"""Grid driver tests: synthetic zarr in -> zarr out, and consistency of the
grid path against a direct single-pixel kernel run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aquagrid.io import generate_synthetic, open_climate, open_sowing
from aquagrid.io.schema import sowing_to_plant_idx
from aquagrid.pipeline import run_grid


NY, NX, DAYS = 6, 5, 400


@pytest.fixture(scope="module")
def grid_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("grid")
    generate_synthetic(out, ny=NY, nx=NX, days=DAYS)
    store = run_grid(
        out / "climate.zarr", out / "sowing.zarr", out / "output.zarr",
        "Maize", "SandyLoam",
        save_daily=True, tile=4, parallel=False,
    )
    return out, store


def test_output_final_fields(grid_run):
    src, store = grid_run
    res = xr.open_zarr(store, decode_timedelta=False)
    sow = open_sowing(src / "sowing.zarr").values

    status = res.status.values
    assert status.shape == (NY, NX)
    # masked pixels flagged, simulated pixels ok
    assert (status[sow <= 0] == 1).all()
    assert (status[sow > 0] == 0).all()
    assert (res.dry_yield.values[sow > 0] > 0).all()
    assert (res.dry_yield.values[sow <= 0] == 0).all()
    assert (res.dap_end.values[sow > 0] > 50).all()


def test_output_daily_group(grid_run):
    src, store = grid_run
    daily = xr.open_zarr(store, group="daily", decode_timedelta=False)
    sow = open_sowing(src / "sowing.zarr").values

    cc = daily.canopy_cover.values
    assert cc.shape == (DAYS, NY, NX)
    ys, xs = np.nonzero(sow > 0)
    y, x = ys[0], xs[0]
    assert np.nanmax(cc[:, y, x]) > 0.5     # canopy actually developed
    ym, xm = np.nonzero(sow <= 0)[0][0], np.nonzero(sow <= 0)[1][0]
    assert np.isnan(cc[:, ym, xm]).all()    # masked pixel: all NaN


def test_grid_matches_single_pixel(grid_run):
    """One pixel run through the tiled grid driver must equal the same
    pixel run alone through run_grid_arrays."""
    from aquacrop import Crop, Soil

    from aquagrid.engine.run import run_grid_arrays
    from aquagrid.kernels import constants as C
    from aquagrid.params import (
        co2_concentration_for_year,
        crop_params_array,
        initial_water_content,
        soil_params,
    )

    src, store = grid_run
    ds = open_climate(src / "climate.zarr")
    sow = open_sowing(src / "sowing.zarr").values
    time = pd.DatetimeIndex(ds.time.values)
    plant_idx = sowing_to_plant_idx(sow, time)

    ys, xs = np.nonzero(sow > 0)
    y, x = ys[-1], xs[-1]   # last valid pixel (deep in the tile loop)

    # params exactly as the pipeline builds them (median sowing year/doy)
    valid = sow > 0
    year = int(np.median(sow[valid] // 1000))
    doy = int(np.median(sow[valid] % 1000))
    nominal = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    crop = Crop("Maize", planting_date=nominal.strftime("%m/%d"))
    soil = Soil("SandyLoam")
    cp = crop_params_array(crop, co2_conc=co2_concentration_for_year(year))
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")

    col = ds.isel(y=y, x=x)
    fin, _ = run_grid_arrays(
        tmin=col.tmin.values.astype(np.float64)[:, None],
        tmax=col.tmax.values.astype(np.float64)[:, None],
        prcp=col.precip.values.astype(np.float64)[:, None],
        et0=col.eto.values.astype(np.float64)[:, None],
        plant_idx=np.array([plant_idx[y, x]], np.int64),
        cp=cp, sp=sp, profile=prof, th_init=thini, parallel=False,
    )

    res = xr.open_zarr(store, decode_timedelta=False)
    assert res.dry_yield.values[y, x] == pytest.approx(
        fin[0, C.OF_DRY_YIELD], rel=1e-6)
    assert res.biomass.values[y, x] == pytest.approx(
        fin[0, C.OF_BIOMASS], rel=1e-6)
