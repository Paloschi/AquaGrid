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


def test_crop_cube_hi0_scales_yield(tmp_path):
    """A per-pixel HI0 override scales dry yield against the named crop."""
    from aquacrop import Crop

    from aquagrid.io import generate_synthetic
    from aquagrid.kernels import constants as C
    from aquagrid.params import crop_params_array, co2_concentration_for_year

    generate_synthetic(tmp_path, ny=1, nx=2, days=250, seed=1)
    sow = xr.open_zarr(tmp_path / "sowing.zarr")
    sow["sowing"] = (("y", "x"), np.array([[2019135, 2019135]], np.int32))
    sow.to_zarr(tmp_path / "sowing.zarr", mode="w")

    base = run_grid(
        tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
        tmp_path / "base.zarr", "Maize", "SandyLoam", parallel=False,
    )
    crop = Crop("Maize", planting_date="05/15")
    hi0 = float(crop_params_array(
        crop, co2_concentration_for_year(2019))[C.CP_HI0])
    cube = xr.Dataset(
        {"crop": (("param", "y", "x"), np.array([[[hi0, hi0 * 0.5]]], np.float32))},
        coords={"param": ["HI0"], "y": [0], "x": [0, 1]},
    )
    cube.to_zarr(tmp_path / "crop.zarr", mode="w")
    varied = run_grid(
        tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
        tmp_path / "varied.zarr", "Maize", "SandyLoam",
        crop_zarr=tmp_path / "crop.zarr", parallel=False,
    )
    base_y = xr.open_zarr(base).dry_yield.values
    varied_y = xr.open_zarr(varied).dry_yield.values
    assert varied_y[0, 0] == pytest.approx(base_y[0, 0], rel=1e-5)
    assert varied_y[0, 1] / varied_y[0, 0] == pytest.approx(0.5, rel=0.15)


def test_apply_crop_cube_nan_unknown_and_calendar_cd():
    from aquagrid.kernels import constants as C
    from aquagrid.params import apply_crop_cube

    base = np.zeros(C.CP_N)
    base[C.CP_HI0] = 0.4
    base[C.CP_CALENDAR_TYPE] = 1
    base[C.CP_MATURITY] = 100
    base[C.CP_MATURITY_CD] = 100

    hi = xr.DataArray(
        np.array([[[0.5, np.nan]]], np.float64),
        dims=("param", "y", "x"),
        coords={"param": ["HI0"], "y": [0], "x": [0, 1]},
    )
    out = apply_crop_cube(base, hi)
    assert out[0, 0, C.CP_HI0] == pytest.approx(0.5)
    assert out[0, 1, C.CP_HI0] == pytest.approx(0.4)

    bad = hi.assign_coords(param=["NotAParam"])
    with pytest.raises(ValueError, match="unknown crop parameter"):
        apply_crop_cube(base, bad)

    maturity = xr.DataArray(
        np.array([[[80.0, np.nan]]], np.float64),
        dims=("param", "y", "x"),
        coords={"param": ["Maturity"], "y": [0], "x": [0, 1]},
    )
    out = apply_crop_cube(base, maturity)
    assert out[0, 0, C.CP_MATURITY] == pytest.approx(80)
    assert out[0, 0, C.CP_MATURITY_CD] == pytest.approx(80)
    assert out[0, 1, C.CP_MATURITY] == pytest.approx(100)
    assert out[0, 1, C.CP_MATURITY_CD] == pytest.approx(100)

    gdd = base.copy()
    gdd[C.CP_CALENDAR_TYPE] = 2
    gdd[C.CP_MATURITY_CD] = 7
    out = apply_crop_cube(gdd, maturity)
    assert out[0, 0, C.CP_MATURITY] == pytest.approx(80)
    assert out[0, 0, C.CP_MATURITY_CD] == pytest.approx(7)

