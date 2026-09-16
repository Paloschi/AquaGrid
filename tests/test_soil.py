"""Soil zarr I/O, PTF, per-pixel profiles, and grid integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aquacrop import Crop, Soil

from aquacrop_grid.engine.run import run_grid_arrays
from aquacrop_grid.io.soil import (
    ACTIVE_DEPTHS,
    canonicalize_depth,
    open_soil,
    validate_soil_grid,
)
from aquacrop_grid.io.synthetic import (
    generate_synthetic,
    synthetic_soil_hydraulic,
    synthetic_soil_texture,
    synthetic_sowing_grid,
)
from aquacrop_grid.kernels import constants as C
from aquacrop_grid.params import (
    co2_concentration_for_year,
    crop_params_array,
    initial_water_content,
    soil_params,
)
from aquacrop_grid.pipeline import run_grid
from aquacrop_grid.soil_grid import (
    curve_number_from_ksat,
    profiles_from_store,
    saxton_rawls,
)


HYDRO_KEYS = (
    "th_fc", "th_s", "th_wp", "th_dry", "ksat", "tau", "penetrability",
)


def _write(ds: xr.Dataset, path) -> None:
    ds.to_zarr(path, mode="w")


def test_depth_aliases():
    assert canonicalize_depth("000-005cm") == "0-5"
    assert canonicalize_depth("0-5cm") == "0-5"
    assert canonicalize_depth("05-15cm") == "5-15"
    assert canonicalize_depth("100-200cm") == "100-200"
    assert canonicalize_depth("nope") is None


def test_open_hydro_aliases_and_depths(tmp_path):
    depths = ["000-005cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
    ny, nx = 3, 2
    base = synthetic_soil_hydraulic(ny, nx, depths=tuple(ACTIVE_DEPTHS),
                                    gradient=False)
    ds = xr.Dataset(
        {
            "Ksat": (("band", "y", "x"), base.ksat.values),
            "WCsat": (("band", "y", "x"), base.wcsat.values),
            "WCpF2": (("band", "y", "x"), base.wcpf2.values),
            "WCpF3": (("band", "y", "x"), base.wcpf3.values),
        },
        coords={"band": depths, "y": np.arange(ny), "x": np.arange(nx)},
    )
    path = tmp_path / "soil.zarr"
    _write(ds, path)
    grid = open_soil(path)
    assert grid.kind == "hydraulic"
    assert grid.depths == ACTIVE_DEPTHS
    assert set(grid.ds.data_vars) == {"ksat", "wcsat", "wcpf2", "wcpf3"}
    # physical float WC, ksat cm/d -> mm/d
    np.testing.assert_allclose(grid.ds.wcsat.values, 0.41, atol=1e-5)
    np.testing.assert_allclose(grid.ds.ksat.values, 1200.0, atol=1e-3)


def test_integer_scale_factors(tmp_path):
    ny, nx, nd = 2, 2, 5
    ds = xr.Dataset(
        {
            "ksat": (("depth", "y", "x"),
                     np.full((nd, ny, nx), 1_200_000, dtype=np.int32)),
            "wcsat": (("depth", "y", "x"),
                      np.full((nd, ny, nx), 4100, dtype=np.int32)),
            "wcpf2": (("depth", "y", "x"),
                      np.full((nd, ny, nx), 2200, dtype=np.int32)),
            "wcpf3": (("depth", "y", "x"),
                      np.full((nd, ny, nx), 1000, dtype=np.int32)),
        },
        coords={"depth": list(ACTIVE_DEPTHS),
                "y": np.arange(ny), "x": np.arange(nx)},
    )
    path = tmp_path / "soil.zarr"
    _write(ds, path)
    grid = open_soil(path)
    np.testing.assert_allclose(grid.ds.wcsat.values, 0.41)
    np.testing.assert_allclose(grid.ds.wcpf2.values, 0.22)
    np.testing.assert_allclose(grid.ds.wcpf3.values, 0.10)
    np.testing.assert_allclose(grid.ds.ksat.values, 1200.0)


def test_missing_depth_raises(tmp_path):
    ds = synthetic_soil_hydraulic(2, 2, depths=("0-5", "5-15", "15-30", "30-60"))
    path = tmp_path / "soil.zarr"
    _write(ds, path)
    with pytest.raises(ValueError, match="missing required depth"):
        open_soil(path)


def test_validate_misaligned_grid(tmp_path):
    _write(synthetic_soil_hydraulic(2, 2, gradient=False), tmp_path / "s.zarr")
    grid = open_soil(tmp_path / "s.zarr")
    with pytest.raises(ValueError, match="does not match"):
        validate_soil_grid(grid, 3, 2)


def test_open_texture_percent(tmp_path):
    ds = synthetic_soil_texture(2, 2, depths=None)
    _write(ds, tmp_path / "tex.zarr")
    grid = open_soil(tmp_path / "tex.zarr")
    assert grid.kind == "texture"
    assert grid.depths == ()
    np.testing.assert_allclose(grid.ds.sand.values, 65.0)


def test_saxton_rawls_matches_aquacrop():
    soil = Soil("custom")
    pairs = [(90.0, 5.0, 0.0), (65.0, 10.0, 2.5), (20.0, 50.0, 1.0)]
    for sand, clay, om in pairs:
        ref = soil.calculate_soil_hydraulic_properties(
            sand / 100.0, clay / 100.0, om)
        got = saxton_rawls(sand / 100.0, clay / 100.0, om)
        for a, b in zip(got, ref):
            assert float(a) == pytest.approx(float(b), rel=0, abs=1e-12)


def test_sandyloam_zarr_profile_and_kernel(tmp_path, weather_df):
    ds = xr.Dataset(
        {
            "ksat": (("y", "x"), np.array([[120.0]], np.float32)),
            "wcsat": (("y", "x"), np.array([[0.41]], np.float32)),
            "wcpf2": (("y", "x"), np.array([[0.22]], np.float32)),
            "wcpf3": (("y", "x"), np.array([[0.10]], np.float32)),
        },
        coords={"y": [0], "x": [0]},
    )
    _write(ds, tmp_path / "soil.zarr")
    grid = open_soil(tmp_path / "soil.zarr")
    crop = Crop("Maize", planting_date="05/15")
    sp, prof, thini, valid = profiles_from_store(grid, zmax=crop.Zmax)
    assert valid[0]
    assert prof["th_wp"][0, 0] == pytest.approx(0.10)
    assert prof["th_fc"][0, 0] == pytest.approx(0.22)
    assert prof["th_s"][0, 0] == pytest.approx(0.41)
    assert prof["ksat"][0, 0] == pytest.approx(1200.0)
    assert curve_number_from_ksat(np.array([1200.0]))[0] == 46.0

    start = pd.Timestamp("2019/05/15")
    plant_idx = int((weather_df.Date == start).idxmax())
    sub = weather_df.iloc[plant_idx:].reset_index(drop=True)
    cp = crop_params_array(crop, co2_conc=co2_concentration_for_year(2019))
    fin, _ = run_grid_arrays(
        tmin=sub.MinTemp.to_numpy()[:, None],
        tmax=sub.MaxTemp.to_numpy()[:, None],
        prcp=sub.Precipitation.to_numpy()[:, None],
        et0=sub.ReferenceET.to_numpy()[:, None],
        plant_idx=np.array([0], np.int64),
        cp=cp, sp=sp, profile=prof, th_init=thini, parallel=False,
    )
    assert fin[0, C.OF_STATUS] == C.STATUS_OK
    assert fin[0, C.OF_DRY_YIELD] > 0

    named = Soil("SandyLoam")
    sp_n, prof_n = soil_params(named, zmax=crop.Zmax)
    th_n = initial_water_content(named, "FC")
    fin_n, _ = run_grid_arrays(
        tmin=sub.MinTemp.to_numpy()[:, None],
        tmax=sub.MaxTemp.to_numpy()[:, None],
        prcp=sub.Precipitation.to_numpy()[:, None],
        et0=sub.ReferenceET.to_numpy()[:, None],
        plant_idx=np.array([0], np.int64),
        cp=cp, sp=sp_n, profile=prof_n, th_init=th_n, parallel=False,
    )
    assert fin[0, C.OF_DRY_YIELD] == pytest.approx(
        fin_n[0, C.OF_DRY_YIELD], rel=0.25)


def test_broadcast_2d_matches_1d(weather_df):
    start = pd.Timestamp("2019/05/15")
    plant_idx = int((weather_df.Date == start).idxmax())
    sub = weather_df.iloc[plant_idx:].reset_index(drop=True)
    crop = Crop("Maize", planting_date="05/15")
    soil = Soil("SandyLoam")
    cp = crop_params_array(crop, co2_conc=co2_concentration_for_year(2019))
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")

    def weather(npix):
        tmin = np.repeat(sub.MinTemp.to_numpy()[:, None], npix, axis=1)
        tmax = np.repeat(sub.MaxTemp.to_numpy()[:, None], npix, axis=1)
        prcp = np.repeat(sub.Precipitation.to_numpy()[:, None], npix, axis=1)
        et0 = np.repeat(sub.ReferenceET.to_numpy()[:, None], npix, axis=1)
        return tmin, tmax, prcp, et0

    tmin, tmax, prcp, et0 = weather(2)
    pidx = np.array([0, 0], np.int64)
    fin_1d, _ = run_grid_arrays(
        tmin=tmin, tmax=tmax, prcp=prcp, et0=et0, plant_idx=pidx,
        cp=cp, sp=sp, profile=prof, th_init=thini, parallel=False,
    )
    sp2 = np.stack([sp, sp])
    prof2 = {k: (np.stack([v, v]) if k in HYDRO_KEYS else v)
             for k, v in prof.items()}
    th2 = np.stack([thini, thini])
    fin_2d, _ = run_grid_arrays(
        tmin=tmin, tmax=tmax, prcp=prcp, et0=et0, plant_idx=pidx,
        cp=cp, sp=sp2, profile=prof2, th_init=th2, parallel=False,
    )
    np.testing.assert_allclose(fin_1d, fin_2d, rtol=0, atol=0)


def test_texture_profiles_finite(tmp_path):
    ds = synthetic_soil_texture(2, 3)
    _write(ds, tmp_path / "tex.zarr")
    grid = open_soil(tmp_path / "tex.zarr")
    sp, prof, thini, valid = profiles_from_store(grid, zmax=2.3)
    assert valid.all()
    assert np.isfinite(prof["ksat"]).all()
    assert (prof["th_s"] > prof["th_fc"]).all()
    assert (prof["th_fc"] > prof["th_wp"]).all()
    assert sp.shape == (6, C.SP_N)


def test_nodata_soil_masks_pixel(tmp_path):
    generate_synthetic(tmp_path, ny=2, nx=2, days=250, seed=1)
    soil = synthetic_soil_hydraulic(2, 2, gradient=False)
    soil["ksat"].values[:, 0, 1] = np.nan
    _write(soil, tmp_path / "soil.zarr")
    sow = synthetic_sowing_grid("2019-05-15", 2, 2, mask_frac=0.0, seed=1)
    sow.to_dataset().to_zarr(tmp_path / "sowing.zarr", mode="w")
    store = run_grid(
        tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
        tmp_path / "out.zarr", "Maize",
        soil_zarr=tmp_path / "soil.zarr",
        save_daily=False, tile=2, parallel=False,
    )
    res = xr.open_zarr(store, decode_timedelta=False)
    assert res.status.values[0, 1] == C.STATUS_NOT_SIMULATED
    assert res.status.values[0, 0] == C.STATUS_OK
    assert res.dry_yield.values[0, 0] > 0
    assert res.dry_yield.values[0, 1] == 0


def test_grid_two_soils_differ(tmp_path):
    generate_synthetic(tmp_path, ny=2, nx=2, days=250, seed=2)
    soil = synthetic_soil_hydraulic(2, 2, gradient=False)
    soil["ksat"].loc[{"x": 1}] = 3.5
    soil["wcsat"].loc[{"x": 1}] = 0.55
    soil["wcpf2"].loc[{"x": 1}] = 0.54
    soil["wcpf3"].loc[{"x": 1}] = 0.39
    _write(soil, tmp_path / "soil.zarr")
    sow = synthetic_sowing_grid("2019-05-15", 2, 2, mask_frac=0.0, seed=2)
    sow.to_dataset().to_zarr(tmp_path / "sowing.zarr", mode="w")
    store = run_grid(
        tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
        tmp_path / "out.zarr", "Maize",
        soil_zarr=tmp_path / "soil.zarr",
        save_daily=False, tile=2, parallel=False,
    )
    res = xr.open_zarr(store, decode_timedelta=False)
    y0 = float(res.dry_yield.values[0, 0])
    y1 = float(res.dry_yield.values[0, 1])
    assert y0 > 0 and y1 > 0
    assert y0 != pytest.approx(y1, rel=1e-3, abs=1e-3)


def test_run_grid_rejects_both_soil_sources(tmp_path):
    generate_synthetic(tmp_path, ny=2, nx=2, days=40)
    with pytest.raises(ValueError, match="exactly one"):
        run_grid(
            tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
            tmp_path / "out.zarr", "Maize", "SandyLoam",
            soil_zarr=tmp_path / "soil.zarr",
        )
