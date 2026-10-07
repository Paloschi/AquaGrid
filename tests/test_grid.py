"""Grid driver tests: synthetic zarr in -> zarr out, and consistency of the
grid path against a direct single-pixel kernel run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aquagrid.io import generate_synthetic, open_climate, open_sowing
from aquagrid.io.schema import sowing_to_plant_idx
from aquagrid.pipeline import run_from_config, run_grid


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


def test_prange_grid_matches_single_pixel_bitwise(tmp_path):
    """The default grid path (prange) reproduces one pixel, bit for bit."""
    import inspect

    from aquacrop import Crop, Soil

    from aquagrid.engine.run import run_grid_arrays
    from aquagrid.io import generate_synthetic
    from aquagrid.kernels import constants as C
    from aquagrid.params import (
        co2_concentration_for_year,
        crop_params_array,
        initial_water_content,
        soil_params,
    )
    from aquagrid.pipeline import DAILY_VARS, FINAL_VARS

    assert inspect.signature(run_grid).parameters["parallel"].default is True

    ny, nx, days = 2, 2, 200
    generate_synthetic(tmp_path, ny=ny, nx=nx, days=days, seed=3)
    sow = np.array([[2019135, 2019135], [2019135, 0]], np.int32)
    xr.Dataset(
        {"sowing": (("y", "x"), sow)},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    ).to_zarr(tmp_path / "sowing.zarr", mode="w")

    store = run_grid(
        tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
        tmp_path / "output.zarr", "Maize", "SandyLoam",
        save_daily=True,
    )

    ds = open_climate(tmp_path / "climate.zarr")
    time = pd.DatetimeIndex(ds.time.values)
    plant_idx = sowing_to_plant_idx(sow, time)
    year = 2019
    crop = Crop("Maize", planting_date="05/15")
    soil = Soil("SandyLoam")
    cp = crop_params_array(crop, co2_conc=co2_concentration_for_year(year))
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")

    nt = days
    npix = ny * nx
    weather = {
        name: ds[name].values.astype(np.float64).reshape(nt, npix)
        for name in ("tmin", "tmax", "precip", "eto")
    }
    pidx = plant_idx.reshape(npix)
    common = dict(
        tmin=weather["tmin"], tmax=weather["tmax"],
        prcp=weather["precip"], et0=weather["eto"],
        plant_idx=pidx, cp=cp, sp=sp, profile=prof, th_init=thini,
        save_daily=True,
    )
    fin_s, day_s = run_grid_arrays(parallel=False, **common)
    fin_p, day_p = run_grid_arrays(parallel=True, **common)
    assert np.array_equal(fin_p, fin_s)
    assert np.array_equal(day_p, day_s, equal_nan=True)

    res = xr.open_zarr(store, decode_timedelta=False)
    daily_xr = xr.open_zarr(store, group="daily", decode_timedelta=False)
    for y in range(ny):
        for x in range(nx):
            p = y * nx + x
            for name, (of, dt) in FINAL_VARS.items():
                got = res[name].values[y, x]
                exp = np.asarray(fin_p[p, of]).astype(dt)
                assert np.array_equal(
                    np.asarray(got), np.asarray(exp), equal_nan=True), name
            for name, od in DAILY_VARS.items():
                got = daily_xr[name].values[:, y, x]
                exp = day_p[od, :, p].astype(np.float32)
                assert np.array_equal(got, exp, equal_nan=True), name
    assert res.status.values[1, 1] == C.STATUS_NOT_SIMULATED
    assert res.status.values[0, 0] == C.STATUS_OK


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


def test_crop_cube_hi0_matches_ospy(tmp_path):
    """A per-pixel HI0 override matches AquaCrop-OSPy with that same HI0."""
    from aquacrop import Crop, Soil

    from aquagrid.engine.run import run_grid_arrays
    from aquagrid.io import generate_synthetic
    from aquagrid.io.schema import sowing_to_plant_idx
    from aquagrid.kernels import constants as C
    from aquagrid.params import (
        apply_crop_cube,
        co2_concentration_for_year,
        crop_params_array,
        initial_water_content,
        soil_params,
    )
    from test_parity import assert_ospy_parity, run_reference

    generate_synthetic(tmp_path, ny=1, nx=2, days=250, seed=1)
    sow = xr.open_zarr(tmp_path / "sowing.zarr")
    sow["sowing"] = (("y", "x"), np.array([[2019135, 2019135]], np.int32))
    sow.to_zarr(tmp_path / "sowing.zarr", mode="w")

    base_cp = crop_params_array(
        Crop("Maize", planting_date="05/15"),
        co2_concentration_for_year(2019),
    )
    hi0 = float(base_cp[C.CP_HI0])
    hi = np.array([[[hi0, hi0 * 0.5]]], np.float64)
    cube = xr.Dataset(
        {"crop": (("param", "y", "x"), hi)},
        coords={"param": ["HI0"], "y": [0], "x": [0, 1]},
    )
    cube.to_zarr(tmp_path / "crop.zarr", mode="w")
    store = run_grid(
        tmp_path / "climate.zarr", tmp_path / "sowing.zarr",
        tmp_path / "varied.zarr", "Maize", "SandyLoam",
        crop_zarr=tmp_path / "crop.zarr", parallel=False,
    )

    ds = open_climate(tmp_path / "climate.zarr")
    time = pd.DatetimeIndex(ds.time.values)
    plant_i = int(sowing_to_plant_idx(sow["sowing"].values, time)[0, 0])
    cp = apply_crop_cube(base_cp, cube["crop"])
    assert cp[0, 0, C.CP_HI0] == pytest.approx(hi0)
    assert cp[0, 1, C.CP_HI0] == pytest.approx(hi0 * 0.5)

    crop = Crop("Maize", planting_date="05/15")
    soil = Soil("SandyLoam")
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")
    cols = []
    for x in (0, 1):
        col = ds.isel(y=0, x=x).isel(time=slice(plant_i, None))
        cols.append(col)
    fin, daily = run_grid_arrays(
        tmin=np.column_stack([c.tmin.values.astype(np.float64) for c in cols]),
        tmax=np.column_stack([c.tmax.values.astype(np.float64) for c in cols]),
        prcp=np.column_stack([c.precip.values.astype(np.float64) for c in cols]),
        et0=np.column_stack([c.eto.values.astype(np.float64) for c in cols]),
        plant_idx=np.zeros(2, np.int64),
        cp=cp.reshape(2, -1), sp=sp, profile=prof, th_init=thini,
        save_daily=True, parallel=False,
    )

    published = xr.open_zarr(store, decode_timedelta=False)
    for x, hi_pix in ((0, hi0), (1, hi0 * 0.5)):
        col = cols[x]
        weather = pd.DataFrame({
            "MinTemp": col.tmin.values.astype(np.float64),
            "MaxTemp": col.tmax.values.astype(np.float64),
            "Precipitation": col.precip.values.astype(np.float64),
            "ReferenceET": col.eto.values.astype(np.float64),
            "Date": pd.DatetimeIndex(col.time.values),
        })
        model = run_reference(
            weather, "Maize", "SandyLoam", "2019/05/15",
            crop=Crop("Maize", planting_date="05/15", HI0=hi_pix),
        )
        assert_ospy_parity(model, fin, daily, pixel=x)
        assert published.dry_yield.values[0, x] == np.float32(
            fin[x, C.OF_DRY_YIELD])
        assert published.yield_pot.values[0, x] == np.float32(
            fin[x, C.OF_YIELD_POT])
        assert published.biomass_ns.values[0, x] == np.float32(
            fin[x, C.OF_BIOMASS_NS])
        assert published.hi_adj.values[0, x] == np.float32(fin[x, C.OF_HI_ADJ])


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


def _tiny_grid(tmp_path):
    from aquagrid.io import generate_synthetic

    generate_synthetic(tmp_path, ny=2, nx=2, days=8, seed=0)
    return tmp_path / "climate.zarr", tmp_path / "sowing.zarr"


def test_run_from_config_matches_run_grid(tmp_path):
    """A YAML config is the same run as the arguments it names."""
    import yaml

    generate_synthetic(tmp_path, ny=2, nx=2, days=200, seed=5)
    climate = tmp_path / "climate.zarr"
    sowing = tmp_path / "sowing.zarr"
    output = tmp_path / "yaml.zarr"
    direct = tmp_path / "direct.zarr"
    cfg = {
        "climate": str(climate),
        "sowing": str(sowing),
        "output": str(output),
        "crop": {"name": "Maize"},
        "soil": {"name": "SandyLoam", "scale_factors": {"Ksat": 10}},
        "backend": "cpu",
        "options": {
            "save_daily": True,
            "tile": 2,
            "parallel": True,
            "max_season_days": 400,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    store = run_from_config(tmp_path / "config.yaml")
    run_grid(
        climate, sowing, direct, "Maize", "SandyLoam",
        scale_factors={"ksat": 10.0},
        save_daily=True, tile=2, parallel=True, max_season_days=400,
    )

    got = xr.open_zarr(store, decode_timedelta=False)
    ref = xr.open_zarr(direct, decode_timedelta=False)
    np.testing.assert_array_equal(got.dap_end.values, ref.dap_end.values)
    np.testing.assert_array_equal(got.dry_yield.values, ref.dry_yield.values)
    sow = open_sowing(sowing).values
    assert (got.dap_end.values[sow > 0] > 50).all()
    daily = xr.open_zarr(store, group="daily", decode_timedelta=False)
    assert daily.canopy_cover.shape[0] == 200


def test_run_grid_rejects_neither_soil_source(tmp_path):
    climate, sowing = _tiny_grid(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        run_grid(climate, sowing, tmp_path / "out.zarr", "Maize")


def test_run_grid_rejects_sowing_shape(tmp_path):
    climate, _sowing = _tiny_grid(tmp_path)
    xr.Dataset(
        {"sowing": (("y", "x"), np.array([[2019135]], np.int32))},
        coords={"y": [0], "x": [0]},
    ).to_zarr(tmp_path / "sowing_small.zarr", mode="w")
    with pytest.raises(ValueError, match="sowing grid"):
        run_grid(
            climate, tmp_path / "sowing_small.zarr", tmp_path / "out.zarr",
            "Maize", "SandyLoam",
        )


def test_run_grid_rejects_no_valid_sowing(tmp_path):
    climate, sowing = _tiny_grid(tmp_path)
    xr.Dataset(
        {"sowing": (("y", "x"), np.zeros((2, 2), np.int32))},
        coords={"y": [0, 1], "x": [0, 1]},
    ).to_zarr(sowing, mode="w")
    with pytest.raises(ValueError, match="no valid"):
        run_grid(climate, sowing, tmp_path / "out.zarr", "Maize", "SandyLoam")


def test_run_grid_rejects_misaligned_soil(tmp_path):
    from aquagrid.io.synthetic import synthetic_soil_hydraulic

    climate, sowing = _tiny_grid(tmp_path)
    synthetic_soil_hydraulic(1, 1, gradient=False).to_zarr(
        tmp_path / "soil_small.zarr", mode="w")
    with pytest.raises(ValueError, match="soil grid"):
        run_grid(
            climate, sowing, tmp_path / "out.zarr", "Maize",
            soil_zarr=tmp_path / "soil_small.zarr",
        )


def test_run_grid_rejects_crop_store_without_cube(tmp_path):
    climate, sowing = _tiny_grid(tmp_path)
    xr.Dataset(
        {"nope": (("y", "x"), np.zeros((2, 2)))},
        coords={"y": [0, 1], "x": [0, 1]},
    ).to_zarr(tmp_path / "crop.zarr", mode="w")
    with pytest.raises(ValueError, match="needs a variable 'crop'"):
        run_grid(
            climate, sowing, tmp_path / "out.zarr", "Maize", "SandyLoam",
            crop_zarr=tmp_path / "crop.zarr",
        )


def test_run_grid_rejects_crop_cube_dims(tmp_path):
    climate, sowing = _tiny_grid(tmp_path)
    xr.Dataset(
        {"crop": (("y", "x"), np.zeros((2, 2)))},
        coords={"y": [0, 1], "x": [0, 1]},
    ).to_zarr(tmp_path / "crop.zarr", mode="w")
    with pytest.raises(ValueError, match=r"\(param, y, x\)"):
        run_grid(
            climate, sowing, tmp_path / "out.zarr", "Maize", "SandyLoam",
            crop_zarr=tmp_path / "crop.zarr",
        )


def test_run_grid_rejects_misaligned_crop_cube(tmp_path):
    climate, sowing = _tiny_grid(tmp_path)
    xr.Dataset(
        {"crop": (("param", "y", "x"), np.zeros((1, 1, 1), np.float32))},
        coords={"param": ["HI0"], "y": [0], "x": [0]},
    ).to_zarr(tmp_path / "crop.zarr", mode="w")
    with pytest.raises(ValueError, match="crop cube"):
        run_grid(
            climate, sowing, tmp_path / "out.zarr", "Maize", "SandyLoam",
            crop_zarr=tmp_path / "crop.zarr",
        )

