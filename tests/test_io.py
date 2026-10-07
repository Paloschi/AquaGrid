"""Tests for the zarr schema helpers and synthetic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aquagrid.io import (
    generate_synthetic,
    open_climate,
    open_sowing,
    open_soil,
    sowing_to_plant_idx,
    validate_climate,
)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    generate_synthetic(out, ny=6, nx=5, days=200)
    return out


def test_synthetic_climate_schema(synth_dir):
    ds = open_climate(synth_dir / "climate.zarr")
    assert ds.tmin.shape == (200, 6, 5)
    assert (ds.tmax.values > ds.tmin.values).all()
    assert (ds.precip.values >= 0).all()
    assert (ds.eto.values > 0).all()


def test_synthetic_sowing(synth_dir):
    sow = open_sowing(synth_dir / "sowing.zarr")
    assert sow.dtype == np.int32
    valid = sow.values[sow.values > 0]
    assert valid.size > 0
    assert ((valid // 1000) == 2019).all()


def test_sowing_to_plant_idx(synth_dir):
    ds = open_climate(synth_dir / "climate.zarr")
    sow = open_sowing(synth_dir / "sowing.zarr")
    time = pd.DatetimeIndex(ds.time.values)
    idx = sowing_to_plant_idx(sow.values, time)

    assert idx.shape == sow.shape
    assert (idx[sow.values <= 0] == -1).all()
    ok = idx >= 0
    assert ok.any()
    # spot-check round trip: time[idx] must match the YYYYDDD date
    ys, xs = np.nonzero(ok)
    for y, x in list(zip(ys, xs))[:10]:
        v = int(sow.values[y, x])
        expect = (pd.Timestamp(year=v // 1000, month=1, day=1)
                  + pd.Timedelta(days=v % 1000 - 1))
        assert time[idx[y, x]] == expect


def test_sowing_outside_time_axis():
    time = pd.date_range("2019-05-01", periods=30, freq="D")
    sow = np.array([[2019120, 2019121, 2019200], [0, -1, 2020001]], np.int32)
    idx = sowing_to_plant_idx(sow, time)
    assert idx[0, 0] == -1        # 2019-04-30, before axis
    assert idx[0, 1] == 0         # 2019-05-01
    assert idx[0, 2] == -1        # beyond 30 days
    assert (idx[1] == -1).all()


def _climate(time, ny=2, nx=2, *, variables=None, coords=None):
    n = len(time)
    data = np.zeros((n, ny, nx), np.float32)
    names = variables or ("tmin", "tmax", "precip", "eto")
    fields = {
        "tmin": data,
        "tmax": data + 10,
        "precip": np.abs(data),
        "eto": data + 3,
    }
    if coords is None:
        coords = {"time": time, "y": np.arange(ny), "x": np.arange(nx)}
    return xr.Dataset(
        {name: (("time", "y", "x"), fields[name]) for name in names},
        coords=coords,
    )


def test_climate_rejects_time_gap(tmp_path):
    time = pd.to_datetime(["2019-05-01", "2019-05-02", "2019-05-04"])
    path = tmp_path / "climate.zarr"
    _climate(time).to_zarr(path, mode="w")
    with pytest.raises(ValueError, match="gap-free"):
        open_climate(path)


def test_climate_rejects_missing_variable(tmp_path):
    time = pd.date_range("2019-05-01", periods=4, freq="D")
    path = tmp_path / "climate.zarr"
    _climate(time, variables=("tmin", "tmax", "precip")).to_zarr(path, mode="w")
    with pytest.raises(ValueError, match="missing variable 'eto'"):
        open_climate(path)


def test_climate_rejects_variable_dims(tmp_path):
    time = pd.date_range("2019-05-01", periods=4, freq="D")
    ds = _climate(time)
    ds["tmin"] = ds.tmin.transpose("y", "x", "time")
    path = tmp_path / "climate.zarr"
    ds.to_zarr(path, mode="w")
    with pytest.raises(ValueError, match="has dims"):
        open_climate(path)


def test_climate_rejects_missing_time_coordinate():
    time = pd.date_range("2019-05-01", periods=3, freq="D")
    ds = _climate(
        time,
        ny=1,
        nx=1,
        coords={"y": [0], "x": [0]},
    )
    with pytest.raises(ValueError, match="missing 'time' coordinate"):
        validate_climate(ds)


def test_sowing_rejects_missing_variable(tmp_path):
    path = tmp_path / "sowing.zarr"
    xr.Dataset(
        {"other": (("y", "x"), np.zeros((2, 2), np.int32))},
        coords={"y": [0, 1], "x": [0, 1]},
    ).to_zarr(path, mode="w")
    with pytest.raises(ValueError, match="missing variable 'sowing'"):
        open_sowing(path)


def test_sowing_rejects_wrong_dims(tmp_path):
    path = tmp_path / "sowing.zarr"
    xr.Dataset(
        {"sowing": (("x", "y"), np.zeros((2, 2), np.int32))},
        coords={"y": [0, 1], "x": [0, 1]},
    ).to_zarr(path, mode="w")
    with pytest.raises(ValueError, match=r"\(y, x\)"):
        open_sowing(path)


def test_synthetic_soil_schema(synth_dir):
    ds = open_soil(synth_dir / "soil.zarr")
    assert ds.kind == "hydraulic"
    assert ds.depths == ("0-5", "5-15", "15-30", "30-60", "60-100")
    assert ds.shape == (6, 5)
    assert ds.ds.ksat.dims == ("depth", "y", "x")
