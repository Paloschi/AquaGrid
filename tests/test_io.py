"""Tests for the zarr schema helpers and synthetic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacrop_grid.io import (
    generate_synthetic,
    open_climate,
    open_sowing,
    open_soil,
    sowing_to_plant_idx,
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


def test_synthetic_soil_schema(synth_dir):
    ds = open_soil(synth_dir / "soil.zarr")
    assert ds.kind == "hydraulic"
    assert ds.depths == ("0-5", "5-15", "15-30", "30-60", "60-100")
    assert ds.shape == (6, 5)
    assert ds.ds.ksat.dims == ("depth", "y", "x")
