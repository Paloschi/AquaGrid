"""Input schema validation and sowing-grid conversion.

Schema reference: ``docs/zarr-schema.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CLIMATE_VARS = ("tmin", "tmax", "precip", "eto")
CLIMATE_DIMS = ("time", "y", "x")


def validate_climate(ds: xr.Dataset) -> None:
    """Raise ``ValueError`` if ``ds`` does not follow the climate schema."""
    for var in CLIMATE_VARS:
        if var not in ds:
            raise ValueError(f"climate store missing variable {var!r}")
        if tuple(ds[var].dims) != CLIMATE_DIMS:
            raise ValueError(
                f"variable {var!r} has dims {ds[var].dims}, "
                f"expected {CLIMATE_DIMS}"
            )
    if "time" not in ds.coords:
        raise ValueError("climate store missing 'time' coordinate")
    time = pd.DatetimeIndex(ds.time.values)
    step = time[1:] - time[:-1]
    if len(time) > 1 and not (step == pd.Timedelta(days=1)).all():
        raise ValueError("'time' coordinate must be daily and gap-free")


def open_climate(store: str | Path) -> xr.Dataset:
    """Open and validate a climate zarr store."""
    ds = xr.open_zarr(store, decode_timedelta=False)
    validate_climate(ds)
    return ds


def open_sowing(store: str | Path, var: str = "sowing") -> xr.DataArray:
    """Open a sowing grid (YYYYDDD int32, <=0 = not simulated)."""
    ds = xr.open_zarr(store, decode_timedelta=False)
    if var not in ds:
        raise ValueError(f"sowing store missing variable {var!r}")
    da = ds[var]
    if tuple(da.dims) != ("y", "x"):
        raise ValueError(f"sowing variable must be (y, x), got {da.dims}")
    return da


def sowing_to_plant_idx(sowing: np.ndarray,
                        time: pd.DatetimeIndex) -> np.ndarray:
    """Convert a YYYYDDD sowing grid to indices into the time axis.

    Returns int64 array of the same shape; -1 where the pixel is masked
    (``sowing <= 0``) or the date falls outside the time axis.
    """
    sow = np.asarray(sowing, np.int64)
    out = np.full(sow.shape, -1, np.int64)
    valid = sow > 0
    if not valid.any():
        return out

    years = sow[valid] // 1000
    doys = sow[valid] % 1000
    dates = (pd.to_datetime(years * 10000 + 101, format="%Y%m%d")
             + pd.to_timedelta(doys - 1, unit="D"))

    t0 = time[0]
    idx = np.array((dates - t0).days, np.int64)
    idx[(idx < 0) | (idx >= len(time))] = -1
    out[valid] = idx
    return out
