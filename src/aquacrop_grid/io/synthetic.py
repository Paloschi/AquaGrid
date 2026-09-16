"""Small synthetic zarr dataset for development and end-to-end tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def synthetic_weather_grid(start: str, days: int, ny: int, nx: int,
                           seed: int = 42) -> xr.Dataset:
    """Build a (time, y, x) climate cube: one synthetic base series plus a
    smooth spatial gradient, so neighbouring pixels differ but stay
    physically plausible."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=days, freq="D")
    doy = time.dayofyear.to_numpy()
    season = np.sin((doy - 105) / 365 * 2 * np.pi)

    tmin = 14 + 6 * season + rng.normal(0, 2, days)
    tmax = tmin + 9 + rng.normal(0, 1.5, days)
    rain_prob = 0.25 + 0.15 * season
    prcp = np.where(rng.random(days) < rain_prob,
                    rng.gamma(2.0, 6.0, days), 0.0)
    et0 = np.clip(3.5 + 1.5 * season + rng.normal(0, 0.6, days), 0.3, None)

    # smooth spatial fields
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    dtemp = 2.0 * (yy / max(ny - 1, 1) - 0.5)          # +-1 degC N-S
    fprcp = 1.0 + 0.3 * (xx / max(nx - 1, 1) - 0.5)    # +-15% E-W
    fet0 = 1.0 + 0.1 * (xx / max(nx - 1, 1) - 0.5)

    def cube(base, spatial, mode):
        b = base[:, None, None]
        s = spatial[None, :, :]
        return (b + s if mode == "add" else b * s).astype(np.float32)

    ds = xr.Dataset(
        {
            "tmin": (("time", "y", "x"), cube(tmin, dtemp, "add")),
            "tmax": (("time", "y", "x"), cube(tmax, dtemp, "add")),
            "precip": (("time", "y", "x"), cube(prcp, fprcp, "mul")),
            "eto": (("time", "y", "x"), cube(et0, fet0, "mul")),
        },
        coords={"time": time, "y": np.arange(ny), "x": np.arange(nx)},
    )
    return ds


def synthetic_sowing_grid(base_date: str, ny: int, nx: int,
                          spread_days: int = 20, seed: int = 42,
                          mask_frac: float = 0.1) -> xr.DataArray:
    """(y, x) int32 YYYYDDD sowing grid spread around ``base_date``,
    with ``mask_frac`` of pixels masked out (0 = not simulated)."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(base_date)
    offsets = rng.integers(0, spread_days + 1, size=(ny, nx))
    dates = base + pd.to_timedelta(offsets.ravel(), unit="D")
    yyyyddd = (dates.year * 1000 + dates.dayofyear).to_numpy()
    sow = yyyyddd.reshape(ny, nx).astype(np.int32)
    sow[rng.random((ny, nx)) < mask_frac] = 0
    return xr.DataArray(
        sow, dims=("y", "x"),
        coords={"y": np.arange(ny), "x": np.arange(nx)}, name="sowing",
    )


def synthetic_soil_hydraulic(
    ny: int, nx: int, *,
    depths: tuple[str, ...] = ("0-5", "5-15", "15-30", "30-60", "60-100"),
    ksat_cm_d: float = 120.0,
    wcsat: float = 0.41,
    wcpf2: float = 0.22,
    wcpf3: float = 0.10,
    gradient: bool = True,
) -> xr.Dataset:
    """(depth, y, x) hydraulic cube in physical HiHydroSoil units (Ksat cm/d).

    Default values match AquaCrop ``SandyLoam``. A mild E–W gradient makes
    neighbouring pixels differ when ``gradient`` is true.
    """
    n_d = len(depths)
    _, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    fx = xx / max(nx - 1, 1)
    g = (0.85 + 0.3 * fx) if gradient else 1.0

    def cube(base, scale=1.0):
        field = np.full((n_d, ny, nx), base, dtype=np.float32) * scale
        if gradient:
            field = field * g[None, :, :]
        return field

    return xr.Dataset(
        {
            "ksat": (("depth", "y", "x"), cube(ksat_cm_d)),
            "wcsat": (("depth", "y", "x"), np.clip(cube(wcsat), 0.05, 0.7)),
            "wcpf2": (("depth", "y", "x"), np.clip(cube(wcpf2), 0.04, 0.6)),
            "wcpf3": (("depth", "y", "x"), np.clip(cube(wcpf3), 0.02, 0.5)),
        },
        coords={
            "depth": list(depths),
            "y": np.arange(ny),
            "x": np.arange(nx),
        },
    )


def synthetic_soil_texture(
    ny: int, nx: int, *,
    depths: tuple[str, ...] | None = ("0-5", "5-15", "15-30", "30-60", "60-100"),
    sand: float = 65.0,
    silt: float = 25.0,
    clay: float = 10.0,
) -> xr.Dataset:
    """Texture percents, 2-D (homogeneous profile) or 3-D with ``depths``."""
    if depths:
        shape = (len(depths), ny, nx)
        dims = ("depth", "y", "x")
        coords = {
            "depth": list(depths),
            "y": np.arange(ny),
            "x": np.arange(nx),
        }
    else:
        shape = (ny, nx)
        dims = ("y", "x")
        coords = {"y": np.arange(ny), "x": np.arange(nx)}
    return xr.Dataset(
        {
            "sand": (dims, np.full(shape, sand, dtype=np.float32)),
            "silt": (dims, np.full(shape, silt, dtype=np.float32)),
            "clay": (dims, np.full(shape, clay, dtype=np.float32)),
        },
        coords=coords,
    )


def generate_synthetic(out: str | Path, *, ny: int = 10, nx: int = 10,
                       days: int = 540, start: str = "2019-05-01",
                       sowing_base: str = "2019-05-15",
                       seed: int = 42) -> tuple[Path, Path]:
    """Write ``climate.zarr``, ``sowing.zarr`` and ``soil.zarr`` under ``out``.

    Returns ``(climate_path, sowing_path)``; soil is written beside them.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    clim = synthetic_weather_grid(start, days, ny, nx, seed=seed)
    clim = clim.chunk({"time": -1, "y": min(ny, 256), "x": min(nx, 256)})
    clim_path = out / "climate.zarr"
    clim.to_zarr(clim_path, mode="w")

    sow = synthetic_sowing_grid(sowing_base, ny, nx, seed=seed)
    sow_path = out / "sowing.zarr"
    sow.to_dataset().to_zarr(sow_path, mode="w")

    soil = synthetic_soil_hydraulic(ny, nx)
    soil.chunk({"depth": -1, "y": min(ny, 256), "x": min(nx, 256)}).to_zarr(
        out / "soil.zarr", mode="w")

    return clim_path, sow_path
