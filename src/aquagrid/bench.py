"""CPU vs GPU throughput benchmark on synthetic data."""

from __future__ import annotations

import time

import numpy as np

from aquagrid.engine.run import run_grid_arrays
from aquagrid.io.synthetic import synthetic_weather_grid


def _inputs(npix: int, days: int, seed: int = 42):
    from aquacrop import Crop, Soil

    from aquagrid.params import (
        co2_concentration_for_year,
        crop_params_array,
        initial_water_content,
        soil_params,
    )

    base = synthetic_weather_grid("2019-05-01", days, 1, 1, seed=seed)
    rng = np.random.default_rng(seed)

    def tile(name):
        col = base[name].values[:, 0, 0].astype(np.float64)
        return np.ascontiguousarray(np.repeat(col[:, None], npix, axis=1))

    tmin = tile("tmin") + rng.normal(0, 0.5, npix)[None, :]
    tmax = tile("tmax") + rng.normal(0, 0.5, npix)[None, :]
    prcp = np.clip(tile("precip") * (1 + rng.normal(0, 0.05, npix))[None, :],
                   0, None)
    et0 = np.clip(tile("eto") * (1 + rng.normal(0, 0.05, npix))[None, :],
                  0.1, None)
    plant_idx = rng.integers(0, 21, npix).astype(np.int64)

    crop = Crop("Maize", planting_date="05/15")
    soil = Soil("SandyLoam")
    cp = crop_params_array(crop, co2_conc=co2_concentration_for_year(2019))
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")

    return dict(tmin=tmin, tmax=tmax, prcp=prcp, et0=et0,
                plant_idx=plant_idx, cp=cp, sp=sp, profile=prof,
                th_init=thini)


def benchmark(npix: int = 16384, days: int = 540,
              backends: tuple[str, ...] = ("cpu", "gpu"),
              seed: int = 42) -> dict[str, float]:
    """Run ``npix`` pixels of ``days`` days on each backend.

    Returns {backend: pixels_per_second}; compilation time is excluded
    (a small warm-up run is executed first).
    """
    kw = _inputs(npix, days, seed=seed)
    warm = _inputs(64, days, seed=seed)

    results: dict[str, float] = {}
    for backend in backends:
        if backend == "gpu":
            from numba import cuda

            if not cuda.is_available():
                print("gpu: CUDA not available, skipping")
                continue
        run_grid_arrays(backend=backend, **warm)   # compile
        t0 = time.perf_counter()
        run_grid_arrays(backend=backend, **kw)
        dt = time.perf_counter() - t0
        results[backend] = npix / dt
        print(f"{backend}: {npix} px x {days} d in {dt:.2f} s "
              f"-> {npix / dt:,.0f} px/s")
    return results
