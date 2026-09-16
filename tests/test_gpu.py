"""CPU vs GPU parity (skipped when no CUDA device is available)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacrop_grid.engine.run import run_grid_arrays
from aquacrop_grid.kernels import constants as C

cuda_available = False
try:
    from numba import cuda

    cuda_available = cuda.is_available()
except Exception:  # pragma: no cover
    pass

pytestmark = pytest.mark.skipif(not cuda_available,
                                reason="no CUDA GPU available")


@pytest.fixture(scope="module")
def grid_inputs(weather_df):
    """A small multi-pixel problem with per-pixel sowing and a masked pixel."""
    from aquacrop import Crop, Soil

    from aquacrop_grid.params import (
        co2_concentration_for_year,
        crop_params_array,
        initial_water_content,
        soil_params,
    )

    start = pd.Timestamp("2019/05/15")
    pi = int((weather_df.Date == start).idxmax())
    sub = weather_df.iloc[pi:].reset_index(drop=True)
    nt = len(sub)

    npix = 16
    rng = np.random.default_rng(7)
    plant_idx = rng.integers(0, 20, npix).astype(np.int64)
    plant_idx[3] = -1  # masked pixel

    # per-pixel weather perturbations so pixels are not identical
    base = np.stack([
        sub.MinTemp.to_numpy(), sub.MaxTemp.to_numpy(),
        sub.Precipitation.to_numpy(), sub.ReferenceET.to_numpy(),
    ])
    dt = rng.normal(0, 0.5, npix)
    fp = 1 + rng.normal(0, 0.05, npix)

    crop = Crop("Maize", planting_date="05/15")
    soil = Soil("SandyLoam")
    cp = crop_params_array(crop, co2_conc=co2_concentration_for_year(2019))
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")

    return dict(
        tmin=base[0][:, None] + dt[None, :],
        tmax=base[1][:, None] + dt[None, :],
        prcp=np.clip(base[2][:, None] * fp[None, :], 0, None),
        et0=np.clip(base[3][:, None] * fp[None, :], 0.1, None),
        plant_idx=plant_idx, cp=cp, sp=sp, profile=prof, th_init=thini,
        save_daily=True,
    )


def test_gpu_matches_cpu(grid_inputs):
    fin_cpu, day_cpu = run_grid_arrays(backend="cpu", parallel=False,
                                       **grid_inputs)
    fin_gpu, day_gpu = run_grid_arrays(backend="gpu", **grid_inputs)

    # status/flags must be identical
    assert np.array_equal(fin_cpu[:, C.OF_STATUS], fin_gpu[:, C.OF_STATUS])
    assert fin_cpu[3, C.OF_STATUS] == C.STATUS_NOT_SIMULATED

    # numeric outputs: tiny FMA-level differences allowed
    np.testing.assert_allclose(fin_gpu, fin_cpu, rtol=1e-9, atol=1e-9)

    assert np.array_equal(np.isnan(day_cpu), np.isnan(day_gpu))
    m = ~np.isnan(day_cpu)
    np.testing.assert_allclose(day_gpu[m], day_cpu[m], rtol=1e-9, atol=1e-9)
