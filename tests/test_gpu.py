"""CPU vs GPU parity.

Skipped unless a CUDA device is available, or ``NUMBA_ENABLE_CUDASIM=1``
is set before Numba is imported. The simulator compiles the same CUDA
kernel and runs it on the CPU. CI sets that variable so this test is part
of the hosted matrix; a runner with a GPU should leave it unset and
execute the same test on the device.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquagrid.engine.run import run_grid_arrays
from aquagrid.kernels import constants as C

cuda_available = False
try:
    from numba import cuda

    cuda_available = cuda.is_available()
except Exception:  # pragma: no cover
    pass

pytestmark = pytest.mark.skipif(
    not cuda_available,
    reason="no CUDA GPU available (set NUMBA_ENABLE_CUDASIM=1 to simulate)",
)

from test_parity import (  # noqa: E402
    IRRIGATION_CASES,
    _events_frame,
    assert_ospy_parity,
    run_grid_kernel,
    run_reference,
)


@pytest.fixture(scope="module")
def grid_inputs(weather_df):
    """A small multi-pixel problem with per-pixel sowing and a masked pixel."""
    from aquacrop import Crop, Soil

    from aquagrid.params import (
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


@pytest.mark.parametrize(
    "crop_name,soil_name,planting,method,options", IRRIGATION_CASES)
def test_gpu_irrigation_matches_ospy(weather_df, crop_name, soil_name,
                                     planting, method, options):
    """Methods 1–5 match AquaCrop-OSPy on the CUDA kernel, including IrrDay."""
    from aquacrop import IrrigationManagement

    irrigation = IrrigationManagement(method, **options)
    model = run_reference(
        weather_df, crop_name, soil_name, planting, irrigation=irrigation)
    fin, daily, _nt = run_grid_kernel(
        weather_df, crop_name, soil_name, planting,
        irrigation=irrigation, backend="gpu")
    assert_ospy_parity(model, fin, daily)
    assert fin[0, C.OF_IRR_CUM] > 0


def test_gpu_schedule_matches_ospy(weather_df):
    from aquacrop import IrrigationManagement

    planting = "2019/05/15"
    events = [("2019-06-10", 20.0), ("2019-07-20", 15.0), ("2019-08-15", 25.0)]
    schedule, frame = _events_frame(weather_df, planting, events)
    irrigation = IrrigationManagement(
        3, Schedule=frame, MaxIrr=40, AppEff=80, WetSurf=70)
    model = run_reference(
        weather_df, "Maize", "SandyLoam", planting, irrigation=irrigation)
    fin, daily, _nt = run_grid_kernel(
        weather_df, "Maize", "SandyLoam", planting,
        irrigation=irrigation, schedule=schedule, backend="gpu")
    assert_ospy_parity(model, fin, daily)
    assert fin[0, C.OF_IRR_CUM] == pytest.approx(20 + 15 + 25)


def test_gpu_day1_pre_irrigation_matches_ospy(weather_df):
    from aquacrop import InitialWaterContent, IrrigationManagement

    planting = "2019/06/01"
    irrigation = IrrigationManagement(4, NetIrrSMT=70)
    initial = InitialWaterContent(value=["WP"])
    model = run_reference(
        weather_df, "SoybeanGDD", "ClayLoam", planting,
        irrigation=irrigation, initial_wc=initial)
    fin, daily, _nt = run_grid_kernel(
        weather_df, "SoybeanGDD", "ClayLoam", planting,
        irrigation=irrigation, initial_wc="WP", backend="gpu")
    assert_ospy_parity(model, fin, daily)
    assert daily[C.OD_IRR, 0, 0] > 1.0
