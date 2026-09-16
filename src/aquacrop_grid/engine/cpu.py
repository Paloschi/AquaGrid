"""CPU backend: one prange thread per pixel, full season per thread."""

from __future__ import annotations

from numba import njit, prange

from aquacrop_grid.kernels.constants import OF_STATUS, STATUS_NOT_SIMULATED
from aquacrop_grid.kernels.loader import load_kernels

_RUNNERS: dict = {}


def get_runner(parallel: bool = True):
    key = parallel
    if key in _RUNNERS:
        return _RUNNERS[key]

    ns = load_kernels("cpu")
    run_pixel = ns["run_pixel"]

    @njit(parallel=parallel)
    def run_all(plant_idx, nt, tmin2d, tmax2d, prcp2d, et02d,
                cp, sp, ncomp, nlayer,
                dz, dzsum, th_fc, th_s, th_wp, th_dry, ksat, tau,
                layer, penetrability, th_init,
                th2d, thnew2d, flux2d, aer2d,
                evap_time_steps, max_season_days,
                out_final, save_daily, out_daily):
        npix = plant_idx.shape[0]
        for p in prange(npix):
            pi = plant_idx[p]
            if pi < 0:
                out_final[p, OF_STATUS] = STATUS_NOT_SIMULATED
            else:
                run_pixel(p, pi, nt, tmin2d, tmax2d, prcp2d, et02d,
                          cp, sp[p], ncomp, nlayer,
                          dz, dzsum, th_fc[p], th_s[p], th_wp[p], th_dry[p],
                          ksat[p], tau[p],
                          layer, penetrability[p], th_init[p],
                          th2d[p], thnew2d[p], flux2d[p], aer2d[p],
                          evap_time_steps, max_season_days,
                          out_final, save_daily, out_daily)

    _RUNNERS[key] = run_all
    return run_all
