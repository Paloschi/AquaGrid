"""CPU backend: one prange thread per pixel, full season per thread."""

from __future__ import annotations

from numba import njit, prange

from aquagrid.kernels.constants import OF_STATUS, STATUS_NOT_SIMULATED
from aquagrid.kernels.loader import load_kernels

# Bound by get_runner before either runner is called. A module global, not a
# closure: Numba's cache key hashes closure cells, and the two parallel flags
# would otherwise share one nested function and one cache entry.
_run_pixel = None


def get_runner(parallel: bool = True):
    global _run_pixel
    if _run_pixel is None:
        _run_pixel = load_kernels("cpu")["run_pixel"]
    if parallel:
        return _run_all_parallel
    return _run_all_serial


@njit(parallel=True, cache=True)
def _run_all_parallel(plant_idx, nt, tmin2d, tmax2d, prcp2d, et02d,
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
            _run_pixel(p, pi, nt, tmin2d, tmax2d, prcp2d, et02d,
                       cp[p], sp[p], ncomp, nlayer,
                       dz, dzsum, th_fc[p], th_s[p], th_wp[p], th_dry[p],
                       ksat[p], tau[p],
                       layer, penetrability[p], th_init[p],
                       th2d[p], thnew2d[p], flux2d[p], aer2d[p],
                       evap_time_steps, max_season_days,
                       out_final, save_daily, out_daily)


@njit(parallel=False, cache=True)
def _run_all_serial(plant_idx, nt, tmin2d, tmax2d, prcp2d, et02d,
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
            _run_pixel(p, pi, nt, tmin2d, tmax2d, prcp2d, et02d,
                       cp[p], sp[p], ncomp, nlayer,
                       dz, dzsum, th_fc[p], th_s[p], th_wp[p], th_dry[p],
                       ksat[p], tau[p],
                       layer, penetrability[p], th_init[p],
                       th2d[p], thnew2d[p], flux2d[p], aer2d[p],
                       evap_time_steps, max_season_days,
                       out_final, save_daily, out_daily)
