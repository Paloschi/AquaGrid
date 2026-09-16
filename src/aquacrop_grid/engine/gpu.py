"""GPU backend: numba.cuda, one thread per pixel, state on device.

All per-pixel state (soil water, scratch buffers, outputs) lives on the
device for the whole simulation; only the final/daily outputs are copied
back to the host at the end.
"""

from __future__ import annotations

from numba import cuda

from aquacrop_grid.kernels.constants import OF_STATUS, STATUS_NOT_SIMULATED
from aquacrop_grid.kernels.loader import load_kernels

_KERNEL = None
THREADS_PER_BLOCK = 64


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL

    ns = load_kernels("gpu")
    run_pixel = ns["run_pixel"]

    @cuda.jit
    def run_all(plant_idx, nt, tmin2d, tmax2d, prcp2d, et02d,
                cp, sp, ncomp, nlayer,
                dz, dzsum, th_fc, th_s, th_wp, th_dry, ksat, tau,
                layer, penetrability, th_init,
                th2d, thnew2d, flux2d, aer2d,
                evap_time_steps, max_season_days,
                out_final, save_daily, out_daily):
        p = cuda.grid(1)
        if p >= plant_idx.shape[0]:
            return
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

    _KERNEL = run_all
    return _KERNEL


def run_gpu(plant_idx, nt, tmin2d, tmax2d, prcp2d, et02d,
            cp, sp, ncomp, nlayer,
            dz, dzsum, th_fc, th_s, th_wp, th_dry, ksat, tau,
            layer, penetrability, th_init,
            th2d, thnew2d, flux2d, aer2d,
            evap_time_steps, max_season_days,
            out_final, save_daily, out_daily):
    """Device-transfer wrapper matching the CPU runner signature."""
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available (numba.cuda)")

    kern = _get_kernel()
    npix = plant_idx.shape[0]

    d = cuda.to_device
    d_out_final = d(out_final)
    d_out_daily = d(out_daily)

    blocks = (npix + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    kern[blocks, THREADS_PER_BLOCK](
        d(plant_idx), nt, d(tmin2d), d(tmax2d), d(prcp2d), d(et02d),
        d(cp), d(sp), ncomp, nlayer,
        d(dz), d(dzsum), d(th_fc), d(th_s), d(th_wp), d(th_dry),
        d(ksat), d(tau), d(layer), d(penetrability), d(th_init),
        d(th2d), d(thnew2d), d(flux2d), d(aer2d),
        evap_time_steps, max_season_days,
        d_out_final, save_daily, d_out_daily,
    )
    cuda.synchronize()

    d_out_final.copy_to_host(out_final)
    if save_daily:
        d_out_daily.copy_to_host(out_daily)
