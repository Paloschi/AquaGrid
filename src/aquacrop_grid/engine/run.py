"""Backend-agnostic grid runner operating on flat pixel arrays."""

from __future__ import annotations

import numpy as np

from aquacrop_grid.kernels.constants import OD_N, OF_N

_HYDRAULIC = (
    "th_fc", "th_s", "th_wp", "th_dry", "ksat", "tau", "penetrability",
)
_GEOMETRY = ("dz", "dzsum", "layer")


def _pixel_rows(arr: np.ndarray, npix: int, name: str) -> np.ndarray:
    """Broadcast a 1-D ``(n,)`` vector to ``(npix, n)``, or check a 2-D array."""
    a = np.ascontiguousarray(arr, np.float64)
    if a.ndim == 1:
        out = np.empty((npix, a.shape[0]), dtype=np.float64)
        out[:] = a
        return out
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {a.shape}")
    if a.shape[0] != npix:
        raise ValueError(
            f"{name} shape {a.shape} does not match npixel={npix}")
    return a


def run_grid_arrays(
    *,
    tmin: np.ndarray,
    tmax: np.ndarray,
    prcp: np.ndarray,
    et0: np.ndarray,
    plant_idx: np.ndarray,
    cp: np.ndarray,
    sp: np.ndarray,
    profile: dict[str, np.ndarray],
    th_init: np.ndarray,
    backend: str = "cpu",
    save_daily: bool = False,
    evap_time_steps: int = 20,
    max_season_days: int = 400,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run one AquaCrop season for every pixel.

    Arguments:
        tmin/tmax/prcp/et0: weather arrays, shape (time, npixel), float64.
        plant_idx: (npixel,) int64 index into the time axis of the sowing
            date for each pixel; negative = do not simulate.
        cp: crop scalar parameter array (see aquacrop_grid.params).
        sp: soil scalars, shape (SP_N,) shared or (npixel, SP_N).
        profile: per-compartment arrays. Geometry (``dz``, ``dzsum``,
            ``layer``) is 1-D ``(ncomp,)``. Hydraulic properties may be
            ``(ncomp,)`` (shared) or ``(npixel, ncomp)``.
        th_init: ``(ncomp,)`` or ``(npixel, ncomp)`` initial θ.

    Returns:
        (out_final (npixel, OF_N), out_daily (OD_N, time, npixel) or None)
    """
    nt, npix = tmin.shape
    ncomp = profile["dz"].shape[0]
    nlayer = int(np.unique(profile["layer"]).shape[0])

    f8 = np.float64
    tmin = np.ascontiguousarray(tmin, f8)
    tmax = np.ascontiguousarray(tmax, f8)
    prcp = np.ascontiguousarray(prcp, f8)
    et0 = np.ascontiguousarray(et0, f8)
    plant_idx = np.ascontiguousarray(plant_idx, np.int64)

    sp2 = _pixel_rows(sp, npix, "sp")
    thini = _pixel_rows(th_init, npix, "th_init")
    hyd = {k: _pixel_rows(profile[k], npix, k) for k in _HYDRAULIC}
    for k in _HYDRAULIC:
        if hyd[k].shape[1] != ncomp:
            raise ValueError(
                f"profile[{k!r}] ncomp {hyd[k].shape[1]} != dz {ncomp}")
    geom = {k: np.ascontiguousarray(profile[k], f8) for k in _GEOMETRY}

    # per-pixel scratch
    th2d = np.zeros((npix, ncomp), f8)
    thnew2d = np.zeros((npix, ncomp), f8)
    flux2d = np.zeros((npix, ncomp), f8)
    aer2d = np.zeros((npix, ncomp), f8)

    out_final = np.zeros((npix, OF_N), f8)
    if save_daily:
        out_daily = np.full((OD_N, nt, npix), np.nan, f8)
    else:
        out_daily = np.zeros((OD_N, 1, 1), f8)

    args = (
        plant_idx, nt, tmin, tmax, prcp, et0,
        np.ascontiguousarray(cp, f8), sp2,
        ncomp, nlayer,
        geom["dz"], geom["dzsum"],
        hyd["th_fc"], hyd["th_s"], hyd["th_wp"], hyd["th_dry"],
        hyd["ksat"], hyd["tau"],
        geom["layer"], hyd["penetrability"], thini,
        th2d, thnew2d, flux2d, aer2d,
        int(evap_time_steps), int(max_season_days),
        out_final, save_daily, out_daily,
    )

    if backend == "cpu":
        from aquacrop_grid.engine.cpu import get_runner

        get_runner(parallel=parallel)(*args)
    elif backend == "gpu":
        from aquacrop_grid.engine.gpu import run_gpu

        run_gpu(*args)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    return out_final, (out_daily if save_daily else None)
