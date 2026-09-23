"""Build per-pixel AquaCrop soil profiles from a SoilGrid tile."""

from __future__ import annotations

import numpy as np

from aquagrid.io.soil import SoilGrid
from aquagrid.kernels import constants as C

DEPTH_THICKNESS_M = {
    "0-5": 0.05,
    "5-15": 0.10,
    "15-30": 0.15,
    "30-60": 0.30,
    "60-100": 0.40,
    "100-200": 1.00,
}

DEFAULT_HOMOGENEOUS_DZ = np.full(12, 0.1, dtype=np.float64)
EVAP_Z_SURF = 0.04
TEXTURE_SUM_TOLERANCE = 5.0  # percent


def saxton_rawls(
    sand: np.ndarray,
    clay: np.ndarray,
    orgmat: np.ndarray | float = 0.0,
    df: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Saxton & Rawls (2006) PTF, matching aquacrop ``Soil``.

    ``sand`` and ``clay`` are mass fractions (0–1); ``orgmat`` is percent.
    Returns ``(th_wp, th_fc, th_s, ksat_mm_day)`` with AquaCrop rounding.
    """
    sand = np.asarray(sand, np.float64)
    clay = np.asarray(clay, np.float64)
    orgmat = np.asarray(orgmat, np.float64)

    pred_th_wp = (
        -(0.024 * sand)
        + (0.487 * clay)
        + (0.006 * orgmat)
        + (0.005 * sand * orgmat)
        - (0.013 * clay * orgmat)
        + (0.068 * sand * clay)
        + 0.031
    )
    th_wp = pred_th_wp + (0.14 * pred_th_wp) - 0.02

    pred_th_fc = (
        -(0.251 * sand)
        + (0.195 * clay)
        + (0.011 * orgmat)
        + (0.006 * sand * orgmat)
        - (0.027 * clay * orgmat)
        + (0.452 * sand * clay)
        + 0.299
    )
    pred_adj_th_fc = pred_th_fc + (
        (1.283 * np.power(pred_th_fc, 2)) - (0.374 * pred_th_fc) - 0.015
    )

    pred_th_s33 = (
        (0.278 * sand)
        + (0.034 * clay)
        + (0.022 * orgmat)
        - (0.018 * sand * orgmat)
        - (0.027 * clay * orgmat)
        - (0.584 * sand * clay)
        + 0.078
    )
    pred_adj_th_s33 = pred_th_s33 + ((0.636 * pred_th_s33) - 0.107)
    pred_th_s = (pred_adj_th_fc + pred_adj_th_s33) + ((-0.097 * sand) + 0.043)

    p_n = (1 - pred_th_s) * 2.65
    p_df = p_n * df
    poros_comp = (1 - (p_df / 2.65)) - (1 - (p_n / 2.65))
    poros_comp_om = 1 - (p_df / 2.65)

    th_fc = pred_adj_th_fc + (0.2 * poros_comp)
    th_s = poros_comp_om

    with np.errstate(divide="ignore", invalid="ignore"):
        lmbda = 1.0 / (
            (np.log(1500.0) - np.log(33.0)) / (np.log(th_fc) - np.log(th_wp))
        )
        ksat = (1930.0 * (th_s - th_fc) ** (3.0 - lmbda)) * 24.0

    th_wp = np.round(th_wp, 3)
    th_fc = np.round(th_fc, 3)
    th_s = np.round(th_s, 3)
    ksat = np.round(ksat, 1)
    return th_wp, th_fc, th_s, ksat


def tau_from_ksat(ksat: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        tau = np.round(0.0866 * np.power(ksat, 0.35), 2)
    return np.clip(tau, 0.0, 1.0)


def curve_number_from_ksat(ksat0: np.ndarray) -> np.ndarray:
    ksat0 = np.asarray(ksat0, np.float64)
    cn = np.full(ksat0.shape, 77.0, dtype=np.float64)
    cn = np.where(ksat0 > 36.0, 72.0, cn)
    cn = np.where(ksat0 > 347.0, 61.0, cn)
    cn = np.where(ksat0 > 864.0, 46.0, cn)
    return cn


def rew_from_theta(th_fc0: np.ndarray, th_dry0: np.ndarray,
                   evap_z_surf: float = EVAP_Z_SURF) -> np.ndarray:
    return np.round(1000.0 * (th_fc0 - th_dry0) * evap_z_surf, 2)


def deepen_dz(dz: np.ndarray, zmax: float) -> np.ndarray:
    """Append 0.1 m compartments until the profile reaches ``zmax + 0.1``."""
    dz = np.asarray(dz, np.float64).copy()
    z = float(dz.sum())
    target = float(zmax) + 0.1
    extra = []
    while z < target - 1e-12:
        extra.append(0.1)
        z += 0.1
    if extra:
        dz = np.concatenate([dz, np.asarray(extra, np.float64)])
    return dz


def _scalar_sp(npix: int, dz0: float) -> np.ndarray:
    sp = np.zeros((npix, C.SP_N), dtype=np.float64)
    sp[:, C.SP_ADJ_CN] = 1.0
    sp[:, C.SP_Z_CN] = 0.3
    sp[:, C.SP_Z_GERM] = 0.3
    sp[:, C.SP_Z_TOP] = max(0.1, float(dz0))
    sp[:, C.SP_EVAP_Z_MIN] = 0.15
    sp[:, C.SP_EVAP_Z_MAX] = 0.30
    sp[:, C.SP_KEX] = 1.1
    sp[:, C.SP_FWCC] = 50.0
    sp[:, C.SP_F_WREL_EXP] = 0.4
    sp[:, C.SP_F_EVAP] = 4.0
    return sp


def _pack_profile(
    dz: np.ndarray,
    layer_idx: np.ndarray,
    th_wp: np.ndarray,
    th_fc: np.ndarray,
    th_s: np.ndarray,
    ksat: np.ndarray,
    valid: np.ndarray,
    initial_wc: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """``th_*`` / ``ksat`` are (nlayer, npix); ``layer_idx`` maps comps → layer."""
    ncomp = int(layer_idx.shape[0])
    npix = int(th_wp.shape[1])
    th_wp_c = np.ascontiguousarray(th_wp[layer_idx].T, np.float64)
    th_fc_c = np.ascontiguousarray(th_fc[layer_idx].T, np.float64)
    th_s_c = np.ascontiguousarray(th_s[layer_idx].T, np.float64)
    ksat_c = np.ascontiguousarray(ksat[layer_idx].T, np.float64)
    th_dry = th_wp_c / 2.0
    tau = tau_from_ksat(ksat_c)
    penetrability = np.full((npix, ncomp), 100.0, dtype=np.float64)
    dzsum = np.round(np.cumsum(dz), 2)
    layer = (layer_idx + 1).astype(np.float64)

    sp = _scalar_sp(npix, float(dz[0]))
    sp[:, C.SP_CN] = curve_number_from_ksat(ksat_c[:, 0])
    sp[:, C.SP_REW] = rew_from_theta(th_fc_c[:, 0], th_dry[:, 0])

    profile = {
        "dz": np.ascontiguousarray(dz, np.float64),
        "dzsum": np.ascontiguousarray(dzsum, np.float64),
        "th_fc": th_fc_c,
        "th_s": th_s_c,
        "th_wp": th_wp_c,
        "th_dry": np.ascontiguousarray(th_dry, np.float64),
        "ksat": ksat_c,
        "tau": np.ascontiguousarray(tau, np.float64),
        "layer": np.ascontiguousarray(layer, np.float64),
        "penetrability": penetrability,
    }
    kind = initial_wc.upper()
    if kind == "FC":
        th_init = th_fc_c.copy()
    elif kind == "WP":
        th_init = th_wp_c.copy()
    elif kind == "SAT":
        th_init = th_s_c.copy()
    else:
        raise ValueError(f"unsupported initial water content: {initial_wc!r}")
    return sp, profile, th_init, valid


def _layer_index(n_src: int, ncomp: int) -> np.ndarray:
    idx = np.empty(ncomp, dtype=np.int64)
    last = n_src - 1
    for i in range(ncomp):
        idx[i] = i if i < n_src else last
    return idx


def _finite_layers(*arrays: np.ndarray) -> np.ndarray:
    valid = np.ones(arrays[0].shape[1], dtype=bool)
    for arr in arrays:
        valid &= np.isfinite(arr).all(axis=0)
    return valid


def _reshape_layers(da: np.ndarray, npix: int) -> np.ndarray:
    """(nlayer, ny, nx) or (ny, nx) → (nlayer, npix)."""
    if da.ndim == 2:
        return da.reshape(1, npix)
    if da.ndim == 3:
        nlayer = da.shape[0]
        return da.reshape(nlayer, npix)
    raise ValueError(f"expected 2-D or 3-D soil array, got shape {da.shape}")


def _texture_to_hydraulic(store: SoilGrid, npix: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sand = _reshape_layers(np.asarray(store.ds["sand"].values, np.float64), npix)
    clay = _reshape_layers(np.asarray(store.ds["clay"].values, np.float64), npix)
    if "silt" in store.ds:
        silt = _reshape_layers(np.asarray(store.ds["silt"].values, np.float64), npix)
        if silt.shape[0] == 1 and sand.shape[0] > 1:
            silt = np.broadcast_to(silt, sand.shape)
        tex_sum = sand + silt + clay
        valid_sum = np.isfinite(tex_sum) & (
            np.abs(tex_sum - 100.0) <= TEXTURE_SUM_TOLERANCE)
    else:
        valid_sum = np.isfinite(sand) & np.isfinite(clay)

    if "orgmat" in store.ds:
        orgmat = _reshape_layers(
            np.asarray(store.ds["orgmat"].values, np.float64), npix)
        if orgmat.shape[0] == 1 and sand.shape[0] > 1:
            orgmat = np.broadcast_to(orgmat, sand.shape)
    else:
        orgmat = 0.0

    th_wp, th_fc, th_s, ksat = saxton_rawls(sand / 100.0, clay / 100.0, orgmat)
    valid = valid_sum.all(axis=0) & _finite_layers(th_wp, th_fc, th_s, ksat)
    return th_wp, th_fc, th_s, ksat, valid


def _hydraulic_layers(store: SoilGrid, npix: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ksat = _reshape_layers(np.asarray(store.ds["ksat"].values, np.float64), npix)
    th_s = _reshape_layers(np.asarray(store.ds["wcsat"].values, np.float64), npix)
    th_fc = _reshape_layers(np.asarray(store.ds["wcpf2"].values, np.float64), npix)
    th_wp = _reshape_layers(np.asarray(store.ds["wcpf3"].values, np.float64), npix)
    valid = _finite_layers(ksat, th_s, th_fc, th_wp)
    return th_wp, th_fc, th_s, ksat, valid


def _geometry(store: SoilGrid, n_src: int, zmax: float
              ) -> tuple[np.ndarray, np.ndarray]:
    if store.depths:
        dz_src = np.array(
            [DEPTH_THICKNESS_M[d] for d in store.depths], dtype=np.float64)
        if dz_src.shape[0] != n_src:
            raise ValueError(
                f"depth count {len(store.depths)} != layer count {n_src}")
        dz = deepen_dz(dz_src, zmax)
        layer_idx = _layer_index(n_src, dz.shape[0])
        return dz, layer_idx
    dz = deepen_dz(DEFAULT_HOMOGENEOUS_DZ, zmax)
    layer_idx = np.zeros(dz.shape[0], dtype=np.int64)
    return dz, layer_idx


def profiles_from_store(
    store: SoilGrid,
    zmax: float,
    initial_wc: str = "FC",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Flatten a soil tile into kernel arrays.

    Returns ``(sp, profile, th_init, valid)`` with ``sp`` / hydraulic profile
    arrays shaped ``(npix, ...)`` and geometry 1-D. ``valid`` is ``(npix,)``.
    """
    ny, nx = store.shape
    npix = ny * nx
    if store.kind == "hydraulic":
        th_wp, th_fc, th_s, ksat, valid = _hydraulic_layers(store, npix)
    elif store.kind == "texture":
        th_wp, th_fc, th_s, ksat, valid = _texture_to_hydraulic(store, npix)
    else:
        raise ValueError(f"unknown soil kind {store.kind!r}")

    dz, layer_idx = _geometry(store, th_wp.shape[0], zmax)
    return _pack_profile(
        dz, layer_idx, th_wp, th_fc, th_s, ksat, valid, initial_wc)
