"""Soil zarr schema: hydraulic (HiHydroSoil-style) or texture (sand/silt/clay).

See ``docs/zarr-schema.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

HIHYDRO_KSAT_FACTOR = 0.001  # integer DN * 0.0001 cm/d * 10 -> mm/d
HIHYDRO_WC_FACTOR = 0.0001

HIHYDRO_VAR_ORDER = ("Ksat", "WCsat", "WCpF2", "WCpF3")
ACTIVE_DEPTHS = ("0-5", "5-15", "15-30", "30-60", "60-100")
OPTIONAL_DEPTHS = ("100-200",)

HYDRO_SCALE_FACTORS = {
    "ksat": HIHYDRO_KSAT_FACTOR,
    "wcsat": HIHYDRO_WC_FACTOR,
    "wcpf2": HIHYDRO_WC_FACTOR,
    "wcpf3": HIHYDRO_WC_FACTOR,
}

DEPTH_ALIASES = {
    "000-005cm": "0-5",
    "0-5cm": "0-5",
    "0-5": "0-5",
    "05-15cm": "5-15",
    "5-15cm": "5-15",
    "5-15": "5-15",
    "15-30cm": "15-30",
    "15-30": "15-30",
    "30-60cm": "30-60",
    "30-60": "30-60",
    "60-100cm": "60-100",
    "60-100": "60-100",
    "100-200cm": "100-200",
    "100-200": "100-200",
}

HYDRO_VARS = ("ksat", "wcsat", "wcpf2", "wcpf3")
HYDRO_ALIASES = {
    "ksat": ("ksat", "Ksat", "KSAT"),
    "wcsat": ("wcsat", "WCsat", "WCSAT"),
    "wcpf2": ("wcpf2", "WCpF2", "WCPF2"),
    "wcpf3": ("wcpf3", "WCpF3", "WCPF3"),
}
TEXTURE_ALIASES = {
    "sand": ("sand", "Sand", "areia", "Areia"),
    "silt": ("silt", "Silt", "silte", "Silte"),
    "clay": ("clay", "Clay", "argila", "Argila"),
    "orgmat": ("orgmat", "OrgMat", "om", "OM", "ormc", "ORMC"),
}


@dataclass
class SoilGrid:
    """Opened, unit-converted soil store ready to tile."""

    kind: str  # "hydraulic" | "texture"
    ds: xr.Dataset
    depths: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.ds.sizes["y"]), int(self.ds.sizes["x"])

    def isel(self, **indexers) -> SoilGrid:
        return SoilGrid(self.kind, self.ds.isel(**indexers), self.depths)


def canonicalize_depth(label) -> str | None:
    """Map a depth label to a canonical ``'0-5'``-style key, or None."""
    s = str(label).strip()
    if s in DEPTH_ALIASES:
        return DEPTH_ALIASES[s]
    sl = s.lower().replace(" ", "")
    for key, canon in DEPTH_ALIASES.items():
        if key.lower().replace(" ", "") == sl:
            return canon
    return None


def _lookup_var(ds: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray | None:
    lower = {str(k).lower(): k for k in ds.data_vars}
    for name in names:
        if name in ds:
            return ds[name]
        key = lower.get(name.lower())
        if key is not None:
            return ds[key]
    return None


def _depth_dim(da: xr.DataArray) -> str | None:
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(
            f"soil variable {da.name!r} must include dims y, x; got {da.dims}")
    extra = [d for d in da.dims if d not in ("y", "x")]
    if not extra:
        return None
    if len(extra) == 1:
        return extra[0]
    raise ValueError(
        f"soil variable {da.name!r} has extra dims {extra}, expected one depth dim")


def _select_depths(da: xr.DataArray, depth_dim: str) -> tuple[xr.DataArray, tuple[str, ...]]:
    labels = [_canonicalize_or_keep(v) for v in da[depth_dim].values]
    da = da.assign_coords({depth_dim: labels})
    present = []
    for depth in ACTIVE_DEPTHS + OPTIONAL_DEPTHS:
        if depth in labels:
            present.append(depth)
    missing = [d for d in ACTIVE_DEPTHS if d not in present]
    if missing:
        raise ValueError(
            f"soil store missing required depth(s) {missing}; "
            f"have {sorted(set(labels))}")
    ordered = tuple(d for d in ACTIVE_DEPTHS + OPTIONAL_DEPTHS if d in present)
    return da.sel({depth_dim: list(ordered)}).rename({depth_dim: "depth"}), ordered


def _canonicalize_or_keep(label):
    canon = canonicalize_depth(label)
    return canon if canon is not None else str(label).strip()


def _apply_hydro_scale(
    ds: xr.Dataset,
    *,
    ksat_unit: str,
    scale_factors: dict[str, float] | None,
) -> xr.Dataset:
    out = {}
    integerish = any(
        np.issubdtype(ds[name].dtype, np.integer) for name in HYDRO_VARS)
    if not integerish:
        wc = np.asarray(ds["wcsat"].values, np.float64)
        wc_max = np.nanmax(wc) if wc.size else 0.0
        integerish = np.isfinite(wc_max) and wc_max > 1.5

    for name in HYDRO_VARS:
        da = ds[name].astype(np.float64)
        if scale_factors is not None and name in scale_factors:
            out[name] = da * float(scale_factors[name])
            continue
        if integerish:
            out[name] = da * float(HYDRO_SCALE_FACTORS[name])
            continue
        if name == "ksat":
            unit = ksat_unit.lower().replace(" ", "")
            if unit in ("cm/d", "cm/day", "cmd"):
                da = da * 10.0
            elif unit not in ("mm/d", "mm/day", "mmd"):
                raise ValueError(
                    f"unsupported ksat_unit {ksat_unit!r}; use 'cm/d' or 'mm/d'")
        out[name] = da
    return xr.Dataset(out, coords=ds.coords)


def _normalize_texture_percent(da: xr.DataArray) -> xr.DataArray:
    """Return mass percent 0–100 (input may already be a 0–1 fraction)."""
    arr = da.astype(np.float64)
    finite = np.asarray(arr.values)
    finite = finite[np.isfinite(finite)]
    if finite.size and np.nanmax(np.abs(finite)) <= 1.5:
        arr = arr * 100.0
    return arr


def open_soil(
    store: str | Path,
    *,
    ksat_unit: str = "cm/d",
    scale_factors: dict[str, float] | None = None,
) -> SoilGrid:
    """Open and validate a soil zarr store; convert to AquaCrop units."""
    ds = xr.open_zarr(store, decode_timedelta=False)
    hydro = {name: _lookup_var(ds, aliases) for name, aliases in HYDRO_ALIASES.items()}
    texture = {name: _lookup_var(ds, aliases) for name, aliases in TEXTURE_ALIASES.items()}
    has_hydro = all(hydro[n] is not None for n in HYDRO_VARS)
    has_texture = texture["sand"] is not None and texture["clay"] is not None
    if has_hydro:
        return _open_hydraulic(hydro, ksat_unit=ksat_unit,
                               scale_factors=scale_factors)
    if has_texture:
        return _open_texture(texture)
    raise ValueError(
        "soil store must contain hydraulic variables "
        f"{list(HYDRO_VARS)} or texture variables sand/clay "
        "(silt optional); none found")


def _align_vars(vars_: dict[str, xr.DataArray]) -> tuple[dict[str, xr.DataArray], tuple[str, ...]]:
    depth_dims = {name: _depth_dim(da) for name, da in vars_.items() if da is not None}
    names = [n for n, da in vars_.items() if da is not None]
    dims = {depth_dims[n] for n in names}
    if len(dims) != 1:
        raise ValueError(
            f"soil variables have inconsistent depth dims: {depth_dims}")
    depth_dim = next(iter(dims))
    if depth_dim is None:
        out = {n: vars_[n].transpose("y", "x") for n in names}
        return out, ()
    selected = None
    ordered = None
    out = {}
    for name in names:
        da, ordered_i = _select_depths(vars_[name], depth_dim)
        if selected is None:
            selected, ordered = da, ordered_i
        elif ordered_i != ordered:
            raise ValueError("soil variables have inconsistent depth coordinates")
        out[name] = da.transpose("depth", "y", "x")
    return out, ordered or ()


def _open_hydraulic(
    hydro: dict[str, xr.DataArray],
    *,
    ksat_unit: str,
    scale_factors: dict[str, float] | None,
) -> SoilGrid:
    aligned, depths = _align_vars(hydro)
    ds = xr.Dataset(aligned)
    ds = _apply_hydro_scale(ds, ksat_unit=ksat_unit, scale_factors=scale_factors)
    return SoilGrid("hydraulic", ds, depths)


def _open_texture(texture: dict[str, xr.DataArray | None]) -> SoilGrid:
    aligned, depths = _align_vars({"sand": texture["sand"], "clay": texture["clay"]})
    for opt in ("silt", "orgmat"):
        da = texture.get(opt)
        if da is None:
            continue
        ddim = _depth_dim(da)
        if ddim is None:
            aligned[opt] = da.transpose("y", "x")
        else:
            extra, ordered = _select_depths(da, ddim)
            if depths and ordered != depths:
                raise ValueError(
                    f"{opt} depths {ordered} do not match sand/clay {depths}")
            aligned[opt] = extra.transpose("depth", "y", "x")
    data = {}
    for name, da in aligned.items():
        if name == "orgmat":
            data[name] = da.astype(np.float64)
        else:
            data[name] = _normalize_texture_percent(da)
    return SoilGrid("texture", xr.Dataset(data), depths)


def validate_soil_grid(soil: SoilGrid, ny: int, nx: int) -> None:
    """Raise if the soil grid is not aligned with a (ny, nx) climate grid."""
    sy, sx = soil.shape
    if (sy, sx) != (ny, nx):
        raise ValueError(
            f"soil grid {(sy, sx)} does not match climate grid {(ny, nx)}")
