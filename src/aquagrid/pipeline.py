"""Grid pipeline: zarr in -> tiled CPU/GPU simulation -> zarr out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aquagrid.engine.run import run_grid_arrays
from aquagrid.io.schema import open_climate, open_sowing, sowing_to_plant_idx
from aquagrid.io.soil import SoilGrid, open_soil, validate_soil_grid
from aquagrid.kernels import constants as C
from aquagrid.params import (
    apply_crop_cube,
    co2_concentration_for_year,
    crop_params_array,
    initial_water_content,
    soil_params,
)
from aquagrid.soil_grid import profiles_from_store

FINAL_VARS = {
    # name -> (OF index, dtype)
    "status": (C.OF_STATUS, np.int8),
    "dry_yield": (C.OF_DRY_YIELD, np.float32),
    "fresh_yield": (C.OF_FRESH_YIELD, np.float32),
    "yield_pot": (C.OF_YIELD_POT, np.float32),
    "biomass": (C.OF_BIOMASS, np.float32),
    "biomass_ns": (C.OF_BIOMASS_NS, np.float32),
    "hi_adj": (C.OF_HI_ADJ, np.float32),
    "dap_end": (C.OF_DAP_END, np.int16),
}

DAILY_VARS = {
    "canopy_cover": C.OD_CANOPY_COVER,
    "biomass": C.OD_BIOMASS,
    "z_root": C.OD_Z_ROOT,
    "gdd_cum": C.OD_GDD_CUM,
    "es": C.OD_ES,
    "tr": C.OD_TR,
    "wr": C.OD_WR,
}


def _output_templates(time, y, x, save_daily):
    """Lazy (dask) datasets used to initialise the output zarr stores."""
    import dask.array as da

    ny, nx = len(y), len(x)
    final = xr.Dataset(
        {
            name: (("y", "x"), da.zeros((ny, nx), chunks=(ny, nx), dtype=dt))
            for name, (_, dt) in FINAL_VARS.items()
        },
        coords={"y": y, "x": x},
    )
    daily = None
    if save_daily:
        nt = len(time)
        daily = xr.Dataset(
            {
                name: (("time", "y", "x"),
                       da.full((nt, ny, nx), np.nan,
                               chunks=(nt, ny, nx), dtype=np.float32))
                for name in DAILY_VARS
            },
            coords={"time": time, "y": y, "x": x},
        )
    return final, daily


def _open_crop_cube(store: str | Path) -> xr.DataArray:
    """Open a crop-parameter cube. The data variable is ``crop``, or the
    only array with dims ``(param, y, x)``."""
    ds = xr.open_zarr(store)
    if "crop" in ds.data_vars:
        da = ds["crop"]
    else:
        found = [name for name, var in ds.data_vars.items()
                 if set(var.dims) == {"param", "y", "x"}]
        if len(found) != 1:
            raise ValueError(
                "crop store needs a variable 'crop' with dims (param, y, x)")
        da = ds[found[0]]
    if set(da.dims) != {"param", "y", "x"}:
        raise ValueError(
            f"crop cube dims must be (param, y, x), got {da.dims}")
    return da.transpose("param", "y", "x")


def run_grid(
    climate: str | Path,
    sowing: str | Path,
    output: str | Path,
    crop_name: str,
    soil_name: str | None = None,
    *,
    soil_zarr: str | Path | None = None,
    crop_zarr: str | Path | None = None,
    ksat_unit: str = "cm/d",
    scale_factors: dict[str, float] | None = None,
    sowing_var: str = "sowing",
    initial_wc: str = "FC",
    backend: str = "cpu",
    save_daily: bool = False,
    tile: int = 128,
    parallel: bool = True,
    evap_time_steps: int = 20,
    max_season_days: int = 400,
) -> Path:
    """Run AquaCrop over a grid, tile by tile, writing results to zarr.

    ``climate`` and ``sowing`` follow ``docs/zarr-schema.md``. ``output``
    holds final fields at the root and, when ``save_daily`` is true, daily
    fields in group ``daily``. A day outside a pixel's season is NaN.

    Provide exactly one of ``soil_name`` (one AquaCrop preset for the
    grid) or ``soil_zarr`` (per-pixel hydraulic or texture raster).
    ``ksat_unit`` (``cm/d`` or ``mm/d``) converts float hydraulic Ksat.
    ``scale_factors`` maps ``ksat``, ``wcsat``, ``wcpf2`` and ``wcpf3``
    to multipliers and replaces that conversion for each name present.
    Both apply only to a hydraulic raster; a preset or a texture raster
    ignores them. Pass lowercase names here; ``run_from_config``
    lowercases YAML keys.

    ``crop_name`` selects the AquaCrop crop. ``crop_zarr``, when given,
    is a ``(param, y, x)`` cube of per-pixel overrides. A missing layer
    or a NaN cell keeps the named-crop value.

    ``sowing_var`` is the variable in the sowing store (default
    ``sowing``). ``initial_wc`` is ``FC``, ``WP`` or ``SAT`` on either
    soil mode. A pixel is not simulated when sowing is ``<= 0``, the
    sowing date falls outside the climate time axis, or the soil is NaN
    (status 1).

    ``backend`` is ``cpu`` or ``gpu``. ``tile`` is the tile edge in
    pixels. ``parallel`` runs one CPU thread per pixel; the GPU backend
    ignores it. ``evap_time_steps`` is the number of soil-evaporation
    substeps per day. ``max_season_days`` caps days after planting.
    Reaching that cap, maturity, or canopy death records status 0. Status
    3 means the climate series ended before any of those. The schema
    lists all four codes.
    """
    from aquacrop import Crop, Soil

    if (soil_name is None) == (soil_zarr is None):
        raise ValueError("provide exactly one of soil_name or soil_zarr")

    ds = open_climate(climate)
    sow = open_sowing(sowing, var=sowing_var)
    if sow.shape != (ds.sizes["y"], ds.sizes["x"]):
        raise ValueError(
            f"sowing grid {sow.shape} does not match climate grid "
            f"({ds.sizes['y']}, {ds.sizes['x']})")

    soil_grid: SoilGrid | None = None
    if soil_zarr is not None:
        soil_grid = open_soil(
            soil_zarr, ksat_unit=ksat_unit, scale_factors=scale_factors)
        validate_soil_grid(soil_grid, ds.sizes["y"], ds.sizes["x"])

    time = pd.DatetimeIndex(ds.time.values)
    sow_np = sow.values
    plant_idx = sowing_to_plant_idx(sow_np, time)

    valid = sow_np > 0
    if not valid.any():
        raise ValueError("sowing grid has no valid (>0) pixels")
    year = int(np.median(sow_np[valid] // 1000))
    base_doy = int(np.median(sow_np[valid] % 1000))
    nominal = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
        days=base_doy - 1)

    crop = Crop(crop_name, planting_date=nominal.strftime("%m/%d"))
    co2 = co2_concentration_for_year(year)
    cp = crop_params_array(crop, co2_conc=co2)
    if crop_zarr is not None:
        cube = _open_crop_cube(crop_zarr)
        if cube.sizes["y"] != ds.sizes["y"] or cube.sizes["x"] != ds.sizes["x"]:
            raise ValueError(
                f"crop cube {(cube.sizes['y'], cube.sizes['x'])} does not "
                f"match climate grid ({ds.sizes['y']}, {ds.sizes['x']})")
        cp = apply_crop_cube(cp, cube)
        zmax = float(np.nanmax(cp[:, :, C.CP_ZMAX]))
    else:
        zmax = float(crop.Zmax)

    if soil_grid is None:
        soil = Soil(soil_name)
        sp, prof = soil_params(soil, zmax=zmax)
        thini = initial_water_content(soil, initial_wc)
        soil_ctx = {"mode": "named", "sp": sp, "prof": prof, "thini": thini}
    else:
        soil_ctx = {
            "mode": "zarr",
            "grid": soil_grid,
            "zmax": zmax,
            "initial_wc": initial_wc,
        }

    # initialise output stores (metadata only)
    output = Path(output)
    final_t, daily_t = _output_templates(time, ds.y.values, ds.x.values,
                                         save_daily)
    final_t.to_zarr(output, mode="w", compute=False)
    if daily_t is not None:
        daily_t.to_zarr(output, group="daily", mode="a", compute=False)

    ny, nx = sow_np.shape
    for y0 in range(0, ny, tile):
        y1 = min(y0 + tile, ny)
        for x0 in range(0, nx, tile):
            x1 = min(x0 + tile, nx)
            _run_tile(ds, plant_idx, cp, soil_ctx, time,
                      y0, y1, x0, x1, output, backend, save_daily,
                      parallel, evap_time_steps, max_season_days)
    return output


def _run_tile(ds, plant_idx, cp, soil_ctx, time,
              y0, y1, x0, x1, output, backend, save_daily,
              parallel, evap_time_steps, max_season_days):
    tny, tnx = y1 - y0, x1 - x0
    npix = tny * tnx
    nt = len(time)

    sub = ds.isel(y=slice(y0, y1), x=slice(x0, x1))
    weather = {
        name: sub[name].values.astype(np.float64).reshape(nt, npix)
        for name in ("tmin", "tmax", "precip", "eto")
    }
    pidx = plant_idx[y0:y1, x0:x1].reshape(npix)
    if cp.ndim == 3:
        cp = cp[y0:y1, x0:x1].reshape(npix, cp.shape[-1])

    if soil_ctx["mode"] == "named":
        sp, prof, thini = soil_ctx["sp"], soil_ctx["prof"], soil_ctx["thini"]
    else:
        tile_soil = soil_ctx["grid"].isel(y=slice(y0, y1), x=slice(x0, x1))
        sp, prof, thini, soil_ok = profiles_from_store(
            tile_soil, soil_ctx["zmax"], soil_ctx["initial_wc"])
        pidx = pidx.copy()
        pidx[~soil_ok] = -1

    fin, daily = run_grid_arrays(
        tmin=weather["tmin"], tmax=weather["tmax"],
        prcp=weather["precip"], et0=weather["eto"],
        plant_idx=pidx, cp=cp, sp=sp, profile=prof, th_init=thini,
        backend=backend, save_daily=save_daily, parallel=parallel,
        evap_time_steps=evap_time_steps, max_season_days=max_season_days,
    )

    region = {"y": slice(y0, y1), "x": slice(x0, x1)}
    final_ds = xr.Dataset(
        {
            name: (("y", "x"),
                   fin[:, of].reshape(tny, tnx).astype(dt))
            for name, (of, dt) in FINAL_VARS.items()
        },
        coords={"y": ds.y.values[y0:y1], "x": ds.x.values[x0:x1]},
    )
    final_ds.drop_vars(["y", "x"]).to_zarr(output, region=region)

    if save_daily:
        daily_ds = xr.Dataset(
            {
                name: (("time", "y", "x"),
                       daily[od].reshape(nt, tny, tnx).astype(np.float32))
                for name, od in DAILY_VARS.items()
            },
            coords={"time": time,
                    "y": ds.y.values[y0:y1], "x": ds.x.values[x0:x1]},
        )
        daily_ds.drop_vars(["time", "y", "x"]).to_zarr(
            output, group="daily",
            region={"time": slice(0, nt), **region})


def run_from_config(config: str | Path, backend: str | None = None) -> Path:
    """Run a gridded simulation described by a YAML config file.

    ``backend``, when passed, overrides the file. Keys and defaults:

    - ``climate``, ``sowing``, ``output`` — zarr paths
    - ``sowing_var`` (``sowing``) — variable name in the sowing store
    - ``initial_water_content`` (``FC``) — ``FC``, ``WP`` or ``SAT``
    - ``crop.name`` — AquaCrop crop; ``crop.zarr`` — optional cube
    - ``soil.name`` or ``soil.zarr`` — exactly one soil source
    - ``soil.ksat_unit`` (``cm/d``) — float hydraulic Ksat unit
    - ``soil.scale_factors`` — hydraulic multipliers; ignored otherwise
    - ``backend`` (``cpu``) — ``cpu`` or ``gpu``
    - ``options.save_daily`` (false), ``options.tile`` (128),
      ``options.parallel`` (true), ``options.evap_time_steps`` (20),
      ``options.max_season_days`` (400)

    Parameter meanings match ``run_grid``. Field layout and status codes
    are in ``docs/zarr-schema.md``.
    """
    import yaml

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    opts = cfg.get("options", {})
    soil_cfg = cfg.get("soil") or {}
    scale = soil_cfg.get("scale_factors")
    if scale is not None:
        scale = {str(k).lower(): float(v) for k, v in scale.items()}
    return run_grid(
        climate=cfg["climate"],
        sowing=cfg["sowing"],
        output=cfg["output"],
        crop_name=cfg["crop"]["name"],
        soil_name=soil_cfg.get("name"),
        soil_zarr=soil_cfg.get("zarr"),
        crop_zarr=(cfg.get("crop") or {}).get("zarr"),
        ksat_unit=soil_cfg.get("ksat_unit", "cm/d"),
        scale_factors=scale,
        sowing_var=cfg.get("sowing_var", "sowing"),
        initial_wc=cfg.get("initial_water_content", "FC"),
        backend=backend or cfg.get("backend", "cpu"),
        save_daily=bool(opts.get("save_daily", False)),
        tile=int(opts.get("tile", 128)),
        parallel=bool(opts.get("parallel", True)),
        evap_time_steps=int(opts.get("evap_time_steps", 20)),
        max_season_days=int(opts.get("max_season_days", 400)),
    )
