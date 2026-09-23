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


def run_grid(
    climate: str | Path,
    sowing: str | Path,
    output: str | Path,
    crop_name: str,
    soil_name: str | None = None,
    *,
    soil_zarr: str | Path | None = None,
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

    ``climate``/``sowing`` are input zarr stores following the schema in
    ``docs/zarr-schema.md``; ``output`` is the target store (final outputs
    at the root group, daily outputs in group ``daily`` if ``save_daily``).

    Provide exactly one of ``soil_name`` (AquaCrop preset, one profile for
    the grid) or ``soil_zarr`` (per-pixel hydraulic or texture raster).
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

    if soil_grid is None:
        soil = Soil(soil_name)
        sp, prof = soil_params(soil, zmax=crop.Zmax)
        thini = initial_water_content(soil, initial_wc)
        soil_ctx = {"mode": "named", "sp": sp, "prof": prof, "thini": thini}
    else:
        soil_ctx = {
            "mode": "zarr",
            "grid": soil_grid,
            "zmax": crop.Zmax,
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
    """Run a gridded simulation described by a YAML config file."""
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
