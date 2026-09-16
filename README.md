# AquaCrop-Grid

Pixel-wise AquaCrop on rasters: climate as zarr cubes `(time, y, x)`,
per-pixel sowing dates `(y, x)`, and zarr outputs. Daily-step kernels from
[AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) 3.x are vendored and
recompiled with Numba — `njit` + `prange` on CPU (one thread per pixel) and
`numba.cuda` on GPU — with **bit-exact parity** against AquaCrop-OSPy
(`tests/test_parity.py`).

This is not official FAO AquaCrop and not an aquacropos extension. It started
in [CyMP](https://github.com/Paloschi/CyMP) (Unioeste-LEA).

Current scope: rainfed (no irrigation, no groundwater), one season per pixel.
Soil: a single AquaCrop preset **or** a per-pixel zarr raster (HiHydroSoil
hydraulics or sand/silt/clay texture).

## Install

```bash
pip install -e .[dev]
```

GPU: NVIDIA GPU with a CUDA driver (Numba talks to the driver; the full CUDA
toolkit is not required).

## Quick start

```bash
# 1. synthetic 10x10 pixels / 540 days
aquacrop-grid synth --out ./data

# 2. config
cp examples/config.example.yaml ./data/config.yaml
# (adjust paths if needed)

# 3. run
aquacrop-grid run --config ./data/config.yaml            # CPU
aquacrop-grid run --config ./data/config.yaml -b gpu     # GPU
```

Output: `output.zarr` with final yield/biomass `(y, x)` and, with
`save_daily: true`, daily series `(time, y, x)` in group `daily`.
Full schema: [`docs/zarr-schema.md`](docs/zarr-schema.md).

Programmatic:

```python
from aquacrop_grid.pipeline import run_grid

run_grid("climate.zarr", "sowing.zarr", "output.zarr",
         crop_name="Maize", soil_name="SandyLoam",
         backend="cpu", save_daily=False)
```

## YAML config

```yaml
climate: data/climate.zarr      # cube (time, y, x): tmin, tmax, precip, eto
sowing: data/sowing.zarr        # grid (y, x) int32 YYYYDDD; <=0 = masked
output: data/output.zarr
crop:
  name: Maize                   # any AquaCrop-OSPy crop
soil:
  name: SandyLoam               # AquaCrop preset (xor with zarr below)
  # zarr: data/soil.zarr        # ksat/wcsat/wcpf2/wcpf3 or sand/silt/clay
  # ksat_unit: cm/d
backend: cpu                    # cpu | gpu
options:
  save_daily: false
  tile: 128                     # spatial tile size (pixels)
  max_season_days: 400
```

## Benchmark

`aquacrop-grid bench --pixels 65536 --days 540` (excluding JIT compile):

| backend | reference hardware | throughput |
|---------|--------------------|-----------|
| CPU (`njit`+`prange`) | Ryzen (all threads) | ~58k pixels/s |
| GPU (`numba.cuda`)    | RTX 3060            | ~61k pixels/s |

One full season (540 days) per pixel. On larger grids the GPU scales better
(climate transfer dominates on small grids).

## Tests

```bash
pytest            # parity vs AquaCrop-OSPy, io, grid driver, gpu (if present)
```

- `test_parity.py` — single pixel vs AquaCrop-OSPy, 1e-12 tolerance
  (maize/soybean, calendar and GDD, 3 soils).
- `test_grid.py` — grid driver: mask, per-pixel sowing, tiles, grid vs
  single-pixel equality.
- `test_soil.py` — hydraulic/texture zarr, Saxton–Rawls PTF, per-pixel soil.
- `test_gpu.py` — CPU vs GPU parity (skipped without CUDA).

## Architecture

- `src/aquacrop_grid/kernels/impl.py` — AquaCrop daily step ported to a
  common `njit`/`cuda.jit` subset (no allocation in kernels, scalar state
  per pixel + 1-D compartment views). One source compiled for both backends
  by `kernels/loader.py`.
- `src/aquacrop_grid/params.py` — flattens aquacrop `Crop`/`Soil` into kernel
  arrays (replicates weather-independent init, including deepening the
  profile to `Zmax + 0.1`).
- `src/aquacrop_grid/engine/` — `cpu.py` (prange), `gpu.py` (cuda, state on
  device), `run.py` (shared runner over flat arrays).
- `src/aquacrop_grid/pipeline.py` — zarr in → tiles → zarr out.
- `src/aquacrop_grid/io/` — schema (climate, sowing, soil), validation, and
  synthetic data.
- `src/aquacrop_grid/soil_grid.py` — PTF and HiHydro layers → per-pixel
  compartments.

The GDD phenology calendar is computed inside the kernel per pixel (sowing
date changes each pixel's thermal accumulation), mirroring aquacrop
`compute_crop_calendar`.

## License and attribution

GPL-3.0-or-later. Kernels in `src/aquacrop_grid/kernels/impl.py` are derived
from [AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) (MIT).
