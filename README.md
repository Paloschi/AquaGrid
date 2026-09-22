<div align="center">
<a href="https://github.com/Paloschi/aquacrop-grid"><img alt="AquaCrop-Grid" src="https://img.shields.io/badge/AquaCrop--Grid-raster%20AquaCrop-0F766E?style=for-the-badge&labelColor=134E4A"></a>
<h1>AquaCrop-Grid</h1>
<p><strong>Pixel-wise AquaCrop on rasters: zarr in, Numba kernels out — CPU or GPU.</strong></p>
<p>
<a href="https://pypi.org/project/aquacrop-grid/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aquacrop-grid.svg?style=flat-square&color=0F766E"></a>
<a href="https://github.com/Paloschi/aquacrop-grid/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/Paloschi/aquacrop-grid/actions/workflows/test.yml/badge.svg?branch=main"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
<a href="#license-and-attribution"><img alt="License" src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square"></a>
<img alt="Visitors" src="https://api.visitorbadge.io/api/visitors?path=github.com%2FPaloschi%2Facquacrop-grid&label=Visitors&countColor=%230F766E&style=flat">
</p>
<p>
<img alt="Numba" src="https://img.shields.io/badge/Numba-CPU%20%7C%20CUDA-00A3E0?style=flat-square">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat-square&logo=numpy&logoColor=white">
<img alt="xarray" src="https://img.shields.io/badge/xarray-2024%2B-4D8BBD?style=flat-square">
<img alt="zarr" src="https://img.shields.io/badge/zarr-2.16%2B-F9C642?style=flat-square&labelColor=1A1A1A">
<img alt="Dask" src="https://img.shields.io/badge/Dask-2024%2B-FFC107?style=flat-square&logo=dask&logoColor=black">
<img alt="pytest" src="https://img.shields.io/badge/pytest-8-0A9EDC?style=flat-square&logo=pytest&logoColor=white">
<img alt="AquaCrop-OSPy" src="https://img.shields.io/badge/AquaCrop--OSPy-3.x-0F766E?style=flat-square">
</p>
<p>
<img alt="Scope" src="https://img.shields.io/badge/scope-rainfed%20%7C%20one%20season-134E4A?style=flat-square">
<img alt="Parity" src="https://img.shields.io/badge/parity-1e--12%20vs%20OSPy-0F766E?style=flat-square">
<a href="https://www.conventionalcommits.org/"><img alt="Conventional Commits" src="https://img.shields.io/badge/commits-conventional-FE5196?style=flat-square&logo=conventionalcommits&logoColor=white"></a>
</p>
<p>
<a href="#quick-start">Quick start</a> ·
<a href="#yaml-config">Config</a> ·
<a href="#architecture">Architecture</a> ·
<a href="#io-schema">Schema</a> ·
<a href="#testing">Testing</a> ·
<a href="#citing-this-project">Citing</a> ·
<a href="#license-and-attribution">License</a>
</p>
</div>

---

Climate as zarr cubes `(time, y, x)`, per-pixel sowing dates `(y, x)`, zarr
outputs. Daily-step kernels from
[AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) 3.x are vendored and
recompiled with Numba — `njit` + `prange` on CPU (one thread per pixel) and
`numba.cuda` on GPU — with **bit-exact parity** against AquaCrop-OSPy
(`tests/test_parity.py`).

This is **not** official FAO AquaCrop and not an aquacropos extension. It
started in [CyMP](https://github.com/Paloschi/CyMP) (Unioeste-LEA).

| Layer      | Stack                                                                      |
| ---------- | -------------------------------------------------------------------------- |
| Language   | **Python 3.11+**                                                           |
| Kernels    | **Numba** (`njit` + `prange` / `numba.cuda`)                               |
| Arrays     | **NumPy**, **xarray**, **zarr**, **Dask**                                  |
| Reference  | **AquaCrop-OSPy** ≥ 3.0.11 (crop/soil params + parity tests)               |
| CLI        | **Typer** (`aquacrop-grid`)                                                |
| Tests      | **pytest** 8 — Ubuntu & Windows, Python 3.11 / 3.12                        |

**Current scope:** rainfed (no irrigation, no groundwater), one season per
pixel. Soil: a single AquaCrop preset **or** a per-pixel zarr raster
(HiHydroSoil hydraulics or sand/silt/clay texture).

---

## Prerequisites

- **Python 3.11+**
- GPU (optional): NVIDIA GPU with a CUDA **driver** — Numba talks to the
  driver; the full CUDA toolkit is not required

---

## Quick start

```bash
pip install aquacrop-grid
```

From Git (no clone, before or besides PyPI):

```bash
pip install "git+https://github.com/Paloschi/aquacrop-grid.git"
```

For development, clone and install editable:

```bash
git clone https://github.com/Paloschi/aquacrop-grid.git
cd aquacrop-grid
pip install -e ".[dev]"
```

```bash
# 1. synthetic 10×10 pixels / 540 days
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

### Programmatic

```python
from aquacrop_grid.pipeline import run_grid

run_grid("climate.zarr", "sowing.zarr", "output.zarr",
         crop_name="Maize", soil_name="SandyLoam",
         backend="cpu", save_daily=False)
```

---

## CLI

| Command | Purpose |
| ------- | ------- |
| `aquacrop-grid synth -o ./data` | Synthetic climate + sowing zarr for tests |
| `aquacrop-grid run -c config.yaml` | Gridded simulation from YAML (`-b cpu\|gpu`) |
| `aquacrop-grid bench` | Throughput (pixels/s), excluding JIT compile |

---

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

Copy from [`examples/config.example.yaml`](examples/config.example.yaml).

---

## Project structure

```
aquacrop-grid/
├── docs/                    # zarr I/O schema
├── examples/                # sample YAML config
├── src/aquacrop_grid/
│   ├── cli.py               # Typer: synth / run / bench
│   ├── pipeline.py          # zarr in → tiles → zarr out
│   ├── params.py            # Crop / Soil → kernel arrays
│   ├── soil_grid.py         # PTF + HiHydro layers → compartments
│   ├── bench.py
│   ├── engine/
│   │   ├── cpu.py           # njit + prange
│   │   ├── gpu.py           # numba.cuda
│   │   └── run.py           # shared runner over flat arrays
│   ├── kernels/
│   │   ├── impl.py          # daily step (njit / cuda.jit subset)
│   │   └── loader.py        # compile one source for both backends
│   └── io/                  # schema, validation, synthetic data
└── tests/                   # pytest: parity, grid, soil, io, gpu
```

### Where to look first

| Concern            | Location                                      |
| ------------------ | --------------------------------------------- |
| Daily AquaCrop step | `src/aquacrop_grid/kernels/impl.py`          |
| CPU / GPU backends | `src/aquacrop_grid/engine/`                   |
| Raster pipeline    | `src/aquacrop_grid/pipeline.py`               |
| Crop / soil params | `src/aquacrop_grid/params.py`                 |
| Soil rasters / PTF | `src/aquacrop_grid/soil_grid.py`              |
| Zarr schema        | [`docs/zarr-schema.md`](docs/zarr-schema.md)  |
| Bit-exact parity   | `tests/test_parity.py`                        |

---

## Architecture

```
climate.zarr (time, y, x) + sowing.zarr (y, x)  [+ soil.zarr]
                         ↓
                   pipeline.py  (spatial tiles)
                         ↓
              params.py  →  kernel arrays
                         ↓
              engine/run.py  →  cpu (prange)  |  gpu (cuda)
                         ↓
         kernels/impl.py   one source, Numba njit / cuda.jit
                         ↓
                    output.zarr
```

- `kernels/impl.py` — AquaCrop daily step ported to a common `njit` /
  `cuda.jit` subset (no allocation in kernels, scalar state per pixel +
  1-D compartment views). One source compiled for both backends by
  `kernels/loader.py`.
- `params.py` — flattens aquacrop `Crop` / `Soil` into kernel arrays
  (weather-independent init, including deepening the profile to
  `Zmax + 0.1`).
- GDD phenology is computed **inside the kernel per pixel** (sowing date
  changes each pixel's thermal accumulation), mirroring aquacrop
  `compute_crop_calendar`.

---

## I/O schema

| Store    | Shape        | Role |
| -------- | ------------ | ---- |
| Climate  | `(time, y, x)` | `tmin`, `tmax`, `precip`, `eto` (daily, gap-free) |
| Sowing   | `(y, x)`     | `int32` YYYYDDD; `<= 0` = mask |
| Soil     | name **or** zarr | AquaCrop preset, or hydraulic / texture raster |
| Output   | `(y, x)`     | yield, biomass, status; optional `daily/` group |

Grids must already be aligned — AquaCrop-Grid does not reproject.
Details, units, and soil bands: [`docs/zarr-schema.md`](docs/zarr-schema.md).

---

## Benchmark

`aquacrop-grid bench --pixels 65536 --days 540` (excluding JIT compile):

| Backend | Reference hardware | Throughput |
| ------- | ------------------ | ---------- |
| CPU (`njit` + `prange`) | Ryzen (all threads) | ~58k pixels/s |
| GPU (`numba.cuda`) | RTX 3060 | ~61k pixels/s |

One full season (540 days) per pixel. On larger grids the GPU scales better
(climate transfer dominates on small grids).

---

## Testing

[![Tests](https://github.com/Paloschi/aquacrop-grid/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Paloschi/aquacrop-grid/actions/workflows/test.yml)

```bash
pytest            # parity vs AquaCrop-OSPy, io, grid driver, gpu (if present)
```

| File | What it covers |
| ---- | -------------- |
| `test_parity.py` | Single pixel vs AquaCrop-OSPy, 1e-12 (maize/soybean, calendar and GDD, 3 soils) |
| `test_grid.py` | Mask, per-pixel sowing, tiles, grid vs single-pixel equality |
| `test_soil.py` | Hydraulic/texture zarr, Saxton–Rawls PTF, per-pixel soil |
| `test_io.py` | Synthetic I/O, sowing → plant index |
| `test_gpu.py` | CPU vs GPU parity (skipped without CUDA) |

CI (GitHub Actions) runs on every **pull request** against `main`, and again
on push to `main`: Ubuntu and Windows × Python 3.11 / 3.12. Hosted runners
are CPU-only; `test_gpu.py` is skipped.

---

## Contributing

1. Use **conventional commits** (`feat:`, `fix:`, `chore:`, `ci:`, `docs:`).
2. Put tests in the same change as the behaviour they cover.
3. Open a pull request against `main` — CI must pass before merge.

---

## Citing this project

If you use AquaCrop-Grid in research or operational work, please cite this repository:

```bibtex
@software{paloschi_aquacrop_grid,
  author  = {Paloschi, Rennan Andres},
  title   = {AquaCrop-Grid: pixel-wise AquaCrop on rasters (Numba CPU/GPU)},
  year    = {2026},
  url     = {https://github.com/Paloschi/aquacrop-grid},
  version = {0.1.0}
}
```

Also acknowledge [AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) (and FAO AquaCrop) for the underlying crop water-productivity model that the kernels follow.


---

## License and attribution

**MIT.** Kernels in `src/aquacrop_grid/kernels/impl.py` are derived from
[AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) (**Apache-2.0**).
