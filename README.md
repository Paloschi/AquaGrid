# AquaCrop-Grid

AquaCrop pixel a pixel sobre rasters: clima em cubos zarr `(time, y, x)`,
datas de semeadura por pixel `(y, x)` e saídas em zarr. Os kernels do passo
diário do [AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) 3.x foram
vendorizados e recompilados com Numba — `njit` + `prange` na CPU (1 thread
por pixel) e `numba.cuda` na GPU — com **paridade bit-exata** contra o
AquaCrop-OSPy original (testes em `tests/test_parity.py`).

Não é o AquaCrop oficial da FAO nem uma extensão do projeto aquacropos.
Nasceu no [CyMP](https://github.com/Paloschi/CyMP) (Unioeste-LEA).

Escopo atual: sequeiro (sem irrigação e sem lençol freático), uma safra
por pixel. Solo: preset AquaCrop único **ou** raster zarr por pixel
(hidráulico HiHydroSoil ou textura areia/silte/argila).

## Instalação

```bash
pip install -e .[dev]
```

GPU: requer uma GPU NVIDIA com driver CUDA instalado (o numba usa o driver
diretamente; não é preciso o toolkit completo).

## Uso rápido (exemplo end-to-end)

```bash
# 1. gerar dado sintético 10x10 pixels / 540 dias
aquacrop-grid synth --out ./dados

# 2. criar config
cp examples/config.example.yaml ./dados/config.yaml
# (ajustar caminhos se necessário)

# 3. rodar
aquacrop-grid run --config ./dados/config.yaml            # CPU
aquacrop-grid run --config ./dados/config.yaml -b gpu     # GPU
```

Saída: `output.zarr` com yield/biomassa finais `(y, x)` e, com
`save_daily: true`, série diária `(time, y, x)` no grupo `daily`.
Esquema completo em [`docs/zarr-schema.md`](docs/zarr-schema.md).

Uso programático:

```python
from aquacrop_grid.pipeline import run_grid

run_grid("clima.zarr", "semeadura.zarr", "saida.zarr",
         crop_name="Maize", soil_name="SandyLoam",
         backend="cpu", save_daily=False)
```

## Config YAML

```yaml
climate: dados/climate.zarr      # cubo (time, y, x): tmin, tmax, precip, eto
sowing: dados/sowing.zarr        # grid (y, x) int32 YYYYDDD; <=0 = mascarado
output: dados/output.zarr
crop:
  name: Maize                    # qualquer cultura do aquacrop-ospy
soil:
  name: SandyLoam                # preset AquaCrop (xor com zarr abaixo)
  # zarr: dados/soil.zarr        # ksat/wcsat/wcpf2/wcpf3 ou sand/silt/clay
  # ksat_unit: cm/d
backend: cpu                     # cpu | gpu
options:
  save_daily: false
  tile: 128                      # tile espacial (pixels) por lote
  max_season_days: 400
```

## Benchmark

`aquacrop-grid bench --pixels 65536 --days 540` (excluindo compilação JIT):

| backend | hardware de referência | throughput |
|---------|------------------------|-----------|
| CPU (`njit`+`prange`) | Ryzen (todas as threads) | ~58 mil pixels/s |
| GPU (`numba.cuda`)    | RTX 3060                 | ~61 mil pixels/s |

Uma safra completa (540 dias) por pixel. Em grids maiores a GPU escala
melhor (a transferência do clima domina em grids pequenos).

## Testes

```bash
pytest            # paridade vs aquacrop-ospy, io, grid driver, gpu (se houver)
```

- `test_parity.py` — pixel único vs AquaCrop-OSPy original, tolerância 1e-12
  (milho/soja, calendário e GDD, 3 solos).
- `test_grid.py` — driver de grid: máscara, semeadura por pixel, tiles, e
  igualdade grid vs pixel único.
- `test_soil.py` — zarr hidráulico/textura, PTF Saxton–Rawls, solo por pixel.
- `test_gpu.py` — paridade CPU vs GPU (pulado sem CUDA).

## Arquitetura

- `src/aquacrop_grid/kernels/impl.py` — passo diário do AquaCrop portado para um
  subconjunto comum de `njit`/`cuda.jit` (sem alocação nos kernels, estado
  escalar por pixel + views 1-D por compartimento). Fonte única compilada
  para os dois backends por `kernels/loader.py`.
- `src/aquacrop_grid/params.py` — achata `Crop`/`Soil` do aquacrop nos arrays de
  parâmetros dos kernels (replica a inicialização weather-independent,
  incluindo o aprofundamento do perfil até `Zmax + 0.1`).
- `src/aquacrop_grid/engine/` — `cpu.py` (prange), `gpu.py` (cuda, estado no
  device), `run.py` (runner comum sobre arrays achatados).
- `src/aquacrop_grid/pipeline.py` — zarr in → tiles → zarr out.
- `src/aquacrop_grid/io/` — esquema (clima, semeadura, solo), validação e sintético.
- `src/aquacrop_grid/soil_grid.py` — PTF e camadas HiHydro → compartimentos por pixel.

O calendário fenológico dependente de GDD é calculado dentro do kernel por
pixel (a data de semeadura muda o acúmulo térmico de cada pixel), espelhando
`compute_crop_calendar` do aquacrop.

## Licença e atribuição

GPL-3.0-or-later. Os kernels em `src/aquacrop_grid/kernels/impl.py` são
derivados do [AquaCrop-OSPy](https://github.com/aquacropos/aquacrop) (MIT).
