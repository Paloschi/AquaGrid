# Esquema zarr do AquaCrop-Grid

Todas as entradas e saídas em grid são stores zarr lidos/escritos com xarray.
Convenção de eixos: `time` (diário, contíguo), `y`, `x` (projeção livre; o
AquaCrop-Grid não reprojeta — os grids de clima e semeadura devem estar
alinhados).

## Clima (entrada)

Store com dims `(time, y, x)` e as variáveis:

| var      | unidade | descrição                    |
|----------|---------|------------------------------|
| `tmin`   | °C      | temperatura mínima diária    |
| `tmax`   | °C      | temperatura máxima diária    |
| `precip` | mm      | precipitação diária          |
| `eto`    | mm      | evapotranspiração de ref.    |

- coord `time`: `datetime64`, diário e sem lacunas.
- dtype recomendado `float32`; o driver converte para `float64` no kernel.
- chunks recomendados `(time=-1, y=256, x=256)`: a simulação é sequencial
  no tempo e paralela no espaço, então o eixo do tempo inteiro fica no chunk.

## Semeadura (entrada)

Variável 2-D `(y, x)` inteira (`int32`), com a data de semeadura de cada
pixel na convenção `YYYYDDD` do CyMP legado (ex.: `2019135` = DOY 135 de
2019). Valores `<= 0` marcam pixels não simulados (nodata/máscara).

Pode morar no mesmo store do clima ou em um store próprio (var `sowing`).

## Solo (entrada)

Dois modos, exclusivos no YAML: nome AquaCrop (`soil.name`, um perfil para
o grid) **ou** raster zarr por pixel (`soil.zarr`).

O perfil é aprofundado até `Zmax + 0.1` m da cultura. No modo nome, os
compartimentos são engrossados como no aquacrop; no modo zarr, anexam-se
compartimentos de 0.1 m com as propriedades da camada mais profunda.

### Raster hidráulico (HiHydroSoil)

Variáveis `ksat`, `wcsat`, `wcpf2`, `wcpf3` (aliases `Ksat`, `WCsat`,
`WCpF2`, `WCpF3`) com dims `(depth, y, x)` alinhadas ao clima.

| var     | unidade AquaCrop | origem HiHydroSoil              |
|---------|------------------|---------------------------------|
| `ksat`  | mm/d             | Ksat cm/d × 10                  |
| `wcsat` | m³/m³            | WCsat → `th_s`                  |
| `wcpf2` | m³/m³            | WCpF2 → `th_fc`                 |
| `wcpf3` | m³/m³            | WCpF3 → `th_wp`                 |

Coord `depth` com labels canónicos `0-5`, `5-15`, `15-30`, `30-60`,
`60-100` (obrigatórios) e `100-200` (opcional). Aliases aceites:
`000-005cm`, `0-5cm`, `05-15cm`, `5-15cm`, `15-30cm`, `30-60cm`,
`60-100cm`, `100-200cm`.

Cada banda vira um compartimento (`dz` = 5, 10, 15, 30, 40 cm). GeoTIFF
HiHydroSoil v2 inteiro: × `0.0001` (WC) e × `0.001` (Ksat, já inclui
cm/d → mm/d). Float com WC ∈ [0, 1] não reaplica `0.0001`; Ksat ainda
converte cm/d → mm/d se `soil.ksat_unit` for `cm/d` (default). Override:
`soil.scale_factors`.

Grids 2-D (sem `depth`) são um perfil homogéneo de 12 × 0.1 m.

### Raster de textura

Variáveis `sand`, `clay` e opcionalmente `silt` / `orgmat` (aliases
`areia`, `argila`, `silte`). 2-D homogéneo ou `(depth, y, x)` nas mesmas
bandas. Percentagens 0–100 ou fracções 0–1. A PTF é Saxton & Rawls
(2006), como `Soil.add_layer_from_texture` do AquaCrop-OSPy (areia +
argila + matéria orgânica; silte só valida soma ≈ 100 %).

Pixel com NaN no solo não é simulado (`status = 1`), mesmo com semeadura
válida.

## Saídas

Store zarr com:

- finais `(y, x)`, `float32`: `dry_yield`, `fresh_yield`, `yield_pot`
  (tonne/ha), `biomass`, `biomass_ns` (g/m²), `hi_adj` (-), `dap_end`
  (dias), `status` (int8: 0=ok, 1=sem maturidade, 2=truncado, <0=não
  simulado).
- diárias opcionais `(time, y, x)`, `float32`, no grupo `daily` do mesmo
  store: `canopy_cover` (-), `biomass` (g/m²), `z_root` (m), `gdd_cum`
  (°C dia), `es` (mm), `tr` (mm), `wr` (mm). Dias fora da estação do
  pixel ficam `NaN`.
