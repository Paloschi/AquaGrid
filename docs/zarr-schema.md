# AquaGrid zarr schema

All gridded inputs and outputs are zarr stores read/written with xarray.
Axis convention: `time` (daily, contiguous), `y`, `x` (any projection;
AquaGrid does not reproject — climate and sowing grids must be
aligned).

## Climate (input)

Store with dims `(time, y, x)` and variables:

| var      | unit | description                    |
|----------|------|--------------------------------|
| `tmin`   | °C   | daily minimum temperature      |
| `tmax`   | °C   | daily maximum temperature      |
| `precip` | mm   | daily precipitation            |
| `eto`    | mm   | reference evapotranspiration   |

- coord `time`: `datetime64`, daily and gap-free.
- recommended dtype `float32`; the driver converts to `float64` in the kernel.
- recommended chunks `(time=-1, y=256, x=256)`: the simulation is sequential
  in time and parallel in space, so the full time axis stays in the chunk.

## Sowing (input)

2-D `(y, x)` integer (`int32`) variable with each pixel's sowing date in the
legacy CyMP `YYYYDDD` convention (e.g. `2019135` = DOY 135 of 2019). Values
`<= 0` mark pixels that are not simulated (nodata/mask).

May live in the same store as climate or in its own store (var `sowing`).

## Soil (input)

Two exclusive YAML modes: an AquaCrop name (`soil.name`, one profile for the
grid) **or** a per-pixel zarr raster (`soil.zarr`).

The profile is deepened to crop `Zmax + 0.1` m. In name mode, compartments
are thickened as in aquacrop; in zarr mode, 0.1 m compartments are appended
with the deepest layer's properties.

### Hydraulic raster (HiHydroSoil)

Variables `ksat`, `wcsat`, `wcpf2`, `wcpf3` (aliases `Ksat`, `WCsat`,
`WCpF2`, `WCpF3`) with dims `(depth, y, x)` aligned to climate.

| var     | AquaCrop unit | HiHydroSoil source             |
|---------|---------------|--------------------------------|
| `ksat`  | mm/d          | Ksat cm/d × 10                 |
| `wcsat` | m³/m³         | WCsat → `th_s`                 |
| `wcpf2` | m³/m³         | WCpF2 → `th_fc`                |
| `wcpf3` | m³/m³         | WCpF3 → `th_wp`                |

Coord `depth` with canonical labels `0-5`, `5-15`, `15-30`, `30-60`,
`60-100` (required) and `100-200` (optional). Accepted aliases:
`000-005cm`, `0-5cm`, `05-15cm`, `5-15cm`, `15-30cm`, `30-60cm`,
`60-100cm`, `100-200cm`.

Each band becomes one compartment (`dz` = 5, 10, 15, 30, 40 cm). Integer
HiHydroSoil v2 GeoTIFF: × `0.0001` (WC) and × `0.001` (Ksat, already includes
cm/d → mm/d). Float WC in [0, 1] does not re-apply `0.0001`; Ksat still
converts cm/d → mm/d if `soil.ksat_unit` is `cm/d` (default). Override:
`soil.scale_factors`.

2-D grids (no `depth`) are a homogeneous 12 × 0.1 m profile.

### Texture raster

Variables `sand`, `clay`, and optionally `silt` / `orgmat` (aliases
`areia`, `argila`, `silte`). Homogeneous 2-D or `(depth, y, x)` on the same
bands. Percent 0–100 or fractions 0–1. The PTF is Saxton & Rawls (2006), as
in AquaCrop-OSPy `Soil.add_layer_from_texture` (sand + clay + organic matter;
silt only checks that the sum is ≈ 100%).

A pixel with NaN in the soil is not simulated (`status = 1`), even if sowing
is valid.

## Outputs

Zarr store with:

- finals `(y, x)`, `float32`: `dry_yield`, `fresh_yield`, `yield_pot`
  (tonne/ha), `biomass`, `biomass_ns` (g/m²), `hi_adj` (-), `dap_end`
  (days), `status` (int8: 0=ok, 1=not simulated, 2=no maturity,
  3=truncated).
- optional daily `(time, y, x)`, `float32`, in group `daily` of the same
  store: `canopy_cover` (-), `biomass` (g/m²), `z_root` (m), `gdd_cum`
  (°C day), `es` (mm), `tr` (mm), `wr` (mm). Days outside the pixel's season
  are `NaN`.
