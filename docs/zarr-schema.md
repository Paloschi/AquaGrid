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

May live in the same store as climate or in its own store. The variable
name defaults to `sowing` (`sowing_var` in the YAML config). A date that
falls outside the climate `time` axis is not simulated.

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
converts cm/d → mm/d if `soil.ksat_unit` is `cm/d` (default; `mm/d`
leaves float Ksat unchanged).

`soil.scale_factors` maps a hydraulic name to a multiplier: `ksat`,
`wcsat`, `wcpf2`, `wcpf3`. YAML keys are lowercased. A name listed there
replaces the integer scale and the `ksat_unit` conversion for that
variable. The map is applied only in this hydraulic branch. An AquaCrop
preset (`soil.name`) and a texture raster ignore `soil.scale_factors`
and `soil.ksat_unit`.

2-D grids (no `depth`) are a homogeneous 12 × 0.1 m profile.

### Texture raster

Variables `sand`, `clay`, and optionally `silt` / `orgmat` (aliases
`areia`, `argila`, `silte`). Homogeneous 2-D or `(depth, y, x)` on the same
bands. Percent 0–100 or fractions 0–1. The PTF is Saxton & Rawls (2006), as
in AquaCrop-OSPy `Soil.add_layer_from_texture` (sand + clay + organic matter;
silt only checks that the sum is ≈ 100%).

A pixel with NaN in the soil is not simulated (`status = 1`), even if sowing
is valid.

## Crop parameters (input, optional)

Store aligned to climate with one data variable `crop` and dims
`(param, y, x)`. Each `param` name is an AquaCrop `Crop` attribute already
stored in the kernel vector (`HI0`, `WP`, `Zmax`, `Maturity`, `CGC`, …).
Any name in that map is accepted. A name outside it is an error.

Three AquaCrop vectors of length 4 are one layer each: `p_up1`–`p_up4`,
`p_lo1`–`p_lo4`, `fshape_w1`–`fshape_w4`.

`fCO2` and the CO2 concentration are not layers; they come from the sowing
year. Names ending in `CD` (`MaturityCD`, …) are not layers either. On a
calendar-day pixel (`CalendarType` 1), an override of `Maturity`,
`MaxCanopy`, `CanopyDevEnd`, `HIstart`, `HIend`, `YldForm`, or `Flowering`
is copied into the matching calendar-day slot.

`crop_name` is still required. A missing layer, or a NaN cell, keeps that
parameter from the named crop. A NaN does not mask the pixel. Without this
store every pixel uses the named crop alone. Soil compartments are deepened
to the deepest finite `Zmax` in the cube.

## Outputs

Rainfed, one season per pixel, no irrigation and no groundwater table.

Final fields are `(y, x)`: `dry_yield`, `fresh_yield`, `yield_pot`
(float32, tonne/ha), `biomass`, `biomass_ns` (float32, g/m²), `hi_adj`
(float32, -), `dap_end` (int16, days after planting), `status` (int8).

`status`:

| code | meaning |
|------|---------|
| 0 | Season closed: the crop reached maturity, the canopy died, or days after planting reached `max_season_days`. |
| 1 | Not simulated: sowing is `<= 0`, the sowing date falls outside the climate `time` axis, or the soil pixel is NaN. Other final fields stay 0. |
| 2 | GDD calendar was not built (`CalendarType` 2). Growing degree-days never rise strictly above maturity, or that crossing falls on day 365 or later of the season. A calendar-day crop (`CalendarType` 1) keeps the calendar prepared in Python. Other final fields stay 0. |
| 3 | The climate series ended before maturity, canopy death, or the `max_season_days` cap. Final fields hold the season accumulated through the last climate day. |

Optional daily fields are `(time, y, x)`, float32, in group `daily` of
the same store: `canopy_cover` (-), `biomass` (g/m²), `z_root` (m),
`gdd_cum` (°C day), `es` (mm), `tr` (mm), `wr` (mm). A day outside that
pixel's season is `NaN`. A pixel with status 1 or 2 is `NaN` on every day.
