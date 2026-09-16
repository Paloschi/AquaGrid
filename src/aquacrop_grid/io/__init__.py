"""zarr I/O: schema validation, sowing-grid conversion, synthetic data.

See ``docs/zarr-schema.md`` for the full schema definition.
"""

from aquacrop_grid.io.schema import (
    CLIMATE_VARS,
    open_climate,
    open_sowing,
    sowing_to_plant_idx,
    validate_climate,
)
from aquacrop_grid.io.soil import (
    ACTIVE_DEPTHS,
    DEPTH_ALIASES,
    HYDRO_SCALE_FACTORS,
    SoilGrid,
    open_soil,
    validate_soil_grid,
)
from aquacrop_grid.io.synthetic import generate_synthetic

__all__ = [
    "CLIMATE_VARS",
    "open_climate",
    "open_sowing",
    "sowing_to_plant_idx",
    "validate_climate",
    "ACTIVE_DEPTHS",
    "DEPTH_ALIASES",
    "HYDRO_SCALE_FACTORS",
    "SoilGrid",
    "open_soil",
    "validate_soil_grid",
    "generate_synthetic",
]
