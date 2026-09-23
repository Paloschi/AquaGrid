"""AquaGrid: FAO AquaCrop on rasters.

Runs AquaCrop over raster grids: zarr climate cubes and per-pixel
sowing-date grids in, yield/biomass/canopy grids out. Compute kernels
are vendored from AquaCrop-OSPy (Apache-2.0) and compiled with Numba for CPU
(njit + prange) and GPU (numba.cuda).

This is not the official FAO AquaCrop nor an aquacropos/AquaCrop-OSPy
extension.
"""

__version__ = "0.2.0"
