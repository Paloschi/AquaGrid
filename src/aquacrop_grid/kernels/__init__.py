"""Compiled AquaCrop kernels (vendored from AquaCrop-OSPy, MIT license).

`impl.py` holds backend-neutral kernel source; `loader.py` compiles it
for the CPU (numba.njit) or GPU (numba.cuda device functions).
"""
