"""Compile the backend-neutral kernel source for a given backend.

The kernel source in :mod:`aquacrop_grid.kernels.impl` is written in a common
subset of what ``numba.njit`` and ``numba.cuda.jit`` support. This module
exec's that source with the appropriate ``kernel`` decorator injected,
producing an independent, fully-jitted namespace per backend.
"""

from __future__ import annotations

import inspect

_CACHE: dict[str, dict] = {}


def load_kernels(backend: str = "cpu") -> dict:
    """Return a dict of compiled kernel functions for ``backend``
    ("cpu" -> numba.njit, "gpu" -> numba.cuda device functions)."""
    if backend in _CACHE:
        return _CACHE[backend]

    import aquacrop_grid.kernels.impl as impl_mod

    if backend == "cpu":
        from numba import njit

        def kernel(f):
            return njit(f, cache=False)

    elif backend == "gpu":
        from numba import cuda

        def kernel(f):
            return cuda.jit(f, device=True)

    else:
        raise ValueError(f"unknown backend: {backend!r}")

    src = inspect.getsource(impl_mod)
    ns: dict = {"kernel": kernel, "__name__": f"aquacrop_grid.kernels.impl_{backend}"}
    code = compile(src, impl_mod.__file__, "exec")
    exec(code, ns)

    _CACHE[backend] = ns
    return ns
