"""Compile the backend-neutral kernel source for a given backend.

The kernel source in :mod:`aquagrid.kernels.impl` is written in a common
subset of what ``numba.njit`` and ``numba.cuda.jit`` support. This module
exec's that source with the appropriate ``kernel`` decorator injected,
producing an independent, fully-jitted namespace per backend.
"""

from __future__ import annotations

import inspect
import sys
import types

_CACHE: dict[str, dict] = {}


def load_kernels(backend: str = "cpu") -> dict:
    """Return a dict of compiled kernel functions for ``backend``
    ("cpu" -> numba.njit, "gpu" -> numba.cuda device functions)."""
    if backend in _CACHE:
        return _CACHE[backend]

    import aquagrid.kernels.impl as impl_mod

    if backend == "cpu":
        from numba import njit

        def kernel(f):
            # Disk cache: a later process loads the machine code instead of
            # compiling again. NUMBA_CACHE_DIR relocates the files.
            return njit(f, cache=True)

    elif backend == "gpu":
        from numba import cuda

        def kernel(f):
            return cuda.jit(f, device=True)

    else:
        raise ValueError(f"unknown backend: {backend!r}")

    # A real module, registered before compilation. Numba reloads a cached
    # kernel by importing its module; an exec dict with no sys.modules entry
    # is pickled as "<dynamic>" and the next process cannot load it.
    modname = f"aquagrid.kernels.impl_{backend}"
    module = types.ModuleType(modname)
    module.__file__ = impl_mod.__file__
    module.__package__ = impl_mod.__package__
    module.kernel = kernel
    sys.modules[modname] = module
    code = compile(inspect.getsource(impl_mod), impl_mod.__file__, "exec")
    exec(code, module.__dict__)

    ns = module.__dict__
    _CACHE[backend] = ns
    return ns
