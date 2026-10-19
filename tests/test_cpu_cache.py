"""Numba disk cache for the CPU kernels and the two grid runners."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _disk_cached(fn) -> bool:
    return type(fn._cache).__name__ == "FunctionCache"


def test_cpu_runners_request_disk_cache():
    from aquagrid.engine.cpu import get_runner
    from aquagrid.kernels.loader import load_kernels

    kernels = load_kernels("cpu")
    assert _disk_cached(kernels["run_pixel"])
    assert _disk_cached(kernels["roundn"])

    parallel = get_runner(True)
    serial = get_runner(False)
    assert parallel is not serial
    assert _disk_cached(parallel)
    assert parallel.targetoptions["parallel"] is True
    assert _disk_cached(serial)
    assert serial.targetoptions["parallel"] is False


def test_cached_kernel_hits_in_a_new_process():
    """roundn calls round_half_even, so the child process must load both."""
    from aquagrid.kernels.loader import load_kernels

    assert load_kernels("cpu")["roundn"](1.25, 1) == 1.2

    script = textwrap.dedent(
        """
        from aquagrid.kernels.loader import load_kernels
        fn = load_kernels("cpu")["roundn"]
        assert fn(1.25, 1) == 1.2
        if sum(fn._cache_hits.values()) < 1:
            raise SystemExit(
                "cache miss: " + repr(dict(fn._cache_misses)))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
