"""Stop the GPU assembler from fusing multiply and add.

NVVM emits a separate multiply and subtract. ptxas then fuses them, so a
demand that was met exactly stays positive and a root-zone loop keeps
walking. The kernel PTX is assembled with ``--fmad=false`` into a normal
cubin, so the product stays rounded and the register allocation stays the
one ptxas would use for a release build.

The CUDA simulator never calls the assembler.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_INSTALLED = False


def _ptxas() -> str | None:
    found = shutil.which("ptxas")
    if found:
        return found
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        for name in ("ptxas.exe", "ptxas"):
            candidate = Path(cuda_path) / "bin" / name
            if candidate.is_file():
                return str(candidate)
    roots = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
        Path("/usr/local"),
    ]
    hits: list[Path] = []
    for root in roots:
        if root.is_dir():
            hits.extend(root.glob("*/bin/ptxas.exe"))
            hits.extend(root.glob("cuda*/bin/ptxas"))
    if not hits:
        return None
    return str(sorted(hits)[-1])


def _assemble(ptx: bytes) -> bytes:
    text = ptx.decode("utf-8", errors="replace")
    arch = "sm_70"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".target"):
            arch = stripped.split()[1]
            break
    ptxas = _ptxas()
    if ptxas is None:
        raise RuntimeError(
            "ptxas was not found; the GPU kernel needs it to match "
            "AquaCrop-OSPy rounding")
    cache = Path(tempfile.gettempdir()) / "aquagrid"
    cache.mkdir(parents=True, exist_ok=True)
    src = cache / "kernel.ptx"
    out = cache / "kernel.cubin"
    src.write_bytes(ptx)
    # A full cubin, not relocatable (-c). Relocatable code links without
    # register allocation and made a season take seconds per pixel.
    proc = subprocess.run(
        [ptxas, "--fmad=false", f"-arch={arch}", "-o", str(out), str(src)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return out.read_bytes()


def disable_fma_contraction() -> None:
    """Load a fully optimized cubin assembled with ``--fmad=false``."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
        return
    from numba.cuda.codegen import CUDACodeLibrary

    def get_cubin(self, cc=None):
        cc = self._ensure_cc(cc)
        cached = self._cubin_cache.get(cc, None)
        if cached is not None:
            return cached
        ptx = self.get_asm_str(cc=cc)
        if isinstance(ptx, str):
            ptx = ptx.encode()
        cubin = _assemble(ptx)
        self._cubin_cache[cc] = cubin
        return cubin

    CUDACodeLibrary.get_cubin = get_cubin
    _INSTALLED = True
