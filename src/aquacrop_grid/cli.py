"""AquaCrop-Grid command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="AquaCrop-Grid - AquaCrop on rasters (zarr in/out, Numba CPU/GPU)")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML run config"),
    backend: str = typer.Option("cpu", "--backend", "-b", help="cpu | gpu"),
):
    """Run a gridded AquaCrop simulation described by a YAML config."""
    from aquacrop_grid.pipeline import run_from_config

    out = run_from_config(config, backend=backend)
    typer.echo(f"results written to {out}")


@app.command()
def synth(
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    ny: int = typer.Option(10, help="Grid rows"),
    nx: int = typer.Option(10, help="Grid cols"),
    days: int = typer.Option(540, help="Length of the daily time axis"),
):
    """Generate a small synthetic zarr dataset for testing."""
    from aquacrop_grid.io.synthetic import generate_synthetic

    generate_synthetic(out, ny=ny, nx=nx, days=days)
    typer.echo(f"synthetic data written to {out}")


@app.command()
def bench(
    pixels: int = typer.Option(16384, help="Number of pixels"),
    days: int = typer.Option(540, help="Length of the time axis"),
    backend: str = typer.Option("both", help="cpu | gpu | both"),
):
    """Benchmark simulation throughput (pixels/s) on CPU and/or GPU."""
    from aquacrop_grid.bench import benchmark

    backends = ("cpu", "gpu") if backend == "both" else (backend,)
    benchmark(npix=pixels, days=days, backends=backends)


if __name__ == "__main__":
    app()
