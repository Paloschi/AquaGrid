# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-10-07

### Fixed

- Read `__version__` from the installed package metadata. The 0.3.0 wheel still reported 0.2.0.
- Point the license attribution and the README citation at the 0.3.0 tree.

### Changed

- Cap supported Python at 3.11 and 3.12 (`requires-python` is `>=3.11,<3.13`), the interpreters the release workflow runs.
- Publish the GitHub Release from this section, after PyPI accepts the files.
- Document every `run_from_config` key and the four output `status` codes.

## [0.3.0] - 2026-10-07

### Added

- Optional crop-parameter zarr `(param, y, x)` so each pixel can override AquaCrop crop attributes.

### Changed

- Require xarray 2026.9 and zarr 3. xarray 2026.9 dropped the zarr 2 compatibility layer.

## [0.2.0] - 2026-09-22

### Changed

- Rename the project to AquaGrid. The distribution, import, and command are `aquagrid`.

## [0.1.1] - 2026-09-16

### Changed

- PyPI-friendly README hero (HTML only inside the centered block; version badge from PyPI).
- CI matrix again covers Ubuntu and Windows on Python 3.11 and 3.12.

## [0.1.0] - 2026-09-16

### Added

- Initial AquaGrid package: zarr climate/sowing in, Numba CPU and GPU
  kernels, YAML CLI, and pytest suite with bit-exact AquaCrop-OSPy parity.

[Unreleased]: https://github.com/Paloschi/aquagrid/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Paloschi/aquagrid/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Paloschi/aquagrid/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Paloschi/aquagrid/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Paloschi/aquagrid/releases/tag/v0.1.1
[0.1.0]: https://github.com/Paloschi/aquagrid/releases/tag/v0.1.0
