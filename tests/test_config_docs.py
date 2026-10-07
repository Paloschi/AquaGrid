"""Published surfaces stay aligned with the code a release ships."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_GUIDE = (
    ROOT / "examples" / "config.example.yaml",
    ROOT / "README.md",
)


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _flatten(data: object, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if not isinstance(data, dict):
        return keys
    for key, value in data.items():
        path = _join(prefix, str(key))
        keys.add(path)
        keys |= _flatten(value, path)
    return keys


def _loader_config_keys() -> set[str]:
    """Dotted YAML paths read inside run_from_config."""
    src = (ROOT / "src" / "aquagrid" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_from_config"
    )
    alias = {"cfg": ""}

    def resolve(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return alias.get(node.id)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if not isinstance(node.slice.value, str):
                return None
            base = resolve(node.value)
            if base is None:
                return None
            return _join(base, node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            base = resolve(node.func.value)
            if base is None:
                return None
            return _join(base, node.args[0].value)
        if isinstance(node, ast.BoolOp):
            found = [path for path in (resolve(value) for value in node.values) if path]
            if len(found) == 1:
                return found[0]
        return None

    def add(path: str | None, keys: set[str]) -> None:
        if not path:
            return
        parts: list[str] = []
        for part in path.split("."):
            parts.append(part)
            keys.add(".".join(parts))

    keys: set[str] = set()
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            path = resolve(stmt.value)
            if path is not None:
                alias[stmt.targets[0].id] = path
                add(path, keys)
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Subscript, ast.Call)):
                add(resolve(node), keys)
    return keys


def _readme_config_keys() -> set[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    for block in blocks:
        data = yaml.safe_load(block)
        if isinstance(data, dict) and "climate" in data:
            return _flatten(data)
    raise AssertionError("README.md has no YAML config block")


def test_config_guide_lists_loader_keys():
    """A new run_from_config key fails CI until the example and README list it."""
    loader = _loader_config_keys()
    example = _flatten(yaml.safe_load(_GUIDE[0].read_text(encoding="utf-8")))
    readme = _readme_config_keys()
    assert {
        "sowing_var",
        "initial_water_content",
        "soil.scale_factors",
        "options.evap_time_steps",
    } <= loader
    assert loader == example
    assert loader == readme


def test_schema_documents_status_codes():
    text = (ROOT / "docs" / "zarr-schema.md").read_text(encoding="utf-8")
    block = text.split("`status`:", 1)[1].split("Optional daily", 1)[0]
    rows = {}
    for line in block.splitlines():
        if not line.startswith("| ") or line.startswith("| code") or line.startswith("|-"):
            continue
        code, meaning = (cell.strip() for cell in line.strip("|").split("|", 1))
        rows[code] = meaning
    assert set(rows) == {"0", "1", "2", "3"}
    assert "max_season_days" in rows["0"]
    assert "time" in rows["1"]
    assert "365" in rows["2"] and "CalendarType" in rows["2"]
    assert "ended before" in rows["3"] and "max_season_days" in rows["3"]


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _fallback_version() -> str:
    """The string used when the package is not installed."""
    src = (ROOT / "src" / "aquagrid" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            found.append(node.value.value)
    assert len(found) == 1
    return found[0]


def _bibtex_version() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"@software\{paloschi_aquagrid,.*?version\s*=\s*\{([^}]+)\}",
        text,
        flags=re.DOTALL,
    )
    assert match, "README citation has no version"
    return match.group(1).strip()


def _changelog_section(version: str) -> str:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text,
        flags=re.M | re.S,
    )
    assert match and match.group(1).strip(), f"CHANGELOG.md has no notes for {version}"
    return match.group(0).strip()


def test_version_copies_match():
    """pyproject.toml, the import fallback, and the README citation are one version."""
    version = _project()["version"]
    assert _fallback_version() == version
    assert _bibtex_version() == version
    assert _changelog_section(version)


def _next_minor(version: str) -> str:
    major, minor = version.split(".")[:2]
    return f"{major}.{int(minor) + 1}"


def test_requires_python_is_the_ci_matrix():
    """The wheel claims the interpreters the release workflow just ran."""
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    )
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    ordered = sorted(versions, key=lambda item: tuple(int(part) for part in item.split(".")))
    ceiling = _next_minor(ordered[-1])
    assert _project()["requires-python"] == f">={ordered[0]},<{ceiling}"


def test_github_release_follows_pypi_and_changelog():
    """A rejected PyPI upload does not publish, and the body is the changelog section."""
    path = ROOT / ".github" / "workflows" / "release.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]
    assert jobs["publish-to-pypi"]["needs"] == "build"
    assert jobs["github-release"]["needs"] == "publish-to-pypi"
    assert "--generate-notes" not in text
    assert "--notes-file release-notes.md" in text
    assert "CHANGELOG.md has no notes" in text
