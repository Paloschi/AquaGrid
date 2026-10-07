"""Published surfaces stay aligned with the code a release ships."""

from __future__ import annotations

import ast
import re
import textwrap
import tomllib
import urllib.parse
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


def _schema_status_rows() -> dict[str, str]:
    text = (ROOT / "docs" / "zarr-schema.md").read_text(encoding="utf-8")
    block = text.split("`status`:", 1)[1].split("Optional daily", 1)[0]
    rows = {}
    for line in block.splitlines():
        if not line.startswith("| ") or line.startswith("| code") or line.startswith("|-"):
            continue
        code, meaning = (cell.strip() for cell in line.strip("|").split("|", 1))
        rows[code] = meaning
    return rows


def test_schema_documents_status_codes():
    rows = _schema_status_rows()
    assert set(rows) == {"0", "1", "2", "3"}
    assert "max_season_days" in rows["0"]
    assert "time" in rows["1"]
    assert "365" in rows["2"] and "CalendarType" in rows["2"]
    assert "ended before" in rows["3"] and "max_season_days" in rows["3"]


# Words each surface must repeat for the branch the kernel actually takes.
_STATUS_FACTS = {
    "0": ("maturity", "canopy", "max_season_days"),
    "1": ("sowing", "<= 0", "time", "NaN"),
    "2": ("CalendarType", "365", "maturity"),
    "3": ("before", "maturity", "canopy", "max_season_days"),
}


def _status_comments() -> dict[str, str]:
    text = (ROOT / "src" / "aquagrid" / "kernels" / "constants.py").read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        match = re.match(r"^STATUS_[A-Z0-9_]+\s*=\s*(\d+)\.0\s*#\s*(.+)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _docstring(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    doc = ast.get_docstring(func)
    assert doc, f"{name} has no docstring"
    return doc


def _doc_status_spans(doc: str) -> dict[str, str]:
    """Text from each ``Status N`` mention until the next status mention."""
    marks = list(re.finditer(r"[Ss]tatus\s+(\d)", doc))
    spans: dict[str, str] = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(doc)
        code = mark.group(1)
        spans[code] = spans.get(code, "") + doc[mark.start():end]
    return spans


def _function(src: str, name: str) -> ast.AST:
    tree = ast.parse(src)
    found = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(found) == 1
    return found[0]


def _assigns_name(stmt: ast.stmt, value: str) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id == value
    )


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _is_neg_one(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value == -1:
        return True
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and node.operand.value == 1
    )


def test_status_texts_match_the_kernel():
    """A shorter comment, schema row, or run_grid sentence fails while the branch stays."""
    impl = (ROOT / "src" / "aquagrid" / "kernels" / "impl.py").read_text(encoding="utf-8")
    impl_tree = ast.parse(impl)
    closed = [
        node for node in ast.walk(impl_tree)
        if isinstance(node, ast.If) and any(_assigns_name(stmt, "STATUS_OK") for stmt in node.body)
    ]
    assert len(closed) == 1
    assert {"crop_mature", "crop_dead", "dap", "max_season_days"} <= _names(closed[0].test)

    calendar = _function(impl, "compute_calendar_cds")
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "maturity_cd"
        and any(isinstance(op, ast.GtE) for op in node.ops)
        and any(isinstance(item, ast.Constant) and item.value == 365 for item in node.comparators)
        for node in ast.walk(calendar)
    )
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Subscript)
        and isinstance(node.left.slice, ast.Name)
        and node.left.slice.id == "CP_CALENDAR_TYPE"
        and any(isinstance(op, ast.Eq) for op in node.ops)
        and any(isinstance(item, ast.Constant) and item.value == 1 for item in node.comparators)
        for node in ast.walk(calendar)
    )
    assert any(_assigns_name(node, "STATUS_TRUNCATED") for node in ast.walk(impl_tree))
    assert any(_assigns_name(node, "STATUS_NO_MATURITY_GDD") for node in ast.walk(impl_tree))

    for backend in ("cpu.py", "gpu.py"):
        tree = ast.parse(
            (ROOT / "src" / "aquagrid" / "engine" / backend).read_text(encoding="utf-8")
        )
        assert any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "pi"
            and any(isinstance(op, ast.Lt) for op in node.test.ops)
            and any(isinstance(item, ast.Constant) and item.value == 0 for item in node.test.comparators)
            and any(_assigns_name(stmt, "STATUS_NOT_SIMULATED") for stmt in node.body)
            for node in ast.walk(tree)
        )

    sowing = _function(
        (ROOT / "src" / "aquagrid" / "io" / "schema.py").read_text(encoding="utf-8"),
        "sowing_to_plant_idx",
    )
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "sow"
        and any(isinstance(op, ast.Gt) for op in node.ops)
        and any(isinstance(item, ast.Constant) and item.value == 0 for item in node.comparators)
        for node in ast.walk(sowing)
    )
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "idx"
        and any(isinstance(op, ast.GtE) for op in node.ops)
        and any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "len"
            and item.args
            and isinstance(item.args[0], ast.Name)
            and item.args[0].id == "time"
            for item in node.comparators
        )
        for node in ast.walk(sowing)
    )

    pipeline = (ROOT / "src" / "aquagrid" / "pipeline.py").read_text(encoding="utf-8")
    assert any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "pidx"
        and isinstance(node.targets[0].slice, ast.UnaryOp)
        and isinstance(node.targets[0].slice.op, ast.Invert)
        and isinstance(node.targets[0].slice.operand, ast.Name)
        and node.targets[0].slice.operand.id == "soil_ok"
        and _is_neg_one(node.value)
        for node in ast.walk(ast.parse(pipeline))
    )
    soil = _function(
        (ROOT / "src" / "aquagrid" / "soil_grid.py").read_text(encoding="utf-8"),
        "_finite_layers",
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "isfinite"
        for node in ast.walk(soil)
    )

    comment = _status_comments()
    schema = _schema_status_rows()
    spans = _doc_status_spans(_docstring(ROOT / "src" / "aquagrid" / "pipeline.py", "run_grid"))
    assert set(comment) == set(schema) == set(spans) == set(_STATUS_FACTS)
    for code, facts in _STATUS_FACTS.items():
        for fact in facts:
            assert fact in comment[code]
            assert fact in schema[code]
            assert fact in spans[code]


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


def _ci_python_versions() -> list[str]:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    )
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    return sorted(versions, key=lambda item: tuple(int(part) for part in item.split(".")))


def _py_versions(snippet: str) -> set[str]:
    assert re.search(r"\d+\.\d+\s*\+", snippet) is None
    return set(re.findall(r"\d+\.\d+", snippet))


def test_requires_python_is_the_ci_matrix():
    """The wheel claims the interpreters the release workflow just ran."""
    ordered = _ci_python_versions()
    ceiling = _next_minor(ordered[-1])
    assert _project()["requires-python"] == f">={ordered[0]},<{ceiling}"


def test_readme_python_versions_are_the_ci_matrix():
    """Badge, prerequisites, and the testing lines name the matrix interpreters."""
    expected = set(_ci_python_versions())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    badge = re.search(r'<img alt="Python" src="([^"]+)">', readme)
    language = re.search(r"\|\s*Language\s*\|([^|\n]+)", readme)
    tests_row = re.search(r"\|\s*Tests\s*\|([^|\n]+)", readme)
    prerequisites = re.search(r"^-\s+\*\*Python[^\n]+", readme, flags=re.M)
    ci_line = re.search(r"^.*Ubuntu and Windows × Python.*$", readme, flags=re.M)
    claims = {
        "badge": urllib.parse.unquote(badge.group(1)) if badge else "",
        "language": language.group(1) if language else "",
        "tests": tests_row.group(1) if tests_row else "",
        "prerequisites": prerequisites.group(0) if prerequisites else "",
        "testing": ci_line.group(0) if ci_line else "",
    }
    assert all(claims.values())
    for name, snippet in claims.items():
        assert _py_versions(snippet) == expected, name


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


def _workflow_python(src: str) -> str:
    match = re.search(r"python - << 'PY'\n(.*)\n[ \t]*PY\b", src, flags=re.S)
    assert match, "release workflow has no Python heredoc"
    return textwrap.dedent(match.group(1))


def _changelog_search_dump(src: str) -> str:
    """Pattern and flags of the one f-string re.search, which slices the changelog."""
    dumps = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "search" and node.args):
            continue
        if not isinstance(node.args[0], ast.JoinedStr):
            continue
        flags = next((kw.value for kw in node.keywords if kw.arg == "flags"), None)
        dumps.append(ast.dump(node.args[0]) + "|" + ast.dump(flags))
    assert len(dumps) == 1
    return dumps[0]


def test_release_changelog_regex_is_one_expression():
    """The release notes and the version check slice the changelog with one pattern."""
    workflow = _workflow_python(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    checker = Path(__file__).read_text(encoding="utf-8")
    assert _changelog_search_dump(workflow) == _changelog_search_dump(checker)
