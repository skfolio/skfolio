"""Keep the documented dependency floors aligned with `pyproject.toml`.

The installation guide lists the minimum supported versions and `README.rst` repeats
the Python floor, for readers who never open `pyproject.toml`. These tests make the
duplication safe by failing when a floor is bumped in only one place.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.rst"
INSTALL_GUIDE = ROOT / "docs" / "user_guide" / "install.rst"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.name} is not available outside a source checkout")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def minimum_versions() -> dict[str, str]:
    """Map each required distribution to its minimum version in `pyproject.toml`.

    Parsed with regular expressions rather than `tomllib`, which is only available
    from Python 3.11 onward.
    """
    pyproject = _read(PYPROJECT)

    dependencies = re.search(
        r"^dependencies = \[(.*?)^\]", pyproject, flags=re.DOTALL | re.MULTILINE
    )
    assert dependencies is not None, "pyproject.toml has no `dependencies` array"

    requirements = dict(
        re.findall(r"\"([A-Za-z0-9_.-]+)>=([^\"]+)\"", dependencies.group(1))
    )
    assert requirements, "no pinned minimum versions found in `dependencies`"

    python_requires = re.search(r"^requires-python = \">=([^\"]+)\"", pyproject, re.M)
    assert python_requires is not None, "pyproject.toml has no `requires-python`"
    requirements["python"] = python_requires.group(1)
    return requirements


def test_readme_python_version_matches_pyproject(minimum_versions):
    readme = _read(README)
    match = re.search(r"\|PythonMinVersion\| replace:: (\S+)", readme)
    assert match is not None, "README.rst no longer defines |PythonMinVersion|"
    assert match.group(1) == minimum_versions["python"]


def test_install_guide_matches_pyproject(minimum_versions):
    install_guide = _read(INSTALL_GUIDE)
    for distribution, version in minimum_versions.items():
        expected = f"- {distribution} (>= {version})"
        assert expected in install_guide, (
            f"install.rst does not document {distribution} >= {version}"
        )
