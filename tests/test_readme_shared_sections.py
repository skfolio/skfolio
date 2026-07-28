"""Guard the README regions shared with the documentation homepage.

`docs/index.rst` reuses parts of `README.rst` through partial `include` directives
delimited by `skfolio-shared-*` comments. Those comments are invisible once rendered,
so these tests keep them from being renamed, removed or moved.
"""

from pathlib import Path

import pytest

README = Path(__file__).parents[1] / "README.rst"

# Region name -> texts that the region must contain, near its start and near its end.
REGIONS = {
    "introduction": ("**skfolio** is a Python library", "Skfolio Labs"),
    "body": ("Important links", "Citation"),
}


@pytest.fixture(scope="module")
def readme() -> str:
    if not README.exists():
        pytest.skip("README.rst is not available outside a source checkout")
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize(("region", "expected"), REGIONS.items())
def test_shared_region_is_included_in_docs_index(readme, region, expected):
    start = f"skfolio-shared-{region}-start"
    end = f"skfolio-shared-{region}-end"
    assert readme.count(start) == 1, f"{start} must appear exactly once in README.rst"
    assert readme.count(end) == 1, f"{end} must appear exactly once in README.rst"
    body = readme.split(start, 1)[1].split(end, 1)[0]
    for text in expected:
        assert text in body, f"{region} region no longer contains {text!r}"


def test_readme_is_standalone(readme):
    # GitHub and PyPI do not process `include`, so the README must remain self-contained.
    assert ".. include::" not in readme
