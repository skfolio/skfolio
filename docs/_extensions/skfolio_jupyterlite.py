"""Configure and validate the JupyterLite gallery integration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import nbformat
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

CONTENT_DIR = "_contents"
INIT_CELL_TAG = "skfolio-jupyterlite-init"
NOTEBOOK_FORMAT = 4
NOTEBOOK_FORMAT_MINOR = 5
_CELL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Pure-Python packages that are unavailable in the Pyodide distribution but are
# required by skfolio.
PACKAGES_TO_INSTALL = ["plotly", "nbformat"]

# Import runtime dependencies before importing skfolio. This avoids loading them
# concurrently on the first notebook import in Pyodide.
PACKAGES_TO_IMPORT = [
    "pandas",
    "sklearn",
    "plotly",
    "cvxpy",
    "nbformat",
    "skfolio",
]

INIT_CELL_SOURCE = f"""
# JupyterLite Initialization

# Suppress non-actionable runtime warnings in this browser session.
import warnings
warnings.filterwarnings("ignore")

# Install packages that are not included in Pyodide.
import piplite
await piplite.install({json.dumps(PACKAGES_TO_INSTALL)})
await piplite.install(["skfolio"], deps=False)

# Allow examples to download external datasets.
import pyodide_http
pyodide_http.patch_all()

# Preload runtime dependencies used by the examples.
import {", ".join(PACKAGES_TO_IMPORT)}
""".strip()


def _upgrade_notebook_format(notebook_content: dict[str, Any]) -> None:
    """Upgrade a v4 notebook to the latest v4 schema without replacing valid IDs."""
    if notebook_content.get("nbformat") != NOTEBOOK_FORMAT:
        raise ValueError(
            "JupyterLite gallery notebooks must use notebook format 4; "
            f"got {notebook_content.get('nbformat')!r}."
        )

    minor_version = notebook_content.get("nbformat_minor", 0)
    if not isinstance(minor_version, int):
        minor_version = 0
    notebook_content["nbformat_minor"] = max(minor_version, NOTEBOOK_FORMAT_MINOR)

    used_ids: set[str] = set()
    for cell in notebook_content.get("cells", []):
        cell_id = cell.get("id")
        if (
            isinstance(cell_id, str)
            and _CELL_ID_PATTERN.fullmatch(cell_id)
            and cell_id not in used_ids
        ):
            used_ids.add(cell_id)
            continue

        cell_id = uuid4().hex
        while cell_id in used_ids:
            cell_id = uuid4().hex
        cell["id"] = cell_id
        used_ids.add(cell_id)


def modify_jupyterlite_notebook(
    notebook_content: dict[str, Any], notebook_filename: str
) -> dict[str, Any]:
    """Add one visible initialization cell to a gallery notebook."""
    cells = notebook_content.setdefault("cells", [])
    tagged_cells = [
        cell
        for cell in cells
        if INIT_CELL_TAG in cell.get("metadata", {}).get("tags", [])
    ]

    if len(tagged_cells) > 1:
        raise ValueError(
            f"Found multiple {INIT_CELL_TAG!r} cells in {notebook_filename}."
        )

    if tagged_cells:
        init_cell = tagged_cells[0]
        cells.remove(init_cell)

        metadata = init_cell.setdefault("metadata", {})
        tags = [tag for tag in metadata.get("tags", []) if tag != INIT_CELL_TAG]
        metadata["tags"] = [*tags, INIT_CELL_TAG]

        jupyter_metadata = metadata.get("jupyter")
        if isinstance(jupyter_metadata, dict):
            jupyter_metadata.pop("source_hidden", None)
            if not jupyter_metadata:
                metadata.pop("jupyter")

        init_cell.update(
            cell_type="code",
            source=INIT_CELL_SOURCE,
            execution_count=None,
            outputs=[],
        )
        init_cell.pop("attachments", None)
    else:
        init_cell = nbformat.v4.new_code_cell(
            INIT_CELL_SOURCE,
            metadata={"tags": [INIT_CELL_TAG]},
        )

    cells.insert(0, init_cell)
    _upgrade_notebook_format(notebook_content)
    return notebook_content


def _resolve_from_srcdir(app: Sphinx, path: str | Path) -> Path:
    """Resolve a Sphinx path relative to the documentation source directory."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(app.srcdir, resolved)
    return resolved.resolve()


def configure_jupyterlite_contents(app: Sphinx, config: Any) -> None:
    """Use the JupyterLite content root directly, without a staging prefix."""
    gallery_jupyterlite = config.sphinx_gallery_conf.get("jupyterlite")
    if not gallery_jupyterlite:
        return

    gallery_content_dir = gallery_jupyterlite.get("jupyterlite_contents")
    if gallery_content_dir is None:
        return

    content_root = _resolve_from_srcdir(app, config.jupyterlite_content_dir)
    if _resolve_from_srcdir(app, gallery_content_dir) != content_root:
        raise SphinxError(
            "Sphinx-Gallery and JupyterLite must use the same content directory."
        )

    # Sphinx-Gallery registers its output directory as additional user content.
    # jupyterlite-sphinx 0.23+ preserves the name of such directories, which would
    # expose `_contents/` inside JupyterLite. The directory is already the canonical
    # JupyterLite content root, so remove only that duplicate registration.
    extra_contents = config.jupyterlite_contents or []
    if isinstance(extra_contents, (str, Path)):
        extra_contents = [extra_contents]
    config.jupyterlite_contents = [
        path
        for path in extra_contents
        if _resolve_from_srcdir(app, path) != content_root
    ]


def _read_json(path: Path) -> dict[str, Any]:
    """Read a generated JSON file and report a useful build error."""
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise SphinxError(
            f"Unable to read JupyterLite JSON file {path}: {error}"
        ) from error


def _content_names(index: dict[str, Any]) -> set[str]:
    """Return the entry names from a JupyterLite contents response."""
    return {
        item["name"]
        for item in index.get("content", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def validate_jupyterlite_gallery(app: Sphinx, exception: Exception | None) -> None:
    """Fail the build when gallery URLs and the Lite file tree diverge."""
    if exception is not None or app.builder.name not in {"html", "readthedocs"}:
        return

    gallery_dirs = app.config.sphinx_gallery_conf.get("gallery_dirs", [])
    if isinstance(gallery_dirs, (str, Path)):
        gallery_dirs = [gallery_dirs]

    lite_dir = Path(app.outdir, "lite")
    root_index_path = lite_dir / "api" / "contents" / "all.json"
    root_names = _content_names(_read_json(root_index_path))
    expected_roots = {Path(gallery_dir).name for gallery_dir in gallery_dirs}

    issues: list[str] = []
    missing_roots = expected_roots - root_names
    if missing_roots:
        issues.append(
            "missing gallery directories at the Lite root: "
            + ", ".join(sorted(missing_roots))
        )

    leaked_roots = {CONTENT_DIR, "jupyterlite_contents"} & root_names
    if leaked_roots:
        issues.append(
            "build-only directories exposed at the Lite root: "
            + ", ".join(sorted(leaked_roots))
        )

    notebook_count = 0
    checked_indexes: dict[Path, set[str]] = {}
    for gallery_dir in gallery_dirs:
        gallery_path = _resolve_from_srcdir(app, gallery_dir)
        for source_notebook in sorted(gallery_path.rglob("*.ipynb")):
            notebook_count += 1
            relative_path = source_notebook.relative_to(app.srcdir)
            relative_posix = relative_path.as_posix()
            built_notebook = lite_dir / "files" / relative_path

            if not built_notebook.is_file():
                issues.append(f"missing Lite notebook: {relative_posix}")
                continue

            notebook = _read_json(built_notebook)
            try:
                nbformat.validate(notebook)
            except nbformat.ValidationError as error:
                issues.append(
                    f"invalid notebook schema for {relative_posix}: "
                    f"{str(error).splitlines()[0]}"
                )

            cells = notebook.get("cells", [])
            tagged_indexes = [
                index
                for index, cell in enumerate(cells)
                if INIT_CELL_TAG in cell.get("metadata", {}).get("tags", [])
            ]
            if tagged_indexes != [0]:
                issues.append(
                    f"{relative_posix} must have exactly one initialization cell "
                    "in first position"
                )
            else:
                init_metadata = cells[0].get("metadata", {})
                if init_metadata.get("jupyter", {}).get("source_hidden") is True:
                    issues.append(f"initialization cell is hidden: {relative_posix}")

            api_index_path = (
                lite_dir / "api" / "contents" / relative_path.parent / "all.json"
            )
            if api_index_path not in checked_indexes:
                checked_indexes[api_index_path] = _content_names(
                    _read_json(api_index_path)
                )
            if relative_path.name not in checked_indexes[api_index_path]:
                issues.append(f"notebook absent from Lite API index: {relative_posix}")

            gallery_html = Path(app.outdir, relative_path.with_suffix(".html"))
            try:
                html = gallery_html.read_text(encoding="utf-8")
            except OSError as error:
                issues.append(f"unable to read gallery page {gallery_html}: {error}")
            else:
                if f"path={relative_posix}" not in html:
                    issues.append(
                        f"gallery Lite link does not target {relative_posix}: "
                        f"{gallery_html}"
                    )

    if notebook_count == 0:
        issues.append("no generated gallery notebooks were found")

    if issues:
        details = "\n- ".join(issues[:20])
        if len(issues) > 20:
            details += f"\n- ... and {len(issues) - 20} more"
        raise SphinxError(f"JupyterLite gallery validation failed:\n- {details}")

    LOGGER.info("validated %d JupyterLite gallery notebook(s)", notebook_count)


def setup(app: Sphinx) -> dict[str, bool]:
    """Register JupyterLite configuration and validation hooks."""
    app.connect("config-inited", configure_jupyterlite_contents, priority=999)
    app.connect("build-finished", validate_jupyterlite_gallery, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
