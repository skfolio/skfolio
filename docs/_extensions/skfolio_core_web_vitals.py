"""Core Web Vitals optimizations for interactive tutorial pages."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any

from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

_ASSET_ROOT = Path("_static") / "plotly"
_ASSET_URL_ROOT = "_static/plotly"
_LOADER_PATH = "scripts/lazy-plotly.js"
_HTML_SCRIPT_RE = re.compile(
    r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
    flags=re.IGNORECASE | re.DOTALL,
)
_PLOTLY_CDN_SRC_RE = re.compile(
    r"""https://cdn\.plot\.ly/plotly-[^"' ]+\.min\.js""",
    flags=re.IGNORECASE,
)
_PLOTLY_GRAPH_DIV_RE = re.compile(
    r"""<div(?P<attrs>[^>]*\bclass=(?P<quote>["'])"""
    r"""[^"']*\bplotly-graph-div\b[^"']*(?P=quote)[^>]*)>""",
    flags=re.IGNORECASE,
)
_HTML_CLASS_RE = re.compile(
    r"""\bclass\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""",
    flags=re.IGNORECASE | re.DOTALL,
)
_LAUNCH_BADGE_IMG_RE = re.compile(
    r"""<img(?P<attrs>[^>]*\balt=["']Launch """
    r"""(?P<service>binder|JupyterLite)["'][^>]*)/?>""",
    flags=re.IGNORECASE,
)
_LAUNCH_BADGE_DIMENSIONS = {
    "binder": (109, 20),
    "jupyterlite": (91, 20),
}
_GALLERY_THUMBNAIL_IMG_RE = re.compile(
    r"""<img(?P<attrs>(?=[^>]*\bsrc=["'][^"']*"""
    r"""sphx_glr_[^"']+_thumb\.png["'])[^>]*)/?>""",
    flags=re.IGNORECASE,
)
_GALLERY_THUMBNAIL_DIMENSIONS = (400, 280)


@dataclass(frozen=True)
class _PlotlyMetadata:
    """Metadata for the Plotly runtime shared by a tutorial page."""

    source: str
    integrity: str | None
    crossorigin: str | None


@dataclass(frozen=True)
class _ChartAsset:
    """A content-addressed Plotly initializer asset."""

    name: str
    payload: bytes
    url: str


@dataclass(frozen=True)
class _PagePlan:
    """A validated HTML transformation and its generated assets."""

    body: str
    metadata: _PlotlyMetadata | None
    assets: tuple[_ChartAsset, ...]
    runtime_scripts: int
    payload_bytes: int


def _page_path(pagename: str) -> PurePosixPath:
    """Return a validated documentation page path."""
    path = PurePosixPath(pagename)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise SphinxError(f"Core Web Vitals: unsafe page name '{pagename}'")
    return path


def _is_tutorial_page(pagename: str) -> bool:
    """Return whether a page is an individual Sphinx-Gallery tutorial."""
    path = _page_path(pagename)
    return path.parts[0] == "auto_examples" and path.name != "index"


def _is_gallery_index_page(pagename: str) -> bool:
    """Return whether a page is a Sphinx-Gallery index."""
    path = _page_path(pagename)
    return path.parts[0] == "auto_examples" and path.name == "index"


def _get_html_attribute(attrs: str, name: str) -> str | None:
    """Return an attribute value from an HTML start tag fragment."""
    match = re.search(
        rf"""\b{re.escape(name)}\s*=\s*(?P<quote>["'])(?P<value>.*?)"""
        rf"""(?P=quote)""",
        attrs,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group("value") if match else None


def _add_html_attribute(attrs: str, name: str, value: str) -> str:
    """Add an escaped HTML attribute unless the start tag already contains it."""
    if re.search(rf"\b{re.escape(name)}\s*=", attrs, flags=re.IGNORECASE):
        return attrs
    return f'{attrs} {name}="{escape(value, quote=True)}"'


def _add_html_class(attrs: str, class_name: str) -> str:
    """Add a class to an HTML start tag fragment."""

    def replace(match: re.Match[str]) -> str:
        classes = match.group("value").split()
        if class_name not in classes:
            classes.append(class_name)
        quote = match.group("quote")
        return f"class={quote}{' '.join(classes)}{quote}"

    return _HTML_CLASS_RE.sub(replace, attrs, count=1)


def _collect_plotly_metadata(
    scripts: list[re.Match[str]], pagename: str
) -> tuple[_PlotlyMetadata | None, list[str], int]:
    """Collect and validate Plotly runtime metadata and initializer bodies."""
    runtimes: list[tuple[str, str | None, str | None]] = []
    initializers: list[str] = []
    for script in scripts:
        attrs = script.group("attrs")
        script_body = script.group("body")
        source = _get_html_attribute(attrs, "src")
        if source and _PLOTLY_CDN_SRC_RE.fullmatch(source):
            runtimes.append(
                (
                    source,
                    _get_html_attribute(attrs, "integrity"),
                    _get_html_attribute(attrs, "crossorigin"),
                )
            )
        elif "Plotly.newPlot(" in script_body:
            initializers.append(script_body)

    if not initializers and not runtimes:
        return None, [], 0
    if not initializers:
        raise SphinxError(
            f"Core Web Vitals: Plotly CDN script found without an initializer on "
            f"'{pagename}'"
        )
    if not runtimes:
        raise SphinxError(
            f"Core Web Vitals: Plotly initializers found without a CDN script on "
            f"'{pagename}'"
        )

    sources = {runtime[0] for runtime in runtimes}
    integrities = {runtime[1] for runtime in runtimes}
    crossorigins = {runtime[2] for runtime in runtimes}
    if len(sources) != 1 or len(integrities) != 1 or len(crossorigins) != 1:
        raise SphinxError(
            f"Core Web Vitals: inconsistent Plotly CDN metadata on '{pagename}'"
        )

    metadata = _PlotlyMetadata(
        source=next(iter(sources)),
        integrity=next(iter(integrities)),
        crossorigin=next(iter(crossorigins)),
    )
    return metadata, initializers, len(runtimes)


def _build_chart_assets(
    context: dict[str, Any], pagename: str, initializers: list[str]
) -> tuple[_ChartAsset, ...]:
    """Build content-addressed asset descriptions without writing files."""
    asset_url_dir = f"{_ASSET_URL_ROOT}/{pagename}"
    assets = []
    for index, initializer in enumerate(initializers, start=1):
        payload = initializer.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:20]
        name = f"chart-{index:02d}-{digest}.js"
        url = context["pathto"](f"{asset_url_dir}/{name}", 1)
        assets.append(_ChartAsset(name=name, payload=payload, url=url))
    return tuple(assets)


def _remove_plotly_scripts(body: str, pagename: str, expected_initializers: int) -> str:
    """Remove Plotly runtime, configuration, and initializer scripts."""
    removed_initializers = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed_initializers

        attrs = match.group("attrs")
        script_body = match.group("body")
        source = _get_html_attribute(attrs, "src")
        if source and _PLOTLY_CDN_SRC_RE.fullmatch(source):
            return ""
        if "Plotly.newPlot(" in script_body:
            removed_initializers += 1
            return ""
        if "window.PlotlyConfig" in script_body:
            return ""
        return match.group(0)

    transformed = _HTML_SCRIPT_RE.sub(replace, body)
    if removed_initializers != expected_initializers:
        raise SphinxError(
            f"Core Web Vitals: removed {removed_initializers} of "
            f"{expected_initializers} Plotly initializers on '{pagename}'"
        )
    return transformed


def _mark_plot_containers(
    body: str, pagename: str, assets: tuple[_ChartAsset, ...]
) -> str:
    """Attach lazy-loading metadata directly to each Plotly container."""
    asset_iterator = iter(assets)

    def replace(match: re.Match[str]) -> str:
        asset = next(asset_iterator)
        attrs = _add_html_class(match.group("attrs"), "skfolio-lazy-plot")
        attrs = _add_html_attribute(attrs, "data-skfolio-plotly-src", asset.url)
        return f"<div{attrs}>"

    transformed, plot_count = _PLOTLY_GRAPH_DIV_RE.subn(replace, body)
    if plot_count != len(assets):
        raise SphinxError(
            f"Core Web Vitals: found {plot_count} Plotly containers and "
            f"{len(assets)} initializers on '{pagename}'"
        )
    return transformed


def _optimize_launch_badges(body: str) -> str:
    """Set stable dimensions and deferred decoding on tutorial launch badges."""

    def replace(match: re.Match[str]) -> str:
        attrs = match.group("attrs").rstrip().removesuffix("/").rstrip()
        service = match.group("service").lower()
        width, height = _LAUNCH_BADGE_DIMENSIONS[service]
        attrs = _add_html_attribute(attrs, "loading", "lazy")
        attrs = _add_html_attribute(attrs, "decoding", "async")
        attrs = _add_html_attribute(attrs, "width", str(width))
        attrs = _add_html_attribute(attrs, "height", str(height))
        return f"<img{attrs} />"

    return _LAUNCH_BADGE_IMG_RE.sub(replace, body)


def _optimize_gallery_thumbnails(body: str) -> str:
    """Set stable dimensions and defer below-the-fold gallery thumbnails."""
    thumbnail_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal thumbnail_index
        attrs = match.group("attrs").rstrip().removesuffix("/").rstrip()
        width, height = _GALLERY_THUMBNAIL_DIMENSIONS
        attrs = _add_html_attribute(
            attrs, "loading", "eager" if not thumbnail_index else "lazy"
        )
        attrs = _add_html_attribute(attrs, "decoding", "async")
        attrs = _add_html_attribute(attrs, "width", str(width))
        attrs = _add_html_attribute(attrs, "height", str(height))
        thumbnail_index += 1
        return f"<img{attrs} />"

    return _GALLERY_THUMBNAIL_IMG_RE.sub(replace, body)


def _create_page_plan(context: dict[str, Any], pagename: str) -> _PagePlan:
    """Create and fully validate a page transformation before writing assets."""
    original_body = context.get("body", "")
    scripts = list(_HTML_SCRIPT_RE.finditer(original_body))
    metadata, initializers, runtime_scripts = _collect_plotly_metadata(
        scripts, pagename
    )
    plot_count = sum(1 for _ in _PLOTLY_GRAPH_DIV_RE.finditer(original_body))
    if metadata is None:
        if plot_count:
            raise SphinxError(
                f"Core Web Vitals: found {plot_count} Plotly containers without "
                f"initializers on '{pagename}'"
            )
        return _PagePlan(
            body=_optimize_launch_badges(original_body),
            metadata=None,
            assets=(),
            runtime_scripts=0,
            payload_bytes=0,
        )

    if plot_count != len(initializers):
        raise SphinxError(
            f"Core Web Vitals: found {plot_count} Plotly containers and "
            f"{len(initializers)} initializers on '{pagename}'"
        )

    assets = _build_chart_assets(context, pagename, initializers)
    body = _remove_plotly_scripts(
        original_body, pagename, expected_initializers=len(initializers)
    )
    body = _mark_plot_containers(body, pagename, assets)
    body = _optimize_launch_badges(body)
    return _PagePlan(
        body=body,
        metadata=metadata,
        assets=assets,
        runtime_scripts=runtime_scripts,
        payload_bytes=sum(len(asset.payload) for asset in assets),
    )


def _page_asset_dir(app: Sphinx, pagename: str) -> Path:
    """Return the generated asset directory for a validated page name."""
    page_path = _page_path(pagename)
    return Path(app.outdir) / _ASSET_ROOT.joinpath(*page_path.parts)


def _write_asset_atomically(path: Path, payload: bytes) -> None:
    """Write an asset atomically unless the expected content already exists."""
    if path.exists() and path.read_bytes() == payload:
        return

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_page_assets(
    app: Sphinx, pagename: str, assets: tuple[_ChartAsset, ...]
) -> int:
    """Write a validated page plan and prune obsolete chart assets."""
    asset_dir = _page_asset_dir(app, pagename)
    generated_names = {asset.name for asset in assets}
    if not assets and not asset_dir.exists():
        return 0

    try:
        if assets:
            asset_dir.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            _write_asset_atomically(asset_dir / asset.name, asset.payload)
        stale_assets = 0
        for asset_path in asset_dir.glob("chart-*.js"):
            if asset_path.is_file() and asset_path.name not in generated_names:
                asset_path.unlink()
                stale_assets += 1
    except OSError as error:
        raise SphinxError(
            f"Core Web Vitals: unable to write Plotly assets for '{pagename}': {error}"
        ) from error
    return stale_assets


def _register_loader(app: Sphinx, metadata: _PlotlyMetadata) -> None:
    """Register the page-specific Plotly loader through the Sphinx asset API."""
    attributes = {
        "class": "skfolio-plotly-loader",
        "data-plotly-src": metadata.source,
    }
    if metadata.integrity:
        attributes["data-plotly-integrity"] = metadata.integrity
    if metadata.crossorigin:
        attributes["data-plotly-crossorigin"] = metadata.crossorigin
    app.add_js_file(
        _LOADER_PATH,
        priority=900,
        loading_method="defer",
        **attributes,
    )


def optimize_tutorial_core_web_vitals(
    app: Sphinx,
    pagename: str,
    _templatename: str,
    context: dict[str, Any],
    _doctree: Any,
) -> None:
    """Optimize Sphinx-Gallery thumbnails, tutorial media, and Plotly charts."""
    if app.builder.name not in {"html", "dirhtml"}:
        return

    if _is_gallery_index_page(pagename):
        context["body"] = _optimize_gallery_thumbnails(context.get("body", ""))
        return
    if not _is_tutorial_page(pagename):
        return

    plan = _create_page_plan(context, pagename)
    stale_assets = _write_page_assets(app, pagename, plan.assets)
    if plan.metadata:
        _register_loader(app, plan.metadata)
    context["body"] = plan.body
    if not plan.assets and not stale_assets:
        return

    LOGGER.info(
        "Core Web Vitals: externalized %d Plotly chart(s) (%d bytes), "
        "removed %d embedded CDN script(s), and pruned %d stale asset(s) from %s",
        len(plan.assets),
        plan.payload_bytes,
        plan.runtime_scripts,
        stale_assets,
        pagename,
    )


def _remove_empty_asset_directories(root: Path) -> None:
    """Remove empty directories below the generated Plotly asset root."""
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


def cleanup_orphaned_plotly_assets(app: Sphinx, exception: Exception | None) -> None:
    """Remove generated Plotly assets that final HTML pages do not reference."""
    if exception is not None or app.builder.name not in {"html", "dirhtml"}:
        return

    root = Path(app.outdir) / _ASSET_ROOT
    if not root.exists():
        return

    removed_assets = 0
    removed_temporary_files = 0
    html_by_page: dict[str, str] = {}
    try:
        for asset_path in root.rglob("chart-*.js"):
            page = asset_path.parent.relative_to(root).as_posix()
            if page not in html_by_page:
                output_path = Path(app.builder.get_outfilename(page))
                html_by_page[page] = (
                    output_path.read_text(encoding="utf-8")
                    if output_path.is_file()
                    else ""
                )
            if asset_path.name not in html_by_page[page]:
                asset_path.unlink()
                removed_assets += 1
        for temporary_path in root.rglob(".chart-*.tmp"):
            temporary_path.unlink()
            removed_temporary_files += 1
    except OSError as error:
        raise SphinxError(
            f"Core Web Vitals: unable to prune orphaned Plotly assets: {error}"
        ) from error
    _remove_empty_asset_directories(root)

    if removed_assets or removed_temporary_files:
        LOGGER.info(
            "Core Web Vitals: pruned %d orphaned Plotly asset(s) and %d temporary "
            "file(s)",
            removed_assets,
            removed_temporary_files,
        )


def setup(app: Sphinx) -> dict[str, object]:
    """Register the Core Web Vitals documentation extension."""
    app.connect("html-page-context", optimize_tutorial_core_web_vitals)
    app.connect("build-finished", cleanup_orphaned_plotly_assets)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
