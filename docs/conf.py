"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------
import json
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import nbformat
import plotly.io as pio
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx.errors import SphinxError
from sphinx_gallery.sorting import FileNameSortKey

import skfolio

# Configure plotly to integrate its output into the HTML pages generated by
# sphinx-gallery.
pio.renderers.default = "sphinx_gallery_png"  # "sphinx_gallery"

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=(
        "Values in x were outside bounds during a minimize step, clipping to bounds"
    ),
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -- Project information -----------------------------------------------------

project = "skfolio"
copyright = "2025, skfolio developers (BSD License)"
author = "Hugo Delatte"

# -- SEO meta tags ------------------------------------------------------------
html_meta = {
    "robots": "index, follow",
}

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_togglebutton",
    "sphinx_favicon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "sphinx.ext.githubpages",
    "jupyterlite_sphinx",
]

# Produce `plot::` directives for examples that contain `import matplotlib` or
# `from matplotlib import`.
numpydoc_use_plots = True

# Options for the `::plot` directive:
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_formats = ["png"]
plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False

autodoc_default_options = {"members": True, "inherited-members": True}

# Don't show type hint in functions and classes
autodoc_typehints = "none"

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False

# If false, no module index is generated.
latex_domain_indices = False

# this is needed to remove warnings on the missing methods docstrings.
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# Copy robots.txt into the HTML root
html_extra_path = ["robots.txt"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Example section order  ------------------------------------------------
# We don't insert the number in the name other the link would change each time
# we want to re-order the examples.
ORDER_OF_EXAMPLES = {
    "mean_risk": 1,
    "risk_budgeting": 2,
    "synthetic_data": 3,
    "entropy_pooling": 4,
    "clustering": 5,
    "maximum_diversification": 6,
    "distributionally_robust_cvar": 7,
    "ensemble": 8,
    "model_selection": 9,
    "pre_selection": 10,
    "metadata_routing": 11,
    "data_preparation": 12,
}


# -- sphinxext-opengraph ----------------------------------------------------

ogp_site_url = "https://skfolio.org/"
ogp_site_name = "skfolio"
ogp_image = "https://skfolio.org/_images/expo.jpg"
ogp_enable_meta_description = True

# -- autosummary -------------------------------------------------------------

autosummary_generate = True

# -- sphinx_sitemap -------------------------------------------------------------
html_baseurl = "https://skfolio.org/"
sitemap_url_scheme = "{link}"

sitemap_excludes = [
    "search.html",
]
# -- Internationalization ----------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- MyST options ------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- sphinx-favicons ------------------------------------------------------------
favicons = [
    {
        "rel": "shortcut icon",
        "type": "image/svg+xml",
        "sizes": "any",
        "href": "favicon.svg",
    },
    {
        "rel": "icon",
        "type": "image/svg+xml",
        "sizes": "any",
        "href": "favicon.svg",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "16x16",
        "href": "favicon-16.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "48x48",
        "href": "favicon-48.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "144x144",
        "href": "favicon-144.png",
    },
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_sourcelink_suffix = ""

# Define the version we use for matching in the version switcher.
# For local development, infer the version to match from the package.
release = skfolio.__version__
version_match = "v" + release

html_theme_options = {
    "pygments_light_style": "friendly",  # "friendly",
    "pygments_dark_style": "dracula",  # "monokai", # dracula highlight print
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/skfolio/skfolio",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "skfolio",
        "alt_text": "skfolio documentation - Home",
        "image_light": "_static/favicon.svg",
        "image_dark": "_static/favicon.svg",
    },
    # "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": (
        "left"
    ),  # [left, content, right] For testing that the navbar items align properly
    "announcement": """<div class="sidebar-message">
    If you'd like to contribute,
    <a href="https://github.com/skfolio/skfolio">check out our GitHub repository.</a>
    Your contributions are welcome!</div>""",
    "secondary_sidebar_items": [],  # No secondary sidebar due to bug with plotly
}

html_sidebars = {
    "auto_examples/*/*": [],  # no primary sidebar
    # "examples/persistent-search-field": ["search-field"],
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

html_context = {
    "github_user": "skfolio",
    "github_repo": "skfolio",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "dark",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_js_files = ["custom-icon.js"]
# todo_include_todos = True

# -- gallery  ----------------------------------------------------------------

image_scrapers = (
    "matplotlib",
    plotly_sg_scraper,
)


class FileNameNumberSortKey(FileNameSortKey):
    """Sort examples in src_dir by file name number.

    Parameters
    ----------
    src_dir : str
        The source directory.
    """

    def __call__(self, filename) -> float:
        # filename="plot_10_tracking_error.py"
        return float(filename.split("_")[1])


def custom_section_order(section_name) -> int:
    # section_name = "..\\examples\\10_data_preparation"
    return ORDER_OF_EXAMPLES[Path(section_name).name]


sphinx_gallery_conf = {
    "doc_module": "skfolio",
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": False,
    "reference_url": {
        "skfolio": None,
    },
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "subsection_order": custom_section_order,
    "within_subsection_order": FileNameNumberSortKey,
    "image_scrapers": image_scrapers,
    # avoid generating too many cross links
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
    "binder": {
        "org": "skfolio",
        "repo": "skfolio",
        "branch": "gh-pages",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "./binder/requirements.txt",
        "use_jupyter_lab": True,
    },
    "write_computation_times": False,
    # 'compress_images': ('images', 'thumbnails'),
    # 'promote_jupyter_magic': False,
    # 'junit': os.path.join('sphinx-gallery', 'junit-results.xml'),
    # # capture raw HTML or, if not present, __repr__ of last expression in
    # # each code block
    # 'capture_repr': ('_repr_html_', '__repr__'),
    # 'matplotlib_animations': True,
    # 'image_srcset': ["2x"],
    # 'nested_sections': False,
    # 'show_api_usage': True,
}

# -- jupyterlite  ----------------------------------------------------------------
# Read more at https://jupyterlite-sphinx.readthedocs.io/en/latest/configuration.html#configuration

# We use the current directory
jupyterlite_dir = str(Path(__file__).parent.absolute())

# Pure-python packages which are not available in Pyodide distribution but `skfolio`
# depends on
PACKAGES_TO_PRE_INSTALL = ["plotly", "nbformat"]

# Runtime dependencies of `skfolio` which need to be pre-imported by the Pyodide kernel
# before running notebooks
PACKAGES_TO_PRE_IMPORT = ["pandas", "sklearn", "plotly", "cvxpy", "nbformat", "skfolio"]

# Each JupyterLite notebook's first cell will be set to this piece of code ensuring that
# the environment is set up correctly
PATCH_CELL_CODE = f"""
# JupyterLite Initialization

# We want the execution to be quiet
import warnings
warnings.filterwarnings('ignore')

# Install missing deps into Pyodide Kernel
import piplite
await piplite.install({json.dumps(PACKAGES_TO_PRE_INSTALL)})
await piplite.install(['skfolio'], deps=False)

# Allows external dataset download
import pyodide_http
pyodide_http.patch_all()

# Run top-level imports
import {", ".join(PACKAGES_TO_PRE_IMPORT)}
""".strip()

# Actual notebook node with hidden source (collapsed by default in the UI)
PATCH_CELL = nbformat.v4.new_code_cell(
    PATCH_CELL_CODE,
    metadata={"tags": ["jupyterlite"], "jupyter": {"source_hidden": True}},
)


# -- Sphinx Hooks ----------------------------------------------------------------


def patch_jupyterlite_notebooks(app, exception):
    """
    Iterates over all ipynb files in the _build/lite/files directory and prepends the
    `PATCH_CELL` node to each notebook.

    We assume that the entire Sphinx build has been completed prior to running this
    function.

    :raises FileNotFoundError if the JupyterLite build directory is not found
    """
    print("Running Patch jupyterlite notebooks...")
    # 1) Skip on build errors
    if exception:
        warnings.warn(
            f"Sitemap hook: skipping because build failed ({exception!r})", stacklevel=2
        )
        return

    built_jupyterlite_dir = Path(jupyterlite_dir, "_build", "lite")
    if not built_jupyterlite_dir.exists():
        raise FileNotFoundError(
            f"JupyterLite build directory not found at {built_jupyterlite_dir}."
        )

    notebook_paths = Path(built_jupyterlite_dir, "files").rglob("*.ipynb")
    for notebook_path in notebook_paths:
        print(f"Patching {notebook_path}")
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            nb.cells.insert(0, PATCH_CELL)
            # Remove any 'id' fields
            for cell in nb.cells:
                cell.pop("id", None)
            with open(notebook_path, "w") as file:
                nbformat.write(nb, file, version=nbformat.NO_CONVERT)


def prune_and_fix_sitemap(app, exception):
    print("Running Prune and fix sitemap...")
    # 1) Skip on build errors
    if exception:
        warnings.warn(
            f"Sitemap hook: skipping because build failed ({exception!r})", stacklevel=2
        )
        return

    # 2) Only for HTML builder
    if app.builder.name != "html":
        warnings.warn(
            f"Sitemap hook: builder is '{app.builder.name}', not 'html' — skipping",
            stacklevel=2,
        )
        return

    sitemap_path = Path(app.outdir) / app.config.sitemap_filename

    # 3) Ensure sitemap exists
    if not sitemap_path.exists():
        warnings.warn(
            f"Sitemap hook: '{sitemap_path}' not found, skipping", stacklevel=2
        )
        return

    try:
        # Parse existing sitemap
        tree = ET.parse(sitemap_path)
        root = tree.getroot()
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        removed = 0

        for url in list(root.findall("sm:url", ns)):
            loc = url.find("sm:loc", ns)
            if loc is None or not loc.text:
                continue
            href = loc.text
            path = urlparse(href).path

            # drop any viewcode pages under /_modules/
            if path.startswith("/_modules/"):
                root.remove(url)
                removed += 1
                continue

            # rewrite only the root index.html → /
            if path == "/index.html":
                loc.text = app.config.html_baseurl.rstrip("/") + "/"

        if removed:
            warnings.warn(
                f"Sitemap hook: removed {removed} entries under '/_modules/'",
                stacklevel=2,
            )

        # Register default namespace so no ns0 prefix appears
        ET.register_namespace("", ns["sm"])

        # Write back, using default_namespace (Python 3.8+)
        tree.write(
            sitemap_path,
            encoding="utf-8",
            xml_declaration=True,
            default_namespace=ns["sm"],
        )

    except ET.ParseError as pe:
        raise SphinxError(
            f"Sitemap hook: XML parse error in '{sitemap_path}': {pe}"
        ) from pe
    except Exception as e:
        raise SphinxError(
            f"Sitemap hook: unexpected error during post‑processing: {e}"
        ) from e


def override_canonical(app, pagename, templatename, context, doctree):
    # only run if you have a base URL set
    if not app.config.html_baseurl:
        return
    # Homepage → slash-only
    if pagename == "index":
        context["pageurl"] = html_baseurl.rstrip("/") + "/"

def setup(app):
    """Setup function to register the build-finished hook."""
    # register existing hook
    app.connect("build-finished", patch_jupyterlite_notebooks)
    app.connect("build-finished", prune_and_fix_sitemap)
    # add the canonical-URL hook
    app.connect("html-page-context", override_canonical)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
