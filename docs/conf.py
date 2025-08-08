"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------
import datetime as dt
import json
import os
import re
import warnings
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urlparse

import nbformat
import plotly.io as pio
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx.errors import SphinxError
from sphinx_gallery.sorting import FileNameSortKey

import skfolio

EXAMPLE_DESCRIPTIONS = {
    # Data Preparation
    "auto_examples/data_preparation/plot_1_investment_horizon": (
        "Exploring how investment horizons influence portfolio outcomes"
    ),
    # Pre-selection
    "auto_examples/pre_selection/plot_1_drop_correlated": (
        "Enhancing stability by removing highly correlated assets"
    ),
    "auto_examples/pre_selection/plot_2_select_best_performers": (
        "Pre-selecting top assets based on out-of-sample Sharpe ratios"
    ),
    "auto_examples/pre_selection/plot_3_custom_pre_selection_volumes": (
        "Building a custom filter to retain assets with highest trading volumes"
    ),
    "auto_examples/pre_selection/plot_4_incomplete_dataset": (
        "Managing asset inception, expiry, and defaults within pipelines"
    ),
    # Model Selection
    "auto_examples/model_selection/plot_1_multiple_randomized_cv": (
        "Using Monte Carlo-style Multiple Randomized CV for robust model evaluation"
    ),
    # Mean-Risk Optimization
    "auto_examples/mean_risk/plot_1_maximum_sharpe_ratio": (
        "Maximizing risk-adjusted returns via the Sharpe ratio"
    ),
    "auto_examples/mean_risk/plot_2_minimum_CVaR": (
        "Minimizing Conditional Value-at-Risk (CVaR) in portfolio construction"
    ),
    "auto_examples/mean_risk/plot_3_efficient_frontier": (
        "Visualizing the mean-variance efficient frontier"
    ),
    "auto_examples/mean_risk/plot_4_mean_variance_cdar": (
        "Comparing efficient frontiers under variance and CDaR constraints"
    ),
    "auto_examples/mean_risk/plot_5_weight_constraints": (
        "Imposing upper and lower bounds on asset weights"
    ),
    "auto_examples/mean_risk/plot_6_transaction_costs": (
        "Incorporating transaction costs into rebalancing optimization"
    ),
    "auto_examples/mean_risk/plot_7_management_fees": (
        "Adjusting for ongoing management fees in portfolio design"
    ),
    "auto_examples/mean_risk/plot_8_regularization": (
        "Applying L1/L2 penalties to improve sparsity and out-of-sample performance"
    ),
    "auto_examples/mean_risk/plot_9_uncertainty_set": (
        "Building robust portfolios using uncertainty sets"
    ),
    "auto_examples/mean_risk/plot_10_tracking_error": (
        "Constraining tracking error relative to a benchmark"
    ),
    "auto_examples/mean_risk/plot_11_empirical_prior": (
        "Empirically estimating expected return priors"
    ),
    "auto_examples/mean_risk/plot_12_black_and_litterman": (
        "Integrating market equilibrium and views via Black-Litterman"
    ),
    "auto_examples/mean_risk/plot_13_factor_model": (
        "Modeling returns and covariance with factor-based priors"
    ),
    "auto_examples/mean_risk/plot_14_black_litterman_factor_model": (
        "Enhancing Black-Litterman with factor-model priors"
    ),
    "auto_examples/mean_risk/plot_15_mip_cardinality_constraints": (
        "Limiting portfolio complexity through cardinality constraints"
    ),
    "auto_examples/mean_risk/plot_16_mip_threshold_constraints": (
        "Enforcing long/short threshold constraints via mixed-integer programming"
    ),
    # Risk Budgeting
    "auto_examples/risk_budgeting/plot_1_risk_parity_variance": (
        "Allocating capital by equalizing variance contributions"
    ),
    "auto_examples/risk_budgeting/plot_2_risk_budgeting_CVaR": (
        "Balancing risk contributions under a CVaR budget"
    ),
    "auto_examples/risk_budgeting/plot_3_risk_parity_ledoit_wolf": (
        "Stabilizing risk parity with covariance shrinkage"
    ),
    # Synthetic Data & Stress Testing
    "auto_examples/synthetic_data/plot_1_bivariate_copulas": (
        "Simulating asset dependencies using bivariate copulas"
    ),
    "auto_examples/synthetic_data/plot_2_vine_copula": (
        "Stress-testing portfolios under vine-copula dependency shocks"
    ),
    "auto_examples/synthetic_data/plot_3_min_CVaR_stressed_factors": (
        "Designing portfolios to minimize CVaR under stressed factor scenarios"
    ),
    # Entropy & Opinion Pooling
    "auto_examples/entropy_pooling/plot_1_entropy_pooling": (
        "Integrating scenario views through entropy pooling"
    ),
    "auto_examples/entropy_pooling/plot_2_opinion_pooling": (
        "Combining expert forecasts via opinion pooling"
    ),
    # Ensemble Optimizations
    "auto_examples/ensemble/plot_1_stacking": (
        "Combining multiple portfolio strategies using stacking optimization"
    ),
    # Hierarchical Clustering & NCO
    "auto_examples/clustering/plot_1_hrp_cvar": (
        "Allocating by CVaR-based hierarchical risk parity"
    ),
    "auto_examples/clustering/plot_2_herc_cdar": (
        "Hierarchical equal-risk contribution under CDaR"
    ),
    "auto_examples/clustering/plot_3_hrp_vs_herc": (
        "Comparing HRP and HERC hierarchical portfolios"
    ),
    "auto_examples/clustering/plot_4_nco": (
        "Nested cluster optimization for hierarchical groups"
    ),
    "auto_examples/clustering/plot_5_nco_grid_search": (
        "Merging NCO with combinatorial purged CV cross-validation"
    ),
    # Maximum Diversification
    "auto_examples/maximum_diversification/plot_1_maximum_diversification": (
        "Maximizing the diversification ratio in portfolio selection"
    ),
    # Distributionally Robust CVaR
    "auto_examples/distributionally_robust_cvar/plot_1_distributionally_robust_cvar": (
        "Optimizing CVaR under distributional robustness"
    ),
    # Metadata Routing
    "auto_examples/metadata_routing/plot_1_implied_volatility": (
        "Routing implied volatility data into optimization models"
    ),
}


EXAMPLE_LAST_UPDATED = {
    "auto_examples/index": str(dt.date.today()),
    # Data Preparation
    "auto_examples/data_preparation/plot_1_investment_horizon": "2023-12-18",
    # Pre-selection
    "auto_examples/pre_selection/plot_1_drop_correlated": "2023-12-18",
    "auto_examples/pre_selection/plot_2_select_best_performers": "2023-12-18",
    "auto_examples/pre_selection/plot_3_custom_pre_selection_volumes": "2025-04-05",
    "auto_examples/pre_selection/plot_4_incomplete_dataset": "2025-04-05",
    # Model Selection
    "auto_examples/model_selection/plot_1_multiple_randomized_cv": "2025-07-26",
    # Mean-Risk Optimization
    "auto_examples/mean_risk/plot_1_maximum_sharpe_ratio": "2023-12-18",
    "auto_examples/mean_risk/plot_2_minimum_CVaR": "2023-12-18",
    "auto_examples/mean_risk/plot_3_efficient_frontier": "2023-12-18",
    "auto_examples/mean_risk/plot_4_mean_variance_cdar": "2023-12-18",
    "auto_examples/mean_risk/plot_5_weight_constraints": "2023-12-18",
    "auto_examples/mean_risk/plot_6_transaction_costs": "2023-12-18",
    "auto_examples/mean_risk/plot_7_management_fees": "2023-12-18",
    "auto_examples/mean_risk/plot_8_regularization": "2023-12-18",
    "auto_examples/mean_risk/plot_9_uncertainty_set": "2023-12-18",
    "auto_examples/mean_risk/plot_10_tracking_error": "2023-12-18",
    "auto_examples/mean_risk/plot_11_empirical_prior": "2023-12-18",
    "auto_examples/mean_risk/plot_12_black_and_litterman": "2023-12-18",
    "auto_examples/mean_risk/plot_13_factor_model": "2023-12-18",
    "auto_examples/mean_risk/plot_14_black_litterman_factor_model": "2023-12-18",
    "auto_examples/mean_risk/plot_15_mip_cardinality_constraints": "2024-11-19",
    "auto_examples/mean_risk/plot_16_mip_threshold_constraints": "2024-11-19",
    # Risk Budgeting
    "auto_examples/risk_budgeting/plot_1_risk_parity_variance": "2023-12-18",
    "auto_examples/risk_budgeting/plot_2_risk_budgeting_CVaR": "2023-12-18",
    "auto_examples/risk_budgeting/plot_3_risk_parity_ledoit_wolf": "2023-12-18",
    # Synthetic Data & Stress Testing
    "auto_examples/synthetic_data/plot_1_bivariate_copulas": "2025-03-21",
    "auto_examples/synthetic_data/plot_2_vine_copula": "2025-03-21",
    "auto_examples/synthetic_data/plot_3_min_CVaR_stressed_factors": "2025-03-21",
    # Entropy & Opinion Pooling
    "auto_examples/entropy_pooling/plot_1_entropy_pooling": "2025-06-09",
    "auto_examples/entropy_pooling/plot_2_opinion_pooling": "2025-06-09",
    # Ensemble Optimizations
    "auto_examples/ensemble/plot_1_stacking": "2023-12-18",
    # Hierarchical Clustering & NCO
    "auto_examples/clustering/plot_1_hrp_cvar": "2023-12-18",
    "auto_examples/clustering/plot_2_herc_cdar": "2023-12-18",
    "auto_examples/clustering/plot_3_hrp_vs_herc": "2023-12-18",
    "auto_examples/clustering/plot_4_nco": "2023-12-18",
    "auto_examples/clustering/plot_5_nco_grid_search": "2023-12-18",
    # Maximum Diversification
    "auto_examples/maximum_diversification/plot_1_maximum_diversification": "2023-12-18",
    # Distributionally Robust CVaR
    "auto_examples/distributionally_robust_cvar/plot_1_distributionally_robust_cvar": "2023-12-18",
    # Metadata Routing
    "auto_examples/metadata_routing/plot_1_implied_volatility": "2023-12-18",
}


# Map old *docname* (no .rst/.html) -> new URL (root-relative or absolute)
REDIRECTS = {
    "auto_examples/5_distributionally_robust_cvar/plot_1_distributionally_robust_cvar": "/auto_examples/distributionally_robust_cvar/plot_1_distributionally_robust_cvar.html",
    "auto_examples/6_clustering/plot_5_nco_grid_search": "/auto_examples/clustering/plot_5_nco_grid_search.html",
    "auto_examples/6_clustering/plot_4_nco": "/auto_examples/clustering/plot_4_nco.html",
    "auto_examples/7_ensemble/plot_1_stacking": "/auto_examples/ensemble/plot_1_stacking.html",
    "auto_examples/6_clustering/plot_3_hrp_vs_herc": "/auto_examples/clustering/plot_3_hrp_vs_herc.html",
    "auto_examples/2_risk_budgeting/plot_3_risk_parity_ledoit_wolf": "/auto_examples/risk_budgeting/plot_3_risk_parity_ledoit_wolf.html",
    "auto_examples/6_clustering/index": "/auto_examples/clustering/index.html",
    "auto_examples/1_mean_risk/plot_8_regularization": "/auto_examples/mean_risk/plot_8_regularization.html",
    "auto_examples/6_ensemble/plot_1_stacking": "/auto_examples/ensemble/plot_1_stacking.html",
    "auto_examples/1_mean_risk/plot_7_management_fees": "/auto_examples/mean_risk/plot_7_management_fees.html",
    "auto_examples/1_mean_risk/plot_1_maximum_sharpe_ratio": "/auto_examples/mean_risk/plot_1_maximum_sharpe_ratio.html",
    "auto_examples/1_mean_risk/plot_13_factor_model": "/auto_examples/mean_risk/plot_13_factor_model.html",
    "auto_examples/1_mean_risk/plot_15_mip_cardinality_constraints": "/auto_examples/mean_risk/plot_15_mip_cardinality_constraints.html",
    "auto_examples/1_mean_risk/plot_2_minimum_CVaR": "/auto_examples/mean_risk/plot_2_minimum_CVaR.html",
    "auto_examples/2_risk_budgeting/index": "/auto_examples/risk_budgeting/index.html",
    "auto_examples/8_pre_selection/plot_4_incomplete_dataset": "/auto_examples/pre_selection/plot_4_incomplete_dataset.html",
    "auto_examples/1_mean_risk/plot_12_black_and_litterman": "/auto_examples/mean_risk/plot_12_black_and_litterman.html",
    "auto_examples/2_risk_budgeting/plot_2_risk_budgeting_CVaR": "/auto_examples/risk_budgeting/plot_2_risk_budgeting_CVaR.html",
    "auto_examples/1_mean_risk/index": "/auto_examples/mean_risk/index.html",
    "auto_examples/1_mean_risk/plot_10_tracking_error": "/auto_examples/mean_risk/plot_10_tracking_error.html",
    "auto_examples/3_synthetic_data/plot_1_bivariate_copulas": "/auto_examples/synthetic_data/plot_1_bivariate_copulas.html",
    "auto_examples/1_mean_risk/plot_16_mip_threshold_constraints": "/auto_examples/mean_risk/plot_16_mip_threshold_constraints.html",
    "auto_examples/5_clustering/plot_5_nco_grid_search": "/auto_examples/clustering/plot_5_nco_grid_search.html",
    "auto_examples/5_clustering/plot_3_hrp_vs_herc": "/auto_examples/clustering/plot_3_hrp_vs_herc.html",
    "auto_examples/5_clustering/plot_4_nco": "/auto_examples/clustering/plot_4_nco.html",
    "auto_examples/3_maxiumum_diversification/index": "/auto_examples/maximum_diversification/index.html",
    "auto_examples/9_data_preparation/index": "/auto_examples/data_preparation/index.html",
    "auto_examples/7_pre_selection/index": "/auto_examples/pre_selection/index.html",
    "auto_examples/4_distributionally_robust_cvar/plot_1_distributionally_robust_cvar": "/auto_examples/distributionally_robust_cvar/plot_1_distributionally_robust_cvar.html",
    "auto_examples/6_ensemble/index": "/auto_examples/ensemble/index.html",
    "auto_examples/5_clustering/index": "/auto_examples/clustering/index.html",
    "auto_examples/4_distributionally_robust_cvar/index": "/auto_examples/distributionally_robust_cvar/index.html",
    "auto_examples/8_metadata_routing/index": "/auto_examples/metadata_routing/index.html",
    "auto_examples/8_data_preparation/index": "/auto_examples/data_preparation/index.html",
}


def get_example_headline_and_description(app, pagename) -> tuple[str, str]:
    title = get_doc_title(app, pagename)

    headline = f"Tutorial on {title}"

    end_example = (
        "using skfolio, a Python library for portfolio optimization and "
        "risk management."
    )

    example_desc = EXAMPLE_DESCRIPTIONS.get(pagename)
    if example_desc:
        description = f"{example_desc} {end_example}"
    else:
        warnings.warn(f"Description missing for example {pagename}", stacklevel=2)
        description = f"{headline} {end_example}"

    return headline, description


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

html_title = "skfolio"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_favicon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "sphinx.ext.githubpages",
    "jupyterlite_sphinx",
    "sphinx_last_updated_by_git",
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

# Copy robots.txt into the HTML root
html_extra_path = ["robots.txt"]

# Last updated date format
html_last_updated_fmt = "%Y-%m-%d"


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
ogp_description_length = 160

# -- sphinx_last_updated_by_git  ----------------------------------------------------

git_untracked_check_dependencies = True

# -- autosummary ------------------------------- ------------------------------

autosummary_generate = True

# -- sphinx_sitemap -------------------------------------------------------------
html_baseurl = "https://skfolio.org/"
sitemap_url_scheme = "{link}"
sitemap_show_lastmod = True
sitemap_excludes = ["search.html"]
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

            # Remove nested index.html pages under auto_examples
            if re.match(r"^/auto_examples/.+/index\.html$", path):
                root.remove(url)
                removed += 1
                continue

            # rewrite only the root index.html → /
            if path == "/index.html":
                loc.text = app.config.html_baseurl.rstrip("/") + "/"

                # Inject <priority>1.0</priority> for the homepage
                priority_el = url.find("sm:priority", ns)
                if priority_el is None:
                    priority_el = ET.SubElement(url, f"{{{ns['sm']}}}priority")
                priority_el.text = "1.0"

            # Inject known <lastmod>
            lastmod = EXAMPLE_LAST_UPDATED.get(path.lstrip("/").removesuffix(".html"))
            if lastmod:
                lastmod_el = url.find("sm:lastmod", ns)
                if lastmod_el is None:
                    lastmod_el = ET.SubElement(url, f"{{{ns['sm']}}}lastmod")
                lastmod_el.text = lastmod

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
            f"Sitemap hook: unexpected error during post-processing: {e}"
        ) from e


def override_canonical(app, pagename, templatename, context, doctree):
    # only run if you have a base URL set
    if not app.config.html_baseurl:
        return
    # Homepage → slash-only
    if pagename == "index":
        context["pageurl"] = html_baseurl.rstrip("/") + "/"


def get_doc_title(app, pagename) -> str:
    """Retrieve the title text for a given docname from the Sphinx environment."""
    title_node = app.env.titles.get(pagename)
    if title_node:
        return title_node.astext()

    raise ValueError(f"Failed to retrieve title from {pagename}")


def inject_schema(app, pagename, templatename, context, doctree):
    base = app.config.html_baseurl.rstrip("/")
    schema = None
    date_published = str(dt.date(2023, 12, 18))
    date_modified = context.get("last_updated", str(dt.date.today()))

    # Main Documentation Site
    if pagename == "index":
        schema = {
            "@context": "https://schema.org/",
            "@type": "WebSite",
            "name": "skfolio Documentation",
            "url": f"{base}/",
            "description": (
                "Python library for portfolio optimization and risk management built "
                "on scikit-learn to create, fine-tune, cross-validate and stress-test "
                "portfolio models."
            ),
            "datePublished": date_published,
            "dateModified": date_modified,
            "hasPart": [
                {"@type": "HowTo", "url": "https://skfolio.org/user_guide/index.html"},
                {
                    "@type": "TechArticle",
                    "url": "https://skfolio.org/auto_examples/index.html",
                },
                {"@type": "APIReference", "url": "https://skfolio.org/api.html"},
            ],
        }

    # User Guide Index: dynamically discover all subpages
    elif pagename == "user_guide/index":
        # discover docs under user_guide/, excluding index
        all_steps = sorted(
            [
                doc
                for doc in app.env.found_docs
                if doc.startswith("user_guide/") and doc != "user_guide/index"
            ]
        )
        steps = []
        for doc in all_steps:
            title = get_doc_title(app, doc)
            steps.append(
                {"@type": "HowToStep", "name": title, "url": f"{base}/{doc}.html"}
            )
        schema = {
            "@context": "https://schema.org/",
            "@type": "HowTo",
            "name": "skfolio User Guide",
            "description": (
                "Comprehensive guide to installing, configuring, and using "
                "the skfolio Python library."
            ),
            "url": f"{base}/user_guide/index.html",
            "step": steps,
            "author": {"@type": "Organization", "name": "skfolio developers"},
            "datePublished": date_published,
            "dateModified": date_modified,
        }

        # Individual User Guide Steps: any page under user_guide/ except index

    elif pagename.startswith("user_guide/") and pagename != "user_guide/index":
        title = get_doc_title(app, pagename)

        schema = {
            "@context": "https://schema.org/",
            "@type": "HowToStep",
            "name": title,
            "url": f"{base}/{pagename}.html",
            "text": f"User Guide section: {title}.",
            "author": {"@type": "Organization", "name": "skfolio developers"},
            "datePublished": date_published,
            "dateModified": date_modified,
        }

    # API Reference Page
    elif pagename == "api":
        schema = {
            "@context": "https://schema.org/",
            "@type": "APIReference",
            "name": "skfolio API Reference",
            "description": (
                "Complete reference for the skfolio Python library's API: "
                "functions, classes, and modules."
            ),
            "url": f"{base}/api.html",
            "programmingModel": "Python",
            "targetPlatform": "Any platform running Python 3.10+",
            "version": app.config.release,
            "author": {"@type": "Organization", "name": "skfolio developers"},
            "publisher": {
                "@type": "Organization",
                "name": "skfolio",
                "url": "https://skfolio.org/",
            },
            "datePublished": date_published,
            "dateModified": date_modified,
        }

    # 4) Main Example Page as TechArticle with hasPart listing example TechArticles
    elif pagename == "auto_examples/index":
        all_examples = sorted(
            [
                doc
                for doc in app.env.found_docs
                if doc.startswith("auto_examples/")
                and not doc.endswith("index")
                and "/index" not in doc  # filters nested auto_examples/**/index
            ]
        )
        parts = []
        for doc in all_examples:
            headline, _ = get_example_headline_and_description(app, doc)
            parts.append(
                {
                    "@type": "TechArticle",
                    "headline": headline,
                    "url": f"{base}/{doc}.html",
                }
            )
        schema = {
            "@context": "https://schema.org/",
            "@type": "TechArticle",
            "headline": "Code Examples & Tutorials",
            "description": (
                "A gallery of code examples and tutorials demonstrating how to "
                "use skfolio for portfolio optimization."
            ),
            "url": f"{base}/auto_examples/index.html",
            "hasPart": parts,
            "author": {"@type": "Organization", "name": "skfolio developers"},
            "datePublished": date_published,
            "dateModified": date_modified,
        }

    # 2) Examples Page as TechArticle
    elif pagename.startswith("auto_examples/") and not pagename.endswith("index"):
        headline, desc = get_example_headline_and_description(app, pagename)
        example_date = EXAMPLE_LAST_UPDATED.get(pagename, str(dt.date.today()))
        schema = {
            "@context": "https://schema.org/",
            "@type": "TechArticle",
            "headline": headline,
            "description": desc,
            "url": f"{base}/{pagename}.html",
            "author": {"@type": "Organization", "name": "skfolio developers"},
            "datePublished": example_date,
            "dateModified": example_date,
        }

    # Inject into context if schema defined
    if schema:
        print(f"Inject Schema into {pagename}")
        script = json.dumps(schema, indent=2)
        # Append inline script directly to metatags
        context.setdefault("metatags", "")
        context["metatags"] += (
            '\n<script type="application/ld+json">\n' + script + "\n</script>\n"
        )


def override_html_title(app, pagename, templatename, context, doctree):
    # only on the main index
    if pagename == "index":
        print("Running Override HTML Title...")
        context["title"] = "Portfolio Optimization in Python"


def override_example_meta_descriptions(app, exception):
    if exception:
        return

    print("Running meta description override (examples only)...")
    output_dir = Path(app.outdir)

    for html_file in output_dir.rglob("*.html"):
        pagename = html_file.relative_to(output_dir).with_suffix("").as_posix()
        if (
            not (
                pagename.startswith("auto_examples/")
                and pagename != "auto_examples/index"
            )
            or pagename in REDIRECTS
        ):
            continue

        headline, desc = get_example_headline_and_description(app, pagename)

        html = html_file.read_text(encoding="utf-8")

        # Replace or insert <meta name="description">
        html = re.sub(
            r'<meta\s+name=["\']description["\']\s+content=["\'].*?["\']\s*/?>',
            f'<meta name="description" content="{escape(desc)}">',
            html,
            flags=re.IGNORECASE,
        )

        # Replace or insert <meta property="og:description">
        html = re.sub(
            r'<meta\s+property=["\']og:description["\']\s+content=["\'].*?["\']\s*/?>',
            f'<meta property="og:description" content="{escape(desc)}">',
            html,
            flags=re.IGNORECASE,
        )

        html_file.write_text(html, encoding="utf-8")
        print(f"Updated: {html_file.relative_to(output_dir)}")


def replace_index_links(app, exception):
    """
    Normalize homepage links in the built HTML.

    After a successful Sphinx build, this hook scans the output directory and
    rewrites any anchor that points to the homepage via an explicit file URL
    (e.g., `href="index.html"` or `href="../index.html"`) to use a cleaner
    directory-style URL for the homepage.

    Specifically, it replaces those patterns with `href="/"`, which helps:
      - keep internal links consistent with SEO best practices,
      - avoid duplicate homepage URLs (`/` vs `/index.html`),
    """
    if exception:
        return

    # Match either href="index.html" or href="../index.html"
    pattern = re.compile(r'href="(?:\.\./)?index\.html"')

    for root, _, files in os.walk(app.outdir):
        for fname in files:
            if not fname.endswith(".html"):
                continue
            path = os.path.join(root, fname)
            text = open(path, encoding="utf-8").read()
            # Replace any matched href with an absolute root link
            new_text = pattern.sub('href="/"', text)
            if new_text != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)


# Accessible + bot-friendly template: meta refresh + canonical + JS + link
REDIRECT_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Redirecting…</title>
<link rel="canonical" href="${canonical}">
<meta http-equiv="refresh" content="0;url=${to_uri}">
<p>If you are not redirected, <a href="${to_uri}">click here</a>.</p>
<script>
  (function() {
    var target = "${to_uri}";
    if (window.location.hash) target += window.location.hash;
    window.location.replace(target);
  })();
</script>
"""


def _canonical(app, target: str) -> str:
    """Simple canonical: if html_baseurl is set and target starts with '/', join them; otherwise use target."""
    base = (getattr(app.config, "html_baseurl", "") or "").rstrip("/")
    if base and target.startswith("/"):
        return base + target
    return target


def create_redirects(app, exception):
    if exception:
        return  # skip on failed builds

    outdir = Path(app.outdir)
    suffix = getattr(app.builder, "out_suffix", ".html")  # default HTML builder
    for src_docname, target in REDIRECTS.items():
        out_path = outdir / f"{src_docname}{suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = Template(REDIRECT_HTML).substitute(
            to_uri=target,
            canonical=_canonical(app, target),
        )
        out_path.write_text(html, encoding="utf-8")
        print(f"[redirects] {src_docname}{suffix} → {target}")


def setup(app):
    """Setup function to register the build-finished hook."""
    # html page context
    app.connect("html-page-context", override_canonical)
    app.connect("html-page-context", inject_schema)
    app.connect("html-page-context", override_html_title)

    # Build finished
    app.connect("build-finished", patch_jupyterlite_notebooks)
    app.connect("build-finished", prune_and_fix_sitemap)
    app.connect("build-finished", override_example_meta_descriptions)
    app.connect("build-finished", replace_index_links)
    app.connect("build-finished", create_redirects)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
