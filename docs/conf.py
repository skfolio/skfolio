"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup ------------------------------------------------------------------------
from __future__ import annotations

import enum
import functools
import importlib
import inspect
import json
import logging
import os
import pkgutil
import queue
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urlparse

import plotly.io as pio
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx.errors import SphinxError

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
        "Evaluating portfolio models with Monte Carlo-style Multiple Randomized CV"
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
        "Building robust portfolios with uncertainty sets"
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
    "auto_examples/mean_risk/plot_17_failure_and_fallbacks": (
        "Handling optimization failures with fallback estimators and diagnostics"
    ),
    # Factor Models
    "auto_examples/factor_models/plot_characteristics_factor_model": (
        "Building and evaluating a characteristics-based cross-sectional factor model"
    ),
    "auto_examples/factor_models/plot_factor_constrained_portfolio": (
        "Optimizing and attributing a factor-constrained long-short portfolio"
    ),
    "auto_examples/factor_models/plot_alpha_factor_neutral_portfolio": (
        "Researching idiosyncratic alpha in a factor-neutral long-short portfolio"
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
        "Modeling pairwise asset dependence with bivariate copulas"
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
        "Combining multiple portfolio strategies through stacking optimization"
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
    "auto_examples/clustering/plot_6_schur": (
        "Interpolating between hierarchical risk parity and minimum variance "
        "with Schur allocation"
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
    # Online Learning
    "auto_examples/online_learning/plot_1_online_covariance_forecast_evaluation": (
        "Evaluating online covariance forecasts with walk-forward calibration diagnostics"
    ),
    "auto_examples/online_learning/plot_2_online_hyperparameter_tuning": (
        "Tuning online covariance estimators with grid and randomized search"
    ),
    "auto_examples/online_learning/plot_3_online_portfolio_optimization_evaluation": (
        "Online evaluation of portfolio optimization"
    ),
}

USER_GUIDE_DESCRIPTIONS = {
    "user_guide/cluster": (
        "The clustering module provides hierarchical clustering from asset distance "
        "matrices for skfolio's hierarchical portfolio optimizers."
    ),
    "user_guide/covariance": (
        "Covariance estimators compute covariance matrices used by skfolio optimizers."
    ),
    "user_guide/cross_sectional_transformers": (
        "Cross-sectional transformers normalize each observation across assets using "
        "z-score, percentile-rank, or Gaussian-rank scaling."
    ),
    "user_guide/data_preparation": (
        "Linear and logarithmic returns aggregate differently across assets and time, "
        "affecting portfolio return and risk calculations."
    ),
    "user_guide/data_representation": (
        "skfolio uses wide, date-by-asset data, provides the dedicated AssetPanel "
        "container, and defines conventions for missing observations, changing asset "
        "universes, and investability."
    ),
    "user_guide/datasets": (
        "skfolio provides loader functions for included financial datasets and larger "
        "datasets downloaded on demand."
    ),
    "user_guide/distance": (
        "Distance estimators compute asset codependence and distance matrices."
    ),
    "user_guide/expected_returns": (
        "Expected-return estimators provide estimates of each asset's expected return "
        "for portfolio optimization."
    ),
    "user_guide/factor_models": (
        "This guide explains different factor models and focuses on skfolio's "
        "characteristics-based cross-sectional factor model."
    ),
    "user_guide/hyper_parameters_tuning": (
        "skfolio estimators can be tuned with scikit-learn-compatible hyperparameter "
        "search."
    ),
    "user_guide/install": (
        "skfolio can be installed from PyPI with pip or from conda-forge with conda."
    ),
    "user_guide/metadata_routing": (
        "Metadata routing passes additional fit data through nested estimators to the "
        "components that consume it."
    ),
    "user_guide/migration": (
        "The migration guide documents the changes required to upgrade between major "
        "skfolio versions."
    ),
    "user_guide/model_selection": (
        "The model-selection module provides portfolio-specific cross-validation and "
        "prediction utilities compatible with scikit-learn."
    ),
    "user_guide/online_learning": (
        "Online-learning utilities update estimators incrementally through time instead "
        "of refitting them independently on each split."
    ),
    "user_guide/optimization": (
        "The optimization module provides scikit-learn-compatible estimators for "
        "constructing portfolio weights from asset returns."
    ),
    "user_guide/population": (
        "Population is a container for comparing, manipulating, and analyzing "
        "collections of portfolios."
    ),
    "user_guide/portfolio": (
        "Portfolio classes provide measures and methods for analyzing portfolio "
        "returns."
    ),
    "user_guide/pre_selection": (
        "Pre-selection transformers filter the initial asset universe before fitting "
        "a portfolio model."
    ),
    "user_guide/prior": (
        "Prior estimators provide the return distribution used by portfolio "
        "optimization models."
    ),
    "user_guide/uncertainty_set": (
        "Uncertainty-set estimators model estimation uncertainty in expected returns "
        "or covariance for worst-case portfolio optimization."
    ),
    "user_guide/variance": (
        "Variance estimators compute the variance vector of the assets."
    ),
}

EXAMPLE_DATE_PUBLISHED = {
    # Factor Models
    "auto_examples/factor_models/plot_characteristics_factor_model": "2026-07-29",
    "auto_examples/factor_models/plot_factor_constrained_portfolio": "2026-07-29",
    "auto_examples/factor_models/plot_alpha_factor_neutral_portfolio": "2026-07-29",
}

EXAMPLE_DATE_MODIFIED = {
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
    "auto_examples/mean_risk/plot_17_failure_and_fallbacks": "2026-04-21",
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
    "auto_examples/clustering/plot_6_schur": "2026-07-30",
    # Maximum Diversification
    "auto_examples/maximum_diversification/plot_1_maximum_diversification": "2023-12-18",
    # Distributionally Robust CVaR
    "auto_examples/distributionally_robust_cvar/plot_1_distributionally_robust_cvar": "2023-12-18",
    # Metadata Routing
    "auto_examples/metadata_routing/plot_1_implied_volatility": "2023-12-18",
    # Online Learning
    "auto_examples/online_learning/plot_1_online_covariance_forecast_evaluation": "2026-04-09",
    "auto_examples/online_learning/plot_2_online_hyperparameter_tuning": "2026-04-09",
    "auto_examples/online_learning/plot_3_online_portfolio_optimization_evaluation": "2026-04-09",
}


def get_example_lastmod(pagename: str) -> str | None:
    """Return the latest factual publication or modification date for an example page."""
    if not pagename.startswith("auto_examples/"):
        return None

    if pagename.endswith("/index") or pagename == "auto_examples/index":
        prefix = pagename.removesuffix("index")
        dates = [
            date
            for registry in (EXAMPLE_DATE_PUBLISHED, EXAMPLE_DATE_MODIFIED)
            for example, date in registry.items()
            if example.startswith(prefix)
        ]
        return max(dates, default=None)

    return EXAMPLE_DATE_MODIFIED.get(pagename) or EXAMPLE_DATE_PUBLISHED.get(pagename)


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
    """Return the structured-data headline and description for an example page."""
    title = get_doc_title(app, pagename)

    headline = f"Tutorial on {title}"

    end_example = (
        "using skfolio, a Python library for portfolio optimization, factor model "
        "construction, and risk management."
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
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in reduce",
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Loky workers start a new interpreter and do not inherit filterwarnings.
_reduce_warning = "ignore:invalid value encountered in reduce:RuntimeWarning"
_python_warnings = os.environ.get("PYTHONWARNINGS", "")
if _reduce_warning not in _python_warnings:
    os.environ["PYTHONWARNINGS"] = (
        f"{_python_warnings},{_reduce_warning}" if _python_warnings else _reduce_warning
    )

# -- Project information ---------------------------------------------------------------

project = "skfolio"
copyright = "2026, skfolio developers (BSD License)"  # noqa: A001
author = "Hugo Delatte"

html_title = "skfolio"

# -- General configuration -------------------------------------------------------------

_LOCAL_EXTENSION_PATH = str(Path(__file__).parent / "_extensions")
if _LOCAL_EXTENSION_PATH not in sys.path:
    sys.path.insert(0, _LOCAL_EXTENSION_PATH)

extensions = [
    "skfolio_core_web_vitals",
    "skfolio_jupyterlite",
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
    "sphinx_last_updated_by_git",
    "sphinx_llm.txt",  # llms.txt / llms-full.txt / per-page .md
    "jupyterlite_sphinx",
]

# `sphinx-llm` builds the docs a second time with the markdown builder
# (`sphinx-build -b markdown`), re-running this conf. Keep the extension list identical
# so the sequential sub-build can reuse the primary build's doctree environment.
_is_markdown_subbuild = "markdown" in sys.argv

# Fast mode checks that the documentation sources build. It skips gallery execution,
# the JupyterLite site build and the sphinx-llm Markdown sub-build, which the
# deployment workflow runs in full. Used by the CI docs job.
_fast_docs_build = os.environ.get("SKFOLIO_DOCS_FAST") == "1"
if _fast_docs_build:
    for _extension in ("skfolio_jupyterlite", "sphinx_llm.txt", "jupyterlite_sphinx"):
        extensions.remove(_extension)

templates_path = ["_templates"]

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


def _numpydoc_show_inherited_class_members() -> dict[str, bool]:
    """Hide inherited members in numpydoc Methods/Attributes for AutoEnum classes.

    AutoEnum subclasses inherit public `str` methods. Numpydoc lists them in the
    class Methods autosummary, and Sphinx fails while formatting some signatures.
    Estimator classes keep the default (`True`) so inherited sklearn helpers stay
    in the summary table. Autodoc still skips `str` methods via
    `skip_str_inherited_members`.
    """
    from skfolio.utils.tools import AutoEnum

    mapping: dict[str, bool] = {}
    for module_info in pkgutil.walk_packages(skfolio.__path__, skfolio.__name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for attr_name, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            try:
                is_autoenum = issubclass(obj, AutoEnum)
            except TypeError:
                continue
            if not is_autoenum:
                continue
            mapping[f"{obj.__module__}.{obj.__qualname__}"] = False
            mapping[f"{module.__name__}.{attr_name}"] = False
    return mapping


numpydoc_show_inherited_class_members = _numpydoc_show_inherited_class_members()

# Copy robots.txt into the HTML root
html_extra_path = ["robots.txt"]

# Last updated date format
html_last_updated_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build*", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Gallery order  --------------------------------------------------------------------
# Section and tutorial names do not include their order so that reordering does
# not change URLs. Dictionary key order controls sections and tuple order controls
# tutorials within each section.
TUTORIAL_ORDER = {
    "mean_risk": (
        "plot_1_maximum_sharpe_ratio.py",
        "plot_2_minimum_CVaR.py",
        "plot_3_efficient_frontier.py",
        "plot_4_mean_variance_cdar.py",
        "plot_5_weight_constraints.py",
        "plot_6_transaction_costs.py",
        "plot_7_management_fees.py",
        "plot_8_regularization.py",
        "plot_9_uncertainty_set.py",
        "plot_10_tracking_error.py",
        "plot_11_empirical_prior.py",
        "plot_12_black_and_litterman.py",
        "plot_13_factor_model.py",
        "plot_14_black_litterman_factor_model.py",
        "plot_15_mip_cardinality_constraints.py",
        "plot_16_mip_threshold_constraints.py",
        "plot_17_failure_and_fallbacks.py",
    ),
    "factor_models": (
        "plot_characteristics_factor_model.py",
        "plot_factor_constrained_portfolio.py",
        "plot_alpha_factor_neutral_portfolio.py",
    ),
    "risk_budgeting": (
        "plot_1_risk_parity_variance.py",
        "plot_2_risk_budgeting_CVaR.py",
        "plot_3_risk_parity_ledoit_wolf.py",
    ),
    "synthetic_data": (
        "plot_1_bivariate_copulas.py",
        "plot_2_vine_copula.py",
        "plot_3_min_CVaR_stressed_factors.py",
    ),
    "entropy_pooling": (
        "plot_1_entropy_pooling.py",
        "plot_2_opinion_pooling.py",
    ),
    "clustering": (
        "plot_1_hrp_cvar.py",
        "plot_2_herc_cdar.py",
        "plot_3_hrp_vs_herc.py",
        "plot_4_nco.py",
        "plot_5_nco_grid_search.py",
        "plot_6_schur.py",
    ),
    "maximum_diversification": ("plot_1_maximum_diversification.py",),
    "distributionally_robust_cvar": ("plot_1_distributionally_robust_cvar.py",),
    "ensemble": ("plot_1_stacking.py",),
    "model_selection": ("plot_1_multiple_randomized_cv.py",),
    "online_learning": (
        "plot_1_online_covariance_forecast_evaluation.py",
        "plot_2_online_hyperparameter_tuning.py",
        "plot_3_online_portfolio_optimization_evaluation.py",
    ),
    "pre_selection": (
        "plot_1_drop_correlated.py",
        "plot_2_select_best_performers.py",
        "plot_3_custom_pre_selection_volumes.py",
        "plot_4_incomplete_dataset.py",
    ),
    "metadata_routing": ("plot_1_implied_volatility.py",),
    "data_preparation": ("plot_1_investment_horizon.py",),
}


def validate_tutorial_order() -> None:
    """Validate that the tutorial registry matches the gallery source tree."""
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    gallery_sections = {
        path.name
        for path in examples_dir.iterdir()
        if path.is_dir() and (path / "README.txt").is_file()
    }
    configured_sections = set(TUTORIAL_ORDER)

    missing_sections = gallery_sections - configured_sections
    unknown_sections = configured_sections - gallery_sections
    if missing_sections or unknown_sections:
        raise SphinxError(
            "Tutorial order sections do not match gallery sections. "
            f"Missing sections: {sorted(missing_sections)}. "
            f"Unknown sections: {sorted(unknown_sections)}."
        )

    for section_name, configured_files in TUTORIAL_ORDER.items():
        duplicate_files = sorted(
            {
                filename
                for filename in configured_files
                if configured_files.count(filename) > 1
            }
        )
        actual_files = {
            path.name for path in (examples_dir / section_name).glob("plot_*.py")
        }
        missing_files = actual_files - set(configured_files)
        unknown_files = set(configured_files) - actual_files
        if duplicate_files or missing_files or unknown_files:
            raise SphinxError(
                f"Tutorial order for section {section_name!r} does not match its "
                f"gallery files. Duplicate files: {duplicate_files}. "
                f"Missing files: {sorted(missing_files)}. "
                f"Unknown files: {sorted(unknown_files)}."
            )


validate_tutorial_order()

# -- sphinxext-opengraph ---------------------------------------------------------------

ogp_site_url = "https://skfolio.org/"
ogp_site_name = "skfolio"
ogp_image = "https://skfolio.org/_static/expo.jpg"
ogp_enable_meta_description = True
ogp_description_length = 160

# -- sphinx_last_updated_by_git  -------------------------------------------------------

git_untracked_check_dependencies = True

# -- autosummary -----------------------------------------------------------------------

autosummary_generate = True

# -- sphinx_sitemap --------------------------------------------------------------------
html_baseurl = "https://skfolio.org/"
sitemap_url_scheme = "{link}"
sitemap_show_lastmod = True
sitemap_excludes = ["search.html"]

# -- sphinx-llm ------------------------------------------------------------------------
# https://github.com/NVIDIA/sphinx-llm
# Builds the docs a second time with the markdown builder and writes, into the HTML
# build dir: a `<page>.html.md` markdown copy of every page, plus `llms-full.txt` (all
# pages concatenated) and `llms.txt` (a markdown sitemap with per-page descriptions),
# following the https://llmstxt.org/ convention. Since these land in the HTML output dir
# they get published with the site, e.g. at https://skfolio.org/llms.txt ,
# https://skfolio.org/llms-full.txt and https://skfolio.org/<page>.html.md
llms_txt_description = (
    "Python library for portfolio optimization, factor model construction, and risk "
    "management, built on top of scikit-learn: create, fine-tune, cross-validate, "
    "and stress-test portfolio models."
)

# make the links in llms.txt absolute
markdown_http_base = "https://skfolio.org"

# sphinx-markdown-builder defaults its cross-reference suffix to ".md", but sphinx-llm
# writes per-page files as "<page>.html.md" to follow llmstxt.org's "append .md to any
# HTML URL" convention. Override the URI suffix so internal references inside the
# generated markdown resolve to the files that actually ship on the site.
markdown_uri_doc_suffix = ".html.md"

# Preserve Sphinx targets so cross-references to sections, modules, classes, methods,
# functions and properties keep working in the generated Markdown pages.
markdown_anchor_sections = True
markdown_anchor_signatures = True

# Run the Markdown sub-build after the HTML build so it consumes completed gallery
# sources and avoids concurrent writes to gallery and JupyterLite state. JupyterLite
# remains loaded for doctree compatibility, but its Markdown cleanup hook is disabled.
llms_txt_build_parallel = False

# -- Internationalization --------------------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- MyST options ----------------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- sphinx-favicons -------------------------------------------------------------------
favicons = [
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
        "sizes": "96x96",
        "href": "favicon-96.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "144x144",
        "href": "favicon-144.png",
    },
    {
        "rel": "shortcut icon",
        "type": "image/x-icon",
        "href": "favicon.ico",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "apple-touch-icon.png",
    },
]

# -- Options for HTML output -----------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_sourcelink_suffix = ""

# Inferred from the installed package and surfaced in the page metadata.
release = skfolio.__version__

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
    "show_toc_level": 2,
    "navbar_align": (
        "left"
    ),  # [left, content, right] For testing that the navbar items align properly
    "secondary_sidebar_items": {
        "**": [],
        "user_guide/*": ["page-toc"],
        "user_guide/*/*": ["page-toc"],
    },
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

# -- gallery  --------------------------------------------------------------------------

image_scrapers = (
    "matplotlib",
    plotly_sg_scraper,
)


class TutorialOrderSortKey:
    """Sort tutorials according to the explicit registry.

    Parameters
    ----------
    src_dir : str
        The source directory.
    """

    def __init__(self, src_dir: str) -> None:
        self.section_name = Path(src_dir).name
        self.positions = {
            filename: position
            for position, filename in enumerate(
                TUTORIAL_ORDER.get(self.section_name, ())
            )
        }

    def __call__(self, filename: str) -> int:
        """Return the configured position of a tutorial."""
        filename = Path(filename).name
        try:
            return self.positions[filename]
        except KeyError as error:
            raise SphinxError(
                f"Tutorial {filename!r} is not registered in the explicit order "
                f"for section {self.section_name!r}."
            ) from error


def custom_section_order(section_name) -> int:
    """Return the configured position of a gallery section."""
    return tuple(TUTORIAL_ORDER).index(Path(section_name).name)


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
    "within_subsection_order": TutorialOrderSortKey,
    "image_scrapers": image_scrapers,
    # avoid generating too many cross links
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
    "jupyterlite": {
        "jupyterlite_contents": "_contents",
        "notebook_modification_function": (
            "skfolio_jupyterlite.modify_jupyterlite_notebook"
        ),
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

# The HTML build owns tutorial execution. The sequential sphinx-llm Markdown build
# consumes the generated gallery sources without executing the examples a second time.
if _is_markdown_subbuild:
    plot_gallery = "False"

# Fast mode generates the gallery pages without executing the examples and without
# the JupyterLite integration, whose extensions are not loaded.
if _fast_docs_build:
    plot_gallery = "False"
    sphinx_gallery_conf["jupyterlite"] = None

# -- jupyterlite  ----------------------------------------------------------------------
# Read more at https://jupyterlite-sphinx.readthedocs.io/en/latest/configuration.html#configuration

# Build the Lite site in the documentation output directory and publish the gallery
# notebooks directly from one canonical content root.
jupyterlite_dir = str(Path(__file__).parent.absolute())
jupyterlite_content_dir = "_contents"


# -- Sphinx Hooks ----------------------------------------------------------------------


def _html_builders_only(handler):
    """No-op a `build-finished` handler unless the active builder emits HTML.

    The `sphinx-llm` extension runs a second `sphinx-build -b markdown` pass through
    this conf. The handlers below manipulate (or assume the existence of) HTML build
    output and would crash or be pointless under another builder.
    """

    @functools.wraps(handler)
    def wrapper(app, exception):
        if app.builder.name not in ("html", "dirhtml"):
            return
        return handler(app, exception)

    return wrapper


def _disable_jupyterlite_markdown_cleanup(app):
    """Disable JupyterLite's build-finished handler in the Markdown sub-build.

    The handler skips the JupyterLite build for non-HTML builders but still removes
    shared JupyterLite state. Keeping the extension loaded preserves doctree cache
    compatibility with the primary build; disconnecting this handler prevents the
    Markdown pass from modifying that state.
    """
    if not _is_markdown_subbuild:
        return

    from jupyterlite_sphinx.jupyterlite_sphinx import jupyterlite_build

    for listener in tuple(app.events.listeners["build-finished"]):
        if listener.handler is jupyterlite_build:
            app.disconnect(listener.id)
            return

    warnings.warn(
        "Unable to disable the JupyterLite Markdown build-finished handler.",
        RuntimeWarning,
        stacklevel=2,
    )


def patch_markdown_builder(app):
    """Configure the Markdown builder for the documentation structure.

    Numpydoc emits the type for each parameter as a docutils `classifier` node.
    `sphinx-markdown-builder` has no `visit_classifier`, so the type is silently dropped
    in the markdown output (e.g. `returns : ndarray of shape (n,)` becomes just
    `returns`). Docutils and Sphinx-Gallery also emit nodes that the Markdown translator
    does not support. Register visitors that preserve their content or intentionally
    omit metadata.

    The generic Markdown builder also treats HTML static and extra paths as source
    documents. Excluding those asset trees matches the HTML builder and prevents
    static reStructuredText fragments from entering the LLM artifacts.
    """
    if app.builder.name != "markdown":
        return
    from docutils import nodes
    from sphinx_markdown_builder.builder import MarkdownBuilder
    from sphinx_markdown_builder.translator import MarkdownTranslator

    def get_asset_paths(_builder):
        return [*html_extra_path, *html_static_path]

    MarkdownBuilder.get_asset_paths = get_asset_paths

    def visit_classifier(self, _node):
        self.add(" *")

    def depart_classifier(self, _node):
        self.add("*")

    def visit_meta(self, _node):
        raise nodes.SkipNode

    def visit_abbreviation(self, _node):
        return

    def depart_abbreviation(self, _node):
        return

    def visit_citation(self, node):
        self.visit_footnote(node)

    def depart_citation(self, node):
        self._pop_context(node)

    def visit_imgsgnode(self, node):
        self.visit_image(node)

    def depart_imgsgnode(self, _node):
        return

    def make_admonition_visitor(title):
        def visit_admonition(self, _node):
            self._push_box(title)

        return visit_admonition

    def depart_admonition(self, node):
        self._pop_context(node)

    MarkdownTranslator.visit_classifier = visit_classifier
    MarkdownTranslator.depart_classifier = depart_classifier
    MarkdownTranslator.visit_meta = visit_meta
    MarkdownTranslator.visit_abbreviation = visit_abbreviation
    MarkdownTranslator.depart_abbreviation = depart_abbreviation
    MarkdownTranslator.visit_citation = visit_citation
    MarkdownTranslator.depart_citation = depart_citation
    MarkdownTranslator.visit_imgsgnode = visit_imgsgnode
    MarkdownTranslator.depart_imgsgnode = depart_imgsgnode
    for node_name, title in {
        "caution": "CAUTION",
        "danger": "DANGER",
        "error": "ERROR",
        "tip": "TIP",
    }.items():
        setattr(
            MarkdownTranslator,
            f"visit_{node_name}",
            make_admonition_visitor(title),
        )
        setattr(MarkdownTranslator, f"depart_{node_name}", depart_admonition)


@_html_builders_only
def populate_complete_sitemap(app, exception):
    """Populate sphinx-sitemap with every discovered document.

    sphinx-sitemap normally records only pages written during the current invocation,
    which truncates the sitemap during incremental builds. Rebuild its queue from the
    cached Sphinx environment before the extension writes `sitemap.xml`.
    """
    if exception is not None:
        return

    from sphinx_sitemap import add_html_link

    app.sitemap_links = queue.Queue()
    for docname in sorted(app.env.found_docs - REDIRECTS.keys()):
        add_html_link(app, docname, None, None, None)


@_html_builders_only
def prune_and_fix_sitemap(app, exception):
    """Remove non-document pages and normalize sitemap metadata."""
    if exception:
        warnings.warn(
            f"Sitemap hook: skipping because build failed ({exception!r})", stacklevel=2
        )
        return

    sitemap_path = Path(app.outdir) / app.config.sitemap_filename

    if not sitemap_path.exists():
        warnings.warn(
            f"Sitemap hook: '{sitemap_path}' not found, skipping", stacklevel=2
        )
        return

    try:
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

            # Drop viewcode pages.
            if path.startswith("/_modules/"):
                root.remove(url)
                removed += 1
                continue

            # rewrite only the root index.html → /
            if path == "/index.html":
                loc.text = app.config.html_baseurl.rstrip("/") + "/"

                priority_el = url.find("sm:priority", ns)
                if priority_el is None:
                    priority_el = ET.SubElement(url, f"{{{ns['sm']}}}priority")
                priority_el.text = "1.0"

            # Use factual example publication and modification dates.
            lastmod = get_example_lastmod(path.lstrip("/").removesuffix(".html"))
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

        ET.register_namespace("", ns["sm"])

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
    """Use the canonical root URL for homepage metadata."""
    if app.config.html_baseurl and pagename == "index":
        context["pageurl"] = app.config.html_baseurl.rstrip("/") + "/"


def get_doc_title(app, pagename) -> str:
    """Retrieve the title text for a given docname from the Sphinx environment."""
    title_node = app.env.titles.get(pagename)
    if title_node:
        return title_node.astext()

    raise ValueError(f"Failed to retrieve title from {pagename}")


_API_DESCRIPTION_MAX_LENGTH = 160
_DESCRIPTION_META_RE = re.compile(
    r"""<meta\b"""
    r"""(?=[^>]*\b(?:name|property)\s*=\s*["']"""
    r"""(?:description|og:description|twitter:description)["'])"""
    r"""[^>]*>\s*""",
    flags=re.IGNORECASE,
)
_OG_TITLE_META_RE = re.compile(
    r"""<meta\b(?=[^>]*\bproperty\s*=\s*["']og:title["'])[^>]*>\s*""",
    flags=re.IGNORECASE,
)
_OG_IMAGE_META_RE = re.compile(
    r"""<meta\b(?=[^>]*\bproperty\s*=\s*["']og:image(?::alt)?["'])[^>]*>\s*""",
    flags=re.IGNORECASE,
)
_FACTOR_MODELS_IMAGE = (
    "_static/factor_model/plots/factor_model_exposure_correlation.webp"
)


def _set_description_meta(context, description: str) -> None:
    """Replace standard and social description metadata in a page context."""
    escaped_description = escape(description, quote=True)
    metatags = _DESCRIPTION_META_RE.sub("", context.get("metatags", "")).rstrip()
    context["metatags"] = (
        f"{metatags}\n"
        f'<meta name="description" content="{escaped_description}" />\n'
        f'<meta property="og:description" content="{escaped_description}" />\n'
        f'<meta name="twitter:description" content="{escaped_description}" />\n'
    )


def _set_og_title(context, title: str) -> None:
    """Replace Open Graph title metadata in a page context."""
    escaped_title = escape(title, quote=True)
    metatags = _OG_TITLE_META_RE.sub("", context.get("metatags", "")).rstrip()
    context["metatags"] = (
        f'{metatags}\n<meta property="og:title" content="{escaped_title}" />\n'
    )


def _truncate_at_word_boundary(text: str, max_length: int) -> str:
    """Truncate text at a word boundary and append an ellipsis."""
    if len(text) <= max_length:
        return text

    truncated = text[: max_length - 1].rsplit(" ", maxsplit=1)[0]
    truncated = truncated.rstrip(" ,.;:-")
    if not truncated:
        truncated = text[: max_length - 1].rstrip()
    return f"{truncated}…"


def get_api_meta_description(pagename, doctree) -> str:
    """Build an API description from the documented object's summary."""
    from docutils import nodes
    from sphinx import addnodes

    qualified_name = pagename.removeprefix("generated/")
    content = next(doctree.findall(addnodes.desc_content), None)
    summary_node = (
        next(
            (
                child
                for child in content.children
                if isinstance(child, (nodes.paragraph, nodes.definition_list))
            ),
            None,
        )
        if content is not None
        else None
    )
    if summary_node is None:
        raise SphinxError(
            f"API description: no summary paragraph found for '{pagename}'"
        )

    summary = " ".join(summary_node.astext().split())
    if not summary:
        raise SphinxError(f"API description: empty summary found for '{pagename}'")
    if summary[-1] not in ".!?":
        summary += "."

    suffix = f"API reference for {qualified_name}."
    summary = _truncate_at_word_boundary(
        summary,
        _API_DESCRIPTION_MAX_LENGTH - len(suffix) - 1,
    )
    return f"{summary} {suffix}"


def get_representative_image(app, pagename: str) -> tuple[str, str] | None:
    """Return a representative image URL and alternative text for selected pages."""
    base = app.config.html_baseurl.rstrip("/")
    if pagename == "user_guide/factor_models":
        return (
            f"{base}/{_FACTOR_MODELS_IMAGE}",
            "Correlation heatmap of market and style factor exposures",
        )

    if pagename in REDIRECTS or not pagename.startswith("auto_examples/"):
        return None

    parts = pagename.split("/")
    if pagename.endswith("/index") and len(parts) == 3:
        tutorials = TUTORIAL_ORDER.get(parts[1])
        if not tutorials:
            return None
        stem = Path(tutorials[0]).stem
        title = get_doc_title(app, pagename)
        return (
            f"{base}/_images/sphx_glr_{stem}_thumb.png",
            f"{title} examples",
        )

    if pagename != "auto_examples/index" and not pagename.endswith("/index"):
        stem = parts[-1]
        title = get_doc_title(app, pagename)
        return (
            f"{base}/_images/sphx_glr_{stem}_thumb.png",
            f"{title} tutorial visualization",
        )

    return None


def inject_representative_image_meta(
    app, pagename, templatename, context, doctree
) -> None:
    """Use representative Open Graph images where relevant assets exist."""
    representative_image = get_representative_image(app, pagename)
    if representative_image is None:
        return

    image_url, image_alt = representative_image
    metatags = _OG_IMAGE_META_RE.sub("", context.get("metatags", "")).rstrip()
    context["metatags"] = (
        f"{metatags}\n"
        f'<meta property="og:image" content="{escape(image_url, quote=True)}" />\n'
        f'<meta property="og:image:alt" content="{escape(image_alt, quote=True)}" />\n'
    )


def inject_api_meta_description(app, pagename, templatename, context, doctree):
    """Add standard and social descriptions to generated API pages."""
    if not pagename.startswith("generated/skfolio.") or doctree is None:
        return

    _set_description_meta(context, get_api_meta_description(pagename, doctree))


def inject_example_meta_description(app, pagename, templatename, context, doctree):
    """Add standard and social descriptions to gallery categories and examples."""
    if (
        not pagename.startswith("auto_examples/")
        or pagename == "auto_examples/index"
        or pagename in REDIRECTS
    ):
        return

    if pagename.endswith("/index"):
        title = get_doc_title(app, pagename)
        description = (
            f"Examples and tutorials for {title} using skfolio, a Python library "
            "for portfolio optimization, factor model construction, and risk "
            "management."
        )
    else:
        _, description = get_example_headline_and_description(app, pagename)

    _set_description_meta(context, description)


def inject_noindex_meta(app, pagename, templatename, context, doctree):
    """Exclude internal search and rendered source pages from search results."""
    if pagename != "search" and not pagename.startswith("_modules/"):
        return

    metatags = context.get("metatags", "").rstrip()
    context["metatags"] = f'{metatags}\n<meta name="robots" content="noindex" />\n'


def inject_user_guide_meta_description(app, pagename, templatename, context, doctree):
    """Use reviewed descriptions for user-guide articles."""
    description = USER_GUIDE_DESCRIPTIONS.get(pagename)
    if description is not None:
        _set_description_meta(context, description)


# Stable entity IDs shared across skfolio.org and skfoliolabs.com
ORG_ID = "https://skfoliolabs.com#organization"
CODE_ID = "https://github.com/skfolio/skfolio#code"
WEBSITE_ID = "https://skfolio.org#website"
DOCS_DATE_PUBLISHED = "2023-12-18"
_ARTICLE_AUTHOR = {
    "@type": "Organization",
    "@id": ORG_ID,
    "name": "Skfolio Labs",
    "url": "https://skfoliolabs.com",
}


def _breadcrumb(id_url: str, items: list[tuple[str, str]]) -> dict[str, object]:
    """Build a Google-compatible breadcrumb with sequential positions."""
    if len(items) < 2:
        raise ValueError("A breadcrumb requires at least two items")

    return {
        "@type": "BreadcrumbList",
        "@id": f"{id_url}#breadcrumb",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": position,
                "name": name,
                "item": url,
            }
            for position, (name, url) in enumerate(items, start=1)
        ],
    }


def inject_schema(app, pagename, templatename, context, doctree):
    """Add page-specific JSON-LD structured data."""
    base = app.config.html_baseurl.rstrip("/")
    in_lang = "en"

    # helper: always return a usable URL for this page
    def _url_for(page: str) -> str:
        return context.get("pageurl") or (
            f"{base}/" if page == "index" else f"{base}/{page}.html"
        )

    page_dates = {"datePublished": DOCS_DATE_PUBLISHED}
    if date_modified := context.get("last_updated"):
        page_dates["dateModified"] = date_modified
    representative_image = get_representative_image(app, pagename)
    image_properties = (
        {"image": representative_image[0]} if representative_image else {}
    )

    # Always initialize metatags safely
    context["metatags"] = context.get("metatags", "")

    # Homepage site entities.
    homepage_graph = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "WebSite",
                "@id": WEBSITE_ID,
                "url": f"{base}/",
                "name": "skfolio",
                "alternateName": ["skfolio documentation", "skfolio.org"],
                "inLanguage": in_lang,
                "publisher": {"@id": ORG_ID},
            },
            # Primary navigation links shown throughout the documentation.
            {
                "@type": "SiteNavigationElement",
                "@id": f"{base}/#site-nav",
                "name": "Primary navigation",
                "url": f"{base}/",
                "about": {"@id": WEBSITE_ID},
                "inLanguage": in_lang,
                "hasPart": [
                    {
                        "@type": "WebPage",
                        "name": "User Guide",
                        "url": f"{base}/user_guide/index.html",
                    },
                    {
                        "@type": "WebPage",
                        "name": "Examples",
                        "url": f"{base}/auto_examples/index.html",
                    },
                    {
                        "@type": "WebPage",
                        "name": "API Reference",
                        "url": f"{base}/api.html",
                    },
                ],
            },
            # Cross-domain entities with stable identifiers.
            {
                "@type": "Corporation",
                "@id": ORG_ID,
                "name": "Skfolio Labs",
                "url": "https://skfoliolabs.com",
                "logo": "https://skfoliolabs.com/icon.svg",
            },
            {
                "@type": "SoftwareSourceCode",
                "@id": CODE_ID,
                "name": "Skfolio Source Code",
                "url": "https://github.com/skfolio/skfolio",
                "codeRepository": "https://github.com/skfolio/skfolio",
                "programmingLanguage": "Python",
                "author": {"@id": ORG_ID},
                "publisher": {"@id": ORG_ID},
            },
        ],
    }

    if pagename == "index":
        context["metatags"] += (
            '\n<script type="application/ld+json">\n'
            + json.dumps(homepage_graph, indent=2)
            + "\n</script>\n"
        )

    # Page-level entities.
    page_schema = None
    url = _url_for(pagename)

    # Docs home (/) as CollectionPage with a featured ItemList
    if pagename == "index":
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "CollectionPage",
                    "@id": f"{base}/#docs-home",
                    "name": "Skfolio Documentation",
                    "url": url,
                    "inLanguage": in_lang,
                    "isPartOf": {"@id": WEBSITE_ID},
                    "about": {"@id": CODE_ID},
                    "publisher": {"@id": ORG_ID},
                    "author": _ARTICLE_AUTHOR,
                    "copyrightHolder": {"@id": ORG_ID},
                    "mainEntityOfPage": url,
                    **page_dates,
                    "primaryImageOfPage": {
                        "@type": "ImageObject",
                        "url": f"{base}/_static/expo.jpg",
                    },
                },
                {
                    "@type": "ItemList",
                    "@id": f"{base}/#featured-sections",
                    "itemListElement": [
                        {
                            "@type": "ListItem",
                            "position": 1,
                            "url": f"{base}/user_guide/index.html",
                        },
                        {
                            "@type": "ListItem",
                            "position": 2,
                            "url": f"{base}/auto_examples/index.html",
                        },
                        {"@type": "ListItem", "position": 3, "url": f"{base}/api.html"},
                    ],
                },
            ],
        }

    # User Guide index as CollectionPage + ItemList of child pages + breadcrumb
    elif pagename == "user_guide/index":
        all_steps = sorted(
            [
                doc
                for doc in app.env.found_docs
                if doc.startswith("user_guide/") and doc != "user_guide/index"
            ]
        )
        step_items = [
            {"@type": "ListItem", "position": i, "url": f"{base}/{doc}.html"}
            for i, doc in enumerate(all_steps, start=1)
        ]

        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "CollectionPage",
                    "@id": f"{url}#guide-home",
                    "name": "skfolio User Guide",
                    "description": (
                        "Comprehensive guide to installing, configuring, and using "
                        "skfolio for portfolio optimization, factor model construction, "
                        "and risk management."
                    ),
                    "url": url,
                    "inLanguage": in_lang,
                    "isPartOf": {"@id": WEBSITE_ID},
                    "about": {"@id": CODE_ID},
                    "publisher": {"@id": ORG_ID},
                    "author": _ARTICLE_AUTHOR,
                    "copyrightHolder": {"@id": ORG_ID},
                    "mainEntityOfPage": url,
                    **page_dates,
                },
                {
                    "@type": "ItemList",
                    "@id": f"{url}#guide-list",
                    "itemListElement": step_items,
                },
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("User Guide", url),
                    ],
                ),
            ],
        }

    # Individual User Guide pages -> TechArticle + breadcrumb
    elif pagename.startswith("user_guide/") and pagename != "user_guide/index":
        title = get_doc_title(app, pagename)
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "TechArticle",
                    "@id": f"{url}#article",
                    "headline": title,
                    "url": url,
                    "inLanguage": in_lang,
                    "isPartOf": {"@id": WEBSITE_ID},
                    "about": {"@id": CODE_ID},
                    "publisher": {"@id": ORG_ID},
                    "author": _ARTICLE_AUTHOR,
                    "copyrightHolder": {"@id": ORG_ID},
                    "mainEntityOfPage": url,
                    **page_dates,
                    **image_properties,
                },
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("User Guide", f"{base}/user_guide/index.html"),
                        (title, url),
                    ],
                ),
            ],
        }

    # API Reference
    elif pagename == "api":
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "APIReference",
                    "@id": f"{url}#article",
                    "headline": "skfolio API Reference",
                    "description": (
                        "Complete reference for skfolio's portfolio optimization, "
                        "factor model construction, and risk management API: functions, "
                        "classes, and modules."
                    ),
                    "url": url,
                    "version": app.config.release,
                    "programmingModel": "Python",
                    "targetPlatform": "Any platform running Python 3.10+",
                    "inLanguage": in_lang,
                    "isPartOf": {"@id": WEBSITE_ID},
                    "about": {"@id": CODE_ID},
                    "publisher": {"@id": ORG_ID},
                    "author": _ARTICLE_AUTHOR,
                    "copyrightHolder": {"@id": ORG_ID},
                    "mainEntityOfPage": url,
                    **page_dates,
                },
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("API Reference", url),
                    ],
                ),
            ],
        }

    # Generated API object pages
    elif pagename.startswith("generated/skfolio."):
        title = get_doc_title(app, pagename)
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("API Reference", f"{base}/api.html"),
                        (title, url),
                    ],
                ),
            ],
        }

    # Examples index as a collection page with Google-supported breadcrumbs.
    elif pagename == "auto_examples/index":
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "CollectionPage",
                    "@id": f"{url}#examples",
                    "name": "Code Examples & Tutorials",
                    "description": (
                        "Code examples and tutorials for portfolio optimization, risk "
                        "management, and factor models with skfolio."
                    ),
                    "url": url,
                    "inLanguage": in_lang,
                    "isPartOf": {"@id": WEBSITE_ID},
                    "about": {"@id": CODE_ID},
                    "publisher": {"@id": ORG_ID},
                    **page_dates,
                },
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("Examples", url),
                    ],
                ),
            ],
        }

    # Gallery category indexes
    elif pagename.startswith("auto_examples/") and pagename.endswith("/index"):
        title = get_doc_title(app, pagename)
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("Examples", f"{base}/auto_examples/index.html"),
                        (title, url),
                    ],
                ),
            ],
        }

    # Individual example pages as TechArticle + breadcrumb (+ provenance)
    elif pagename.startswith("auto_examples/") and not pagename.endswith("index"):
        headline, desc = get_example_headline_and_description(app, pagename)
        example_dates = {}
        if example_date_published := EXAMPLE_DATE_PUBLISHED.get(pagename):
            example_dates["datePublished"] = example_date_published
        if example_date_modified := EXAMPLE_DATE_MODIFIED.get(pagename):
            example_dates["dateModified"] = example_date_modified
        page_schema = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "TechArticle",
                    "@id": f"{url}#article",
                    "headline": headline,
                    "description": desc,
                    "url": url,
                    "inLanguage": in_lang,
                    "isPartOf": {"@id": WEBSITE_ID},
                    "about": {"@id": CODE_ID},
                    "publisher": {"@id": ORG_ID},
                    "author": _ARTICLE_AUTHOR,
                    "copyrightHolder": {"@id": ORG_ID},
                    "mainEntityOfPage": url,
                    "isBasedOn": {"@id": CODE_ID},
                    **example_dates,
                    **image_properties,
                },
                _breadcrumb(
                    url,
                    [
                        ("Docs Home", f"{base}/"),
                        ("Examples", f"{base}/auto_examples/index.html"),
                        (headline, url),
                    ],
                ),
            ],
        }

    # Inject page-level block when defined
    if page_schema:
        context["metatags"] += (
            '\n<script type="application/ld+json">\n'
            + json.dumps(page_schema, indent=2)
            + "\n</script>\n"
        )


def override_html_title(app, pagename, templatename, context, doctree):
    """Set search-oriented titles where the Sphinx title is insufficient."""
    if pagename == "index":
        # Keep the visible RST title as the library name, following Python
        # documentation conventions, while making browser/social titles descriptive.
        seo_title = "Portfolio Optimization in Python"
        context["title"] = seo_title
        _set_og_title(context, seo_title)
        return

    is_category_index = (
        pagename.startswith("auto_examples/")
        and pagename.endswith("/index")
        and pagename != "auto_examples/index"
        and pagename not in REDIRECTS
    )
    if not is_category_index:
        return

    seo_title = f"{get_doc_title(app, pagename)} examples"
    context["title"] = seo_title
    _set_og_title(context, seo_title)


@_html_builders_only
def replace_index_links(app, exception):
    """Normalize only links that truly point to the root homepage.

      - href="/index.html"                    -> href="/"
      - href="{html_baseurl}/index.html"      -> href="{html_baseurl}/"
      - href="../index.html", "../../index.html", ...  (only if they resolve to root)
      - href="index.html" or "./index.html"   (only from files in the root outdir)

    Do NOT touch:
      - section indexes like /auto_examples/index.html
      - ../index.html that resolve to a section index
      - links with fragments or queries (index.html#..., index.html?...).
    """
    if exception:
        return

    base = app.config.html_baseurl.rstrip("/")
    outdir = Path(app.outdir)
    root_index_abs = (outdir / "index.html").resolve()

    # 1) Absolute root link: href="/index.html"
    abs_root_pattern = re.compile(
        r'href\s*=\s*(["\'])/index\.html\1(?![#?])',
        flags=re.IGNORECASE,
    )

    # 2) Fully-qualified root link: href="{base}/index.html"
    fq_root_pattern = re.compile(
        r'href\s*=\s*(["\'])' + re.escape(base) + r"/index\.html\1(?![#?])",
        flags=re.IGNORECASE,
    )

    # 3) Relative links with one-or-more "../" segments: href="../index.html", "../../index.html", ...
    rel_up_pattern = re.compile(
        r'href\s*=\s*(["\'])(?P<prefix>(?:\.\./)+)index\.html\1(?![#?])',
        flags=re.IGNORECASE,
    )

    # 4) Plain or "./" relative link: href="index.html" or href="./index.html"
    rel_same_pattern = re.compile(
        r'href\s*=\s*(["\'])(?:\./)?index\.html\1(?![#?])',
        flags=re.IGNORECASE,
    )

    def _rel_up_repl(current_html_path: Path):
        """Return a callable that rewrites ../index.html → / only if it resolves to root index."""
        current_dir = current_html_path.parent

        def _repl(m: re.Match) -> str:
            quote = m.group(1)
            prefix = m.group("prefix")  # e.g., "../" or "../../"
            resolved = (current_dir / prefix / "index.html").resolve()
            if resolved == root_index_abs:
                return f"href={quote}/{quote}"
            # Not the root index → leave untouched
            return m.group(0)

        return _repl

    for path in outdir.rglob("*.html"):
        text = path.read_text(encoding="utf-8")

        # Absolute / fully-qualified root → "/"
        new_text = abs_root_pattern.sub(r'href="/"', text)
        new_text = fq_root_pattern.sub(lambda m: f'href="{base}/"', new_text)

        # ../index.html (or deeper) → resolve; rewrite only if it maps to root index
        new_text = rel_up_pattern.sub(_rel_up_repl(path), new_text)

        # "index.html" or "./index.html" → rewrite only if *this file* lives in outdir root
        if path.parent == outdir:
            new_text = rel_same_pattern.sub(r'href="/"', new_text)

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


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


@_html_builders_only
def create_redirects(app, exception):
    """Write redirect pages for legacy documentation URLs."""
    if exception:
        return  # skip on failed builds

    outdir = Path(app.outdir)
    suffix = getattr(app.builder, "out_suffix", ".html")  # default HTML builder
    updated_redirects = 0
    for src_docname, target in REDIRECTS.items():
        out_path = outdir / f"{src_docname}{suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = Template(REDIRECT_HTML).substitute(
            to_uri=target,
            canonical=_canonical(app, target),
        )
        if not out_path.exists() or out_path.read_text(encoding="utf-8") != html:
            out_path.write_text(html, encoding="utf-8")
            updated_redirects += 1

    if updated_redirects:
        print(f"Updated {updated_redirects} redirect page(s)")


_HEAD_RE = re.compile(r"<head\b[^>]*>.*?</head>", flags=re.IGNORECASE | re.DOTALL)
_VIEWPORT_META_RE = re.compile(
    r"""<meta\b(?=[^>]*\bname\s*=\s*["']viewport["'])[^>]*>""",
    flags=re.IGNORECASE,
)
_CANONICAL_VIEWPORT_META = (
    '<meta name="viewport" content="width=device-width, initial-scale=1" />'
)


@_html_builders_only
def normalize_viewport_meta(app, exception):
    """Keep one canonical viewport declaration in each generated HTML head."""
    if exception:
        return

    updated_pages = 0
    for path in Path(app.outdir).rglob("*.html"):
        original = path.read_text(encoding="utf-8")
        head_match = _HEAD_RE.search(original)
        if head_match is None:
            continue

        head = head_match.group()
        if len(_VIEWPORT_META_RE.findall(head)) <= 1:
            continue

        replacement_count = 0

        def replace_viewport(match: re.Match) -> str:
            nonlocal replacement_count
            replacement_count += 1
            return _CANONICAL_VIEWPORT_META if replacement_count == 1 else ""

        normalized_head = _VIEWPORT_META_RE.sub(replace_viewport, head)
        html = (
            original[: head_match.start()]
            + normalized_head
            + original[head_match.end() :]
        )
        path.write_text(html, encoding="utf-8")
        updated_pages += 1

    if updated_pages:
        print(f"Normalized viewport metadata in {updated_pages} page(s)")


def skip_str_inherited_members(app, what, name, obj, skip, options):
    """Skip str methods pulled in by AutoEnum via inherited-members."""
    if skip:
        return True
    if name not in vars(str):
        return False
    qualname = getattr(obj, "__qualname__", "") or ""
    if qualname.startswith("str."):
        return True
    if getattr(obj, "__objclass__", None) is str:
        return True
    if obj is vars(str).get(name):
        return True
    return False


class _SkfolioAutodocStrEnumNoise(logging.Filter):
    """Hide autodoc/autosummary warnings for inherited `str` methods on skfolio enums.

    `AutoEnum` subclasses `(str, Enum)`, so every skfolio enum inherits ~40 public `str`
    methods. Combined with `autodoc_default_options = {"inherited-members": True}` this
    produces a flood of "error while formatting signature for X" / "failed to import
    object X" warnings from `sphinx.ext.autodoc` and `sphinx.ext.autosummary`. The
    companion `skip_strenum_public_str_methods` callback prevents them from appearing
    in the rendered docs. This filter silences the matching log lines.
    """

    def __init__(self) -> None:
        super().__init__()
        tail = "|".join(
            re.escape(n)
            for n in dir(str)
            if not n.startswith("_") and callable(getattr(str, n, None))
        )
        self._pat = re.compile(
            rf"(?:error while formatting signature for |failed to import object )"
            rf"skfolio\.[\w.]+\.(?:{tail})\b"
        )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if self._pat.search(record.getMessage()):
                return False
        except Exception:
            pass
        return True


def skip_strenum_public_str_methods(
    app, _obj_type, member_name, member_obj, skip, options
):
    """Skip documenting inherited public `str` methods on `str` + `Enum` classes.

    Returning `True` skips the member, `None` defers to the default. The callback only
    suppresses methods that are (a) public, (b) genuinely the inherited `str.<name>`
    object, and (c) not overridden in the enum subclass's own `__dict__`.
    """
    if skip or member_name.startswith("_"):
        return None
    sm = getattr(str, member_name, None)
    if sm is None or not callable(sm):
        return None
    same = member_obj is sm or (
        getattr(member_obj, "__objclass__", None) is str
        and getattr(member_obj, "__name__", None) == member_name
    )
    if not same:
        return None

    modname = app.env.current_document.autodoc_module
    cls_name = app.env.current_document.autodoc_class
    if not modname or not cls_name:
        return True

    try:
        module = importlib.import_module(modname)
        subject = getattr(module, cls_name)
    except (AttributeError, ImportError):
        return True

    if not isinstance(subject, type):
        return True
    if issubclass(subject, enum.Enum):
        if member_name in subject.__dict__:
            return None
        return True
    return None


# Each Plotly figure is emitted by sphinx-markdown-builder as a verbatim HTML block
# containing a `plotly-graph-div` and a `Plotly.newPlot(...)` payload (often ~200 KB per
# figure). The payload is useless to an LLM and bloats both the per-page `*.html.md` and
# the concatenated `llms-full.txt`. Two wrappers occur: sphinx-gallery's
# `<div class="output_subarea ...">` cell wrapper and the `<html><body>...` standalone
# document emitted by `plotly.io.show()`.
_PLOTLY_BLOCK_RES = [
    re.compile(
        r'<div class="output_subarea output_html rendered_html output_result">'
        r".*?plotly-graph-div"
        r".*?</div>\s*</div>",
        flags=re.DOTALL,
    ),
    re.compile(
        r"<html>.*?plotly-graph-div.*?</html>",
        flags=re.DOTALL,
    ),
]
_PLOTLY_PLACEHOLDER = "[plotly figure stripped from llms output]"

# Remove metadata comments (e.g. `<!-- !! processed by numpydoc !! -->`) from
# sphinx-llm pages
_LLMS_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_LLMS_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_LLMS_SITE_URL_RE = re.compile(
    r"(?P<prefix>\]\(https://skfolio\.org/)"
    r"(?P<path>[^)\r\n]*\\[^)\r\n]*)(?=\))"
)
_LLMS_TRUNCATED_SECONDARY_LINK_RE = re.compile(
    r"\[(?P<label>[^\]\r\n]+)\]\("
    r"https://skfolio\.org(?:/[^)\r\n]*)?\.\.\.(?=\r?$)",
    flags=re.MULTILINE,
)


def _sanitize_llm_markdown_text(text: str) -> tuple[str, int, int]:
    """Prepare sphinx-llm markdown text for LLM-oriented output.

    Order of operations: replace Plotly HTML regions first, then remove all HTML
    comments, collapse runs of three or more newlines to two and strip leading newlines
    so files do not start with empty lines after comment removal.
    """
    n_plotly = 0
    for regex in _PLOTLY_BLOCK_RES:
        text, n = regex.subn(_PLOTLY_PLACEHOLDER, text)
        n_plotly += n
    text, n_comments = _LLMS_HTML_COMMENT_RE.subn("", text)
    text = _LLMS_EXCESS_BLANK_LINES_RE.sub("\n\n", text).lstrip("\n")
    return text, n_plotly, n_comments


def _normalize_llms_txt_urls(text: str) -> tuple[str, int]:
    """Use URL separators in links generated for the `llms.txt` sitemap."""

    def replace_url(match: re.Match) -> str:
        return match.group("prefix") + match.group("path").replace("\\", "/")

    return _LLMS_SITE_URL_RE.subn(replace_url, text)


def _repair_truncated_llms_txt_links(text: str) -> tuple[str, int]:
    """Replace truncated secondary links with their readable labels."""
    return _LLMS_TRUNCATED_SECONDARY_LINK_RE.subn(
        lambda match: match.group("label"),
        text,
    )


def postprocess_llm_markdown_artifacts(app, exception):
    """Post-process sphinx-llm markdown files after the HTML build.

    Runs after sphinx-llm's own `build-finished` hook (priority > 500) so the per-page
    `*.html.md`, `llms-full.txt` and `llms.txt` files already exist on disk. Strips
    inline Plotly HTML, removes leftover numpydoc/sphinx-gallery metadata comments,
    normalises whitespace, repairs truncated description links and ensures sitemap
    links use URL separators on every platform.
    """
    if exception is not None:
        return
    if app.builder.name not in ("html", "dirhtml"):
        return
    outdir = Path(app.outdir)

    stale_files = 0
    if app.builder.name == "html":
        expected_files = {Path(f"{docname}.html.md") for docname in app.env.found_docs}
        for path in outdir.rglob("*.html.md"):
            if path.relative_to(outdir) not in expected_files:
                path.unlink()
                stale_files += 1

    targets = list(outdir.rglob("*.html.md"))
    llms_full = outdir / "llms-full.txt"
    if llms_full.exists():
        targets.append(llms_full)
    llms_index = outdir / "llms.txt"
    if llms_index.exists():
        targets.append(llms_index)
    total_plotly = 0
    total_comments = 0
    total_normalized_urls = 0
    total_repaired_links = 0
    total_files = 0
    for path in targets:
        original = path.read_text(encoding="utf-8")
        text, n_plotly, n_comments = _sanitize_llm_markdown_text(original)
        n_normalized_urls = 0
        n_repaired_links = 0
        if path == llms_index:
            text, n_normalized_urls = _normalize_llms_txt_urls(text)
            text, n_repaired_links = _repair_truncated_llms_txt_links(text)
        if (
            n_plotly
            or n_comments
            or n_normalized_urls
            or n_repaired_links
            or text != original
        ):
            path.write_text(text, encoding="utf-8")
            total_plotly += n_plotly
            total_comments += n_comments
            total_normalized_urls += n_normalized_urls
            total_repaired_links += n_repaired_links
            total_files += 1
    print(
        f"[llms-postprocess] updated {total_files} file(s): "
        f"{total_plotly} Plotly block(s), {total_comments} HTML comment(s), "
        f"{total_normalized_urls} URL(s) normalized, "
        f"{total_repaired_links} truncated link(s) repaired, "
        f"{stale_files} stale file(s) removed"
    )


def setup(app):
    """Setup function to register autodoc, HTML, and build-finished hooks."""
    _disable_jupyterlite_markdown_cleanup(app)

    # sphinx-llm re-executes this conf under the markdown builder.
    if getattr(app, "_skfolio_autodoc_str_enum_noise_filter", None) is None:
        noise_filter = _SkfolioAutodocStrEnumNoise()
        app._skfolio_autodoc_str_enum_noise_filter = noise_filter
        for _log in ("sphinx.ext.autodoc", "sphinx.ext.autosummary"):
            logging.getLogger(_log).addFilter(noise_filter)

    # Skip inherited str methods that should not appear in API documentation.
    app.connect("autodoc-skip-member", skip_str_inherited_members)

    # Filter inherited str methods out of every (str, Enum) class. We set priority below
    # the default 500. The callback returns None outside its narrow target so it
    # composes with later handlers.
    app.connect("autodoc-skip-member", skip_strenum_public_str_methods, priority=-100)

    # Configure sphinx-markdown-builder for documentation-specific nodes and assets.
    app.connect("builder-inited", patch_markdown_builder)

    # html page context
    # Set the homepage URL before other page-context consumers.
    app.connect("html-page-context", override_canonical, priority=499)
    app.connect("html-page-context", inject_schema)
    app.connect("html-page-context", override_html_title)
    app.connect("html-page-context", inject_noindex_meta, priority=900)
    app.connect("html-page-context", inject_representative_image_meta, priority=900)
    app.connect("html-page-context", inject_api_meta_description, priority=900)
    app.connect("html-page-context", inject_example_meta_description, priority=900)
    app.connect("html-page-context", inject_user_guide_meta_description, priority=900)

    # Build finished
    # Populate before sphinx-sitemap's default-priority writer, then post-process its
    # complete output.
    app.connect("build-finished", populate_complete_sitemap, priority=499)
    app.connect("build-finished", prune_and_fix_sitemap, priority=501)
    app.connect("build-finished", replace_index_links)
    app.connect("build-finished", create_redirects)
    app.connect("build-finished", normalize_viewport_meta, priority=998)

    # We set priority>500 so this runs after sphinx-llm's build-finished hook, which
    # generates the .html.md / llms-full.txt that we then post-process.
    app.connect("build-finished", postprocess_llm_markdown_artifacts, priority=999)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
