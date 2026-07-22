.. _factor_models:

.. currentmodule:: skfolio.prior

=============
Factor Models
=============

This guide covers skfolio's factor model implementations, their API and their
theoretical foundations. It focuses on the characteristics-based cross-sectional factor
model :class:`~skfolio.prior.CharacteristicsFactorModel`. This model family has
become foundational in quantitative asset management, and implementing it correctly
requires addressing many practical challenges (e.g. point-in-time data, changing 
universes, look-ahead bias, zero-sum constraints, alpha integration, diagnostics, 
attribution and computational performance on large universes).

The results in this guide were obtained by fitting a 58-factor US equity model on
the FactSet
`Fundamentals Point-in-Time <https://www.factset.com/marketplace/catalog/product/factset-fundamentals-point-in-time>`_,
`Estimates Point-in-Time Consensus <https://www.factset.com/marketplace/catalog/product/factset-estimates-point-in-time-consensus>`_ 
and `RBICS <https://www.factset.com/marketplace/catalog/product/factset-rbics-api>`_
datasets. The gallery examples use synthetic characteristics data, so users can run the full API
while respecting data vendor licences.

For complementary references, see "The Elements of Quantitative Investing" by Giuseppe
Paleologo [1]_, "Active Portfolio Management" by Grinold and Kahn [2]_, and
"Portfolio Optimization: Theory and Application" by Daniel P. Palomar [3]_.

Introduction
------------

Motivation and Use Cases
~~~~~~~~~~~~~~~~~~~~~~~~

Directly estimating the covariance matrix of a large asset universe is impractical.
The sample covariance matrix of 5,000 assets has over 12 million free parameters and
requires more than 5,000 observations (~20 years of daily data) in order to reach full 
rank. The resulting estimate is ill-conditioned, unstable and slow to react to change, 
and an optimizer using that estimate will also tend to allocate toward the parts of the 
covariance structure where estimation noise is largest.

A factor model addresses this problem by assuming that a small set of pervasive
factors (e.g. market, industries, countries, currencies, styles) drives the
co-movement of assets, and the remainder is asset-specific (i.e. idiosyncratic).
Estimation then reduces to a small factor covariance plus one idiosyncratic variance
per asset, requiring far fewer parameters and a much shorter history.

Factor models are used for:

* **Risk forecasting**: estimate a stable asset covariance matrix for large universes.
* **Risk decomposition**: express portfolio risk as systematic exposures plus
  idiosyncratic risk, separating intended from unintended exposures.
* **Performance attribution**: explain realized returns factor by factor,
  distinguishing systematic factor premia from asset-specific returns, and separating 
  skill from luck.
* **Portfolio construction**: supply expected returns, covariance, scenarios and
  factor exposures to an optimizer, with exposures that can be monitored and
  constrained.
* **Alpha research**: provide the ingredients of the alpha workflow, such as
  idiosyncratic returns as a prediction target, idiosyncratic variances
  for signal scaling, and factor exposures for neutralization.
* **Alpha decomposition**: split expected returns into factor-spanned and orthogonal
  components before optimization.

Model Definition
~~~~~~~~~~~~~~~~

A factor model decomposes the return of asset :math:`i` at time :math:`t` into:

.. math::

    r_{t,i} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} \, f_{t,k} + \epsilon_{t,i}

or in matrix form:

.. math::

    r_t = \alpha + B \, f_t + \epsilon_t

where:

* :math:`B \in \mathbb{R}^{N \times K}` is the factor exposure matrix, also called
  the loading matrix: the sensitivity of each asset to each factor.
* :math:`f_t \in \mathbb{R}^{K}` is the vector of factor returns: the return per unit
  of exposure at time :math:`t`, common to all assets.
* :math:`\epsilon_t \in \mathbb{R}^{N}` is the vector of idiosyncratic returns, the
  part of each asset's return not explained by the factors.
* :math:`\alpha \in \mathbb{R}^{N}` is the asset-level alpha, or intercept.

The vector of expected asset returns is:

.. math::

    \mu = \alpha + B \, \mu_f

where :math:`\mu_f = \mathbb{E}[f_t]` is the vector of expected factor returns.
When factor returns are centered, :math:`\alpha = \mu`; otherwise, factor premia
explain part of :math:`\mu`. Pure risk models set :math:`\alpha = 0`.

When an alpha estimator is provided,
:class:`~skfolio.prior.CharacteristicsFactorModel` treats :math:`\alpha` as an
asset-level forecast and decomposes it into a factor-spanned part and an orthogonal
part:

.. math::

    \alpha = \alpha^{\parallel} + \alpha^{\perp}
    \qquad \text{with} \qquad
    \alpha^{\parallel} = B \, g

The spanned part :math:`\alpha^{\parallel}` is explained by factor exposures and can
be written as :math:`B g`. The vector :math:`g` contains the expected factor returns
that reproduce the spanned alpha. The orthogonal part :math:`\alpha^{\perp}` is the
asset-level alpha left outside the factor space. This gives two estimates of
expected factor returns: :math:`\mu_f`, estimated from the factor return time series,
and :math:`g`, obtained by projecting the alpha forecast onto the factor exposure
space.
skfolio blends these sources with `spanned_alpha_shrinkage`, which is covered in
:ref:`Spanned and Orthogonal Alpha <factor_model_spanned_alpha>`.

The factor structure assumes that common co-movement is captured by the factors, and
that the remaining idiosyncratic covariance is diagonal or sparse. The asset
covariance matrix is:

.. math::

    \Sigma = B \, F \, B^\top + D

where :math:`F \in \mathbb{R}^{K \times K}` is the factor covariance matrix and
:math:`D` is the diagonal or sparse idiosyncratic covariance matrix.

The rest of this guide covers the estimation of :math:`B`, :math:`f_t`, :math:`F`,
:math:`D` and :math:`\alpha`, and their use in optimization and attribution.

Types of Factor Models
~~~~~~~~~~~~~~~~~~~~~~

Factor model families differ by what is observed before fitting: factor returns,
factor exposures, or neither.

**Time-series factor models** observe the factor returns and estimate asset exposures.
Factors are observable time series such as factor ETF returns, long-short factor portfolio
returns, or macroeconomic series (e.g. inflation, rates, GDP growth), hence the
alternative name "macroeconomic factor models". Each asset's exposures come from a
time-series regression of its returns on those factor series. They are interpretable,
straightforward to estimate, and they can work on small universes because each
asset is estimated independently. However, they require long return histories per asset,
exposures react slowly to asset-level change because they only update through the
regression window, and the risk of misspecification is higher. When factors are
tradable, their return history can be used to estimate factor premia. For
non-tradable variables, factor premia are estimated with a second-pass
cross-sectional procedure such as Fama-MacBeth on a sufficiently broad universe.
Implemented in :class:`~skfolio.prior.TimeSeriesFactorModel`.

**Characteristics-based factor models**, also called "fundamental factor models", observe 
exposures and estimate factor returns. Exposures are built from point-in-time
asset characteristics (e.g. industry classification, country, market capitalization,
book equity, sales, operating cash flow, analyst estimates), and factor returns are derived
from one cross-sectional regression of asset returns on exposures at each date.
Exposures react immediately to asset-level change, new assets need no return history to
receive exposures, and factors can be neutralized against each other to produce pure
factor definitions. The challenges of this model are the complexity of the estimation 
procedure and the heavier data requirement, both of which are covered in this guide. 
Implemented in :class:`~skfolio.prior.CharacteristicsFactorModel`.

.. note::

    The Fama-French procedure also starts from characteristics, but follows a different
    construction. It uses characteristics to sort assets into quantile long-short
    portfolios whose returns define the factors. It is primarily a factor-pricing
    framework for explaining the cross-section of expected returns, as opposed to a full 
    risk model. To use those factor returns as a risk model for arbitrary assets, exposures
    must be estimated by time-series regression, inheriting the drawbacks of
    time-series factor models.

    Moreover, quantile portfolios are not pure factors in the characteristics
    risk-model sense. A value portfolio, for example, can also carry industry, size,
    profitability, investment or momentum tilts because the sorting procedure only 
    controls the characteristics used to build the portfolio. Intersection procedures 
    reduce this contamination by sorting on several characteristics at the same time, 
    but they do not scale to a large number of factors.

**Statistical factor models** observe neither factor returns nor exposures and instead 
extract both from asset returns using methods such as PCA. They are adaptive and need 
returns only, but factors have no economic identity and may drift across regimes. 
Currently not implemented in skfolio.


Historical Background
~~~~~~~~~~~~~~~~~~~~~

Characteristics factor models originate with Barr Rosenberg, whose 1974 work on
extra-market components of covariance [4]_ showed that asset characteristics predict
return covariation, and that a small set of common factors explains most
cross-sectional variation while the residual is asset-specific. Rosenberg founded
Barra (now part of MSCI) to commercialize the approach, and the first US equity model
(USE1) was released in 1975. Specialist competitors followed, including Northfield
(1985), Axioma (1998) and Wolfe Research (2008). Data vendors later entered the market
by adding proprietary factor models to their data and analytics platforms, including
Bloomberg, FactSet, S&P Global and Morningstar. Factor risk models have since become 
standard infrastructure across quantitative hedge funds, asset managers and banks.

These models were initially built for risk estimation, but practitioners observed
that some risk factors (e.g. value, momentum) also earn persistent premia. Modern
implementations therefore estimate expected factor returns alongside factor
covariances, enabling deliberate tilts toward specific factors in optimization, the
foundation of what became known as "smart beta".

Today, commercial factor risk models are often distributed as model catalogues, usually segmented by 
region and horizon. skfolio now provides a toolkit for building and customizing factor models directly 
from user-defined data, rather than selecting a single predefined model from a catalogue. This enables a 
continuum of specifications, recognizing the view that there is no "one size fits all" factor model.

Alternatives to a Single Centralized Factor Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

A factor model is an estimate. Its exposures, factor covariance and
idiosyncratic variances carry estimation error, and its specification is never
complete, as no factor set captures all common co-movement.

The traditional commercial offering can lead to a single house risk model 
being shared by all strategies and risk management teams. skfolio supports 
this approach but also allows for an alternative approach where the factor model is
itself an estimator parameter, such that each strategy can embed its own specific 
factor model. The strategy and its factor model then form a single meta-model that is fitted,
evaluated and tuned jointly.

Multiple factor models allow portfolio construction and risk monitoring to use different model 
specifications. A portfolio may be constructed with a factor model tailored to its universe, 
horizon and alpha process, while risk management evaluates exposures and covariance with a 
broader firm-wide model. At the book level, using different factor models across portfolios 
and strategies introduces model diversification. Estimation error and specification error are 
then distributed across model specifications rather than concentrated in a single shared model.

Model Overview
--------------

Estimation Pipeline
~~~~~~~~~~~~~~~~~~~

:class:`~skfolio.prior.CharacteristicsFactorModel` is a meta-estimator following
scikit-learn conventions. It composes sub-estimators, descriptors, factor exposure
estimators, a cross-sectional regressor, prior estimators for factor and idiosyncratic
risk and an optional alpha estimator, each of which can be replaced, tuned and
customized independently. The input is a point-in-time
:class:`~skfolio.containers.AssetPanel` of asset characteristics and the output is a
:class:`~skfolio.prior.ReturnDistribution` consumed by all skfolio optimizations,
together with a :class:`~skfolio.prior.FactorModel` container holding the full
decomposition and diagnostics.

The model is fitted as follows:

1. Start from point-in-time asset characteristics stored as panel fields (e.g.
   `returns`, `market_cap`, `book_equity`, `industry`).
2. Compute descriptor values from these fields, or pass through existing fields
   unchanged, using descriptor estimators.
3. Build factor exposures. Style factors are typically formed by combining one or
   more descriptors and applying cross-sectional transformations (e.g. winsorization,
   z-scoring). Categorical factors (e.g. industry, country, currency) are represented
   by one-hot exposures.
4. Orthogonalize selected exposures against other factors or families when
   `neutralize_against` is provided.
5. Reparameterize constrained families when `constrained_families` is provided. This
   enforces the benchmark-weighted zero-sum constraint on factor returns within each
   constrained family and produces a full-rank basis for factor-level estimators.
6. Estimate realized factor returns with
   `cs_regressor` on the estimation universe defined by the panel's
   `estimation_mask`. By default, regression weights are based on market
   capitalization through `regression_mcap_power`. When
   `inv_idio_variance_weight_shrinkage > 0`, a two-pass procedure blends those weights
   with inverse-idiosyncratic-variance weights estimated from first-pass residuals.
7. Estimate the factor return distribution with `factor_prior_estimator`, including
   expected factor returns (factor premia), factor covariance and factor return
   scenarios. This step can introduce factor covariance shrinkage, short-term
   volatility updating or Newey-West HAC correction.
8. Estimate idiosyncratic variances with `idio_variance_estimator`, then form the
   idiosyncratic covariance as a diagonal matrix or, when `idio_corr_threshold > 0`,
   as a sparse covariance using correlation thresholding.
9. If provided, fit `alpha_estimator` to produce an alpha forecast. Decompose it into
   factor-spanned and orthogonal alphas, blend the spanned alpha with the expected
   factor returns using `spanned_alpha_shrinkage`, shrink the orthogonal alpha with
   `orthogonal_alpha_confidence` and assemble the final :math:`\mu`, :math:`\Sigma`
   and asset return scenarios on the investment universe.

Each step is detailed in the following sections.

.. _factor_model_code_example:

Code Example
~~~~~~~~~~~~

The model below is used throughout this guide. It is a medium-horizon US equity model
with 58 factors: 1 global factor, 44 industry factors and 13 style factors built from
29 descriptors. It uses within-industry scoring, neutralization, a zero-sum constraint
on the industry family, regression weights that blend market-capitalization weights
with inverse-idiosyncratic-variance weights and a regime-adjusted factor covariance
estimator. The model was fitted on the FactSet point-in-time datasets of 2,000 US
equities with daily data from 2013 to 2026. The parametrization is intentionally 
simple and has not been fine-tuned. This simplified model is used as the standard 
example for presenting the API. See :ref:`Hyperparameter Tuning
<factor_model_hyper_parameter_tuning>` for guidance on tuning the model to your data
and goals.

.. code-block:: python

    from skfolio.descriptor import (
        AssetsGrowthRate,
        AssetTurnover,
        BookLeverage,
        BookToPrice,
        CapexToAssetsChangeInIntensity,
        CashFlowToAssets,
        CashFlowToPrice,
        DebtToAssets,
        DividendToPrice,
        EarningsChangeToPrice,
        EarningsToPrice,
        EbitdaToEnterpriseValue,
        EWAmihudIlliquidity,
        EWMarketBeta,
        EWMomentum,
        EWResidualVolatility,
        EWShareTurnover,
        EWVolatility,
        ForwardEarningsToPrice,
        GrossMargin,
        GrossProfitability,
        IssuanceGrowthRate,
        LogMarketCap,
        MarketLeverage,
        ReturnOnAssets,
        ReturnOnEquity,
        SalesGrowthRate,
        SalesToPrice,
        ShareholderYield,
    )
    from skfolio.factor_exposure import (
        DerivedFactor,
        FixedWeightedFactor,
        GlobalFactor,
        OneHotCategoricalFactors,
    )
    from skfolio.moments import EWMu, RegimeAdjustedEWCovariance
    from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior

    month = 21
    quarter = 3 * month
    half_year = 6 * month
    year = 12 * month

    # Global factor
    global_factor = GlobalFactor(family="market")

    # Industry factors
    industry_factors = OneHotCategoricalFactors(category="industry", family="industry")

    # Style factors
    beta_factor = FixedWeightedFactor(
        descriptors=[("market_beta", EWMarketBeta(half_life=year))],
        transform_by_group="industry",
    )

    momentum_factor = FixedWeightedFactor(
        descriptors=[("momentum", EWMomentum(half_life=half_year, skip=month))],
        transform_by_group="industry",
    )

    size_factor = FixedWeightedFactor(
        descriptors=[("log_mcap", LogMarketCap())], transform_by_group="industry"
    )

    non_linear_size_factor = DerivedFactor(
        source="size", func=lambda x: x**3, transform_by_group="industry"
    )

    value_factor = FixedWeightedFactor(
        descriptors=[
            ("book_to_price", BookToPrice()),
            ("sales_to_price", SalesToPrice()),
            ("cash_flow_to_price", CashFlowToPrice()),
        ],
        weights=[0.8, 0.1, 0.1],
        transform_by_group="industry",
    )

    earnings_yield_factor = FixedWeightedFactor(
        descriptors=[
            ("fwd_earnings_to_price", ForwardEarningsToPrice()),
            ("earnings_to_price", EarningsToPrice()),
            ("enterprise_multiple", EbitdaToEnterpriseValue()),
        ],
        transform_by_group="industry",
    )

    growth_factor = FixedWeightedFactor(
        descriptors=[
            ("earnings_change_to_price", EarningsChangeToPrice(lag=year)),
            ("sales_growth", SalesGrowthRate(lag=year)),
        ],
        transform_by_group="industry",
    )

    profitability_factor = FixedWeightedFactor(
        descriptors=[
            ("asset_turnover", AssetTurnover()),
            ("gross_profitability", GrossProfitability()),
            ("gross_margin", GrossMargin()),
            ("return_on_assets", ReturnOnAssets()),
            ("return_on_equity", ReturnOnEquity()),
            ("cash_flow_to_assets", CashFlowToAssets()),
        ],
        transform_by_group="industry",
    )

    investment_factor = FixedWeightedFactor(
        descriptors=[
            ("asset_growth", AssetsGrowthRate(lag=year)),
            ("issuance_growth", IssuanceGrowthRate(lag=year)),
            ("capex_growth", CapexToAssetsChangeInIntensity(lag=year)),
        ],
        transform_by_group="industry",
    )

    dividend_yield_factor = FixedWeightedFactor(
        descriptors=[
            ("dividend_to_price", DividendToPrice()),
            ("shareholder_yield", ShareholderYield()),
        ],
        weights=[0.7, 0.3],
        transform_by_group="industry",
    )

    leverage_factor = FixedWeightedFactor(
        descriptors=[
            ("market_leverage", MarketLeverage()),
            ("debt_to_assets", DebtToAssets()),
            ("book_leverage", BookLeverage()),
        ],
        transform_by_group="industry",
    )

    liquidity_factor = FixedWeightedFactor(
        descriptors=[
            ("share_turnover", EWShareTurnover(half_life=quarter)),
            ("amihud_illiquidity", EWAmihudIlliquidity()),
        ],
        transform_by_group="industry",
    )

    volatility_factor = FixedWeightedFactor(
        descriptors=[
            ("vol", EWVolatility(half_life=quarter)),
            (
                "residual_vol",
                EWResidualVolatility(half_life=quarter, beta_half_life=quarter),
            ),
        ],
        transform_by_group="industry",
    )

    # Characteristics Factor Model
    model = CharacteristicsFactorModel(
        factors=[
            ("market", global_factor),
            ("industry", industry_factors),
            ("beta", beta_factor),
            ("momentum", momentum_factor),
            ("size", size_factor),
            ("non_linear_size", non_linear_size_factor),
            ("value", value_factor),
            ("earnings_yield", earnings_yield_factor),
            ("growth", growth_factor),
            ("profitability", profitability_factor),
            ("investment", investment_factor),
            ("dividend_yield", dividend_yield_factor),
            ("leverage", leverage_factor),
            ("liquidity", liquidity_factor),
            ("volatility", volatility_factor),
        ],
        neutralize_against={
            "non_linear_size": ["size"],
            "volatility": ["beta"],
        },
        constrained_families=[("industry", None)],
        exposure_lag=1,
        inv_idio_variance_weight_shrinkage=0.5,
        factor_prior_estimator=EmpiricalPrior(
            covariance_estimator=RegimeAdjustedEWCovariance(
                half_life=half_year, corr_half_life=year, regime_half_life=month
            ),
            mu_estimator=EWMu(half_life=year),
        ),
        n_jobs=-1,
    )

The model is fitted on a point-in-time :class:`~skfolio.containers.AssetPanel` whose
construction is covered in the :ref:`Input Data <factor_model_input_data>` section:

.. code-block:: python

    model.fit(characteristics=characteristics)

For incremental updates on new observations, use `partial_fit`, covered in the
:ref:`Online Learning <factor_model_online_learning>` section:

.. code-block:: python

    model.partial_fit(characteristics=new_characteristics)


After fitting, the model exposes two main attributes.

`return_distribution_` is the :class:`~skfolio.prior.ReturnDistribution` consumed by
skfolio optimizations. It contains the expected asset returns `mu`, the asset
covariance matrix `covariance` and the asset return scenarios `returns`, all on the
investment universe. Assets that are not investable at the current point in time
(e.g. delisted, not yet listed, or still in :ref:`warmup <factor_model_warmup>`)
are represented with NaN.

`factor_model_` is the :class:`~skfolio.prior.FactorModel` container holding the full
decomposition: factor exposures, loading matrix, factor returns, expected factor
returns and factor covariance, idiosyncratic returns, variances and covariance,
regression and benchmark weights. It provides DataFrame accessors (`factor_returns_df`,
`idio_returns_df`, `exposures_df`), slicing (`select_assets`, `select_observations`),
a `summary` method and the diagnostic statistics and plots covered in the
:ref:`Diagnostics <factor_model_diagnostics>` section.

.. code-block:: python

    distribution = model.return_distribution_
    distribution.mu          # expected returns, shape (n_assets,)
    distribution.covariance  # asset covariance, shape (n_assets, n_assets)
    distribution.returns     # return scenarios, shape (n_observations, n_assets)

    factor_model = model.factor_model_
    factor_model.summary()
    factor_model.factor_returns_df()


The fitted sub-estimators are also available with the usual scikit-learn trailing
underscore convention: `cs_regressor_`, `factor_prior_estimator_`,
`idio_variance_estimator_`, `idio_corr_estimator_` and `alpha_estimator_`.

The `X` argument of `fit` is optional. When provided, `X` selects the investment
universe: its columns define the assets returned in `return_distribution_` and
`factor_model_`. Factor estimation still uses the `returns` field of
`characteristics`, which can cover a broader point-in-time universe. This keeps the
estimator compatible with skfolio pipelines, cross-validation, prediction and scoring.
When `X` is `None`, the investment universe equals the coverage universe.

.. _factor_model_input_data:

Input Data
----------

The model consumes a point-in-time :class:`~skfolio.containers.AssetPanel` of asset
characteristics. The panel must include a `returns` field and, when market-cap
weighting is used, a `market_cap` field. The remaining fields depend on the chosen
descriptors and alpha estimators. The :ref:`example <factor_model_code_example>`
model uses market data (e.g.
`adj_close`, `adj_volume`, `adj_shares_outstanding`), fundamentals (e.g.
`book_equity`, `sales_ttm`, `total_assets`, `operating_cash_flow_ttm`), analyst
estimates (e.g. `eps_ntm`, `dps_ntm`) and a categorical `industry` field.

AssetPanel Container
~~~~~~~~~~~~~~~~~~~~

A factor model pipeline applies many cross-sectional and time-series transformations
to the same date-by-asset data. With general-purpose containers (e.g. DataFrames, xarray),
each step must re-align indexes, group by date or pivot before computing. This adds
overhead and increases the risk of indexing errors, either on the time index, which
can introduce look-ahead bias, or on the asset index. To address this, skfolio provides
:class:`~skfolio.containers.AssetPanel`, a dedicated container for aligned
cross-sectional asset data. An :class:`~skfolio.containers.AssetPanel` validates 
alignment once so that estimators operate on already-aligned numeric arrays and stores 
universe masks explicitly, which is needed for point-in-time estimation on changing 
universes (e.g. listing, delisting, inclusion/exclusion rules).

The rationale behind the container, the wide-format convention and the NaN-handling
conventions are covered in :ref:`Asset Data Representation
<asset_data_representation>`.

Every field shares the same two axes with shape `(n_observations, n_assets)`. Three
kinds of fields are supported:

* 2D numeric fields (e.g. `returns`, `market_cap`)
* 2D categorical fields (e.g. `industry`, `country`), stored as integer codes with
  their category labels
* 3D numeric fields (e.g. factor exposures), with labeled third axis

The panel also carries two boolean masks aligned with the data, `active_mask` and
`estimation_mask`, described in :ref:`Coverage, Estimation and Investment
Universes <factor_model_universes>`.

This layout has additional practical benefits:

* The container is scikit-learn compatible: `len(panel)` returns the number of
  observations and `panel[start:stop]` returns a zero-copy view, so the panel can be
  passed directly to cross-validation and hyper-parameter tuning utilities, and
  walk-forward folds reuse the same field arrays instead of copying them.
* With thread-based parallelism, workers read the same panel in memory instead of
  receiving separate process copies, which is significant for large panels.
* Panels are saved as `.npy` files and support memory-mapped loading for fast
  startup on large datasets.

.. code-block:: python

    import numpy as np
    from skfolio.containers import AssetPanel
    from skfolio.datasets import make_synthetic_characteristics

    # Build a panel from aligned date-by-asset arrays.
    panel = AssetPanel(
        fields={
            "returns": returns,        # ndarray (n_observations, n_assets)
            "market_cap": market_cap,  # ndarray (n_observations, n_assets)
        },
        observations=dates,
        asset_names=assets,
        active_mask=active_mask,
        estimation_mask=estimation_mask,
    )
    panel.add_categorical_field(
        name="industry",
        values=industry_codes,  # integer codes (n_observations, n_assets)
        levels=["energy", "bank", "technology"],
    )

    # Inspect dimensions, fields, missing values and mask coverage.
    panel.info()

    # Save and reload a panel.
    panel.save("path/to/saved_panel")
    saved_panel = AssetPanel.load("path/to/saved_panel")

    # Generate a synthetic panel for examples and tests.
    characteristics = make_synthetic_characteristics()


.. _factor_model_universes:

Coverage, Estimation and Investment Universes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A factor model distinguishes three universes:

* The **coverage universe** is the full set of assets stored in the
  :class:`~skfolio.containers.AssetPanel`. It contains the estimation universe and
  the investment universe.
* The **estimation universe**, defined by the panel's `estimation_mask`, is the
  subset of observation-asset pairs used to fit cross-sectional statistics,
  factor-return regressions, benchmark and regression weights, alpha estimators and
  regime statistics. Pairs outside it still receive transformed values, exposures and
  forecasts, but do not contribute to those fitted statistics.
* The **investment universe** is the set of assets selected for portfolio
  optimization. It is returned in `return_distribution_` and
  `factor_model_`. When `X` is provided to `fit`, its columns select this universe.
  When `X` is `None`, the investment universe equals the coverage universe. This lets
  the model estimate factors on a broader cross-section, then return outputs only for
  the assets used downstream.

Membership within the coverage universe varies through time and is tracked by the
panel's `active_mask`. `False` marks an asset outside the universe at that date
(e.g. pre-listing, post-delisting), while `True` with a NaN value marks missing data
for an active asset (e.g. holiday, missing quote). The `estimation_mask` is enforced
as a subset of `active_mask`.

The estimation universe must be broad enough to represent the investment opportunity
set, liquid enough to avoid spurious return relationships and stable enough for factor
exposures to behave consistently through time. For a US equity model, a typical
estimation universe is 1,000 to 3,000 names. It should also be large enough relative to the 
factor set. The number of estimation assets :math:`N` should be well above the number of 
factors :math:`K` for a stable cross-sectional regression, and best practice 
at least 20 estimation assets per industry (e.g. 50 industries should require at least
1,000 assets in a well-balanced universe, and more if some industries are sparsely
represented).

The effective size of the estimation universe also depends on the regression weights.
The default square-root market-capitalization weights are commonly used as a proxy for
inverse idiosyncratic variance. Under square-root market-capitalization weighting, the
contribution of small stocks decreases rapidly: in a broad US universe, the largest
2,000 stocks carry about 98% of the total regression weight. Extending the universe 
beyond this point can still be useful where the additional names improve estimation 
of small-cap-sensitive factors or increase coverage of sparsely represented industries. 
Otherwise, extending the estimation universe further mostly adds memory and compute cost 
and may add noise from illiquid securities (e.g. stale prices, zero-return days, 
bid-ask bounce, missing fundamentals).

.. _factor_model_point_in_time_data:

Point-in-Time Data
~~~~~~~~~~~~~~~~~~

AssetPanel characteristics should be built from point-in-time (PIT) datasets. For
fundamentals, this means respecting reporting lags and using the figures as reported
at the time, before later restatements. Data vendors offer point-in-time datasets for
this purpose, and the example in this guide is constructed using FactSet point-in-time datasets. 
When actual availability dates are not available, a common fallback is to apply a
conservative lag from the fiscal period end, such as 90 days, before making the value
available to the model.

Let's suppose ABC's fiscal Q2 ends on 2024-06-30, Q2 sales are reported as 100m
on 2024-08-05, and later restated to 96m on 2024-11-12. If the latest available
Q1 sales value is 90m before the Q2 report, a daily `AssetPanel` stores:

.. list-table::
   :header-rows: 1

   * - Observation date
     - Sales value in the PIT panel
     - Source available on that date
   * - 2024-08-01
     - 90m
     - Latest available Q1 value
   * - 2024-08-02
     - 90m
     - Latest available Q1 value
   * - 2024-08-05
     - 100m
     - 2024 Q2 report
   * - 2024-08-06
     - 100m
     - Latest available Q2 value
   * - 2024-11-12
     - 96m
     - 2024 Q2 restatement

Universe membership must also be point-in-time. The universe is formed by applying
eligibility rules at each observation date, using fields known on that date such as
listing status, market capitalization, liquidity, sector, country, exchange and
security type. Fitting history on the current constituent list introduces survivorship
bias as assets that were delisted, acquired or defaulted disappear from the sample,
biasing both estimates. See
:ref:`data representation <asset_data_representation>` for missing data, changing
universes and `active_mask`.

.. _factor_model_time_alignment:

Time Alignment and Look-Ahead Bias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid look-ahead bias from inconsistent time indexing and manual lagging across
characteristic fields, exposures and returns, skfolio uses a single as-of
time-indexing convention across all estimators, so data is aligned once and lags 
enter as model parameters.

Under this convention, all time-varying inputs at observation :math:`t` reflect
information available up to and including the end of period :math:`t`. Point-in-time
fields and derived values (e.g. prices, fundamentals, industry labels and factor
exposures) store the latest available value for observation :math:`t`. Returns
stored at observation :math:`t` cover the period ending at :math:`t`, namely
:math:`(t-1, t]`.

Factor-return regressions estimate the factor returns realized over
:math:`(t-1, t]`. The exposure matrix must therefore describe the assets before that
return interval begins, at :math:`t-1`. `exposure_lag` selects that
exposure date and defaults to 1:

.. math::

    R(t) = B(t - \ell)\,f(t) + \epsilon(t)

where :math:`\ell` is `exposure_lag`. With the default :math:`\ell = 1`, returns over
:math:`(t-1, t]` are regressed on exposures measured at :math:`t-1`.

The same alignment applies to regression weights. Market capitalization weights are
lagged by `exposure_lag`, and inverse-idiosyncratic-variance weights at date
:math:`t` are estimated from residuals up to :math:`t-1`. Both are detailed in the
:ref:`Regression Weights <factor_model_regression_weights>` section.

Stored outputs follow the as-of time-indexing convention. `factor_model_.exposures` at :math:`t`
stores the exposures measured at :math:`t`, and the covariance forecast pairs the
factor covariance with the latest exposures :math:`B(T)`. The lag is applied
internally when estimating realized factor returns and regression-based statistics.

Split, Dividend and Excess Return Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `returns` field should contain total returns, computed from split and dividend
adjusted prices. It is the dependent variable of the cross-sectional regression and
the input to all return-based descriptors (beta, volatility, momentum, reversal).

Other market data used by descriptors, such as prices, trading volumes and shares
outstanding, should be split adjusted but not dividend adjusted. These enter
descriptors as price and quantity levels, for example as the denominator of a
valuation ratio or in turnover and liquidity measures, where reinvested dividends
would distort the level. The corresponding fields are detailed in the
:ref:`Descriptors <factor_model_descriptor_input_fields>` section. The `market_cap`
field should contain the market value of common equity at each date.

In a single-currency model, raw and excess returns give nearly identical regression
results, as subtracting a common risk-free rate shifts the cross-section by a constant
that the global factor absorbs. Excess returns are nonetheless still preferred because 
some descriptors estimate time-series regressions (e.g.
:class:`~skfolio.descriptor.EWMarketBeta`,
:class:`~skfolio.descriptor.EWResidualVolatility`) whose intercept should capture
residual return without absorbing the risk-free rate. In a multi-currency model,
`returns` should contain local excess returns, and currency excess returns are
supplied through the `currency_excess_returns` argument. See the :ref:`Currency
Factors <factor_model_currency_factors>` section.

.. _factor_model_factor_exposures:

Factor Exposures
----------------

Factor exposure estimators transform an :class:`~skfolio.containers.AssetPanel` into
factor exposure arrays. Some estimators compute
exposures without descriptors, such as the global market factor or one-hot industry
factors. Others compute one or more descriptors and transform them into standardized
style exposures.

skfolio provides four factor exposure estimators:

* :class:`~skfolio.factor_exposure.GlobalFactor` creates one common factor by
  assigning every asset an exposure of 1.0. In the cross-sectional regression,
  this factor acts as the intercept or broad market factor. With the
  benchmark-weighted centering and zero-sum constraints described in
  :ref:`Global Factor and Benchmark Portfolio <factor_model_global_factor>`, its
  estimated factor return captures the benchmark portfolio return.
* :class:`~skfolio.factor_exposure.OneHotCategoricalFactors` turns a categorical
  field into binary membership factors, one factor per category level. An asset has
  exposure 1.0 to the factor matching its category and 0.0 to the other
  category factors.
* :class:`~skfolio.factor_exposure.FixedWeightedFactor` builds a factor from one or
  more descriptors. Each descriptor is computed, passed through the outlier and
  scoring transforms, and converted into a cross-sectional score. The descriptor
  scores are then aggregated with the fixed descriptor weights. For each
  asset-observation pair, non-finite scores are ignored and the weighted average is
  divided by the weight assigned to the remaining finite scores, so an asset missing
  a descriptor (e.g. gross margin for financial firms, which do not report cost of
  goods sold) can still receive a composite score from its valid descriptors. The
  `min_coverage` parameter sets the minimum fraction of descriptor weight that must
  be finite and when below this threshold, the composite is NaN. When multiple descriptors 
  are combined and scoring is enabled, the composite is scored again cross-sectionally 
  so partial-coverage composites remain on a comparable scale.
* :class:`~skfolio.factor_exposure.DerivedFactor` applies a function to another
  factor's exposure, then optionally applies outlier and scoring transforms. In
  the :ref:`example <factor_model_code_example>`, it builds the non-linear size
  factor from the size exposure (`func=lambda x: x**3`). Dependencies between
  factors are resolved automatically through topological sorting.

Custom exposure estimators are created by subclassing
:class:`~skfolio.factor_exposure.BaseFactorExposure`.

Every factor exposure estimator carries a `family` attribute (e.g. `"market"`,
`"style"`, `"industry"`, `"country"`, `"currency"`). Families group related factors and
are used in neutralization, zero-sum constraints, attribution and reporting.

.. _factor_model_descriptors:

Descriptors
~~~~~~~~~~~

A descriptor is a transformer that reads one or more
:class:`~skfolio.containers.AssetPanel` fields and returns a value array of shape
`(n_observations, n_assets)`. Descriptors follow the as-of time-indexing convention: values at
observation :math:`t` use information available up to and including the end of period
:math:`t`.


skfolio provides descriptors covering the standard factor literature:

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :class: catalog-table

   * - Category
     - Descriptors
   * - Value
     - * :class:`~skfolio.descriptor.BookToPrice`
       * :class:`~skfolio.descriptor.CashFlowToPrice`
       * :class:`~skfolio.descriptor.SalesToPrice`
   * - Earnings yield
     - * :class:`~skfolio.descriptor.EarningsToPrice`
       * :class:`~skfolio.descriptor.ForwardEarningsToPrice`
       * :class:`~skfolio.descriptor.EbitdaToEnterpriseValue`
   * - Growth
     - * :class:`~skfolio.descriptor.AssetsGrowthRate`
       * :class:`~skfolio.descriptor.SalesGrowthRate`
       * :class:`~skfolio.descriptor.EarningsChangeToPrice`
       * :class:`~skfolio.descriptor.IssuanceGrowthRate`
       * :class:`~skfolio.descriptor.CapexToAssetsChangeInIntensity`
       * :class:`~skfolio.descriptor.GrowthRate`
       * :class:`~skfolio.descriptor.ChangeToScale`
       * :class:`~skfolio.descriptor.ChangeInIntensity`
   * - Profitability
     - * :class:`~skfolio.descriptor.GrossProfitability`
       * :class:`~skfolio.descriptor.GrossMargin`
       * :class:`~skfolio.descriptor.ReturnOnAssets`
       * :class:`~skfolio.descriptor.ReturnOnEquity`
       * :class:`~skfolio.descriptor.AssetTurnover`
       * :class:`~skfolio.descriptor.CashFlowToAssets`
       * :class:`~skfolio.descriptor.SalesToEnterpriseValue`
   * - Earnings quality
     - * :class:`~skfolio.descriptor.AccrualsCashFlow`
       * :class:`~skfolio.descriptor.AnalystDispersionToPrice`
   * - Dividend yield
     - * :class:`~skfolio.descriptor.DividendToPrice`
       * :class:`~skfolio.descriptor.ForwardDividendToPrice`
       * :class:`~skfolio.descriptor.ShareholderYield`
   * - Leverage
     - * :class:`~skfolio.descriptor.MarketLeverage`
       * :class:`~skfolio.descriptor.BookLeverage`
       * :class:`~skfolio.descriptor.DebtToAssets`
   * - Size
     - * :class:`~skfolio.descriptor.LogMarketCap`
   * - Momentum
     - * :class:`~skfolio.descriptor.EWMomentum`
       * :class:`~skfolio.descriptor.RollingMomentum`
   * - Reversal
     - * :class:`~skfolio.descriptor.Reversal`
   * - Volatility
     - * :class:`~skfolio.descriptor.EWVolatility`
       * :class:`~skfolio.descriptor.EWResidualVolatility`
       * :class:`~skfolio.descriptor.EWDownsideVolatility`
       * :class:`~skfolio.descriptor.EWResidualDownsideVolatility`
   * - Sensitivity
     - * :class:`~skfolio.descriptor.EWMarketBeta`
       * :class:`~skfolio.descriptor.EWMacroSensitivity`
   * - Downside risk
     - * :class:`~skfolio.descriptor.EWDownsideBeta`
   * - Liquidity
     - * :class:`~skfolio.descriptor.EWShareTurnover`
       * :class:`~skfolio.descriptor.EWAmihudIlliquidity`
   * - Lottery demand
     - * :class:`~skfolio.descriptor.MaxReturn`
   * - Short interest
     - * :class:`~skfolio.descriptor.ShortInterest`
       * :class:`~skfolio.descriptor.DaysToCover`

:class:`~skfolio.descriptor.Passthrough` exposes an existing panel field
as a descriptor without transformation, which is useful for vendor-supplied or
externally computed values. Custom descriptors are created by subclassing
:class:`~skfolio.descriptor.BaseDescriptor`.

.. _factor_model_descriptor_input_fields:

Each descriptor requires specific panel fields. The table below covers all fields used
by the built-in descriptors. A model only needs the fields used by the descriptors it
configures.

.. list-table::
   :header-rows: 1
   :widths: 25 75
   :class: catalog-table

   * - Field
     - Description
   * - `returns`
     - Asset returns
   * - `market_cap`
     - Market value of common equity
   * - `adj_close`
     - Split-adjusted close price
   * - `adj_volume`
     - Split-adjusted traded volume
   * - `adj_shares_outstanding`
     - Split-adjusted common shares outstanding
   * - `book_equity`
     - Common shareholders' equity
   * - `sales_ttm`
     - Trailing 12-month sales
   * - `operating_cash_flow_ttm`
     - Trailing 12-month operating cash flow
   * - `net_income_ttm`
     - Trailing 12-month net income
   * - `cost_of_revenue_ttm`
     - Trailing 12-month cost of revenue
   * - `ebitda_ttm`
     - Trailing 12-month EBITDA
   * - `dividends_ttm`
     - Trailing 12-month cash dividends paid on common shares
   * - `net_buybacks_ttm`
     - Trailing 12-month net share repurchases, positive when repurchases exceed issuance
   * - `total_assets`
     - Total assets
   * - `total_debt`
     - Total debt
   * - `enterprise_value`
     - Enterprise value (market capitalization plus debt minus cash)
   * - `capex_ttm`
     - Trailing 12-month capital expenditures
   * - `eps_ntm`
     - Consensus next-12-month earnings per share
   * - `dps_ntm`
     - Consensus next-12-month dividends per share
   * - `eps_ntm_std`
     - Cross-analyst standard deviation of next-12-month EPS estimates
   * - `short_interest`
     - Shares sold short

The `adj_` fields should be split adjusted but not dividend adjusted, since descriptors use them as price and
quantity levels where reinvested dividends would distort the level. The per-share estimate fields `eps_ntm`,
`dps_ntm` and `eps_ntm_std` must share the same split-adjustment basis as `adj_close`,
so that per-share ratios such as forward earnings yield are consistent.


.. _factor_model_cross_sectional_transformers:

Cross-Sectional Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw descriptor values have heavy tails and incomparable units (a book-to-price ratio
and a turnover rate are on different scales). Each descriptor is therefore passed
through two cross-sectional transformers before being combined:

* `outlier_transformer`, defaulting to
  :class:`~skfolio.preprocessing.CSWinsorizer`, caps extreme values.
* `scoring_transformer`, defaulting to
  :class:`~skfolio.preprocessing.CSStandardScaler`, converts values into
  cross-sectional z-scores.

With the default scoring, each descriptor is centered on the benchmark-weighted
mean (market-cap-weighted by default) and scaled by the equal-weighted standard
deviation. Benchmark-weighted centering gives the benchmark portfolio zero exposure
to every style factor. Equal-weighted scaling keeps the dispersion estimate from
being dominated by the largest assets.

An exposure of 1.0 means the asset is one standard deviation above the
benchmark-weighted average. The zero benchmark exposure determines the
interpretation of the global factor return, described in
:ref:`Global Factor and Benchmark Portfolio <factor_model_global_factor>`.
Transform statistics are computed on the estimation universe and applied to the
full coverage universe.

The `transform_by_group` parameter applies both transforms within groups defined by a 
categorical panel field (e.g. "industry", "country"). For example, with 
`transform_by_group="industry"`, each descriptor is scored within 
its own industry group, making cross-sectional scores more comparable across industries. 
This is useful because book-to-price, profitability, leverage, and similar descriptors 
can have very different distributions across industries. Group-level scoring also makes 
the resulting factor industry-neutral (see :ref:`Neutralization <factor_model_neutralization>`) 
and prevents a single industry from dominating the factor's variation.

The following transformers are available:

**Outlier transformers**

* :class:`~skfolio.preprocessing.CSWinsorizer`
* :class:`~skfolio.preprocessing.CSTanhShrinker`

**Scoring transformers**

* :class:`~skfolio.preprocessing.CSStandardScaler`
* :class:`~skfolio.preprocessing.CSGaussianRankScaler`
* :class:`~skfolio.preprocessing.CSPercentileRankScaler`

See :ref:`Cross-Sectional Transformers <cross_sectional_transformers>` for details.

.. _factor_model_neutralization:

Neutralization
~~~~~~~~~~~~~~

Style exposures are often correlated with other factors. For example, volatility 
correlates with beta, and an un-neutralized value exposure can carry industry tilts. 
Neutralization (also called orthogonalization) removes these overlaps, producing "pure" 
factor definitions, reducing collinearity in the cross-sectional regression and making 
factor returns easier to interpret.

The `neutralize_against` parameter maps each factor (or family) to the target
factors or families. The :ref:`example <factor_model_code_example>` uses:

.. code-block:: python

    neutralize_against={
        "non_linear_size": ["size"],
        "volatility": ["beta"],
    }

Neutralization is a weighted least squares projection. For a style exposure :math:`z`
and target exposures :math:`D` under benchmark weights :math:`W`:

.. math::

    z^{\perp} = z - D\,(D^\top W D)^{-1}\,D^\top W z

The residual :math:`z^{\perp}` is orthogonal to the target factors under the
benchmark-weighted inner product and is re-standardized afterwards. Entries are
processed in insertion order such that later entries operate on exposures already 
modified by earlier ones.

When the target factors form a one-hot categorical family, the projection reduces
to demeaning within each group. Taking industry as an example, neutralizing a
style against "industry" through `neutralize_against` and scoring it within
industries through `transform_by_group="industry"` will produce the same neutrality 
because the exposure has zero benchmark-weighted mean within every industry
(:math:`D^\top W z = 0`). The resulting exposures differ only in per-industry
scaling: the projection removes the per-industry mean, while within-industry
scoring also divides each industry by its own standard deviation, keeping the
factor from being dominated by the industry with the largest spread in the raw
descriptor. For one-hot target factors, `transform_by_group` is preferred  as it both 
normalizes per-industry scale and it is cheaper (group demeaning instead of a full 
projection). When both are applied, the projection has no effect because the 
exposure already satisfies the same orthogonality condition.

.. _factor_model_diagnostics:

Exposure Diagnostics
~~~~~~~~~~~~~~~~~~~~

Exposure diagnostics assess the stability and conditioning of the factor
exposure matrix. Strong collinearity between exposures inflates the variance of
the estimated factor returns and can make attribution unstable. 
:meth:`~skfolio.prior.FactorModel.exposure_correlation` reports the
time-average pairwise correlation of exposures,
:attr:`~skfolio.prior.FactorModel.exposure_vif` the per-factor variance inflation
factors and :attr:`~skfolio.prior.FactorModel.exposure_condition_number` the
conditioning of the regression design. When zero-sum constraints are active, VIF
and condition number are computed in the reduced full-rank basis.

`exposure_correlation` accepts a `cs_weighting` parameter. The default
`"benchmark"` measures orthogonality in the metric used by neutralization,
`"identity"` (equal weighting) uses a different inner product and can show a
residual correlation of 0.1 to 0.3 even when the factor is exactly
benchmark-neutral, and `"regression"` measures multicollinearity as seen by the
WLS regression.

.. code-block:: python

    factor_model.plot_exposure_correlation(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_exposure_correlation.inc.rst

In the :ref:`example <factor_model_code_example>`, the market exposure is constant
(every asset has unit exposure), so its correlation row is uninformative and
displays as zero. Market neutrality is instead guaranteed by benchmark-weighted
centering: every exposure has a zero benchmark-weighted mean, so no factor
carries net market exposure. The volatility-beta correlation and the correlation
between non-linear size and size are zero, as expected from the neutralization.
The remaining correlations are moderate, indicating no redundant factors. The
`families` argument excludes the 44 industry factors for readability. When
included, their correlations with the style factors are zero as well, the result
of within-industry scoring through `transform_by_group="industry"`, and the
industry-industry correlations are slightly negative rather than zero. Each asset
belongs to exactly one industry, so membership in one industry rules out
membership in all others. Zero correlation would mean industry memberships are
independent, while this exclusion is a negative relationship.

.. note::

    For two one-hot exposures :math:`x` and :math:`y` with benchmark weights
    :math:`p_x` and :math:`p_y`, the product :math:`xy` is always zero, so the
    covariance :math:`\mathbb{E}[xy] - \mathbb{E}[x]\mathbb{E}[y] = -p_x p_y` is
    negative, giving a correlation of
    :math:`-\sqrt{p_x p_y / ((1-p_x)(1-p_y))}`, about -0.02 for the 44
    industries of the example when weights are similar.

:meth:`~skfolio.prior.FactorModel.plot_exposure_stability` shows the
cross-sectional correlation of each factor's exposures between observations
with `step` determining how far apart the observations are sampled  (21 by default). 
Slow-moving factors (e.g. value, size) should stay highly correlated at a monthly 
step. Fast-turnover factors (e.g. reversal, short-term momentum) reshuffle quickly 
by construction and naturally show lower stability at that horizon. It is therefore 
recommended to assess these factors with a shorter `step` (e.g. 1 to 5 days).

.. code-block:: python

    factor_model.plot_exposure_stability(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_exposure_stability.inc.rst

In the :ref:`example <factor_model_code_example>`, monthly stability stays above
0.8 for nearly all factors and dates, with short-lived dips during stress
episodes (e.g. March 2020). Stable exposures keep the risk decomposition
consistent between rebalancings and limit the turnover induced by exposure noise.

Additional plots on :class:`~skfolio.prior.FactorModel`:

* :meth:`~skfolio.prior.FactorModel.plot_exposure_vif`
* :meth:`~skfolio.prior.FactorModel.plot_exposure_condition_number`
* :meth:`~skfolio.prior.FactorModel.plot_exposure_distribution`
* :meth:`~skfolio.prior.FactorModel.plot_exposure_dispersion`

.. _factor_model_cross_sectional_regression:

Cross-Sectional Regression
--------------------------

After factor exposures are computed, the cross-sectional regression uses the
exposure tensor of shape `(n_observations, n_assets, n_factors)` together with
the asset return matrix of shape `(n_observations, n_assets)`. For each
observation, the model estimates factor returns and idiosyncratic returns by 
regressing the cross-section of asset returns on the corresponding lagged exposures.

Regression Model
~~~~~~~~~~~~~~~~

For each observation :math:`t`, factor returns solve the weighted least squares
problem:

.. math::

    \hat{f}(t) = \arg\min_{f} \sum_{i} w_i(t)\,
    \big(R_i(t) - B_i(t - \ell)^\top f\big)^2

where :math:`w_i(t)` are the weights described in
:ref:`Regression Weights <factor_model_regression_weights>`,
:math:`B_i(t - \ell)` is asset :math:`i`'s lagged exposure vector and :math:`\ell` is
`exposure_lag`. Idiosyncratic returns are the regression residuals:

.. math::

    \hat{\epsilon}_i(t) = R_i(t) - B_i(t - \ell)^\top \hat{f}(t)

The regression is performed by `cs_regressor`, defaulting to
:class:`~skfolio.linear_model.CSLinearRegression`, a weighted least-squares
estimator that solves all observations in one vectorized pass over the exposure
tensor. For robust or regularized cross-sectional estimation, skfolio allows
scikit-learn-compatible estimators (e.g. `HuberRegressor` for outlier-robust
regression) to be passed through
:class:`~skfolio.linear_model.CSLinearRegressorWrapper`, which applies the
wrapped estimator separately to each observation.

.. note::

    Unlike factor exposures, which are typically winsorized and standardized by
    the exposure estimators, asset returns enter the cross-sectional regression
    unadjusted. Cleaning return data errors is an upstream responsibility, since
    the same returns drive benchmark weights, realized performance and downstream
    optimization. Winsorizing legitimate extreme returns would break the
    reconciliation of :math:`R = B\,f + \epsilon` and understate idiosyncratic
    risk for heavy-tailed assets. Extreme observations are already moderated by
    regression weights (`regression_mcap_power` and `inv_idio_variance_weight_shrinkage`). 
    When the factor-return regression needs to further reduce the influence of extreme 
    asset returns, use a robust regressor such as `HuberRegressor` through 
    :class:`~skfolio.linear_model.CSLinearRegressorWrapper` as described above.

.. _factor_model_regression_weights:

Regression Weights
~~~~~~~~~~~~~~~~~~

Regression residuals are heteroskedastic, meaning idiosyncratic variance differs widely
across the cross-section. An unweighted regression would give excessive influence to the 
noisiest names. Under heteroskedasticity, the best linear unbiased estimator (BLUE) is the 
weighted least squares regression whose weights are proportional to inverse idiosyncratic
variance, known as generalized least squares (GLS).

Idiosyncratic variances are not observable before the regression is run. It is common practice
to approximate the inverse-variance weights with square-root market-cap weights (empirically, 
idiosyncratic variance decreases roughly as the inverse square root of market capitalization).
The `regression_mcap_power` parameter controls this weighting as a power
:math:`p` of market capitalization:

.. math::

    w_i \propto \mathrm{mcap}_i^{\,p}

with `0.5` (default) for square-root cap weighting, `0.0` for equal weighting and
`1.0` for cap weighting. Market cap moves with the return being regressed,
:math:`\mathrm{mcap}_i(t) \approx \mathrm{mcap}_i(t-1)\,(1 + R_i(t))`, and
weighting the regression at :math:`t` by caps at :math:`t` would correlate the
weights with the regressed returns and bias the estimated factor returns. 
Because of this, market caps are also lagged by `exposure_lag`.

The model also supports inverse-idiosyncratic-variance weighting with a two-step
feasible GLS. A first pass runs with the cap-based weights and then the variances
estimated from its residuals (up to :math:`t - 1` only) feed the weights of the
second pass. The regression weights never depend on their own output, avoiding
the feedback loop of recursive weighting schemes where a low estimated variance
increases an asset's weight and, in turn, its influence on later residuals.
`inv_idio_variance_max_weight_ratio` (default 20) caps each inverse-variance
weight at a multiple of the cross-sectional median, preventing assets with very
low estimated variance from dominating the regression.

Estimated variances are noisy, and `inv_idio_variance_weight_shrinkage` blends
the two weightings for robustness:

.. math::

    w_i = \lambda\,w_i^{\text{inv-var}} + (1 - \lambda)\,w_i^{\text{cap}}

where :math:`\lambda = 0` (default) uses the cap-based weights only and
:math:`\lambda = 1` the inverse-variance weights only.

The weights used at each date are stored in `factor_model_.regression_weights`: row
:math:`t` holds the weights used by the regression at :math:`t`, built from market
caps at :math:`t - \ell` and idiosyncratic variances estimated up to :math:`t - 1`.

.. _factor_model_zero_sum_constraints:

Zero-Sum Constraints
~~~~~~~~~~~~~~~~~~~~

One-hot factor families are exactly collinear with the global factor: each
asset's industry exposures sum to one, which is the global exposure. Including
both leaves the factor returns unidentified as adding a constant to every industry
return and subtracting it from the global return leaves the fit unchanged, and
the exposure design is rank-deficient.

The `constrained_families` parameter resolves this by imposing a benchmark-weighted
zero-sum constraint on the factor returns of each constrained family. Economically,
the global factor captures the benchmark portfolio return and the constrained family
factors capture relative effects around it.

Each tuple `(family, factor_to_drop)` specifies a family to reparameterize.
Instead of solving a constrained regression, the model applies an equivalent
change of basis. For a constrained family with exposure columns :math:`x_j` and
factor returns :math:`\beta_j`, the constraint reads
:math:`\sum_j w_j \beta_j = 0`, where :math:`w_j` is the benchmark weight
aggregated per factor (for one-hot industries, the benchmark weight of each
industry). Solving the constraint for one factor :math:`k` and substituting into
the regression yields an unconstrained regression on transformed features
:math:`z_j`, with the dropped factor's return reconstructed from the constraint
after fitting:

.. math::

    z_j = x_j - \frac{w_j}{w_k}\,x_k,
    \qquad
    \hat{\beta}_k = -\frac{1}{w_k} \sum_{j \neq k} w_j \hat{\beta}_j

The full family is reported with no loss of information, and all downstream
computations such as the factor covariance estimator and the 
:ref:`regression diagnostics <factor_model_regression_diagnostics>` (t-statistics, 
VIF, condition number, adjusted :math:`R^2`) operate on a full-rank design.
Any choice of :math:`k` yields the same constrained solution. When `factor_to_drop` is `None`,
the model drops the factor with the largest :math:`|w_k|`, keeping the ratios
:math:`|w_j / w_k|` bounded by 1.0 for most factors and preserving the
conditioning of the reduced design. The basis is stored in
`factor_model_.family_constraint_basis` and follows the same timing as the
regression. Realized factor returns are reconstructed with the lagged ratios used
by the regression, while expected factor returns and covariance are expanded with
the latest as-of basis.

.. _factor_model_global_factor:

Global Factor and Benchmark Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combined with the exposure centering and zero-sum constraints described in :ref:`Zero-Sum Constraints <factor_model_zero_sum_constraints>`, the
global factor return has a direct economic interpretation. Style exposures are
centered so that their benchmark-weighted average is zero (the default
:class:`~skfolio.preprocessing.CSStandardScaler` behavior) and
constrained family returns average to zero under benchmark weights, so all
non-global factors satisfy:

.. math::

    \sum_i w_i^{\text{bench}}\,B_{ij} = 0 \quad \forall\; j \neq 0

The benchmark portfolio therefore has zero exposure to every factor except the global
one, and the estimated global factor return tracks the benchmark portfolio return on
the estimation universe (e.g. the market-cap portfolio when
`benchmark_mcap_power=1`). Small deviations remain when regression weights differ
from benchmark weights. When `regression_mcap_power == benchmark_mcap_power` and
`inv_idio_variance_weight_shrinkage == 0`, the regression weights are proportional to
the benchmark weights and the identity becomes exact:

.. math::

    \hat{f}_0(t) = \sum_i w_i^{\text{bench}}\,R_i(t)

The benchmark weights are stored in `factor_model_.benchmark_weights`. Under this
structure, each factor return reads as a portfolio return relative to the
benchmark. The global factor is the benchmark return itself. An industry factor
return is a benchmark-relative industry effect meaning it captures the return earned by that industry
in excess of the benchmark, net of the other factors. A style factor return is the
return to a standardized characteristic tilt, representing the return earned by holding one
standard deviation of exposure to that characteristic while keeping all other
factor exposures at zero.

Missing Data and Changing Universes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the regression is re-estimated independently at each date, changing universes
are handled naturally, with newly listed assets joining the regression as soon as their
exposures are available, delisted assets dropping out, and assets on holiday
excluded for that date only. No realignment or imputation is needed.

At each date, an asset participates in the regression when it belongs to the
estimation universe and its lagged exposures and return are finite. All other pairs
receive zero regression weight. Assets outside the estimation universe still receive 
exposures, idiosyncratic returns and forecasts where computable.

As a guard against underspecified regressions, `min_regression_assets` sets the
minimum number of participating assets required at every observation after the
:ref:`warmup period <factor_model_warmup>` (default
`max(2 * n_factors, 30)`). A ValueError is raised when a cross-section falls below
this minimum.

.. _factor_model_currency_factors:

Currency Factors
~~~~~~~~~~~~~~~~

In a multi-currency universe, asset returns in the investor's numeraire mix equity
risk with currency risk. The model separates the two components by estimating non-currency 
factors from local excess returns and adding currency risk through dedicated currency factors.

The `currency_factor` parameter takes a one-hot exposure estimator on the asset currency
field (typically :class:`~skfolio.factor_exposure.OneHotCategoricalFactors`),
and the `currency_excess_returns` argument of `fit` supplies the currency excess return
series. The base-currency excess return of asset :math:`i` decomposes as:

.. math::

    R^{excess,base}_i(t) = R^{excess,local}_i(t) + R^{ccy}_{C_i(t)}(t)

where :math:`R^{excess,local}_i(t)` is the asset's local-currency return in
excess of the cash rate of its currency :math:`C_i(t)` and
:math:`R^{ccy}_{C_i(t)}(t)` is the currency excess return from converting that
local-currency asset return into the investor's base currency:

.. math::

    R^{ccy}_{C_i(t)}(t) = R^{FX}_{C_i(t)}(t) + r^{cash}_{C_i(t)}(t)
    - r^{cash}_{base}(t) + R^{local}_i(t)\,R^{FX}_{C_i(t)}(t)

The identity follows from compounding the local return with the FX return and
subtracting the cash rates defining each excess return (time indices omitted):

.. math::

    R^{base}_i &= (1 + R^{local}_i)(1 + R^{FX}_{C_i}) - 1
    = R^{local}_i + R^{FX}_{C_i} + R^{local}_i\,R^{FX}_{C_i} \\
    R^{excess,base}_i &= R^{base}_i - r^{cash}_{base} \\
    &= \big(R^{local}_i - r^{cash}_{C_i}\big)
    + R^{FX}_{C_i} + r^{cash}_{C_i} - r^{cash}_{base}
    + R^{local}_i\,R^{FX}_{C_i} \\
    &= R^{excess,local}_i + R^{ccy}_{C_i}

The currency excess return series are computed by the user and supplied through
`currency_excess_returns`, with one column per currency factor.

Unlike equity factor returns, currency factor returns are not estimated by the
regression but instead are observed FX series in the investor's numeraire which are then
appended directly to the factor return distribution with family `"currency"`. The factor
covariance then captures both equity and currency factor risk, and each asset loads
on its currency through the one-hot exposures.

.. _factor_model_regression_diagnostics:

Regression Diagnostics
~~~~~~~~~~~~~~~~~~~~~~

:meth:`~skfolio.prior.FactorModel.plot_factor_cumulative_returns` displays the
estimated factor returns accumulated through time:

.. code-block:: python

    fig = factor_model.plot_factor_cumulative_returns(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_factor_cumulative_returns.inc.rst

Summary statistics of the style factors from the fitted model:

.. code-block:: python

    factor_model.summary(families="style")[
        ["annualized_mean", "annualized_vol", "annualized_sharpe", "mean_vif"]
    ]

.. include:: ../_static/factor_model/tables/style_factor_summary.inc.rst

In the :ref:`example <factor_model_code_example>`, momentum carries the highest
annualized Sharpe ratio (0.62) and the largest cumulative return, with the sharp
2020 reversal characteristic of momentum crashes. The beta, size, earnings yield
and profitability factors show positive premia over the sample while value is
flat. The mean VIFs are all below 5, consistent with the moderate exposure
correlations observed in :ref:`Exposure Diagnostics <factor_model_diagnostics>`.


.. note::

    These are pure-factor returns, not sorted long-short factor portfolio returns
    such as the Fama-French factors. Their numerical scale is generally smaller
    than that of familiar academic factor portfolios. Each factor return can be
    interpreted as the return of a factor-mimicking portfolio constructed by the
    cross-sectional regression. This portfolio is not rescaled to a fixed gross
    exposure or volatility, whereas academic factors are often constructed as
    100% long and 100% short (200% gross exposure) and may carry several units
    of factor exposure together with incidental exposures to other factors.
    Factor-return series should therefore only be compared after matching
    exposure scaling and portfolio normalization conventions. Sharpe ratios are
    more comparable because rescaling changes the mean and volatility proportionally.

    In a characteristics factor model, each factor return is the cross-sectional
    regression coefficient for one unit of exposure. For style factors built from
    standardized exposures, this corresponds to one cross-sectional standard
    deviation of exposure after winsorization, within-industry scoring and
    neutralization. Equivalently, the factor-mimicking portfolio has unit
    exposure to that factor and zero exposure to the other factors.

    Factor-return signs follow the exposure convention. The size factor in this
    model is built from `LogMarketCap` with positive exposure meaning larger companies
    and negative exposure meaning smaller companies. This large-minus-small
    convention is standard in characteristics risk models and is the opposite of
    the Fama-French SMB factor, defined as small minus big. A positive size
    factor return in this model means that large-cap exposure was rewarded. The
    sign convention does not affect covariance decomposition, attribution or
    optimization because exposures and factor returns are used consistently
    within the model. It only matters when comparing to external factor series
    with different naming conventions.


:attr:`~skfolio.prior.FactorModel.cs_regression_scores` returns per-observation
fit statistics: `r2`, `adjusted_r2` (adjusted for the effective number of
regressors, reduced when constraints are active), `aic` and `bic`.

.. code-block:: python

    factor_model.cs_regression_scores.mean()

.. code-block:: text

    r2                 0.312856
    adjusted_r2        0.278087
    aic           -10183.548079
    bic            -9894.130723

:meth:`~skfolio.prior.FactorModel.plot_cs_regression_scores` displays them
through time:

.. code-block:: python

    factor_model.plot_cs_regression_scores(score="adjusted_r2", window=20)

.. include:: ../_static/factor_model/fragments/factor_model_cs_regression_scores_adjusted_r2.inc.rst

The mean :math:`R^2` is an in-sample quantity and does not account for model
complexity or overfitting, making it a weak measure of model quality. It does however
remain useful as a sanity check. For daily US equity models, the mean :math:`R^2` typically
falls between 25% and 40% and the mean adjusted :math:`R^2` between 20% and 35%.
Values outside these ranges typically warrant investigation. Three caveats apply:

* A mean :math:`R^2` of 30% does not mean the model explains only 30% of portfolio risk:
  the daily cross-sectional :math:`R^2` is an in-sample fit statistic for that
  day's stock returns, while a portfolio's risk can still be dominated by common
  factors because idiosyncratic terms diversify away and factor terms do not. 
* Some vendors report :math:`R^2` on monthly returns, which is mechanically higher than
  on daily returns, and figures are only comparable at the same frequency. 
* Finally, :math:`R^2` should not be used to compare different models as the addition of any 
  factor will lead to a larger value.

:attr:`~skfolio.prior.FactorModel.cs_regression_t_stats` returns the
per-observation t-statistic of each factor return, computed in the reduced basis
when constraints are active. The usual rule of thumb applies: :math:`|t| > 2`
suggests significance at approximately the 5% level.
:meth:`~skfolio.prior.FactorModel.plot_cs_regression_t_stats` displays them
through time.

:meth:`~skfolio.prior.FactorModel.cs_regression_t_stat_exceedance_rate`
aggregates this through time. A factor whose true coefficient is zero would
exceed the threshold about 5% of the time. Rates persistently above this
reference indicate a factor that is repeatedly significant in the cross-section.
:meth:`~skfolio.prior.FactorModel.plot_cs_regression_t_stat_exceedance_rate`
displays the exceedance rates:

.. code-block:: python

    factor_model.plot_cs_regression_t_stat_exceedance_rate(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_cs_regression_t_stat_exceedance_rate.inc.rst

.. _factor_model_risk_forecasting:

Risk Forecasting
----------------

The risk forecast combines the distribution of factor returns and the idiosyncratic risk of each asset. 
This section details their estimation and how they are assembled into the asset covariance and 
return scenarios.

.. _factor_model_factor_return_distribution:

Factor Return Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

The estimated factor return time series is passed to `factor_prior_estimator`, a
:ref:`prior estimator <prior>` that produces the expected factor returns (factor
premia), the factor covariance and factor return scenarios. When zero-sum constraints
are present, the estimation runs in the reduced full-rank basis, making the factor
covariance positive definite by construction.

The default is :class:`~skfolio.prior.EmpiricalPrior` with
:class:`~skfolio.moments.EWMu` for expected returns and
:class:`~skfolio.moments.RegimeAdjustedEWCovariance` for covariance. The latter
addresses two known weaknesses of plain exponentially weighted covariance:

* It applies a scalar regime multiplier (Short-Term Volatility Update) that improves
  risk calibration when volatility regimes change faster than the EWMA half-life can
  track.
* It supports separate half-lives for variance and correlation. Empirically,
  volatility mean-reverts faster than correlation: a shorter variance half-life adapts
  quickly to volatility shifts while a longer correlation half-life reduces
  estimation noise on co-movements.

It also supports optional Newey-West HAC correction through `hac_lags` to adjust for
serial correlation in factor returns.

Because the factor return series is low-dimensional (:math:`K \ll N`), estimators
that would be unstable or expensive on thousands of assets are cheap and reliable on
factors. The factor prior is fully replaceable, e.g. covariance shrinkage or denoising can
be applied at the factor level, and views on factors can be expressed by using
:class:`~skfolio.prior.EntropyPooling` or :class:`~skfolio.prior.OpinionPooling` as
`factor_prior_estimator`. The factor return scenarios produced here feed
scenario-based risk measures (e.g. CVaR) downstream.

:meth:`~skfolio.prior.FactorModel.plot_factor_forecast_correlation` shows the
factor return correlation forecast from the fitted factor covariance. High values
flag factors whose returns move together and carry overlapping risk. Industry
factors are omitted below for readability:

.. code-block:: python

    factor_model.plot_factor_forecast_correlation(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_factor_forecast_correlation.inc.rst

In the :ref:`example <factor_model_code_example>`, most correlations are
moderate, indicating that the factors capture distinct risk dimensions. The 0.80 market-beta
correlation is the known exception and is accepted. The beta factor return is the
reward for holding high-beta names over low-beta names, a spread that widens in
rising markets and reverses in falling markets, making it co-move with the market return
by construction. The two factors remain separated in the regression because their
exposures are near-orthogonal in the cross-section, and the co-movement of their
returns is captured by the factor covariance.

:meth:`~skfolio.prior.FactorModel.plot_factor_forecast_volatilities` shows each
factor's annualized volatility forecast which is calculated as the square root 
of the factor covariance diagonal. It ranks the factors by their standalone risk 
contribution:

.. code-block:: python

    factor_model.plot_factor_forecast_volatilities(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_factor_forecast_volatilities.inc.rst

The market factor dominates at close to 20% annualized volatility. Its return is
the benchmark return, while style factor returns are long-short spreads across
standardized exposures. The beta, momentum, size and liquidity factors form the
next tier around 5% to 8%, and the remaining styles sit near 2%.


Idiosyncratic Risk
~~~~~~~~~~~~~~~~~~

Per-asset idiosyncratic variances are estimated from the idiosyncratic returns by
`idio_variance_estimator`, defaulting to
:class:`~skfolio.moments.RegimeAdjustedEWVariance`. The estimator must support
`partial_fit` so the model can recover per-asset variance estimates at each
observation, stored in `factor_model_.idio_variances`. Assets still in their
:ref:`warmup period <factor_model_warmup>` have NaN variances, which propagate
to the fitted moments and mark them as not yet investable.

A factor model assumes the factor structure captures all common risk, leaving
idiosyncratic returns uncorrelated across assets. The idiosyncratic covariance is
therefore diagonal by default. In practice, linked securities remain correlated
after the factor structure is removed (e.g. multiple share classes, ADRs versus
ordinary shares, or dual listings). Without correction, optimizers treat such
pairs as diversified sources of idiosyncratic risk and may over-allocate to them.

Correlation thresholding addresses these cases. When `idio_corr_threshold` is set to
:math:`\tau > 0`, the `idio_corr_estimator` (defaulting to
:class:`~skfolio.moments.EWCovariance`) is fitted on idiosyncratic returns
standardized by their contemporaneous idiosyncratic volatility. Only the correlation
component of its output is retained: off-diagonal correlations with
:math:`|\rho_{ij}| \le \tau` are set to zero, and the surviving correlations are
recombined with the latest per-asset variances to form a sparse idiosyncratic
covariance. Variances and correlations are estimated separately because a single full
covariance estimator would mix per-asset variance estimation with off-diagonal
correlation noise. This keeps variances driven by `idio_variance_estimator` and
applies the correlation overlay only where residual correlations are large enough to
retain.

Persistent residual correlation across a broader group, such as sub-industry
peers under a coarse industry classification, signals a missing factor rather
than a thresholding problem. Adding the factor keeps the idiosyncratic covariance
sparse, while lowering :math:`\tau` to absorb the group reintroduces the
estimation noise the diagonal assumption avoids.

Idiosyncratic Risk Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These diagnostics test the idiosyncratic volatility forecasts through the
standardized idiosyncratic returns :math:`z_{it} = u_{it} / \hat\sigma_{it}`. Under
correct calibration, :math:`z` has cross-sectional standard deviation 1.0.

:meth:`~skfolio.prior.FactorModel.idio_calibration_summary` aggregates the main
statistics:

* `mean_cs_std` close to 1.0 indicates correctly scaled idiosyncratic risk. Values
  persistently above 1.0 indicate underestimated risk and persistently below 1.0
  indicate overestimated risk.
* `mean_tail_rate_3sigma` is the fraction of standardized returns beyond three
  standard deviations. The Gaussian reference is 0.27%, and values of 1% to 3% are
  common for equity factor models due to fat tails.
* `mean_cs_excess_kurtosis` above zero and a moderate `mean_cs_skewness` are typical.

Two complementary statistics separate ranking power from calibration.
:attr:`~skfolio.prior.FactorModel.idio_vol_ic` is the Spearman correlation between
predicted volatility and the next-period absolute idiosyncratic return with high
values meaning that the model ranks cross-sectional volatility differences well.
:attr:`~skfolio.prior.FactorModel.idio_vol_residual_dependence` is the same
correlation after standardizing the next-period return by the predicted
volatility: under correct calibration it should be close to 0. The desirable
pattern is a high `idio_vol_ic` combined with residual dependence near 0.

:meth:`~skfolio.prior.FactorModel.plot_idio_calibration` tracks the
cross-sectional standard deviation of the standardized returns through time:

.. code-block:: python

    factor_model.plot_idio_calibration(window=20)

.. include:: ../_static/factor_model/fragments/factor_model_idio_calibration.inc.rst

In the :ref:`example <factor_model_code_example>`, the series oscillates around a
mean of 1.06, a slight average underestimation of idiosyncratic risk. The spike
above 1.5 at the COVID shock shows realized dispersion outrunning the forecasts,
followed by a dip below 0.7 as the variance estimator caught up while volatility
dissipated.

:meth:`~skfolio.prior.FactorModel.plot_idio_vol_ic` displays the volatility rank
IC through time:

.. code-block:: python

    factor_model.plot_idio_vol_ic()

.. include:: ../_static/factor_model/fragments/factor_model_idio_vol_ic.inc.rst

The rolling mean holds near 0.4 across the sample, which identifies a strong and stable
ranking of cross-sectional volatility differences. Combined with the calibration series
close to 1.0, we can conclude that the idiosyncratic risk forecasts are both well ordered
and well scaled.

Additional plots on :class:`~skfolio.prior.FactorModel`:

* :meth:`~skfolio.prior.FactorModel.plot_idio_tail_rate`
* :meth:`~skfolio.prior.FactorModel.plot_idio_kurtosis`
* :meth:`~skfolio.prior.FactorModel.plot_idio_skewness`
* :meth:`~skfolio.prior.FactorModel.plot_idio_vol_residual_dependence`

.. _factor_model_asset_covariance_forecast:

Asset Covariance Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~

The asset covariance forecast assembles the pieces estimated above:

.. math::

    \Sigma = B(T)\,F\,B(T)^\top + D

where :math:`B(T)` is the latest loading matrix, :math:`F` the factor covariance
and :math:`D` the idiosyncratic covariance. The result is positive definite and
well conditioned by construction as the systematic part is low-rank positive
semidefinite and :math:`D` is positive diagonal (or sparse positive definite).

Asset return scenarios follow the same construction. Factor scenarios from the
factor prior are mapped through the latest loading matrix, and idiosyncratic
scenarios calibrated to the latest idiosyncratic risk forecast are added on top.
Optimizers using scenario-based risk measures (e.g. CVaR) therefore see both
return components.

skfolio exploits this structure when passing the asset covariance to downstream
optimizers. Portfolio variance splits into a factor contribution and an
idiosyncratic contribution:

.. math::

    w^\top \Sigma\, w
    = \lVert F^{1/2} B^\top w \rVert^2 + \lVert D^{1/2} w \rVert^2

Both terms involve only small matrices: the :math:`n \times K` loading matrix,
the :math:`K \times K` factor covariance and the :math:`n` idiosyncratic
variances. skfolio's convex optimizers build their risk constraints on these
directly, through `factor_model_.covariance_sqrt`, instead of assembling and
factorizing the dense :math:`n \times n` covariance. On a universe of thousands
of assets driven by a few dozen factors, this keeps the risk constraints small
and the optimization fast.

.. _factor_model_covariance_forecast_evaluation:

Covariance Forecast Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In-sample fit says little about forecast quality, so the covariance forecast is
evaluated out of sample with
:func:`~skfolio.model_selection.covariance_forecast_evaluation` or, using online
learning for speed, :func:`~skfolio.model_selection.online_covariance_forecast_evaluation`.
Both walk forward through the data, compare each covariance forecast with the
subsequently realized returns over an evaluation window and return a
:class:`~skfolio.model_selection.CovarianceForecastEvaluation` with summary
statistics and plots. Four diagnostics are computed:

* The **Mahalanobis calibration ratio** tests the full covariance structure across
  all eigenvalue directions. The target is 1.0, with values above 1.0 indicating
  underestimated risk and below 1.0 indicating overestimated risk.
* The **diagonal calibration ratio** applies the same test to individual asset
  variances, ignoring correlations.
* The **portfolio bias statistic** tests covariance calibration along
  user-supplied portfolio directions. For each test portfolio, realized
  portfolio returns are divided by their forecast volatility, and the standard
  deviation of that standardized series should be 1.0 under correct
  calibration. Evaluating several representative portfolios can reveal
  direction-specific under or over-estimated risk.
* The **portfolio QLIKE** scores portfolio variance forecasts, with lower values
  indicating better forecasts.

The figures below evaluate the model with
:func:`~skfolio.model_selection.online_covariance_forecast_evaluation` and
`test_size=5`:

.. code-block:: python

    from skfolio.model_selection import online_covariance_forecast_evaluation

    evaluation = online_covariance_forecast_evaluation(
        model,
        X,
        params={"characteristics": characteristics},
        warmup_size=2 * 252 + 21,
        test_size=5,
    )
    evaluation.plot_calibration()

.. include:: ../_static/factor_model/fragments/covariance_eval_calibration.inc.rst

In the :ref:`example <factor_model_code_example>`, the diagonal ratio and the bias
statistic oscillate around the target, implying individual asset variances and
test-portfolio volatilities are well scaled. The Mahalanobis ratio stays near 1.5,
indicating that the remaining underestimation is concentrated in the covariance's
low-variance directions. The Mahalanobis distance gives more weight to errors in
those directions, while the diagonal ratio and portfolio bias statistic are less
sensitive to them. All three ratios spike at the 2020 shock, then fall below
1.0 through 2021 as the forecasts lag the post-crisis decline in volatility.

The summary table aggregates the four diagnostics over the full evaluation
period:

.. code-block:: python

    evaluation.summary()

.. include:: ../_static/factor_model/tables/covariance_evaluation_summary.inc.rst

Additional plots on
:class:`~skfolio.model_selection.CovarianceForecastEvaluation`:

* :meth:`~skfolio.model_selection.CovarianceForecastEvaluation.plot_qlike_loss`
* :meth:`~skfolio.model_selection.CovarianceForecastEvaluation.plot_exceedance`

:class:`~skfolio.model_selection.CovarianceForecastComparison` runs the same
evaluation over several models and aggregates the results for side-by-side
comparison. Missing data follows the panel conventions: only finite observations
contribute and inactive assets are excluded.

The comparison below contrasts two
:class:`~skfolio.moments.RegimeAdjustedEWCovariance` regime half-lives for the
factor prior: `regime_half_life=month` (21 trading days) against
`regime_half_life=quarter` (63 trading days), which reacts more slowly to
volatility regime shifts.

.. code-block:: python

    from skfolio.model_selection import CovarianceForecastComparison

    comparison = CovarianceForecastComparison(
        [eval_month, eval_quarter],
        names=["regime_half_life=month", "regime_half_life=quarter"],
    )
    comparison.plot_calibration(diagnostics=("bias",))

.. include:: ../_static/factor_model/fragments/covariance_cmp_calibration.inc.rst

.. code-block:: python

    comparison.plot_qlike_loss()

.. include:: ../_static/factor_model/fragments/covariance_cmp_qlike_loss.inc.rst

The shorter month half-life adapts faster, its bias statistic overshoots less at
the 2020 shock, recovers sooner from the 2021 over-forecast and its QLIKE loss is
lower through the 2020-2021 stress. In calm periods the two models are nearly
indistinguishable, and the summary table shows similar aggregate diagnostics, with
the month half-life slightly lower on mean QLIKE.

.. code-block:: python

    comparison.summary()

.. include:: ../_static/factor_model/tables/covariance_comparison_summary.inc.rst


Additional plots on
:class:`~skfolio.model_selection.CovarianceForecastComparison`:

* :meth:`~skfolio.model_selection.CovarianceForecastComparison.plot_calibration`
  with full diagnostics (Mahalanobis, diagonal, bias)
* :meth:`~skfolio.model_selection.CovarianceForecastComparison.plot_exceedance`

.. _factor_model_expected_returns:

Expected Returns
----------------

The model is not only a risk model but also estimates expected asset returns
used by downstream optimizers. Two components contribute here: expected factor returns
(factor premia), estimated by the factor prior and mapped to assets through their
exposures, and an optional asset-level alpha forecast built from quantities the
pipeline has already estimated (idiosyncratic returns, factor exposures,
idiosyncratic variances). This section covers both components, how the alpha
forecast is decomposed and combined with the factor premia, and the diagnostics
used to evaluate the results.

Expected Factor Returns
~~~~~~~~~~~~~~~~~~~~~~~

By default (`alpha_estimator=None`), expected asset returns are determined
entirely by the factor premia:

.. math::

    \mu = B(T)\,\mu_f

where :math:`\mu_f` holds the expected factor returns estimated by
`factor_prior_estimator` (see :ref:`Factor Return Distribution
<factor_model_factor_return_distribution>`) and :math:`B(T)` is the latest
loading matrix. Each asset's expected return is the sum of the premia of the
factors it is exposed to, weighted by its exposures.

Information Coefficient
~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~skfolio.prior.FactorModel.exposure_ic_summary` measures the
cross-sectional correlation between factor exposures at :math:`t` and the forward
mean asset return from :math:`t+1` to :math:`t+h`, where :math:`h` is the
`horizon` parameter. The default `correlation_method` computes the Spearman rank
IC, with Pearson IC weighted by the regression weights as the alternative. The
summary reports `mean_ic`, `std_ic`, `ic_ir` (mean over standard deviation) and
`hit_rate` per factor.
:meth:`~skfolio.prior.FactorModel.plot_cumulative_exposure_ic` shows the
cumulative IC through time:

.. code-block:: python

    factor_model.plot_cumulative_exposure_ic(families=["market", "style"])

.. include:: ../_static/factor_model/fragments/factor_model_cumulative_exposure_ic.inc.rst


.. code-block:: python

    factor_model.exposure_ic_summary(families=["market", "style"])

.. include:: ../_static/factor_model/tables/factor_model_exposure_ic_summary.inc.rst


Daily ICs are small in absolute value, and the persistence of their sign matters
more than their level. In the :ref:`example <factor_model_code_example>`,
momentum and profitability accumulate positive IC steadily across the sample
(hit rates of 56% and 55%), while volatility, liquidity and value accumulate
negative IC.

The IC quantifies return-predictive power. In a risk model, factors are designed
to forecast covariance, not expected returns. A factor can
therefore be an excellent risk factor with an IC near zero, and a low IC is not
a reason to discard it. IC is mainly useful for evaluating alpha signals and
factor premia, not for deciding whether a factor should remain in a risk model.

.. _factor_model_alpha:

Alpha Estimators
~~~~~~~~~~~~~~~~

The `alpha_estimator` parameter accepts any :class:`~skfolio.alpha.BaseAlpha`
estimator producing asset-level expected returns. Before it is fitted, the factor
model enriches the panel with the quantities it has already estimated:
idiosyncratic returns, idiosyncratic variances, regression weights, benchmark
weights and the factor exposure tensor. In typical alpha research workflows,
idiosyncratic returns serve as the prediction target (typically after
cross-sectional transformation), idiosyncratic variances scale the target and
factor exposures neutralize the features. Targeting idiosyncratic returns rather
than raw returns removes the factor-driven component from the target. The
cross-sectional variation of raw returns includes each asset's factor exposures
times the factor returns, so a signal correlated with the exposures would pick
up factor premia already captured by the factor model. The
idiosyncratic target also carries less noise, since common factor volatility is
removed.

The alpha forecast should be expressed in expected-return units when it is combined
with expected factor returns or used in an optimization alongside return-denominated 
quantities (e.g. transaction costs, turnover constraints, return targets). Unitless 
cross-sectional scores are appropriate only when the downstream objective treats 
them as ordinal signals.

skfolio provides alpha estimators following the same signal pipeline as
factor exposures with descriptors being transformed into cross-sectional scores
(`outlier_transformer`, `scoring_transformer`, `transform_by_group`), optionally
neutralized against factor exposures (`neutralize_against`) and re-scored. The
estimators differ mainly in how the scored descriptors are combined and how the
result is scaled into expected-return units.


* :class:`~skfolio.alpha.FixedWeightedAlpha` combines multiple descriptors
  with fixed signed weights and a fixed `forecast_scale`. The weights define the
  direction and relative contribution of each descriptor, while `forecast_scale`
  converts one unit of composite score into the selected `forecast_unit`. This is useful
  when the signal combination is specified outside the model and the main decision is
  how strongly it should affect expected returns. `forecast_scale` requires careful
  calibration as a value too large leads the optimizer to over-allocate to the alpha
  forecast, whereas a value which is too small leaves the signal with little effect after
  costs, risk limits and turnover constraints.

* :class:`~skfolio.alpha.EWSharpeOptimalAlpha` combines descriptors linearly and
  estimates their coefficients with exponentially weighted least squares on forward
  idiosyncratic returns, using inverse idiosyncratic variance as regression weights.
  It learns both sign and scale from realized data and accounts for cross-descriptor
  correlations, scaling the signal blend according to its estimated idiosyncratic
  return payoff. `forecast_scale` is then applied to the learned return-unit
  forecast as a final strength multiplier. The learned coefficients are subject to
  sampling error: short half-lives, noisy targets, weak descriptors and descriptors
  whose predictive content is already absorbed by the factor model produce unstable
  coefficients that can degrade an otherwise useful raw signal.

* :class:`~skfolio.alpha.PredictorAlpha` wraps a user-provided
  scikit-learn regressor and treats each observation-asset pair as one training sample.
  It supports nonlinear interactions and non-additive signal effects, for example with
  tree-based models or regularized regressors. With `calibrate_to_return_units=True`,
  the raw predictor output is calibrated to expected-return units by exponentially
  weighted least squares, so `alpha_` remains usable alongside factor premia.
  `forecast_scale` is applied after this optional calibration. This flexibility
  increases the need for robust validation, because nonlinear models can fit noise,
  require more data and be sensitive to target construction and cross-validation
  design.

All three estimators produce `alpha_` in expected-return units. The
`forecast_unit` parameter controls the unit of the intermediate forecast
(:class:`~skfolio.alpha.ForecastUnit`). With `ForecastUnit.IDIO_RETURN` (default),
the forecast is interpreted directly as expected idiosyncratic return. With
`ForecastUnit.IDIO_SHARPE`, the forecast is interpreted as idiosyncratic Sharpe and
multiplied by the current idiosyncratic volatility before being passed to the
factor model. The Sharpe unit is preferable when a signal is expected to rank
risk-adjusted opportunities rather than raw returns. `forecast_scale` is the
common final multiplier controlling alpha strength.

The following example combines two reversal descriptors, :class:`return on assets <skfolio.descriptor.ReturnOnAssets>`
and :class:`Amihud illiquidity <skfolio.descriptor.EWAmihudIlliquidity>` with fixed signed weights and utilises a Gaussian rank scorer
for the cross-sectional scoring:

.. code-block:: python

    from skfolio.alpha import FixedWeightedAlpha
    from skfolio.descriptor import EWAmihudIlliquidity, ReturnOnAssets, Reversal
    from skfolio.preprocessing import CSGaussianRankScaler

    alpha_estimator = FixedWeightedAlpha(
        descriptors=[
            ("reversal_5d", Reversal(window=5)),
            ("reversal_21d", Reversal(window=21)),
            ("return_on_assets", ReturnOnAssets()),
            ("amihud_illiquidity", EWAmihudIlliquidity()),
        ],
        weights=[1.0, 1.0, -1.0, -1.0],
        forecast_scale=0.0001,
        scoring_transformer=CSGaussianRankScaler(),
        n_jobs=-1,
    )

In practice, the simplest estimator that matches the research assumption is usually
the most robust starting point. Use :class:`~skfolio.alpha.FixedWeightedAlpha`
when signal direction and relative weights are already specified and the main
decision is scale. Use :class:`~skfolio.alpha.EWSharpeOptimalAlpha` when
the signal combination is expected to be approximately linear and there is enough
history to estimate stable payoffs. Use :class:`~skfolio.alpha.PredictorAlpha`
when nonlinear effects are important and the additional validation burden is
acceptable. Custom estimators are created by subclassing
:class:`~skfolio.alpha.BaseAlpha`. During the alpha estimator's
:ref:`warmup period <factor_model_warmup>`, its forecast is treated as zero.

.. _factor_model_spanned_alpha:

Spanned and Orthogonal Alpha
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitted alpha forecast is decomposed into a factor-spanned part and an
orthogonal part by projecting it onto the factor exposure space with a weighted
cross-sectional regression, using the latest exposures and regression weights:

.. math::

    \alpha = \alpha^{\parallel} + \alpha^{\perp}
    \qquad \text{with} \qquad
    \alpha^{\parallel} = B(T)\,g

where :math:`g` is the vector of expected factor returns that reproduces the spanned
alpha and :math:`\alpha^{\perp}` is the orthogonal residual, the asset-specific
part of the forecast that the factor exposures cannot explain.

This gives two estimates of expected factor returns: :math:`\mu_f`, estimated
from the factor return time series (see :ref:`Factor Return Distribution
<factor_model_factor_return_distribution>`), and :math:`g`, obtained by
projecting the alpha forecast onto the factor exposure space.
`spanned_alpha_shrinkage` blends these two sources:

.. math::

    \mu^{\text{span}} =
    (1 - \lambda)\,\mu^{\text{span}}_{\text{alpha}}
    + \lambda\,\mu^{\text{span}}_{\text{factor}}

where :math:`\lambda = 1` (default) keeps only the time-series factor premia,
:math:`\lambda = 0` keeps only the alpha-implied premia and intermediate values
blend the two. With the default, the alpha forecast contributes to expected
returns only through its orthogonal part.

The orthogonal part is shrunk towards zero by `orthogonal_alpha_confidence`:

.. math::

    \mu = \mu^{\text{span}} + c\,\mu^{\perp}

where :math:`c = 1` (default) uses the orthogonal alpha as-is and :math:`c = 0`
discards it. Orthogonal directions are penalized only through idiosyncratic
variances in the covariance forecast, so an optimizer allocates to them
aggressively when they carry alpha. Reducing `orthogonal_alpha_confidence`
tempers this incentive when confidence in the forecast is limited.
:ref:`Orthogonal Space Regularization
<factor_model_orthogonal_space_regularization>` covers this behavior and the
optimizer-level alternatives.

The orthogonal alpha is stored in `factor_model_.idio_mu`. When currency factors
are present, direct currency expected returns are added to :math:`\mu`.

Alpha and Risk Factor Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A signal may closely resemble an existing risk-model factor while using a modified
definition that produces a stronger expected return. The key question is whether
the modified definition also provides a better representation of systematic risk.

If it explains common return variation better or improves risk forecasts, it should
replace the existing factor definition. Otherwise, the additional systematic
component would remain in idiosyncratic returns, causing the optimizer to underestimate
its risk.

If the modified definition improves expected returns but does not improve the risk
model, the existing risk factor should remain. The signal can then be decomposed
into a component spanned by the risk model and an orthogonal component. The
orthogonal component represents diversifiable alpha relative to the validated risk
model, so allocating to it is intentional.

A historical example comes from momentum. Earlier commercial risk models measured
momentum over the most recent 12 months, including the latest month. Later
research separated medium-term momentum from short-term reversal by excluding
that latest month. A manager using the revised definition while retaining the older risk
model created a mismatch: the difference between the two definitions appeared
outside the modelled momentum factor and was treated as idiosyncratic risk. The optimizer
could therefore take a large position in that difference without accounting for its
systematic risk.

With skfolio, the definition can be tested directly and, when it provides a better
representation of systematic risk, used immediately in the risk model.

.. _factor_model_orthogonal_space_regularization:

Orthogonal Space Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The factor structure has an asymmetric effect on the covariance forecast. Systematic
directions carry the full factor covariance, while directions orthogonal to the
factor span are penalized only through the per-asset idiosyncratic variances, since
the idiosyncratic covariance is diagonal or sparse. To an optimizer, orthogonal
directions therefore appear cheap in risk, and any orthogonal alpha makes them
attractive, leading to concentrated allocations in the orthogonal space.

If the model were correctly specified, complete and free of estimation error, this
behavior would be desirable as factor-neutral strategies (e.g. statistical arbitrage)
would exploit these directions. In practice the model is neither complete nor
error-free, so orthogonal risk is understated and some regularization is needed,
without giving up the orthogonal space entirely.

skfolio provides three mechanisms to achieve this. The first is a
:class:`~skfolio.prior.CharacteristicsFactorModel` parameter, the other two are
configured at the optimization step:

* `orthogonal_alpha_confidence` shrinks the orthogonal alpha point estimate toward
  zero, reducing the incentive to allocate in orthogonal directions.
* :class:`~skfolio.uncertainty_set.OrthogonalMuUncertaintySet`, passed to the
  optimizer's `mu_uncertainty_set_estimator`, applies robust optimization to the
  expected returns in the orthogonal space, penalizing allocations proportionally
  to the uncertainty of the orthogonal alpha.
* :class:`~skfolio.uncertainty_set.OrthogonalCovarianceUncertaintySet`, passed to
  the optimizer's `covariance_uncertainty_set_estimator`, inflates the covariance
  in orthogonal directions, raising their variance directly.

An optimizer can allocate in orthogonal directions even when the orthogonal alpha
is zero as binding constraints (e.g. factor-neutrality, long-only) act through
shadow prices and push weights into the orthogonal space. In a model without an
alpha estimator, such allocations carry idiosyncratic risk with no expected
reward, making the covariance-side regularization relevant beyond the alpha term
itself. The :ref:`Portfolio Construction <factor_model_portfolio_construction>`
section shows the optimizer-level configuration, and the regularization strength
(`radius`) can be selected by walk-forward evaluation or
:ref:`hyperparameter tuning <factor_model_hyper_parameter_tuning>`.

Alpha Forecast Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Alpha research can be organized in two ways. The alpha estimator can either be developed
jointly with the factor model, attached through `alpha_estimator` and evaluated
end to end or it can be developed independently by first fitting a factor model without
an alpha estimator and then adding its fitted outputs to the panel with
:meth:`~skfolio.prior.FactorModel.enrich_asset_panel` and finally iterating over
the alpha estimator directly using the enriched panel. The independent workflow avoids
refitting the factor model at each research iteration and is used below.

The :func:`~skfolio.alpha.alpha_forecast_evaluation` function evaluates the
historical forecasts produced by an alpha estimator against a forward target
field in an :class:`~skfolio.containers.AssetPanel`. The default target is
`idio_returns`, the part of returns not explained by the factor model. The
function fits the estimator with `fit_transform`, then compares the alpha
forecasts at observation :math:`t` with the forward mean target over
:math:`[t + \ell, t + \ell + h)`, where :math:`h` is `holding_period` and
:math:`\ell` is `signal_lag`:

.. code-block:: python

    from skfolio.utils.stats import CSWeighting
    from skfolio.alpha import alpha_forecast_evaluation

    characteristics_enriched = factor_model.enrich_asset_panel(characteristics)

    evaluation = alpha_forecast_evaluation(
        alpha_estimator,
        characteristics_enriched,
        holding_period=5,
        signal_lag=1,
        cs_weighting=CSWeighting.REGRESSION,
        quantiles=(0.1, 0.25),
    )

    evaluation.ic_summary()

.. include:: ../_static/factor_model/tables/alpha_eval_ic_summary.inc.rst


.. code-block:: python

    evaluation.portfolio_summary()

.. include:: ../_static/factor_model/tables/alpha_eval_portfolio_summary.inc.rst

The example alpha is deliberately simple. Its Spearman IC averages 0.012 with an
ICIR of 0.16 (t-stat 3.97) and a hit rate of 56%: a genuine but modest
predictive signal, in line with typical daily alpha signals. The simple
portfolios earn an annualized information ratio above 1.7 with substantial
turnover, gross of any trading friction. Whether such a signal can be monetized
depends on transaction costs, borrow costs, market impact and turnover
constraints, which enter at the optimization step and are covered in
:ref:`Portfolio Construction <factor_model_portfolio_construction>`.

All diagnostics are computed on the final alpha forecast returned by the
estimator, after any rank transformation. With
`scoring_transformer=CSGaussianRankScaler()`, `spearman_ic` measures the ordering
quality of the forecast and `pearson_ic` the linear relation between
Gaussian-rank scores and future target returns. `zscore_weighted_portfolio` is
the simple portfolio closest to the expected-return vector consumed by an
optimizer, since the optimizer receives alpha values proportional to those
scores.

The :class:`~skfolio.alpha.AlphaForecastEvaluation` result groups the diagnostics
as follows:

* :meth:`~skfolio.alpha.AlphaForecastEvaluation.ic_summary`: `spearman_ic`
  measures ordering quality and is invariant to monotonic transformations of the
  forecast. `pearson_ic`, weighted by `cs_weighting`, measures whether forecast
  magnitudes are linearly related to realized targets.
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.portfolio_summary`: annualized
  statistics of alpha-only, gross-normalized long-short portfolios built from
  centered ranks (`rank_weighted_portfolio`) or centered forecast values
  (`zscore_weighted_portfolio`), before covariance, costs and constraints are
  introduced by the optimizer.
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.quantile_summary`: annualized
  top-minus-bottom target returns per tail quantile, showing whether predictive
  content is concentrated in the tails.
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.calibration_summary`:
  `calibration_slope` is the scale multiplier from a weighted zero-intercept
  regression of realized target on forecast. A slope near 1.0 indicates the
  forecast is already scaled to target units. Values above 1.0 indicate that
  `forecast_scale` is too small, while values below 1.0 indicate that it is too
  large.
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.coverage_summary`: fraction and
  number of assets with finite forecast and target values. Low coverage can make
  the other statistics unstable even when their averages look acceptable.
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.factor_correlation_summary`:
  contemporaneous cross-sectional correlation between the forecast and factor
  exposures, testing whether the forecast is neutral to existing factors.
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.decay_summary` and
  :meth:`~skfolio.alpha.AlphaForecastEvaluation.holding_period_summary`: IC and
  simple-portfolio statistics across forward periods, showing whether the signal
  is short-lived, persistent or delayed relative to the selected holding horizon.

Together, these diagnostics separate ordering quality from magnitude quality:
`spearman_ic` and `rank_weighted_portfolio` isolate the ordering, while
`pearson_ic`, `zscore_weighted_portfolio` and `calibration_summary` evaluate the
forecast values that an optimizer receives. `coverage_summary` checks data
availability and `decay_summary` aligns the alpha horizon with the intended
rebalancing and holding period. When the rank-based diagnostics are stronger
than the magnitude-based ones, the optimizer should not receive raw alpha
magnitudes as if they were reliable expected-return intensities.

.. code-block:: python

    evaluation.plot_cumulative_ic()

.. include:: ../_static/factor_model/fragments/alpha_eval_cumulative_ic.inc.rst

.. code-block:: python

    evaluation.plot_factor_correlation()

.. include:: ../_static/factor_model/fragments/alpha_eval_factor_correlation.inc.rst

.. code-block:: python

    evaluation.plot_cumulative_returns()

.. include:: ../_static/factor_model/fragments/alpha_eval_cumulative_returns.inc.rst

Both ICs accumulate steadily across the sample, with the strongest run in
2020-2021 and no prolonged negative stretch, indicating stable predictive power
rather than a few favorable periods. The factor correlation plot shows the
forecast is not fully factor-neutral as it carries a positive size correlation and
negative profitability and liquidity correlations, inherited from its
descriptors. If these tilts are unwanted, they can be removed with
`neutralize_against`. The simple portfolios compound consistently, with a sharp
drawdown and recovery around the COVID shock.

Additional plots on :class:`~skfolio.alpha.AlphaForecastEvaluation`:

* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_rolling_ic`
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_quantile_returns`
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_calibration`
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_ic_by_holding_period`
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_portfolio_by_holding_period`
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_ic_decay`
* :meth:`~skfolio.alpha.AlphaForecastEvaluation.plot_portfolio_decay`

When the diagnostics show reliable ordering but weak magnitude calibration, a
practical approach is to keep the alpha shape rank-based and control its strength
with a single scale parameter. For descriptor-composition estimators, this is
done with a rank-based `scoring_transformer` (e.g.
:class:`~skfolio.preprocessing.CSGaussianRankScaler`), as in the
:class:`~skfolio.alpha.FixedWeightedAlpha` example of the
:ref:`Alpha Estimators <factor_model_alpha>` section. The resulting alpha is
still in expected-return units:

.. math::

    \alpha_i = s \, z_i

where :math:`z_i` is the cross-sectional Gaussian rank score and :math:`s` is
`forecast_scale`. The rank transformation defines the relative shape of the forecast,
while `forecast_scale` maps one rank-normal score unit to expected-return units.

This unit conversion is important in mean-risk optimization as the optimizer trades off
expected return, risk, constraints and transaction costs in the same objective. If
transaction costs are expressed in return units, the alpha forecast must also be in
return units. A rank-based alpha satisfies this requirement once it is multiplied by
`forecast_scale`.

The calibration problem is therefore not rank versus cost, but scale versus cost. If
`forecast_scale` is too small, transaction costs and risk penalties may dominate the
signal and the optimizer will keep positions close to their starting weights. If it is 
too large, the optimizer may overtrade the ranked signal. The `calibration_summary`, 
`portfolio_summary` and `decay_summary` diagnostics help choose a scale that is 
consistent with realized target returns and the expected holding horizon.


.. _factor_model_portfolio_construction:

Portfolio Construction
----------------------

The factor model is a prior estimator and can be passed to any skfolio optimizer
through the `prior_estimator` parameter. The optimizer fits the
prior internally, then consumes its expected returns, covariance and scenarios.
The `characteristics` panel is forwarded to the prior through scikit-learn
metadata routing.

This section builds two portfolios on the fitted model: a factor-constrained
portfolio that trades factor premia through explicit exposure targets, and a
factor-neutral alpha portfolio that allocates to the orthogonal alpha component.

Factor-Constrained Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example below builds a dollar-neutral long-short portfolio with positive
momentum and profitability exposures, together with a negative exposure to the
non-linear size factor. The exposure levels are chosen for illustration so their
effects remain clear in the later attribution analysis.

The -2.0 target for `non_linear_size` shows how factor constraints can express
more complex patterns than a simple large-versus-small tilt. After removing its
linear size component, the factor produces the following approximate tilts:

.. list-table::
   :header-rows: 1
   :widths: 2 1

   * - Size region
     - Portfolio tilt
   * - Very small
     - Long
   * - Moderately small
     - Short
   * - Moderately large
     - Long
   * - Very large
     - Short

In skfolio, covariance uncertainty sets are applied to variance. With `MAXIMIZE_RATIO`
and no additional objective penalty, minimizing variance or standard deviation
produces the same maximum Sharpe ratio portfolio.

.. code-block:: python

    from sklearn import set_config
    from skfolio import RiskMeasure
    from skfolio.optimization import MeanRisk, ObjectiveFunction

    set_config(enable_metadata_routing=True)

    X = characteristics.to_dataframe(fields="returns")
    industry_names = characteristics.fields["industry"].levels

    mvo = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
        prior_estimator=model,  # factor model as prior
        max_weights=0.05,  # limit individual positions to 5%
        min_weights=-0.05,  # allow short positions and limit to -5%
        budget=0.0,  # dollar neutral
        max_long=1.0,  # 100% long and 100% short: 200% gross exposure
        transaction_costs=0.001 / month,  # 10 bps amortized over one month
        fallback="previous_weights",  # keep last valid weights if a fit fails
        linear_constraints=[
            "momentum >= 1.0",
            "profitability == 1.0",
            "non_linear_size == -2.0",

            # Exact neutrality
            "beta == 0",
            "size == 0",
            "volatility == 0",

            # Small bands on the remaining styles
            "growth <= 0.05",
            "growth >= -0.05",

            "investment <= 0.05",
            "investment >= -0.05",

            "value <= 0.05",
            "value >= -0.05",

            "liquidity <= 0.05",
            "liquidity >= -0.05",

            "earnings_yield <= 0.05",
            "earnings_yield >= -0.05",

            "leverage <= 0.05",
            "leverage >= -0.05",

            "dividend_yield <= 0.05",
            "dividend_yield >= -0.05",

            # Industry neutrality
            *[f"{name} == 0.0" for name in industry_names],
        ],
    )

    mvo.fit(X, characteristics=characteristics)

    print(mvo.weights_)

    # Factor model fitted inside the optimizer, reused below for attribution
    factor_model = mvo.prior_estimator_.factor_model_


Here the full coverage universe serves as the investment universe: `X` contains
the returns of every asset in the panel. The factor model fitted inside the optimizer is
available through `mvo.prior_estimator_.factor_model_` and is reused in the
:ref:`Attribution <factor_model_attribution>` section. A separate factor model
can also be fitted for attribution only.

The portfolio is dollar neutral with `budget=0.0` and limited to 100% long
exposure with `max_long=1.0`. Dollar neutrality implies an equally sized short
position, so the maximum gross exposure is:

.. math::

    100\% \text{ long} + 100\% \text{ short} = 200\%.

Individual positions are limited to :math:`\pm 5\%`. Transaction costs follow the
skfolio convention: a linear cost per unit traded, deducted from the portfolio
expected return, which is expressed per observation period (here daily). A
transaction cost is paid once per rebalancing while a position earns its return
on every period it is held, so the 10 basis points are amortized over the
one-month expected holding period to convert them to a daily cost,
`0.001 / month` (see
:ref:`Periodicity Convention <periodicity_convention>`). Market
impact and borrow costs can be added through the optimizer's `add_objective` and
`add_constraints` parameters, with native support planned for a future release.

The portfolio targets three equity styles: momentum, profitability and non-linear size.
The beta, size and volatility exposures are set to zero. The remaining styles are
constrained within :math:`\pm 0.05`, and industry exposures are neutralized. In
`linear_constraints`, an expression on a factor name (e.g. `"momentum >= 1.0"`)
applies to the portfolio exposure to that factor, while an expression on a family
name (e.g. `"style <= 0.5"`) applies to the sum of exposures over the family's
factors. Industry neutrality therefore uses one constraint per industry
factor rather than a single `"industry == 0"` constraint, which would only force
industry exposures to offset each other. No explicit
`"market == 0"` constraint is needed because `budget=0.0` already sets the
exposure to the market intercept to zero.

`fallback="previous_weights"` keeps the latest valid allocation when a
rebalancing problem is infeasible, for example on dates where strict constraints
cannot be satisfied. Fallback estimators and the fallback audit trail are covered
in :ref:`sphx_glr_auto_examples_mean_risk_plot_17_failure_and_fallbacks.py`.

The portfolio is evaluated using monthly walk-forward rebalancing:

.. code-block:: python

    from skfolio.model_selection import online_predict

    # Two years plus one month of observations warm up the model
    # before the first rebalancing
    warmup_size = 252 * 2 + 21

    mpp = online_predict(
        estimator=mvo,
        X=X,
        warmup_size=warmup_size,
        test_size=month,
        params={"characteristics": characteristics},
    )
    print(mpp.n_fallback_portfolios)
    print(mpp.summary())
    print(mpp.annualized_sharpe_ratio)
    mpp.plot_cumulative_returns()

.. include:: ../_static/factor_model/fragments/mpp_cumulative_returns.inc.rst

.. code-block:: python

    mpp.plot_composition()

.. include:: ../_static/factor_model/fragments/mpp_composition.inc.rst

`online_predict` returns a :class:`~skfolio.portfolio.MultiPeriodPortfolio`, one
portfolio per rebalancing. The :ref:`Portfolio <portfolio>` user guide covers the
portfolio objects and their analytics (e.g. `plot_long_short_exposure`,
`plot_contribution`, `summary`). The two-year `warmup_size` covers the stacked
descriptor and estimator warmups (see :ref:`Warmup Periods
<factor_model_warmup>`). The out-of-sample portfolio achieves an annualized
Sharpe ratio of 0.91.

This example trades factor premia and does not use an alpha forecast. The
factor-neutral case is covered in the :ref:`Factor-Neutral Alpha Portfolio
<factor_model_factor_neutral_alpha_portfolio>` section below.

The optimizer consumes the factor model with the following conventions:

* For variance-based risk measures, the optimizer consumes the covariance square
  root described in the :ref:`Asset Covariance Forecast <factor_model_asset_covariance_forecast>` section, 
  operating in the factor space instead of forming the dense asset covariance.
* For scenario-based risk measures (e.g. `RiskMeasure.CVAR`), the optimizer consumes
  the asset return scenarios from `return_distribution_.returns`, which combine
  factor and idiosyncratic components. With :ref:`online learning <factor_model_online_learning>`, `max_history` keeps the
  scenarios on a rolling window.
* Assets that are not investable at the current date (e.g. delisted, in warmup)
  carry NaN moments. The optimizer solves on the investable subset and assigns them
  zero weight, as described in the :ref:`Input Data <factor_model_input_data>` section.

Robust optimization in the orthogonal space is configured at the optimizer level,
following the :ref:`Orthogonal Space Regularization <factor_model_orthogonal_space_regularization>` section:

.. code-block:: python

    from skfolio.uncertainty_set import OrthogonalCovarianceUncertaintySet

    mvo.set_params(
        covariance_uncertainty_set_estimator=OrthogonalCovarianceUncertaintySet(radius=1.0)
    )


Walk-forward evaluation and hyperparameter tuning of the full
optimization-plus-prior pipeline are covered in the :ref:`Walk-Forward Evaluation <factor_model_walk_forward_evaluation>` and :ref:`Hyperparameter Tuning <factor_model_hyper_parameter_tuning>` sections.

.. note::

    When the research focus is the optimization itself and the factor model is
    fixed, refitting the prior at each iteration can dominate the runtime on
    large universes. The factor model outputs can be precomputed with
    `partial_fit` over the chosen observations, stored in a database or local
    cache, and served back by a custom prior estimator that reads them for the
    requested observation range. Native factor model caching is planned for a
    future release.

.. _factor_model_factor_neutral_alpha_portfolio:

Factor-Neutral Alpha Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous example earns its return from factor premia through explicit
exposure targets. This portfolio takes the opposite approach, common in
statistical arbitrage: all factor exposures are kept close to zero and the
return comes from the orthogonal alpha component. This setup requires an alpha
estimator producing orthogonal alpha. Without one, expected returns come entirely
from factor exposures, and once those exposures are constrained to zero, every
feasible portfolio has zero expected return and the optimal allocation is empty.

The validated alpha estimator from the :ref:`Alpha Estimators
<factor_model_alpha>` section is attached to the factor model, and a
dollar-neutral long-short portfolio is optimized with weekly rebalancing. Style
exposures are bounded within :math:`\pm 0.05`, industry exposures are set to
zero, and individual positions are limited to :math:`\pm 3\%`. Because
factor-neutral strategies typically run at higher gross leverage, `max_long` is raised to
3.0, allowing up to 600% gross exposure. The optimization objective is a mean-variance 
utility, balancing the alpha forecast against risk and transaction costs:

.. code-block:: python

    from skfolio.model_selection import online_predict
    from skfolio import RiskMeasure
    from skfolio.optimization import MeanRisk, ObjectiveFunction

    week = 5
    model.set_params(alpha_estimator=alpha_estimator)

    X = characteristics.to_dataframe(fields="returns")
    industry_names = characteristics.fields["industry"].levels
    style_factors = [
        "beta", "momentum", "profitability", "non_linear_size", "size", "volatility",
        "growth", "investment", "value", "liquidity", "earnings_yield",
        "leverage", "dividend_yield",
    ]

    mvo = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=RiskMeasure.VARIANCE,
        risk_aversion=1,
        prior_estimator=model,
        max_weights=0.03,
        min_weights=-0.03,
        budget=0.0,
        max_long=3.0,
        transaction_costs=0.001 / week,
        fallback="previous_weights",
        linear_constraints=[
            *[f"{name}  <= 0.05" for name in style_factors],
            *[f"{name}  >= -0.05" for name in style_factors],
            *[f"{name} == 0" for name in industry_names],
        ],
    )

    warmup_size = 2 * 252 + 21
    mpp = online_predict(
        estimator=mvo,
        X=X,
        warmup_size=warmup_size,
        test_size=week,
        params={"characteristics": characteristics},
        entry_rebalancing_params={"transaction_costs": 0.0},
    )
    print(mpp.annualized_sharpe_ratio)

`entry_rebalancing_params` applies estimator parameters only while constructing
the first portfolio, which starts from cash while later portfolios rebalance
from the previous weights. Setting `transaction_costs=0.0` at entry avoids
charging costs on the full initial ramp-up from cash, letting the first
rebalancing use the desired allocation directly, up to the 600% gross-exposure
limit, instead of building exposure over several rebalancings. The regular
parameters are then restored for subsequent updates. At
this level of gross and short exposure, borrow costs and market impact become
material. They can be added through the optimizer's `add_objective` and
`add_constraints` parameters, with native support planned for a future release.
:meth:`~skfolio.portfolio.MultiPeriodPortfolio.plot_cumulative_returns` and
:meth:`~skfolio.portfolio.MultiPeriodPortfolio.plot_long_short_exposure` display
the resulting path and the long and short books.

Realized attribution verifies that the portfolio behaves as intended:

.. code-block:: python

    realized_attrib = mpp.realized_attribution(
        factor_model=factor_model,
        compute_uncertainty=True,
        compute_asset_breakdowns=False,
    )
    realized_attrib.plot_return_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/alpha_realized_return_contrib.inc.rst

Factor contributions are negligible and nearly all of the realized return comes
from the idiosyncratic component, around 6.6% annualized with a narrow 95%
confidence interval. The :ref:`Attribution <factor_model_attribution>` section
covers the methodology.


Evaluation and Tuning
---------------------

A factor model pipeline can be evaluated at three complementary levels:

* :ref:`Regression diagnostics <factor_model_regression_diagnostics>` measure how
  well the factor structure explains the cross-section of returns.
* :ref:`Covariance forecast evaluation <factor_model_covariance_forecast_evaluation>`
   tests the out-of-sample calibration of the risk forecasts.
* :ref:`Walk-forward evaluation <factor_model_walk_forward_evaluation>` measures the realized portfolio outcomes of
  the full pipeline, including the optimizer.

.. _factor_model_warmup:

Warmup Periods
~~~~~~~~~~~~~~

Walk-forward evaluation starts with a warmup period. The `warmup_size` observations
are fitted before the first prediction. Its minimum value follows from the history
each stage of the pipeline consumes before producing its first output.

Rolling and exponentially weighted descriptors (e.g.
:class:`~skfolio.descriptor.RollingMomentum`,
:class:`~skfolio.descriptor.EWMarketBeta`) return NaN during their warmup, and
the cross-sectional regression requires finite exposures, so the longest
descriptor warmup determines the first observation of the factor return
history. The estimators consuming the factor and idiosyncratic return series
add their own warmup on top: the factor prior (covariance and expected returns)
and the idiosyncratic variance estimator. These warmup periods are cumulative. 
For example, with one year of descriptor warmup and one year of covariance warmup, 
the first usable forecast arrives after about two years of data. The walk-forward
examples in this guide use `warmup_size = 2 * 252 + 21` for this reason.

Alpha estimators also have warmup periods from their own descriptors, but these
run concurrently with the model's warmup rather than after it. The model passes
the full :class:`~skfolio.containers.AssetPanel`, enriched with idiosyncratic
returns, idiosyncratic variances and exposures that keep their leading warmup
NaN values, instead of truncating the panel.

.. _factor_model_walk_forward_evaluation:

Walk-Forward Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

:func:`~skfolio.model_selection.online_predict` simulates the strategy through
time by updating a single stateful estimator with `partial_fit` and predicting
on each subsequent test window:

.. code-block:: python

    from skfolio.model_selection import online_predict

    portfolio = online_predict(
        estimator=optimization,
        X=X,
        warmup_size=2 * 252 + 21,
        test_size=21,
        params={"characteristics": characteristics},
    )
    portfolio.summary()

Unlike :func:`~skfolio.model_selection.cross_val_predict`, which clones and refits
the estimator on each fold, `online_predict` updates a single stateful estimator
and carries its state forward, making long walk-forward backtests practical for
models of this size. The result is a multi-period portfolio with the usual
skfolio analytics (summary, plots, risk measures).
:func:`~skfolio.model_selection.online_score` follows the same pattern and
returns a score.

.. _factor_model_hyper_parameter_tuning:

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

All model parameters, from descriptor half-lives to weighting powers, shrinkages
and thresholds, follow the scikit-learn convention and can be tuned with
standard model-selection utilities. For walk-forward tuning,
:class:`~skfolio.model_selection.OnlineGridSearch` and
:class:`~skfolio.model_selection.OnlineRandomizedSearch` evaluate each parameter
combination in a single walk-forward pass using `partial_fit`, instead of refitting
the estimator on every fold:

.. code-block:: python

    from skfolio.model_selection import OnlineGridSearch

    search = OnlineGridSearch(
        estimator=optimization,
        param_grid={
            "prior_estimator__inv_idio_variance_weight_shrinkage": [0.0, 0.5, 1.0],
            "prior_estimator__exposure_lag": [1, 2],
        },
        warmup_size=2 * 252 + 21,
        test_size=21,
    )
    search.fit(X, characteristics=characteristics)
    search.best_params_

Scoring can target two levels. Portfolio-level scores (e.g. ratio measures)
evaluate the full pipeline including the optimizer, as in the example above.
Covariance losses from :mod:`skfolio.metrics` evaluate the risk model itself 
with the factor model being passed directly as the search estimator, without 
an optimizer, and scored on its covariance forecast. In `make_scorer`, 
`response_method=None` indicates a non-predictor estimator and 
`greater_is_better=False` a loss to minimize:

.. code-block:: python

    from skfolio.metrics import make_scorer, portfolio_variance_qlike_loss
    from skfolio.model_selection import OnlineGridSearch

    qlike_scorer = make_scorer(
        portfolio_variance_qlike_loss,
        greater_is_better=False,
        response_method=None,
    )

    search = OnlineGridSearch(
        estimator=model,
        param_grid={
            "factor_prior_estimator__covariance_estimator__regime_half_life": [10, 21, 63],
        },
        scoring=qlike_scorer,
        warmup_size=2 * 252 + 21,
        test_size=21,
    )
    search.fit(X, characteristics=characteristics)
    search.best_params_

The same workflow is shown on a covariance estimator in
:ref:`sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py`.


.. _factor_model_attribution:

Attribution
-----------

Attribution decomposes portfolio risk and return into the contributions of
individual factors, factor families, the idiosyncratic component and,
optionally, individual assets. Applied ex ante, it shows where the forecast
risk and expected return come from. Applied ex post, it shows which factors
delivered the realized performance, with standard errors that separate genuine
contributions from estimation noise.

Attribution is accessed from a :class:`~skfolio.portfolio.Portfolio` or
:class:`~skfolio.portfolio.MultiPeriodPortfolio`, which supplies the weights
and portfolio returns. Three methods take the fitted factor model as argument:

* `predicted_attribution` computes ex-ante attribution from the fitted loading
  matrix, factor covariance and idiosyncratic covariance. The volatility forecast
  is decomposed using the exposure-volatility-correlation framework
  (:math:`x`-:math:`\sigma`-:math:`\rho`) and, when expected factor returns are
  available, expected return is decomposed into spanned and orthogonal
  components.
* `realized_attribution` computes ex-post attribution from the realized factor
  returns, exposures and idiosyncratic returns.
* `rolling_realized_attribution` runs the realized attribution over rolling
  windows, showing how contributions evolve through time.

The same methods are available at a lower level on the fitted
:class:`~skfolio.prior.FactorModel`, taking `weights` (a single vector or a
time-varying array) and `portfolio_returns` explicitly.

All three return an :class:`~skfolio.attribution.Attribution` object
with the same structure:

* `systematic`, `idio` and `total`: component-level breakdowns with volatility,
  volatility contribution, share of total variance, return and correlation with
  the portfolio. Realized attribution adds `unexplained`, the residual between
  observed portfolio returns and model-attributed returns.
* `factors` and `families`: per-factor breakdowns with exposures, standalone
  statistics and contributions, with the same information aggregated by factor
  family.
* `assets` and `asset_by_factor_contrib`: the per-asset systematic/idiosyncratic
  decomposition and, optionally, the full asset-by-factor contribution matrix.

Realized attribution supports uncertainty estimates (`compute_uncertainty=True`,
the default). Using the stored regression weights and idiosyncratic variances, it
computes standard errors on the factor and idiosyncratic return contributions,
exposed as `mu_contrib_uncertainty` in the factor breakdown.

Results are available as DataFrames through `summary_df`, `families_df` and
`factors_df`, and as plots through `plot_exposure`, `plot_vol_contrib`,
`plot_return_contrib` and `plot_return_vs_vol_contrib`. Rolling attributions
carry an `observations` axis and can be indexed (`attribution[i]`) to retrieve
the attribution of a single window.

Return contributions are reported directly in additive return units. Risk
contributions are additionally normalized as shares of total variance, the
standard scale for comparing risk attribution across portfolios and periods.

The figures below use the constrained mean-variance portfolio from the
:ref:`Portfolio Construction <factor_model_portfolio_construction>` section. That
portfolio is dollar neutral and imposes three main style constraints:
`momentum >= 1.0`, `profitability == 1.0` and `non_linear_size == -2.0`.
It also sets beta, size, volatility and industry exposures to zero and allows
only small exposures to the remaining style factors.

Ex-Ante Attribution
~~~~~~~~~~~~~~~~~~~

Ex-ante attribution decomposes the risk and expected-return forecasts of the
optimized portfolio:

.. code-block:: python

    # Access via the Portfolio or MultiPeriodPortfolio API
    portfolio = mvo.predict(X)
    predicted_attrib = portfolio.predicted_attribution(factor_model=factor_model)

    # Equivalent access via the FactorModel API, passing the weights explicitly
    predicted_attrib = factor_model.predicted_attribution(weights=mvo.weights_)

    predicted_attrib.summary_df()

.. include:: ../_static/factor_model/tables/attribution_predicted_summary.inc.rst

The predicted annualized volatility of 17.5% splits into a 16.0% systematic
volatility contribution and a 1.5% idiosyncratic volatility contribution. Because
each volatility contribution is the corresponding variance contribution divided
by total volatility, dividing again by total volatility gives the variance share:
91.6% systematic and 8.4% idiosyncratic. The expected return is entirely
systematic: no alpha estimator is attached, so the model forecasts no return in
the orthogonal space.

.. code-block:: python

    predicted_attrib.families_df()

.. include:: ../_static/factor_model/tables/attribution_predicted_families.inc.rst

In the family breakdown, the industry and market rows carry zero exposure and
zero contribution, as imposed by the neutrality constraints. All systematic
risk and expected return come from the style family.

.. code-block:: python

    predicted_attrib.factors_df().head()

.. include:: ../_static/factor_model/tables/attribution_predicted_factors_head.inc.rst

The per-factor breakdown reports each factor's standalone volatility and
expected return (the statistics of the factor's own return series) next to its
contributions. The volatility contribution follows the
exposure-volatility-correlation decomposition: portfolio exposure multiplied by
standalone volatility multiplied by correlation with the portfolio. The momentum
factor accounts for 83% of the predicted variance and has a 0.90 correlation
with the portfolio.

.. code-block:: python

    predicted_attrib.plot_exposure(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_predicted_exposure.inc.rst

The profitability factor sits at its 1.0 target and the non-linear size factor at
its -2.0 target. The momentum exposure reaches about 2.6, well above its 1.0
floor as the ratio-maximizing objective concentrates in the factor with the
strongest forecast premium.
The market, industry and neutralized style exposures are zero, and the
remaining styles stay within their :math:`\pm 0.05` bands.

.. code-block:: python

    predicted_attrib.plot_vol_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_predicted_vol_contrib.inc.rst

The momentum factor dominates predicted risk with about 14.5% of the 17.5% total
annualized volatility. The idiosyncratic contribution of about 1.5% is the
second largest: with market and industry exposures forced to zero, part of the
allocation moves into orthogonal directions, which enter the risk forecast only
through the idiosyncratic variances (see :ref:`Orthogonal Space Regularization
<factor_model_orthogonal_space_regularization>`).

.. code-block:: python

    predicted_attrib.plot_return_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_predicted_return_contrib.inc.rst

The momentum factor contributes about 14% of annualized expected return, the
non-linear size factor contributes 3%, and the profitability factor contributes
a small negative amount, in line with their exposures and forecast premia. The
idiosyncratic contribution is exactly zero.

.. code-block:: python

    predicted_attrib.plot_return_vs_vol_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_predicted_return_vs_vol_contrib.inc.rst

The scatter plots expected return contribution against volatility contribution.
The momentum factor sits in the top right, driving both, while the idiosyncratic
component lies on the zero-return axis, carrying risk without forecast reward.

Ex-Post Attribution
~~~~~~~~~~~~~~~~~~~

Ex-post attribution decomposes the realized performance of the walk-forward
portfolio, whose weights vary through time:

.. code-block:: python

    # Access via the Portfolio or MultiPeriodPortfolio API
    realized_attrib = mpp.realized_attribution(factor_model=factor_model)

    # Equivalent access via the FactorModel API, passing the time-varying
    # weights and the realized portfolio returns explicitly
    realized_attrib = factor_model.realized_attribution(
        weights=weights,
        portfolio_returns=portfolio_returns,
    )

    realized_attrib.summary_df()

.. include:: ../_static/factor_model/tables/attribution_realized_summary.inc.rst

Out of sample, the systematic component earned 7.5% ± 1.5% annualized mean
return for a 5.1% volatility contribution. The idiosyncratic component cost
-1.4% ± 1.5%: the risk taken in orthogonal directions was not compensated,
consistent with the zero orthogonal alpha forecast.

.. code-block:: python

    realized_attrib.families_df()

.. include:: ../_static/factor_model/tables/attribution_realized_families.inc.rst

The realized family breakdown reports the mean and standard deviation of each
exposure over the backtest. Industry and market exposures stay near zero, so
the neutrality constraints held at every rebalancing.

.. code-block:: python

    realized_attrib.factors_df().head()

.. include:: ../_static/factor_model/tables/attribution_realized_factors_head.inc.rst

At the factor level, momentum and non-linear size each contributed about 3.2%
of annualized return, but momentum consumed three times the risk budget (53%
of total variance against 17%). Out of sample, the short non-linear-size
position was the more efficient trade.

.. code-block:: python

    realized_attrib.plot_exposure(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_realized_exposure.inc.rst

Realized exposures are averaged over the backtest, with error bars showing one
standard deviation of their variation through time. The equality-constrained
factors (profitability, non-linear size) show tight bands, while momentum
averages about 1.1 with a wider band: its floor constraint leaves the optimizer
free to exceed 1.0 when the forecast premium justifies it.

.. code-block:: python

    realized_attrib.plot_vol_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_realized_vol_contrib.inc.rst

.. code-block:: python

    realized_attrib.plot_return_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_realized_return_contrib.inc.rst

The error bars show the 95% confidence intervals on the mean return
contributions. The momentum, non-linear size and profitability factors are clearly
positive, while the -1.4% idiosyncratic contribution has an interval crossing
zero, so it is not distinguishable from estimation noise.

.. code-block:: python

    realized_attrib.plot_return_vs_vol_contrib(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_realized_return_vs_vol_contrib.inc.rst

The scatter shows that non-linear size delivered the same return as momentum at
a third of the risk, while the idiosyncratic component sits below the axis,
carrying risk without reward.

Rolling Attribution
~~~~~~~~~~~~~~~~~~~

Rolling attribution repeats the realized attribution over rolling windows,
showing how contributions evolve through time:

.. code-block:: python

    rolling_realized_attrib = mpp.rolling_realized_attribution(
        factor_model=factor_model,
        compute_uncertainty=True,
        compute_asset_breakdowns=False,
    )
    rolling_realized_attrib.summary_df().head(8)

.. include:: ../_static/factor_model/tables/attribution_rolling_realized_summary.inc.rst

The summary carries one component breakdown per window, dated at the window end.

.. code-block:: python

    rolling_realized_attrib.plot_exposure(top_n=15)

.. include:: ../_static/factor_model/fragments/attribution_rolling_realized_exposure.inc.rst

The constraints hold throughout the backtest: profitability stays at its 1.0
target, non-linear size at -2.0, and momentum near its 1.0 floor, rising above
it when the forecast premium strengthens (e.g. in 2018 and from late 2025).
Indexing the rolling attribution (`rolling_realized_attrib[i]`) retrieves a
single window with the same plots and DataFrames as above.


Model Validation and Review
---------------------------

Users who wish to review the implementation can start with the statistical
recovery suite in
`tests/test_prior/test_characteristics_factor_model/test_statistical_recovery.py`.
The tests build synthetic panels from known data-generating processes, fit
:class:`~skfolio.prior.CharacteristicsFactorModel` and verify that the estimated
quantities recover the intended model structure.

The suite covers factor returns, loadings, the covariance identity
:math:`\Sigma = B F B^\top + D`, idiosyncratic risk, residual orthogonality,
exposure lagging, estimation masks, zero-sum constraints, neutralization,
inverse-idiosyncratic-variance weighting, time-varying market capitalizations,
changing universes and currency factors. These checks support implementation
review alongside the empirical diagnostics and walk-forward evaluation described
in this guide.


.. _factor_model_performance:

Computational Performance
-------------------------

A production factor model processes large datasets. For example, a coverage 
universe of 5,000 assets with 10 years of daily data and 80 characteristic 
fields holds over a billion entries. This section describes how the 
implementation handles this scale and what to expect in terms of fitting time and
memory usage.

.. _factor_model_online_learning:

Online Learning
~~~~~~~~~~~~~~~

The factor model supports online learning. `partial_fit` appends new
observations without refitting the history, and the result is identical to a
batch `fit` on the concatenated data. This is used in three ways:

* An :class:`~skfolio.containers.AssetPanel` that does not fit in memory is
  processed chunk by chunk, with only the current chunk held in memory (see
  :ref:`Memory Usage <factor_model_memory_usage>`).
* Walk-forward evaluation and hyper-parameter tuning run in a single pass over
  the data through the online utilities `online_predict`,
  `online_covariance_forecast_evaluation` and `OnlineGridSearch`.
* In production, daily updates only fit the new observation instead of refitting
  the full history. Even for large models, this update runs in well under a
  second in the benchmarks below.

Internal state (estimator warmup, exposure and weight lag buffers, constraint
bases) is carried across calls, and `max_history` bounds the retained
time-series outputs to a rolling window.

.. code-block:: python

    model.fit(characteristics=characteristics[:warmup])
    for i in range(warmup, len(characteristics), 5):
        model.partial_fit(characteristics=characteristics[i : i + 5])

See :ref:`Online Learning <online_learning>` for the general framework.

Benchmarks
~~~~~~~~~~

The following fitting times were measured on 10 years of daily data (2,520
observations) on a laptop (Ultra 9 275HX, 24 cores, 32 GB RAM), for two models:

* Model 1: 16 factors (1 global, 10 industries, 5 styles) with default parameters.
* Model 2: the 58-factor model used throughout this guide (1 global, 44 industries, 13 styles
  from 29 descriptors), with within-industry scoring, neutralization, zero-sum
  constraints, two-pass inverse-idiosyncratic-variance regression weights and the
  regime-adjusted covariance estimator.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Assets
     - Model 1
     - Model 2
   * - 500
     - 3 s
     - 14 s
   * - 1,000
     - 5 s
     - 22 s
   * - 5,000
     - 25 s
     - 87 s

An incremental `partial_fit` on the next observation runs in under a second for both
models.

Achieving these performances relies on two implementation choices. Firstly, the
hot paths are vectorized NumPy operations backed by parallel BLAS kernels.  
Secondly, factor exposures are computed with thread-based parallelism (`n_jobs`), 
which avoids copying the panel to worker processes (the computations are 
NumPy-dominated and release the GIL, so threads provide effective parallelism 
while sharing the panel in memory).

.. _factor_model_memory_usage:

Memory Usage
~~~~~~~~~~~~

By default, all panel data is held in memory for vectorized operations and
thread-based parallelism. A coverage
universe of 5,000 assets with 10 years of daily data (2,500 observations) and 80
characteristic fields holds :math:`5{,}000 \times 80 \times 2{,}500 = 10^9` entries,
or 8 GB in float64. This fits on a typical 32 GB machine.

When memory becomes a constraint, three options are available:

* Process the data in chunks with `partial_fit`, keeping only the current batch in
  memory and bounding the retained outputs with `max_history`.
* Store selected fields as float32 in the :class:`~skfolio.containers.AssetPanel`,
  halving their footprint.
* Subclass :class:`~skfolio.containers.AssetPanel` to load characteristic fields
  lazily and release them once descriptors have consumed them.


References
----------

.. [1] "The Elements of Quantitative Investing",
    Giuseppe A. Paleologo (2025).

.. [2] "Active Portfolio Management: A Quantitative Approach for Producing Superior
    Returns and Controlling Risk", Richard C. Grinold & Ronald N. Kahn, McGraw-Hill
    (1999).

.. [3] "Portfolio Optimization: Theory and Application", Chapter 3,
    Daniel P. Palomar (2025).

.. [4] "Extra-Market Components of Covariance in Security Returns",
    Barr Rosenberg, Journal of Financial and Quantitative Analysis (1974).
