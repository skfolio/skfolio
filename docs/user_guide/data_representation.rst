.. _asset_data_representation:

*************************
Asset Data Representation
*************************

The choice of data structure, data container and missing-data handling matters
for portfolio workflows, cross-sectional factor models and alpha pipelines. This page discusses the main
choices, their trade-offs and the convention used by `skfolio`.

.. _wide_and_long_format:

Wide and Long Format
====================

Asset data (e.g. market data, fundamental data) can be represented in either wide
or long format.

In **wide format**, each field is stored as a date-by-asset matrix, with dates as
rows and assets as columns. A single field (e.g. returns) is a 2D table.
Missingness is represented as NaNs:

.. code-block:: text

    date          AAPL    MSFT    BMW
    2024-01-01    0.01    0.02    NaN
    2024-01-02   -0.01    NaN     0.03

In **long format**, each row represents a `(date, asset)` pair. For a single field
(e.g. returns), the values are stored in one column. Missingness can be represented
either by a NaN or by the absence of a row:

.. code-block:: text

    date          asset   returns
    2024-01-01    AAPL     0.01
    2024-01-01    MSFT     0.02
    2024-01-02    AAPL    -0.01
    2024-01-02    BMW      0.03

With several fields, long format keeps the same `(date, asset)` rows and adds one
column per field:

.. code-block:: text

    date          asset   returns   volume   industry
    2024-01-01    AAPL      0.01      1200    tech
    2024-01-01    MSFT      0.02       900    tech
    2024-01-02    AAPL     -0.01       NaN    tech
    2024-01-02    BMW       0.03       700    auto

In wide format, the multi-field case is less direct and discussed below.

Both representations have trade-offs and serve different purposes.

Long format is often convenient for storage, database queries, joins, filtering and are
more memory efficient when the universe changes through time. It can also naturally
distinguish missing data for an asset that belongs to the universe (e.g. holidays),
represented by a NaN, from an asset that is not in the universe (e.g. delisting),
represented by the absence of a row.

The drawback is that most estimators in an end-to-end quant pipeline do not directly
consume long-format data. When they do (e.g. ML alpha prediction treating each
`(date, asset)` pair as one sample with one column per feature), they often sit in
the middle of the pipeline. Before them, the data usually needs time-aware and/or
cross-sectional transformations (e.g. cross-sectional z-scores, ranking,
winsorization, time-series estimates, factor neutralization). After them, their
outputs usually need to be time and asset aligned again for risk estimation,
portfolio optimization and evaluation. Many steps would need to handle date group-by,
pivoting, reindexing and asset alignment internally before the data can be used. These
transformations add overhead and code complexity and they increase the risk of indexing
mistakes, either on the time index, which can introduce look-ahead bias, or on the asset
index. They also make cross-validation and hyper-parameter tuning more complex when the
whole workflow must remain time-aware.

On the contrary, wide format uses more memory when the universe changes through time,
because assets that are not present at a given date are represented by NaNs. For example,
if a universe changes by about 2% per year, a 10-year history carries roughly 20%
additional entries for assets that were not present during the full period. In return,
wide format keeps the data in a dense `date × asset` representation expected by most
transformers and estimators. This allows vectorized implementations to operate on
already-aligned arrays, avoids repeated date group-by, pivoting and reindexing,
simplifies cross-validation and hyper-parameter tuning and reduces asset-alignment
errors.


`skfolio` is opinionated and follows the wide-format convention because it often provides
a worthwhile trade-off: higher memory usage, which is cheap for typical use cases,
in exchange for improved computational efficiency, simpler code and clearer temporal
and asset alignment.

One challenge is representing the multi-field case in wide format. Several choices
are possible, such as a 3D array, xarray object, DataFrame with MultiIndex columns,
or dictionary of 2D arrays. However, these choices are not optimal for portfolio workflows:
they either lack field metadata, require repeated expensive reindexing, don't handle
zero-copy views or expose a suboptimal API for this use case.

For this reason, `skfolio` developed :class:`~skfolio.containers.AssetPanel`, a
dedicated container for aligned cross-sectional asset data. It keeps the wide layout,
stores field metadata, supports categorical and tensor fields, and keeps masks
aligned with the data. See :class:`~skfolio.containers.AssetPanel` for details.

.. _missing_data:

Missing Data and Changing Universes
===================================

Financial datasets often contain missing returns and changing asset universes. Over a
given history, assets can:

* enter the investment universe (e.g. new listing)
* leave the investment universe (e.g. delisting, default, expiry)
* remain in the universe while having missing data on some dates (e.g. holidays,
  trading interruptions, missing quotes)

Each source of missingness has different modelling implications. The right treatment
depends on the modelling choice and on what the downstream estimators support. A poor
choice can introduce bias, including universe-selection bias, survivorship bias,
non-synchronous trading bias or imputation bias.

Because wide format can encode distinct data states with the same NaN marker,
`skfolio` uses explicit conventions to distinguish:

* missing data for assets that belong to the universe (e.g. holidays)
* assets that are outside the universe at a given date (e.g. new listing, delisting)
* assets that are in the universe but not yet investable (e.g. not enough estimation data)

`skfolio` provides two main ways to handle missing data:

* make the input finite before fitting, using pre-selection, imputation, or both via a
  scikit-learn `Pipeline`
* use estimators that handle NaNs natively when they support it

Pre-Selection and Imputation
============================

When an estimator requires finite input, NaNs must be handled before the estimator is
fitted. This can be done with :ref:`pre-selection transformers <pre_selection>`, with
imputation, or with both.

For example, :class:`~skfolio.pre_selection.SelectComplete` keeps only assets with a
complete history over the fitted period, and
:class:`~skfolio.pre_selection.SelectNonExpiring` can remove assets according to known
expiration dates. These transformers can be combined with imputers and optimizers in a
standard `Pipeline`.

This approach is useful when:

* the downstream estimator requires finite inputs
* the missingness rule can be expressed as an asset selection rule
* imputing missing data is an acceptable modelling assumption

See :ref:`sphx_glr_auto_examples_pre_selection_plot_4_incomplete_dataset.py` for an
example using inception, default, expiration, imputation and walk-forward validation in
a single pipeline.

This approach makes the input finite before estimation. Asset selection removes columns
from the fitted dataset, while imputation inserts chosen values for the remaining
missing observations. These rules are appropriate when they match the intended
modelling choice. They are not equivalent to native missing-data handling: for example,
filling a holiday return with zero is different from freezing the estimator state for
that observation. They also cannot represent estimator-specific readiness in the same
way: an EWMA covariance estimator can keep an asset in the universe while exposing NaNs
in its fitted covariance until the asset has enough observations for the estimate to be
used.

Native NaN-Aware Approach
=========================

Some estimators explicitly accept NaNs.

This is useful when the estimator can work with partial information. For example, a
covariance estimator can update covariance entries from non-missing pairs instead of
dropping the asset or imputing the missing returns. It can also keep its own state, such
as freezing an estimate during a holiday or exposing NaNs while an asset is still in
its warmup period. This avoids replacing missing returns with artificial values that
can bias expected returns, volatilities or correlations, and lets the estimator signal
when an estimate is not ready yet.

The native approach is also better suited to online learning. NaN-aware estimators can
update their state with `partial_fit`. The pipeline-based approach described above
cannot currently be applied in `skfolio` online learning workflows, because
scikit-learn pipelines do not provide the required online update interface for
pre-selection and imputation.

.. _native_nan_aware:

Native NaN-Aware Convention
===========================

This section applies to the native NaN-aware approach. It describes the convention used
by compatible estimators and by optimizers that consume their outputs.

For this approach, `skfolio` separates three concepts:

* missing observations in `X` (e.g. holidays)
* universe membership through time (e.g. new listings, delistings, defaults,
  expirations)
* investability at optimization time (e.g. an asset that has entered the universe but
  has not yet accumulated enough data for stable moment estimation)

.. _fixed_asset_schema:

Fixed Asset Schema in Online Learning
-------------------------------------

Consider three assets, A, B and C, where C lists after the first two observations:

.. code-block:: python

    import numpy as np
    import pandas as pd

    history = pd.DataFrame(
        {
            "A": [0.01, -0.02,  0.01,  0.02],
            "B": [0.02,  0.01, -0.01,  0.01],
            "C": [np.nan, np.nan, 0.03, -0.01],
        }
    )

C's column is present throughout the dataset. Its pre-listing returns are NaN.
For estimators accepting `active_mask`, C is marked inactive during that period
and active afterward. A holiday also produces a missing return, but the asset
remains active.

In research, this fixed schema provides the alignment and computational benefits
described in :ref:`wide_and_long_format`. With compatible estimators, C can
accumulate observations and become investable through `partial_fit` without
restarting the model.

In production, C may not have been planned as part of the universe when the model
was first fitted on A and B. When the decision is made to include C, the fixed
schema requires a new learning run with C included. Call `fit` on the expanded
historical dataset, or replay historical batches through a fresh estimator.
Subsequent `partial_fit` calls retain columns A, B and C in the same order.

This rebuilding step allows C's available history to contribute before
allocation. An asset joining a strategy may have traded for months or years. Its
past returns can initialize expected returns, volatility, and dependence with
the existing assets. Estimating those dependencies requires aligned historical
observations for all relevant assets.

An alternative design would be to accept dictionaries of returns, similar to
single-date slices of long-format data, and support new asset identifiers during
updates. Preserving existing state could reduce latency when historical replay
is expensive. However, incorporating a new asset's earlier observations would
still require a separate historical initialization mechanism. Otherwise,
learning would begin with its first supplied observation.

Skfolio accepts the cost of rebuilding when new columns are introduced. A fixed
schema keeps returns, moments, weights, constraints, costs, and metadata aligned,
avoids dynamically remapping state across components, and follows scikit-learn's
feature-consistency convention.

For a worked example with late listings, delistings, holidays and asset warm-up,
see :ref:`sphx_glr_auto_examples_online_learning_plot_online_schur_changing_universe.py`.

Universe Membership
-------------------

Some estimators accept an `active_mask` parameter. It is a boolean array with the same
shape as `X`:

.. math::

    active\_mask_{t,i} \in \{\mathrm{True}, \mathrm{False}\}

It indicates whether asset :math:`i` is active in the universe at observation
:math:`t`.

If `active_mask=True` and `X` is NaN, the value is considered missing for that
observation (e.g. holiday). NaN-aware estimators handle this according to their own rule
(e.g. skipping the missing pairwise update or freezing the current estimate).

If `active_mask=False`, the asset is inactive for that observation (e.g.
pre-listing or post-delisting periods). Estimators use this information to mark the
asset as unavailable.

EW moment estimators request `active_mask` by default in both `fit` and
`partial_fit`. Enable metadata routing to pass the mask through a prior or
optimizer. See :ref:`default_metadata_requests`. Without a mask, trailing NaNs
are treated as missing observations and do not signal a delisting.

When data is stored in an :class:`~skfolio.containers.AssetPanel`, each field applies
its `inactive_policy` outside `active_mask`. The default policy stores NaN for
floating numeric fields and `MISSING=-1` for categorical fields. Some generated
fields can use zero or leave inactive values unchanged when that is the field's
convention.

Estimation Universe
-------------------

Some estimators also accept an `estimation_mask` parameter. It is used for
estimator-specific calculations.

For example, a covariance estimator may compute a regime statistic on a restricted set
of liquid assets while still updating pairwise covariance estimates for all assets that
belong to the universe.

`estimation_mask` should be read as "use this asset in this estimator statistic".

Moment Estimators
-----------------

In the native NaN-aware convention, moment estimators expose unavailable assets through
NaNs in their fitted outputs.

If an expected return cannot be estimated for asset :math:`i`, then :math:`\mu_i` is
set to NaN. If a variance cannot be estimated for asset :math:`i`, then
:math:`\Sigma_{i,i}` is set to NaN.

Covariance estimators keep this convention consistent across the covariance matrix. If
an asset cannot belong to a finite covariance block, the corresponding row and column
of :math:`\Sigma` are set to NaN.

A NaN in the fitted moments marks the asset as not usable by downstream optimization,
even though the asset remains present in the full asset universe.

Prior Estimators
----------------

Prior estimators store a full-universe
:class:`~skfolio.prior.ReturnDistribution` in `return_distribution_`.

The full universe contains all assets passed to `fit`, including assets that are not
currently investable. Non-investable assets remain present in the arrays, but are
represented by NaNs in :math:`\mu`, :math:`\Sigma`, or both.

The investable universe is inferred from the fitted moments:

.. math::

    investable_i =
    \operatorname{isfinite}(\mu_i) \land
    \operatorname{isfinite}(\Sigma_{i,i})

An asset is investable only when both its expected return and variance are finite.

Optimization
------------

Before building the optimization problem, compatible portfolio optimizers extract the
investable subset from the prior's full-universe
:class:`~skfolio.prior.ReturnDistribution`.

The optimization problem is solved only on assets with finite :math:`\mu_i` and finite
:math:`\Sigma_{i,i}`. After solving, the weights are expanded back to the full input
universe. Assets outside the investable subset receive a weight of zero.

This keeps `weights_` aligned with the original columns of `X`, while ensuring that
the solver only receives a finite optimization problem.

Native Convention Summary
-------------------------

The convention is:

#. `X` may contain NaNs.
#. `active_mask` identifies whether each asset belongs to the universe at each
   observation.
#. `estimation_mask` optionally restricts estimator-specific statistics.
#. Moment estimators encode unavailable assets with NaNs in :math:`\mu` or
   :math:`\Sigma`.
#. Prior estimators keep the full asset universe in `return_distribution_`.
#. Optimizers solve on the investable subset and expand `weights_` back to the full
   universe.
