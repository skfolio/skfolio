.. _missing_data:

************
Missing Data
************

Financial datasets often contain missing returns and changing asset universes. Over a
given history, assets can:

* enter the investment universe (e.g. new listing)
* leave the investment universe (e.g. delisting, default, expiry)
* remain in the universe while having missing observations on some dates (e.g.
  holidays, trading interruptions, missing quotes)

Each source of missingness has different modelling implications. The right treatment
depends on the modelling choice and on what the downstream estimators support. A poor
choice can introduce bias, including universe-selection bias, survivorship bias,
non-synchronous trading bias or imputation bias.

Data Representation
===================

Input data such as asset returns is commonly represented in one of two forms.

The first is a **wide format**: a rectangular matrix indexed by date, with one column per
asset. Missing returns are represented as NaNs:

.. code-block:: text

    date          AAPL    MSFT    BMW
    2024-01-01    0.01    0.02    NaN
    2024-01-02   -0.01    NaN     0.03

The second is a **long format**: one row per `(date, asset)` pair. Missingness can be
represented either by a NaN value or by the absence of a row:

.. code-block:: text

    date          asset   return
    2024-01-01    AAPL     0.01
    2024-01-01    MSFT     0.02
    2024-01-02    AAPL    -0.01
    2024-01-02    BMW      0.03

Both representations have trade-offs. Long format can be more memory efficient when
the universe changes composition through time. It can also naturally distinguish a
missing return for an asset that belongs to the universe, represented by a NaN, from an
asset that is not in the universe, represented by the absence of a row.

The drawback is that most estimators do not directly consume raw long-format data.
When they do, it is often after transformations that must be aware of the time
dimension, such as cross-sectional z-scores. This means that many estimators would need
to handle pivoting, reindexing and asset alignment internally before the data can be
used. These transformations add overhead and complexity, and they increase the risk of
indexing mistakes, either on the time index, which can introduce look-ahead bias, or on
the asset index. They also make cross-validation and hyper-parameter tuning more
complex, because the whole workflow must often remain time-aware.

`skfolio` is opinionated and follows the wide format convention.

Wide format may use more memory when the universe changes through time, because assets
that are not present at a given date are represented by NaNs. For example, if an
index universe changes by about 2% per year, a 10-year history carries roughly 20%
additional cells for assets that were not present during the full period. In return,
wide format keeps the data in the 2D representation expected by estimators. This allows
vectorized implementations to operate on already-aligned arrays, avoids repeated pivoting and
reindexing, simplifies cross-validation and hyper-parameter tuning, and reduces
asset-alignment errors.

Because wide format can encode distinct data states with the same NaN marker,
`skfolio` uses explicit conventions to distinguish:

* missing observations for assets that belong to the universe
* assets that are outside the universe at a given date
* assets that are in the universe but not yet investable

`skfolio` provides two main ways to handle missing data:

* make the input finite before fitting, using pre-selection, imputation, or both via a scikit-learn `Pipeline`;
* use estimators that handle NaNs natively when they support it.

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

Native NaN-Aware Convention
===========================

This section applies to the native NaN-aware approach. It describes the convention used
by compatible estimators and by optimizers that consume their outputs.

For this approach, `skfolio` separates three concepts:

* missing observations in `X` (e.g. holidays);
* universe membership through time (e.g. new listings, delistings, defaults or
  expirations);
* investability at optimization time (e.g. an asset that has entered the universe but
  has not yet accumulated enough data for stable moment estimation).

Universe Membership
-------------------

Some estimators accept an `active_mask` parameter. It is a boolean array with the same
shape as `X`:

.. math::

    active\_mask_{t,i} \in \{\mathrm{True}, \mathrm{False}\}

It indicates whether asset :math:`i` belongs to the universe at observation :math:`t`.

If `active_mask=True` and `X` is NaN, the asset remains in the universe but its return
is missing for that observation (e.g. holiday). NaN-aware estimators handle this
according to their own rule (e.g. skipping the missing pairwise update or freezing the
current estimate).

If `active_mask=False`, the asset is outside the universe for that observation. This
is used for cases such as pre-listing and post-delisting periods. Estimators use this
information to mark the asset as unavailable when its fitted moments cannot be used.

Estimation Universe
-------------------

Some estimators also accept an `estimation_mask` parameter. It is used for
estimator-specific calculations. It is not an investability mask.

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

This means that NaNs in fitted moments have a specific meaning. They signal that the
asset is not usable by downstream optimization, even if the asset remains present in
the full asset universe.

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

Compatible portfolio optimizers do not solve optimization problems with non-investable
assets. Before building the optimization problem, they extract the investable subset
from the prior's full-universe :class:`~skfolio.prior.ReturnDistribution`.

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
