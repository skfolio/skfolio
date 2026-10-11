.. _migration:

.. currentmodule:: skfolio

===============
Migration Guide
===============

`skfolio` follows `semantic versioning <https://semver.org>`_. The public API remains
backward compatible within a major series. Deprecated functionality raises a
`FutureWarning` and is removed in the next major release.

This page documents the changes required to upgrade between major versions.

.. _migration_2_0:

Preparing for 2.0
-----------------

The following deprecated APIs remain available throughout 1.x and will be removed
in 2.0. Their replacements are available now.

Hierarchical Optimizer Imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import HRP, HERC, Schur Complementary and NCO from `skfolio.optimization` or
`skfolio.optimization.hierarchical`:

.. code-block:: python

    from skfolio.optimization.hierarchical import (
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        NestedClustersOptimization,
        SchurComplementary,
    )

Imports from `skfolio.optimization.cluster` and
`skfolio.optimization.cluster.hierarchical` raise a `FutureWarning` and return
the same estimator classes. The `skfolio.cluster` module for clustering
estimators is unchanged.

HRP and Schur Ordering
~~~~~~~~~~~~~~~~~~~~~~

Configure hierarchical clustering through :class:`~skfolio.seriation.HierarchicalSeriation`
on HRP and Schur Complementary.

Before:

.. code-block:: python

    model = HierarchicalRiskParity(
        hierarchical_clustering_estimator=HierarchicalClustering()
    )

After:

.. code-block:: python

    from skfolio.cluster import HierarchicalClustering
    from skfolio.seriation import HierarchicalSeriation

    model = HierarchicalRiskParity(
        seriation_estimator=HierarchicalSeriation(
            hierarchical_clustering_estimator=HierarchicalClustering()
        )
    )

The old parameter warns during `fit` or the first `partial_fit` of a learning
run. Supplying both estimators raises an error. The default ordering is unchanged.

After fitting, access the clustering estimator through
`model.seriation_estimator_.hierarchical_clustering_estimator_`. The old
`model.hierarchical_clustering_estimator_` attribute raises a `FutureWarning`.
This alias is only available when the fitted seriator is hierarchical.

HERC's `hierarchical_clustering_estimator` and NCO's `clustering_estimator`
remain supported.

Custom Hierarchical Optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`BaseHierarchicalOptimization` is deprecated. Existing custom subclasses remain
supported throughout 1.x.

HRP, HERC and Schur Complementary no longer inherit from this base.
`isinstance` and `issubclass` checks against `BaseHierarchicalOptimization`
now return `False` for these estimators. Use `BaseOptimization` to recognize
portfolio optimizers generally, or a tuple of concrete optimizer classes to
recognize a specific family.

For new custom optimizers, inherit from
:class:`~skfolio.optimization.BaseOptimization` and implement `fit` to set
`weights_`. When migrating an existing subclass, provide any risk calculations
and bound conversion previously inherited through `_risk`, `_unitary_risks`
and `_convert_weights_bounds`. The private seriation base and risk mixin are
implementation details, not supported replacements for the public base.

.. _migration_1_0:

Migrating to 1.0
----------------

Version 1.0 introduces the stable public API. The parameters and aliases deprecated
during the 0.x series are removed in this release.

Exponentially Weighted Moments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~skfolio.moments.EWMu` and :class:`~skfolio.moments.EWCovariance` no longer
accept `alpha`. Use `half_life`, the number of observations for a weight to decay
to 50%.

Before:

.. code-block:: python

    EWMu(alpha=0.2)
    EWCovariance(alpha=0.2)

After:

.. code-block:: python

    EWMu(half_life=3.11)
    EWCovariance(half_life=3.11)

The half-life equivalent of a given `alpha` is

.. math:: \text{half-life} = \frac{-1}{\log_2(1 - \alpha)}

For example, `alpha=0.2` corresponds to a `half_life` of approximately :math:`3.11`
and `alpha=0.02` to :math:`34.31`. The decay factor is
:math:`\lambda = 2^{-1/\text{half-life}}`, computed by
:func:`~skfolio.utils.tools.half_life_to_decay_factor`.

Passing `alpha` raises a `TypeError`.

Walk-Forward Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~skfolio.model_selection.WalkForward` no longer accepts `expend_train`. Use
`expand_train`, which has identical behavior.

Before:

.. code-block:: python

    WalkForward(test_size=60, train_size=252, expend_train=True)

After:

.. code-block:: python

    WalkForward(test_size=60, train_size=252, expand_train=True)

Factor Models
~~~~~~~~~~~~~

The `FactorModel` prior estimator is replaced by
:class:`~skfolio.prior.TimeSeriesFactorModel`, and `factors` is now a keyword-only
argument of `fit`.

Before:

.. code-block:: python

    from skfolio.optimization import MeanRisk
    from skfolio.prior import FactorModel

    model = MeanRisk(prior_estimator=FactorModel())
    model.fit(X_train, y_train)

After:

.. code-block:: python

    from skfolio.optimization import MeanRisk
    from skfolio.prior import TimeSeriesFactorModel

    model = MeanRisk(prior_estimator=TimeSeriesFactorModel())
    model.fit(X_train, factors=factors_train)

.. warning::

    `FactorModel` now refers to a different object: the fitted factor model container
    exposed on :attr:`~skfolio.prior.ReturnDistribution.factor_model`, holding the
    loading matrix, the factor and idiosyncratic moments, and the realized factor
    returns. The import therefore still resolves, and estimator arguments passed to
    `FactorModel` raise a `TypeError` for unexpected keyword arguments rather than an
    `ImportError`.

:class:`~skfolio.prior.CharacteristicsFactorModel` provides a cross-sectional
alternative, fitted from point-in-time asset characteristics rather than factor return
time series. See :ref:`Factor Models <factor_models>`.

Uncertainty Sets
~~~~~~~~~~~~~~~~

:class:`~skfolio.uncertainty_set.UncertaintySet` now describes a general norm-ball
rather than an ellipsoid, which allows box and diamond sets to use the same
representation. The ellipsoid is the :math:`p = 2` case.

The field names changed as follows:

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Before
      - After
      - Description
    * - `k`
      - `radius`
      - Size :math:`\kappa` of the normalized uncertainty ball.
    * - `sigma`
      - `geometry`
      - Linear map :math:`L` with :math:`S = L L^{T}` for an ellipsoid with shape
        matrix :math:`S`. May be low-rank.
    * - not applicable
      - `norm`
      - Norm :math:`p` selecting the shape, defaulting to :math:`2` for an ellipsoid.

This only affects code that constructs an `UncertaintySet` directly or reads the fitted
`uncertainty_set_` attribute. Passing an uncertainty set estimator to
:class:`~skfolio.optimization.MeanRisk` is unchanged.

Two factor-model estimators are added:
:class:`~skfolio.uncertainty_set.OrthogonalMuUncertaintySet` and
:class:`~skfolio.uncertainty_set.OrthogonalCovarianceUncertaintySet`. See
:ref:`Uncertainty Set <uncertainty_set_estimator>`.

.. _migration_scheduled_removals:

Scheduled for Removal in 2.0
----------------------------

The following remain available throughout 1.x and raise a `FutureWarning`:

* `annualized_factor`, on :class:`~skfolio.portfolio.Portfolio` and
  :class:`~skfolio.moments.ImpliedCovariance`. Use `annualization_factor`.
* `non_denominated_sort`, as a function and as a
  :class:`~skfolio.population.Population` method. Use `non_dominated_sort`.
