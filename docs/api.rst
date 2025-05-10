.. _api:

=============
API Reference
=============

This is the class and function reference of ``skfolio``. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. _measures_ref:

:mod:`skfolio.measures`: Measures
=================================

.. automodule:: skfolio.measures
    :no-members:
    :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    measures.BaseMeasure

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    measures.PerfMeasure
    measures.RiskMeasure
    measures.ExtraRiskMeasure
    measures.RatioMeasure

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    measures.mean
    measures.get_cumulative_returns
    measures.get_drawdowns
    measures.variance
    measures.semi_variance
    measures.standard_deviation
    measures.semi_deviation
    measures.third_central_moment
    measures.fourth_central_moment
    measures.fourth_lower_partial_moment
    measures.cvar
    measures.mean_absolute_deviation
    measures.value_at_risk
    measures.worst_realization
    measures.first_lower_partial_moment
    measures.entropic_risk_measure
    measures.evar
    measures.drawdown_at_risk
    measures.cdar
    measures.max_drawdown
    measures.average_drawdown
    measures.edar
    measures.ulcer_index
    measures.gini_mean_difference
    measures.owa_gmd_weights
    measures.effective_number_assets
    measures.correlation

.. _portfolio_ref:

:mod:`skfolio.portfolio`: Portfolio
===================================

.. automodule:: skfolio.portfolio
    :no-members:
    :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    portfolio.BasePortfolio

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    portfolio.Portfolio
    portfolio.MultiPeriodPortfolio


.. _population_ref:

:mod:`skfolio.population`: Population
=====================================

.. automodule:: skfolio.population
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    population.Population


.. _optimization_naive_ref:

:mod:`skfolio.optimization.naive`: Naive Optimization Estimators
================================================================

.. automodule:: skfolio.optimization.naive
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    optimization.EqualWeighted
    optimization.InverseVolatility
    optimization.Random


.. _optimization_convex_ref:

:mod:`skfolio.optimization.convex`: Convex Optimization Estimators
==================================================================

.. automodule:: skfolio.optimization.convex
   :no-members:
   :no-inherited-members:

Enum
----
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    optimization.ObjectiveFunction

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    optimization.ConvexOptimization
    optimization.MeanRisk
    optimization.RiskBudgeting
    optimization.MaximumDiversification
    optimization.DistributionallyRobustCVaR


.. _optimization_cluster_ref:

:mod:`skfolio.optimization.cluster`: Clustering Optimization Estimators
==========================================================================

.. automodule:: skfolio.optimization.cluster
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    optimization.BaseHierarchicalOptimization
    optimization.HierarchicalRiskParity
    optimization.HierarchicalEqualRiskContribution
    optimization.NestedClustersOptimization

.. _optimization_ensemble_ref:

:mod:`skfolio.optimization.ensemble`: Ensemble Optimization Estimators
======================================================================

.. automodule:: skfolio.optimization.ensemble
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    optimization.BaseComposition
    optimization.StackingOptimization

.. _prior_ref:

:mod:`skfolio.prior`: Prior Estimators
======================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Model Dataclass
---------------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    prior.ReturnDistribution

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    prior.BasePrior

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    prior.EmpiricalPrior
    prior.BlackLitterman
    prior.FactorModel
    prior.SyntheticData
    prior.EntropyPooling
    prior.OpinionPooling

Loading Matrix Classes for Factor Models
----------------------------------------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    prior.BaseLoadingMatrix
    prior.LoadingMatrixRegression

.. _mu_ref:

:mod:`skfolio.moments.mu`: Mu Estimators
========================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    moments.BaseMu

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    moments.EmpiricalMu
    moments.EWMu
    moments.ShrunkMu
    moments.EquilibriumMu
    moments.ShrunkMuMethods

.. _covariance_ref:

:mod:`skfolio.moments.covariance`: Covariance Estimators
========================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    moments.BaseCovariance

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    moments.EmpiricalCovariance
    moments.EWCovariance
    moments.GerberCovariance
    moments.DenoiseCovariance
    moments.DetoneCovariance
    moments.LedoitWolf
    moments.OAS
    moments.ShrunkCovariance
    moments.GraphicalLassoCV
    moments.ImpliedCovariance

.. _distance_ref:

:mod:`skfolio.distance`: Distance Estimators
============================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distance.BaseDistance

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distance.PearsonDistance
    distance.KendallDistance
    distance.SpearmanDistance
    distance.CovarianceDistance
    distance.DistanceCorrelation
    distance.MutualInformation

.. _cluster_ref:

:mod:`skfolio.cluster`: Cluster Estimators
===================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    cluster.HierarchicalClustering
    cluster.LinkageMethod

.. _uncertainty_set_ref:

:mod:`skfolio.uncertainty_set`: Uncertainty set Estimators
==========================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Model Dataclass
---------------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    uncertainty_set.UncertaintySet


Base Classes
------------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    uncertainty_set.BaseMuUncertaintySet
    uncertainty_set.BaseCovarianceUncertaintySet

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    uncertainty_set.EmpiricalMuUncertaintySet
    uncertainty_set.EmpiricalCovarianceUncertaintySet
    uncertainty_set.BootstrapMuUncertaintySet
    uncertainty_set.BootstrapCovarianceUncertaintySet


.. _pre_selection_ref:

:mod:`skfolio.pre_selection`: Pre-selection Transformers
========================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    pre_selection.DropCorrelated
    pre_selection.DropZeroVariance
    pre_selection.SelectKExtremes
    pre_selection.SelectNonDominated
    pre_selection.SelectComplete
    pre_selection.SelectNonExpiring

.. _model_selection_ref:

:mod:`skfolio.model_selection`: Model Selection
===============================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Base Classes
------------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    model_selection.BaseCombinatorialCV

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    model_selection.CombinatorialPurgedCV
    model_selection.WalkForward

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    model_selection.cross_val_predict
    model_selection.optimal_folds_number

.. _metrics_ref:

:mod:`skfolio.metrics`: Metrics
===============================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    metrics.make_scorer

.. _datasets_ref:

:mod:`skfolio.datasets`: Datasets
=================================

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    datasets.load_sp500_dataset
    datasets.load_sp500_index
    datasets.load_factors_dataset
    datasets.load_ftse100_dataset
    datasets.load_nasdaq_dataset


.. _preprocessing_ref:

:mod:`skfolio.preprocessing`: Preprocessing
===========================================

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    preprocessing.prices_to_returns

.. _stats_ref:

:mod:`skfolio.utils.stats`: Stats
=================================

Functions
---------
.. currentmodule:: skfolio.utils

.. autosummary::
    :toctree: generated/
    :template: function.rst

    stats.NBinsMethod
    stats.n_bins_freedman
    stats.n_bins_knuth
    stats.is_cholesky_dec
    stats.assert_is_square
    stats.assert_is_symmetric
    stats.assert_is_distance
    stats.cov_nearest
    stats.cov_to_corr
    stats.corr_to_cov
    stats.commutation_matrix
    stats.compute_optimal_n_clusters
    stats.rand_weights
    stats.rand_weights_dirichlet
    stats.minimize_relative_weight_deviation


.. _univariate_distribution_ref:

:mod:`skfolio.distribution.univariate`: Univariate Distribution Estimators
==========================================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.BaseUnivariateDist

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.Gaussian
    distribution.StudentT
    distribution.JohnsonSU
    distribution.NormalInverseGaussian

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    distribution.select_univariate_dist

.. _multivariate_distribution_ref:

:mod:`skfolio.distribution.multivariate`: Multivariate Distribution Estimators
==============================================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.VineCopula

Enum
----
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.DependenceMethod

.. _bivariate_copula_ref:

:mod:`skfolio.distribution.copula`: Bivariate Copula Estimators
===============================================================

.. automodule:: skfolio
   :no-members:
   :no-inherited-members:

Base Class
----------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.BaseBivariateCopula

Classes
-------
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.GaussianCopula
    distribution.StudentTCopula
    distribution.ClaytonCopula
    distribution.GumbelCopula
    distribution.JoeCopula
    distribution.IndependentCopula

Functions
---------
.. currentmodule:: skfolio

.. autosummary::
    :toctree: generated/
    :template: function.rst

    distribution.compute_pseudo_observations
    distribution.empirical_tail_concentration
    distribution.plot_tail_concentration
    distribution.select_bivariate_copula

Enum
----
.. currentmodule:: skfolio

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    distribution.CopulaRotation
