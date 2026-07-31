.. _uncertainty_set_estimator:

.. currentmodule:: skfolio.uncertainty_set

*************************
Uncertainty Set Estimator
*************************

An :ref:`uncertainty set estimator <uncertainty_set_ref>` builds the region in which a
distribution moment is assumed to lie under estimation error. When one is provided to
:class:`~skfolio.optimization.MeanRisk`, the objective is evaluated at the least
favorable moment within that region instead of at the point estimate. This is called
worst-case optimization and is a class of robust optimization. It reduces the
instability that arises from the estimation errors of the expected returns and the
covariance matrix.

It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as
the assets returns and stores the fitted set in its `uncertainty_set_` attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)

Norm-ball representation
************************

:class:`UncertaintySet` represents deviations of a parameter vector :math:`z` from its
estimate :math:`\hat{z}` as :math:`z - \hat{z} = L u` with
:math:`\lVert u \rVert_p \leq \kappa`:

    .. math:: \mathcal{U}=\left\{\hat{z} + L u\,:\,\lVert u \rVert_p \leq \kappa\right\}

The parameter :math:`z` is :math:`\mu` for expected return uncertainty, and
:math:`\text{vec}(\Sigma)`, the vector obtained by stacking the columns of
:math:`\Sigma`, for covariance uncertainty. The set is defined by three fields:

* `radius`, the size :math:`\kappa` of the normalized ball.
* `norm`, the norm :math:`p` that selects its shape: :math:`2` for an ellipsoid,
  :math:`\infty` for a box and :math:`1` for a diamond.
* `geometry`, the linear map :math:`L` that scales and mixes the uncertainty
  directions.

For :math:`p = 2` and shape matrix :math:`S = L L^{T}`, the set is ellipsoidal:

    .. math:: U_{\mu}=\left\{\mu\,|\left(\mu-\hat{\mu}\right)S^{-1}\left(\mu-\hat{\mu}\right)^{T}\leq\kappa^{2}\right\}

Optimizers use the worst-case deviation over :math:`\mathcal{U}`, which for a linear
exposure vector :math:`e` is :math:`\kappa \lVert L^{T} e \rVert_q`, with :math:`q` the
dual norm of :math:`p`. The map :math:`L` may be low-rank, which keeps this penalty
tractable for covariance uncertainty.

Available estimators
********************

For the expected returns:
    * :class:`EmpiricalMuUncertaintySet`
    * :class:`BootstrapMuUncertaintySet`
    * :class:`OrthogonalMuUncertaintySet`

For the covariance:
    * :class:`EmpiricalCovarianceUncertaintySet`
    * :class:`BootstrapCovarianceUncertaintySet`
    * :class:`OrthogonalCovarianceUncertaintySet`

The size of the set is controlled by `confidence_level`. The empirical and bootstrap
estimators derive the radius from the quantile of a chi-squared distribution at that
level, so a higher confidence level widens the set and increases the penalty.
:class:`OrthogonalCovarianceUncertaintySet` is parameterized by `radius` instead.

The `Orthogonal` estimators require the optimizer's `prior_estimator` to be a factor
model, such as :class:`~skfolio.prior.CharacteristicsFactorModel`. The optimizer passes
the fitted return distribution to them. They confine the uncertainty to the subspace
orthogonal to the factor-model loading matrix, which penalizes allocations in directions
that the factor model prices only through idiosyncratic variance. See
:ref:`Orthogonal Space Regularization <factor_model_orthogonal_space_regularization>`.

Covariance estimators may store a :class:`CompactCovarianceUncertaintySet` instead of a
norm ball. It expresses the penalty as a reduced quadratic form that the optimizer adds
directly to the variance term, avoiding the lifted semidefinite formulation required by
a generic set.

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.uncertainty_set import EmpiricalMuUncertaintySet

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = EmpiricalMuUncertaintySet()
    model.fit(X)
    print(model.uncertainty_set_)

Worst-case optimization
***********************

Uncertainty set estimators are provided to :class:`~skfolio.optimization.MeanRisk`
through `mu_uncertainty_set_estimator` and `covariance_uncertainty_set_estimator`. The
optimizer fits them and subtracts the resulting penalty from the portfolio expected
return:

    .. math:: w^{T}\hat{\mu} - \kappa\lVert L^{T}w\rVert_{q}

Covariance uncertainty is applied when `risk_measure=RiskMeasure.VARIANCE` or when
`max_variance` is set.

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns
    from skfolio.uncertainty_set import (
        BootstrapMuUncertaintySet,
        EmpiricalCovarianceUncertaintySet,
    )

    prices = load_sp500_dataset()
    prices = prices["2020":]
    X = prices_to_returns(prices)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        mu_uncertainty_set_estimator=BootstrapMuUncertaintySet(confidence_level=0.5),
        covariance_uncertainty_set_estimator=EmpiricalCovarianceUncertaintySet(
            confidence_level=0.5
        ),
    )
    model.fit(X)
    print(model.weights_)
