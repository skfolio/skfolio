.. _uncertainty_set_estimator:

.. currentmodule:: skfolio.uncertainty_set

*************************
Uncertainty Set Estimator
*************************

The :ref:`Uncertainty Set estimator <uncertainty_set_ref>` build an ellipsoidal
:class:`UncertaintySet` of the distribution moments.

An ellipsoidal uncertainty set is defined by its size :math:`\kappa` and
shape :math:`S`. Ellipsoidal uncertainty set can be used with both expected returns
and covariance:

Expected returns ellipsoidal uncertainty set:

    .. math:: U_{\mu}=\left\{\mu\,|\left(\mu-\hat{\mu}\right)S^{-1}\left(\mu-\hat{\mu}\right)^{T}\leq\kappa^{2}\right\}

Covariance ellipsoidal uncertainty set:

    .. math:: U_{\Sigma}=\left\{\Sigma\,|\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)S^{-1}\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)^{T}\leq k^{2}\,,\,\Sigma\succeq 0\right\}


It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as
the assets returns and stores the :class:`UncertaintySet` in its `uncertainty_set_`
attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc...)


Available estimators for the expected returns are:
    * :class:`EmpiricalMuUncertaintySet`
    * :class:`BootstrapMuUncertaintySet`

Available estimators for the covariance are:
    * :class:`EmpiricalCovarianceUncertaintySet`
    * :class:`BootstrapCovarianceUncertaintySet`

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


It is used to solve worst-case optimization using the
:class:`~skfolio.optimization.MeanRisk` estimator. Worst-case optimization is a class of
robust optimization. It reduces the instability that arises from the estimation errors
of the expected returns and the covariance matrix.

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

