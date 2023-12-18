.. _prior:

.. currentmodule:: skfolio.prior

***************
Prior Estimator
***************

A prior estimator fits a :class:`PriorModel` containing the distribution estimate of asset returns.
It represents the investor's prior beliefs about the model used to estimate such distribution.

A prior estimator follow the same API as scikit-learn's `estimator`: the `fit` method
takes `X` as the assets returns and stores the :class:`PriorModel` in its
`prior_model_` attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc...)

.. warning::

    The prior of one model can be the posterior of another one. For example, :class:`BlackLitterman` takes as input a
    prior estimator used to compute the prior expected returns and prior covariance matrix which are updated using the
    analyst's views to get the posterior expected returns and posterior covariance matrix. These posterior estimates
    will be saved in a new :class:`PriorModel` that can be used in another estimator.




The :class:`PriorModel` is a dataclass containing:

    * `mu`: Expected returns estimation
    * `covariance`: Covariance matrix estimation
    * `returns`: Assets returns estimation
    * `cholesky` : Lower-triangular Cholesky factor of the covariance estimation (optional)

Empirical Prior
***************

The :class:`EmpiricalPrior` estimator estimates the :class:`PriorModel` by fitting a
`mu_estimator` and a `covariance_estimator` separately.

**Example:**

Empirical prior with James-Stein shrinkage for the estimation of expected returns and
Denoising for the estimation of the covariance matrix:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.moments import DenoiseCovariance, ShrunkMu
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import EmpiricalPrior

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = EmpiricalPrior(
        mu_estimator=ShrunkMu(), covariance_estimator=DenoiseCovariance()
    )
    model.fit(X)
    print(model.prior_model_)


Black & Litterman
*****************

The :class:`BlackLitterman` estimator estimates the :class:`PriorModel` using the
Black & Litterman model. It takes a Bayesian approach by using a prior estimate
of the assets expected returns and covariance matrix which are updated using the
analyst views to get the posterior estimate.

**Example:**

.. code-block:: python

    from skfolio.preprocessing import prices_to_returns
    from skfolio.datasets import load_sp500_dataset
    from skfolio.prior import BlackLitterman

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    analyst_views = [
        "AAPL - BBY == 0.0003",
        "CVX - KO == 0.0004",
        "MSFT == 0.0006",
    ]

    model = BlackLitterman(views=analyst_views)
    model.fit(X)
    print(model.prior_model_)


Factor Model
************

The :class:`FactorModel` estimator estimates the :class:`PriorModel` using a factor
model and a :ref:`prior estimator <prior>` of the factor's returns.

The purpose of factor models is to impose a structure on financial variables and
their covariance matrix by explaining them through a small number of common factors.
This can help to overcome estimation error by reducing the number of parameters,
i.e., the dimensionality of the estimation problem, making portfolio optimization
more robust against noise in the data. Factor models also provide a decomposition of
financial risk to systematic and security specific components.

To be fully compatible with `scikit-learn`, the `fit` method takes `X` as the assets
returns and `y` as the factors returns. Note that `y` is in lower case even for a 2D
array (more than one factor). This is for consistency with the scikit-learn API.

**Example:**

.. code-block:: python

    from skfolio.datasets import load_factors_dataset, load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import FactorModel

    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()
    X, y = prices_to_returns(prices, factor_prices)

    model = FactorModel()
    model.fit(X, y)
    print(model.prior_model_)


The loading matrix (betas) of the factors is estimated using a ``loading_matrix_estimator``. By Default we use
the :class:`LoadingMatrixRegression` which fits the factors using a :class:`sklean.linear_model.LassoCV` on each asset
separately.

Combining Multiple Prior Estimators
***********************************
Prior estimators can be combined. For example, it is possible to create a Black &
Litterman Factor Model by simply using a :class:`BlackLitterman` estimator for the prior
estimator of the :class:`FactorModel`:

**Example:**

Factor model for the estimation of the assets expected retruns and covariance matrix
with a Black & Litterman model for the estimation of the factors expected reruns and
covariance matrix incorporating the analyst views on the factors.

.. code-block:: python

    from skfolio.datasets import load_factors_dataset, load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import BlackLitterman, FactorModel

    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()
    X, y = prices_to_returns(prices, factor_prices)

    views = [
        "MTUM - QUAL == 0.0003",
        "SIZE - USMV == 0.0004",
        "VLUE == 0.0006",
    ]

    model = FactorModel(
        factor_prior_estimator=BlackLitterman(views=views),
    )
    model.fit(X, y)
    print(model.prior_model_)

