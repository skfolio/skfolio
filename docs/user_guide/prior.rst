.. _prior:

.. currentmodule:: skfolio.prior

***************
Prior Estimator
***************

A Prior Estimator in `skfolio` fits a :class:`ReturnDistribution` containing your
pre-optimization inputs (:math:`\mu`, :math:`\Sigma`, returns, sample weight, Cholesky decomposition).

The term "prior" is used in a general optimization sense, not confined to Bayesian
priors. It denotes any **a priori** assumption or estimation method for the return
distribution before optimization, unifying both **Frequentist**, **Bayesian** and
**Information-theoretic** approaches into a single cohesive framework:

1. Frequentist:
    * :class:`EmpiricalPrior`
    * :class:`FactorModel`
    * :class:`SyntheticData`

2. Bayesian:
    * :class:`BlackLitterman`

3. Information-theoretic:
    * :class:`EntropyPooling`
    * :class:`OpinionPooling`


In skfolio's API, all such methods share the same interface and adhere to scikit-learn's
estimator API: the `fit` method accepts `X` (the asset returns) and stores the
resulting :class:`ReturnDistribution` in its `return_distribution_` attribute.

`X` can be any array-like structure (NumPy array, pandas DataFrame, etc.).

The :class:`ReturnDistribution` is a dataclass containing:

    * `mu`: Estimated expected returns of shape (n_assets,)
    * `covariance`: Estimated covariance matrix of shape (n_assets, n_assets)
    * `returns`: (Estimated) asset returns of shape (n_observations, n_assets)
    * `sample_weight` : Sample weight for each observation of shape (n_observations,) (optional)
    * `cholesky` : Lower-triangular Cholesky factor of the covariance (optional)

.. note::

    The posterior of one model can serve as the prior for another. In skfolio,
    Prior Estimators can be composed into complex pre-optimization pipelines.
    For example, :class:`BlackLitterman` accepts a fitted Prior Estimator that computes
    initial expected returns and covariance, applies the analyst's views to update them,
    and then stores the resulting posterior expected returns and covariance in a new
    :class:`ReturnDistribution`, which can be passed into another estimator.

Empirical Prior
***************

The :class:`EmpiricalPrior` estimator estimates the :class:`ReturnDistribution` by
fitting its `mu_estimator` and `covariance_estimator` independently.

**Example:**

An `EmpiricalPrior` configured with James–Stein shrinkage to estimate expected returns
and a denoising method to estimate the covariance matrix:

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
    print(model.return_distribution_)


Black & Litterman
*****************

The :class:`BlackLitterman` estimator estimates the :class:`ReturnDistribution` using
the Black & Litterman model. It takes a Bayesian approach by starting from a prior
estimate of the assets' expected returns and covariance matrix, then updating them with
the analyst's views to obtain the posterior estimates.

Tutorials:
    * :ref:`Black & Litterman <sphx_glr_auto_examples_mean_risk_plot_12_black_and_litterman.py>`
    * :ref:`Black & Litterman Factor Model <sphx_glr_auto_examples_mean_risk_plot_14_black_litterman_factor_model.py>`

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
    print(model.return_distribution_)


Factor Model
************

The :class:`FactorModel` estimator estimates the :class:`ReturnDistribution` by fitting
a factor model on asset returns alongside a specified :ref:`prior estimator <prior>`
for the factor returns.

The purpose of factor models is to impose a structure on financial variables and
their covariance matrix by explaining them through a small number of common factors.
This can help overcome estimation error by reducing the number of parameters,
i.e., the dimensionality of the estimation problem, making portfolio optimization
more robust against noise in the data. Factor models also provide a decomposition of
financial risk into systematic and security-specific components.

To be fully compatible with `scikit-learn`, the `fit` method takes `X` as the assets
returns and `y` as the factors returns. Note that `y` is in lowercase even for a 2D
array (more than one factor). This is for consistency with the scikit-learn API.

Tutorials:
    * :ref:`Factor Model <sphx_glr_auto_examples_mean_risk_plot_13_factor_model.py>`
    * :ref:`Black & Litterman Factor Model <sphx_glr_auto_examples_mean_risk_plot_14_black_litterman_factor_model.py>`
    * :ref:`Hierarchical Risk Parity - CVaR <sphx_glr_auto_examples_clustering_plot_1_hrp_cvar.py>`
    * :ref:`Minimize CVaR on Stressed Factors - CVaR <sphx_glr_auto_examples_synthetic_data_plot_3_min_CVaR_stressed_factors.py>`

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
    print(model.return_distribution_)


The loading matrix (betas) of the factors is estimated using a
`loading_matrix_estimator`. By default, we use the :class:`LoadingMatrixRegression`
which fits the factors using a :class:`sklean.linear_model.LassoCV` on each asset
separately.

Synthetic Data
**************

The :class:`SyntheticData` estimator bridges scenario generation and portfolio
optimization. It estimates the :class:`ReturnDistribution` by fitting a
`distribution_estimator` and sampling new data from it.

The default `distribution_estimator` is a Regular
:class:`~skfolio.distribution.VineCopula` estimator. Other common choices are Generative
Adversarial Networks (GANs) or Variational Autoencoders (VAEs).

It is particularly useful when the historical distribution tail dependencies are
sparse and need extrapolation for tail optimizations or when optimizing under
conditional or stressed scenarios.

Tutorials:
    * :ref:`Vine Copula <sphx_glr_auto_examples_synthetic_data_plot_2_vine_copula.py>`
    * :ref:`Minimize CVaR on Stressed Factors <sphx_glr_auto_examples_synthetic_data_plot_3_min_CVaR_stressed_factors.py>`
    * :ref:`Entropy Pooling <sphx_glr_auto_examples_entropy_pooling_plot_1_entropy_pooling.py>`
    * :ref:`Opinion Pooling <sphx_glr_auto_examples_entropy_pooling_plot_2_opinion_pooling.py>`

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset, load_factors_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.distribution import VineCopula
    from skfolio.optimization import MeanRisk
    from skfolio.prior import FactorModel, SyntheticData
    from skfolio import RiskMeasure
   
    # Load historical prices and convert them to returns
    prices = load_sp500_dataset()
    X = prices_to_returns(prices, factors)
   
    # Instanciate the SyntheticData model and fit it
    model = SyntheticData()
    model.fit(X)
    print(model.return_distribution_)

    # Minimum CVaR optimization on synthetic returns
    vine = VineCopula(log_transform=True, n_jobs=-1)
    prior = =SyntheticData(distribution_estimator=vine, n_samples=2000)
    model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=prior)
    model.fit(X)
    print(model.weights_)

    # Stress Test
    vine = VineCopula(log_transform=True, central_assets=["BAC"]  n_jobs=-1)
    vine.fit(X)
    X_stressed = vine.sample(n_samples=10000, conditioning = {"BAC": -0.2})
    ptf_stressed = model.predict(X_stressed)

Entropy Pooling
***************  

:class:`EntropyPooling`, introduced by Attilio Meucci in 2008 as a generalization of the
Black-Litterman framework, is a nonparametric method for adjusting a baseline ("prior")
probability distribution to incorporate user-defined views by finding the posterior
distribution closest to the prior while satisfying those views.

User-defined views can be **elicited** from domain experts or **derived** from
quantitative analyses.

Grounded in information theory, it updates the distribution in the least-informative
way by minimizing the Kullback-Leibler divergence (relative entropy) under the
specified view constraints.

Tutorials:
    * :ref:`Entropy Pooling <sphx_glr_auto_examples_entropy_pooling_plot_1_entropy_pooling.py>`
    * :ref:`Opinion Pooling <sphx_glr_auto_examples_entropy_pooling_plot_2_opinion_pooling.py>`

**Example:**

.. code-block:: python

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import EntropyPooling
    from skfolio.optimization import HierarchicalRiskParity
    
    prices = load_sp500_dataset()
    prices = prices[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY", "PG"]]
    X = prices_to_returns(prices)
    
    groups = {
        "AMD": ["Technology", "Growth"],
        "BAC": ["Financials", "Value"],
        "GE": ["Industrials", "Value"],
        "JNJ": ["Healthcare", "Defensive"],
        "JPM": ["Financials", "Income"],
        "LLY": ["Healthcare", "Defensive"],
        "PG": ["Consumer", "Defensive"],
    }
    
    entropy_pooling = EntropyPooling(
        mean_views=[
            "JPM == -0.002",
            "PG >= LLY",
            "BAC >= prior(BAC) * 1.2",
            "Financials == 2 * Growth",
        ],
        variance_views=[
            "BAC == prior(BAC) * 4",
        ],
        correlation_views=[
            "(BAC,JPM) == 0.80",
            "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5",
        ],
        skew_views=[
            "BAC == -0.05",
        ],
        cvar_views=[
            "GE == 0.08",
        ],
        cvar_beta=0.90,
        groups=groups,
    )
    
    entropy_pooling.fit(X)
    
    print(entropy_pooling.relative_entropy_)
    print(entropy_pooling.effective_number_of_scenarios_)
    print(entropy_pooling.return_distribution_.sample_weight)
    
    # CVaR Hierarchical Risk Parity optimization on Entropy Pooling
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=entropy_pooling
    )
    model.fit(X)
    print(model.weights_)
    
    # Stress Test the Portfolio
    entropy_pooling = EntropyPooling(cvar_views=["AMD == 0.10"])
    entropy_pooling.fit(X)
    
    stressed_dist = entropy_pooling.return_distribution_

    stressed_ptf = model.predict(stressed_dist)


Opinion Pooling
***************  

:class:`OpinionPooling` (also called Belief Aggregation or Risk Aggregation) is a
process in which different probability distributions (opinions), produced by different
experts, are combined to yield a single probability distribution (consensus).

Expert opinions (also called individual prior distributions) can be
**elicited** from domain experts or **derived** from quantitative analyses.

The `OpinionPooling` estimator takes a list of prior estimators, each of which
produces scenario probabilities (`sample_weight`), and pools them into a single
consensus probability .

You can choose between linear (arithmetic) pooling or logarithmic (geometric)
pooling, and optionally apply robust pooling using a Kullback-Leibler divergence
penalty to down-weight experts whose views deviate strongly from the group
consensus.

Tutorials:
    * :ref:`Opinion Pooling <sphx_glr_auto_examples_entropy_pooling_plot_2_opinion_pooling.py>`

**Example:**

.. code-block:: python

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import EntropyPooling, OpinionPooling
    from skfolio.optimization import RiskBudgeting
    
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    
    # We consider two expert opinions, each generated via Entropy Pooling with
    # user-defined views.
    # We assign probabilities of 40% to Expert 1, 50% to Expert 2, and by default
    # the remaining 10% is allocated to the prior distribution:
    opinion_1 = EntropyPooling(cvar_views=["AMD == 0.10"])
    opinion_2 = EntropyPooling(
        mean_views=["AMD >= BAC", "JPM <= prior(JPM) * 0.8"],
        cvar_views=["GE == 0.12"],
    )
    
    opinion_pooling = OpinionPooling(
        estimators=[("opinion_1", opinion_1), ("opinion_2", opinion_2)],
        opinion_probabilities=[0.4, 0.5],
    )
    
    opinion_pooling.fit(X)
    
    print(opinion_pooling.return_distribution_.sample_weight)
    
    # CVaR Risk Parity optimization on opinion Pooling
    model = RiskBudgeting(
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=opinion_pooling
    )
    model.fit(X)
    print(model.weights_)
    
    # Stress Test the Portfolio
    opinion_1 = EntropyPooling(cvar_views=["AMD == 0.05"])
    opinion_2 = EntropyPooling(cvar_views=["AMD == 0.10"])
    opinion_pooling = OpinionPooling(
        estimators=[("opinion_1", opinion_1), ("opinion_2", opinion_2)],
        opinion_probabilities=[0.6, 0.4],
    )
    opinion_pooling.fit(X)
    
    stressed_dist = opinion_pooling.return_distribution_

    stressed_ptf = model.predict(stressed_dist)

Combining Multiple Prior Estimators
***********************************
Prior estimators can be composed to build more sophisticated models. For example,
you can create a Black & Litterman Factor Model by supplying :class:`BlackLitterman`
as the prior estimator of the :class:`FactorModel` and impose views on the factors.

**Example:**

Below is a factor model that estimates the **assets'** expected returns and covariance
matrix, where the **factors'** expected returns and covariance are themselves
estimated via a Black & Litterman model that incorporates the analyst's views on those
**factors**.

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
    print(model.return_distribution_)



**Example:**

By combining :class:`SyntheticData` with :class:`FactorModel` you can generate
synthetic data of your factors then project them to your assets.
This is often used for factor stress test.

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset, load_factors_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.distribution import VineCopula
    from skfolio.optimization import MeanRisk
    from skfolio.prior import FactorModel, SyntheticData
    from skfolio import RiskMeasure

    # Load historical prices and convert them to returns
    prices = load_sp500_dataset()
    factors = load_factors_dataset()
    X, y = prices_to_returns(prices, factors)


    # Minimum CVaR optimization on Stressed Factors
    vine = VineCopula(central_assets=["QUAL"], log_transform=True, n_jobs=-1)
    factor_prior = SyntheticData(
        distribution_estimator=vine,
        n_samples=10000,
        sample_args=dict(conditioning={"QUAL": -0.2}),
    )
    factor_model = FactorModel(factor_prior_estimator=factor_prior)
    model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model)
    model.fit(X, y)
    print(model.weights_)

    # Stress Test the Portfolio
    factor_model.set_params(factor_prior_estimator__sample_args=dict(
        conditioning={"QUAL": -0.5}
    ))
    factor_model.fit(X,y)
    stressed_dist = factor_model.return_distribution_
    stressed_ptf = model.predict(stressed_dist)

**Example:**

To impose extreme views using Entropy Pooling on a sparse historical distribution,
we must generate synthetic data capable of extrapolating tail dependencies.
This can be achieved by combining :class:`EntropyPooling` with :class:`SyntheticData`:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.distribution import VineCopula
    from skfolio.prior import EntropyPooling, SyntheticData

    # Load historical prices and convert them to returns
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    # Regular Vine Copula and sampling of 100,000 synthetic returns
    synth = SyntheticData(
        n_samples=100_000,
        distribution_estimator=VineCopula(log_transform=True, n_jobs=-1, random_state=0)
    )

    # Entropy Pooling by imposing a CVaR-95% of 10% on Apple
    entropy_pooling = EntropyPooling(
        prior_estimator=factor_synth,
        cvar_views=["AAPL == 0.10"],
    )

    entropy_pooling.fit(X)


**Example:**

Instead of applying extreme Entropy Pooling views directly to asset returns, we can
embed it within a Factor Model.
This allows us to impose views on factor data such at the quality factor "QUAL".
This can be achieved by combining :class:`EntropyPooling` with :class:`SyntheticData`
and with :class:`FactorModel`:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset, load_factors_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.distribution import VineCopula
    from skfolio.optimization import MeanRisk
    from skfolio.prior import FactorModel, SyntheticData
    from skfolio import RiskMeasure

    # Load historical prices and convert them to returns
    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()
    X, factors = prices_to_returns(prices, factor_prices)

    # Regular Vine Copula and sampling of 100,000 synthetic factor returns
    factor_synth = SyntheticData(
        n_samples=100_000,
        distribution_estimator=VineCopula(log_transform=True, n_jobs=-1, random_state=0)
    )

    # Entropy Pooling by imposing a CVaR-95% of 10% on the Quality factor
    factor_entropy_pooling = EntropyPooling(
        prior_estimator=factor_synth,
        cvar_views=["QUAL == 0.10"],
    )

    factor_entropy_pooling.fit(X, factors)

