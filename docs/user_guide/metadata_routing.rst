.. _metadata_routing:

.. currentmodule:: skfolio

****************
Metadata Routing
****************
This document shows how you can use the metadata routing mechanism to route metadata
to the estimators consuming them.
For a complete explanation, you can refer to the `scikit-learn documentation <https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_metadata_routing.html#sphx-glr-auto-examples-miscellaneous-plot-metadata-routing-py>`_


A full example is available here: :ref:`sphx_glr_auto_examples_metadata_routing_plot_1_implied_volatility.py`

Let's suppose you use the :class:`~skfolio.moments.ImpliedCovariance` estimator
inside a :class:`~skfolio.optimization.MeanRisk` estimator.
In addition to the assets' returns `X`, the `ImpliedCovariance` estimator also needs
the assets' implied volatilities passed to its `fit` method.
In order to route the implied volatilities time series from the `MeanRisk` estimator
to the `ImpliedCovariance` estimator, we need metadata routing.

First, a few imports and some random data for the rest of the script:

.. code-block:: python

    from sklearn import set_config

    from skfolio.moments import ImpliedCovariance
    from skfolio.optimization import MeanRisk
    from skfolio.prior import EmpiricalPrior
    from skfolio.preprocessing import prices_to_returns
    from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset

    prices = load_sp500_dataset()
    implied_vol = load_sp500_implied_vol_dataset()

    X = prices_to_returns(prices)
    X = X.loc["2010":]

Metadata routing is available only if explicitly enabled:

.. code-block:: python

    set_config(enable_metadata_routing=True)


Then, in order to route the metadata, you must use `set_fit_request`:

.. code-block:: python

    model = MeanRisk(
        prior_estimator=EmpiricalPrior(
            covariance_estimator=ImpliedCovariance(
            ).set_fit_request(implied_vol=True)
        )
    )
    model.fit(X, implied_vol=implied_vol)
    print(model.weights_)


