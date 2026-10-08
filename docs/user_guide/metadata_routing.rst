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


`ImpliedCovariance` requires an explicit request for `implied_vol`:

.. code-block:: python

    model = MeanRisk(
        prior_estimator=EmpiricalPrior(
            covariance_estimator=ImpliedCovariance(
            ).set_fit_request(implied_vol=True)
        )
    )
    model.fit(X, implied_vol=implied_vol)
    print(model.weights_)


.. _default_metadata_requests:

Default Requests
****************

:class:`~skfolio.moments.EWMu`, :class:`~skfolio.moments.EWCovariance`,
:class:`~skfolio.moments.EWVariance`,
:class:`~skfolio.moments.RegimeAdjustedEWCovariance` and
:class:`~skfolio.moments.RegimeAdjustedEWVariance` request `active_mask` by
default for both `fit` and `partial_fit`. With metadata routing enabled, pass
the mask to the enclosing prior or optimizer without configuring each child:

.. code-block:: python

    from skfolio.moments import EWCovariance, EWMu

    prior = EmpiricalPrior(
        mu_estimator=EWMu(),
        covariance_estimator=EWCovariance(),
    )
    prior.partial_fit(X_batch, active_mask=active_batch)

The mask is optional. Omitting it treats all assets as active.
Use `set_fit_request` or `set_partial_fit_request` to override a default:
`False` stops forwarding the mask to that child, `None` raises if it is
supplied, and a string requests it under another name. These settings affect
routing through a parent, not direct calls to the child.

:class:`~skfolio.prior.CharacteristicsFactorModel` also requests
`characteristics` by default, and :class:`~skfolio.prior.TimeSeriesFactorModel`
requests `factors` by default.

Masks must match the observations and columns consumed by each estimator.
When a factor prior and an asset learner need different masks, use an alias
such as `set_fit_request(active_mask="factor_active_mask")` on the factor
moment estimators.

`CharacteristicsFactorModel` supplies its residual variance and correlation
estimators with masks derived from `characteristics`. To pass a separate
`active_mask` to another learner in the same model, disable the external mask
requests on these residual estimators. Their internally supplied panel masks
are still used.


.. _distance_metadata_routing:

Distance Inputs in HRP and Schur
********************************

HRP and Schur can compute distances from prior scenarios, the prior covariance or
input returns, as described in :ref:`asset_seriation`. These inputs differ in how
observation metadata is routed.

During `partial_fit`, with `distance_from_prior=True`, distances fitted on prior
scenarios receive neither the incoming targets nor the incoming observation metadata.
For example, an `active_mask` describes the new return batch, which may differ from
the prior's scenarios. If that distance uses an EW covariance estimator, setting its
`set_fit_request(active_mask=False)` allows the mask to reach the prior without
forwarding it to the distance.

Scenario distances use the prior's `sample_weight` when their configured `fit`
supports it. For `CovarianceDistance`, this requires the covariance estimator's
`sample_weight` metadata request. Otherwise, the distance remains unweighted even
when portfolio risk uses weighted scenarios.

With `distance_from_prior=False`, the distance receives targets and requested
metadata aligned with the optimizer's input returns. An EW covariance estimator
requests `active_mask` by default. Precomputed covariance distances and seriation
receive no observation metadata, so `CovarianceDistance("precomputed")` needs no
mask configuration.

During batch `fit`, distances fitted on prior scenarios retain observation metadata
routing. The caller is responsible for aligning that metadata with the scenarios.
