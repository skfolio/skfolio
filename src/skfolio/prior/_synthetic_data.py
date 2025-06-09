"""Synthetic Data Prior estimator."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.distribution import VineCopula
from skfolio.prior._base import BasePrior
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.utils.tools import check_estimator


class SyntheticData(BasePrior):
    """Synthetic Data Estimator.

    The Synthetic Data model estimates a :class:`~skfolio.prior.ReturnDistribution` by
    fitting a `distribution_estimator` and sampling new returns data from it.

    The default ``distribution_estimator`` is a Regular Vine Copula model. Other common
    choices are Generative Adversarial Networks (GANs) or Variational Autoencoders
    (VAEs).

    This class is particularly useful when the historical distribution tail dependencies
    are sparse and need extrapolation for tail optimizations or when optimizing under
    conditional or stressed scenarios.

    Parameters
    ----------
    distribution_estimator : BaseEstimator, optional
        Estimator to model the distribution of asset returns. It must inherit from
        `BaseEstimator` and implements a `sample` method. If None, the default
        `VineCopula()` model is used.

    n_samples : int, default=1000
        Number of samples to generate from the `distribution_estimator`, default is
        1000.

    sample_args : dict, optional
        Additional keyword arguments to pass to the `sample` method of the
        `distribution_estimator`.

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` to be used by the optimization
        estimators, containing the assets syntehtic data distribution and moments
        estimation.

    distribution_estimator_ : BaseEstimator
        The fitted distribution estimator.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.datasets import load_sp500_dataset, load_factors_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution import VineCopula
    >>> from skfolio.optimization import MeanRisk
    >>> from skfolio.prior import FactorModel, SyntheticData
    >>> from skfolio import RiskMeasure
    >>>
    >>> # Load historical prices and convert them to returns
    >>> prices = load_sp500_dataset()
    >>> factors = load_factors_dataset()
    >>> X, y = prices_to_returns(prices, factors)
    >>>
    >>> # Instanciate the SyntheticData model and fit it
    >>> model = SyntheticData()
    >>> model.fit(X)
    >>> print(model.return_distribution_)
    >>>
    >>> # Minimum CVaR optimization on synthetic returns
    >>> model = MeanRisk(
    ...    risk_measure=RiskMeasure.CVAR,
    ...    prior_estimator=SyntheticData(
    ...        distribution_estimator=VineCopula(log_transform=True, n_jobs=-1),
    ...        n_samples=2000,
    ...    )
    ... )
    >>> model.fit(X)
    >>> print(model.weights_)
    >>>
    >>> # Minimum CVaR optimization on Stressed Factors
    >>> factor_model = FactorModel(
    ...    factor_prior_estimator=SyntheticData(
    ...        distribution_estimator=VineCopula(
    ...            central_assets=["QUAL"],
    ...            log_transform=True,
    ...            n_jobs=-1,
    ...        ),
    ...        n_samples=5000,
    ...        sample_args=dict(conditioning={"QUAL": -0.2}),
    ...    )
    ... )
    >>> model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model)
    >>> model.fit(X, y)
    >>> print(model.weights_)
    >>>
    >>> # Stress Test the Portfolio
    >>> factor_model.set_params(factor_prior_estimator__sample_args=dict(
    ...     conditioning={"QUAL": -0.5}
    ... ))
    >>> factor_model.fit(X,y)
    >>> stressed_dist = factor_model.return_distribution_
    >>> stressed_ptf = model.predict(stressed_dist)
    """

    distribution_estimator_: skb.BaseEstimator
    prior_estimator_: BasePrior
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        distribution_estimator: skb.BaseEstimator | None = None,
        n_samples: int = 1000,
        sample_args: dict | None = None,
    ):
        self.distribution_estimator = distribution_estimator
        self.n_samples = n_samples
        self.sample_args = sample_args

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            distance_estimator=self.distribution_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "SyntheticData":
        """Fit the Synthetic Data estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : SyntheticData
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        self.distribution_estimator_ = check_estimator(
            self.distribution_estimator,
            default=VineCopula(),
            check_type=skb.BaseEstimator,
        )
        _check_sample_method(self.distribution_estimator_)

        # fitting distribution estimator on prior returns
        # noinspection PyUnresolvedReferences
        self.distribution_estimator_.fit(
            X, y, **routed_params.distribution_estimator.fit
        )

        # We validate after all models have been fitted to keep feature names
        # information.
        skv.validate_data(self, X)

        # sample from the distribution estimator
        sample_args = self.sample_args if self.sample_args is not None else {}
        # noinspection PyUnresolvedReferences
        synthetic_data = self.distribution_estimator_.sample(
            n_samples=self.n_samples, **sample_args
        )

        # When performing conditional sampling, the conditioning samples are often
        # constant. To avoid null variance, we add a small white noise.
        constant_returns = np.var(synthetic_data, axis=0) < 1e-14
        if np.any(constant_returns):
            noise = 1e-6 * np.random.randn(len(synthetic_data), 1)
            synthetic_data[:, constant_returns] += noise

        # Fit empirical posterior estimator
        posterior_estimator = EmpiricalPrior()
        posterior_estimator.fit(synthetic_data)
        self.return_distribution_ = posterior_estimator.return_distribution_

        return self


def _check_sample_method(distribution_estimator: skb.BaseEstimator) -> None:
    """Check that the distribution_estimator implements a valid 'sample' method.

    This helper function verifies that the given estimator has a callable 'sample'
    method and that this method accepts an 'n_samples' parameter.

    Parameters
    ----------
    distribution_estimator : BaseEstimator
        The estimator whose 'sample' method is to be validated.

    Raises
    ------
    ValueError
        If the 'sample' method is missing or does not have an 'n_samples' parameter.
    """
    # Get the 'sample' attribute; if it doesn't exist, return False.
    sample_method = getattr(distribution_estimator, "sample", None)
    if sample_method is None or not callable(sample_method):
        raise ValueError(
            f"The distribution_estimator {distribution_estimator} must implement a "
            "`sample` method"
        )

    sig = inspect.signature(sample_method)

    # Check if the parameter 'n_samples' is in the method's parameters.
    if "n_samples" not in sig.parameters:
        raise ValueError(
            "The `sample` method of the distribution_estimator "
            f"{distribution_estimator} must have `n_samples` as parameter"
        )
