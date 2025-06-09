"""Opinion Pooling estimator."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Vincent Maladière, Matteo Manzi, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.special as scs
import scipy.special as sp
import sklearn as sk
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.parallel as skp
import sklearn.utils.validation as skv

import skfolio.measures as sm
from skfolio.prior._base import BasePrior, ReturnDistribution
from skfolio.utils.composition import BaseComposition
from skfolio.utils.tools import check_estimator, fit_single_estimator


class OpinionPooling(BasePrior, BaseComposition):
    r"""Opinion Pooling estimator.

    Opinion Pooling (also called Belief Aggregation or Risk Aggregation) is a process
    in which different probability distributions (opinions), produced by different
    experts, are combined to yield a single probability distribution (consensus).

    Expert opinions (also called individual prior distributions) can be
    **elicited** from domain experts or **derived** from quantitative analyses.

    The `OpinionPooling` estimator takes a list of prior estimators, each of which
    produces scenario probabilities (which we use as `sample_weight`), and pools them
    into a single consensus probability .

    You can choose between linear (arithmetic) pooling or logarithmic (geometric)
    pooling, and optionally apply robust pooling using a Kullback-Leibler divergence
    penalty to down-weight experts whose views deviate strongly from the group
    consensus.

    Parameters
    ----------
    estimators : list of (str, BasePrior)
        A list of :ref:`prior estimators <prior>` representing opinions to be pooled
        into a single consensus.
        Each element of the list is defined as a tuple of string (i.e. name) and an
        estimator instance. Each must expose `sample_weight` such as in
        :class:`~skfolio.prior.EntropyPooling`.

    opinion_probabilities : array-like of float, optional
        Probability mass assigned to each opinion, in [0,1] summing to ≤1.
        Any leftover mass is assigned to the uniform (uninformative) prior.
        The default (None), is to assign the same probability to each opinion.

    prior_estimator : BasePrior, optional
        Common prior for all `estimators`. If provided, each estimator from `estimators`
        will be fitted using this common prior before pooling. Setting `prior_estimator`
        inside individual `estimators` is disabled to avoid mixing different prior
        scenarios (each estimator must have the same underlying distribution).
        For example, using `prior_estimator = SyntheticData(n_samples=10_000)` will
        generate 10,000 synthetic data points from a Vine Copula before fitting the
        estimators on this common distribution.

    is_linear_pooling : bool, default=True
        If True, combine each opinion via Linear Opinion Pooling
        (arithmetic mean); if False, use Logarithmic Opinion Pooling (geometric
        mean).

        Linear Opinion Pooling:
            * Retains all nonzero support (no "zero-forcing").
            * Produces an averaging that is more evenly spread across all expert opinions.

        Logarithmic Opinion Pooling:
            * Zero-Preservation. Any scenario assigned zero probability by any expert
              remains zero in the aggregate.
            * Information-Theoretic Optimality. Yields the distribution that minimizes
              the weighted sum of KL-divergences from each expert's distribution.
            * Robust to Extremes: down-weight extreme or contrarian views more severely.

    divergence_penalty : float, default=0.0
        Non-negative factor (:math:`\alpha`) that penalizes each opinion's divergence
        from the group consensus, yielding more robust pooling.
        A higher value more strongly down-weights deviating opinions.

        The robust opinion probabilities are given by:

        .. math::
            \tilde{p}_i = \frac{p_i \exp\bigl(-\alpha D_i\bigr)}
            {\displaystyle \sum_{k=1}^N p_k \exp\bigl(-\alpha D_k\bigr)}
            \quad\text{for }i = 1,\dots,N

        where

        * :math:`N` is the number of experts `len(estimators)`

        * :math:`M` is the number of scenarios `len(observations)`

        * :math:`D_i` is the KL-divergence of expert *i*'s distribution from consensus:

          .. math::
             D_i = \mathrm{KL}\bigl(w_i \,\|\, c\bigr)
                 = \sum_{j=1}^M w_{ij}\,\ln\!\frac{w_{ij}}{c_j}
             \quad\text{for }i = 1,\dots,N.

        * :math:`w_i` is the sample-weight vector (scenario probabilities) from expert
          *i*,  with :math:`\sum_{j=1}^M w_{ij} = 1`.

        * :math:`p_i` is the initial opinion probability of expert *i*, with
          :math:`\sum_{i=1}^N p_i \le 1` (any leftover mass goes to a uniform prior).

        * :math:`c_j` is the consensus of scenario :math:`j`:

          .. math::
             c_j = \sum_{i=1}^N p_i \, w_{ij} \quad\text{for }j = 1,\dots,M.

    n_jobs : int, optional
        The number of jobs to run in parallel for `fit` of all `estimators`.
        The value `-1` means using all processors.
        The default (`None`) means 1 unless in a `joblib.parallel_backend` context.

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` to be used by the optimization
        estimators, containing the assets distribution, moments estimation and the
        opinion-pooling sample weights.

    estimators_ : list[BasePrior]
        The elements of the `estimators` parameter, having been fitted on the
        training data.

    named_estimators_ : dict[str, BasePrior]
        Attribute to access any fitted sub-estimators by name.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator` if provided.

    opinion_probabilities_ : ndarray of shape (n_opinions,)
        Final opinion probabilities after applying the KL-divergence penalty.
        If the initial `opinion_probabilities` doesn't sum to one, the last element of
        `opinion_probabilities_` is the probability assigned to the uniform prior.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Probabilistic opinion pooling generalized",
            Social Choice and Welfare, Dietrich & List (2017)

    .. [2] "Opinion Aggregation and Individual Expertise",
            Oxford University Press, Martini & Sprenger (2017)

    .. [3] "Rational Decisions",
            Journal of the Royal Statistical Society, Good  (1952)

    Examples
    --------
    For a full tutorial on entropy pooling, see :ref:`sphx_glr_auto_examples_entropy_pooling_plot_2_opinion_pooling.py`.

    >>> from skfolio import RiskMeasure
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.prior import EntropyPooling, OpinionPooling
    >>> from skfolio.optimization import RiskBudgeting
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>>
    >>> # We consider two expert opinions, each generated via Entropy Pooling with
    >>> # user-defined views.
    >>> # We assign probabilities of 40% to Expert 1, 50% to Expert 2, and by default
    >>> # the remaining 10% is allocated to the prior distribution:
    >>> opinion_1 = EntropyPooling(cvar_views=["AMD == 0.10"])
    >>> opinion_2 = EntropyPooling(
    ...     mean_views=["AMD >= BAC", "JPM <= prior(JPM) * 0.8"],
    ...     cvar_views=["GE == 0.12"],
    ... )
    >>>
    >>> opinion_pooling = OpinionPooling(
    ...     estimators=[("opinion_1", opinion_1), ("opinion_2", opinion_2)],
    ...     opinion_probabilities=[0.4, 0.5],
    ... )
    >>>
    >>> opinion_pooling.fit(X)
    >>>
    >>> print(opinion_pooling.return_distribution_.sample_weight)
    >>>
    >>> # CVaR Risk Parity optimization on opinion Pooling
    >>> model = RiskBudgeting(
    ...     risk_measure=RiskMeasure.CVAR,
    ...     prior_estimator=opinion_pooling
    ... )
    >>> model.fit(X)
    >>> print(model.weights_)
    >>>
    >>> # Stress Test the Portfolio
    >>> opinion_1 = EntropyPooling(cvar_views=["AMD == 0.05"])
    >>> opinion_2 = EntropyPooling(cvar_views=["AMD == 0.10"])
    >>> opinion_pooling = OpinionPooling(
    ...     estimators=[("opinion_1", opinion_1), ("opinion_2", opinion_2)],
    ...     opinion_probabilities=[0.6, 0.4],
    ... )
    >>> opinion_pooling.fit(X)
    >>>
    >>> stressed_dist = opinion_pooling.return_distribution_
    >>>
    >>> stressed_ptf = model.predict(stressed_dist)
    """

    estimators_: list[BasePrior]
    named_estimators_: dict[str, BasePrior]
    opinion_probabilities_: np.ndarray
    prior_estimator_: BasePrior
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        estimators: list[tuple[str, BasePrior]],
        opinion_probabilities: list[float] | None = None,
        prior_estimator: BasePrior | None = None,
        is_linear_pooling: bool = True,
        divergence_penalty: float = 0.0,
        n_jobs: int | None = None,
    ):
        self.estimators = estimators
        self.opinion_probabilities = opinion_probabilities
        self.prior_estimator = prior_estimator
        self.divergence_penalty = divergence_penalty
        self.is_linear_pooling = is_linear_pooling
        self.n_jobs = n_jobs

    @property
    def named_estimators(self):
        """Dictionary to access any fitted sub-estimators by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return sku.Bunch(**dict(self.estimators))

    def _validate_estimators(self) -> tuple[list[str], list[BasePrior]]:
        """Validate the `estimators` parameter.

        Returns
        -------
        names : list[str]
            The list of estimators names.
        estimators : list[BaseOptimization
            The list of optimization estimators.
        """
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators, strict=True)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        for estimator in estimators:
            if getattr(estimator, "prior_estimator", None) is not None:
                raise ValueError(
                    "Cannot set `prior_estimator` on individual estimators within "
                    "`OpinionPooling` to avoid mixing different prior scenarios. "
                    "Please leave those as `None` and specify your prior directly via "
                    "the `prior_estimator` parameter of the `OpinionPooling` class."
                )

        return names, estimators

    def set_params(self, **params):
        """Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("estimators", **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("estimators", deep=deep)

    def get_metadata_routing(self):
        router = skm.MetadataRouter(owner=self.__class__.__name__)
        for name, estimator in self.estimators:
            router.add(
                **{name: estimator},
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "OpinionPooling":
        """Fit the Opinion Pooling estimator.

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
        self : OpinionPooling
           Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        skv.validate_data(self, X)

        names, all_estimators = self._validate_estimators()

        opinion_probabilities = self._validate_opinion_probabilities()

        if self.prior_estimator is not None:
            self.prior_estimator_ = check_estimator(
                self.prior_estimator,
                default=None,
                check_type=BasePrior,
            )
            # fitting prior estimator
            self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
            returns = self.prior_estimator_.return_distribution_.returns
            # To keep the asset_names
            if hasattr(self, "feature_names_in_"):
                returns = pd.DataFrame(returns, columns=self.feature_names_in_)
        else:
            returns = X

        # Fit the prior estimators on the whole training data. Those
        # prior estimators will be used to retrieve the sample weights.
        self.estimators_ = skp.Parallel(n_jobs=self.n_jobs)(
            skp.delayed(fit_single_estimator)(
                sk.clone(est), returns, None, routed_params[name]["fit"]
            )
            for name, est in zip(names, all_estimators, strict=True)
        )

        self.named_estimators_ = {
            name: estimator
            for name, estimator in zip(names, self.estimators_, strict=True)
        }

        sample_weights = []
        for estimator in self.estimators_:
            if estimator.return_distribution_.sample_weight is None:
                raise ValueError(
                    f"Estimator `{estimator.__class__.__name__}` did not produce "
                    "a `return_distribution_.sample_weight`. OpinionPooling requires "
                    "each estimator to expose sample weights (e.g. via EntropyPooling)."
                )
            sample_weights.append(estimator.return_distribution_.sample_weight)
        sample_weights = np.array(sample_weights)

        returns = np.asarray(returns)
        n_observations = len(returns)

        # Add the remaining part of the opinion_probabilities to the uniform prior
        q_weight = 1.0 - opinion_probabilities.sum()
        if q_weight > 1e-8:
            opinion_probabilities = np.append(opinion_probabilities, q_weight)
            q = np.ones(n_observations) / n_observations
            sample_weights = np.vstack((sample_weights, q))

        opinion_probabilities = self._compute_robust_opinion_probabilities(
            opinion_probabilities=opinion_probabilities, sample_weights=sample_weights
        )

        if self.is_linear_pooling:
            sample_weight = opinion_probabilities @ sample_weights
        else:
            # let exact 0 in sample weights flow through
            with np.errstate(divide="ignore"):
                u = opinion_probabilities @ np.log(sample_weights)
                sample_weight = np.exp(u - sp.logsumexp(u))

        self.opinion_probabilities_ = opinion_probabilities
        self.return_distribution_ = ReturnDistribution(
            mu=sm.mean(returns, sample_weight=sample_weight),
            covariance=np.cov(returns, rowvar=False, aweights=sample_weight),
            returns=returns,
            sample_weight=sample_weight,
        )
        return self

    def _validate_opinion_probabilities(self) -> np.ndarray:
        """Validate `opinion_probabilities`."""
        n_opinions = len(self.estimators)
        if self.opinion_probabilities is None:
            return np.ones(n_opinions) / n_opinions

        opinion_probabilities = np.asarray(self.opinion_probabilities)

        if len(opinion_probabilities) != n_opinions:
            raise ValueError(
                f"`opinion_probabilities` length ({len(opinion_probabilities)}) "
                f"does not match number of estimators ({n_opinions})."
            )

        if np.any(opinion_probabilities < 0) or np.any(opinion_probabilities > 1):
            raise ValueError(
                "`The entries of `opinion_probabilities` must be between 0 and 1"
            )
        if opinion_probabilities.sum() > 1.0:
            raise ValueError(
                "The entries of `opinion_probabilities` must sum to at most 1; "
                "any remaining mass (1-sum) is allocated to the uniform prior."
            )
        return opinion_probabilities

    def _compute_robust_opinion_probabilities(
        self, opinion_probabilities: np.ndarray, sample_weights: np.ndarray
    ) -> np.ndarray:
        """Compute the robust `opinion_probabilities` using KL-divergence."""
        if self.divergence_penalty < 0:
            raise ValueError("`divergence_penalty` cannot be negative")

        if self.divergence_penalty == 0:
            return opinion_probabilities

        consensus = opinion_probabilities @ sample_weights
        divergences = np.sum(scs.rel_entr(sample_weights, consensus), axis=1)
        opinion_probabilities *= np.exp(-self.divergence_penalty * divergences)
        opinion_probabilities /= opinion_probabilities.sum()
        return opinion_probabilities
