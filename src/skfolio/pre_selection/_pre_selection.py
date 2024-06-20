"""pre-selection estimators module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.measures import RatioMeasure
from skfolio.population import Population
from skfolio.portfolio import Portfolio


class DropCorrelated(skf.SelectorMixin, skb.BaseEstimator):
    """Transformer for dropping highly correlated assets.

    Simply removing all correlation pairs above the threshold will remove more assets
    than necessary and a naive sequential removal is suboptimal and depends on the
    initial assets ordering.

    Let's suppose X,Y,Z are three random variables with corr(X,Y) and corr(X,Z) above
    the threshold and corr(Y,Z) below.
    The first approach would remove X,Y,Z and the second approach would remove either
    Y and Z or X depending on the initial ordering.

    To avoid these shortcomings, we implement the below algorithm:

        * Step 1: select all correlation pairs above the threshold.
        * Step 2: sort all the selected correlation pairs from highest to lowest.
        * Step 3: for each pair, if none of the two assets has been removed, keep the
          asset with the lowest average correlation against the other assets.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold. The default value is `0.95`.

    absolute : bool, default=False
        If this is set to True, we take the absolute value of the correlation. This has
        for effect to also include negatively correlated assets.

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
        Boolean array indicating which assets are remaining.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    to_keep_: np.ndarray

    def __init__(self, threshold: float = 0.95, absolute: bool = False):
        self.threshold = threshold
        self.absolute = absolute

    def fit(self, X: npt.ArrayLike, y=None):
        """Run the correlation transformer and get the appropriate assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : DropCorrelated
            Fitted estimator.
        """
        X = self._validate_data(X)
        if not -1 <= self.threshold <= 1:
            raise ValueError("`threshold` must be between -1 and 1")

        n_assets = X.shape[1]
        corr = np.corrcoef(X.T)
        mean_corr = corr.mean(axis=0)

        triu_idx = np.triu_indices(n_assets, 1)

        # select all correlation pairs above the threshold
        selected_idx = np.argwhere(corr[triu_idx] > self.threshold).flatten()

        # sort all the selected correlation pairs from highest to lowest
        selected_idx = selected_idx[np.argsort(-corr[triu_idx][selected_idx])]

        # for each pair, if none of the two assets has been removed, keep the asset with
        # the lowest average correlation with other assets
        to_remove = set()
        for idx in selected_idx:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            if i not in to_remove and j not in to_remove:
                if mean_corr[i] > mean_corr[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
        self.to_keep_ = ~np.isin(np.arange(n_assets), list(to_remove))
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_


class SelectKExtremes(skf.SelectorMixin, skb.BaseEstimator):
    """Transformer for selecting the `k` best or worst assets.

    Keep the `k` best or worst assets according to a given measure.

    Parameters
    ----------
    k : int, default=10
        Number of assets to select. If `k` is higher than the number of assets, all
        assets are selected.

    measure : Measure, default=RatioMeasure.SHARPE_RATIO
        The :ref:`measure <measures_ref>` used to sort the assets.
        The default is `RatioMeasure.SHARPE_RATIO`.

    highest : bool, default=True
        If this is set to True, the `k` assets with the highest `measure` are selected,
        otherwise it is the `k` lowest.

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
       Boolean array indicating which assets are remaining.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    to_keep_: np.ndarray

    def __init__(
        self,
        k: int = 10,
        measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        highest: bool = True,
    ):
        self.k = k
        self.measure = measure
        self.highest = highest

    def fit(self, X: npt.ArrayLike, y=None) -> "SelectKExtremes":
        """Run the SelectKExtremes transformer and get the appropriate assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SelectKExtremes
            Fitted estimator.
        """
        X = self._validate_data(X)
        k = int(self.k)
        if k <= 0:
            raise ValueError("`k` must be strictly positive")
        n_assets = X.shape[1]
        # Build a population of single assets portfolio
        population = Population([])
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1
            population.append(Portfolio(X=X, weights=weights))

        selected = population.sort_measure(measure=self.measure, reverse=self.highest)[
            :k
        ]
        selected_idx = [x.nonzero_assets_index[0] for x in selected]
        self.to_keep_ = np.isin(np.arange(n_assets), selected_idx)
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_


class SelectNonDominated(skf.SelectorMixin, skb.BaseEstimator):
    """Transformer for selecting non dominated assets.

    Pre-selection based on the Assets Preselection Process 2 [1]_.

    Good single asset (for example with high return and low risk) is likely to
    contribute to the final optimized portfolio. Each asset is considered as a portfolio
    and these assets are ranked using the non-domination sorting method. The selection
    is based on the ranks assigned to each asset based on their fitness until the number
    of selected assets reaches the user-defined number.

    Considering only the fitness of individual asset is insufficient because a pair of
    negatively correlated assets has the potential to reduce the risk. Therefore,
    negatively correlated pairs of assets are also considered.

    Parameters
    ----------
    min_n_assets : int, optional
        The minimum number of assets to select. If `min_n_assets` is reached before the
        end of the current non-dominated front, we return the remaining assets of this
        front. This is because all assets in the same front have same rank.
        The default (`None`) is to select the first front.

    threshold : float, default=0.0
        Asset pair with a correlation below this threshold are included in the
        non-domination sorting. The default value is `0.0`.

    fitness_measures : list[Measure], optional
        A list of :ref:`measure <measures_ref>` used to compute the portfolio fitness.
        The fitness is used to compare portfolios in terms of domination, compute the
        pareto fronts and run the portfolio selection using non-denominated sorting.
        The default (`None`) is to use the list [PerfMeasure.MEAN, RiskMeasure.VARIANCE]

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
        Boolean array indicating which assets are remaining.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1]  "Large-Scale Portfolio Optimization Using Multi-objective Evolutionary
        Algorithms and Preselection Methods",
        B.Y. Qu and Q.Zhou (2017).
    """

    to_keep_: np.ndarray

    def __init__(
        self,
        min_n_assets: int | None = None,
        threshold: float = -0.5,
        fitness_measures: list[skt.Measure] | None = None,
    ):
        self.min_n_assets = min_n_assets
        self.threshold = threshold
        self.fitness_measures = fitness_measures

    def fit(self, X: npt.ArrayLike, y=None):
        """Run the Non Dominated transformer and get the appropriate assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SelectNonDominated
            Fitted estimator.
        """
        X = self._validate_data(X)
        if not -1 <= self.threshold <= 1:
            raise ValueError("`threshold` must be between -1 and 1")
        n_assets = X.shape[1]

        if self.min_n_assets is not None and self.min_n_assets >= n_assets:
            self.to_keep_ = np.full(n_assets, True)
            return self

        # Build a population of portfolio
        population = Population([])
        # Add single assets
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1
            population.append(
                Portfolio(X=X, weights=weights, fitness_measures=self.fitness_measures)
            )

        # Add pairs with correlation below threshold with minimum variance
        # ptf_variance = sigma1^2 w1^2 + sigma2^2 w2^2 + 2 sigma12 w1 w2 (1)
        # with w1 + w2 = 1
        # To find the minimum we substitute w2 = 1 - w1 in (1) and differentiate with
        # respect to w1 and set to zero.
        # By solving the obtained equation, we get:
        # w1 = (sigma2^2 - sigma12) / (sigma1^2 + sigma2^2 - 2 sigma12)
        # w2 = 1 - w1

        corr = np.corrcoef(X.T)
        covariance = np.cov(X.T)
        for i, j in zip(*np.triu_indices(n_assets, 1), strict=True):
            if corr[i, j] < self.threshold:
                cov = covariance[i, j]
                var1 = covariance[i, i]
                var2 = covariance[j, j]
                weights = np.zeros(n_assets)
                weights[i] = (var2 - cov) / (var1 + var2 - 2 * cov)
                weights[j] = 1 - weights[i]
                population.append(
                    Portfolio(
                        X=X, weights=weights, fitness_measures=self.fitness_measures
                    )
                )

        fronts = population.non_denominated_sort(
            first_front_only=self.min_n_assets is None
        )
        new_assets_idx = set()
        i = 0
        while i < len(fronts):
            if (
                self.min_n_assets is not None
                and len(new_assets_idx) > self.min_n_assets
            ):
                break
            for idx in fronts[i]:
                new_assets_idx.update(population[idx].nonzero_assets_index)
            i += 1
        self.to_keep_ = np.isin(np.arange(n_assets), list(new_assets_idx))
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_
