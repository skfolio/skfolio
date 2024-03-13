"""Distance Estimators"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.distance as scd
import scipy.stats as sct
import sklearn.metrics as skm

from skfolio.distance._base import BaseDistance
from skfolio.moments import BaseCovariance, GerberCovariance
from skfolio.utils.stats import (
    NBinsMethod,
    cov_to_corr,
    n_bins_freedman,
    n_bins_knuth,
)
from skfolio.utils.tools import check_estimator


class PearsonDistance(BaseDistance):
    r"""Pearson Distance estimator.

    The codependence is computed from the Pearson correlation to which is applied a
    power and/or absolute transformation.
    This codependence is then used to compute the distance matrix.
    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`

    Parameters
    ----------
    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        Lòpez de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1):
        self.absolute = absolute
        self.power = power

    def fit(self, X: npt.ArrayLike, y=None) -> "PearsonDistance":
        """Fit the Pearson Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : PearsonDistance
            Fitted estimator.
        """
        X = self._validate_data(X)
        corr = np.corrcoef(X.T)
        self.codependence_, self.distance_ = _corr_to_distance(
            corr, absolute=self.absolute, power=self.power
        )
        return self


class KendallDistance(BaseDistance):
    r"""Kendall Distance estimator.

    The codependence is computed from the Kendall correlation to which is applied a
    power and/or absolute transformation.
    This codependence is then used to compute the distance matrix.
    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`

    Parameters
    ----------
    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.
        The default is `False`.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.
        The default value is `1`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        Lòpez de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1):
        self.absolute = absolute
        self.power = power

    def fit(self, X: npt.ArrayLike, y=None) -> "KendallDistance":
        """Fit the Kendall estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : KendallDistance
            Fitted estimator.
        """
        X = self._validate_data(X)
        corr = pd.DataFrame(X).corr(method="kendall").to_numpy()
        self.codependence_, self.distance_ = _corr_to_distance(
            corr, absolute=self.absolute, power=self.power
        )
        return self


class SpearmanDistance(BaseDistance):
    r"""Spearman Distance estimator.

    The codependence is computed from the Spearman correlation to which is applied a
    power and/or absolute transformation.
    This codependence is then used to compute the distance matrix.
    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`

    Parameters
    ----------
    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.
        The default is `False`.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.
        The default value is `1`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        Lòpez de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1):
        self.absolute = absolute
        self.power = power

    def fit(self, X: npt.ArrayLike, y=None) -> "SpearmanDistance":
        """Fit the Spearman Kendall estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SpearmanDistance
            Fitted estimator.
        """
        X = self._validate_data(X)
        corr = pd.DataFrame(X).corr(method="spearman").to_numpy()
        self.codependence_, self.distance_ = _corr_to_distance(
            corr, absolute=self.absolute, power=self.power
        )
        return self


class CovarianceDistance(BaseDistance):
    r"""Covariance Distance estimator.

    The codependence is computed from the correlation matrix of a chosen
    :ref:`covariance estimator <covariance_estimator>` to which is applied
    a power and/or absolute transformation.
    This codependence is then used to compute the distance matrix.
    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
       :ref:`Covariance estimator <covariance_estimator>`.
       The default (`None`) is to use :class:`~skfolio.moments.GerberCovariance`.

    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.
        The default is `False`.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.
        The default value is `1`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    covariance_estimator_: BaseCovariance
        Fitted `covariance_estimator`

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        Lòpez de Prado, Journal of Portfolio Management (2016)
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        absolute: bool = False,
        power: float = 1,
    ):
        self.covariance_estimator = covariance_estimator
        self.absolute = absolute
        self.power = power

    def fit(self, X: npt.ArrayLike, y=None) -> "CovarianceDistance":
        """Fit the Covariance Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : CovarianceDistance
            Fitted estimator.
        """
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=GerberCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X)

        # we validate and convert to numpy after all models have been fitted to keep the
        # features names information.
        _ = self._validate_data(X)

        corr, _ = cov_to_corr(self.covariance_estimator_.covariance_)
        self.codependence_, self.distance_ = _corr_to_distance(
            corr, absolute=self.absolute, power=self.power
        )
        return self


class DistanceCorrelation(BaseDistance):
    """Distance Correlation estimator.

    Distance Correlation was introduced by Szekely [1]_ to capture non-linear
    dependencies.

    Parameters
    ----------
    threshold : float, default=0.5
        Distance correlation threshold.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Measuring and testing independence by correlation of distances"
        Gábor J. Szekely , 2005
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _dcorr(x: np.ndarray, y: np.ndarray):
        """Calculate the distance correlation between two variables"""
        x = scd.squareform(scd.pdist(x.reshape(-1, 1)))
        y = scd.squareform(scd.pdist(y.reshape(-1, 1)))
        x = x - x.mean(axis=0)[np.newaxis, :] - x.mean(axis=1)[:, np.newaxis] + x.mean()
        y = y - y.mean(axis=0)[np.newaxis, :] - y.mean(axis=1)[:, np.newaxis] + y.mean()
        value = np.sqrt((x * y).sum()) / np.sqrt(
            np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
        )
        return value

    def fit(self, X: npt.ArrayLike, y=None) -> "DistanceCorrelation":
        """Fit the Distance Correlation estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : DistanceCorrelation
            Fitted estimator.
        """
        X = self._validate_data(X)
        n_assets = X.shape[1]
        corr = np.ones((n_assets, n_assets))
        # TODO: parallelize
        for i, j in zip(*np.triu_indices(n_assets, 1), strict=True):
            corr[i, j] = self._dcorr(x=X[:, i], y=X[:, j])
            corr[j, i] = corr[i, j]
        self.codependence_ = corr
        self.distance_ = np.sqrt(np.clip(1 - self.codependence_, a_min=0.0, a_max=1.0))
        return self


class MutualInformation(BaseDistance):
    r"""Mutual Information estimator.

    In information theory, the mutual information is a measure of the mutual dependence
    between variables.
    The related distance metric is called the variation of information.

    For two random variables X and Y, the mutual information I(X,Y) is defined as:

    .. math:: I(X,Y) = H(X) + H(Y) - H(X,Y)

    with H(X) and H(Y) the marginal entropies and H(X,Y) the joint entropy.

    The related distance metric known as the  variation of information is defined as:

    .. math:: d(X,Y) = H(X,Y) - I(X,Y) =  H(X) + H(Y) - 2 \times I(X,Y)

    and its normalization as:

    .. math:: D(X,Y) = \frac{d(X,Y)}{H(X,Y)} = \frac{H(X) + H(Y) - 2 \times I(X,Y)}{H(X) + H(Y) - I(X,Y)}

    Parameters
    ----------
    n_bins_method : NBinsMethod, default=NBinsMethod.FREEDMAN
        Method to compute the number of bins for the contingency matrix estimation used
        for the computation of the mutual information.
        Possible values are:

            * FREEDMAN (`default`)
            * KNUTH

    n_bins : int, optional
        Instead of using `n_bins_method`, you can directly specify the number of bins
        with `n_bins`.

    normalize : bool, default=True
        If this is set to True, the variation of information is normalized.
        The default is `True`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.
    """

    def __init__(
        self,
        n_bins_method: NBinsMethod = NBinsMethod.FREEDMAN,
        n_bins: int | None = None,
        normalize: bool = True,
    ):
        self.n_bins_method = n_bins_method
        self.n_bins = n_bins
        self.normalize = normalize

    def fit(self, X: npt.ArrayLike, y=None) -> "MutualInformation":
        """Fit the Mutual Information estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : MutualInformation
            Fitted estimator.
        """
        X = self._validate_data(X)
        n_assets = X.shape[1]
        if self.n_bins is None:
            match self.n_bins_method:
                case NBinsMethod.FREEDMAN:
                    n_bins_func = n_bins_freedman
                case NBinsMethod.KNUTH:
                    n_bins_func = n_bins_knuth
                case _:
                    raise ValueError(f"n_bins_method {self.n_bins_method} is not valid")
            n_bins_list = [n_bins_func(x=X[:, i]) for i in range(n_assets)]
        else:
            n_bins_list = [self.n_bins] * n_assets

        corr = np.full((n_assets, n_assets), np.nan)
        dist = corr.copy()
        for i, j in zip(*np.triu_indices(n_assets), strict=True):
            n_bins = max(n_bins_list[i], n_bins_list[j])
            x = X[:, i]
            y = X[:, j]
            contingency = np.histogram2d(x, y, bins=n_bins)[0]
            mutual_information = skm.mutual_info_score(
                None, None, contingency=contingency
            )
            entropy_x = sct.entropy(np.histogram(x, n_bins)[0])
            entropy_y = sct.entropy(np.histogram(y, n_bins)[0])
            if self.normalize:
                corr[i, j] = mutual_information / min(entropy_x, entropy_y)
                dist[i, j] = max(
                    0.0,
                    (entropy_x + entropy_y - 2 * mutual_information)
                    / (entropy_x + entropy_y - mutual_information),
                )
            else:
                corr[i, j] = mutual_information
                dist[i, j] = max(0.0, entropy_x + entropy_y - 2 * mutual_information)
            corr[j, i] = corr[i, j]
            dist[j, i] = dist[i, j]
        self.codependence_ = corr
        self.distance_ = dist
        return self


def _corr_to_distance(
    corr: np.ndarray, absolute: bool, power: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Transform a correlation matrix to a codependence and distance matrix.

    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`


    Parameters
    ----------
    corr : ndarray of shape (n_assets, n_assets)
        Correlation matrix.

    absolute : bool
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.

    power : float
        Exponent of the power transformation applied to the correlation matrix.

    Returns
    -------
    codependence, distance : tuple[np.ndarray, np.ndarray]
        Codependence and distance matrices.
    """
    bounds = np.array([-1, 0, 1])
    if absolute:
        corr = np.abs(corr)
        bounds = np.abs(bounds)
    corr = np.power(corr, power)
    bounds = np.power(bounds, power)
    scaler = 1 / (1 - min(bounds))
    distance = np.sqrt(np.clip(scaler * (1 - corr), a_min=0.0, a_max=1.0))
    return corr, distance
