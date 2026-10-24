"""Distance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.spatial.distance as scd
import scipy.stats as sct
import sklearn.metrics as skmc
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.distance._base import BaseDistance
from skfolio.moments import BaseCovariance, GerberCovariance
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.stats import (
    NBinsMethod,
    assert_is_symmetric,
    cov_to_corr,
    is_positive_semidefinite,
    n_bins_freedman,
    n_bins_knuth,
    symmetrize,
)
from skfolio.utils.tools import (
    _call_estimator,
    _validate_positive_real,
    check_estimator,
)
from skfolio.utils.validation import _validate_pairwise_matrix

_FITTED_ATTR = "distance_"


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
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.

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
        López de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1) -> None:
        self.absolute = absolute
        self.power = power

    def fit(self, X: ArrayLike, y: None = None) -> PearsonDistance:
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
        X = skv.validate_data(self, X)
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
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.
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
        López de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1) -> None:
        self.absolute = absolute
        self.power = power

    def fit(self, X: ArrayLike, y: None = None) -> KendallDistance:
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
        X = skv.validate_data(self, X)
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
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.
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
        López de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1) -> None:
        self.absolute = absolute
        self.power = power

    def fit(self, X: ArrayLike, y: None = None) -> SpearmanDistance:
        """Fit the Spearman estimator.

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
        X = skv.validate_data(self, X)
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
    covariance_estimator : BaseCovariance or {"precomputed"}, optional
       :ref:`Covariance estimator <covariance_estimator>`.
       The default (`None`) is to use :class:`~skfolio.moments.GerberCovariance`.
       With `"precomputed"`, fit converts a covariance matrix directly.
       NaN diagonal entries exclude assets, whose output rows and columns stay
       NaN. The available block must be symmetric and positive semidefinite,
       with strictly positive variances. Validation allows float64 roundoff of
       `64 * eps * n_available_assets` in the implied correlation. No repair is
       applied.

    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.
        The default is `False`.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.
        The default value is `1`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    covariance_estimator_ : BaseCovariance or None
        Fitted covariance estimator, or None in precomputed mode.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    Notes
    -----
    Both learned and precomputed covariances must be symmetric and positive
    semidefinite within numerical tolerance. Singular covariances are accepted.
    Invalid covariances raise `ValueError`. For covariance estimators supporting
    `nearest`, keep `nearest=True` to repair their output before conversion.
    Precomputed covariance can be repaired explicitly with
    :func:`~skfolio.utils.stats.cov_nearest`.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        López de Prado, Journal of Portfolio Management (2016)
    """

    covariance_estimator_: BaseCovariance | None

    def __init__(
        self,
        covariance_estimator: BaseCovariance | Literal["precomputed"] | None = None,
        absolute: bool = False,
        power: float = 1,
    ) -> None:
        self.covariance_estimator = covariance_estimator
        self.absolute = absolute
        self.power = power

    def fit(
        self,
        X: ArrayLike,
        y: None = None,
        **fit_params: Any,
    ) -> CovarianceDistance:
        """Fit the Covariance Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets) or (n_assets, n_assets)
            Returns in learned mode, or a covariance matrix in precomputed mode.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Metadata routed to the covariance estimator.

        Returns
        -------
        self : CovarianceDistance
            Fitted estimator.
        """
        self._reset()
        return self._fit(X, y, method="fit", **fit_params)

    def partial_fit(
        self,
        X: ArrayLike,
        y: None = None,
        **fit_params: Any,
    ) -> CovarianceDistance:
        """Update covariance from new returns and recompute the distance.

        The covariance estimator must implement `partial_fit`. The first call
        also uses the child's `partial_fit`. With `covariance_estimator="precomputed"`,
        call `fit` with the updated covariance matrix instead.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            New asset returns with the same full schema as previous calls.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Metadata routed to the covariance estimator's partial_fit.

        Returns
        -------
        self : CovarianceDistance
            Updated estimator.

        Raises
        ------
        TypeError
            If the covariance estimator does not implement `partial_fit` or
            `covariance_estimator="precomputed"`.
        """
        if self.requires_covariance_input:
            raise TypeError(
                'partial_fit is not supported with covariance_estimator="precomputed". '
                "Call fit with the updated covariance matrix instead."
            )
        return self._fit(X, y, method="partial_fit", **fit_params)

    @property
    def requires_covariance_input(self) -> bool:
        """Whether `covariance_estimator` is set to `"precomputed"`."""
        return (
            isinstance(self.covariance_estimator, str)
            and self.covariance_estimator == "precomputed"
        )

    def get_metadata_routing(self) -> skm.MetadataRouter:
        """Get metadata routing for this estimator.

        Routes metadata to the corresponding `fit` or `partial_fit` method of
        `covariance_estimator`.

        Returns
        -------
        routing : MetadataRouter
            Metadata routing configuration.
        """
        router = skm.MetadataRouter(owner=self.__class__.__name__)
        if not isinstance(self.covariance_estimator, str):
            router.add(
                covariance_estimator=self.covariance_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        return router

    def _fit(
        self,
        X: ArrayLike,
        y: None,
        method: str,
        **fit_params: Any,
    ) -> CovarianceDistance:
        """Share initialization, metadata routing and covariance conversion."""
        routed = skm.process_routing(self, method, **fit_params)
        first_call = not hasattr(self, _FITTED_ATTR)
        if self.requires_covariance_input:
            covariance = X
            skv.validate_data(self, X, reset=True, skip_check_array=True)
            self.covariance_estimator_ = None
        else:
            if first_call and isinstance(self.covariance_estimator, str):
                raise ValueError(
                    "covariance_estimator must be 'precomputed' or a covariance estimator."
                )
            # The covariance estimator validates values and missing observations.
            skv.validate_data(self, X, reset=first_call, skip_check_array=True)
            if first_call:
                self.covariance_estimator_ = check_estimator(
                    self.covariance_estimator,
                    default=GerberCovariance(),
                    check_type=BaseCovariance,
                )
            _call_estimator(
                self.covariance_estimator_,
                method,
                X,
                y,
                routed_params=routed.covariance_estimator,
            )
            covariance = self.covariance_estimator_.covariance_  # ty: ignore[unresolved-attribute]
        self.codependence_, self.distance_ = _cov_to_distance(
            covariance, absolute=self.absolute, power=self.power
        )
        return self

    def _reset(self) -> None:
        """Reset fitted state for a new learning run."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def __sklearn_tags__(self) -> sku.Tags:
        """Declare missing-data support for the configured input."""
        tags = super().__sklearn_tags__()
        if self.requires_covariance_input:
            tags.input_tags.allow_nan = True
        elif self.covariance_estimator is not None and not isinstance(
            self.covariance_estimator, str
        ):
            tags.input_tags.allow_nan = sku.get_tags(
                self.covariance_estimator
            ).input_tags.allow_nan
        return tags


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

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @staticmethod
    def _dcorr(x: FloatArray, y: FloatArray) -> float:
        """Calculate the distance correlation between two variables."""
        x = scd.squareform(scd.pdist(x.reshape(-1, 1)))
        y = scd.squareform(scd.pdist(y.reshape(-1, 1)))
        x = x - x.mean(axis=0)[np.newaxis, :] - x.mean(axis=1)[:, np.newaxis] + x.mean()
        y = y - y.mean(axis=0)[np.newaxis, :] - y.mean(axis=1)[:, np.newaxis] + y.mean()
        value = np.sqrt((x * y).sum()) / np.sqrt(
            np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
        )
        return value

    def fit(self, X: ArrayLike, y: None = None) -> DistanceCorrelation:
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
        X = skv.validate_data(self, X)
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
    ) -> None:
        self.n_bins_method = n_bins_method
        self.n_bins = n_bins
        self.normalize = normalize

    def fit(self, X: ArrayLike, y: None = None) -> MutualInformation:
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
        X = skv.validate_data(self, X)
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
            x_i = X[:, i]
            x_j = X[:, j]
            contingency = np.histogram2d(x_i, x_j, bins=n_bins)[0]
            mutual_information = skmc.mutual_info_score(
                None, None, contingency=contingency
            )
            entropy_x = sct.entropy(np.histogram(x_i, n_bins)[0])
            entropy_y = sct.entropy(np.histogram(x_j, n_bins)[0])
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


def _cov_to_distance(
    cov: ArrayLike, absolute: bool, power: float
) -> tuple[FloatArray, FloatArray]:
    """Convert a covariance matrix to codependence and distance matrices.

    NaN diagonal entries exclude assets. Only the available block is converted,
    and excluded rows and columns remain NaN in both outputs. The input is not
    modified.

    Parameters
    ----------
    cov : array-like of shape (n_assets, n_assets)
        Covariance matrix. The available block must be finite, symmetric and
        positive semidefinite, with strictly positive variances. DataFrame row
        and column labels must match in the same order.

    absolute : bool
        If True, apply the absolute transformation to the correlation matrix.

    power : float
        Exponent applied to the correlation matrix after the optional absolute
        transformation. Must be finite and strictly positive, with an integer
        value when `absolute=False`.

    Returns
    -------
    codependence : ndarray of shape (n_assets, n_assets)
        Transformed correlation matrix, with NaN rows and columns for excluded
        assets.

    distance : ndarray of shape (n_assets, n_assets)
        Distance matrix, with zero diagonal entries for available assets and
        NaN rows and columns for excluded assets. If no assets are available,
        both outputs contain only NaN.

    Raises
    ------
    ValueError
        If `cov` is not square, contains infinite values, or has mismatched
        asset labels. Also raised if the available block contains
        missing values, has nonpositive variances, or fails the correlation
        checks described below, or if `power` is invalid.

    See Also
    --------
    _corr_to_distance : Convert correlations to codependence and distances.

    Notes
    -----
    Symmetry and positive semidefiniteness are checked on the implied
    correlation matrix with absolute tolerance
    `64 * eps * n_available_assets`, where `eps` is float64 machine precision.
    Accepted roundoff is handled by symmetrizing correlations and clipping them
    to [-1, 1]. Singular covariances are accepted without eigenvalue correction.
    """
    cov, available = _validate_pairwise_matrix(cov)
    codependence = np.full_like(cov, np.nan)
    distance = np.full_like(cov, np.nan)
    ix = np.ix_(available, available)
    available_cov = cov[ix]
    if np.any(np.diag(available_cov) <= 0):
        raise ValueError("Available covariance variances must be strictly positive.")
    corr, _ = cov_to_corr(available_cov)
    tolerance = 64 * np.finfo(np.float64).eps * len(available_cov)
    assert_is_symmetric(corr, rtol=0, atol=tolerance)
    symmetrize(corr)
    if not is_positive_semidefinite(corr, atol=tolerance):
        raise ValueError("The covariance must be positive semidefinite.")
    np.clip(corr, -1, 1, out=corr)
    codependence[ix], distance[ix] = _corr_to_distance(
        corr, absolute=absolute, power=power
    )
    return codependence, distance


def _corr_to_distance(
    corr: FloatArray, absolute: bool, power: float
) -> tuple[FloatArray, FloatArray]:
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
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.

    Returns
    -------
    codependence, distance : tuple[FloatArray, FloatArray]
        Codependence and distance matrices.

    Raises
    ------
    ValueError
        If `power` is not finite and strictly positive, or has a fractional
        value when `absolute=False`.
    """
    _validate_positive_real(power, "power")
    if not absolute and power % 1 != 0:
        raise ValueError("power must have an integer value when absolute=False.")
    bounds = np.array([-1, 0, 1])
    if absolute:
        corr = np.abs(corr)
        bounds = np.abs(bounds)
    corr = np.power(corr, power)
    bounds = np.power(bounds, power)
    scaler = 1 / (1 - min(bounds))
    distance = np.sqrt(np.clip(scaler * (1 - corr), a_min=0.0, a_max=1.0))
    return corr, distance
