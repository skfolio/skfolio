"""Base Covariance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.utils.validation as skv

from skfolio.exceptions import NonPositiveVarianceError
from skfolio.utils.stats import (
    _squared_mahalanobis_dist_from_cholesky,
    cov_nearest,
    safe_cholesky,
    squared_mahalanobis_dist,
)


class BaseCovariance(skb.BaseEstimator, ABC):
    """Base class for all covariance estimators in `skfolio`.

    Parameters
    ----------
    assume_centered : bool, default=False
        If False (default), the data are mean-centered before computing the covariance.
        This is the standard behavior when working with raw returns where the mean is
        not guaranteed to be zero. If True, the estimator assumes the input data are
        already centered. Use this when you know the returns have zero mean, such as
        pre-demeaned data or regression residuals.

    nearest : bool, default=True
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition than can be
        computed. The variance is left unchanged.
        A covariance matrix that is not positive definite often occurs in high
        dimensional problems. It can be due to multicollinearity, floating-point
        inaccuracies, or when the number of observations is smaller than the number of
        assets. For more details, see :func:`~skfolio.utils.stats.cov_nearest`.
        The default is `True`.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm is used to find the
        nearest PD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and uses the clipping method as the
        Higham algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iterations of the Higham (2002) algorithm.
        The default value is `100`.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance matrix.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.
        Use for compatibility with scikit-learn Covariance estimators and for
        mahalanobis and score methods.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    covariance_: np.ndarray
    location_: np.ndarray
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        assume_centered: bool = False,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        self.assume_centered = assume_centered
        self.nearest = nearest
        self.higham = higham
        self.higham_max_iteration = higham_max_iteration

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None, **fit_params):
        pass

    def score(self, X_test: npt.ArrayLike, y=None) -> float:
        r"""Compute the mean log-likelihood of observations under the estimated model.

        Evaluates how well the fitted covariance matrix explains new observations,
        assuming a multivariate Gaussian distribution. This is useful for:

        * Model selection (comparing different covariance estimators)
        * Cross-validation of covariance estimation methods
        * Assessing goodness-of-fit

        The log-likelihood for a single observation :math:`r` is:

        .. math::
            \log p(r | \mu, \Sigma) = -\frac{1}{2} \left[
                n \log(2\pi) + \log|\Sigma| + (r - \mu)^T \Sigma^{-1} (r - \mu)
            \right]

        where :math:`n` is the number of assets, :math:`\Sigma` is the estimated
        covariance matrix (`self.covariance_`), and :math:`\mu` is the estimated
        mean (`self.location_` if available, otherwise zero).

        Parameters
        ----------
        X_test : array-like of shape (n_observations, n_assets)
            Observations for which to compute the log-likelihood.
            Typically held-out test data not used during fitting.
            Assets with non-finite fitted variance are excluded from inference. This
            typically happens when the fitted covariance cannot be estimated for an
            asset, for example before listing, after delisting, or during a warmup
            period. After this asset-level filtering, each row of `X_test` is scored
            using the remaining available values only. This covers row-level missing
            values in `X_test`, such as market holidays or pre/post-listing.

        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        score : float
            Mean log-likelihood of the observations. Higher values indicate better fit.
            The score is averaged over all observations.

        Examples
        --------
        >>> import numpy as np
        >>> from skfolio.moments import EmpiricalCovariance, LedoitWolf
        >>> X_train = np.random.randn(100, 5)
        >>> X_test = np.random.randn(50, 5)
        >>> emp = EmpiricalCovariance().fit(X_train)
        >>> lw = LedoitWolf().fit(X_train)
        >>> # Compare models on held-out data
        >>> print(f"Empirical: {emp.score(X_test):.2f}")
        >>> print(f"LedoitWolf: {lw.score(X_test):.2f}")
        """
        skv.check_is_fitted(self, "covariance_")
        X_test = skv.validate_data(
            self,
            X_test,
            reset=False,
            dtype=float,
            ensure_all_finite="allow-nan",
        )
        mask = np.isfinite(np.diag(self.covariance_))
        mean = self.location_ if hasattr(self, "location_") else None
        if mean is not None and not self.assume_centered:
            mask &= np.isfinite(mean)
        if not np.any(mask):
            raise ValueError("No finite fitted assets available for inference.")

        if not np.all(mask):
            X_test = X_test[:, mask]
            covariance = self.covariance_[np.ix_(mask, mask)]
            if mean is not None:
                mean = mean[mask]
        else:
            covariance = self.covariance_
        if np.isfinite(X_test).all():
            _, n_assets = X_test.shape
            chol = safe_cholesky(covariance=covariance)
            d2 = _squared_mahalanobis_dist_from_cholesky(
                X_test, cholesky=chol, mean=mean
            )
            score = 0.5 * (
                -2.0 * np.sum(np.log(np.diag(chol)))
                - np.mean(d2)
                - n_assets * np.log(2.0 * np.pi)
            )
            return float(score)

        row_scores = _score_observed_subspaces(X_test, covariance, mean)
        if np.all(np.isnan(row_scores)):
            raise ValueError("X_test has no row with any finite retained observation.")
        return float(np.nanmean(row_scores))

    def mahalanobis(self, X_test: npt.ArrayLike) -> np.ndarray:
        r"""Compute the squared Mahalanobis distance of observations.

        The squared Mahalanobis distance of an observation :math:`r` is defined as:

        .. math:: d^2 = (r - \mu)^T \Sigma^{-1} (r - \mu)

        where :math:`\Sigma` is the estimated covariance matrix (`self.covariance_`)
        and :math:`\mu` is the estimated mean (`self.location_` if available, otherwise
        zero).

        This distance measure accounts for correlations between assets and is useful
        for:

        * Outlier detection in portfolio returns
        * Risk-adjusted distance calculations
        * Identifying unusual market regimes

        Parameters
        ----------
        X_test : array-like of shape (n_observations, n_assets) or (n_assets,)
            Observations for which to compute the squared Mahalanobis distance.
            Each row represents one observation. If 1D, treated as a single
            observation. Assets with non-finite fitted variance are excluded from
            inference. Inside the retained inference subspace, the observations
            must be finite.

        Returns
        -------
        distances : ndarray of shape (n_observations,) or float
            Squared Mahalanobis distance for each observation. Returns a scalar
            if input is 1D.

        Examples
        --------
        >>> import numpy as np
        >>> from skfolio.moments import EmpiricalCovariance
        >>> X = np.random.randn(100, 3)
        >>> model = EmpiricalCovariance()
        >>> model.fit(X)
        >>> distances = model.mahalanobis(X)
        >>> # Distances follow approximately chi-squared distribution with n_assets DoF
        >>> print(f"Mean distance: {distances.mean():.2f}, Expected: {3:.2f}")
        """
        skv.check_is_fitted(self, "covariance_")

        is_1d = np.asarray(X_test).ndim == 1
        X_test = np.atleast_2d(X_test) if is_1d else X_test
        X_test = skv.validate_data(
            self,
            X_test,
            reset=False,
            dtype=float,
            ensure_all_finite="allow-nan",
        )
        mask = np.isfinite(np.diag(self.covariance_))
        mean = self.location_ if hasattr(self, "location_") else None
        if mean is not None and not self.assume_centered:
            mask &= np.isfinite(mean)
        if not np.any(mask):
            raise ValueError("No finite fitted assets available for inference.")

        if not np.all(mask):
            X_test = X_test[:, mask]
            covariance = self.covariance_[np.ix_(mask, mask)]
            if mean is not None:
                mean = mean[mask]
        else:
            covariance = self.covariance_
        if not np.isfinite(X_test).all():
            raise ValueError(
                "X_test contains non-finite values in the fitted inference subspace."
            )
        distances = squared_mahalanobis_dist(X_test, covariance, mean=mean)
        return float(distances[0]) if is_1d else distances

    def _sanity_check(self, covariance: np.ndarray) -> None:
        """Perform a sanity check on the covariance matrix by verifying that all
        finite diagonal elements are strictly positive.

        NaN diagonal entries (e.g., from assets not yet active or still in
        warm-up) are skipped. The goal is to early detect corrupted asset data
        (with zero variance) that would lead to optimization errors.
        """
        diag = np.diag(covariance)
        finite_mask = np.isfinite(diag)
        cond = finite_mask & (diag < 1e-15)
        if np.any(cond):
            corrupted_assets = list(np.argwhere(cond).flatten())
            detail = "assets indices"
            if hasattr(self, "feature_names_in_"):
                corrupted_assets = list(self.feature_names_in_[corrupted_assets])
                detail = "assets"
            raise NonPositiveVarianceError(
                f"The following {detail} have a non positive variance:"
                f" {corrupted_assets}"
            )

    def _set_covariance(self, covariance: np.ndarray) -> None:
        """Perform checks, convert to nearest PSD if specified and save the covariance.

        NaN-aware: if the covariance matrix contains NaN entries (e.g., from assets
        that are inactive or still in warm-up), the sanity check skips NaN diagonal
        entries and the nearest PD projection operates only on the active (non-NaN)
        submatrix. NaN entries are preserved in the output for inactive assets.

        If that submatrix contains NaN or infinity (e.g. missing pairwise estimates
        while marginal variances are finite), inactive rows and columns are extended
        by :func:`_reduce_to_finite_active_block` and a :class:`UserWarning` is
        emitted.

        Parameters
        ----------
        covariance : array-like of shape (n_assets, n_assets)
            Estimated covariance matrix to be stored. May contain NaN for inactive
            assets.

        Warns
        -----
        UserWarning
            When the active submatrix is incomplete and assets are peeled.
        """
        self._sanity_check(covariance)
        diag = np.diag(covariance)
        active = np.isfinite(diag)
        all_active = active.all()
        any_active = active.any()

        if all_active:
            active_block_is_finite = np.isfinite(covariance).all()
        elif any_active:
            active_idx = np.flatnonzero(active)
            active_block_is_finite = np.isfinite(
                covariance[np.ix_(active_idx, active_idx)]
            ).all()
        else:
            active_block_is_finite = True

        if not active_block_is_finite:
            warnings.warn(
                "Covariance has a non-finite entry between two assets with "
                "finite variances (e.g. no overlapping returns). Peeling "
                "assets until the active block is fully finite.",
                UserWarning,
                stacklevel=2,
            )
            _reduce_to_finite_active_block(covariance)
            active = np.isfinite(np.diag(covariance))
            all_active = active.all()
            any_active = active.any()

        if self.nearest:
            if all_active:
                covariance = cov_nearest(
                    covariance,
                    higham=self.higham,
                    higham_max_iteration=self.higham_max_iteration,
                    warn=True,
                )
            elif any_active:
                active_idx = np.flatnonzero(active)
                active_cov = covariance[np.ix_(active_idx, active_idx)]
                active_cov = cov_nearest(
                    active_cov,
                    higham=self.higham,
                    higham_max_iteration=self.higham_max_iteration,
                    warn=True,
                )
                covariance = covariance.copy()
                covariance[np.ix_(active_idx, active_idx)] = active_cov

        # set covariance
        self.covariance_ = covariance


def _score_observed_subspaces(
    X: np.ndarray,
    covariance: np.ndarray,
    mean: np.ndarray | None,
) -> np.ndarray:
    """Compute row-wise Gaussian scores on observed subspaces.

    Each row is scored on the marginal Gaussian distribution of its finite
    coordinates. Rows with no finite coordinate return NaN.
    """
    observed = np.isfinite(X)
    row_scores = np.full(X.shape[0], np.nan, dtype=float)
    log_2pi = np.log(2.0 * np.pi)
    packed = np.ascontiguousarray(np.packbits(observed, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    split_idx = np.flatnonzero(np.diff(sorted_inverse)) + 1
    row_groups = np.split(order, split_idx)

    for row_idx in row_groups:
        obs_idx = np.flatnonzero(observed[row_idx[0]])
        if obs_idx.size == 0:
            continue

        chol = safe_cholesky(covariance=covariance[np.ix_(obs_idx, obs_idx)])
        d2 = _squared_mahalanobis_dist_from_cholesky(
            X[np.ix_(row_idx, obs_idx)],
            cholesky=chol,
            mean=None if mean is None else mean[obs_idx],
        )
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        row_scores[row_idx] = 0.5 * (-logdet - d2 - obs_idx.size * log_2pi)

    return row_scores


def _reduce_to_finite_active_block(covariance: np.ndarray) -> None:
    r"""Drop assets until the finite-diagonal set has a fully finite submatrix.

    Pairwise updates can leave :math:`\Sigma_{ii}` and :math:`\Sigma_{jj}` finite while
    :math:`\Sigma_{ij}` stays NaN when no observation has both returns. Assets are
    removed greedily by largest count of non-finite entries to other currently active
    indices. On ties, the asset with the smallest global index is removed first. Each
    removed asset gets a full NaN row and column.

    Parameters
    ----------
    covariance : ndarray of shape (n_assets, n_assets)
        Square matrix, modified in place.
    """
    active = np.isfinite(np.diag(covariance)).copy()
    non_finite = ~np.isfinite(covariance)

    while True:
        idx_active = np.flatnonzero(active)
        if idx_active.size == 0:
            break
        sub = non_finite[np.ix_(idx_active, idx_active)]
        damage = sub.sum(axis=1)
        if damage.max() == 0:
            break
        victim = idx_active[int(np.argmax(damage))]
        active[victim] = False

    inactive = np.flatnonzero(~active)
    covariance[inactive, :] = np.nan
    covariance[:, inactive] = np.nan
