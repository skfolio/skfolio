"""Implied Covariance Estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._empirical_covariance import EmpiricalCovariance
from skfolio.utils.tools import check_estimator, safe_indexing
from skfolio.utils.validation import check_implied_vol


class ImpliedCovariance(BaseCovariance):
    """Implied Covariance estimator.
    The covariance matrix is first estimated using a Covariance estimator (for example
    `EmpiricalCovariance`) then the diagonal elements are shrunken toward the expected
    variances computed from the implied volatilities.

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` to estimate the covariance
        matrix prior shrinking.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    annualized_factor: float, default=252.0



    nearest : bool, default=False
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition than can be
        computed. The variance is left unchanged. A covariance matrix is in theory PSD.
        However, due to floating-point inaccuracies, we can end up with a covariance
        matrix that is slightly non-PSD or where Cholesky decomposition is failing.
        This often occurs in high dimensional problems.
        For more details, see :func:`~skfolio.units.stats.cov_nearest`.
        The default is `False`.

    higham : bool, default=False
        If this is set to True, the Higham & Nick (2002) algorithm is used to find the
        nearest PSD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and use the clipping method as the
        Higham & Nick algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iteration of the Higham & Nick (2002) algorithm.
        The default value is `100`.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        annualized_factor: float = 252.0,
        alpha: float = 1.0,
        method: str = "last",
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.covariance_estimator = covariance_estimator
        self.annualized_factor = annualized_factor
        self.alpha = alpha
        self.method = method

    def fit(
        self, X: npt.ArrayLike, y=None, implied_vol: npt.ArrayLike = None, **fit_params
    ) -> "ImpliedCovariance":
        """Fit the implied covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        implied_vol : array-like of shape (n_observations, n_assets)
            Implied volatilities of the assets.

        Returns
        -------
        self : ImpliedCovariance
            Fitted estimator.
        """
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X)

        covariance = self.covariance_estimator_.covariance_

        print(implied_vol)

        assets_names = skv._get_feature_names(X)
        if assets_names is not None:
            vol_assets_names = skv._get_feature_names(implied_vol)
            if vol_assets_names is not None:
                missing_assets = assets_names[~np.in1d(assets_names, vol_assets_names)]
                if len(missing_assets) > 0:
                    raise ValueError(
                        f"The following assets are missing from "
                        f"`implied_vol`: {missing_assets}"
                    )
                indices = [
                    np.argwhere(x == vol_assets_names)[0][0] for x in assets_names
                ]
                # Select same columns as X (needed for Pipeline with preselection)
                # and re-order to follow X ordering.
                implied_vol = safe_indexing(implied_vol, indices=indices, axis=1)

        X = self._validate_data(X)

        implied_vol = check_implied_vol(implied_vol=implied_vol, X=X)

        expected_var = implied_vol**2 / self.annualized_factor  # TODO: paper
        shrunk_var = expected_var * self.alpha + np.diag(covariance) * (1 - self.alpha)
        np.fill_diagonal(covariance, shrunk_var)

        self._set_covariance(covariance)
        return self
