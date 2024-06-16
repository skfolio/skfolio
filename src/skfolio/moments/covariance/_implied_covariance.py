"""Implied Covariance Estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn as sk
import sklearn.base as skb
import sklearn.linear_model as skl
import sklearn.metrics as sks
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._empirical_covariance import EmpiricalCovariance
from skfolio.utils.tools import check_estimator, get_feature_names, safe_indexing


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

    volatility_risk_premium_adj : float | dict[str, float] | array-like of shape (n_assets, ), optional
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset fee) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.
        The default value is `0.0`.

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
        Names of assets seen during `fit`. Defined only when `returns`
        has assets names that are all strings.
    """

    covariance_estimator_: BaseCovariance
    pred_realised_vols_: np.ndarray
    linear_regressors_: list
    coefs_: np.ndarray
    intercepts_: np.ndarray
    r2_scores_: np.ndarray

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        annualized_factor: float = 252.0,
        window: int = 21,
        linear_regressor: skb.BaseEstimator | None = None,
        volatility_risk_premium_adj: skt.MultiInput | None = None,
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
        self.linear_regressor = linear_regressor
        self.window = window
        self.volatility_risk_premium_adj = volatility_risk_premium_adj

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                covariance_estimator=self.covariance_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

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

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : ImpliedCovariance
            Fitted estimator.
        """
        if implied_vol is not None:
            # noinspection PyTypeChecker
            fit_params["implied_vol"] = implied_vol

        routed_params = skm.process_routing(self, "fit", **fit_params)

        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        # noinspection PyArgumentList
        self.covariance_estimator_.fit(X, y, **routed_params.covariance_estimator.fit)

        covariance = self.covariance_estimator_.covariance_

        assets_names = get_feature_names(X)
        if assets_names is not None:
            vol_assets_names = get_feature_names(implied_vol)
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
                # Select same columns as returns (needed for Pipeline with preselection)
                # and re-order to follow returns ordering.
                implied_vol = safe_indexing(implied_vol, indices=indices, axis=1)

        X = self._validate_data(X)
        implied_vol = check_implied_vol(implied_vol=implied_vol, X=X)
        implied_vol /= np.sqrt(self.annualized_factor)

        if self.volatility_risk_premium_adj is not None:
            if self.volatility_risk_premium_adj <= 0:
                raise ValueError(
                    "volatility_risk_premium_adj must be strictly positive, "
                    f"received {self.volatility_risk_premium_adj}"
                )
            self.pred_realised_vols_ = (
                implied_vol[-1] / self.volatility_risk_premium_adj
            )
        else:
            if self.window is None or self.window < 3:
                raise ValueError(
                    f"window must be strictly greater than 2, "
                    f"received {self.window}"
                )
            _linear_regressor = check_estimator(
                self.linear_regressor,
                default=skl.LinearRegression(fit_intercept=True),
                check_type=skb.BaseEstimator,
            )
            # OLS of ln(RV(t) = a + b1 ln(IV(t-1)) + b2 ln(RV(t-1)) + epsilon
            self._predict_realised_vols(
                linear_regressor=_linear_regressor, returns=X, implied_vol=implied_vol
            )

        np.fill_diagonal(covariance, self.pred_realised_vols_**2)

        self._set_covariance(covariance)
        return self

    def _predict_realised_vols(
        self,
        linear_regressor: skb.BaseEstimator,
        returns: np.ndarray,
        implied_vol: np.ndarray,
    ) -> None:
        n_observations, n_assets = returns.shape

        n_folds = n_observations // self.window
        if n_folds < 3:
            raise ValueError(
                f"Not enough observations to compute the volatility regression "
                f"coefficients. The window size of {self.window} on {n_observations} "
                f"observations produces {n_folds} non-overlapping folds. "
                f"The minimum number of fold is 3. You can either increase the number "
                f"of observation in your training set or decrease the window size."
            )

        realised_vol = _compute_realised_vol(
            returns=returns, window=self.window, ddof=1
        )

        implied_vol = _compute_implied_vol(implied_vol=implied_vol, window=self.window)

        if realised_vol.shape != implied_vol.shape:
            raise ValueError("`realised_vol`and `implied_vol` must have same shape")

        assert realised_vol.shape[0] == n_folds

        rv = np.log(realised_vol)
        iv = np.log(implied_vol)

        self.linear_regressors_ = []
        self.pred_realised_vols_ = np.zeros(n_assets)
        self.coefs_ = np.zeros((n_assets, 2))
        self.intercepts_ = np.zeros(n_assets)
        self.r2_scores_ = np.zeros(n_assets)
        for i in range(n_assets):
            model = sk.clone(linear_regressor)
            X = np.hstack((iv[:, [i]], rv[:, [i]]))
            X_train = X[:-1]
            X_pred = X[[-1]]
            y_train = rv[1:, i]

            model.fit(X=X_train, y=y_train)
            self.coefs_[i, :] = model.coef_
            self.intercepts_[i] = model.intercept_
            self.r2_scores_[i] = sks.r2_score(y_train, model.predict(X_train))
            rv_pred = model.predict(X_pred)
            self.pred_realised_vols_[i] = np.exp(rv_pred[0])
            self.linear_regressors_.append(model)


def _compute_realised_vol(
    returns: np.ndarray, window: int, ddof: int = 1
) -> np.ndarray:
    n_observations, n_assets = returns.shape
    chunks = n_observations // window

    return np.std(
        np.reshape(
            returns[n_observations - chunks * window :, :], (chunks, window, n_assets)
        ),
        ddof=ddof,
        axis=1,
    )


def _compute_implied_vol(implied_vol: np.ndarray, window: int) -> np.ndarray:
    n_observations, _ = implied_vol.shape
    chunks = n_observations // window
    return implied_vol[
        np.arange(n_observations - (chunks - 1) * window - 1, n_observations, window)
    ]


def check_implied_vol(implied_vol: npt.ArrayLike, X: npt.ArrayLike):
    """Validate implied volatilities.


    Parameters
    ----------
    implied_vol : {ndarray, Number or None}, shape (n_samples,)
        Input sample weights.

    X : {ndarray, list, sparse matrix}
        Input data.


    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    # noinspection PyUnresolvedReferences
    n_observations, n_assets = X.shape

    if implied_vol is None:
        raise ValueError("`implied_vol` cannot be None")
    else:
        implied_vol = skv.check_array(
            implied_vol,
            accept_sparse=False,
            ensure_2d=False,
            dtype=[np.float64, np.float32],
            order="C",
            copy=False,
            input_name="implied_vol",
        )
        if implied_vol.ndim != 2:
            raise ValueError(
                "Sample weights must be 2D array of shape (n_observation, n_assets)"
            )

        if implied_vol.shape != (n_observations, n_assets):
            raise ValueError(
                f"implied_vol.shape == {(implied_vol.shape)}, "
                f"expected {(n_observations, n_assets)}"
            )

    skv.check_non_negative((n_observations, n_assets), "`implied_vol`")

    return implied_vol
