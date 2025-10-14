"""Implied Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
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
from skfolio.utils.stats import corr_to_cov, cov_to_corr
from skfolio.utils.tools import (
    check_estimator,
    get_feature_names,
    input_to_array,
    safe_indexing,
)


class ImpliedCovariance(BaseCovariance):
    r"""Implied Covariance estimator.

    For each asset, the implied volatility time series is used to estimate the realised
    volatility using the non-overlapping log-transformed OLS model [6]_:

    .. math:: \ln(RV_{t}) = \alpha + \beta_{1} \ln(IV_{t-1}) + \beta_{2} \ln(RV_{t-1}) + \epsilon

    with :math:`\alpha`, :math:`\beta_{1}` and :math:`\beta_{2}` the intercept and
    coefficients to estimate, :math:`RV` the realised volatility, and :math:`IV` the
    implied volatility. The training set uses non-overlapping data of sample size
    `window_size` to avoid possible regression errors caused by auto-correlation.
    The logarithmic transformation of volatilities is used for its better finite sample
    properties and distribution, which is closer to normality, less skewed and
    leptokurtic [6]_.

    Alternatively, if `volatility_risk_premium_adj` is provided, the realised
    volatility is estimated using:

    .. math:: RV_{t} = \frac{IV_{t-1}}{VRPA}

    with :math:`VRPA` the volatility risk premium adjustment.

    The final step is the reconstruction of the covariance matrix from the correlation
    and estimated realised volatilities :math:`D`:

    .. math:: \Sigma = D \ Corr \ D

    With :math:`Corr`, the correlation matrix computed from the prior covariance
    estimator. The default is the `EmpiricalCovariance`. It can be changed to any
    covariance estimator using `prior_covariance_estimator`.

    Parameters
    ----------
    prior_covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` to estimate the covariance
        matrix used for the correlation estimates prior the volatilities update.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    annualized_factor : float, default=252
        Annualized factor (AF) used to covert the implied volatilities into the same
        frequency as the returns using :math:`\frac{IV}{\sqrt{AF}}`.
        The default is 252 which corresponds to **daily** returns and implied volatility
        expressed in **p.a.**

    window_size : int, default=20
        Window size used to construct the non-overlapping training set of realised
        volatilities and implied volatilities used in the regression.
        The default is 20 observations.

    linear_regressor : BaseEstimator, optional
        Estimator of the linear regression used to estimate the realised volatilities
        from the implied volatilities. The default is to use the scikit-learn OLS
        estimator `LinearRegression`.

    volatility_risk_premium_adj : float | dict[str, float] | array-like of shape (n_assets, ), optional
        If provided, instead of using the regression model, the realised volatilities
        are estimated using:

        .. math:: RV_{t} = \frac{IV_{t-1}}{VRPA}

        with :math:`VRPA` the volatility risk premium adjustment.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset :math:`VRPA`) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.

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

    prior_covariance_estimator_ : BaseEstimator
        Fitted prior covariance estimator.

    pred_realised_vols_ : ndarray of shape (n_assets,)
        The predicted realised volatilities

    linear_regressors_ : list[BaseEstimator]
        The fitted linear regressions.

    coefs_ : ndarray of shape (n_assets, 2)
        The coefficients of the log transformed regression model for each asset.

    intercepts_ : ndarray of shape (n_assets,)
        The intercepts of the log transformed regression model for each asset.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `returns`
        has assets names that are all strings.

    References
    ----------
    .. [1] "New evidence on the implied-realized volatility relation".
        Christensen & Hansen (2002).

    .. [2] "The relation between implied and realized volatility".
        Christensen & Prabhala (2002).

    .. [3] "Can implied volatility predict returns on the carry trade?".
        Egbers & Swinkels (2015).

    .. [4] "Volatility and correlation forecasting".
        Egbers & Swinkels (2015).

    .. [5] "Volatility and correlation forecasting".
        Andersen, Bollerslev, Christoffersen & Diebol (2006).

    .. [6] "How Well Does Implied Volatility Predict Future Stock Index Returns and
        Volatility? : A Study of Option-Implied Volatility Derived from OMXS30 Index
        Options".
        Sara Vikberg & Julia Björkman (2020).
    """

    prior_covariance_estimator_: BaseCovariance
    pred_realised_vols_: np.ndarray
    linear_regressors_: list
    coefs_: np.ndarray
    intercepts_: np.ndarray
    r2_scores_: np.ndarray

    def __init__(
        self,
        prior_covariance_estimator: BaseCovariance | None = None,
        annualized_factor: float = 252.0,
        window_size: int = 20,
        linear_regressor: skb.BaseEstimator | None = None,
        volatility_risk_premium_adj: skt.MultiInput | None = None,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.prior_covariance_estimator = prior_covariance_estimator
        self.annualized_factor = annualized_factor
        self.linear_regressor = linear_regressor
        self.window_size = window_size
        self.volatility_risk_premium_adj = volatility_risk_premium_adj

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                prior_covariance_estimator=self.prior_covariance_estimator,
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

        window_size = int(self.window_size)
        # fitting estimators
        self.prior_covariance_estimator_ = check_estimator(
            self.prior_covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        # noinspection PyArgumentList
        self.prior_covariance_estimator_.fit(
            X, y, **routed_params.prior_covariance_estimator.fit
        )

        corr, _ = cov_to_corr(self.prior_covariance_estimator_.covariance_)

        assets_names = get_feature_names(X)
        if assets_names is not None:
            vol_assets_names = get_feature_names(implied_vol)
            if vol_assets_names is not None:
                missing_assets = assets_names[~np.isin(assets_names, vol_assets_names)]
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

        X = skv.validate_data(self, X)
        _, n_assets = X.shape
        implied_vol = check_implied_vol(implied_vol=implied_vol, X=X)
        implied_vol /= np.sqrt(self.annualized_factor)

        if self.volatility_risk_premium_adj is not None:
            if np.isscalar(self.volatility_risk_premium_adj):
                volatility_risk_premium_adj = self.volatility_risk_premium_adj
            else:
                volatility_risk_premium_adj = input_to_array(
                    items=self.volatility_risk_premium_adj,
                    n_assets=n_assets,
                    fill_value=np.nan,
                    dim=1,
                    assets_names=getattr(self, "feature_names_in_", None),
                    name="volatility_risk_premium_adj",
                )

            if np.any(np.isnan(volatility_risk_premium_adj)):
                raise ValueError(
                    "volatility_risk_premium_adj must contain a value for each assets, "
                    f"received {self.volatility_risk_premium_adj}"
                )
            if np.any(volatility_risk_premium_adj <= 0):
                raise ValueError(
                    "volatility_risk_premium_adj must be strictly positive, "
                    f"received {self.volatility_risk_premium_adj}"
                )

            self.pred_realised_vols_ = implied_vol[-1] / volatility_risk_premium_adj
        else:
            if window_size is None or window_size < 3:
                raise ValueError(
                    f"window must be strictly greater than 2, "
                    f"received {self.window_size}"
                )
            _linear_regressor = check_estimator(
                self.linear_regressor,
                default=skl.LinearRegression(fit_intercept=True),
                check_type=skb.BaseEstimator,
            )
            # OLS of ln(RV(t) = a + b1 ln(IV(t-1)) + b2 ln(RV(t-1)) + epsilon
            self._predict_realised_vols(
                linear_regressor=_linear_regressor,
                returns=X,
                implied_vol=implied_vol,
                window_size=window_size,
            )

        covariance = corr_to_cov(corr, self.pred_realised_vols_)

        self._set_covariance(covariance)
        return self

    def _predict_realised_vols(
        self,
        linear_regressor: skb.BaseEstimator,
        returns: np.ndarray,
        implied_vol: np.ndarray,
        window_size: int,
    ) -> None:
        n_observations, n_assets = returns.shape

        n_folds = n_observations // window_size
        if n_folds < 3:
            raise ValueError(
                f"Not enough observations to compute the volatility regression "
                f"coefficients. The window size of {window_size} on {n_observations} "
                f"observations produces {n_folds} non-overlapping folds. "
                f"The minimum number of fold is 3. You can either increase the number "
                f"of observation in your training set or decrease the window size."
            )

        realised_vol = _compute_realised_vol(
            returns=returns, window_size=window_size, ddof=1
        )

        implied_vol = _compute_implied_vol(
            implied_vol=implied_vol, window_size=window_size
        )

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
    returns: np.ndarray, window_size: int, ddof: int = 1
) -> np.ndarray:
    """Create the realised volatilities samples for the regression model."""
    n_observations, n_assets = returns.shape
    chunks = n_observations // window_size

    return np.std(
        np.reshape(
            returns[n_observations - chunks * window_size :, :],
            (chunks, window_size, n_assets),
        ),
        ddof=ddof,
        axis=1,
    )


def _compute_implied_vol(implied_vol: np.ndarray, window_size: int) -> np.ndarray:
    """Create the implied volatilities samples for the regression model."""
    n_observations, _ = implied_vol.shape
    chunks = n_observations // window_size
    return implied_vol[
        np.arange(
            n_observations - (chunks - 1) * window_size - 1, n_observations, window_size
        )
    ]


def check_implied_vol(implied_vol: npt.ArrayLike, X: npt.ArrayLike) -> np.ndarray:
    """Validate implied volatilities.

    Parameters
    ----------
    implied_vol : array-like of shape (n_observations, n_assets)
        Implied volatilities of the assets.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.

    Returns
    -------
    implied_vol : ndarray of shape (n_observations, n_assets)
        Validated implied volatilities.
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
                f"implied_vol.shape == {(implied_vol.shape,)}, "
                f"expected {(n_observations, n_assets)}"
            )

    skv.check_non_negative((n_observations, n_assets), "`implied_vol`")
    # noinspection PyTypeChecker
    return implied_vol
