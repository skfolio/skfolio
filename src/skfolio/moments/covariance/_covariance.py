"""Covariance Estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as sco
import sklearn.covariance as skc
import sklearn.neighbors as skn

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.utils.stats import corr_to_cov, cov_to_corr
from skfolio.utils.tools import check_estimator


class EmpiricalCovariance(BaseCovariance):
    """Empirical covariance estimator.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    ddof : int, default=1
        Normalization is by `(n_observations - ddof)`.
        Note that `ddof=1` will return the unbiased estimate, and `ddof=0`
        will return the simple average. The default value is `1`.

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

    def __init__(
        self,
        window_size: int | None = None,
        ddof: int = 1,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.ddof = ddof

    def fit(self, X: npt.ArrayLike, y=None) -> "EmpiricalCovariance":
        """Fit the empirical covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EmpiricalCovariance
            Fitted estimator.
        """
        X = self._validate_data(X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        covariance = np.cov(X.T, ddof=self.ddof)
        self._set_covariance(covariance)
        return self


class GerberCovariance(BaseCovariance):
    """Gerber covariance estimator.

    Robust co-movement measure which ignores fluctuations below a certain threshold
    while simultaneously limiting the effects of extreme movements.
    The Gerber statistic extends Kendall's Tau by counting the proportion of
    simultaneous co-movements in series when their amplitudes exceed data-dependent
    thresholds.

    Three variant has been published:

        * Gerber et al. (2015): tend to produce matrices that are non-PSD.
        * Gerber et al. (2019): alteration of the denominator of the above statistic.
        * Gerber et al. (2022): final alteration to ensure PSD matrix.

    The last two variants are implemented.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    threshold : float, default=0.5
        Gerber threshold. The default value is `0.5`.

    psd_variant : bool, default=True
        If this is set to True, the Gerber et al. (2022) variant is used to ensure a
        positive semi-definite matrix.
        Otherwise, the Gerber et al. (2019) variant is used.
        The default is `True`.

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
        Estimated covariance.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "The gerber statistic: A robust co-movement measure for portfolio
        optimization".
        The Journal of Portfolio Management.
        Gerber, S., B. Javid, H. Markowitz, P. Sargen, and D. Starer (2022).

    .. [2]  "The gerber statistic: A robust measure of correlation".
        Gerber, S., B. Javid, H. Markowitz, P. Sargen, and D. Starer (2019).

    .. [3]  "Enhancing multi-asset portfolio construction under modern portfolio theory
        with a robust co-movement measure".
        Social Science Research network Working Paper Series.
        Gerber, S., H. Markowitz, and P. Pujara (2015).

    .. [4]  "Deconstructing the Gerber Statistic".
        Flint & Polakow, 2023.
    """

    def __init__(
        self,
        window_size: int | None = None,
        threshold: float = 0.5,
        psd_variant: bool = True,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.threshold = threshold
        self.psd_variant = psd_variant

    def fit(self, X: npt.ArrayLike, y=None) -> "GerberCovariance":
        """Fit the Gerber covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
          Price returns of the assets.

        y : Ignored
           Not used, present for API consistency by convention.

        Returns
        -------
        self : GerberCovariance
           Fitted estimator.
        """
        X = self._validate_data(X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        if not (1 > self.threshold > 0):
            raise ValueError("The threshold must be between 0 and 1")
        n_observations = X.shape[0]
        std = X.std(axis=0).reshape((-1, 1))
        u = X >= std.T * self.threshold
        d = X <= -std.T * self.threshold
        n = np.invert(u) & np.invert(d)  # np.invert preferred that ~ for type hint
        n = n.astype(int)
        u = u.astype(int)
        d = d.astype(int)
        concordant = u.T @ u + d.T @ d
        discordant = u.T @ d + d.T @ u
        h = concordant - discordant
        if self.psd_variant:
            corr = h / (n_observations - n.T @ n)
        else:
            h_sqrt = np.sqrt(np.diag(h)).reshape((-1, 1))
            corr = h / (h_sqrt @ h_sqrt.T)
        covariance = corr_to_cov(corr, std.reshape(-1))
        self._set_covariance(covariance)
        return self


class DenoiseCovariance(BaseCovariance):
    """Covariance Denoising estimator.

    The goal of Covariance Denoising is to reduce the noise and enhance the signal of
    the empirical covariance matrix [1]_.
    It reduces the ill-conditioning of the traditional covariance estimate by
    differentiating the eigenvalues associated with noise from the eigenvalues
    associated with signal.
    Denoising replaces the eigenvalues of the eigenvectors classified as random by
    Marčenko-Pastur with a constant eigenvalue.

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` to estimate the covariance
        matrix that will be denoised.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

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
        Estimated covariance.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "Machine Learning for Asset Managers".
        Elements in Quantitative Finance.
        Lòpez de Prado (2020).
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
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

    def fit(self, X: npt.ArrayLike, y=None) -> "DenoiseCovariance":
        """Fit the Covariance Denoising estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
          Price returns of the assets.

        y : Ignored
           Not used, present for API consistency by convention.

        Returns
        -------
        self : DenoiseCovariance
           Fitted estimator.
        """
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = self._validate_data(X)
        n_observations, n_assets = X.shape
        q = n_observations / n_assets
        corr, std = cov_to_corr(self.covariance_estimator_.covariance_)
        e_val, e_vec = np.linalg.eigh(corr)
        indices = e_val.argsort()[::-1]
        e_val, e_vec = e_val[indices], e_vec[:, indices]

        def _marchenko(x_var):
            e_min, e_max = (
                x_var * (1 - (1.0 / q) ** 0.5) ** 2,
                x_var * (1 + (1.0 / q) ** 0.5) ** 2,
            )
            e_val_lin = np.linspace(e_min, e_max, 1000)
            pdf_0 = (
                q
                / (2 * np.pi * x_var * e_val_lin)
                * ((e_max - e_val_lin) * (e_val_lin - e_min)) ** 0.5
            )
            kde = skn.KernelDensity(kernel="gaussian", bandwidth=0.01).fit(
                e_val.reshape(-1, 1)
            )
            # noinspection PyUnresolvedReferences
            pdf_1 = np.exp(kde.score_samples(pdf_0.reshape(-1, 1)))
            return np.sum((pdf_1 - pdf_0) ** 2)

        # noinspection PyTypeChecker
        res = sco.minimize(_marchenko, x0=0.5, bounds=((1e-5, 1 - 1e-5),))

        var = res["x"][0]
        n_facts = e_val.shape[0] - e_val[::-1].searchsorted(
            var * (1 + (1.0 / q) ** 0.5) ** 2
        )
        e_val_ = e_val.copy()
        e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
        corr = e_vec @ np.diag(e_val_) @ e_vec.T
        corr, _ = cov_to_corr(corr)
        covariance = corr_to_cov(corr, std)
        self._set_covariance(covariance)
        return self


class DetoneCovariance(BaseCovariance):
    """Covariance Detoning estimator.

    Financial covariance matrices usually incorporate a market component corresponding
    to the first eigenvectors [1]_.
    For some applications like clustering, removing the market component (loud tone)
    allow a greater portion of the covariance to be explained by components that affect
    specific subsets of the securities.

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` to estimate the covariance
        matrix prior detoning.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    n_markets : int, default=1
        Number of eigenvectors related to the market.
        The default value is `1`.

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
        Estimated covariance.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "Machine Learning for Asset Managers".
        Elements in Quantitative Finance.
        Lòpez de Prado (2020).
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        n_markets: float = 1,
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
        self.n_markets = n_markets

    def fit(self, X: npt.ArrayLike, y=None) -> "DetoneCovariance":
        """Fit the Covariance Detoning estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
          Price returns of the assets.

        y : Ignored
           Not used, present for API consistency by convention.

        Returns
        -------
        self : DetoneCovariance
           Fitted estimator.
        """
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        _ = self._validate_data(X)
        corr, std = cov_to_corr(self.covariance_estimator_.covariance_)
        e_val, e_vec = np.linalg.eigh(corr)
        indices = e_val.argsort()[::-1]
        e_val, e_vec = e_val[indices], e_vec[:, indices]
        # market eigenvalues and eigenvectors
        market_e_val, market_e_vec = e_val[: self.n_markets], e_vec[:, : self.n_markets]
        # market correlation
        market_corr = market_e_vec @ np.diag(market_e_val) @ market_e_vec.T
        # Removing the market correlation
        corr -= market_corr
        corr, _ = cov_to_corr(corr)
        covariance = corr_to_cov(corr, std)
        self._set_covariance(covariance)
        return self


class EWCovariance(BaseCovariance):
    r"""Exponentially Weighted Covariance estimator.

    Estimator of the covariance using the historical exponentially weighted returns.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    alpha : float, default=0.2
       Exponential smoothing factor. The default value is `0.2`.

       :math:`0 < \alpha \leq 1`.

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
        Estimated covariance.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    def __init__(
        self,
        window_size: int | None = None,
        alpha: float = 0.2,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.alpha = alpha

    def fit(self, X: npt.ArrayLike, y=None):
        """Fit the Exponentially Weighted Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : EWCovariance
          Fitted estimator.
        """
        X = self._validate_data(X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        n_observations = X.shape[0]
        covariance = (
            pd.DataFrame(X)
            .ewm(alpha=self.alpha)
            .cov()
            .loc[(n_observations - 1, slice(None)), :]
            .to_numpy()
        )
        self._set_covariance(covariance)
        return self


class LedoitWolf(BaseCovariance, skc.LedoitWolf):
    """LedoitWolf Estimator.

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in [1]_.

    Read more in `scikit-learn
    <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html>`_.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.

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
        Estimated covariance.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_assets, n_assets)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the Ledoit and Wolf formula (see References)

    References
    ----------
    .. [1]  "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices".
        Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2.
        February 2004, pages 365-41.
    """

    def __init__(
        self,
        store_precision=True,
        assume_centered=False,
        block_size=1000,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.LedoitWolf.__init__(
            self,
            store_precision=store_precision,
            assume_centered=assume_centered,
            block_size=block_size,
        )

    def fit(self, X: npt.ArrayLike, y=None) -> "LedoitWolf":
        """Fit the Ledoit-Wolf shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : LedoitWolf
          Fitted estimator.
        """
        skc.LedoitWolf.fit(self, X)
        self._set_covariance(self.covariance_)
        return self


class OAS(BaseCovariance, skc.OAS):
    """Oracle Approximating Shrinkage Estimator as proposed in [1]_.

    Read more in `scikit-learn
    <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html>`_.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_assets, n_assets)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.

    References
    ----------
    .. [1] "Shrinkage algorithms for MMSE covariance estimation".
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
    """

    def __init__(
        self,
        store_precision=True,
        assume_centered=False,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.OAS.__init__(
            self,
            store_precision=store_precision,
            assume_centered=assume_centered,
        )

    def fit(self, X: npt.ArrayLike, y=None) -> "OAS":
        """Fit the Oracle Approximating Shrinkage covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : OAS
          Fitted estimator.
        """
        skc.OAS.fit(self, X)
        self._set_covariance(self.covariance_)
        return self


class ShrunkCovariance(BaseCovariance, skc.ShrunkCovariance):
    """Covariance estimator with shrinkage.

    Read more in `scikit-learn
    <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html>`_.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_assets, n_assets)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    The regularized covariance is given by:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    """

    def __init__(
        self,
        store_precision=True,
        assume_centered=False,
        shrinkage=0.1,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.ShrunkCovariance.__init__(
            self,
            store_precision=store_precision,
            assume_centered=assume_centered,
            shrinkage=shrinkage,
        )

    def fit(self, X: npt.ArrayLike, y=None) -> "ShrunkCovariance":
        """Fit the shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : ShrunkCovariance
          Fitted estimator.
        """
        skc.ShrunkCovariance.fit(self, X)
        self._set_covariance(self.covariance_)
        return self


class GraphicalLassoCV(BaseCovariance, skc.GraphicalLassoCV):
    """Sparse inverse covariance with cross-validated choice of the l1 penalty.

    Read more in `scikit-learn
    <https://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html>`_.

    Parameters
    ----------
    alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details. Range is [1, inf) for an integer.
        Range is (0, inf] for an array-like of floats.

    n_refinements : int, default=4
        The number of times the grid is refined. Not used if explicit
        values of alphas are passed. Range is [1, inf).

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - `CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs :class:`KFold` is used.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        Maximum number of iterations.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where number of features is greater
        than number of samples. Elsewhere prefer cd which is more numerically
        stable.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors.

    verbose : bool, default=False
        If verbose is True, the objective function and duality gap are
        printed at each iteration.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_assets, n_assets)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    alpha_ : float
        Penalization parameter selected.

    cv_results_ : dict of ndarrays
        A dict with keys:

        alphas : ndarray of shape (n_alphas,)
            All penalization parameters explored.

        split(k)_test_score : ndarray of shape (n_alphas,)
            Log-likelihood score on left-out data across (k)th fold.

            .. versionadded:: 1.0

        mean_test_score : ndarray of shape (n_alphas,)
            Mean of scores over the folds.

            .. versionadded:: 1.0

        std_test_score : ndarray of shape (n_alphas,)
            Standard deviation of scores over the folds.

            .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run for the optimal alpha.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    The search for the optimal penalization parameter (`alpha`) is done on an
    iteratively refined grid: first the cross-validated scores on a grid are
    computed, then a new refined grid is centered around the maximum, and so
    on.

    One of the challenges which is faced here is that the solvers can
    fail to converge to a well-conditioned estimate. The corresponding
    values of `alpha` then come out as missing values, but the optimum may
    be close to these missing values.

    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.
    """

    def __init__(
        self,
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        n_jobs=None,
        verbose=False,
        assume_centered=False,
        nearest: bool = False,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.GraphicalLassoCV.__init__(
            self,
            alphas=alphas,
            n_refinements=n_refinements,
            cv=cv,
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            n_jobs=n_jobs,
            verbose=verbose,
            assume_centered=assume_centered,
        )

    def fit(self, X, y=None) -> "GraphicalLassoCV":
        """Fit the GraphicalLasso covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : GraphicalLassoCV
          Fitted estimator.
        """
        skc.GraphicalLassoCV.fit(self, X)
        self._set_covariance(self.covariance_)
        return self
