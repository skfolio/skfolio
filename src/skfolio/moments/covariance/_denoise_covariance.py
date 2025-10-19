"""Covariance Denoising Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import scipy.optimize as sco
import sklearn.neighbors as skn
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._empirical_covariance import EmpiricalCovariance
from skfolio.utils.stats import corr_to_cov, cov_to_corr
from skfolio.utils.tools import check_estimator


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
        If this is set to True, the Higham & Nick (2002) algorithm is used to find the
        nearest PD covariance, otherwise the eigenvalues are clipped to a threshold
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
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.covariance_estimator = covariance_estimator

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            covariance_estimator=self.covariance_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "DenoiseCovariance":
        """Fit the Covariance Denoising estimator.

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
        self : DenoiseCovariance
           Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        # noinspection PyArgumentList
        self.covariance_estimator_.fit(X, y, **routed_params.covariance_estimator.fit)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = skv.validate_data(self, X)
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
