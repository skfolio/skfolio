"""Graphical Lasso CV Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import sklearn.covariance as skc

from skfolio.moments.covariance._base import BaseCovariance


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
        nearest: bool = True,
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

    def fit(self, X, y=None, **fit_params) -> "GraphicalLassoCV":
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
        skc.GraphicalLassoCV.fit(self, X, **fit_params)
        self._set_covariance(self.covariance_)
        return self
