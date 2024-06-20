"""Stacking Optimization estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from copy import deepcopy

import numpy as np
import numpy.typing as npt
import sklearn as sk
import sklearn.model_selection as skm
import sklearn.utils as sku
import sklearn.utils.parallel as skp
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.measures import RatioMeasure
from skfolio.model_selection import BaseCombinatorialCV, cross_val_predict
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex import MeanRisk
from skfolio.optimization.ensemble._base import BaseComposition
from skfolio.utils.tools import check_estimator, fit_single_estimator


class StackingOptimization(BaseOptimization, BaseComposition):
    """Stack of optimizations with a final optimization.

    Stacking Optimization is an ensemble method that consists in stacking the output of
    individual portfolio optimizations with a final portfolio optimization.

    The weights are the dot-product of individual optimizations weights with the final
    optimization weights.

    Stacking allows to use the strength of each individual portfolio optimization by
    using their output as input of a final portfolio optimization.

    To avoid data leakage, out-of-sample estimates are used to fit the outer
    optimization.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Parameters
    ----------
    estimators : list[tuple[str, BaseOptimization]]
        :ref:`Optimization estimators <optimization>` which will be stacked together.
        Each element of the list is defined as a tuple of string (i.e. name) and an
        :ref:`optimization estimator <optimization>`.

    final_estimator : BaseOptimization, optional
        A final :ref:`optimization estimator <optimization>` which will be used to
        combine the base estimators.
        The default (`None`) is to use :class:`~skfolio.optimization.MeanRisk`.

    cv : BaseCrossValidator | BaseCombinatorialCV | int | "prefit" | "ignore", optional
        Determines the cross-validation splitting strategy used in `cross_val_predict`
        to train the `final_estimator`.
        The default (`None`) is to use the 5-fold cross validation `KFold()`.
        Possible inputs for `cv` are:

            * "ignore": no cross-validation is used (note that it will likely lead to data leakage with a high risk of overfitting)
            * integer, to specify the number of folds in a `KFold`
            * An object to be used as a cross-validation generator
            * An iterable yielding train, test splits
            * "prefit" to assume the `estimators` are prefit, and skip cross validation
            * A :class:`~skfolio.model_selection.CombinatorialPurgedCV`

        If a `CombinatorialCV` cross-validator is used, each cluster out-of-sample
        outputs becomes a collection of multiple paths instead of one single path. The
        selected out-of-sample path among this collection of paths is chosen according
        to the `quantile` and `quantile_measure` parameters.

        If "prefit" is passed, it is assumed that all `estimators` have been fitted
        already. The `final_estimator_` is trained on the `estimators` predictions on
        the full training set and are **not** cross validated predictions.
        Please note that if the models have been trained on the same data to train the
        stacking model, there is a very high risk of overfitting.

    n_jobs : int, optional
        The number of jobs to run in parallel for `fit` of all `estimators`.
        The value `-1` means using all processors.
        The default (`None`) means 1 unless in a `joblib.parallel_backend` context.


    quantile : float, default=0.5
        Quantile for a given measure (`quantile_measure`) of the out-of-sample
        inner-estimator paths when the `cv` parameter is a
        :class:`~skfolio.model_selection.CombinatorialPurgedCV` cross-validator.
        The default value is `0.5` corresponding to the path with the median measure.
        (see `cv`)

    quantile_measure : PerfMeasure or RatioMeasure or RiskMeasure or ExtraRiskMeasure, default=RatioMeasure.SHARPE_RATIO
        Measure used for the quantile path selection (see `quantile` and `cv`).
        The default is `RatioMeasure.SHARPE_RATIO`.

    verbose : int, default=0
        The verbosity level. The default value is `0`.

    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name` is copied from the optimization
        model and systematically passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Weights of the assets.

    estimators_ : list[BaseOptimization]
        The elements of the `estimators` parameter, having been fitted on the
        training data. When `cv="prefit"`, `estimators_`
        is set to `estimators` and is not fitted again.

    named_estimators_ : dict[str, BaseOptimization]
        Attribute to access any fitted sub-estimators by name.

    final_estimator_ : BaseOptimization
        The fitted `final_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    estimators_: list[BaseOptimization]
    final_estimator_: BaseOptimization
    named_estimators_: dict[str, BaseOptimization]

    def __init__(
        self,
        estimators: list[tuple[str, BaseOptimization]],
        final_estimator: BaseOptimization | None = None,
        cv: skm.BaseCrossValidator | BaseCombinatorialCV | str | int | None = None,
        quantile: float = 0.5,
        quantile_measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        n_jobs: int | None = None,
        verbose: int = 0,
        portfolio_params: dict | None = None,
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.quantile = quantile
        self.quantile_measure = quantile_measure
        self.n_jobs = n_jobs
        self.verbose = verbose

    @property
    def named_estimators(self):
        """Dictionary to access any fitted sub-estimators by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return sku.Bunch(**dict(self.estimators))

    def _validate_estimators(self) -> tuple[list[str], list[BaseOptimization]]:
        """Validate the `estimators` parameter.

        Returns
        -------
        names : list[str]
            The list of estimators names.
        estimators : list[BaseOptimization
            The list of optimization estimators.
        """
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators, strict=True)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        return names, estimators

    def set_params(self, **params):
        """Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("estimators", **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("estimators", deep=deep)

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> "StackingOptimization":
        """Fit the Stacking Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        Returns
        -------
        self : StackingOptimization
           Fitted estimator.
        """
        names, all_estimators = self._validate_estimators()
        self.final_estimator_ = check_estimator(
            self.final_estimator,
            default=MeanRisk(),
            check_type=BaseOptimization,
        )

        if self.cv == "prefit":
            self.estimators_ = []
            for estimator in all_estimators:
                skv.check_is_fitted(estimator)
                self.estimators_.append(estimator)
        else:
            # Fit the base estimators on the whole training data. Those
            # base estimators will be used to retrieve the inner weights.
            # They are exposed publicly.
            # noinspection PyCallingNonCallable
            self.estimators_ = skp.Parallel(n_jobs=self.n_jobs)(
                skp.delayed(fit_single_estimator)(sk.clone(est), X, y)
                for est in all_estimators
            )

        self.named_estimators_ = {
            name: estimator
            for name, estimator in zip(names, self.estimators_, strict=True)
        }

        inner_weights = np.array([estimator.weights_ for estimator in self.estimators_])

        # To train the final-estimator using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.
        # To ensure that the data provided to each estimator are the same,
        # we need to set the random state of the cv if there is one and we
        # need to take a copy.
        if self.cv in ["prefit", "ignore"]:
            X_pred = np.array(
                [estimator.predict(X) for estimator in self.estimators_]
            ).T
        else:
            cv = skm.check_cv(self.cv)
            if hasattr(cv, "random_state") and cv.random_state is None:
                cv.random_state = np.random.RandomState()
            # noinspection PyCallingNonCallable
            cv_predictions = skp.Parallel(n_jobs=self.n_jobs)(
                skp.delayed(cross_val_predict)(
                    sk.clone(estimator),
                    X,
                    y,
                    cv=deepcopy(cv),
                    method="predict",
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                for estimator in all_estimators
            )

            # We validate and convert to numpy array only after base-estimator fitting
            # to keep the assets names in case they are used in the estimator.
            if y is not None:
                _, y = self._validate_data(X, y, multi_output=True)
            else:
                _ = self._validate_data(X)

            if isinstance(self.cv, BaseCombinatorialCV):
                X_pred = np.array(
                    [
                        pred.quantile(measure=self.quantile_measure, q=self.quantile)
                        for pred in cv_predictions
                    ]
                ).T
            else:
                X_pred = np.array(cv_predictions).T
                if y is not None:
                    test_indices = np.sort(
                        np.concatenate([test for _, test in cv.split(X, y)])
                    )
                    y = y[test_indices]

        fit_single_estimator(self.final_estimator_, X_pred, y)
        outer_weights = self.final_estimator_.weights_
        self.weights_ = outer_weights @ inner_weights
        return self
