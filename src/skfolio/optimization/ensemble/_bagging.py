import numpy as np
from sklearn.ensemble import BaggingRegressor

from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.ensemble._base import BaseComposition


class BaseComposedBaggingRegressor(BaggingRegressor, BaseComposition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SubsetResampling(BaseOptimization, BaseComposedBaggingRegressor):
    """
    A bagging ensemble that performs subset resampling with base optimization models.

    This class extends the `BaggingRegressor` and performs resampling with a set of base optimization models.
    The ensemble uses a specified base optimization model (`estimator`) and creates an ensemble of models by
    training on randomly sampled subsets of features.
    The resampling is performed without replacement.

    Parameters:
    -----------
    estimator : BaseOptimization, optional
        The base optimization model to use for building the ensemble.
        If not provided, the base model is set to None.

    n_estimators : int, optional (default=10)
        The number of base optimization models in the ensemble.

    max_features : float, optional (default=0.7)
        The fraction of features to randomly sample for each base optimization model.
        When the number of features is large decrease the max_features to a smaller percentage
        in order to let optimization methods run faster.

    warm_start : bool, optional (default=False)
        If True, reuse the solution of the previous call to fit as an initialization.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fitting and prediction.
        None means 1 unless in a joblib.parallel_backend context.

    random_state : int, RandomState instance, or None, optional (default=None)
        Controls the random resampling of features.
        Pass an int for reproducible results across multiple function calls.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    portfolio_params : dict or None, optional (default=None)
        Additional parameters to pass to the base optimization model.

    Attributes:
    ------------
    estimators_ : list of BaseOptimization
        The fitted base optimization models in the ensemble.

    named_estimators_ : dict of str to BaseOptimization
        The named fitted base optimization models in the ensemble.
    weights_ : ndarray of shape (n_features,)
        The average weights assigned to each feature across all base optimization models.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    Methods:
    ---------
    fit(X, y=None, sample_weight=None):
        Fit the ensemble to the input data. Overrides the base class method to perform subset resampling.

    Notes:
    ------
    This class inherits from `BaggingRegressor` and utilizes the subset resampling technique for building an ensemble
    of base optimization models.
    The resampling is performed without replacement, and the average weights of features
    across all base optimization models are computed and stored in the `weights_` attribute.
    """

    estimators_: list[BaseOptimization]
    named_estimators_: dict[str, BaseOptimization]

    def __init__(
        self,
        estimator: BaseOptimization = None,
        n_estimators: int = 10,
        max_features: float = 0.7,
        warm_start: bool = False,
        n_jobs: int | None = None,
        random_state=None,
        verbose=0,
        portfolio_params: dict | None = None,
    ):
        super(BaseComposedBaggingRegressor, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=max_features,
            bootstrap=False,
            bootstrap_features=False,
            oob_score=False,
            warm_start=warm_start,
            random_state=random_state,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.portfolio_params = portfolio_params

    def fit(self, X, y=None, sample_weight=None) -> "SubsetResampling":
        super(BaseComposedBaggingRegressor, self).fit(
            X, y=np.ones(X.shape[0]), sample_weight=sample_weight
        )
        inner_weights = np.zeros(X.shape[1])
        # important to collect the indices at which the individual estimator weights refer to
        for estimator, features_indices in zip(
            self.estimators_, self.estimators_features_, strict=True
        ):
            inner_weights[features_indices] += estimator.weights_

        np.divide(inner_weights, self.n_estimators, out=inner_weights)
        self.weights_ = inner_weights
        return self
