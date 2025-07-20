.. _model_selection:

.. currentmodule:: skfolio.model_selection

***************
Model Selection
***************

The Model Selection module extends `sklearn.model_selection` by adding additional
methods tailored for portfolio selection.

.. _cross_validation:

Cross-Validation Prediction
***************************
Every `skfolio` estimator is compatible with `sklearn.model_selection.cross_val_predict`.
We also implement our own :func:`cross_val_predict` for enhanced integration
with `Portfolio` and `Population` objects, as well as compatibility with
`CombinatorialPurgedCV`.

.. _data_leakage:
.. danger::

    When using `scikit-learn` selection tools like `KFold` or `train_test_split`, ensure
    that the parameter `shuffle` is set to `False` to avoid data leakage. Financial
    features often incorporate series that exhibit serial correlation (like ARMA
    processes) and shuffling the data will lead to leakage from the test set to the
    training set.

In `cross_val_predict`, the data is split according to the `cv` parameter.
The portfolio optimization estimator is fitted on the training set and portfolios are
predicted on the corresponding test set.

For `scikit-learn` cross-validations methods such as `Kfold` and `skfolio`'s
`WalkForward`, the output is a :class:`~skfolio.MultiPeriodPortfolio`, where
each :class:`~skfolio.Portfolio` corresponds to the prediction on a single train/test
split (resulting in K portfolios for `KFold`).

For combinatorial cross-validation methods such as :class:`CombinatorialPurgeCV` and
Monte Carlo-style methods such as :class:`MultipleRandomizedCV`, the output a
:class:`~skfolio.Population` containing multiple :class:`~skfolio.MultiPeriodPortfolio`.
This is because each test produces a collection of multiple paths rather than a single
path.

**Example:**

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import KFold

    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import WalkForward CombinatorialPurgedCV, cross_val_predict
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    # KFold
    # One single path -> pred is a MultiPeriodPortfolio
    pred = cross_val_predict(MeanRisk(), X, cv=KFold())
    print(pred.sharpe_ratio)
    np.asarray(pred)  # predicted returns vector

    # WalkForward
    # One single path -> pred is a MultiPeriodPortfolio
    pred = cross_val_predict(
        MeanRisk(),
        X,
        cv=WalkForward(test_size=3, train_size=12, freq="WOM-3FRI")
    )
    print(pred.sharpe_ratio)
    np.asarray(pred)  # predicted returns vector

    # CombinatorialPurgedCV
    # Multiple paths -> pred is a Population of MultiPeriodPortfolio
    pred = cross_val_predict(MeanRisk(), X, cv=CombinatorialPurgedCV())
    print(pred.summary())
    print(np.asarray(pred))  # predicted returns matrix

    # MultipleRandomizedCV
    # Multiple paths -> pred is a Population of MultiPeriodPortfolio
    pred = cross_val_predict(
        MeanRisk(),
        X,
        cv=MultipleRandomizedCV(
            walk_forward=WalkForward(test_size=1, train_size=2),
            n_subsamples=2,
            asset_subset_size=3,
        )
    )
    print(pred.summary())
    print(np.asarray(pred))  # predicted returns matrix



Walk-Forward Cross-Validation
*****************************
The :class:`WalkForward` splitter divides time series data using a walk‑forward approach.
Unlike `sklearn.model_selection.TimeSeriesSplit`, you specify the number of training
and test samples rather than the number of splits, making it more suitable for portfolio
cross‑validation.

If your data is a DataFrame indexed by a :class:`pandas.DatetimeIndex`, you can split it
using specific datetime frequencies and offsets.

Combinatorial Purged Cross-Validation
*************************************
Compared to `KFold`, which splits the data into k folds and generates one single testing
path, the :class:`CombinatorialPurgedCV` uses the combination of multiple
train/test sets to generate multiple testing paths.

To avoid data leakage, purging and embargoing can be performed.

Purging consist of removing from the training set all observations
whose labels overlapped in time with those labels included in the testing set.
Embargoing consist of removing from the training set observations that immediately
follow an observation in the testing set, since financial features often incorporate
series that exhibit serial correlation (like ARMA processes).

When used with :func:`cross_val_predict`, the object returned is a
:class:`~skfolio.population.Population` of
:class:`~skfolio.portfolio.MultiPeriodPortfolio` representing each prediction path.

**Example:**

.. code-block:: python

    from skfolio import RatioMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import CombinatorialPurgedCV, cross_val_predict
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    pred = cross_val_predict(MeanRisk(), X, cv=CombinatorialPurgedCV())
    print(pred.summary())

    portfolio = pred.quantile(measure=RatioMeasure.SHARPE_RATIO, q=0.95)
    print(portfolio.annualized_sharpe_ratio)


The default parameters of the `CombinatorialPurgedCV` are `n_folds=10` and
`n_test_folds=8`. You may want to choose these parameters to target a number of test
paths and an average training size. The later depends on the number of observations.
For that, you can use the function :func:`optimal_folds_number` as shown in the example
:ref:`sphx_glr_auto_examples_clustering_plot_3_hrp_vs_herc.py`.

.. code-block:: python

    n_folds, n_test_folds = optimal_folds_number(
        n_observations=X_test.shape[0],
        target_n_test_paths=100,
        target_train_size=252,
    )

    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)
    cv.summary(X_test)



Multiple Randomized Cross-Validation
************************************
The :class:`MultipleRandomizedCV` cross‑validation strategy, based on the
"Multiple Randomized Backtests" methodology of Palomar & Zhou, performs a Monte
Carlo–style evaluation by repeatedly sampling **distinct** asset subsets (without
replacement) and **contiguous** time windows. It then applies an inner walk‑forward
split to each subsample, capturing both temporal and cross‑sectional variability in
performance.


When used with :func:`cross_val_predict`, the object returned is a
:class:`~skfolio.population.Population` of
:class:`~skfolio.portfolio.MultiPeriodPortfolio` representing each prediction path.

.. code-block:: python

    import numpy as np
    from skfolio.datasets import load_sp500_dataset, load_factors_dataset
    from skfolio.model_selection import WalkForward, MultipleRandomizedCV, cross_val_predict
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(test_size=3, train_size=6, freq="WOM-3FRI"),
        n_subsamples=100,
        asset_subset_size=3,
        window_size=2*252,
    )

    pred = cross_val_predict(MeanRisk(), X, cv=cv)
    print(pred.summary())

    portfolio = pred.quantile(measure=RatioMeasure.SHARPE_RATIO, q=0.95)
    print(portfolio.annualized_sharpe_ratio)
