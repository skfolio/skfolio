.. _model_selection:

.. currentmodule:: skfolio.model_selection

***************
Model Selection
***************

The Model Selection module extends `sklearn.model_selection` by adding additional
methods tailored for portfolio selection.

Online Learning
***************

In addition to fold-based cross-validation utilities,
`skfolio.model_selection` provides stateful online utilities for estimators
that support `partial_fit`. These utilities update a single estimator through
time instead of fitting an independent clone on each split, which is
particularly useful for exponentially weighted moments and portfolio optimizers
built on top of them.

See :ref:`online_learning` for the full workflow and the differences between
online evaluation and standard cross-validation.

.. _cross_validation:

Cross-Validation Prediction
***************************
Every `skfolio` estimator is compatible with `sklearn.model_selection.cross_val_predict`.
We also implement our own :func:`cross_val_predict` for enhanced integration
with `Portfolio` and `Population` objects, as well as compatibility with
:class:`CombinatorialPurgedCV` and :class:`MultipleRandomizedCV`.

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

For `scikit-learn` cross-validation methods such as `KFold` and `skfolio`'s
`WalkForward`, the output is a :class:`~skfolio.portfolio.MultiPeriodPortfolio`, where
each :class:`~skfolio.portfolio.Portfolio` corresponds to the prediction on a single
train/test split (resulting in K portfolios for `KFold`).

For combinatorial cross-validation methods such as :class:`CombinatorialPurgedCV` and
Monte Carlo-style methods such as :class:`MultipleRandomizedCV`, the output is a
:class:`~skfolio.population.Population` containing multiple
:class:`~skfolio.portfolio.MultiPeriodPortfolio`. This is because each test produces a
collection of multiple paths rather than a single path.

Portfolio parameters can be set in the portfolio optimizer's `portfolio_params` or
passed to `cross_val_predict`. The parameters shared by
:class:`~skfolio.portfolio.Portfolio` and
:class:`~skfolio.portfolio.MultiPeriodPortfolio` (`compounded`, `risk_free_rate`,
`annualization_factor`, `fitness_measures` and the risk measure parameters) are applied
to the resulting `MultiPeriodPortfolio` and to each `Portfolio` it contains. A value
passed to `cross_val_predict` takes precedence over the optimizer's `portfolio_params`.
When omitted, it is inherited from the optimizer's `portfolio_params`, and
`risk_free_rate` falls back to the optimizer's `risk_free_rate` parameter when it has
one. For example, `MeanRisk(portfolio_params={"compounded": True})` produces a
compounded `MultiPeriodPortfolio` from `cross_val_predict` without repeating the
setting. These parameters only affect how the portfolios are measured: a
`risk_free_rate` passed to `cross_val_predict` does not change the optimizer's own
`risk_free_rate`.

`weight_drift` applies to each `Portfolio` of the path. With `weight_drift=True`, the
weights held within each test window drift with the asset returns, and the path runs
sequentially. A value passed to `cross_val_predict` overrides the optimizer's
`portfolio_params`. Optimizer parameters such as `transaction_costs`,
`management_fees` and `previous_weights` are not accepted in the function's
`portfolio_params`: set them on the optimizer, which forwards them to the predicted
`Portfolio` objects. `name`, `tag`, `sample_weight` and `check_observations_order`
apply to the resulting `MultiPeriodPortfolio` only.

For example, to evaluate a drifted, compounded path with transaction costs:

.. code-block:: python

    pred = cross_val_predict(
        MeanRisk(transaction_costs=0.001 / 5),
        X,
        cv=WalkForward(test_size=5, train_size=252),
        portfolio_params={"weight_drift": True, "compounded": True},
    )

With a sequential splitter, the `ending_weights` of each portfolio are passed as
`previous_weights` to the next fit. They equal the target weights when
`weight_drift=False` and the weights after the last observation when
`weight_drift=True`, so transaction costs and `max_turnover` are then measured from the
holdings a fund would trade from. A failed period keeps the last successful ending
weights. The sequential path requires one portfolio per fold and raises for estimators
that return a :class:`~skfolio.population.Population`. With non-sequential splitters
such as `KFold`, drift is evaluated inside each test fold and is not propagated. See
:ref:`backtesting_and_evaluation` for the choice between `weight_drift=False` and
`weight_drift=True`.

**Example:**

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import KFold

    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import (
        CombinatorialPurgedCV,
        WalkForward,
        cross_val_predict,
    )
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

Purging consists of removing from the training set all observations
whose labels overlapped in time with those labels included in the testing set.
Embargoing consists of removing from the training set observations that immediately
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
paths and an average training size. The latter depends on the number of observations.
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
"Multiple Randomized Backtests" methodology of Palomar, performs a Monte
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
