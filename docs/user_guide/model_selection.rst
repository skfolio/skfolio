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

For non-combinatorial cross-validation like ``Kfold``, the output is the predicted
:class:`~skfolio.MultiPeriodPortfolio` where each
:class:`~skfolio.Portfolio` corresponds to the prediction on each train/test
pair (K portfolios for ``Kfold``).

For combinatorial cross-validation like :class:`CombinatorialPurgeCV`, the output is the
predicted :class:`~skfolio.Population` of multiple
:class:`~skfolio.MultiPeriodPortfolio`. This is because each test output is a
collection of multiple paths instead of one single path.

**Example:**

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import KFold

    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import CombinatorialPurgedCV, cross_val_predict
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    # One single path -> pred is a MultiPeriodPortfolio
    pred = cross_val_predict(MeanRisk(), X, cv=KFold())
    print(pred.sharpe_ratio)
    np.asarray(pred)  # predicted returns vector

    # Multiple paths -> pred is a Population of MultiPeriodPortfolio
    pred = cross_val_predict(MeanRisk(), X, cv=CombinatorialPurgedCV())
    print(pred.summary())
    print(np.asarray(pred))  # predicted returns matrix



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

