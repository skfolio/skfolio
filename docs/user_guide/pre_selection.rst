.. _pre_selection:

.. currentmodule:: skfolio.pre_selection

***************************
Pre-Selection Transformers
***************************

A :ref:`Pre-Selection transformer <pre_selection_ref>` performs a pre-selection on the
initial assets universe.

It follows the same API as scikit-learn's `estimator`: the `fit_transform` method takes
`X` as the assets returns and returns a new `X` with only the pre-selected assets.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)


Available transformers are:
    * :class:`DropZeroVariance`
    * :class:`DropCorrelated`
    * :class:`SelectComplete`
    * :class:`SelectKExtremes`
    * :class:`SelectNonDominated`
    * :class:`SelectNonExpiring`

**Example:**

.. code-block:: python

    from sklearn import set_config

    from skfolio.datasets import load_sp500_dataset
    from skfolio.pre_selection import DropCorrelated
    from skfolio.preprocessing import prices_to_returns

    set_config(transform_output="pandas")

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    print(X.shape)

    model = DropCorrelated(threshold=0.5)
    new_X = model.fit_transform(X)
    print(new_X.shape)



Pre-Selection transformers are fully compatible with :class:`sklearn.pipeline.Pipeline`:

**Example:**

.. code-block:: python

    from sklearn import set_config
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.pre_selection import DropCorrelated
    from skfolio.preprocessing import prices_to_returns

    set_config(transform_output='pandas')

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, shuffle=False, test_size=0.3)

    pipe = Pipeline([('pre_selection', DropCorrelated(threshold=0.9)),
                     ('mean_risk', MeanRisk())])
    pipe.fit(X_train)

    portfolio = pipe.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)

