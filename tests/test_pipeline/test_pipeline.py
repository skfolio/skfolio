import pytest
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.pre_selection import DropCorrelated, SelectKExtremes, SelectNonDominated
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    return X


def test_transformer(X):
    set_config(transform_output="pandas")

    X_train, X_test = train_test_split(X, shuffle=False, test_size=0.3)

    pipe = Pipeline(
        [("pre_selection", DropCorrelated(threshold=0.9)), ("mean_risk", MeanRisk())]
    )
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    _ = portfolio.sharpe_ratio

    pipe = Pipeline([
        ("pre_selection", SelectNonDominated(min_n_assets=15, threshold=0)),
        ("mean_risk", MeanRisk()),
    ])
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    _ = portfolio.sharpe_ratio

    pipe = Pipeline([
        ("pre_selection", SelectKExtremes(k=10, highest=True)),
        ("optimization", MeanRisk()),
    ])
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    _ = portfolio.sharpe_ratio
