from sklearn import config_context, set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skfolio.moments import (
    ImpliedCovariance,
)
from skfolio.optimization import InverseVolatility, MeanRisk
from skfolio.pre_selection import DropCorrelated, SelectKExtremes, SelectNonDominated
from skfolio.prior import EmpiricalPrior


def test_transformer(X):
    set_config(transform_output="pandas")

    X_train, X_test = train_test_split(X, shuffle=False, test_size=0.3)

    pipe = Pipeline(
        [("pre_selection", DropCorrelated(threshold=0.9)), ("mean_risk", MeanRisk())]
    )
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    _ = portfolio.sharpe_ratio

    pipe = Pipeline(
        [
            ("pre_selection", SelectNonDominated(min_n_assets=15, threshold=0)),
            ("mean_risk", MeanRisk()),
        ]
    )
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    _ = portfolio.sharpe_ratio

    pipe = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10, highest=True)),
            ("optimization", MeanRisk()),
        ]
    )
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    _ = portfolio.sharpe_ratio


def test_meta_data_routing_pipeline(X, implied_vol):
    with config_context(enable_metadata_routing=True, transform_output="pandas"):
        model = InverseVolatility(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        pipe = Pipeline(
            [("pre_selection", SelectKExtremes(k=10)), ("mean_risk", model)]
        )
        pipe.fit(X, implied_vol=implied_vol)

        _ = pipe.predict(X)
