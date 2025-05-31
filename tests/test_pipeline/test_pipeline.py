import pytest
from sklearn import config_context, set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skfolio.moments import (
    ImpliedCovariance,
)
from skfolio.optimization import InverseVolatility, MeanRisk
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectKExtremes,
    SelectNonDominated,
)
from skfolio.prior import EmpiricalPrior


@pytest.mark.parametrize(
    "transformer",
    [
        DropCorrelated(threshold=0.9),
        DropZeroVariance(threshold=1e-6),
        SelectNonDominated(min_n_assets=15, threshold=0),
        SelectKExtremes(k=10, highest=True),
    ],
)
def test_transformer(X, transformer):
    set_config(transform_output="pandas")

    X_train, X_test = train_test_split(X, shuffle=False, test_size=0.3)

    pipe = Pipeline([("pre_selection", transformer), ("mean_risk", MeanRisk())])
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
