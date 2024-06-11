import datetime as dt

import numpy as np
import pytest
from sklearn import set_config, config_context
from sklearn.model_selection import KFold
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.moments import (
    DenoiseCovariance,
    DetoneCovariance,
    EWCovariance,
    EmpiricalCovariance,
    GerberCovariance,
    ImpliedCovariance
)
from skfolio.preprocessing import prices_to_returns
import datetime as dt
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior
from skfolio.optimization import InverseVolatility
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.pre_selection import SelectKExtremes

set_config(enable_metadata_routing=True)


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    X = X.loc[dt.date(2010, 1, 1) :]
    return X

@pytest.fixture(scope="module")
def implied_vol():
    implied_vol = load_sp500_implied_vol_dataset()
    return implied_vol

def test_meta_data_routing(X, implied_vol):

    model = EmpiricalPrior(
        covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
    )

    with pytest.raises(ValueError):
        model.fit(X)

    model.fit(X, implied_vol=implied_vol)


def test_meta_data_routing_optimization(X, implied_vol):

    model = InverseVolatility(
        prior_estimator= EmpiricalPrior(
        covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
    )
    )

    with pytest.raises(ValueError):
        model.fit(X)

    model.fit(X, implied_vol=implied_vol)

def test_meta_data_routing_cross_validation(X, implied_vol):

    model = InverseVolatility(
        prior_estimator=EmpiricalPrior(
            covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
        )
    )

    cv = KFold()

    pred = cross_val_predict(model, X,params={"implied_vol":implied_vol}, cv=cv)


def test_meta_data_routing_pipeline(X, implied_vol):
    set_config(transform_output="pandas")


    model = InverseVolatility(
        prior_estimator=EmpiricalPrior(
            covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
        )
    )

    pipe = Pipeline(
        [("pre_selection", SelectKExtremes(k=10)), ("mean_risk",model)]
    )
    pipe.fit(X,implied_vol=implied_vol)

    portfolio = pipe.predict(X)


