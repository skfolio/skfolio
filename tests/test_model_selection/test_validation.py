"""Test Validation module."""

import numpy as np
import sklearn.model_selection as sks
from sklearn import config_context
from sklearn.model_selection import KFold

from skfolio import MultiPeriodPortfolio, Population
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.moments import (
    ImpliedCovariance,
)
from skfolio.optimization import InverseVolatility, MeanRisk
from skfolio.prior import EmpiricalPrior


def test_validation(X):
    model = MeanRisk()
    n_observations = X.shape[0]
    for cv in [
        sks.KFold(),
        WalkForward(test_size=n_observations // 5, train_size=n_observations // 5),
    ]:
        pred = cross_val_predict(
            model, X, cv=cv, portfolio_params=dict(name="ptf_test")
        )

        pred2 = MultiPeriodPortfolio()
        for train, test in cv.split(X):
            model.fit(X.take(train))
            pred2.append(model.predict(X.take(test)))

        assert isinstance(pred, MultiPeriodPortfolio)
        assert pred.name == "ptf_test"
        assert np.array_equal(pred.returns_df.index, pred2.returns_df.index)
        np.testing.assert_almost_equal(np.asarray(pred), np.asarray(pred2))

        assert len(pred.portfolios) == cv.get_n_splits(X)


def test_validation_combinatorial(X):
    model = MeanRisk()
    n_observations = X.shape[0]
    cv = CombinatorialPurgedCV()

    pred = cross_val_predict(model, X, cv=cv, portfolio_params=dict(name="test"))

    cv.split(X)
    cv.get_path_ids()

    assert isinstance(pred, Population)
    assert len(pred) == cv.n_test_paths
    for p in pred:
        assert isinstance(p, MultiPeriodPortfolio)
        assert len(p.portfolios) == cv.n_folds
        assert len(p) == cv.n_folds
        assert p.n_observations == n_observations


def test_meta_data_routing_cross_validation(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = InverseVolatility(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        cv = KFold()

        _ = cross_val_predict(model, X, params={"implied_vol": implied_vol}, cv=cv)
