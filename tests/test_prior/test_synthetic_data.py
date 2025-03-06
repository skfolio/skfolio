import numpy as np

from skfolio import RiskMeasure
from skfolio.distribution import VineCopula
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk
from skfolio.prior import FactorModel, SyntheticData


def test_synthetic_data(X):
    model = SyntheticData()
    model.fit(X)
    res = model.prior_model_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (1000, 20)
    assert res.cholesky is None


def test_factor_synthetic_data(X, y):
    model = FactorModel(
        factor_prior_estimator=SyntheticData(),
    )
    model.fit(X, y)
    res = model.prior_model_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (1000, 20)
    assert res.cholesky is not None


def test_factor_stress_test(X, y):
    model = FactorModel(
        factor_prior_estimator=SyntheticData(
            distribution_estimator=VineCopula(
                central_assets=["QUAL"],
                log_transform=True,
                n_jobs=-1,
            ),
            n_samples=10000,
            sample_args=dict(random_state=42, conditioning={"QUAL": -0.8}),
        )
    )
    model.fit(X, y)
    res = model.prior_model_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (10000, 20)
    assert res.cholesky is not None
    print(res.returns[:5, :5])
    np.testing.assert_almost_equal(
        res.returns[:5, :5],
        [
            [-0.90533426, -1.07783261, -0.64652144, -0.74521958, -0.55290306],
            [-0.79325235, -0.99783261, -0.87305837, -0.87861587, -0.72493803],
            [-0.3986848, 0.0648375, -0.88926408, -0.85734484, -0.70829534],
            [-0.90957477, -1.23326199, -0.84512581, -0.86158363, -0.701079],
            [-0.86998068, -1.1299855, -0.78104609, -0.83133594, -0.65676247],
        ],
        5,
    )


def test_optimization_synthetic_data(X):
    model = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=SyntheticData(
            distribution_estimator=VineCopula(log_transform=True, n_jobs=-1),
            n_samples=2000,
        ),
    )
    cv = WalkForward(train_size=252, test_size=100)
    _ = cross_val_predict(model, X, cv=cv, n_jobs=-1)
