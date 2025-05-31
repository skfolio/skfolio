import numpy as np

from skfolio import RiskMeasure
from skfolio.distribution import VineCopula
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk
from skfolio.prior import FactorModel, SyntheticData


def test_synthetic_data(X):
    model = SyntheticData()
    model.fit(X)
    res = model.return_distribution_
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
    res = model.return_distribution_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (1000, 20)
    assert res.cholesky is not None


def test_factor_stress_test(X, y):
    model = FactorModel(
        factor_prior_estimator=SyntheticData(
            distribution_estimator=VineCopula(
                central_assets=["QUAL"], log_transform=True, n_jobs=-1, random_state=42
            ),
            n_samples=10000,
            sample_args=dict(conditioning={"QUAL": -0.8}),
        )
    )
    model.fit(X, y)
    res = model.return_distribution_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (10000, 20)
    assert res.cholesky is not None
    np.testing.assert_almost_equal(
        res.returns[:5, :5],
        [
            [-0.86294093, -1.11156425, -0.75854301, -0.82283136, -0.63883654],
            [-0.9419662, -1.27665467, -0.85295726, -0.85935132, -0.70171624],
            [-0.82122944, -1.01823713, -0.82881924, -0.85346634, -0.68590999],
            [-0.89897468, -1.17230283, -0.7521866, -0.81503241, -0.63313694],
            [-0.84638832, -1.08273094, -0.77658093, -0.83103323, -0.65410704],
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
