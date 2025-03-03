import numpy as np

from skfolio import RiskMeasure
from skfolio.distribution import VineCopula
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk
from skfolio.prior import FactorModel, SyntheticData


def test_synthetic_returns(X):
    model = SyntheticData()
    model.fit(X)
    res = model.prior_model_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (1000, 20)
    assert res.cholesky is None


def test_factor_synthetic_returns(X, y):
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
    np.testing.assert_almost_equal(
        res.returns[:5, :5],
        [
            [-0.90499992, -1.07832384, -0.64602842, -0.74537033, -0.55255603],
            [-0.79369733, -0.99770418, -0.872875, -0.87823451, -0.72478694],
            [-0.40084947, 0.05574429, -0.88890724, -0.85822426, -0.7082633],
            [-0.90962344, -1.2332441, -0.84503044, -0.86151357, -0.7010059],
            [-0.86955051, -1.13060295, -0.78101306, -0.83175341, -0.65676623],
        ],
        5,
    )


def test_optimization_synthetic_returns(X):
    model = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=SyntheticData(
            distribution_estimator=VineCopula(log_transform=True, n_jobs=-1),
            n_samples=2000,
        ),
    )
    cv = WalkForward(train_size=252, test_size=100)
    _ = cross_val_predict(model, X, cv=cv, n_jobs=-1)
