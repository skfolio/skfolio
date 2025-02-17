import numpy as np

from skfolio import RiskMeasure
from skfolio.distribution import VineCopula
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk
from skfolio.prior import FactorModel, SyntheticReturns


def test_synthetic_returns(X):
    model = SyntheticReturns()
    model.fit(X)
    res = model.prior_model_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    assert res.returns.shape == (1000, 20)
    assert res.cholesky is None


def test_factor_synthetic_returns(X, y):
    model = FactorModel(
        factor_prior_estimator=SyntheticReturns(),
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
        factor_prior_estimator=SyntheticReturns(
            distribution_estimator=VineCopula(
                central_assets=["QUAL"],
                log_transform=True,
                n_jobs=-1,
            ),
            n_samples=10000,
            sample_args=dict(
                random_state=42, conditioning_samples={"QUAL": -0.8 * np.ones(10000)}
            ),
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
            [-0.63838434, -0.76033558, -0.42009942, -0.49873723, -0.36229766],
            [-0.51528883, -0.68566953, -0.64283515, -0.63888747, -0.53198762],
            [-0.3069065, -0.07478326, -0.64390488, -0.61327874, -0.51388077],
            [-0.63683385, -0.9289238, -0.61233964, -0.61964923, -0.50635838],
            [-0.59912879, -0.8245584, -0.54689312, -0.5874889, -0.46083999],
        ],
        5,
    )


def test_optimization_synthetic_returns(X):
    model = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=SyntheticReturns(
            distribution_estimator=VineCopula(log_transform=True, n_jobs=-1),
            n_samples=2000,
        ),
    )
    cv = WalkForward(train_size=252, test_size=100)
    _ = cross_val_predict(model, X, cv=cv, n_jobs=-1)
