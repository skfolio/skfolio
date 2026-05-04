from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import scipy.stats as scipy_stats

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import (
    CSWeighting,
    FactorModel,
    ReturnDistribution,
    TimeSeriesFactorModel,
)
from skfolio.uncertainty_set import CompactCovarianceUncertaintySet, UncertaintySet
from skfolio.uncertainty_set._orthogonal import (
    OrthogonalCovarianceUncertaintySet,
    OrthogonalMuUncertaintySet,
)


@dataclass(frozen=True)
class FactorCase:
    X: np.ndarray
    return_distribution: ReturnDistribution
    loading: np.ndarray
    idio_variance: np.ndarray
    regression_weights: np.ndarray | None


@pytest.fixture
def make_factor_case():
    def _make_factor_case(
        *,
        n_assets: int = 8,
        n_factors: int = 3,
        n_obs: int = 120,
        seed: int = 0,
        with_regression_weights: bool = False,
    ) -> FactorCase:
        rng = np.random.default_rng(seed)
        loading = rng.standard_normal((n_assets, n_factors))
        factor_covariance = np.eye(n_factors) * 0.01
        factor_mu = np.zeros(n_factors)
        idio_variance = rng.uniform(0.001, 0.01, size=n_assets)

        factor_returns = rng.multivariate_normal(
            factor_mu, factor_covariance, size=n_obs
        )
        idio_returns = rng.standard_normal((n_obs, n_assets)) * np.sqrt(idio_variance)
        returns = factor_returns @ loading.T + idio_returns
        covariance = loading @ factor_covariance @ loading.T + np.diag(idio_variance)
        mu = rng.standard_normal(n_assets) * 0.001

        regression_weights = None
        if with_regression_weights:
            regression_weights = rng.uniform(0.5, 5.0, size=(n_obs, n_assets))

        factor_model = FactorModel(
            observations=np.arange(n_obs),
            asset_names=np.array([f"asset_{i}" for i in range(n_assets)]),
            factor_names=np.array([f"factor_{i}" for i in range(n_factors)]),
            factor_families=None,
            loading_matrix=loading,
            exposures=None,
            factor_covariance=factor_covariance,
            factor_mu=factor_mu,
            factor_returns=factor_returns,
            idio_covariance=idio_variance,
            idio_mu=None,
            idio_returns=idio_returns,
            idio_variances=np.tile(idio_variance, (n_obs, 1)),
            regression_weights=regression_weights,
        )
        return_distribution = ReturnDistribution(
            mu=mu,
            covariance=covariance,
            returns=returns,
            factor_model=factor_model,
        )
        return FactorCase(
            X=returns,
            return_distribution=return_distribution,
            loading=loading,
            idio_variance=idio_variance,
            regression_weights=regression_weights,
        )

    return _make_factor_case


def _fit_mu_uncertainty(
    factor_case: FactorCase, **kwargs
) -> OrthogonalMuUncertaintySet:
    model = OrthogonalMuUncertaintySet(**kwargs)
    model.fit(factor_case.X, return_distribution=factor_case.return_distribution)
    return model


def _fit_covariance_uncertainty(
    factor_case: FactorCase, **kwargs
) -> OrthogonalCovarianceUncertaintySet:
    model = OrthogonalCovarianceUncertaintySet(**kwargs)
    model.fit(factor_case.X, return_distribution=factor_case.return_distribution)
    return model


def _cs_weighting_metric(
    factor_case: FactorCase, cs_weighting: CSWeighting
) -> np.ndarray:
    n_assets = factor_case.loading.shape[0]
    if cs_weighting == CSWeighting.REGRESSION:
        return np.diag(factor_case.regression_weights[-1])
    if cs_weighting == CSWeighting.INVERSE_IDIO_VARIANCE:
        return np.diag(1.0 / factor_case.idio_variance)
    if cs_weighting == CSWeighting.IDENTITY:
        return np.eye(n_assets)
    raise ValueError("Unsupported weighting for this test.")


def _factor_aligned_portfolio(
    factor_case: FactorCase, cs_weighting: CSWeighting
) -> np.ndarray:
    if cs_weighting == CSWeighting.REGRESSION:
        weights = factor_case.regression_weights[-1]
    elif cs_weighting == CSWeighting.INVERSE_IDIO_VARIANCE:
        weights = 1.0 / factor_case.idio_variance
    elif cs_weighting == CSWeighting.IDENTITY:
        weights = np.ones_like(factor_case.idio_variance)
    else:
        raise ValueError("Unsupported weighting for this test.")

    portfolio = np.diag(weights) @ factor_case.loading[:, 0]
    return portfolio / np.linalg.norm(portfolio)


def _compact_inflation_residual(
    uncertainty_set: CompactCovarianceUncertaintySet, weights: np.ndarray
) -> np.ndarray:
    exposure = uncertainty_set.metric_sqrt * weights
    if uncertainty_set.basis.shape[1] == 0:
        return exposure
    return exposure - uncertainty_set.basis @ (uncertainty_set.basis.T @ exposure)


class TestOrthogonalMuUncertaintySet:
    def test_fit_returns_norm_ball_uncertainty_set(self, make_factor_case):
        factor_case = make_factor_case(n_assets=10, n_factors=3, seed=1)

        model = _fit_mu_uncertainty(factor_case)
        uncertainty_set = model.uncertainty_set_

        assert isinstance(uncertainty_set, UncertaintySet)
        assert uncertainty_set.radius > 0
        assert uncertainty_set.norm == 2
        assert uncertainty_set.geometry.shape == (10, 7)

    @pytest.mark.parametrize(
        "cs_weighting",
        [
            CSWeighting.INVERSE_IDIO_VARIANCE,
            CSWeighting.REGRESSION,
            CSWeighting.IDENTITY,
        ],
    )
    def test_geometry_is_orthogonal_to_factor_span(
        self, make_factor_case, cs_weighting
    ):
        factor_case = make_factor_case(
            with_regression_weights=cs_weighting == CSWeighting.REGRESSION,
            seed=2,
        )

        model = _fit_mu_uncertainty(factor_case, cs_weighting=cs_weighting)
        metric = _cs_weighting_metric(factor_case, cs_weighting)

        cross = model.uncertainty_set_.geometry.T @ metric @ factor_case.loading
        np.testing.assert_allclose(cross, 0.0, atol=1e-10)

    @pytest.mark.parametrize(
        "cs_weighting",
        [
            CSWeighting.INVERSE_IDIO_VARIANCE,
            CSWeighting.REGRESSION,
            CSWeighting.IDENTITY,
        ],
    )
    def test_factor_aligned_portfolio_has_zero_mu_penalty(
        self, make_factor_case, cs_weighting
    ):
        factor_case = make_factor_case(
            with_regression_weights=cs_weighting == CSWeighting.REGRESSION,
            seed=3,
        )

        model = _fit_mu_uncertainty(factor_case, cs_weighting=cs_weighting)
        weights = _factor_aligned_portfolio(factor_case, cs_weighting)
        penalty = np.linalg.norm(model.uncertainty_set_.geometry.T @ weights)

        np.testing.assert_allclose(penalty, 0.0, atol=1e-10)

    def test_uncertainty_shape_idio_variance_changes_geometry(self, make_factor_case):
        factor_case = make_factor_case(seed=4)

        identity = _fit_mu_uncertainty(
            factor_case,
            uncertainty_shape="identity",
        )
        idio_variance = _fit_mu_uncertainty(
            factor_case,
            uncertainty_shape="idio_variance",
        )

        assert not np.allclose(
            identity.uncertainty_set_.geometry,
            idio_variance.uncertainty_set_.geometry,
        )

    def test_regression_weighting_changes_geometry(self, make_factor_case):
        factor_case = make_factor_case(with_regression_weights=True, seed=5)

        inverse_idio = _fit_mu_uncertainty(
            factor_case,
            cs_weighting=CSWeighting.INVERSE_IDIO_VARIANCE,
        )
        regression = _fit_mu_uncertainty(
            factor_case,
            cs_weighting=CSWeighting.REGRESSION,
        )

        assert not np.allclose(
            inverse_idio.uncertainty_set_.geometry,
            regression.uncertainty_set_.geometry,
        )

    def test_full_rank_factor_model_has_zero_radius(self, make_factor_case):
        factor_case = make_factor_case(n_assets=5, n_factors=5, seed=6)

        model = _fit_mu_uncertainty(factor_case)

        assert model.uncertainty_set_.radius == 0.0
        assert model.uncertainty_set_.geometry.shape == (5, 1)

    def test_radius_increases_with_confidence_level(self, make_factor_case):
        factor_case = make_factor_case(seed=7)

        low = _fit_mu_uncertainty(factor_case, confidence_level=0.5)
        high = _fit_mu_uncertainty(factor_case, confidence_level=0.99)

        assert high.uncertainty_set_.radius > low.uncertainty_set_.radius

    def test_radius_matches_chi_square_quantile(self, make_factor_case):
        confidence_level = 0.9
        factor_case = make_factor_case(n_assets=11, n_factors=4, seed=18)

        model = _fit_mu_uncertainty(factor_case, confidence_level=confidence_level)
        rank = model.uncertainty_set_.geometry.shape[1]
        assert rank > 0

        expected = np.sqrt(scipy_stats.chi2.ppf(q=confidence_level, df=rank))
        np.testing.assert_allclose(
            model.uncertainty_set_.radius, expected, rtol=1e-12, atol=1e-12
        )

    def test_invalid_uncertainty_shape_raises(self, make_factor_case):
        factor_case = make_factor_case(seed=8)
        model = OrthogonalMuUncertaintySet(uncertainty_shape="bogus")

        with pytest.raises(ValueError, match="uncertainty_shape"):
            model.fit(
                factor_case.X, return_distribution=factor_case.return_distribution
            )

    def test_invalid_cs_weighting_raises(self, make_factor_case):
        factor_case = make_factor_case(seed=9)
        model = OrthogonalMuUncertaintySet(cs_weighting="inverse_idio_variance")

        with pytest.raises(TypeError, match="CSWeighting"):
            model.fit(
                factor_case.X, return_distribution=factor_case.return_distribution
            )

    def test_no_return_distribution_raises(self):
        model = OrthogonalMuUncertaintySet()

        with pytest.raises(ValueError, match="requires `return_distribution`"):
            model.fit(np.zeros((5, 3)))

    def test_no_factor_model_raises(self):
        returns = np.zeros((5, 3))
        return_distribution = ReturnDistribution(
            mu=np.zeros(3),
            covariance=np.eye(3),
            returns=returns,
        )
        model = OrthogonalMuUncertaintySet()

        with pytest.raises(ValueError, match="requires a factor model"):
            model.fit(returns, return_distribution=return_distribution)


class TestOrthogonalCovarianceUncertaintySet:
    def test_fit_returns_compact_covariance_inflation(self, make_factor_case):
        factor_case = make_factor_case(n_assets=9, n_factors=3, seed=10)

        model = _fit_covariance_uncertainty(factor_case, radius=2.0)
        uncertainty_set = model.uncertainty_set_

        assert isinstance(uncertainty_set, CompactCovarianceUncertaintySet)
        assert uncertainty_set.radius == 2.0
        assert uncertainty_set.metric_sqrt.shape == (9,)
        assert uncertainty_set.basis.shape == (9, 3)

    def test_radius_scales_compact_penalty(self, make_factor_case):
        factor_case = make_factor_case(seed=11)
        weights = np.ones(factor_case.loading.shape[0])

        small = _fit_covariance_uncertainty(factor_case, radius=1.0).uncertainty_set_
        large = _fit_covariance_uncertainty(factor_case, radius=3.0).uncertainty_set_
        small_penalty = small.radius * np.sum(
            _compact_inflation_residual(small, weights) ** 2
        )
        large_penalty = large.radius * np.sum(
            _compact_inflation_residual(large, weights) ** 2
        )

        np.testing.assert_allclose(large_penalty, 3.0 * small_penalty)

    @pytest.mark.parametrize(
        "cs_weighting",
        [
            CSWeighting.INVERSE_IDIO_VARIANCE,
            CSWeighting.REGRESSION,
            CSWeighting.IDENTITY,
        ],
    )
    def test_factor_aligned_portfolio_has_zero_compact_penalty(
        self, make_factor_case, cs_weighting
    ):
        factor_case = make_factor_case(
            with_regression_weights=cs_weighting == CSWeighting.REGRESSION,
            seed=12,
        )

        model = _fit_covariance_uncertainty(factor_case, cs_weighting=cs_weighting)
        weights = _factor_aligned_portfolio(factor_case, cs_weighting)
        residual = _compact_inflation_residual(model.uncertainty_set_, weights)

        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_residual_portfolio_has_positive_compact_penalty(self, make_factor_case):
        factor_case = make_factor_case(seed=13)

        model = _fit_covariance_uncertainty(factor_case)
        weights = np.eye(factor_case.loading.shape[0])[0]
        residual = _compact_inflation_residual(model.uncertainty_set_, weights)

        assert np.sum(residual**2) > 1e-8

    def test_regression_weighting_changes_compact_inflation(self, make_factor_case):
        factor_case = make_factor_case(with_regression_weights=True, seed=14)

        inverse_idio = _fit_covariance_uncertainty(
            factor_case,
            cs_weighting=CSWeighting.INVERSE_IDIO_VARIANCE,
        )
        regression = _fit_covariance_uncertainty(
            factor_case,
            cs_weighting=CSWeighting.REGRESSION,
        )

        assert not np.allclose(
            inverse_idio.uncertainty_set_.metric_sqrt,
            regression.uncertainty_set_.metric_sqrt,
        )

    def test_missing_regression_weights_raise(self, make_factor_case):
        factor_case = make_factor_case(seed=15)
        model = OrthogonalCovarianceUncertaintySet(cs_weighting=CSWeighting.REGRESSION)

        with pytest.raises(ValueError, match="regression_weights"):
            model.fit(
                factor_case.X, return_distribution=factor_case.return_distribution
            )

    def test_invalid_radius_raises(self):
        model = OrthogonalCovarianceUncertaintySet(radius=-1.0)

        with pytest.raises(ValueError, match="radius"):
            model.fit(np.zeros((5, 3)), return_distribution=None)

    def test_invalid_cs_weighting_raises(self, make_factor_case):
        factor_case = make_factor_case(seed=16)
        model = OrthogonalCovarianceUncertaintySet(cs_weighting="inverse_idio_variance")

        with pytest.raises(TypeError, match="CSWeighting"):
            model.fit(
                factor_case.X, return_distribution=factor_case.return_distribution
            )

    def test_no_return_distribution_raises(self):
        model = OrthogonalCovarianceUncertaintySet()

        with pytest.raises(ValueError, match="requires `return_distribution`"):
            model.fit(np.zeros((5, 3)))

    def test_no_factor_model_raises(self):
        returns = np.zeros((5, 3))
        return_distribution = ReturnDistribution(
            mu=np.zeros(3),
            covariance=np.eye(3),
            returns=returns,
        )
        model = OrthogonalCovarianceUncertaintySet()

        with pytest.raises(ValueError, match="requires a factor model"):
            model.fit(returns, return_distribution=return_distribution)


class TestMeanRiskIntegration:
    def test_mu_uncertainty_with_factor_model(self, X, y):
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=TimeSeriesFactorModel(),
            mu_uncertainty_set_estimator=OrthogonalMuUncertaintySet(),
        )

        model.fit(X, y)

        assert model.weights_.shape == (X.shape[1],)
        assert np.isfinite(model.weights_).all()
        np.testing.assert_almost_equal(np.sum(model.weights_), 1.0)

    def test_maximize_return_orthogonal_mu_changes_weights(self, make_factor_case):
        factor_case = make_factor_case(n_assets=10, n_factors=3, n_obs=200, seed=101)
        n_obs, n_assets = factor_case.X.shape
        n_factors = factor_case.return_distribution.factor_model.factor_returns.shape[1]
        idx = pd.RangeIndex(n_obs)
        X = pd.DataFrame(
            factor_case.X,
            index=idx,
            columns=[f"asset_{i}" for i in range(n_assets)],
        )
        y = pd.DataFrame(
            factor_case.return_distribution.factor_model.factor_returns,
            index=idx,
            columns=[f"factor_{i}" for i in range(n_factors)],
        )

        common = dict(
            objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=TimeSeriesFactorModel(),
        )
        baseline = MeanRisk(**common)
        baseline.fit(X, y)

        robust = MeanRisk(
            **common,
            mu_uncertainty_set_estimator=OrthogonalMuUncertaintySet(
                confidence_level=0.99,
            ),
        )
        robust.fit(X, y)

        assert robust.mu_uncertainty_set_estimator_ is not None
        u = robust.mu_uncertainty_set_estimator_.uncertainty_set_
        assert isinstance(u, UncertaintySet)
        assert u.radius > 0
        assert "mu_uncertainty_set" in robust.problem_values_
        assert np.isfinite(robust.problem_values_["mu_uncertainty_set"])
        assert robust.problem_values_["mu_uncertainty_set"] >= 0.0

        assert not np.allclose(baseline.weights_, robust.weights_, atol=1e-4)

    def test_covariance_uncertainty_with_factor_model(self, X, y):
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=TimeSeriesFactorModel(),
            covariance_uncertainty_set_estimator=OrthogonalCovarianceUncertaintySet(
                radius=0.5
            ),
        )

        model.fit(X, y)

        assert model.weights_.shape == (X.shape[1],)
        assert np.isfinite(model.weights_).all()
        np.testing.assert_almost_equal(np.sum(model.weights_), 1.0)

    def test_mu_and_covariance_uncertainty_together(self, X, y):
        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=TimeSeriesFactorModel(),
            mu_uncertainty_set_estimator=OrthogonalMuUncertaintySet(
                confidence_level=0.9
            ),
            covariance_uncertainty_set_estimator=OrthogonalCovarianceUncertaintySet(
                radius=1.0
            ),
        )

        model.fit(X, y)

        assert model.weights_.shape == (X.shape[1],)
        assert np.isfinite(model.weights_).all()
        np.testing.assert_almost_equal(np.sum(model.weights_), 1.0)
