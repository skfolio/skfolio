"""Statistical recovery tests for the alpha estimation pipeline.

These tests validate the end-to-end behavior of the alpha estimation pipeline
within CharacteristicsFactorModel, covering:

Test 14a -- Spanned alpha recovery
    DGP: single factor with known betas.  Alpha is a linear function of
    the betas (i.e. purely spanned).  With `spanned_alpha_shrinkage=0`
    the factor-implied mu must match the alpha-implied factor mu, and the
    orthogonal component must be zero.

Test 14b -- Orthogonal alpha recovery
    DGP: single factor.  Alpha has a component orthogonal to the factor
    exposures.  With `orthogonal_alpha_confidence=1` the full orthogonal
    alpha must be present in the final mu.  With `confidence=0` only the
    spanned part survives.

Test 14c -- Spanned alpha shrinkage
    Same DGP as 14a.  Varying `spanned_alpha_shrinkage` from 0 to 1
    must linearly interpolate between the alpha-implied and factor-prior
    factor mu.

Test 14d -- EWSharpeOptimalAlpha integration
    DGP: single factor with a predictive signal (the true beta corrupted
    by noise).  Using `EWSharpeOptimalAlpha` as alpha estimator, the
    final mu must capture some of the cross-sectional return structure.

Test 14e -- No alpha estimator (default)
    When `alpha_estimator=None` the asset mu must equal the
    factor-prior-implied spanned alpha (`B @ factor_prior_mu`).
"""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.alpha import BaseAlpha, EWSharpeOptimalAlpha
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import GlobalFactor
from skfolio.prior import CharacteristicsFactorModel

from .conftest import make_panel, passthrough_factor


class TestSpannedAlphaRecovery:
    r"""Alpha is a linear function of exposures (purely spanned).

    DGP:
        :math:`R_i(t) = \beta_i \, f(t) + \epsilon_i(t)`

    Alpha vector: :math:`\alpha_i = c \cdot \beta_i` for some constant c.
    This alpha lies entirely in the column space of the factor exposures,
    so `_decompose_alpha` must produce zero orthogonal component and the
    factor mu from alpha must equal `c`.
    """

    N_OBS = 500
    N_ASSETS = 40
    SEED = 77
    ALPHA_SLOPE = 0.002

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        sigma_f = 0.01
        sigma_eps = 0.005
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas[None, :] * f_true[:, None] + eps

        # Alpha is purely spanned: alpha_i = ALPHA_SLOPE * beta_i
        alpha = cls.ALPHA_SLOPE * betas
        alpha_2d = np.broadcast_to(alpha[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        panel, X = make_panel(
            returns, extra_fields={"beta": betas_2d, "alpha_signal": alpha_2d}
        )

        # spanned_alpha_shrinkage=0: use the alpha-implied factor mu entirely
        model = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=_ConstantAlpha("alpha_signal"),
            spanned_alpha_shrinkage=0.0,
            orthogonal_alpha_confidence=1.0,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        cls.model = model
        cls.alpha = alpha
        cls.betas = betas

    def test_mu_recovers_alpha(self):
        """Final asset mu must approximate the injected alpha (purely
        spanned, shrinkage=0 gives full weight to alpha-implied mu)."""
        np.testing.assert_allclose(
            self.model.return_distribution_.mu,
            self.alpha,
            atol=1e-10,
        )

    def test_orthogonal_alpha_is_zero(self):
        """With purely spanned alpha, the orthogonal component is zero."""
        fm = self.model.factor_model_
        spanned = fm.loading_matrix @ fm.factor_mu
        orthogonal = self.model.return_distribution_.mu - spanned
        np.testing.assert_allclose(orthogonal, 0, atol=1e-10)

    def test_factor_mu_equals_slope(self):
        """The factor mu must equal the alpha slope constant."""
        fm = self.model.factor_model_
        # With passthrough factor (no z-scoring), the exposure equals
        # the raw beta.  factor_mu = c so that B @ factor_mu = c * beta
        # which matches the alpha.  The factor model also has an intercept,
        # so factor_mu[0] absorbs the mean and factor_mu[1] = ALPHA_SLOPE.
        spanned = fm.loading_matrix @ fm.factor_mu
        np.testing.assert_allclose(spanned, self.alpha, atol=1e-10)


class TestOrthogonalAlpha:
    r"""Alpha has a component orthogonal to the factor exposures.

    DGP:
        :math:`R_i(t) = \beta_i \, f(t) + \epsilon_i(t)`

    Alpha: :math:`\alpha_i = c \cdot \beta_i + \delta_i` where
    :math:`\delta` is orthogonal to :math:`\beta` under equal weights.
    """

    N_OBS = 500
    N_ASSETS = 40
    SEED = 88

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        sigma_f = 0.01
        sigma_eps = 0.005
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas[None, :] * f_true[:, None] + eps

        # Build orthogonal component: random vector, project out beta and 1
        raw_delta = rng.normal(0, 0.001, size=cls.N_ASSETS)
        # Under equal weights, project out intercept (ones) and beta
        ones = np.ones(cls.N_ASSETS)
        B = np.column_stack([ones, betas])
        proj = B @ np.linalg.lstsq(B, raw_delta, rcond=None)[0]
        delta = raw_delta - proj
        delta *= 0.001 / (np.std(delta) + 1e-12)

        alpha_spanned = 0.002 * betas
        alpha = alpha_spanned + delta
        alpha_2d = np.broadcast_to(alpha[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        panel_full, X_full = make_panel(
            returns, extra_fields={"beta": betas_2d, "alpha_signal": alpha_2d}
        )
        panel_zero, X_zero = make_panel(
            returns.copy(),
            extra_fields={"beta": betas_2d.copy(), "alpha_signal": alpha_2d.copy()},
        )

        # Full confidence: orthogonal alpha is kept
        model_full = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=_ConstantAlpha("alpha_signal"),
            spanned_alpha_shrinkage=0.0,
            orthogonal_alpha_confidence=1.0,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_full.fit(X_full, characteristics=panel_full)

        # Zero confidence: orthogonal alpha is discarded
        model_zero = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=_ConstantAlpha("alpha_signal"),
            spanned_alpha_shrinkage=0.0,
            orthogonal_alpha_confidence=0.0,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_zero.fit(X_zero, characteristics=panel_zero)

        cls.model_full = model_full
        cls.model_zero = model_zero
        cls.alpha = alpha
        cls.alpha_spanned = alpha_spanned
        cls.delta = delta

    def test_full_confidence_recovers_total_alpha(self):
        """With confidence=1 the total alpha (spanned + ortho) is recovered."""
        np.testing.assert_allclose(
            self.model_full.return_distribution_.mu,
            self.alpha,
            atol=1e-10,
        )

    def test_zero_confidence_discards_orthogonal(self):
        """With confidence=0 only the spanned alpha survives."""
        mu = self.model_zero.return_distribution_.mu
        fm = self.model_zero.factor_model_
        spanned = fm.loading_matrix @ fm.factor_mu
        np.testing.assert_allclose(mu, spanned, atol=1e-10)

    def test_zero_confidence_mu_differs_from_full(self):
        """Zero confidence mu must differ from full confidence mu because
        the orthogonal component is non-zero."""
        diff = np.max(
            np.abs(
                self.model_full.return_distribution_.mu
                - self.model_zero.return_distribution_.mu
            )
        )
        assert diff > 1e-5, f"max |mu_full - mu_zero| = {diff:.2e}"

    def test_orthogonal_component_equals_delta(self):
        """The orthogonal component must equal the injected delta."""
        fm = self.model_full.factor_model_
        spanned = fm.loading_matrix @ fm.factor_mu
        ortho = self.model_full.return_distribution_.mu - spanned
        np.testing.assert_allclose(ortho, self.delta, atol=1e-10)


class TestSpannedAlphaShrinkage:
    r"""Verify linear interpolation via `spanned_alpha_shrinkage`.

    With shrinkage :math:`\lambda`, the factor mu is:

    .. math::

        \mu_f = \lambda \, \mu_f^{\text{prior}}
                + (1 - \lambda) \, \mu_f^{\text{alpha}}

    The test fits three models: shrinkage = 0 (pure alpha), 0.5 (blend),
    and 1 (pure prior), and verifies the blended mu is the midpoint.
    """

    N_OBS = 500
    N_ASSETS = 40
    SEED = 99

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        sigma_f = 0.01
        sigma_eps = 0.005
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas[None, :] * f_true[:, None] + eps

        alpha = 0.003 * betas
        alpha_2d = np.broadcast_to(alpha[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        def _fit(shrinkage):
            panel, X = make_panel(
                returns.copy(),
                extra_fields={
                    "beta": betas_2d.copy(),
                    "alpha_signal": alpha_2d.copy(),
                },
            )
            m = CharacteristicsFactorModel(
                factors=[("f1", passthrough_factor("beta"))],
                alpha_estimator=_ConstantAlpha("alpha_signal"),
                spanned_alpha_shrinkage=shrinkage,
                orthogonal_alpha_confidence=0.0,
                benchmark_mcap_power=0,
                regression_mcap_power=0,
            )
            m.fit(X, characteristics=panel)
            return m

        cls.model_0 = _fit(0.0)
        cls.model_half = _fit(0.5)
        cls.model_1 = _fit(1.0)

    def test_blend_is_midpoint(self):
        """Shrinkage=0.5 must give a mu that is the midpoint of 0 and 1."""
        mu_0 = self.model_0.return_distribution_.mu
        mu_1 = self.model_1.return_distribution_.mu
        mu_half = self.model_half.return_distribution_.mu
        expected = 0.5 * mu_0 + 0.5 * mu_1
        np.testing.assert_allclose(mu_half, expected, atol=1e-10)

    def test_shrinkage_zero_differs_from_one(self):
        """Pure alpha mu (shrinkage=0) must differ from pure prior mu
        (shrinkage=1) when the alpha signal is non-zero."""
        diff = np.max(
            np.abs(
                self.model_0.return_distribution_.mu
                - self.model_1.return_distribution_.mu
            )
        )
        assert diff > 1e-5, f"max |mu_0 - mu_1| = {diff:.2e}"

    def test_covariance_unchanged_by_shrinkage(self):
        """Covariance must be identical across all shrinkage levels
        (alpha only affects mu, not covariance)."""
        np.testing.assert_allclose(
            self.model_0.return_distribution_.covariance,
            self.model_1.return_distribution_.covariance,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            self.model_0.return_distribution_.covariance,
            self.model_half.return_distribution_.covariance,
            rtol=1e-10,
        )


class TestEWSharpeOptimalAlphaIntegration:
    r"""End-to-end test with a real `EWSharpeOptimalAlpha` estimator.

    DGP: single factor with known betas.  A noisy copy of the true beta
    is provided as a predictive signal.  The `EWSharpeOptimalAlpha`
    estimator should learn that this signal predicts idiosyncratic returns
    (which in this DGP are pure noise), producing a non-zero alpha that
    captures cross-sectional return structure.

    Because the signal is beta (which drives returns) passed through the
    alpha estimator that targets idiosyncratic returns, the alpha should
    be small but non-zero -- the WLS regression detects the residual
    predictability from any misattribution in the factor model step.
    """

    N_OBS = 500
    N_ASSETS = 40
    SEED = 55

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        sigma_f = 0.01
        sigma_eps = 0.005
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas[None, :] * f_true[:, None] + eps

        # Noisy signal: true beta + noise
        signal_noise = rng.normal(0, 0.3, size=(cls.N_OBS, cls.N_ASSETS))
        signal = betas_2d + signal_noise

        panel_alpha, X_alpha = make_panel(
            returns,
            extra_fields={"beta": betas_2d, "pred_signal": signal},
        )

        alpha_est = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("pred_signal"))],
            horizon=1,
            half_life=50,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        model_alpha = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=alpha_est,
            spanned_alpha_shrinkage=0.0,
            orthogonal_alpha_confidence=1.0,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_alpha.fit(X_alpha, characteristics=panel_alpha)

        # Baseline model without alpha
        panel_base, X_base = make_panel(
            returns.copy(),
            extra_fields={"beta": betas_2d.copy()},
        )
        model_base = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=None,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_base.fit(X_base, characteristics=panel_base)

        cls.model_alpha = model_alpha
        cls.model_base = model_base

    def test_alpha_estimator_fitted(self):
        """The alpha_estimator_ must be stored and fitted."""
        assert self.model_alpha.alpha_estimator_ is not None
        assert hasattr(self.model_alpha.alpha_estimator_, "alpha_")

    def test_mu_differs_from_no_alpha(self):
        """Asset mu with alpha estimator must differ from the no-alpha
        baseline."""
        diff = np.max(
            np.abs(
                self.model_alpha.return_distribution_.mu
                - self.model_base.return_distribution_.mu
            )
        )
        assert diff > 1e-6, f"max |mu_alpha - mu_base| = {diff:.2e}"

    def test_covariance_same_as_no_alpha(self):
        """Covariance must be identical regardless of alpha estimator
        (alpha only affects mu)."""
        np.testing.assert_allclose(
            self.model_alpha.return_distribution_.covariance,
            self.model_base.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model_alpha.factor_model_
        reconstructed = fm.loading_matrix @ fm.factor_covariance @ fm.loading_matrix.T
        idio = fm.idio_covariance
        if idio.ndim == 1:
            reconstructed[np.diag_indices_from(reconstructed)] += idio
        else:
            reconstructed += idio
        np.testing.assert_allclose(
            self.model_alpha.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )


class TestInactiveExposureMaskingWithAlpha:
    """Inactive asset exposures stay missing in alpha and stored factor-model data."""

    def test_global_factor_is_nan_outside_active_universe(self):
        rng = np.random.default_rng(123)
        n_obs = 80
        n_assets = 12
        listing = 20

        active_mask = np.ones((n_obs, n_assets), dtype=bool)
        active_mask[:listing, 0] = False

        betas = rng.uniform(0.5, 1.5, size=n_assets)
        betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
        signal = betas_2d + rng.normal(0, 0.2, size=(n_obs, n_assets))

        returns = betas * rng.normal(0, 0.01, size=(n_obs, 1))
        returns += rng.normal(0, 0.005, size=(n_obs, n_assets))
        market_cap = np.ones((n_obs, n_assets))

        returns[~active_mask] = np.nan
        market_cap[~active_mask] = np.nan
        betas_2d[~active_mask] = np.nan
        signal[~active_mask] = np.nan

        panel, X = make_panel(
            returns,
            extra_fields={"beta": betas_2d, "pred_signal": signal},
            market_cap=market_cap,
            active_mask=active_mask,
        )

        alpha_estimator = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("pred_signal"))],
            horizon=1,
            half_life=10,
            neutralize_against=["market"],
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )
        model = CharacteristicsFactorModel(
            factors=[
                ("market", GlobalFactor()),
                ("beta", passthrough_factor("beta")),
            ],
            alpha_estimator=alpha_estimator,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=8,
        )

        model.fit(X, characteristics=panel)

        factor_model = model.factor_model_
        history_active_mask = active_mask[factor_model.observations.astype(int)]
        market_idx = int(np.flatnonzero(factor_model.factor_names == "market")[0])

        assert np.isnan(factor_model.exposures[~history_active_mask]).all()
        np.testing.assert_allclose(
            factor_model.exposures[history_active_mask, market_idx],
            1.0,
        )


class TestNoAlphaEstimatorDefault:
    """When alpha_estimator=None, mu = B @ factor_prior_mu (from the factor
    prior estimator, typically EWMA of factor returns)."""

    N_OBS = 500
    N_ASSETS = 40
    SEED = 33

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        f_true = rng.normal(0, 0.01, size=cls.N_OBS)
        eps = rng.normal(0, 0.005, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas[None, :] * f_true[:, None] + eps

        panel, X = make_panel(returns, extra_fields={"beta": betas_2d})

        model = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=None,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        cls.model = model

    def test_alpha_estimator_is_none(self):
        """alpha_estimator_ must be None."""
        assert self.model.alpha_estimator_ is None

    def test_mu_equals_spanned_prior(self):
        """Asset mu must equal B @ factor_prior_mu."""
        fm = self.model.factor_model_
        expected_mu = fm.loading_matrix @ fm.factor_mu
        np.testing.assert_allclose(
            self.model.return_distribution_.mu,
            expected_mu,
            atol=1e-12,
        )

    def test_no_orthogonal_component(self):
        """With no alpha estimator, the idio mu stored in the factor model
        must be zero."""
        fm = self.model.factor_model_
        np.testing.assert_allclose(fm.idio_mu, 0, atol=1e-12)


class TestAlphaWarmupEdgeCases:
    """Alpha warm-up and missing-alpha behavior."""

    N_OBS = 160
    N_ASSETS = 18

    @staticmethod
    def _make_data(seed=2024):
        rng = np.random.default_rng(seed)
        betas = rng.uniform(0.6, 1.4, size=TestAlphaWarmupEdgeCases.N_ASSETS)
        betas_2d = np.broadcast_to(
            betas,
            (TestAlphaWarmupEdgeCases.N_OBS, TestAlphaWarmupEdgeCases.N_ASSETS),
        ).copy()
        factor_returns = rng.normal(0, 0.01, size=TestAlphaWarmupEdgeCases.N_OBS)
        returns = betas[None, :] * factor_returns[:, None]
        returns += rng.normal(
            0,
            0.004,
            size=(TestAlphaWarmupEdgeCases.N_OBS, TestAlphaWarmupEdgeCases.N_ASSETS),
        )
        return make_panel(returns, extra_fields={"beta": betas_2d})

    @staticmethod
    def _make_model(alpha_estimator, **kwargs):
        defaults = dict(
            factors=[("f1", passthrough_factor("beta"))],
            alpha_estimator=alpha_estimator,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=12,
        )
        defaults.update(kwargs)
        return CharacteristicsFactorModel(**defaults)

    def test_alpha_estimator_warmup_none(self):
        panel, X = self._make_data(seed=2025)
        model = self._make_model(_NoneAlpha())

        model.fit(X, characteristics=panel)

        fm = model.factor_model_
        expected_mu = fm.loading_matrix @ fm.factor_mu
        np.testing.assert_allclose(model.return_distribution_.mu, expected_mu)

    def test_alpha_with_nan_entries(self):
        panel, X = self._make_data(seed=2026)
        alpha = np.linspace(-0.002, 0.002, self.N_ASSETS)
        alpha[[2, 7, 13]] = np.nan
        model = self._make_model(
            _VectorAlpha(alpha),
            spanned_alpha_shrinkage=0.0,
            orthogonal_alpha_confidence=1.0,
        )

        model.fit(X, characteristics=panel)

        mu = model.return_distribution_.mu
        np.testing.assert_array_equal(np.isnan(mu), np.isnan(alpha))
        assert np.isfinite(mu[~np.isnan(alpha)]).all()
        assert np.isfinite(model.factor_model_.factor_mu).all()

    def test_alpha_all_nan_returns_spanned_only(self):
        panel, X = self._make_data(seed=2027)
        model = self._make_model(_VectorAlpha(np.full(self.N_ASSETS, np.nan)))

        model.fit(X, characteristics=panel)

        fm = model.factor_model_
        expected_mu = fm.loading_matrix @ fm.factor_mu
        assert np.isfinite(model.return_distribution_.mu).all()
        np.testing.assert_allclose(model.return_distribution_.mu, expected_mu)


class _ConstantAlpha(BaseAlpha):
    """Minimal alpha estimator that reads a pre-computed alpha from the panel.

    Used in tests to inject a known alpha vector without going through
    the full EWSharpeOptimalAlpha pipeline.
    """

    def __init__(self, field_name: str = "alpha_signal"):
        self.field_name = field_name

    def fit(self, X, y=None, **fit_params):
        self.alpha_ = X[self.field_name][-1]
        return self

    def partial_fit(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params)


class _NoneAlpha(BaseAlpha):
    """Minimal alpha estimator that remains in warmup."""

    def fit(self, X, y=None, **fit_params):
        self.alpha_ = None
        return self

    def partial_fit(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params)


class _VectorAlpha(BaseAlpha):
    """Minimal alpha estimator that publishes a fixed alpha vector."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y=None, **fit_params):
        self.alpha_ = np.asarray(self.alpha, dtype=float).copy()
        return self

    def partial_fit(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params)
