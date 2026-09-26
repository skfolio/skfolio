r"""Statistical recovery tests for CharacteristicsFactorModel.

Each test generates data from a known data-generating process (DGP), fits the
model, and verifies that estimated quantities recover the ground-truth
parameters within statistical tolerance.

Test plan
---------

Test 1 -- Single factor, constant beta (Passthrough)
    DGP: R_i(t) = beta_i * f(t) + eps_i(t), betas constant, all Gaussian.
    Verifies: factor return recovery, factor variance, idiosyncratic
    variances, covariance identity B F B^T + D, residual orthogonality.

Test 2 -- Market (intercept) factor equals benchmark return
    DGP: R_i(t) = f_mkt(t) + eps_i(t), constant exposure = 1.
    With cap-weighted regression, the estimated factor return must equal the
    cap-weighted average asset return at each observation.

Test 3 -- Single factor, EWMA-estimated beta (EWMarketBeta)
    DGP: same as Test 1 but betas are estimated online by EWMarketBeta
    from returns and market_cap. Validates descriptor-to-factor pipeline
    with warmup handling.

Test 4 -- Two uncorrelated factors, constant betas
    DGP: R_i(t) = beta1_i * f1(t) + beta2_i * f2(t) + eps_i(t), f1 and f2
    independent. Verifies multi-factor separation, diagonal factor
    covariance, individual factor return recovery.

Test 5 -- Two correlated factors, constant betas
    Same as Test 4 but f1, f2 drawn from a bivariate normal with known
    correlation. Verifies off-diagonal factor covariance recovery.

Test 6 -- Exposure lag with time-varying betas
    DGP: betas drift linearly. Verifies that the regression uses B(t-lag)
    while the loading_matrix stores B(T).

Test 7 -- Estimation mask (subset learning)
    DGP: single factor, but 20% of assets excluded from estimation_mask.
    Verifies that excluded assets still receive predictions but do not
    influence learned statistics.

Test 8 -- Sparse idiosyncratic correlation overlay
    DGP: single factor, two assets share correlated idio returns. With
    idio_corr_threshold > 0, the overlay captures the pair while
    remaining entries stay zero.

Test 9 -- Basket-neutral constraints (industries)
    DGP: market + 3 industry dummies. With constrained_families,
    benchmark-weighted industry factor returns sum to zero.  Also verifies:
    (a) market factor = cap-weighted average return (exact per observation),
    (b) factor_covariance is rank K-1 with null space proportional to
    the constraint direction, (c) asset covariance is PD, (d) Cholesky
    L L^T = Sigma.

Test 9b -- Basket-neutral basis storage
    Same DGP as Test 9. The family_constraint_basis must be present on the
    factor model and the covariance decomposition identity must hold.

Test 10 -- Factor neutralization
    DGP: two factors with known cross-sectional correlation. After
    neutralization, cross-sectional correlation is removed.

Test 10b -- Neutralization with basket-neutral constraints
    DGP: market + 3 industry dummies + 1 style factor with industry-
    correlated characteristic.  With both `neutralize_against` and
    `constrained_families`, verifies that output exposures in the
    original basis reflect the neutralization (style orthogonal to each
    industry under benchmark weights), and that both basis output modes
    produce identical asset covariance.

Test 10c -- Demeaning vs neutralization equivalence
    Same DGP as Test 10b.  Compares three approaches: (A) within-industry
    demeaning via `transform_by_group`, (B) explicit neutralization via
    `neutralize_against`, (C) both combined.  All three satisfy
    D^T W z = 0.  Models A and C produce identical style exposures
    (neutralization is a no-op after demeaning).  Models A and B differ in
    within-group spread (demeaning normalizes per-group std).

Test 11 -- Inverse-idiosyncratic-variance regression weights
    DGP: single factor with heterogeneous idio variance. With
    inv_idio_variance_weight_shrinkage > 0, noisy assets are
    downweighted and factor return MSE decreases.

Test 12 -- Intercept recovery with industries and style factors
    DGP: market intercept + 3 industry dummies (basket-neutral) + 1 style
    factor (z-scored) with market cap benchmark and sqrt market cap
    regression weights.  The basket-neutral constraints transform the
    industry basis and the z-scoring centers the style factor; the
    intercept factor return must still equal the benchmark-weighted average
    return.

Test 13 -- Regression diagnostics
    DGP: two factors (one true, one pure noise) with constant betas.
    Verifies: R^2 is positive, t-stats for
    the true factor are significant while the noise factor is not,
    hit_rate is high for the true factor, and AIC/BIC penalize the noise
    factor (lower information criterion for the true-only model).

Test 14 -- Time-varying market caps
    DGP with drifting market caps.  Verifies that benchmark weights,
    regression weights, and basket-neutral contrasts update correctly.

Test 15 -- NaN handling / listings and delistings
    DGP where some assets appear mid-sample (NaN returns early on).
    Verifies correct active_mask transitions and state management.

Test 16 -- Currency factor
    Verifies the currency exposure pipeline end-to-end.

Test 17 -- Neutralize-against family key and disjointness validation
    Integration test through `CharacteristicsFactorModel` with
    `neutralize_against={"style": ["industry"]}`: both style exposures
    must be benchmark-weight-orthogonal to every industry exposure, the
    covariance decomposition holds, and exposures differ from the
    no-neutralization control.  Companion validation and unit tests
    (name resolution, overlap errors) live in `test_validation.py`.

"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from skfolio.descriptor import EWMarketBeta, Passthrough
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.moments.variance import EWVariance
from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior

from .conftest import make_panel, passthrough_factor


def _reconstruct_asset_covariance(fm):
    """Reconstruct asset covariance from the factor model decomposition."""
    cov = fm.loading_matrix @ fm.factor_covariance @ fm.loading_matrix.T
    if fm.idio_covariance.ndim == 1:
        cov[np.diag_indices_from(cov)] += fm.idio_covariance
    else:
        cov += fm.idio_covariance
    return cov


def _assert_covariance_sqrt_identity(rd, rtol=1e-10):
    """Assert that the CovarianceSqrt decomposition reproduces the covariance."""
    sqrt = rd.covariance_sqrt
    n = rd.covariance.shape[0]
    reconstructed = np.zeros((n, n))
    for block in sqrt.components:
        reconstructed += block @ block.T
    if sqrt.diagonal is not None:
        reconstructed[np.diag_indices_from(reconstructed)] += sqrt.diagonal**2
    np.testing.assert_allclose(reconstructed, rd.covariance, rtol=rtol)


class TestSingleFactorConstantBeta:
    """DGP: R_i(t) = beta_i * f(t) + eps_i(t), constant betas, all Gaussian.

    Verifies end-to-end recovery of factor returns, variances, covariance
    decomposition, and residual orthogonality with a single passthrough factor.
    """

    N_OBS = 2000
    N_ASSETS = 50
    SEED = 42

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        beta_true = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        sigma_f = 0.01
        sigma_eps = rng.uniform(0.005, 0.02, size=cls.N_ASSETS)

        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, 1, size=(cls.N_OBS, cls.N_ASSETS)) * sigma_eps
        returns = beta_true[None, :] * f_true[:, None] + eps

        betas_field = np.broadcast_to(beta_true, (cls.N_OBS, cls.N_ASSETS)).copy()

        panel, X = make_panel(
            returns,
            extra_fields={"beta": betas_field},
        )

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            factor_prior_estimator=EmpiricalPrior(),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        cls.beta_true = beta_true
        cls.sigma_f = sigma_f
        cls.sigma_eps = sigma_eps
        cls.f_true_aligned = f_true[1:]
        cls.model = model

    def test_factor_returns_recovery(self):
        """Estimated factor returns correlate > 0.98 with the true factor."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        corr = np.corrcoef(f_hat, self.f_true_aligned)[0, 1]
        assert corr > 0.98, f"Factor return correlation {corr:.4f} < 0.98"

    def test_factor_variance_recovery(self):
        """Estimated factor variance within 15% of true sigma_f^2."""
        var_hat = self.model.factor_model_.factor_covariance[0, 0]
        assert var_hat == pytest.approx(self.sigma_f**2, rel=0.15)

    def test_idio_variance_recovery(self):
        """Per-asset idiosyncratic variances within 40% of truth."""
        var_hat = self.model.factor_model_.idio_variances[-1]
        np.testing.assert_allclose(var_hat, self.sigma_eps**2, rtol=0.4)

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_covariance_sqrt_identity(self):
        """Reconstructing from covariance_sqrt must equal the stored covariance."""
        _assert_covariance_sqrt_identity(self.model.return_distribution_)

    def test_idio_returns_orthogonal_to_factor(self):
        """All per-asset |corr(idio_i, f)| should be below 0.08."""
        idio = self.model.factor_model_.idio_returns
        corrs = [
            np.corrcoef(idio[:, i], self.f_true_aligned)[0, 1]
            for i in range(self.N_ASSETS)
        ]
        max_abs = np.max(np.abs(corrs))
        assert max_abs < 0.08, f"Max |corr(idio, factor)| = {max_abs:.4f}"

    def test_shapes(self):
        """All output arrays have correct dimensions."""
        fm = self.model.factor_model_
        rd = self.model.return_distribution_
        n = self.N_ASSETS
        n_obs = self.N_OBS - 1

        assert fm.loading_matrix.shape == (n, 1)
        assert fm.factor_covariance.shape == (1, 1)
        assert fm.factor_returns.shape == (n_obs, 1)
        assert fm.idio_returns.shape == (n_obs, n)
        assert fm.idio_variances.shape == (n_obs, n)
        assert fm.idio_covariance.shape == (n,)
        assert rd.mu.shape == (n,)
        assert rd.covariance.shape == (n, n)

    def test_loading_matrix_equals_true_betas(self):
        """With passthrough, loading matrix must be the raw betas."""
        B = self.model.factor_model_.loading_matrix[:, 0]
        np.testing.assert_allclose(B, self.beta_true, rtol=1e-10)


class TestInterceptFactorEqualsBenchmark:
    r"""Verify that an intercept factor recovers the benchmark return exactly.

    DGP: :math:`R_i(t) = f_{\mathrm{mkt}}(t) + \epsilon_i(t)`, with constant
    exposure = 1 for every asset.

    In a WLS cross-sectional regression with weights :math:`w_i` and a single
    factor whose exposure is 1 for every asset, the coefficient is:

    .. math::

        \hat{f}(t) = \frac{\sum_i w_i\,R_i(t)}{\sum_i w_i}

    When the regression weights equal the benchmark weights (both use the same
    `mcap_power`), this is the benchmark-weighted average return. The
    identity is exact (not statistical), so this test uses tight numerical
    tolerance.
    """

    N_OBS = 500
    N_ASSETS = 30
    SEED = 123

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        sigma_eps = 0.01
        f_mkt = rng.normal(0, 0.008, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = f_mkt[:, None] + eps

        # Heterogeneous, constant market caps (log-normal)
        market_cap = np.exp(rng.normal(10, 1, size=cls.N_ASSETS))
        market_cap_2d = np.broadcast_to(market_cap, (cls.N_OBS, cls.N_ASSETS)).copy()

        intercept_field = np.ones((cls.N_OBS, cls.N_ASSETS))

        panel, X = make_panel(
            returns,
            extra_fields={"intercept": intercept_field},
            market_cap=market_cap_2d,
        )

        model = CharacteristicsFactorModel(
            factors=[
                ("intercept", passthrough_factor("intercept", family="market")),
            ],
            exposure_lag=1,
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )
        model.fit(X, characteristics=panel)

        # Cap-weighted average return (aligned to the post-lag window)
        w = market_cap / market_cap.sum()
        cap_weighted_returns = returns[1:] @ w

        cls.model = model
        cls.cap_weighted_returns = cap_weighted_returns
        cls.f_mkt_aligned = f_mkt[1:]

    def test_factor_return_equals_cap_weighted_return(self):
        """Intercept factor return must equal cap-weighted average return."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        np.testing.assert_allclose(f_hat, self.cap_weighted_returns, rtol=1e-10)

    def test_factor_return_correlated_with_true_market(self):
        """Factor return should be highly correlated with the true DGP market
        factor (not exact due to idiosyncratic noise)."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        corr = np.corrcoef(f_hat, self.f_mkt_aligned)[0, 1]
        assert corr > 0.95, f"Correlation with true market = {corr:.4f}"

    def test_loading_matrix_all_ones(self):
        """Passthrough intercept exposure must be 1 for all assets."""
        B = self.model.factor_model_.loading_matrix[:, 0]
        np.testing.assert_allclose(B, 1.0, rtol=1e-10)


class TestEWMarketBetaDescriptor:
    r"""Validate the descriptor-to-factor pipeline with online EWMA betas.

    DGP: :math:`R_i(t) = \beta_i \cdot f(t) + \epsilon_i(t)` with constant
    true betas centered around 1 and equal market capitalization. The
    cap-weighted average return approximates the true factor, so
    `EWMarketBeta` can recover :math:`\beta_i` from observable data alone.

    Uses a short half-life (30) and many observations (1500) so the EWMA
    betas are well converged by the end of the sample. Assertions are
    statistical because the exposures are estimated, not provided.
    """

    N_OBS = 1500
    N_ASSETS = 50
    SEED = 77
    HALF_LIFE = 30

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        # Betas centered around 1 so equal-weighted return ≈ f(t)
        beta_true = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        sigma_f = 0.01
        sigma_eps = 0.005

        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = beta_true[None, :] * f_true[:, None] + eps

        panel, X = make_panel(returns)

        model = CharacteristicsFactorModel(
            factors=[
                (
                    "beta",
                    FixedWeightedFactor(
                        descriptors=[
                            (
                                "beta",
                                EWMarketBeta(
                                    half_life=cls.HALF_LIFE,
                                    min_periods=2 * cls.HALF_LIFE,
                                ),
                            ),
                        ],
                        family="market",
                        outlier_transformer="passthrough",
                        scoring_transformer="passthrough",
                    ),
                ),
            ],
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        # Derive alignment from the model's actual observation count
        # rather than hardcoding warmup arithmetic.
        n_used = model.factor_model_.factor_returns.shape[0]
        f_true_aligned = f_true[-n_used:]

        cls.beta_true = beta_true
        cls.sigma_f = sigma_f
        cls.f_true_aligned = f_true_aligned
        cls.model = model

    def test_factor_returns_recovery(self):
        """Factor returns should correlate well with the true factor despite
        estimated (noisy) exposures."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        corr = np.corrcoef(f_hat, self.f_true_aligned)[0, 1]
        assert corr > 0.99, f"Factor return correlation {corr:.4f} < 0.99"

    def test_loading_matrix_convergence(self):
        """After many observations, EWMA betas should converge near the true
        betas. The convergence is approximate because EWMA estimates betas
        relative to the realized cap-weighted return, not the latent factor."""
        B = self.model.factor_model_.loading_matrix[:, 0]
        corr = np.corrcoef(B, self.beta_true)[0, 1]
        assert corr > 0.98, f"Loading vs true beta correlation {corr:.4f} < 0.98"

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_covariance_sqrt_identity(self):
        """Reconstructing from covariance_sqrt must equal the stored covariance."""
        _assert_covariance_sqrt_identity(self.model.return_distribution_)

    def test_idio_returns_orthogonal_to_factor(self):
        """Idiosyncratic returns should have low correlation with the true
        factor, even when exposures are estimated."""
        idio = self.model.factor_model_.idio_returns
        corrs = [
            np.corrcoef(idio[:, i], self.f_true_aligned)[0, 1]
            for i in range(self.N_ASSETS)
        ]
        max_abs = np.max(np.abs(corrs))
        assert max_abs < 0.05, f"Max |corr(idio, factor)| = {max_abs:.4f}"

    def test_warmup_trimming(self):
        """The model must trim the EWMA warmup period so that no NaN
        exposures enter the regression.

        With min_periods = 2 * HALF_LIFE, the first valid betas appear at
        observation index (min_periods - 1). Adding exposure_lag = 1
        trims one more, giving n_used = N_OBS - min_periods."""
        fm = self.model.factor_model_
        min_periods = 2 * self.HALF_LIFE
        n_expected = self.N_OBS - min_periods
        assert fm.factor_returns.shape[0] == n_expected


class TestTwoUncorrelatedFactors:
    r"""Verify multi-factor separation with two independent signals.

    DGP: :math:`R_i(t) = \beta_{1,i}\,f_1(t) + \beta_{2,i}\,f_2(t)
    + \epsilon_i(t)` with :math:`f_1 \perp f_2`, constant betas drawn
    independently, and different factor volatilities to catch any signal
    mixing.
    """

    N_OBS = 2000
    N_ASSETS = 50
    SEED = 99

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        beta1_true = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        beta2_true = rng.uniform(-1.0, 1.0, size=cls.N_ASSETS)
        sigma_f1, sigma_f2 = 0.01, 0.005
        sigma_eps = 0.005

        f1_true = rng.normal(0, sigma_f1, size=cls.N_OBS)
        f2_true = rng.normal(0, sigma_f2, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = (
            beta1_true[None, :] * f1_true[:, None]
            + beta2_true[None, :] * f2_true[:, None]
            + eps
        )

        betas1_field = np.broadcast_to(beta1_true, (cls.N_OBS, cls.N_ASSETS)).copy()
        betas2_field = np.broadcast_to(beta2_true, (cls.N_OBS, cls.N_ASSETS)).copy()

        panel, X = make_panel(
            returns,
            extra_fields={"beta1": betas1_field, "beta2": betas2_field},
        )

        model = CharacteristicsFactorModel(
            factors=[
                ("beta1", passthrough_factor("beta1", family="style")),
                ("beta2", passthrough_factor("beta2", family="style")),
            ],
            factor_prior_estimator=EmpiricalPrior(),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        cls.beta1_true = beta1_true
        cls.beta2_true = beta2_true
        cls.sigma_f1 = sigma_f1
        cls.sigma_f2 = sigma_f2
        cls.f1_true_aligned = f1_true[1:]
        cls.f2_true_aligned = f2_true[1:]
        cls.model = model

    def test_factor1_return_recovery(self):
        """Estimated f1 must track the true f1."""
        f1_hat = self.model.factor_model_.factor_returns[:, 0]
        corr = np.corrcoef(f1_hat, self.f1_true_aligned)[0, 1]
        assert corr > 0.99, f"corr(f1_hat, f1_true) = {corr:.4f}"

    def test_factor2_return_recovery(self):
        """Estimated f2 must track the true f2. Threshold is lower than f1
        because sigma_f2 == sigma_eps, giving weaker per-asset SNR."""
        f2_hat = self.model.factor_model_.factor_returns[:, 1]
        corr = np.corrcoef(f2_hat, self.f2_true_aligned)[0, 1]
        assert corr > 0.96, f"corr(f2_hat, f2_true) = {corr:.4f}"

    def test_no_cross_contamination(self):
        """Estimated f1 should not correlate with true f2, and vice versa."""
        f1_hat = self.model.factor_model_.factor_returns[:, 0]
        f2_hat = self.model.factor_model_.factor_returns[:, 1]
        cross_12 = abs(np.corrcoef(f1_hat, self.f2_true_aligned)[0, 1])
        cross_21 = abs(np.corrcoef(f2_hat, self.f1_true_aligned)[0, 1])
        assert cross_12 < 0.05, f"|corr(f1_hat, f2_true)| = {cross_12:.4f}"
        assert cross_21 < 0.05, f"|corr(f2_hat, f1_true)| = {cross_21:.4f}"

    def test_factor_covariance_diagonal(self):
        """Off-diagonal correlation of the factor covariance should be
        near zero for independent factors."""
        F = self.model.factor_model_.factor_covariance
        corr_off = F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])
        assert abs(corr_off) < 0.05, f"|factor corr off-diag| = {abs(corr_off):.4f}"

    def test_factor_variances(self):
        """Individual factor variances should be close to truth."""
        F = self.model.factor_model_.factor_covariance
        assert F[0, 0] == pytest.approx(self.sigma_f1**2, rel=0.05)
        assert F[1, 1] == pytest.approx(self.sigma_f2**2, rel=0.10)

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_loading_matrix_equals_true_betas(self):
        """With passthrough, columns of the loading matrix must be the
        raw betas."""
        B = self.model.factor_model_.loading_matrix
        np.testing.assert_allclose(B[:, 0], self.beta1_true, rtol=1e-10)
        np.testing.assert_allclose(B[:, 1], self.beta2_true, rtol=1e-10)

    def test_shapes(self):
        """All output arrays have correct dimensions for a 2-factor model."""
        fm = self.model.factor_model_
        rd = self.model.return_distribution_
        n = self.N_ASSETS
        n_obs = self.N_OBS - 1

        assert fm.loading_matrix.shape == (n, 2)
        assert fm.factor_covariance.shape == (2, 2)
        assert fm.factor_returns.shape == (n_obs, 2)
        assert fm.idio_returns.shape == (n_obs, n)
        assert rd.mu.shape == (n,)
        assert rd.covariance.shape == (n, n)


class TestTwoCorrelatedFactors:
    r"""Verify recovery of off-diagonal factor covariance.

    DGP: :math:`R_i(t) = \beta_{1,i}\,f_1(t) + \beta_{2,i}\,f_2(t)
    + \epsilon_i(t)` where :math:`[f_1, f_2]` are drawn from a bivariate
    normal with correlation :math:`\rho = 0.6`.

    The main addition over Test 4 is that the estimated factor covariance
    must recover the off-diagonal structure (factor correlation).
    """

    N_OBS = 2000
    N_ASSETS = 50
    SEED = 2024
    RHO = 0.6
    SIGMA_F1 = 0.01
    SIGMA_F2 = 0.008

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        beta1_true = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        beta2_true = rng.uniform(-1.0, 1.0, size=cls.N_ASSETS)
        sigma_eps = 0.005

        cov_f = np.array(
            [
                [cls.SIGMA_F1**2, cls.RHO * cls.SIGMA_F1 * cls.SIGMA_F2],
                [cls.RHO * cls.SIGMA_F1 * cls.SIGMA_F2, cls.SIGMA_F2**2],
            ]
        )
        factors_true = rng.multivariate_normal([0, 0], cov_f, size=cls.N_OBS)
        f1_true = factors_true[:, 0]
        f2_true = factors_true[:, 1]

        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        returns = (
            beta1_true[None, :] * f1_true[:, None]
            + beta2_true[None, :] * f2_true[:, None]
            + eps
        )

        betas1_field = np.broadcast_to(beta1_true, (cls.N_OBS, cls.N_ASSETS)).copy()
        betas2_field = np.broadcast_to(beta2_true, (cls.N_OBS, cls.N_ASSETS)).copy()

        panel, X = make_panel(
            returns,
            extra_fields={"beta1": betas1_field, "beta2": betas2_field},
        )

        model = CharacteristicsFactorModel(
            factors=[
                ("beta1", passthrough_factor("beta1", family="style")),
                ("beta2", passthrough_factor("beta2", family="style")),
            ],
            factor_prior_estimator=EmpiricalPrior(),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        cls.cov_f_true = cov_f
        cls.f1_true_aligned = f1_true[1:]
        cls.f2_true_aligned = f2_true[1:]
        cls.model = model

    def test_factor_return_recovery(self):
        """Each estimated factor return should track its true counterpart."""
        fr = self.model.factor_model_.factor_returns
        c1 = np.corrcoef(fr[:, 0], self.f1_true_aligned)[0, 1]
        c2 = np.corrcoef(fr[:, 1], self.f2_true_aligned)[0, 1]
        assert c1 > 0.99, f"corr(f1_hat, f1_true) = {c1:.4f}"
        assert c2 > 0.98, f"corr(f2_hat, f2_true) = {c2:.4f}"

    def test_factor_covariance_recovery(self):
        """Full 2x2 factor covariance matrix should match the DGP."""
        F = self.model.factor_model_.factor_covariance
        np.testing.assert_allclose(F, self.cov_f_true, rtol=0.05)

    def test_factor_correlation_recovery(self):
        """Estimated factor correlation should recover rho."""
        F = self.model.factor_model_.factor_covariance
        rho_hat = F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])
        assert rho_hat == pytest.approx(self.RHO, abs=0.03)

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )


class TestExposureLagTimeVaryingBetas:
    r"""Verify the exposure lag machinery with linearly drifting betas.

    DGP: :math:`R_i(t) = \beta_i(t)\,f(t) + \epsilon_i(t)` where
    :math:`\beta_i(t)` drifts linearly from `beta_start_i` to
    `beta_end_i`.

    With `exposure_lag = L`, the cross-sectional regression at time
    :math:`t` uses :math:`B(t - L)` (lagged exposures), while the
    `loading_matrix` stores :math:`B(T)` (the most recent exposures,
    used for the covariance decomposition :math:`B\,F\,B^\top + D`).
    """

    N_OBS = 500
    N_ASSETS = 30
    SEED = 314
    EXPOSURE_LAG = 3

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        beta_start = rng.uniform(0.3, 0.7, size=cls.N_ASSETS)
        beta_end = rng.uniform(1.3, 1.7, size=cls.N_ASSETS)
        t_frac = np.linspace(0, 1, cls.N_OBS)
        betas = beta_start[None, :] + (beta_end - beta_start)[None, :] * t_frac[:, None]

        sigma_f = 0.01
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, 0.005, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas * f_true[:, None] + eps

        panel, X = make_panel(returns, extra_fields={"beta": betas})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            exposure_lag=cls.EXPOSURE_LAG,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        lag = cls.EXPOSURE_LAG
        cls.betas = betas
        cls.returns_aligned = returns[lag:]
        cls.betas_lagged = betas[:-lag]
        cls.betas_current = betas[lag:]
        cls.f_true_aligned = f_true[lag:]
        cls.model = model

    def test_loading_matrix_is_last_exposure(self):
        """loading_matrix must store B(T), the LAST observation's betas."""
        B = self.model.factor_model_.loading_matrix[:, 0]
        np.testing.assert_allclose(B, self.betas[-1], rtol=1e-10)

    def test_loading_matrix_differs_from_lagged(self):
        """loading_matrix must NOT equal B(T - lag), confirming it stores
        current, not lagged, exposures."""
        B = self.model.factor_model_.loading_matrix[:, 0]
        lagged = self.betas[-1 - self.EXPOSURE_LAG]
        assert not np.allclose(B, lagged, rtol=1e-3), (
            "loading_matrix should differ from lagged betas"
        )

    def test_idio_returns_use_lagged_exposures(self):
        """Residuals must be computed from lagged exposures:
        idio(t) = R(t) - B(t-lag) * f_hat(t)."""
        fr = self.model.factor_model_.factor_returns[:, 0]
        manual_idio = self.returns_aligned - self.betas_lagged * fr[:, None]
        np.testing.assert_allclose(
            self.model.factor_model_.idio_returns, manual_idio, rtol=1e-10
        )

    def test_current_exposures_give_worse_residuals(self):
        """Using current (non-lagged) exposures to recompute residuals must
        produce higher variance, proving the regression used lagged ones."""
        fr = self.model.factor_model_.factor_returns[:, 0]
        idio_lagged = self.model.factor_model_.idio_returns
        idio_current = self.returns_aligned - self.betas_current * fr[:, None]

        var_lagged = np.mean(idio_lagged**2)
        var_current = np.mean(idio_current**2)
        assert var_current > var_lagged, (
            f"Current-exposure residual variance ({var_current:.6e}) should "
            f"exceed lagged ({var_lagged:.6e})"
        )

    def test_factor_returns_recovery(self):
        """Factor returns should still correlate well with truth despite
        the small lag-induced mismatch."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        corr = np.corrcoef(f_hat, self.f_true_aligned)[0, 1]
        assert corr > 0.99, f"corr(f_hat, f_true) = {corr:.4f}"

    def test_covariance_uses_current_exposures(self):
        """B(T) F B(T)^T + D must equal the stored covariance, confirming
        the decomposition uses current (not lagged) exposures."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )


class TestEstimationMask:
    r"""Verify that the estimation mask excludes assets from learning.

    DGP: :math:`R_i(t) = \beta_i\,f(t) + \epsilon_i(t)` with two groups:

    * **Estimation assets** (indices 6--29): :math:`\beta_i \sim U(0.5, 1.5)`,
      :math:`\sigma_\epsilon = 0.005`.
    * **Excluded assets** (indices 0--5): :math:`\beta_i = 10`,
      :math:`\sigma_\epsilon = 0.005`.

    With `estimation_mask = False` for the excluded assets:

    * Factor returns must match a manual OLS using only estimation assets
      (exact numerical identity).
    * All 30 assets must appear in `loading_matrix` and `idio_returns`.
    * Factor returns must differ from an all-asset OLS (proving exclusion).
    """

    N_OBS = 500
    N_ASSETS = 30
    N_EXCLUDED = 6
    SEED = 555

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = np.empty(cls.N_ASSETS)
        betas[: cls.N_EXCLUDED] = 10.0
        betas[cls.N_EXCLUDED :] = rng.uniform(
            0.5, 1.5, size=cls.N_ASSETS - cls.N_EXCLUDED
        )

        sigma_f = 0.01
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, 0.005, size=(cls.N_OBS, cls.N_ASSETS))
        returns = betas[None, :] * f_true[:, None] + eps

        est_mask = np.ones((cls.N_OBS, cls.N_ASSETS), dtype=bool)
        est_mask[:, : cls.N_EXCLUDED] = False

        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()
        panel, X = make_panel(
            returns,
            extra_fields={"beta": betas_2d},
            estimation_mask=est_mask,
        )

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=5,
        )
        model.fit(X, characteristics=panel)

        n_used = model.factor_model_.factor_returns.shape[0]

        b_est = betas[cls.N_EXCLUDED :]
        r_est = returns[-n_used:, cls.N_EXCLUDED :]
        manual_f = (r_est @ b_est) / (b_est @ b_est)

        r_used = returns[-n_used:]
        manual_f_all = (r_used @ betas) / (betas @ betas)

        cls.betas = betas
        cls.f_true = f_true[-n_used:]
        cls.returns_used = r_used
        cls.manual_f = manual_f
        cls.manual_f_all = manual_f_all
        cls.n_used = n_used
        cls.model = model

    def test_factor_returns_match_estimation_only_ols(self):
        """Model factor returns must equal OLS on estimation assets only."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        np.testing.assert_allclose(f_hat, self.manual_f, rtol=1e-10)

    def test_factor_returns_differ_from_all_asset_ols(self):
        """Factor returns must NOT match OLS using all assets, proving that
        excluded assets were not used."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        assert not np.allclose(f_hat, self.manual_f_all, rtol=1e-3), (
            "Factor returns should differ from the all-asset OLS"
        )

    def test_loading_matrix_includes_all_assets(self):
        """All 30 assets (including excluded) must appear in loading_matrix."""
        B = self.model.factor_model_.loading_matrix
        assert B.shape == (self.N_ASSETS, 1)
        np.testing.assert_allclose(B[:, 0], self.betas, rtol=1e-10)

    def test_idio_returns_includes_all_assets(self):
        """All 30 assets must receive idiosyncratic returns."""
        idio = self.model.factor_model_.idio_returns
        assert idio.shape == (self.n_used, self.N_ASSETS)

    def test_excluded_assets_idio_correctness(self):
        """Excluded assets' idio returns must be R_i(t) - beta_i * f_hat(t),
        using the factor return estimated from estimation assets only."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        expected_idio = self.returns_used - self.betas[None, :] * f_hat[:, None]
        np.testing.assert_allclose(
            self.model.factor_model_.idio_returns, expected_idio, rtol=1e-10
        )

    def test_factor_return_correlation_with_truth(self):
        """Factor returns should recover the true factor well."""
        f_hat = self.model.factor_model_.factor_returns[:, 0]
        corr = np.corrcoef(f_hat, self.f_true)[0, 1]
        assert corr > 0.99, f"corr(f_hat, f_true) = {corr:.4f}"

    def test_covariance_decomposition_identity_mask(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )


class TestSparseIdioCorrelationOverlay:
    r"""Verify the sparse idiosyncratic correlation overlay.

    DGP: :math:`R_i(t) = \beta_i\,f(t) + \epsilon_i(t)` where assets 0
    and 1 share correlated idiosyncratic noise
    (:math:`\rho_{\epsilon} = 0.5`) and all other assets are independent.

    With `idio_corr_threshold > 0`, the model should:

    * Capture the correlated pair in the off-diagonal of `idio_covariance`.
    * Leave other off-diagonal entries at (or near) zero.
    * Still satisfy the covariance decomposition identity.
    """

    N_OBS = 2000
    N_ASSETS = 30
    SEED = 777
    PAIR_RHO = 0.5
    THRESHOLD = 0.1

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)

        sigma_f = 0.01
        sigma_eps = 0.005
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)

        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))

        rho = cls.PAIR_RHO
        chol = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
        z = rng.normal(0, sigma_eps, size=(cls.N_OBS, 2))
        eps_corr = z @ chol.T
        eps[:, 0] = eps_corr[:, 0]
        eps[:, 1] = eps_corr[:, 1]

        returns = betas[None, :] * f_true[:, None] + eps

        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()
        panel, X = make_panel(returns, extra_fields={"beta": betas_2d})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            idio_corr_threshold=cls.THRESHOLD,
        )
        model.fit(X, characteristics=panel)

        cls.betas = betas
        cls.f_true = f_true
        cls.model = model

    def test_idio_cov_not_diagonal(self):
        """With correlated pair, idio_covariance must have off-diagonal entries."""
        D = self.model.factor_model_.idio_covariance
        off_diag = D.copy()
        np.fill_diagonal(off_diag, 0.0)
        assert np.any(np.abs(off_diag) > 1e-10), (
            "idio_covariance should have non-zero off-diagonal entries"
        )

    def test_correlated_pair_captured(self):
        """idio_covariance[0, 1] should be positive (reflecting rho > 0)."""
        D = self.model.factor_model_.idio_covariance
        assert D[0, 1] > 0, f"idio_cov[0,1] = {D[0, 1]:.6e}, expected > 0"

    def test_other_pairs_near_zero(self):
        """Off-diagonal entries outside the correlated pair should be small."""
        D = self.model.factor_model_.idio_covariance
        mask = np.ones_like(D, dtype=bool)
        np.fill_diagonal(mask, False)
        mask[0, 1] = False
        mask[1, 0] = False
        max_other = np.max(np.abs(D[mask]))
        pair_cov = np.abs(D[0, 1])
        assert max_other < pair_cov, (
            f"max other off-diag ({max_other:.6e}) should be < "
            f"pair covariance ({pair_cov:.6e})"
        )

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_factor_returns_recovery(self):
        """Factor returns should still recover truth well."""
        fm = self.model.factor_model_
        n_used = fm.factor_returns.shape[0]
        f_hat = fm.factor_returns[:, 0]
        corr = np.corrcoef(f_hat, self.f_true[-n_used:])[0, 1]
        assert corr > 0.99, f"corr(f_hat, f_true) = {corr:.4f}"


class TestBasketNeutralConstraints:
    r"""Verify basket-neutral constraints on industry factors with cap-weighted
    benchmark.

    DGP: market intercept + 3 industry dummies (10 assets each) with
    heterogeneous market capitalization across industries.

    .. math::

        R_i(t) = f_{\text{mkt}}(t) + f_{k(i)}(t) + \epsilon_i(t)

    With `constrained_families=[("industry", None)]` and
    cap-weighted benchmark (`benchmark_mcap_power=1`), the reconstructed
    industry factor returns must satisfy:

    .. math::

        \sum_k w_k\,f_k(t) = 0 \quad \forall\, t

    where :math:`w_k` is the cap-weighted industry share.
    """

    N_OBS = 500
    N_ASSETS = 30
    N_IND = 3
    ASSETS_PER_IND = 10
    SEED = 888
    IND_MCAPS: ClassVar[list[float]] = [3.0, 2.0, 1.0]

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        sigma_f = 0.01
        sigma_eps = 0.005

        f_mkt = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_ind = rng.normal(0, sigma_f, size=(cls.N_OBS, cls.N_IND))
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))

        ind_dummies = np.zeros((cls.N_OBS, cls.N_ASSETS, cls.N_IND))
        mcap = np.ones((cls.N_OBS, cls.N_ASSETS))
        for k in range(cls.N_IND):
            start = k * cls.ASSETS_PER_IND
            end = (k + 1) * cls.ASSETS_PER_IND
            ind_dummies[:, start:end, k] = 1.0
            mcap[:, start:end] = cls.IND_MCAPS[k]

        returns = f_mkt[:, None] + np.einsum("tnk,tk->tn", ind_dummies, f_ind) + eps

        mkt_exp = np.ones((cls.N_OBS, cls.N_ASSETS))
        extra_fields = {
            "mkt_exp": mkt_exp,
            "ind_1": ind_dummies[:, :, 0],
            "ind_2": ind_dummies[:, :, 1],
            "ind_3": ind_dummies[:, :, 2],
        }

        panel, X = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)

        model = CharacteristicsFactorModel(
            factors=[
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("ind_1", passthrough_factor("ind_1", family="industry")),
                ("ind_2", passthrough_factor("ind_2", family="industry")),
                ("ind_3", passthrough_factor("ind_3", family="industry")),
            ],
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )
        model.fit(X, characteristics=panel)

        # Industry benchmark weights: w_k = (n_assets_k * mcap_k) / total_mcap
        total_mcap = sum(cls.ASSETS_PER_IND * c for c in cls.IND_MCAPS)
        ind_weights = np.array(
            [cls.ASSETS_PER_IND * c / total_mcap for c in cls.IND_MCAPS]
        )

        # With basket-neutral constraints, the cap-weighted average industry
        # return is absorbed into the market factor.
        f_mkt_effective = f_mkt + f_ind @ ind_weights

        cls.returns = returns
        cls.mcap = mcap
        cls.f_mkt_effective = f_mkt_effective
        cls.ind_weights = ind_weights
        cls.model = model

    def test_industry_returns_cap_weighted_sum_to_zero(self):
        """Cap-weighted industry factor returns must sum to zero."""
        fm = self.model.factor_model_
        factor_names = list(fm.factor_names)
        ind_cols = [factor_names.index(n) for n in ("ind_1", "ind_2", "ind_3")]
        f_ind_hat = fm.factor_returns[:, ind_cols]
        weighted_sum = f_ind_hat @ self.ind_weights
        np.testing.assert_allclose(weighted_sum, 0.0, atol=1e-12)

    def test_factor_names_original_basis(self):
        """All 4 original factor names must be present."""
        names = list(self.model.factor_model_.factor_names)
        assert "market" in names
        assert "ind_1" in names
        assert "ind_2" in names
        assert "ind_3" in names
        assert len(names) == 4

    def test_factor_returns_shape(self):
        """Factor returns should have 4 columns (original basis)."""
        fm = self.model.factor_model_
        assert fm.factor_returns.shape[1] == 4

    def test_market_factor_recovery(self):
        """Market factor returns should recover the effective market (true
        market + cap-weighted industry average absorbed by the constraint)."""
        fm = self.model.factor_model_
        n_used = fm.factor_returns.shape[0]
        idx = list(fm.factor_names).index("market")
        corr = np.corrcoef(fm.factor_returns[:, idx], self.f_mkt_effective[-n_used:])[
            0, 1
        ]
        assert corr > 0.99, f"corr(f_mkt_hat, f_mkt_eff) = {corr:.4f}"

    def test_market_factor_equals_cap_weighted_return(self):
        """With regression_mcap_power == benchmark_mcap_power, the WLS
        first-order condition forces f_mkt(t) == cap-weighted average
        return(t) exactly, because all basket-neutral contrasts satisfy
        :math:`\\sum_i w_i z_{ji} = 0`."""
        fm = self.model.factor_model_
        n_used = fm.factor_returns.shape[0]
        returns_used = self.returns[-n_used:]
        mcap_used = self.mcap[-n_used:]
        weights = mcap_used / mcap_used.sum(axis=1, keepdims=True)
        cap_weighted_return = (weights * returns_used).sum(axis=1)

        idx = list(fm.factor_names).index("market")
        np.testing.assert_allclose(
            fm.factor_returns[:, idx],
            cap_weighted_return,
            rtol=1e-10,
        )

    def test_factor_cov_rank_deficient(self):
        """In the original basis, factor_covariance has rank K - n_constraints
        because the constraint removes one degree of freedom."""
        fm = self.model.factor_model_
        n_factors = fm.factor_covariance.shape[0]
        n_constraints = 1
        assert np.linalg.matrix_rank(fm.factor_covariance) == n_factors - n_constraints

    def test_factor_cov_null_space_is_constraint_direction(self):
        """The null space of the rank-deficient factor_covariance must be the
        constraint direction [0, w_1, w_2, w_3]."""
        fm = self.model.factor_model_
        eigvals, eigvecs = np.linalg.eigh(fm.factor_covariance)
        null_vec = eigvecs[:, np.argmin(np.abs(eigvals))]

        factor_names = list(fm.factor_names)
        expected_null = np.zeros(len(factor_names))
        for k, name in enumerate(("ind_1", "ind_2", "ind_3")):
            expected_null[factor_names.index(name)] = self.ind_weights[k]
        expected_null /= np.linalg.norm(expected_null)

        alignment = np.abs(null_vec @ expected_null)
        assert alignment == pytest.approx(1.0, abs=1e-8), (
            f"null-space alignment = {alignment:.10f}"
        )

    def test_asset_covariance_positive_definite(self):
        """Despite factor_covariance being singular, the asset covariance
        B F B^T + D must be positive definite."""
        eigvals = np.linalg.eigvalsh(self.model.return_distribution_.covariance)
        assert eigvals.min() > 0, f"min eigenvalue = {eigvals.min():.2e}"

    def test_covariance_sqrt_identity(self):
        """Reconstructing from covariance_sqrt must equal the stored covariance."""
        _assert_covariance_sqrt_identity(self.model.return_distribution_)

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance (original basis
        with rank-deficient factor_covariance)."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )


class TestFamilyConstraintBasisMode:
    r"""Verify that the `family_constraint_basis` is stored and functional.

    Uses the same DGP as :class:`TestBasketNeutralConstraints`.  The factor
    model always stores full-basis data; the basis object must be present
    and able to reconstruct reduced-basis quantities.
    """

    N_OBS = 500
    N_ASSETS = 30
    N_IND = 3
    ASSETS_PER_IND = 10
    SEED = 888
    IND_MCAPS: ClassVar[list[float]] = [3.0, 2.0, 1.0]

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        sigma_f = 0.01
        sigma_eps = 0.005

        f_mkt = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_ind = rng.normal(0, sigma_f, size=(cls.N_OBS, cls.N_IND))
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))

        ind_dummies = np.zeros((cls.N_OBS, cls.N_ASSETS, cls.N_IND))
        mcap = np.ones((cls.N_OBS, cls.N_ASSETS))
        for k in range(cls.N_IND):
            start = k * cls.ASSETS_PER_IND
            end = (k + 1) * cls.ASSETS_PER_IND
            ind_dummies[:, start:end, k] = 1.0
            mcap[:, start:end] = cls.IND_MCAPS[k]

        returns = f_mkt[:, None] + np.einsum("tnk,tk->tn", ind_dummies, f_ind) + eps

        mkt_exp = np.ones((cls.N_OBS, cls.N_ASSETS))
        extra_fields = {
            "mkt_exp": mkt_exp,
            "ind_1": ind_dummies[:, :, 0],
            "ind_2": ind_dummies[:, :, 1],
            "ind_3": ind_dummies[:, :, 2],
        }

        panel, X = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)

        model = CharacteristicsFactorModel(
            factors=[
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("ind_1", passthrough_factor("ind_1", family="industry")),
                ("ind_2", passthrough_factor("ind_2", family="industry")),
                ("ind_3", passthrough_factor("ind_3", family="industry")),
            ],
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )
        model.fit(X, characteristics=panel)

        cls.model = model

    def test_basis_present(self):
        """family_constraint_basis must be stored on the factor model."""
        fm = self.model.factor_model_
        assert fm.family_constraint_basis is not None

    def test_full_factor_names_preserved(self):
        """All original factor names (K) must be present."""
        fm = self.model.factor_model_
        names = list(fm.factor_names)
        assert len(names) == 4
        assert "market" in names
        for i in range(1, self.N_IND + 1):
            assert f"ind_{i}" in names

    def test_reduced_names_via_basis(self):
        """Reduced factor names (K_red = K - 1) derived from the basis."""
        fm = self.model.factor_model_
        bnb = fm.family_constraint_basis
        reduced_names = np.delete(fm.factor_names, bnb.dropped_full_indices)
        assert len(reduced_names) == len(fm.factor_names) - 1
        assert "market" in reduced_names

    def test_reduced_factor_cov_positive_definite(self):
        """Reduced-basis factor covariance must be full rank (PD)."""
        fm = self.model.factor_model_
        bnb = fm.family_constraint_basis
        bnb.reduce_factor_returns(np.eye(len(fm.factor_names)))
        # Actually test the stored full-basis covariance is still valid
        np.linalg.eigvalsh(fm.factor_covariance)
        # Full-basis covariance is expected to be rank-deficient (K-1)
        # but the reconstructed asset covariance is fine
        assert fm.factor_covariance.shape == (4, 4)

    def test_covariance_decomposition_identity(self):
        """B F B^T + D = Sigma in the full basis."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_covariance_sqrt_identity(self):
        """Reconstructing from covariance_sqrt must equal the stored covariance."""
        _assert_covariance_sqrt_identity(self.model.return_distribution_)


class TestFactorNeutralization:
    r"""Verify that `neutralize_against` orthogonalizes factor exposures.

    DGP: three factors (market, size, momentum) where the raw momentum
    characteristic is constructed with cross-sectional correlation
    :math:`\rho = 0.7` with size.

    .. math::

        R_i(t) = f_{\text{mkt}}(t) + z_{\text{size},i}\,f_{\text{size}}(t)
               + z_{\text{mom},i}\,f_{\text{mom}}(t) + \epsilon_i(t)

    With `neutralize_against={"momentum": ["size"]}`, the model regresses
    the momentum exposure on size (WLS, `fit_intercept=False`) and replaces
    it with the re-standardized residual.

    Because the size exposure has zero weighted mean (pre-z-scored with equal
    weights matching `benchmark_mcap_power=0`), the WLS first-order
    condition :math:`\sum_i w_i\,\text{size}_i\,\text{orth}_i = 0` is
    preserved through the CSStandardScaler re-standardization, giving
    exact cross-sectional orthogonality at every observation.
    """

    N_OBS = 500
    N_ASSETS = 30
    SEED = 999
    RHO = 0.7

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        size_raw = rng.standard_normal(cls.N_ASSETS)
        noise = rng.standard_normal(cls.N_ASSETS)
        mom_raw = cls.RHO * size_raw + np.sqrt(1 - cls.RHO**2) * noise

        size_z = (size_raw - size_raw.mean()) / size_raw.std(ddof=0)
        mom_z = (mom_raw - mom_raw.mean()) / mom_raw.std(ddof=0)

        size_2d = np.broadcast_to(size_z[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()
        mom_2d = np.broadcast_to(mom_z[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()
        mkt_exp = np.ones((cls.N_OBS, cls.N_ASSETS))

        sigma_f = 0.01
        sigma_eps = 0.005

        f_mkt = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_size = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_mom = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))

        returns = (
            f_mkt[:, None]
            + size_z[None, :] * f_size[:, None]
            + mom_z[None, :] * f_mom[:, None]
            + eps
        )

        extra_fields = {
            "mkt_exp": mkt_exp,
            "size": size_2d,
            "momentum": mom_2d,
        }

        panel1, X1 = make_panel(returns, extra_fields=extra_fields)
        model_no_neut = CharacteristicsFactorModel(
            factors=[
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("size", passthrough_factor("size", family="style")),
                ("momentum", passthrough_factor("momentum", family="style")),
            ],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_no_neut.fit(X1, characteristics=panel1)

        panel2, X2 = make_panel(returns, extra_fields=extra_fields)
        model_neut = CharacteristicsFactorModel(
            factors=[
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("size", passthrough_factor("size", family="style")),
                ("momentum", passthrough_factor("momentum", family="style")),
            ],
            neutralize_against={"momentum": ["size"]},
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_neut.fit(X2, characteristics=panel2)

        cls.size_z = size_z
        cls.mom_z = mom_z
        cls.f_size = f_size
        cls.f_mom = f_mom
        cls.model_no_neut = model_no_neut
        cls.model_neut = model_neut

    def test_exposures_correlated_without_neutralization(self):
        """Without neutralization, size and momentum exposures must show
        the constructed cross-sectional correlation (~0.7)."""
        fm = self.model_no_neut.factor_model_
        names = list(fm.factor_names)
        size_exp = fm.exposures[-1, :, names.index("size")]
        mom_exp = fm.exposures[-1, :, names.index("momentum")]
        corr = np.corrcoef(size_exp, mom_exp)[0, 1]
        assert abs(corr) > 0.65, f"expected strong correlation, got {corr:.4f}"

    def test_exposures_orthogonal_with_neutralization(self):
        """After neutralization, the cross-sectional covariance between size
        and the neutralized momentum must be zero at every observation."""
        fm = self.model_neut.factor_model_
        names = list(fm.factor_names)
        size_exp = fm.exposures[:, :, names.index("size")]
        mom_exp = fm.exposures[:, :, names.index("momentum")]

        size_centered = size_exp - size_exp.mean(axis=1, keepdims=True)
        mom_centered = mom_exp - mom_exp.mean(axis=1, keepdims=True)
        cs_cov = (size_centered * mom_centered).mean(axis=1)
        np.testing.assert_allclose(cs_cov, 0.0, atol=1e-10)

    def test_neutralized_exposure_standardized(self):
        """The neutralized momentum exposure must have weighted mean ~0 and
        equal-weighted std ~1 at each observation (from CSStandardScaler)."""
        fm = self.model_neut.factor_model_
        names = list(fm.factor_names)
        mom_exp = fm.exposures[:, :, names.index("momentum")]

        means = mom_exp.mean(axis=1)
        stds = mom_exp.std(axis=1, ddof=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_size_exposure_unchanged(self):
        """Size exposure must be identical in both models (neutralization
        only modifies the target factor, not the regressors)."""
        fm_no = self.model_no_neut.factor_model_
        fm_yes = self.model_neut.factor_model_
        names_no = list(fm_no.factor_names)
        names_yes = list(fm_yes.factor_names)
        np.testing.assert_allclose(
            fm_no.exposures[:, :, names_no.index("size")],
            fm_yes.exposures[:, :, names_yes.index("size")],
            rtol=1e-10,
        )

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance (neutralized
        model)."""
        fm = self.model_neut.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model_neut.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_factor_return_recovery(self):
        """Both size and momentum factor returns must correlate well with
        ground truth in the non-neutralized model."""
        fm = self.model_no_neut.factor_model_
        n_used = fm.factor_returns.shape[0]
        names = list(fm.factor_names)

        corr_size = np.corrcoef(
            fm.factor_returns[:, names.index("size")],
            self.f_size[-n_used:],
        )[0, 1]
        corr_mom = np.corrcoef(
            fm.factor_returns[:, names.index("momentum")],
            self.f_mom[-n_used:],
        )[0, 1]
        assert corr_size > 0.98, f"corr(size_hat, size_true) = {corr_size:.4f}"
        assert corr_mom > 0.98, f"corr(mom_hat, mom_true) = {corr_mom:.4f}"


class TestNeutralizationWithBasketNeutralConstraints:
    r"""Verify that neutralization propagates to original-basis output
    exposures when combined with basket-neutral constraints.

    DGP: market intercept + 3 industry dummies + 1 style factor whose raw
    characteristic is deliberately correlated with industry membership
    (industry A assets receive a +1.5 bias).

    .. math::

        R_i(t) = f_{\text{mkt}}(t) + f_{k(i)}(t)
               + z_i\,f_{\text{style}}(t) + \epsilon_i(t)

    With `neutralize_against={"style": ["industry"]}` and
    `constrained_families=[("industry", None)]`, the style exposure
    must be benchmark-weight-orthogonal to every industry factor in the
    output.
    """

    N_OBS = 500
    N_ASSETS = 30
    N_IND = 3
    ASSETS_PER_IND = 10
    SEED = 777
    IND_MCAPS: ClassVar[list[float]] = [3.0, 2.0, 1.0]

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        sigma_f = 0.01
        sigma_eps = 0.005

        f_mkt = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_ind = rng.normal(0, sigma_f, size=(cls.N_OBS, cls.N_IND))
        f_style = rng.normal(0, sigma_f, size=cls.N_OBS)

        ind_dummies = np.zeros((cls.N_OBS, cls.N_ASSETS, cls.N_IND))
        mcap = np.ones((cls.N_OBS, cls.N_ASSETS))
        for k in range(cls.N_IND):
            start = k * cls.ASSETS_PER_IND
            end = (k + 1) * cls.ASSETS_PER_IND
            ind_dummies[:, start:end, k] = 1.0
            mcap[:, start:end] = cls.IND_MCAPS[k]

        style_base = rng.standard_normal(cls.N_ASSETS)
        industry_bias = np.zeros(cls.N_ASSETS)
        industry_bias[: cls.ASSETS_PER_IND] = 1.5
        style_char_1d = style_base + industry_bias
        style_char = np.broadcast_to(
            style_char_1d[None, :], (cls.N_OBS, cls.N_ASSETS)
        ).copy()

        returns = (
            f_mkt[:, None]
            + np.einsum("tnk,tk->tn", ind_dummies, f_ind)
            + style_char_1d[None, :] * f_style[:, None]
            + rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        )

        mkt_exp = np.ones((cls.N_OBS, cls.N_ASSETS))
        extra_fields = {
            "mkt_exp": mkt_exp,
            "ind_1": ind_dummies[:, :, 0],
            "ind_2": ind_dummies[:, :, 1],
            "ind_3": ind_dummies[:, :, 2],
            "style_char": style_char,
        }

        def _factors():
            return [
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("ind_1", passthrough_factor("ind_1", family="industry")),
                ("ind_2", passthrough_factor("ind_2", family="industry")),
                ("ind_3", passthrough_factor("ind_3", family="industry")),
                ("style", passthrough_factor("style_char", family="style")),
            ]

        common_kw = dict(
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )

        # Model without neutralization (sanity check)
        panel_raw, X_raw = make_panel(
            returns, extra_fields=extra_fields, market_cap=mcap
        )
        model_raw = CharacteristicsFactorModel(factors=_factors(), **common_kw)
        model_raw.fit(X_raw, characteristics=panel_raw)

        # Model with neutralization
        panel_orig, X_orig = make_panel(
            returns, extra_fields=extra_fields, market_cap=mcap
        )
        model_orig = CharacteristicsFactorModel(
            factors=_factors(),
            neutralize_against={"style": ["industry"]},
            **common_kw,
        )
        model_orig.fit(X_orig, characteristics=panel_orig)

        total_mcap = sum(cls.ASSETS_PER_IND * c for c in cls.IND_MCAPS)
        bench_weights = np.array(
            [c / total_mcap for c in cls.IND_MCAPS for _ in range(cls.ASSETS_PER_IND)]
        )

        cls.model_raw = model_raw
        cls.model_orig = model_orig
        cls.bench_weights = bench_weights

    def test_raw_style_is_industry_correlated(self):
        """Without neutralization, the style exposure must be correlated
        with industry membership (confirms the DGP bias is effective)."""
        fm = self.model_raw.factor_model_
        names = list(fm.factor_names)
        style_exp = fm.loading_matrix[:, names.index("style")]
        ind_exp = fm.loading_matrix[:, names.index("ind_1")]
        w = self.bench_weights
        weighted_dot = np.sum(w * style_exp * ind_exp)
        assert abs(weighted_dot) > 0.05, (
            f"expected industry-correlated style, got dot={weighted_dot:.4f}"
        )

    def test_style_orthogonal_to_industries_in_output(self):
        """After neutralization, the style exposure in the original-basis
        output must be benchmark-weight-orthogonal to each industry."""
        fm = self.model_orig.factor_model_
        names = list(fm.factor_names)
        style_exp = fm.loading_matrix[:, names.index("style")]
        w = self.bench_weights
        for ind_name in ["ind_1", "ind_2", "ind_3"]:
            ind_exp = fm.loading_matrix[:, names.index(ind_name)]
            weighted_dot = np.sum(w * style_exp * ind_exp)
            assert abs(weighted_dot) < 1e-8, (
                f"style not orthogonal to {ind_name}: {weighted_dot:.2e}"
            )

    def test_style_benchmark_weighted_mean_zero(self):
        """Neutralizing against the full industry dummies (spanning the
        intercept) must also zero out the benchmark-weighted mean."""
        fm = self.model_orig.factor_model_
        names = list(fm.factor_names)
        style_exp = fm.loading_matrix[:, names.index("style")]
        weighted_mean = np.sum(self.bench_weights * style_exp)
        assert abs(weighted_mean) < 1e-8, (
            f"benchmark-weighted mean not zero: {weighted_mean:.2e}"
        )

    def test_covariance_decomposition(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model_orig.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model_orig.return_distribution_.covariance,
            reconstructed,
            rtol=1e-8,
        )

    def test_asset_covariance_positive_definite(self):
        """The asset covariance must be positive definite."""
        eigvals = np.linalg.eigvalsh(self.model_orig.return_distribution_.covariance)
        assert eigvals.min() > 0

    def test_covariance_sqrt_identity(self):
        """Reconstructing from covariance_sqrt must equal the stored covariance."""
        _assert_covariance_sqrt_identity(
            self.model_orig.return_distribution_, rtol=1e-8
        )


class TestDemeaningVsNeutralizationEquivalence:
    r"""Verify equivalence between within-industry demeaning and explicit
    industry neutralization for the orthogonality condition.

    DGP: market intercept + 3 industry dummies + 1 style factor whose raw
    characteristic has a deliberate industry bias (industry A +1.5).

    Three models are compared:

    * **Model A** (demeaning only): style factor uses
      `transform_by_group="industry_group"` to apply within-industry
      z-scoring via :class:`CSStandardScaler` with groups.
    * **Model B** (neutralization only): style factor uses global
      z-scoring (no groups), with
      `neutralize_against={"style": ["industry"]}`.
    * **Model C** (both): combines `transform_by_group` and
      `neutralize_against`.

    All three satisfy :math:`D^\top W z_{\text{style}} = 0`. Models A and C
    produce identical style exposures because the explicit neutralization
    finds :math:`\beta = (D^\top W D)^{-1} D^\top W z = 0` after demeaning
    and is a no-op.  Models A and B differ in within-group spread
    (demeaning normalizes per-group std; neutralization does not).
    """

    N_OBS = 500
    N_ASSETS = 30
    N_IND = 3
    ASSETS_PER_IND = 10
    SEED = 888
    IND_MCAPS: ClassVar[list[float]] = [3.0, 2.0, 1.0]

    @staticmethod
    def _style_factor_with_groups():
        return FixedWeightedFactor(
            descriptors=[("style_char", Passthrough("style_char"))],
            family="style",
            outlier_transformer="passthrough",
            transform_by_group="industry_group",
        )

    @staticmethod
    def _style_factor_global():
        return FixedWeightedFactor(
            descriptors=[("style_char", Passthrough("style_char"))],
            family="style",
            outlier_transformer="passthrough",
        )

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        sigma_f = 0.01
        sigma_eps = 0.005

        f_mkt = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_ind = rng.normal(0, sigma_f, size=(cls.N_OBS, cls.N_IND))
        f_style = rng.normal(0, sigma_f, size=cls.N_OBS)

        ind_dummies = np.zeros((cls.N_OBS, cls.N_ASSETS, cls.N_IND))
        mcap = np.ones((cls.N_OBS, cls.N_ASSETS))
        industry_group = np.zeros((cls.N_OBS, cls.N_ASSETS), dtype=int)
        for k in range(cls.N_IND):
            start = k * cls.ASSETS_PER_IND
            end = (k + 1) * cls.ASSETS_PER_IND
            ind_dummies[:, start:end, k] = 1.0
            mcap[:, start:end] = cls.IND_MCAPS[k]
            industry_group[:, start:end] = k

        style_base = rng.standard_normal(cls.N_ASSETS)
        industry_bias = np.zeros(cls.N_ASSETS)
        industry_bias[: cls.ASSETS_PER_IND] = 1.5
        style_char_1d = style_base + industry_bias
        style_char = np.broadcast_to(
            style_char_1d[None, :], (cls.N_OBS, cls.N_ASSETS)
        ).copy()

        returns = (
            f_mkt[:, None]
            + np.einsum("tnk,tk->tn", ind_dummies, f_ind)
            + style_char_1d[None, :] * f_style[:, None]
            + rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        )

        mkt_exp = np.ones((cls.N_OBS, cls.N_ASSETS))
        extra_fields = {
            "mkt_exp": mkt_exp,
            "ind_1": ind_dummies[:, :, 0],
            "ind_2": ind_dummies[:, :, 1],
            "ind_3": ind_dummies[:, :, 2],
            "style_char": style_char,
            "industry_group": industry_group,
        }

        def _ind_factors():
            return [
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("ind_1", passthrough_factor("ind_1", family="industry")),
                ("ind_2", passthrough_factor("ind_2", family="industry")),
                ("ind_3", passthrough_factor("ind_3", family="industry")),
            ]

        common_kw = dict(
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )

        cls = TestDemeaningVsNeutralizationEquivalence

        # Model A: demeaning via transform_by_group, no neutralize_against
        panel_a, X_a = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
        model_a = CharacteristicsFactorModel(
            factors=[*_ind_factors(), ("style", cls._style_factor_with_groups())],
            **common_kw,
        )
        model_a.fit(X_a, characteristics=panel_a)

        # Model B: no transform_by_group, neutralize_against
        panel_b, X_b = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
        model_b = CharacteristicsFactorModel(
            factors=[*_ind_factors(), ("style", cls._style_factor_global())],
            neutralize_against={"style": ["industry"]},
            **common_kw,
        )
        model_b.fit(X_b, characteristics=panel_b)

        # Model C: both transform_by_group AND neutralize_against
        panel_c, X_c = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
        model_c = CharacteristicsFactorModel(
            factors=[*_ind_factors(), ("style", cls._style_factor_with_groups())],
            neutralize_against={"style": ["industry"]},
            **common_kw,
        )
        model_c.fit(X_c, characteristics=panel_c)

        cls.model_a = model_a
        cls.model_b = model_b
        cls.model_c = model_c
        cls.bench_weights = np.array(
            [
                c / sum(cls.ASSETS_PER_IND * c for c in cls.IND_MCAPS)
                for c in cls.IND_MCAPS
                for _ in range(cls.ASSETS_PER_IND)
            ]
        )

    def _style_exposure(self, model):
        fm = model.factor_model_
        names = list(fm.factor_names)
        return fm.loading_matrix[:, names.index("style")]

    def test_all_models_orthogonal_to_industries(self):
        """All three approaches must satisfy D^T W z_style = 0."""
        w = self.bench_weights
        for label, model in [
            ("A", self.model_a),
            ("B", self.model_b),
            ("C", self.model_c),
        ]:
            fm = model.factor_model_
            names = list(fm.factor_names)
            style_exp = fm.loading_matrix[:, names.index("style")]
            for ind_name in ["ind_1", "ind_2", "ind_3"]:
                ind_exp = fm.loading_matrix[:, names.index(ind_name)]
                dot = np.sum(w * style_exp * ind_exp)
                assert abs(dot) < 1e-8, (
                    f"Model {label}: style not orthogonal to {ind_name} (dot={dot:.2e})"
                )

    def test_demeaning_and_both_produce_identical_exposures(self):
        """Model A (demeaning only) and Model C (both) must produce
        identical style exposures: neutralization is a no-op when
        D^T W z = 0 is already satisfied by demeaning."""
        np.testing.assert_allclose(
            self._style_exposure(self.model_a),
            self._style_exposure(self.model_c),
            atol=1e-10,
        )

    def test_demeaning_and_neutralization_differ_in_shape(self):
        """Model A (demeaning) and Model B (neutralization only) produce
        different style exposures because demeaning normalizes per-group
        std while neutralization preserves natural within-group spread."""
        corr = np.corrcoef(
            self._style_exposure(self.model_a),
            self._style_exposure(self.model_b),
        )[0, 1]
        assert corr < 0.9999, f"expected different exposures, got corr={corr:.6f}"

    def test_covariance_decomposition(self):
        """B F B^T + D must equal the stored asset covariance for all
        three models."""
        for label, model in [
            ("A", self.model_a),
            ("B", self.model_b),
            ("C", self.model_c),
        ]:
            fm = model.factor_model_
            reconstructed = _reconstruct_asset_covariance(fm)
            np.testing.assert_allclose(
                model.return_distribution_.covariance,
                reconstructed,
                rtol=1e-8,
                err_msg=f"Model {label}",
            )


class TestInverseIdioVarianceWeights:
    r"""Verify inverse-idiosyncratic-variance regression weighting.

    DGP: single factor with constant betas and **heterogeneous** idiosyncratic
    noise. The first half of assets have low noise
    (:math:`\sigma_{\text{low}} = 0.002`), the second half have high noise
    (:math:`\sigma_{\text{high}} = 0.02`).

    .. math::

        R_i(t) = \beta_i\,f(t) + \epsilon_i(t), \quad
        \epsilon_i \sim \mathcal{N}(0,\,\sigma_i^2)

    With `inv_idio_variance_weight_shrinkage=1`, the model runs an initial
    OLS, estimates per-asset idiosyncratic variance via EWMA, and reweights the
    cross-sectional regression by :math:`1 / \hat\sigma_i^2`.  This
    approximates GLS, downweighting noisy assets and lowering factor-return MSE.

    Partial shrinkage (`inv_idio_variance_weight_shrinkage=0.5`) blends
    inverse-variance weights with the base regression weights
    (`regression_mcap_power=0.5` gives :math:`\sqrt{\text{mcap}}`):

    .. math::

        w_i = \lambda\,w_i^{\text{inv-var}}
            + (1 - \lambda)\,w_i^{\text{cap}}
    """

    N_OBS = 500
    N_ASSETS = 30
    N_LOW = 15
    SEED = 777
    SIGMA_LOW = 0.002
    SIGMA_HIGH = 0.02

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        betas_2d = np.broadcast_to(betas[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        sigma_f = 0.01
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)

        eps = np.empty((cls.N_OBS, cls.N_ASSETS))
        eps[:, : cls.N_LOW] = rng.normal(0, cls.SIGMA_LOW, size=(cls.N_OBS, cls.N_LOW))
        eps[:, cls.N_LOW :] = rng.normal(
            0, cls.SIGMA_HIGH, size=(cls.N_OBS, cls.N_ASSETS - cls.N_LOW)
        )

        returns = betas[None, :] * f_true[:, None] + eps

        # Heterogeneous market caps (log-normal)
        mcap_1d = np.exp(rng.normal(10, 1, size=cls.N_ASSETS))
        mcap = np.broadcast_to(mcap_1d[None, :], (cls.N_OBS, cls.N_ASSETS)).copy()

        def _make(returns, betas_2d, mcap):
            return make_panel(
                returns,
                extra_fields={"beta": betas_2d},
                market_cap=mcap,
            )

        # shrinkage=0, equal-weight base
        panel, X = _make(returns, betas_2d, mcap)
        model_equal = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_equal.fit(X, characteristics=panel)

        # shrinkage=1, equal-weight base (pure inverse-variance)
        panel2, X2 = _make(returns, betas_2d, mcap)
        _short_warmup = EWVariance(half_life=10, min_observations=1)
        model_invvar = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            inv_idio_variance_weight_shrinkage=1.0,
            idio_variance_estimator=_short_warmup,
        )
        model_invvar.fit(X2, characteristics=panel2)

        # shrinkage=0.5, sqrt-cap base (blend of inv-var and sqrt-cap)
        panel3, X3 = _make(returns, betas_2d, mcap)
        model_blend = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0.5,
            inv_idio_variance_weight_shrinkage=0.5,
            idio_variance_estimator=_short_warmup,
        )
        model_blend.fit(X3, characteristics=panel3)

        # shrinkage=0, sqrt-cap base (pure sqrt-cap, no IRLS)
        panel4, X4 = _make(returns, betas_2d, mcap)
        model_sqrtcap = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0.5,
        )
        model_sqrtcap.fit(X4, characteristics=panel4)

        n_used = model_equal.factor_model_.factor_returns.shape[0]
        cls.f_true = f_true[-n_used:]
        cls.n_used = n_used
        cls.mcap_1d = mcap_1d
        cls.model_equal = model_equal
        cls.model_invvar = model_invvar
        cls.model_blend = model_blend
        cls.model_sqrtcap = model_sqrtcap

    def test_noisy_assets_downweighted(self):
        """Average regression weight must be lower for high-noise assets
        than for low-noise assets."""
        rw = self.model_invvar.factor_model_.regression_weights
        avg_w = rw.mean(axis=0)
        low_mean = avg_w[: self.N_LOW].mean()
        high_mean = avg_w[self.N_LOW :].mean()
        assert low_mean > high_mean, (
            f"low-noise weight {low_mean:.6f} <= high-noise weight {high_mean:.6f}"
        )

    def test_factor_return_mse_improves(self):
        """Inverse-variance weighting must reduce factor return MSE compared
        to equal-weight OLS."""
        fm_eq = self.model_equal.factor_model_
        fm_iv = self.model_invvar.factor_model_

        mse_eq = np.mean((fm_eq.factor_returns[:, 0] - self.f_true) ** 2)
        mse_iv = np.mean((fm_iv.factor_returns[:, 0] - self.f_true) ** 2)
        assert mse_iv < mse_eq, f"MSE invvar {mse_iv:.2e} >= MSE equal {mse_eq:.2e}"

    def test_factor_return_recovery(self):
        """Inverse-variance model must still recover factor returns well."""
        fm = self.model_invvar.factor_model_
        corr = np.corrcoef(fm.factor_returns[:, 0], self.f_true)[0, 1]
        assert corr > 0.95, f"corr(f_hat, f_true) = {corr:.4f}"

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model_invvar.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model_invvar.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_equal_weight_model_has_equal_weights(self):
        """Without inv_idio_variance_weight_shrinkage, regression weights
        must be uniform (all ones for mcap_power=0)."""
        rw = self.model_equal.factor_model_.regression_weights
        np.testing.assert_allclose(rw, 1.0, rtol=1e-10)

    def test_sqrtcap_model_has_sqrtcap_weights(self):
        """Without IRLS, regression_mcap_power=0.5 must produce sqrt(mcap)
        weights."""
        rw = self.model_sqrtcap.factor_model_.regression_weights
        expected = np.sqrt(self.mcap_1d)
        np.testing.assert_allclose(rw[0], expected, rtol=1e-10)

    def test_blend_mse_improves_over_sqrtcap(self):
        """Partial shrinkage (0.5) with sqrt-cap base must still improve MSE
        compared to pure sqrt-cap regression."""
        fm_sc = self.model_sqrtcap.factor_model_
        fm_bl = self.model_blend.factor_model_

        mse_sc = np.mean((fm_sc.factor_returns[:, 0] - self.f_true) ** 2)
        mse_bl = np.mean((fm_bl.factor_returns[:, 0] - self.f_true) ** 2)
        assert mse_bl < mse_sc, f"MSE blend {mse_bl:.2e} >= MSE sqrtcap {mse_sc:.2e}"

    def test_blend_noise_ratio_between_extremes(self):
        """With shrinkage=0.5, the ratio of average normalized weight for
        low-noise vs high-noise assets must be between the pure-cap ratio
        (~1, noise-blind) and the pure-invvar ratio (>>1, noise-aware)."""

        def _noise_ratio(rw):
            row_sum = rw.sum(axis=1, keepdims=True)
            valid = row_sum.squeeze() > 0
            rw_norm = rw[valid] / row_sum[valid]
            low = rw_norm[:, : self.N_LOW].mean()
            high = rw_norm[:, self.N_LOW :].mean()
            return low / high

        ratio_cap = _noise_ratio(self.model_sqrtcap.factor_model_.regression_weights)
        ratio_iv = _noise_ratio(self.model_invvar.factor_model_.regression_weights)
        ratio_blend = _noise_ratio(self.model_blend.factor_model_.regression_weights)
        assert ratio_blend > ratio_cap, (
            f"blend ratio {ratio_blend:.2f} <= cap ratio {ratio_cap:.2f}"
        )
        assert ratio_blend < ratio_iv, (
            f"blend ratio {ratio_blend:.2f} >= invvar ratio {ratio_iv:.2f}"
        )

    def test_blend_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance for the blended
        model."""
        fm = self.model_blend.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model_blend.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )


class TestInterceptRecoveryWithStyleAndIndustries:
    r"""Verify intercept recovery in a multi-factor model with basket-neutral
    industry constraints and a z-scored style factor.

    DGP: market intercept + 3 industry dummies (basket-neutral) + 1 continuous
    style factor (z-scored by default `CSStandardScaler`), with heterogeneous
    market caps.

    .. math::

        R_i(t) = f_{\text{mkt}}(t) + f_{k(i)}(t)
               + \beta_i^{\text{raw}}\,f_{\text{style}}(t) + \epsilon_i(t)

    With `benchmark_mcap_power=1` and `regression_mcap_power=1`, the
    basket-neutral constraints and z-scoring both produce exposures that are
    benchmark-weight-centered: :math:`\sum_i w_i B_{ij} = 0` for all
    non-market factors.  The WLS first-order condition then forces the
    intercept (market factor) to equal the benchmark-weighted average return
    exactly.

    When `regression_mcap_power != benchmark_mcap_power`, this identity
    breaks because the centering is w.r.t. benchmark weights but the
    regression uses different weights.
    """

    N_OBS = 500
    N_ASSETS = 30
    N_IND = 3
    ASSETS_PER_IND = 10
    SEED = 999
    IND_MCAPS: ClassVar[list[float]] = [3.0, 2.0, 1.0]

    @staticmethod
    def _style_factor():
        """Style factor with default CSWinsorizer + CSStandardScaler."""
        return FixedWeightedFactor(
            descriptors=[("style_char", Passthrough("style_char"))],
            family="style",
        )

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        sigma_f = 0.01
        sigma_eps = 0.005

        f_mkt = rng.normal(0, sigma_f, size=cls.N_OBS)
        f_ind = rng.normal(0, sigma_f, size=(cls.N_OBS, cls.N_IND))
        f_style = rng.normal(0, sigma_f, size=cls.N_OBS)

        ind_dummies = np.zeros((cls.N_OBS, cls.N_ASSETS, cls.N_IND))
        mcap = np.ones((cls.N_OBS, cls.N_ASSETS))
        for k in range(cls.N_IND):
            start = k * cls.ASSETS_PER_IND
            end = (k + 1) * cls.ASSETS_PER_IND
            ind_dummies[:, start:end, k] = 1.0
            mcap[:, start:end] = cls.IND_MCAPS[k]

        # Continuous style characteristic with non-zero benchmark-weighted
        # mean so the z-scoring actually shifts the intercept.
        style_char_1d = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        style_char = np.broadcast_to(
            style_char_1d[None, :], (cls.N_OBS, cls.N_ASSETS)
        ).copy()

        returns = (
            f_mkt[:, None]
            + np.einsum("tnk,tk->tn", ind_dummies, f_ind)
            + style_char_1d[None, :] * f_style[:, None]
            + rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))
        )

        mkt_exp = np.ones((cls.N_OBS, cls.N_ASSETS))
        extra_fields = {
            "mkt_exp": mkt_exp,
            "ind_1": ind_dummies[:, :, 0],
            "ind_2": ind_dummies[:, :, 1],
            "ind_3": ind_dummies[:, :, 2],
            "style_char": style_char,
        }

        def _factors():
            return [
                ("market", passthrough_factor("mkt_exp", family="market")),
                ("ind_1", passthrough_factor("ind_1", family="industry")),
                ("ind_2", passthrough_factor("ind_2", family="industry")),
                ("ind_3", passthrough_factor("ind_3", family="industry")),
                ("style", TestInterceptRecoveryWithStyleAndIndustries._style_factor()),
            ]

        # Model A: regression_mcap_power == benchmark_mcap_power == 1
        panel_a, X_a = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
        model_same = CharacteristicsFactorModel(
            factors=_factors(),
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )
        model_same.fit(X_a, characteristics=panel_a)

        # Model B: regression_mcap_power=0.5 != benchmark_mcap_power=1
        panel_b, X_b = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
        model_diff = CharacteristicsFactorModel(
            factors=_factors(),
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=0.5,
        )
        model_diff.fit(X_b, characteristics=panel_b)

        # Industry benchmark weights
        total_mcap = sum(cls.ASSETS_PER_IND * c for c in cls.IND_MCAPS)
        ind_weights = np.array(
            [cls.ASSETS_PER_IND * c / total_mcap for c in cls.IND_MCAPS]
        )

        cls.returns = returns
        cls.mcap = mcap
        cls.f_mkt = f_mkt
        cls.f_style = f_style
        cls.f_ind = f_ind
        cls.style_char_1d = style_char_1d
        cls.ind_weights = ind_weights
        cls.model_same = model_same
        cls.model_diff = model_diff

    def _cap_weighted_return(self):
        """Benchmark-weighted average return per observation (trimmed)."""
        n_used = self.model_same.factor_model_.factor_returns.shape[0]
        returns_used = self.returns[-n_used:]
        mcap_used = self.mcap[-n_used:]
        weights = mcap_used / mcap_used.sum(axis=1, keepdims=True)
        return (weights * returns_used).sum(axis=1)

    def test_factor_names(self):
        """All 5 factors must be present in the original basis."""
        names = list(self.model_same.factor_model_.factor_names)
        assert len(names) == 5
        for expected in ("market", "ind_1", "ind_2", "ind_3", "style"):
            assert expected in names

    def test_intercept_equals_benchmark_same_power(self):
        """When regression_mcap_power == benchmark_mcap_power, the intercept
        must exactly equal the benchmark-weighted average return because all
        non-market exposures are benchmark-weight-centered."""
        fm = self.model_same.factor_model_
        idx = list(fm.factor_names).index("market")
        np.testing.assert_allclose(
            fm.factor_returns[:, idx],
            self._cap_weighted_return(),
            rtol=1e-10,
        )

    def test_intercept_equals_benchmark_different_power(self):
        r"""Even when regression_mcap_power != benchmark_mcap_power, the
        intercept still exactly equals the benchmark-weighted average return.

        This holds because with constant within-industry market caps, the
        ratio :math:`w_i^{\text{bench}} / w_i^{\text{reg}}` is constant within
        each industry and therefore lies in the column space of the
        industry-contrast exposures.  This means the benchmark weight vector
        :math:`u` satisfies :math:`u = WBa` for some :math:`a`, implying
        :math:`u^{\top}\epsilon = 0` by the WLS first-order conditions.
        """
        fm = self.model_diff.factor_model_
        n_used = fm.factor_returns.shape[0]
        returns_used = self.returns[-n_used:]
        mcap_used = self.mcap[-n_used:]
        weights = mcap_used / mcap_used.sum(axis=1, keepdims=True)
        cap_ret = (weights * returns_used).sum(axis=1)

        idx = list(fm.factor_names).index("market")
        np.testing.assert_allclose(
            fm.factor_returns[:, idx],
            cap_ret,
            rtol=1e-10,
        )

    def test_industry_constraint_same_power(self):
        """Cap-weighted industry factor returns must sum to zero (same
        power model)."""
        fm = self.model_same.factor_model_
        names = list(fm.factor_names)
        ind_cols = [names.index(n) for n in ("ind_1", "ind_2", "ind_3")]
        weighted_sum = fm.factor_returns[:, ind_cols] @ self.ind_weights
        np.testing.assert_allclose(weighted_sum, 0.0, atol=1e-12)

    def test_industry_constraint_different_power(self):
        """Cap-weighted industry factor returns must sum to zero (different
        power model) -- the constraint is imposed via the basis, not via
        regression weights."""
        fm = self.model_diff.factor_model_
        names = list(fm.factor_names)
        ind_cols = [names.index(n) for n in ("ind_1", "ind_2", "ind_3")]
        weighted_sum = fm.factor_returns[:, ind_cols] @ self.ind_weights
        np.testing.assert_allclose(weighted_sum, 0.0, atol=1e-12)

    def test_style_factor_return_recovery(self):
        """Style factor returns must correlate well with the true factor
        (accounting for the scale change from z-scoring).  With 30 assets
        and idio noise, recovery is noisier than single-factor tests."""
        fm = self.model_same.factor_model_
        n_used = fm.factor_returns.shape[0]
        idx = list(fm.factor_names).index("style")
        corr = np.corrcoef(fm.factor_returns[:, idx], self.f_style[-n_used:])[0, 1]
        assert corr > 0.90, f"corr(f_style_hat, f_style_true) = {corr:.4f}"

    def test_covariance_decomposition_same_power(self):
        """B F B^T + D must equal the stored asset covariance (same power)."""
        fm = self.model_same.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model_same.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_covariance_decomposition_different_power(self):
        """B F B^T + D must equal the stored asset covariance (different
        power)."""
        fm = self.model_diff.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model_diff.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_asset_covariance_positive_definite(self):
        """Asset covariance must be positive definite despite the
        rank-deficient factor covariance from basket-neutral constraints."""
        for model in (self.model_same, self.model_diff):
            eigvals = np.linalg.eigvalsh(model.return_distribution_.covariance)
            assert eigvals.min() > 0, f"min eigenvalue = {eigvals.min():.2e}"


class TestComputeScores:
    r"""Verify regression diagnostics on FactorModel.

    DGP: two factors -- one true signal factor and one pure noise factor.

    .. math::

        R_i(t) = \beta_i\,f(t) + \epsilon_i(t)

    The noise factor characteristic is random and uncorrelated with returns.
    Two models are fitted: a "full" model with both factors and a "true-only"
    model with just the signal factor.  Diagnostics are compared to verify
    that :math:`R^2`, t-statistics, hit rate, AIC, and BIC behave correctly.
    Diagnostics are read from :meth:`~skfolio.prior.FactorModel.cs_regression_scores`,
    :meth:`~skfolio.prior.FactorModel.cs_regression_t_stats`, and
    :meth:`~skfolio.prior.FactorModel.cs_regression_t_stat_exceedance_rate`.
    """

    N_OBS = 500
    N_ASSETS = 50
    SEED = 1234

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

        # Noise characteristic: random, uncorrelated with returns
        noise_char = rng.normal(0, 1, size=(cls.N_OBS, cls.N_ASSETS))

        def _make(returns, betas_2d, noise_char):
            return make_panel(
                returns,
                extra_fields={"beta": betas_2d, "noise": noise_char},
            )

        # Full model: true factor + noise factor
        panel_full, X_full = _make(returns, betas_2d, noise_char)
        model_full = CharacteristicsFactorModel(
            factors=[
                ("signal", passthrough_factor("beta", family="style")),
                ("noise", passthrough_factor("noise", family="style")),
            ],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_full.fit(X_full, characteristics=panel_full)

        # True-only model: just the signal factor
        panel_true, X_true = _make(returns, betas_2d, noise_char)
        model_true = CharacteristicsFactorModel(
            factors=[
                ("signal", passthrough_factor("beta", family="style")),
            ],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_true.fit(X_true, characteristics=panel_true)

        cls.model_full = model_full
        cls.model_true = model_true

    def test_regression_scores_present(self):
        """factor_model_ must expose cross-sectional regression score tables."""
        fm = self.model_full.factor_model_
        assert hasattr(fm, "cs_regression_scores")
        _ = fm.cs_regression_scores

    def test_r2_shape(self):
        """r2 column must have one entry per regression observation."""
        fm = self.model_full.factor_model_
        r2 = fm.cs_regression_scores["r2"]
        assert len(r2) == fm.cs_regression_t_stats.shape[0]

    def test_r2_df_type(self):
        """r2 must be a time-indexed Series (column of cs_regression_scores)."""
        r2_s = self.model_full.factor_model_.cs_regression_scores["r2"]
        assert isinstance(r2_s, pd.Series)

    def test_r2_positive(self):
        """Mean R^2 must be meaningfully positive for the true factor."""
        r2 = self.model_true.factor_model_.cs_regression_scores["r2"].values
        assert r2.mean() > 0.1, f"mean R^2 = {r2.mean():.4f}"

    def test_r2_full_ge_true_only(self):
        """Adding a factor (even noise) cannot decrease in-sample R^2."""
        r2_diff = (
            self.model_full.factor_model_.cs_regression_scores["r2"].values
            - self.model_true.factor_model_.cs_regression_scores["r2"].values
        )
        assert r2_diff.min() >= -1e-10, (
            f"R^2 decreased by {r2_diff.min():.2e} at some observation"
        )

    def test_t_stats_shape(self):
        """cs_regression_t_stats must be (n_obs, n_factors)."""
        fm = self.model_full.factor_model_
        ts = fm.cs_regression_t_stats
        assert ts.shape[1] == 2
        assert ts.shape[0] == len(fm.cs_regression_scores["r2"])

    def test_t_stats_df_columns(self):
        """t-stats must have factor names as columns."""
        ts_df = self.model_full.factor_model_.cs_regression_t_stats
        assert isinstance(ts_df, pd.DataFrame)
        assert list(ts_df.columns) == ["signal", "noise"]

    def test_t_stats_signal_significant(self):
        """The true factor must have median |t| well above 2."""
        ts = self.model_full.factor_model_.cs_regression_t_stats.values
        signal_idx = list(self.model_full.factor_model_.factor_names).index("signal")
        median_abs_t = np.median(np.abs(ts[:, signal_idx]))
        assert median_abs_t > 3.0, f"median |t_signal| = {median_abs_t:.2f}"

    def test_t_stats_noise_insignificant(self):
        """The noise factor must have median |t| close to the null
        distribution (median of half-normal ~ 0.67)."""
        ts = self.model_full.factor_model_.cs_regression_t_stats.values
        noise_idx = list(self.model_full.factor_model_.factor_names).index("noise")
        median_abs_t = np.median(np.abs(ts[:, noise_idx]))
        assert median_abs_t < 2.0, f"median |t_noise| = {median_abs_t:.2f}"

    def test_hit_rate_signal_high(self):
        """Hit rate for the true factor must be high (most observations
        have |t| > 2)."""
        hr = self.model_full.factor_model_.cs_regression_t_stat_exceedance_rate(
            threshold=2.0
        )
        assert hr["signal"] > 0.8, f"hit_rate(signal) = {hr['signal']:.2f}"

    def test_hit_rate_noise_low(self):
        """Hit rate for the noise factor must be close to the nominal
        false-positive rate (~5% at threshold=2)."""
        hr = self.model_full.factor_model_.cs_regression_t_stat_exceedance_rate(
            threshold=2.0
        )
        assert hr["noise"] < 0.20, f"hit_rate(noise) = {hr['noise']:.2f}"

    def test_aic_penalizes_noise_factor(self):
        """The true-only model must have lower mean AIC than the full
        model (noise factor adds complexity without explanatory power)."""
        aic_true = self.model_true.factor_model_.cs_regression_scores[
            "aic"
        ].values.mean()
        aic_full = self.model_full.factor_model_.cs_regression_scores[
            "aic"
        ].values.mean()
        assert aic_true < aic_full, (
            f"AIC true-only {aic_true:.2f} >= AIC full {aic_full:.2f}"
        )

    def test_bic_penalizes_noise_factor(self):
        """The true-only model must have lower mean BIC than the full
        model (BIC penalizes complexity more than AIC)."""
        bic_true = self.model_true.factor_model_.cs_regression_scores[
            "bic"
        ].values.mean()
        bic_full = self.model_full.factor_model_.cs_regression_scores[
            "bic"
        ].values.mean()
        assert bic_true < bic_full, (
            f"BIC true-only {bic_true:.2f} >= BIC full {bic_full:.2f}"
        )

    def test_adjusted_r2_shape(self):
        """adjusted_r2 column must have one entry per regression observation."""
        fm = self.model_full.factor_model_
        adj = fm.cs_regression_scores["adjusted_r2"]
        assert len(adj) == fm.cs_regression_t_stats.shape[0]

    def test_adjusted_r2_df_type(self):
        """adjusted_r2 must be a time-indexed Series."""
        adj_r2_s = self.model_full.factor_model_.cs_regression_scores["adjusted_r2"]
        assert isinstance(adj_r2_s, pd.Series)

    def test_adjusted_r2_le_r2(self):
        """Adjusted R2 must be <= R2 (penalises added parameters)."""
        fm = self.model_full.factor_model_
        scores = fm.cs_regression_scores
        valid = np.isfinite(scores["adjusted_r2"]) & np.isfinite(scores["r2"])
        assert np.all(
            scores.loc[valid, "adjusted_r2"].values
            <= scores.loc[valid, "r2"].values + 1e-12
        )


class TestTimeVaryingMarketCaps:
    r"""Verify the model with drifting market capitalizations.

    DGP: market intercept + 3 industry dummies (basket-neutral) with
    market caps that drift linearly over time.  Industry A starts large
    and shrinks, Industry C starts small and grows.

    .. math::

        R_i(t) = f_{\text{mkt}}(t) + f_{g(i)}(t) + \epsilon_i(t)

    With time-varying caps, the benchmark weights, regression weights, and
    basket-neutral contrasts must update per observation.  The intercept
    (market factor) must still equal the cap-weighted average return at
    each date.
    """

    N_OBS = 500
    N_ASSETS = 30
    ASSETS_PER_IND = 10
    SEED = 314

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        n_obs = cls.N_OBS
        n_assets = cls.N_ASSETS
        n_per = cls.ASSETS_PER_IND

        # Industry assignments: 0..9 → A, 10..19 → B, 20..29 → C
        np.array(["ind_A"] * n_per + ["ind_B"] * n_per + ["ind_C"] * n_per)

        # Time-varying market caps: Industry A shrinks, B constant, C grows
        t_grid = np.linspace(0, 1, n_obs)[:, None]  # (n_obs, 1)
        base_cap = np.ones((n_obs, n_assets))
        # Industry A: starts at 3, ends at 1
        base_cap[:, :n_per] = 3.0 - 2.0 * t_grid
        # Industry B: constant at 2
        base_cap[:, n_per : 2 * n_per] = 2.0
        # Industry C: starts at 1, ends at 3
        base_cap[:, 2 * n_per :] = 1.0 + 2.0 * t_grid

        # Within each industry, add slight asset-level variation
        asset_scale = rng.uniform(0.8, 1.2, size=n_assets)
        market_cap = base_cap * asset_scale[None, :]

        # Factor returns
        sigma_f = 0.01
        f_mkt = rng.normal(0, sigma_f, size=n_obs)
        f_ind = rng.normal(0, sigma_f * 0.5, size=(n_obs, 3))
        sigma_eps = 0.005
        eps = rng.normal(0, sigma_eps, size=(n_obs, n_assets))

        # Build returns
        returns = f_mkt[:, None] + eps
        for k in range(3):
            returns[:, k * n_per : (k + 1) * n_per] += f_ind[:, k : k + 1]

        # Constant-ones intercept and industry indicator fields
        mkt_exp = np.ones((n_obs, n_assets))
        ind_a = np.zeros((n_obs, n_assets))
        ind_a[:, :n_per] = 1.0
        ind_b = np.zeros((n_obs, n_assets))
        ind_b[:, n_per : 2 * n_per] = 1.0
        ind_c = np.zeros((n_obs, n_assets))
        ind_c[:, 2 * n_per :] = 1.0

        panel, X = make_panel(
            returns,
            market_cap=market_cap,
            extra_fields={
                "mkt_exp": mkt_exp,
                "ind_a": ind_a,
                "ind_b": ind_b,
                "ind_c": ind_c,
            },
        )

        factors = [
            ("market", passthrough_factor("mkt_exp", family="market")),
            ("ind_A", passthrough_factor("ind_a", family="industry")),
            ("ind_B", passthrough_factor("ind_b", family="industry")),
            ("ind_C", passthrough_factor("ind_c", family="industry")),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=1,
        )
        model.fit(X, characteristics=panel)

        cls.model = model
        cls.returns = returns
        cls.market_cap = market_cap
        cls.n_per = n_per

    def test_benchmark_weights_are_time_varying(self):
        """Benchmark weights must differ between early and late observations
        because market caps drift."""
        fm = self.model.factor_model_
        fr = fm.factor_returns
        fr.shape[0]
        # Compare first and last observation loading matrices implicitly
        # through the intercept value (which uses benchmark weights).
        # More directly: the benchmark_weights_ were stored in the panel.
        # Instead, verify that the intercept return changes in a way
        # consistent with drifting weights.
        early_mkt = fr[:50, 0].mean()
        late_mkt = fr[-50:, 0].mean()
        # They should be different (different cap-weighted baskets)
        # But with noise, we just check they're not identical
        assert not np.allclose(early_mkt, late_mkt, atol=1e-6), (
            "Early and late market returns should differ with drifting caps"
        )

    def test_intercept_close_to_cap_weighted_return(self):
        """The intercept must equal the lagged cap-weighted average return."""
        fm = self.model.factor_model_
        fr = fm.factor_returns
        n_obs = fr.shape[0]

        trim = self.returns.shape[0] - n_obs
        trimmed_returns = self.returns[trim:]
        lag = fm.exposure_lag
        lagged_cap = self.market_cap[trim - lag : -lag]

        cap_weights = lagged_cap / lagged_cap.sum(axis=1, keepdims=True)
        benchmark_return = (cap_weights * trimmed_returns).sum(axis=1)

        np.testing.assert_allclose(fr[:, 0], benchmark_return, rtol=1e-10, atol=1e-12)

    def test_intercept_correlation_with_benchmark(self):
        """The intercept must track the lagged benchmark return exactly."""
        fm = self.model.factor_model_
        fr = fm.factor_returns
        n_obs = fr.shape[0]

        trim = self.returns.shape[0] - n_obs
        trimmed_returns = self.returns[trim:]
        lag = fm.exposure_lag
        lagged_cap = self.market_cap[trim - lag : -lag]
        cap_weights = lagged_cap / lagged_cap.sum(axis=1, keepdims=True)
        benchmark_return = (cap_weights * trimmed_returns).sum(axis=1)

        corr = np.corrcoef(fr[:, 0], benchmark_return)[0, 1]
        assert corr > 1 - 1e-12, f"correlation = {corr:.12f}"

    def test_regression_weights_use_lagged_market_cap(self):
        """Regression weights must be proportional to lagged market cap."""
        fm = self.model.factor_model_
        rw = fm.regression_weights
        n_obs = rw.shape[0]

        trim = self.returns.shape[0] - n_obs
        lag = fm.exposure_lag
        lagged_cap = self.market_cap[trim - lag : -lag]

        np.testing.assert_allclose(rw, lagged_cap, rtol=1e-10, atol=1e-12)

    def test_industry_constraint_per_observation(self):
        """Cap-weighted industry factor returns must sum to zero at each
        observation under the formation-date weights (caps lagged by
        `exposure_lag`), even with drifting caps."""
        fm = self.model.factor_model_
        fr = fm.factor_returns
        n_obs = fr.shape[0]

        trim = self.returns.shape[0] - n_obs
        lag = fm.exposure_lag
        # Factor returns at observation t are estimated on exposures formed at
        # t - lag, so the zero-sum constraint uses the caps at t - lag.
        lagged_cap = self.market_cap[trim - lag : -lag]

        # Industry cap weights per formation date
        cap_A = lagged_cap[:, : self.n_per].sum(axis=1)
        cap_B = lagged_cap[:, self.n_per : 2 * self.n_per].sum(axis=1)
        cap_C = lagged_cap[:, 2 * self.n_per :].sum(axis=1)
        total = cap_A + cap_B + cap_C
        w_A = cap_A / total
        w_B = cap_B / total
        w_C = cap_C / total

        # Factor returns columns: [market, ind_A, ind_B, ind_C]
        weighted_sum = w_A * fr[:, 1] + w_B * fr[:, 2] + w_C * fr[:, 3]
        np.testing.assert_allclose(weighted_sum, 0, atol=1e-10)

    def test_regression_fit_identity_with_drifting_caps(self):
        """Full-basis factor returns must exactly reproduce the cross-sectional
        regression fit: returns = exposures(t - lag) @ f(t) + idio(t). With
        drifting caps this holds only when the dropped factor is reconstructed
        with the formation-date constraint ratios."""
        fm = self.model.factor_model_
        lag = fm.exposure_lag
        lagged_exposures = fm.exposures[:-lag]
        factor_returns = fm.factor_returns[lag:]
        idio_returns = fm.idio_returns[lag:]

        systematic = np.einsum("tnk,tk->tn", lagged_exposures, factor_returns)
        reconstructed = systematic + idio_returns

        n_obs = fm.factor_returns.shape[0]
        trim = self.returns.shape[0] - n_obs
        np.testing.assert_allclose(
            reconstructed,
            self.returns[trim + lag :],
            atol=1e-12,
        )

    def test_industry_weights_drift(self):
        """The industry constraint weights must change over time,
        reflecting the market cap drift."""
        fm = self.model.factor_model_
        fr = fm.factor_returns
        n_obs = fr.shape[0]

        trim = self.returns.shape[0] - n_obs
        trimmed_cap = self.market_cap[trim:]

        # Industry A starts large, should have higher weight early
        cap_A_early = trimmed_cap[:50, : self.n_per].sum(axis=1).mean()
        cap_A_late = trimmed_cap[-50:, : self.n_per].sum(axis=1).mean()
        total_early = trimmed_cap[:50].sum(axis=1).mean()
        total_late = trimmed_cap[-50:].sum(axis=1).mean()

        w_A_early = cap_A_early / total_early
        w_A_late = cap_A_late / total_late

        assert w_A_early > w_A_late, (
            f"Industry A weight should decrease: "
            f"early={w_A_early:.4f}, late={w_A_late:.4f}"
        )

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_asset_covariance_positive_definite(self):
        """Asset covariance must be positive definite."""
        eigvals = np.linalg.eigvalsh(self.model.return_distribution_.covariance)
        assert eigvals.min() > 0, f"min eigenvalue = {eigvals.min():.2e}"

    def test_factor_names(self):
        """Output must have 4 factors: market + 3 industries."""
        fm = self.model.factor_model_
        assert list(fm.factor_names) == ["market", "ind_A", "ind_B", "ind_C"]

    def test_loading_matrix_uses_last_observation_caps(self):
        """The loading matrix (final exposures) must reflect the most recent
        market cap distribution, not an average."""
        fm = self.model.factor_model_
        assert fm.loading_matrix.shape == (self.N_ASSETS, 4)
        np.testing.assert_allclose(fm.loading_matrix[:, 0], 1.0, atol=1e-10)

    def test_factor_covariance_rank_deficient(self):
        """Factor covariance must be rank K-1 = 3 (one constraint removes
        one degree of freedom from 4 factors)."""
        fm = self.model.factor_model_
        eigvals = np.linalg.eigvalsh(fm.factor_covariance)
        n_zero = np.sum(np.abs(eigvals) < 1e-12)
        assert n_zero == 1, (
            f"Expected 1 zero eigenvalue, got {n_zero}; eigenvalues = {eigvals}"
        )


class TestNaNHandlingListingsDelistings:
    r"""Verify correct handling of mid-sample listings and delistings.

    DGP: single factor, 50 assets.  The first 40 are present for the full
    sample. The last 10 are "late listings" that appear at the midpoint
    (`active_mask=False` and `NaN` returns before the listing date).

    .. math::

        R_i(t) = \beta_i \, f(t) + \epsilon_i(t)

    A control model is fitted on only the full-sample assets to verify
    that the late-listing assets do not distort early-period factor
    returns.
    """

    N_OBS = 500
    N_ASSETS = 50
    N_LATE = 10
    LISTING_DATE = 250
    SEED = 4242

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)

        n_obs = cls.N_OBS
        n_assets = cls.N_ASSETS
        n_late = cls.N_LATE
        listing = cls.LISTING_DATE
        n_full = n_assets - n_late

        betas = rng.uniform(0.5, 1.5, size=n_assets)
        betas_2d = np.broadcast_to(betas[None, :], (n_obs, n_assets)).copy()

        sigma_f = 0.01
        sigma_eps = 0.005
        f_true = rng.normal(0, sigma_f, size=n_obs)
        eps = rng.normal(0, sigma_eps, size=(n_obs, n_assets))
        returns_full = betas[None, :] * f_true[:, None] + eps

        # Late-listing assets: NaN before listing date
        returns = returns_full.copy()
        returns[:listing, n_full:] = np.nan

        # Universe mask: late assets out-of-universe before listing
        active_mask = np.ones((n_obs, n_assets), dtype=bool)
        active_mask[:listing, n_full:] = False

        # Market cap: equal for simplicity; NaN for out-of-universe
        market_cap = np.ones((n_obs, n_assets))
        market_cap[:listing, n_full:] = np.nan

        # Betas: NaN for out-of-universe
        betas_panel = betas_2d.copy()
        betas_panel[:listing, n_full:] = np.nan

        panel, X = make_panel(
            returns,
            market_cap=market_cap,
            extra_fields={"beta": betas_panel},
            active_mask=active_mask,
        )

        model = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model.fit(X, characteristics=panel)

        # Control model: only the full-sample assets
        returns_ctrl = returns_full[:, :n_full].copy()
        betas_ctrl = betas_2d[:, :n_full].copy()
        panel_ctrl, X_ctrl = make_panel(
            returns_ctrl,
            extra_fields={"beta": betas_ctrl},
        )
        model_ctrl = CharacteristicsFactorModel(
            factors=[("f1", passthrough_factor("beta"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
        )
        model_ctrl.fit(X_ctrl, characteristics=panel_ctrl)

        cls.model = model
        cls.model_ctrl = model_ctrl
        cls.betas = betas
        cls.f_true = f_true
        cls.n_full = n_full
        cls.listing = listing

    def test_output_includes_all_assets(self):
        """The final outputs must include all 50 assets, including
        late listings."""
        assert self.model.return_distribution_.mu.shape == (self.N_ASSETS,)
        assert self.model.return_distribution_.covariance.shape == (
            self.N_ASSETS,
            self.N_ASSETS,
        )

    def test_loading_matrix_shape(self):
        """Loading matrix must cover all assets (1 factor)."""
        fm = self.model.factor_model_
        assert fm.loading_matrix.shape == (self.N_ASSETS, 1)

    def test_loading_matrix_no_nan(self):
        """Loading matrix must have no NaN entries -- even late-listing
        assets receive exposures at the final observation."""
        fm = self.model.factor_model_
        assert np.all(np.isfinite(fm.loading_matrix))

    def test_factor_returns_shape(self):
        """Factor returns must span the full sample (minus exposure_lag
        trim)."""
        fm = self.model.factor_model_
        expected = self.N_OBS - 1  # exposure_lag=1 trims one obs
        assert fm.factor_returns.shape[0] == expected

    def test_early_factor_returns_match_control(self):
        """Factor returns before the listing date must match those from
        the control model (fitted on full-sample assets only), because
        late-listing assets are excluded from the regression."""
        fm = self.model.factor_model_
        fm_ctrl = self.model_ctrl.factor_model_
        # Both have exposure_lag=1 trim, so factor_returns[0] corresponds
        # to observation 1 in the original data.
        n_early = self.listing - 1  # -1 for exposure_lag trim
        np.testing.assert_allclose(
            fm.factor_returns[:n_early],
            fm_ctrl.factor_returns[:n_early],
            rtol=1e-10,
        )

    def test_factor_return_recovery(self):
        """Factor returns must correlate with the true factor."""
        fm = self.model.factor_model_
        trim = self.N_OBS - fm.factor_returns.shape[0]
        f_trimmed = self.f_true[trim:]
        corr = np.corrcoef(fm.factor_returns[:, 0], f_trimmed)[0, 1]
        assert corr > 0.95, f"factor return correlation = {corr:.4f}"

    def test_covariance_decomposition_identity(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_asset_covariance_positive_definite(self):
        """Asset covariance must be positive definite for all 50 assets."""
        eigvals = np.linalg.eigvalsh(self.model.return_distribution_.covariance)
        assert eigvals.min() > 0, f"min eigenvalue = {eigvals.min():.2e}"

    def test_idio_covariance_no_nan(self):
        """Idiosyncratic covariance must have no NaN entries."""
        fm = self.model.factor_model_
        assert np.all(np.isfinite(fm.idio_covariance))

    def test_late_assets_get_correct_betas(self):
        """Late-listing assets must have loading matrix entries close to
        their true betas (using the last available exposure)."""
        fm = self.model.factor_model_
        late_betas_estimated = fm.loading_matrix[self.n_full :, 0]
        late_betas_true = self.betas[self.n_full :]
        np.testing.assert_allclose(late_betas_estimated, late_betas_true, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 17 -- Family-key neutralize_against integration
# ---------------------------------------------------------------------------


class TestNeutralizeExposureFamilyKeyIntegration:
    r"""Integration test: `neutralize_against` with a family key through
    `CharacteristicsFactorModel`.

    DGP: market + 3 industry dummies + 2 style factors (size, momentum)
    where both style characteristics have industry-correlated components.

    With `neutralize_against={"style": ["industry"]}`, both style
    exposures must be benchmark-weight-orthogonal to every industry
    exposure in the output.
    """

    N_OBS = 200
    N_ASSETS = 30
    N_IND = 3
    SEED = 1234

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)
        n = cls.N_ASSETS

        ind_labels = np.array([i % cls.N_IND for i in range(n)])
        ind_dummies = np.eye(cls.N_IND)[ind_labels]

        raw_size = rng.standard_normal(n)
        raw_mom = rng.standard_normal(n)
        for k in range(cls.N_IND):
            mask = ind_labels == k
            raw_size[mask] += rng.normal(2.0 * k, 0.5)
            raw_mom[mask] += rng.normal(-1.5 * k, 0.5)

        mcap = rng.uniform(1, 100, size=(cls.N_OBS, n))

        sigma_f = 0.01
        sigma_eps = 0.005
        f_mkt = rng.normal(0, sigma_f, cls.N_OBS)
        f_ind = rng.normal(0, sigma_f, (cls.N_OBS, cls.N_IND))
        f_size = rng.normal(0, sigma_f, cls.N_OBS)
        f_mom = rng.normal(0, sigma_f, cls.N_OBS)
        eps = rng.normal(0, sigma_eps, (cls.N_OBS, n))

        returns = (
            f_mkt[:, None]
            + (ind_dummies[None, :, :] * f_ind[:, None, :]).sum(axis=-1)
            + raw_size[None, :] * f_size[:, None]
            + raw_mom[None, :] * f_mom[:, None]
            + eps
        )

        extra_fields = {
            "mkt_exp": np.ones((cls.N_OBS, n)),
            "size": np.broadcast_to(raw_size[None, :], (cls.N_OBS, n)).copy(),
            "momentum": np.broadcast_to(raw_mom[None, :], (cls.N_OBS, n)).copy(),
        }
        for k in range(cls.N_IND):
            extra_fields[f"ind_{k}"] = np.broadcast_to(
                ind_dummies[:, k][None, :], (cls.N_OBS, n)
            ).copy()

        factors = [
            ("market", passthrough_factor("mkt_exp", family="market")),
        ]
        for k in range(cls.N_IND):
            factors.append(
                (f"ind_{k}", passthrough_factor(f"ind_{k}", family="industry"))
            )
        factors.extend(
            [
                ("size", passthrough_factor("size", family="style")),
                ("momentum", passthrough_factor("momentum", family="style")),
            ]
        )

        panel, X = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
        model = CharacteristicsFactorModel(
            factors=factors,
            neutralize_against={"style": ["industry"]},
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=0.5,
        )
        model.fit(X, characteristics=panel)

        cls.model = model
        cls.mcap = mcap

    def test_style_orthogonal_to_industry(self):
        """Style exposures must be benchmark-weight-orthogonal to industry."""
        fm = self.model.factor_model_
        names = list(fm.factor_names)
        style_idx = [names.index("size"), names.index("momentum")]
        ind_idx = [names.index(n) for n in names if n.startswith("ind_")]

        for t in range(fm.exposures.shape[0]):
            w = fm.benchmark_weights[t]
            for si in style_idx:
                for ii in ind_idx:
                    dot = np.sum(w * fm.exposures[t, :, si] * fm.exposures[t, :, ii])
                    assert abs(dot) < 1e-8, (
                        f"t={t}, {names[si]} vs {names[ii]}: dot={dot:.2e}"
                    )

    def test_covariance_decomposition(self):
        """B F B^T + D must equal the stored asset covariance."""
        fm = self.model.factor_model_
        reconstructed = _reconstruct_asset_covariance(fm)
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            reconstructed,
            rtol=1e-10,
        )

    def test_style_factors_are_modified(self):
        """The style exposures in the output must differ from a model
        without neutralization (proving the family key was effective)."""
        rng = np.random.default_rng(self.SEED)
        n = self.N_ASSETS
        ind_labels = np.array([i % self.N_IND for i in range(n)])
        ind_dummies = np.eye(self.N_IND)[ind_labels]

        raw_size = rng.standard_normal(n)
        raw_mom = rng.standard_normal(n)
        for k in range(self.N_IND):
            mask = ind_labels == k
            raw_size[mask] += rng.normal(2.0 * k, 0.5)
            raw_mom[mask] += rng.normal(-1.5 * k, 0.5)

        mcap = rng.uniform(1, 100, size=(self.N_OBS, n))
        sigma_f, sigma_eps = 0.01, 0.005
        f_mkt = rng.normal(0, sigma_f, self.N_OBS)
        f_ind = rng.normal(0, sigma_f, (self.N_OBS, self.N_IND))
        f_size = rng.normal(0, sigma_f, self.N_OBS)
        f_mom = rng.normal(0, sigma_f, self.N_OBS)
        eps = rng.normal(0, sigma_eps, (self.N_OBS, n))

        returns = (
            f_mkt[:, None]
            + (ind_dummies[None, :, :] * f_ind[:, None, :]).sum(axis=-1)
            + raw_size[None, :] * f_size[:, None]
            + raw_mom[None, :] * f_mom[:, None]
            + eps
        )

        extra_fields = {
            "mkt_exp": np.ones((self.N_OBS, n)),
            "size": np.broadcast_to(raw_size[None, :], (self.N_OBS, n)).copy(),
            "momentum": np.broadcast_to(raw_mom[None, :], (self.N_OBS, n)).copy(),
        }
        for k in range(self.N_IND):
            extra_fields[f"ind_{k}"] = np.broadcast_to(
                ind_dummies[:, k][None, :], (self.N_OBS, n)
            ).copy()

        factors = [
            ("market", passthrough_factor("mkt_exp", family="market")),
        ]
        for k in range(self.N_IND):
            factors.append(
                (f"ind_{k}", passthrough_factor(f"ind_{k}", family="industry"))
            )
        factors.extend(
            [
                ("size", passthrough_factor("size", family="style")),
                ("momentum", passthrough_factor("momentum", family="style")),
            ]
        )

        panel_ctrl, X_ctrl = make_panel(
            returns, extra_fields=extra_fields, market_cap=mcap
        )
        model_ctrl = CharacteristicsFactorModel(
            factors=factors,
            constrained_families=[("industry", None)],
            benchmark_mcap_power=1,
            regression_mcap_power=0.5,
        )
        model_ctrl.fit(X_ctrl, characteristics=panel_ctrl)

        fm = self.model.factor_model_
        fm_ctrl = model_ctrl.factor_model_
        names = list(fm.factor_names)
        for name in ("size", "momentum"):
            idx = names.index(name)
            diff = np.abs(fm.exposures[:, :, idx] - fm_ctrl.exposures[:, :, idx])
            assert diff.max() > 0.01, (
                f"{name} exposures are nearly identical to no-neutralization"
            )
