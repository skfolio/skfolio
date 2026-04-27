"""Tests for skfolio.factor_model.utils._basis_transform."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model._family_constraint_basis import (
    ConstrainedFamily,
    FamilyConstraintBasis,
    compute_family_constraint_basis,
)


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def _industry_data(rng, *, T=50, N=30, n_industries=3):
    """Build a simple intercept + industry + style factor setup."""
    K = 1 + n_industries + 1  # mkt + industries + style
    factor_names = np.array(
        ["mkt"] + [f"ind{i}" for i in range(n_industries)] + ["style"]
    )
    factor_families = np.array(["market"] + ["industry"] * n_industries + ["style"])

    exposures = rng.standard_normal((T, N, K))
    exposures[:, :, 0] = 1.0
    for i in range(n_industries):
        group = slice(i * (N // n_industries), (i + 1) * (N // n_industries))
        exposures[:, group, 1 + i] = 1.0
        for j in range(n_industries):
            if j != i:
                exposures[:, group, 1 + j] = 0.0

    bench_w = np.ones((T, N)) / N
    return exposures, bench_w, factor_names, factor_families


# ======================================================================
# ConstrainedFamily
# ======================================================================
class TestConstrainedFamily:
    def test_basic_properties(self):
        constrained_family = ConstrainedFamily(
            family_name="industry",
            full_factor_indices=np.array([1, 2, 3]),
            dropped_index_in_family=0,
            basis_coefficients=np.zeros((4, 2)),
        )
        assert constrained_family.family_size == 3
        assert constrained_family.dropped_full_index == 1
        np.testing.assert_array_equal(constrained_family.kept_full_indices, [2, 3])
        np.testing.assert_array_equal(constrained_family.kept_local_indices, [1, 2])

    def test_dropped_middle(self):
        constrained_family = ConstrainedFamily(
            family_name="sector",
            full_factor_indices=np.array([5, 7, 9, 11]),
            dropped_index_in_family=2,
            basis_coefficients=np.zeros((4, 3)),
        )
        assert constrained_family.dropped_full_index == 9
        np.testing.assert_array_equal(constrained_family.kept_full_indices, [5, 7, 11])
        np.testing.assert_array_equal(constrained_family.kept_local_indices, [0, 1, 3])

    def test_dropped_last(self):
        constrained_family = ConstrainedFamily(
            family_name="country",
            full_factor_indices=np.array([10, 20]),
            dropped_index_in_family=1,
            basis_coefficients=np.zeros((4, 1)),
        )
        assert constrained_family.dropped_full_index == 20
        np.testing.assert_array_equal(constrained_family.kept_full_indices, [10])

    def test_frozen(self):
        constrained_family = ConstrainedFamily(
            family_name="x",
            full_factor_indices=np.array([0, 1]),
            dropped_index_in_family=0,
            basis_coefficients=np.zeros((4, 1)),
        )
        with pytest.raises(AttributeError):
            constrained_family.family_name = "y"


# ======================================================================
# FamilyConstraintBasis — structural properties
# ======================================================================
class TestFamilyConstraintBasisProperties:
    @pytest.fixture()
    def basis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20, N=12)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        return basis

    def test_dimensions(self, basis):
        assert basis.n_factors == 5
        assert basis.n_constraints == 1
        assert basis.n_factors_reduced == 4
        assert basis.n_observations == 20

    def test_dropped_full_indices(self, basis):
        dropped = basis.dropped_full_indices
        assert dropped.shape == (1,)
        assert dropped[0] in [1, 2, 3]

    def test_passthrough_indices(self, basis):
        pt = basis._passthrough_full_indices
        assert 0 in pt  # mkt
        assert 4 in pt  # style

    def test_reduced_column_layout(self, basis):
        pass_cols = basis._passthrough_reduced_cols
        family_slices = basis._family_reduced_slices
        assert len(pass_cols) == 2
        assert len(family_slices) == 1
        total = len(pass_cols) + sum(s.stop - s.start for s in family_slices)
        assert total == basis.n_factors_reduced

    def test_reduced_factor_names(self, basis):
        factor_names = np.array(["mkt", "ind0", "ind1", "ind2", "style"])
        reduced_names = basis.reduced_factor_names(factor_names)

        expected = np.concatenate(
            [
                factor_names[basis._passthrough_full_indices],
                factor_names[basis.constrained_families[0].kept_full_indices],
            ]
        )
        np.testing.assert_array_equal(reduced_names, expected)

    def test_frozen(self, basis):
        with pytest.raises(AttributeError):
            basis.n_factors = 99


# ======================================================================
# FamilyConstraintBasis — conversion roundtrips
# ======================================================================
class TestConversionRoundtrips:
    @pytest.fixture()
    def setup(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        return exposures, bench_w, fnames, ffams, basis

    def test_to_reduced_exposures_shape(self, setup):
        exposures, _, _, _, basis = setup
        reduced = basis.to_reduced_exposures(exposures)
        assert reduced.shape == (50, 30, basis.n_factors_reduced)

    def test_factor_returns_roundtrip(self, setup, rng):
        """full -> reduced -> full is identity for non-dropped columns."""
        _, _, _, _, basis = setup
        K_red = basis.n_factors_reduced
        fr_red = rng.standard_normal((50, K_red))
        fr_full = basis.to_full_factor_returns(fr_red)
        fr_back = basis.to_reduced_factor_returns(fr_full)
        np.testing.assert_allclose(fr_back, fr_red, atol=1e-12)

    def test_factor_returns_roundtrip_1d(self, setup, rng):
        """1D (single observation) factor returns roundtrip."""
        _, _, _, _, basis = setup
        K_red = basis.n_factors_reduced
        fr_red_1d = rng.standard_normal(K_red)
        fr_full_1d = basis.to_full_factor_returns(fr_red_1d)
        assert fr_full_1d.ndim == 1
        assert fr_full_1d.shape == (basis.n_factors,)
        fr_back = basis.to_reduced_factor_returns(fr_full_1d)
        assert fr_back.ndim == 1
        np.testing.assert_allclose(fr_back, fr_red_1d, atol=1e-12)

    def test_fitted_values_invariant(self, setup, rng):
        r"""X_full @ f_full == X_red @ f_red for each (t, asset)."""
        exposures, _, _, _, basis = setup
        K_red = basis.n_factors_reduced
        fr_red = rng.standard_normal((50, K_red))
        fr_full = basis.to_full_factor_returns(fr_red)

        X_red = basis.to_reduced_exposures(exposures)
        fitted_full = np.einsum("tnk,tk->tn", exposures, fr_full)
        fitted_red = np.einsum("tnk,tk->tn", X_red, fr_red)
        np.testing.assert_allclose(fitted_full, fitted_red, atol=1e-10)

    def test_mu_roundtrip(self, setup, rng):
        """Reduced mu -> full mu -> reduced mu roundtrip."""
        _, _, _, _, basis = setup
        mu_red = rng.standard_normal(basis.n_factors_reduced)
        mu_full = basis.to_full_factor_mu(mu_red, observation_index=-1)
        assert mu_full.shape == (basis.n_factors,)
        mu_back = basis.to_reduced_factor_returns(mu_full)
        np.testing.assert_allclose(mu_back, mu_red, atol=1e-12)

    def test_covariance_roundtrip_via_dense(self, setup, rng):
        """Cov_full = R @ Cov_red @ R.T must be PSD."""
        _, _, _, _, basis = setup
        K_red = basis.n_factors_reduced
        A = rng.standard_normal((K_red, K_red))
        cov_red = A @ A.T / K_red
        cov_full = basis.to_full_factor_covariance(cov_red, observation_index=-1)
        assert cov_full.shape == (basis.n_factors, basis.n_factors)
        eigvals = np.linalg.eigvalsh(cov_full)
        assert np.all(eigvals >= -1e-12)

    def test_dense_basis_consistency(self, setup, rng):
        """Dense basis at each date matches block-wise conversion."""
        _, _, _, _, basis = setup
        K_red = basis.n_factors_reduced
        fr_red = rng.standard_normal((50, K_red))

        for t in [0, 25, -1]:
            R = basis._dense_basis_at(t)
            fr_single = fr_red[[t if t >= 0 else 50 + t]]
            fr_via_dense = (R @ fr_single.T).T.ravel()
            fr_via_method = basis.to_full_factor_returns(
                fr_red[t if t >= 0 else 50 + t]
            )
            np.testing.assert_allclose(fr_via_dense, fr_via_method, atol=1e-12)


# ======================================================================
# FamilyConstraintBasis — to_reduced_loading_matrix
# ======================================================================
class TestToReducedLoadingMatrix:
    @pytest.fixture()
    def setup(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        return exposures, basis

    def test_shape(self, setup):
        exposures, basis = setup
        loading = exposures[-1]
        reduced = basis.to_reduced_loading_matrix(loading)
        assert reduced.shape == (exposures.shape[1], basis.n_factors_reduced)

    def test_matches_to_reduced_exposures(self, setup):
        """Must produce the same result as the corresponding slice of the
        full 3D to_reduced_exposures output."""
        exposures, basis = setup
        reduced_3d = basis.to_reduced_exposures(exposures)
        for obs in [0, 25, -1]:
            loading = exposures[obs]
            via_loading = basis.to_reduced_loading_matrix(
                loading, observation_index=obs
            )
            np.testing.assert_allclose(via_loading, reduced_3d[obs], atol=1e-14)

    def test_observation_selects_basis_coefficients(self, setup):
        """Different observation indices may produce different results."""
        exposures, basis = setup
        loading = exposures[0]
        r0 = basis.to_reduced_loading_matrix(loading, observation_index=0)
        r_last = basis.to_reduced_loading_matrix(loading, observation_index=-1)
        family_coefficients = basis.constrained_families[0].basis_coefficients
        if not np.allclose(family_coefficients[0], family_coefficients[-1]):
            assert not np.allclose(r0, r_last)

    def test_fitted_values_invariant(self, setup, rng):
        r"""X_full @ f_full == X_red @ f_red for a single date."""
        exposures, basis = setup
        K_red = basis.n_factors_reduced
        fr_red = rng.standard_normal(K_red)
        fr_full = basis.to_full_factor_returns(fr_red)

        loading_full = exposures[-1]
        loading_red = basis.to_reduced_loading_matrix(
            loading_full, observation_index=-1
        )

        fitted_full = loading_full @ fr_full
        fitted_red = loading_red @ fr_red
        np.testing.assert_allclose(fitted_full, fitted_red, atol=1e-10)

    def test_multi_constraint(self, rng):
        """Works correctly with multiple constrained families."""
        T, N, K = 40, 20, 8
        factor_names = np.array(
            ["mkt", "ind1", "ind2", "ind3", "sec1", "sec2", "style1", "style2"]
        )
        factor_families = np.array(
            [
                "market",
                "industry",
                "industry",
                "industry",
                "sector",
                "sector",
                "style",
                "style",
            ]
        )
        exposures = rng.standard_normal((T, N, K))
        exposures[:, :, 0] = 1.0
        bench_w = np.ones((T, N)) / N

        for i in range(3):
            sl = slice(i * 6, (i + 1) * 6)
            exposures[:, sl, 1 + i] = 1.0
            for j in range(3):
                if j != i:
                    exposures[:, sl, 1 + j] = 0.0

        for i in range(2):
            sl = slice(i * 10, (i + 1) * 10)
            exposures[:, sl, 4 + i] = 1.0
            for j in range(2):
                if j != i:
                    exposures[:, sl, 4 + j] = 0.0

        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None), ("sector", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=factor_names,
            factor_families=factor_families,
        )

        reduced_3d = basis.to_reduced_exposures(exposures)
        loading = exposures[-1]
        reduced = basis.to_reduced_loading_matrix(loading)
        assert reduced.shape == (N, basis.n_factors_reduced)
        np.testing.assert_allclose(reduced, reduced_3d[-1], atol=1e-14)


# ======================================================================
# FamilyConstraintBasis — multiple constrained families
# ======================================================================
class TestMultipleConstraints:
    @pytest.fixture()
    def multi_basis(self, rng):
        T, N, K = 40, 20, 8
        factor_names = np.array(
            ["mkt", "ind1", "ind2", "ind3", "sec1", "sec2", "style1", "style2"]
        )
        factor_families = np.array(
            [
                "market",
                "industry",
                "industry",
                "industry",
                "sector",
                "sector",
                "style",
                "style",
            ]
        )
        exposures = rng.standard_normal((T, N, K))
        exposures[:, :, 0] = 1.0
        bench_w = np.ones((T, N)) / N

        for i in range(3):
            sl = slice(i * 6, (i + 1) * 6)
            exposures[:, sl, 1 + i] = 1.0
            for j in range(3):
                if j != i:
                    exposures[:, sl, 1 + j] = 0.0

        for i in range(2):
            sl = slice(i * 10, (i + 1) * 10)
            exposures[:, sl, 4 + i] = 1.0
            for j in range(2):
                if j != i:
                    exposures[:, sl, 4 + j] = 0.0

        basis, resolved = compute_family_constraint_basis(
            constrained_families=[
                ("industry", None),
                ("sector", None),
            ],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        return exposures, basis, resolved

    def test_two_constraints(self, multi_basis):
        _, basis, resolved = multi_basis
        assert basis.n_constraints == 2
        assert basis.n_factors_reduced == 6
        assert len(resolved) == 2

    def test_roundtrip_multi(self, multi_basis, rng):
        _, basis, _ = multi_basis
        K_red = basis.n_factors_reduced
        fr_red = rng.standard_normal((40, K_red))
        fr_full = basis.to_full_factor_returns(fr_red)
        fr_back = basis.to_reduced_factor_returns(fr_full)
        np.testing.assert_allclose(fr_back, fr_red, atol=1e-12)

    def test_reduced_factor_names_multi(self, multi_basis):
        _, basis, _ = multi_basis
        factor_names = np.array(
            ["mkt", "ind1", "ind2", "ind3", "sec1", "sec2", "style1", "style2"]
        )
        reduced_names = basis.reduced_factor_names(factor_names)

        expected = [factor_names[basis._passthrough_full_indices]]
        for family in basis.constrained_families:
            expected.append(factor_names[family.kept_full_indices])
        np.testing.assert_array_equal(reduced_names, np.concatenate(expected))

    def test_fitted_values_multi(self, multi_basis, rng):
        exposures, basis, _ = multi_basis
        K_red = basis.n_factors_reduced
        fr_red = rng.standard_normal((40, K_red))
        fr_full = basis.to_full_factor_returns(fr_red)
        X_red = basis.to_reduced_exposures(exposures)

        fitted_full = np.einsum("tnk,tk->tn", exposures, fr_full)
        fitted_red = np.einsum("tnk,tk->tn", X_red, fr_red)
        np.testing.assert_allclose(fitted_full, fitted_red, atol=1e-10)


# ======================================================================
# compute_family_constraint_basis — factory function
# ======================================================================
class TestComputeBasisBasketNeutral:
    @pytest.fixture()
    def data(self, rng):
        return _industry_data(rng)

    def test_auto_drop(self, data):
        exposures, bench_w, fnames, ffams = data
        _basis, resolved = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        assert len(resolved) == 1
        family, dropped = resolved[0]
        assert family == "industry"
        assert dropped in ["ind0", "ind1", "ind2"]

    def test_explicit_drop(self, data):
        exposures, bench_w, fnames, ffams = data
        basis, resolved = compute_family_constraint_basis(
            constrained_families=[("industry", "ind1")],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        assert resolved[0][1] == "ind1"
        assert 2 in basis.dropped_full_indices  # ind1 is at index 2

    def test_basis_coefficients_shape(self, data):
        exposures, bench_w, fnames, ffams = data
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        # 3 industries, 1 dropped → 2 retained per family
        assert basis.constrained_families[0].basis_coefficients.shape == (50, 2)

    def test_invalid_exposures_dim(self, data):
        _, bench_w, fnames, ffams = data
        with pytest.raises(ValueError, match="3D array"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=np.zeros((50, 30)),
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_invalid_weights_shape(self, data):
        exposures, _, fnames, ffams = data
        with pytest.raises(ValueError, match="benchmark_weights shape"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=exposures,
                benchmark_weights=np.ones((10, 30)),
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_unknown_family(self, data):
        exposures, bench_w, fnames, ffams = data
        with pytest.raises(ValueError, match="not found"):
            compute_family_constraint_basis(
                constrained_families=[("nonexistent", None)],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_unknown_factor_to_drop(self, data):
        exposures, bench_w, fnames, ffams = data
        with pytest.raises(ValueError, match="not found"):
            compute_family_constraint_basis(
                constrained_families=[("industry", "bogus")],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_single_factor_family(self, rng):
        T, N = 10, 5
        fnames = np.array(["a", "b"])
        ffams = np.array(["x", "y"])
        exposures = rng.standard_normal((T, N, 2))
        bench_w = np.ones((T, N)) / N
        with pytest.raises(ValueError, match="at least two"):
            compute_family_constraint_basis(
                constrained_families=[("x", None)],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_too_many_constraints(self, rng):
        T, N = 10, 8
        fnames = np.array(["a1", "a2", "b1", "b2", "c1", "c2"])
        ffams = np.array(["x", "x", "y", "y", "z", "z"])
        exposures = rng.standard_normal((T, N, 6))
        bench_w = np.ones((T, N)) / N
        with pytest.raises(ValueError, match="appears more than once"):
            compute_family_constraint_basis(
                constrained_families=[
                    ("x", None),
                    ("y", None),
                    ("z", None),
                    ("x", None),
                    ("y", None),
                    ("z", None),
                ],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_factor_to_drop_must_belong_to_family(self, data):
        exposures, bench_w, fnames, ffams = data
        with pytest.raises(ValueError, match="does not belong to family"):
            compute_family_constraint_basis(
                constrained_families=[("industry", "style")],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_near_zero_denominator(self, rng):
        T, N = 10, 6
        fnames = np.array(["a", "b", "c"])
        ffams = np.array(["x", "x", "y"])
        exposures = np.ones((T, N, 3))
        exposures[:, :, 0] = 0.0
        bench_w = np.ones((T, N)) / N
        with pytest.raises(ValueError, match="near-zero"):
            compute_family_constraint_basis(
                constrained_families=[("x", "a")],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_zero_total_benchmark_weight_raises(self, data):
        exposures, bench_w, fnames, ffams = data
        bench_w[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive sum"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )


# ======================================================================
# FamilyConstraintBasis — edge cases
# ======================================================================
class TestEdgeCases:
    def test_ratios_slicing_preserves_structure(self, rng):
        """Observation slicing produces a valid basis with fewer observations."""
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        sliced = basis[10:]
        assert sliced.n_observations == 20
        assert sliced.n_factors_reduced == basis.n_factors_reduced

        fr_red = rng.standard_normal((20, basis.n_factors_reduced))
        fr_full = sliced.to_full_factor_returns(fr_red)
        X_red = sliced.to_reduced_exposures(exposures[10:])
        fitted_full = np.einsum("tnk,tk->tn", exposures[10:], fr_full)
        fitted_red = np.einsum("tnk,tk->tn", X_red, fr_red)
        np.testing.assert_allclose(fitted_full, fitted_red, atol=1e-10)

    def test_getitem_slice_returns_expected_basis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        sliced = basis[5:18]
        for sliced_family, original_family in zip(
            sliced.constrained_families, basis.constrained_families, strict=True
        ):
            np.testing.assert_allclose(
                sliced_family.basis_coefficients,
                original_family.basis_coefficients[5:18],
            )
            assert sliced_family.family_name == original_family.family_name
            np.testing.assert_array_equal(
                sliced_family.full_factor_indices, original_family.full_factor_indices
            )
            assert (
                sliced_family.dropped_index_in_family
                == original_family.dropped_index_in_family
            )
        assert sliced.n_factors == basis.n_factors

    def test_integer_index_returns_single_observation_basis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        single = basis[-1]
        assert isinstance(single, FamilyConstraintBasis)
        assert single.n_observations == 1
        for sliced_family, original_family in zip(
            single.constrained_families, basis.constrained_families, strict=True
        ):
            np.testing.assert_allclose(
                sliced_family.basis_coefficients[0],
                original_family.basis_coefficients[-1],
            )

    def test_tuple_indexing_is_rejected(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        with pytest.raises(TypeError, match="observation axis"):
            _ = basis[:, :]

    def test_single_observation(self, rng):
        """Basis with a single observation works for all conversions."""
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=1, N=12)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        assert basis.n_observations == 1
        fr_red = rng.standard_normal(basis.n_factors_reduced)
        fr_full = basis.to_full_factor_returns(fr_red)
        assert fr_full.ndim == 1

        mu_full = basis.to_full_factor_mu(fr_red, observation_index=0)
        np.testing.assert_allclose(mu_full, fr_full, atol=1e-12)

    def test_to_reduced_exposures_requires_matching_time_axis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.to_reduced_exposures(exposures[:10])

    def test_to_full_factor_returns_requires_matching_time_axis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        factor_returns_reduced = rng.standard_normal((10, basis.n_factors_reduced))
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.to_full_factor_returns(factor_returns_reduced)

    def test_to_reduced_factor_coordinates_requires_matching_time_axis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        v_full = rng.standard_normal((10, basis.n_factors))
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.to_reduced_factor_coordinates(v_full)

    def test_to_full_factor_covariance_requires_matching_time_axis(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=30)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        cov_reduced = np.broadcast_to(
            np.eye(basis.n_factors_reduced),
            (10, basis.n_factors_reduced, basis.n_factors_reduced),
        ).copy()
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.to_full_factor_covariance(cov_reduced)

    def test_to_reduced_exposures_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        wrong = exposures[:, :, :-1]
        with pytest.raises(ValueError, match="factor dimension"):
            basis.to_reduced_exposures(wrong)

    def test_to_reduced_loading_matrix_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        with pytest.raises(ValueError, match="factor dimension"):
            basis.to_reduced_loading_matrix(exposures[-1][:, :-1])

    def test_to_full_factor_returns_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        wrong = rng.standard_normal((20, basis.n_factors_reduced + 1))
        with pytest.raises(ValueError, match="factor dimension"):
            basis.to_full_factor_returns(wrong)

    def test_to_reduced_factor_returns_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        wrong = rng.standard_normal((20, basis.n_factors + 1))
        with pytest.raises(ValueError, match="factor dimension"):
            basis.to_reduced_factor_returns(wrong)

    def test_to_full_factor_mu_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        wrong = rng.standard_normal(basis.n_factors_reduced + 1)
        with pytest.raises(ValueError, match="factor dimension"):
            basis.to_full_factor_mu(wrong)

    def test_to_full_factor_covariance_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        wrong = np.eye(basis.n_factors_reduced + 1)
        with pytest.raises(ValueError, match="trailing shape"):
            basis.to_full_factor_covariance(wrong)

    def test_to_reduced_factor_coordinates_rejects_wrong_factor_dimension(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        wrong = rng.standard_normal((20, basis.n_factors + 1))
        with pytest.raises(ValueError, match="factor dimension"):
            basis.to_reduced_factor_coordinates(wrong)

    def test_nan_benchmark_weights_zeroed(self, rng):
        """Non-finite benchmark weights are coerced to zero."""
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20, N=12)
        bench_w[0, 0] = np.nan
        bench_w[1, 1] = np.inf
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        assert all(
            np.all(np.isfinite(family.basis_coefficients))
            for family in basis.constrained_families
        )

    def test_negative_benchmark_weights_rejected(self, rng):
        """Negative benchmark weights raise."""
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20, N=12)
        bench_w[1, 1] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )

    def test_covariance_symmetry(self, rng):
        """Reconstructed full covariance must be symmetric."""
        exposures, bench_w, fnames, ffams = _industry_data(rng)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        K_red = basis.n_factors_reduced
        A = rng.standard_normal((K_red, K_red))
        cov_red = A @ A.T / K_red
        cov_full = basis.to_full_factor_covariance(cov_red)
        np.testing.assert_allclose(cov_full, cov_full.T, atol=1e-14)

    def test_covariance_rank(self, rng):
        """Full covariance has rank K_red (dropped row is linear combination)."""
        exposures, bench_w, fnames, ffams = _industry_data(rng)
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=bench_w,
            factor_names=fnames,
            factor_families=ffams,
        )
        K_red = basis.n_factors_reduced
        A = rng.standard_normal((K_red, K_red))
        cov_red = A @ A.T / K_red
        cov_full = basis.to_full_factor_covariance(cov_red)
        rank = np.linalg.matrix_rank(cov_full, tol=1e-10)
        assert rank == K_red

    def test_direct_basis_rejects_non_finite_basis_coefficients(self):
        with pytest.raises(ValueError, match="finite values"):
            ConstrainedFamily(
                family_name="industry",
                full_factor_indices=np.array([0, 1]),
                dropped_index_in_family=0,
                basis_coefficients=np.array([[np.nan]]),
            )

    def test_compute_basis_rejects_factor_names_shape_mismatch(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        with pytest.raises(ValueError, match="factor_names shape"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames[:-1],
                factor_families=ffams,
            )

    def test_compute_basis_rejects_duplicate_factor_names(self, rng):
        exposures, bench_w, fnames, ffams = _industry_data(rng, T=20)
        fnames = fnames.copy()
        fnames[1] = fnames[0]
        with pytest.raises(ValueError, match="must be unique"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=exposures,
                benchmark_weights=bench_w,
                factor_names=fnames,
                factor_families=ffams,
            )
