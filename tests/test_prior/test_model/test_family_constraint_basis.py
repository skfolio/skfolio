"""Tests for skfolio.prior._model._family_constraint_basis."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.prior._model._family_constraint_basis import (
    FamilyConstraint,
    FamilyConstraintBasis,
    compute_family_constraint_basis,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def _industry_data(rng, *, T=50, N=30, n_industries=3):
    """Build a simple intercept + industry + style factor setup."""
    n_factors = 1 + n_industries + 1
    factor_names = np.array(
        ["mkt"] + [f"ind{i}" for i in range(n_industries)] + ["style"]
    )
    factor_families = np.array(["market"] + ["industry"] * n_industries + ["style"])

    exposures = rng.standard_normal((T, N, n_factors))
    exposures[:, :, 0] = 1.0
    for i in range(n_industries):
        group = slice(i * (N // n_industries), (i + 1) * (N // n_industries))
        exposures[:, group, 1 + i] = 1.0
        for j in range(n_industries):
            if j != i:
                exposures[:, group, 1 + j] = 0.0

    benchmark_weights = np.ones((T, N)) / N
    return exposures, benchmark_weights, factor_names, factor_families


def _contract_matrix_from_public_state(
    basis: FamilyConstraintBasis,
    observation_index: int,
) -> np.ndarray:
    """Build a dense point-in-time basis in tests from the public contract."""
    contract_matrix = np.zeros((basis.n_full_factors, basis.n_reduced_factors))
    contract_matrix[basis.retained_full_indices, np.arange(basis.n_reduced_factors)] = (
        1.0
    )
    for family, reduced_indices, ratio_slice in basis._iter_families():
        contract_matrix[
            family.dropped_full_index, reduced_indices
        ] = -basis.constraint_ratios[observation_index, ratio_slice]
    return contract_matrix


class TestFamilyConstraint:
    def test_basic_properties(self):
        family = FamilyConstraint(
            family_name="industry",
            full_factor_indices=np.array([1, 2, 3]),
            dropped_index_in_family=0,
        )
        assert family.family_size == 3
        assert family.dropped_full_index == 1
        assert family.dropped_full_slice == slice(1, 2)
        np.testing.assert_array_equal(family.retained_full_indices, [2, 3])
        np.testing.assert_array_equal(family.retained_local_indices, [1, 2])

    def test_dropped_middle(self):
        family = FamilyConstraint(
            family_name="sector",
            full_factor_indices=np.array([5, 7, 9, 11]),
            dropped_index_in_family=2,
        )
        assert family.dropped_full_index == 9
        np.testing.assert_array_equal(family.retained_full_indices, [5, 7, 11])
        np.testing.assert_array_equal(family.retained_local_indices, [0, 1, 3])

    def test_validation(self):
        with pytest.raises(ValueError, match="1D"):
            FamilyConstraint("x", np.array([[0, 1]]), 0)
        with pytest.raises(ValueError, match="at least two"):
            FamilyConstraint("x", np.array([0]), 0)
        with pytest.raises(ValueError, match="unique"):
            FamilyConstraint("x", np.array([0, 0]), 0)
        with pytest.raises(ValueError, match="out of bounds"):
            FamilyConstraint("x", np.array([0, 1]), 2)

    def test_frozen(self):
        family = FamilyConstraint("x", np.array([0, 1]), 0)
        with pytest.raises(AttributeError):
            family.family_name = "y"


class TestFamilyConstraintBasisProperties:
    @pytest.fixture()
    def basis(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=20, N=12
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", "ind2")],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        return basis

    def test_dimensions(self, basis):
        assert basis.n_full_factors == 5
        assert basis.n_constraints == 1
        assert basis.n_reduced_factors == 4
        assert basis.n_observations == 20

    def test_reduced_order_is_full_order_without_dropped(self, basis):
        factor_names = np.array(["mkt", "ind0", "ind1", "ind2", "style"])
        reduced_names = basis.reduced_factor_names(factor_names)
        np.testing.assert_array_equal(reduced_names, ["mkt", "ind0", "ind1", "style"])
        np.testing.assert_array_equal(
            reduced_names, factor_names[basis.retained_full_indices]
        )

    def test_currency_append_preserves_order(self, basis):
        factor_names = np.array(["mkt", "ind0", "ind1", "ind2", "style"])
        ccy_names = np.array(["ccy_USD", "ccy_EUR"])
        basis_with_ccy = basis.append_passthrough_factors(len(ccy_names))

        np.testing.assert_array_equal(
            basis_with_ccy.reduced_factor_names(
                np.concatenate([factor_names, ccy_names])
            ),
            np.concatenate([basis.reduced_factor_names(factor_names), ccy_names]),
        )

    def test_full_to_reduced_index(self, basis):
        np.testing.assert_array_equal(basis.retained_full_indices, [0, 1, 2, 4])
        np.testing.assert_array_equal(basis.full_to_reduced_index, [0, 1, 2, -1, 3])

    def test_with_constraint_ratios(self, basis, rng):
        ratios = rng.standard_normal((12, basis.constraint_ratios.shape[1]))
        updated = basis.with_constraint_ratios(ratios)
        assert updated.n_full_factors == basis.n_full_factors
        assert updated.n_observations == 12
        np.testing.assert_allclose(updated.constraint_ratios, ratios)

    def test_append_passthrough_factors_validation(self, basis):
        assert basis.append_passthrough_factors(0) is basis
        with pytest.raises(ValueError, match="integer"):
            basis.append_passthrough_factors(1.5)
        with pytest.raises(ValueError, match="non-negative"):
            basis.append_passthrough_factors(-1)


class TestBasisTransforms:
    @pytest.fixture()
    def setup(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", "ind2")],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        return exposures, factor_names, basis

    def test_reduce_exposures_shape_and_formula(self, setup):
        exposures, _, basis = setup
        reduced = basis.reduce_exposures(exposures)
        assert reduced.shape == (50, 30, basis.n_reduced_factors)

        family = basis.family_constraints[0]
        ratio_slice = basis._family_ratio_slices[0]
        reduced_indices = basis._family_reduced_indices[0]
        expected_family = (
            exposures[:, :, family.retained_full_indices]
            - exposures[:, :, family.dropped_full_slice]
            * basis.constraint_ratios[:, np.newaxis, ratio_slice]
        )
        np.testing.assert_allclose(reduced[:, :, reduced_indices], expected_family)

        passthrough_full = np.array([0, 4])
        passthrough_reduced = basis.full_to_reduced_index[passthrough_full]
        np.testing.assert_allclose(
            reduced[:, :, passthrough_reduced],
            exposures[:, :, passthrough_full],
        )

    def test_reduce_loading_matrix_matches_3d_transform(self, setup):
        exposures, _, basis = setup
        reduced_3d = basis.reduce_exposures(exposures)
        for observation_index in [0, 25, -1]:
            loading_reduced = basis.reduce_loading_matrix(
                exposures[observation_index],
                observation_index=observation_index,
            )
            np.testing.assert_allclose(
                loading_reduced, reduced_3d[observation_index], atol=1e-14
            )

    def test_factor_returns_roundtrip(self, setup, rng):
        _, _, basis = setup
        factor_returns_reduced = rng.standard_normal((50, basis.n_reduced_factors))
        factor_returns_full = basis.expand_factor_returns(factor_returns_reduced)
        factor_returns_back = basis.reduce_factor_returns(factor_returns_full)
        np.testing.assert_allclose(
            factor_returns_back, factor_returns_reduced, atol=1e-12
        )

    def test_factor_returns_roundtrip_1d(self, setup, rng):
        _, _, basis = setup
        factor_returns_reduced = rng.standard_normal(basis.n_reduced_factors)
        factor_returns_full = basis.expand_factor_returns(factor_returns_reduced)
        factor_returns_back = basis.reduce_factor_returns(factor_returns_full)
        assert factor_returns_full.shape == (basis.n_full_factors,)
        np.testing.assert_allclose(
            factor_returns_back, factor_returns_reduced, atol=1e-12
        )

    def test_fitted_values_invariant(self, setup, rng):
        exposures, _, basis = setup
        factor_returns_reduced = rng.standard_normal((50, basis.n_reduced_factors))
        factor_returns_full = basis.expand_factor_returns(factor_returns_reduced)
        exposures_reduced = basis.reduce_exposures(exposures)

        fitted_full = np.einsum("tnk,tk->tn", exposures, factor_returns_full)
        fitted_reduced = np.einsum(
            "tnk,tk->tn", exposures_reduced, factor_returns_reduced
        )
        np.testing.assert_allclose(fitted_full, fitted_reduced, atol=1e-10)

    def test_mu_expansion(self, setup, rng):
        _, _, basis = setup
        mu_reduced = rng.standard_normal(basis.n_reduced_factors)
        mu_full = basis.expand_factor_mu(mu_reduced, observation_index=-1)
        mu_back = basis.reduce_factor_mu(mu_full)
        np.testing.assert_allclose(mu_back, mu_reduced, atol=1e-12)
        np.testing.assert_allclose(
            basis.reduce_factor_returns(mu_full), mu_reduced, atol=1e-12
        )

    def test_project_factor_coordinates_uses_time_varying_ratios(self, setup, rng):
        _, _, basis = setup
        coordinates_full = rng.standard_normal(
            (basis.n_observations, basis.n_full_factors)
        )
        projected = basis.project_factor_coordinates(coordinates_full)

        for observation_index in [0, 17, -1]:
            contract_matrix = _contract_matrix_from_public_state(
                basis, observation_index
            )
            expected = coordinates_full[observation_index] @ contract_matrix
            np.testing.assert_allclose(
                projected[observation_index], expected, atol=1e-12
            )

    def test_covariance_expansion_2d_matches_contract(self, setup, rng):
        _, _, basis = setup
        matrix = rng.standard_normal((basis.n_reduced_factors, basis.n_reduced_factors))
        covariance_reduced = matrix @ matrix.T / basis.n_reduced_factors
        covariance_full = basis.expand_factor_covariance(covariance_reduced)

        contract_matrix = _contract_matrix_from_public_state(basis, -1)
        expected = contract_matrix @ covariance_reduced @ contract_matrix.T
        np.testing.assert_allclose(covariance_full, expected, atol=1e-12)
        assert covariance_full.shape == (basis.n_full_factors, basis.n_full_factors)
        assert np.all(np.linalg.eigvalsh(covariance_full) >= -1e-12)

    def test_covariance_expansion_3d_matches_contract(self, rng):
        n_observations, n_assets = 40, 20
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
        exposures = rng.standard_normal((n_observations, n_assets, len(factor_names)))
        exposures[:, :, 0] = 1.0
        for i in range(3):
            asset_slice = slice(i * 6, (i + 1) * 6)
            exposures[:, asset_slice, 1 + i] = 1.0
            for j in range(3):
                if j != i:
                    exposures[:, asset_slice, 1 + j] = 0.0
        for i in range(2):
            asset_slice = slice(i * 10, (i + 1) * 10)
            exposures[:, asset_slice, 4 + i] = 1.0
            for j in range(2):
                if j != i:
                    exposures[:, asset_slice, 4 + j] = 0.0

        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", "ind3"), ("sector", "sec2")],
            factor_exposures=exposures,
            benchmark_weights=np.ones((n_observations, n_assets)) / n_assets,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        covariance_reduced = np.empty(
            (n_observations, basis.n_reduced_factors, basis.n_reduced_factors)
        )
        for i in range(n_observations):
            matrix = rng.standard_normal(
                (basis.n_reduced_factors, basis.n_reduced_factors)
            )
            covariance_reduced[i] = matrix @ matrix.T / basis.n_reduced_factors

        covariance_full = basis.expand_factor_covariance(covariance_reduced)
        for observation_index in [0, 13, -1]:
            contract_matrix = _contract_matrix_from_public_state(
                basis, observation_index
            )
            expected = (
                contract_matrix
                @ covariance_reduced[observation_index]
                @ contract_matrix.T
            )
            np.testing.assert_allclose(
                covariance_full[observation_index], expected, atol=1e-12
            )

    def test_reduce_factor_covariance_roundtrip_2d(self, setup, rng):
        _, _, basis = setup
        matrix = rng.standard_normal((basis.n_reduced_factors, basis.n_reduced_factors))
        covariance_reduced = matrix @ matrix.T / basis.n_reduced_factors
        covariance_full = basis.expand_factor_covariance(covariance_reduced)

        covariance_back = basis.reduce_factor_covariance(covariance_full)

        np.testing.assert_allclose(covariance_back, covariance_reduced, atol=1e-12)

    def test_reduce_factor_covariance_roundtrip_3d(self, setup, rng):
        _, _, basis = setup
        covariance_reduced = np.empty(
            (basis.n_observations, basis.n_reduced_factors, basis.n_reduced_factors)
        )
        for i in range(basis.n_observations):
            matrix = rng.standard_normal(
                (basis.n_reduced_factors, basis.n_reduced_factors)
            )
            covariance_reduced[i] = matrix @ matrix.T / basis.n_reduced_factors

        covariance_full = basis.expand_factor_covariance(covariance_reduced)
        covariance_back = basis.reduce_factor_covariance(covariance_full)

        np.testing.assert_allclose(covariance_back, covariance_reduced, atol=1e-12)

    def test_reduce_factor_covariance_returns_full_rank_retained_block(
        self, setup, rng
    ):
        _, _, basis = setup
        matrix = rng.standard_normal((basis.n_reduced_factors, basis.n_reduced_factors))
        covariance_reduced = matrix @ matrix.T / basis.n_reduced_factors
        covariance_full = basis.expand_factor_covariance(covariance_reduced)

        retained = basis.retained_full_indices
        covariance_back = basis.reduce_factor_covariance(covariance_full)

        np.testing.assert_allclose(
            covariance_back, covariance_full[np.ix_(retained, retained)], atol=1e-12
        )
        assert np.linalg.eigvalsh(covariance_back).min() > 0
        assert np.linalg.matrix_rank(covariance_full) == basis.n_reduced_factors
        assert np.linalg.matrix_rank(covariance_full) < basis.n_full_factors


class TestComputeFamilyConstraintBasis:
    @pytest.fixture()
    def data(self, rng):
        return _industry_data(rng)

    def test_auto_drop(self, data):
        exposures, benchmark_weights, factor_names, factor_families = data
        _basis, resolved = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        assert resolved[0][0] == "industry"
        assert resolved[0][1] in ["ind0", "ind1", "ind2"]

    def test_explicit_drop(self, data):
        exposures, benchmark_weights, factor_names, factor_families = data
        basis, resolved = compute_family_constraint_basis(
            constrained_families=[("industry", "ind1")],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        assert resolved[0][1] == "ind1"
        np.testing.assert_array_equal(basis.dropped_full_indices, [2])

    def test_constraint_ratios_shape(self, data):
        exposures, benchmark_weights, factor_names, factor_families = data
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        assert basis.constraint_ratios.shape == (50, 2)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"factor_exposures": np.zeros((50, 30))}, "3D array"),
            ({"benchmark_weights": np.ones((10, 30))}, "benchmark_weights shape"),
        ],
    )
    def test_shape_validation(self, data, kwargs, match):
        exposures, benchmark_weights, factor_names, factor_families = data
        params = dict(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            compute_family_constraint_basis(**params)

    def test_unknown_family(self, data):
        exposures, benchmark_weights, factor_names, factor_families = data
        with pytest.raises(ValueError, match="not found"):
            compute_family_constraint_basis(
                constrained_families=[("missing", None)],
                factor_exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )

    def test_empty_constraints(self, data):
        exposures, benchmark_weights, factor_names, factor_families = data
        with pytest.raises(ValueError, match="at least one family"):
            compute_family_constraint_basis(
                constrained_families=[],
                factor_exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )

    def test_invalid_factor_to_drop(self, data):
        exposures, benchmark_weights, factor_names, factor_families = data
        with pytest.raises(ValueError, match="not found"):
            compute_family_constraint_basis(
                constrained_families=[("industry", "missing")],
                factor_exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )
        with pytest.raises(ValueError, match="does not belong"):
            compute_family_constraint_basis(
                constrained_families=[("industry", "style")],
                factor_exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )

    def test_near_zero_denominator(self, rng):
        factor_names = np.array(["a", "b", "c"])
        factor_families = np.array(["x", "x", "y"])
        exposures = np.ones((10, 6, 3))
        exposures[:, :, 0] = 0.0
        with pytest.raises(ValueError, match="near-zero"):
            compute_family_constraint_basis(
                constrained_families=[("x", "a")],
                factor_exposures=exposures,
                benchmark_weights=np.ones((10, 6)) / 6,
                factor_names=factor_names,
                factor_families=factor_families,
            )

    def test_nan_benchmark_weights_zeroed(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=20, N=12
        )
        benchmark_weights[0, 0] = np.nan
        benchmark_weights[1, 1] = np.inf
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        assert np.all(np.isfinite(basis.constraint_ratios))

    def test_negative_benchmark_weights_rejected(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=20, N=12
        )
        benchmark_weights[1, 1] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            compute_family_constraint_basis(
                constrained_families=[("industry", None)],
                factor_exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )


class TestEdgeCases:
    def test_slicing_preserves_structure(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=30
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        sliced = basis[10:]
        assert sliced.n_observations == 20
        assert sliced.n_reduced_factors == basis.n_reduced_factors
        np.testing.assert_allclose(
            sliced.constraint_ratios, basis.constraint_ratios[10:]
        )
        assert sliced.family_constraints == basis.family_constraints

    def test_integer_index_returns_single_observation_basis(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=30
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        single = basis[-1]
        assert single.n_observations == 1
        np.testing.assert_allclose(
            single.constraint_ratios[0], basis.constraint_ratios[-1]
        )

    def test_tuple_indexing_is_rejected(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=30
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        with pytest.raises(TypeError, match="observation axis"):
            _ = basis[:, :]

    def test_time_axis_validation(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=30
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.reduce_exposures(exposures[:10])
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.expand_factor_returns(
                rng.standard_normal((10, basis.n_reduced_factors))
            )
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.project_factor_coordinates(
                rng.standard_normal((10, basis.n_full_factors))
            )
        with pytest.raises(ValueError, match="Slice the FamilyConstraintBasis first"):
            basis.expand_factor_covariance(
                np.broadcast_to(
                    np.eye(basis.n_reduced_factors),
                    (10, basis.n_reduced_factors, basis.n_reduced_factors),
                ).copy()
            )

    def test_factor_dimension_validation(self, rng):
        exposures, benchmark_weights, factor_names, factor_families = _industry_data(
            rng, T=20
        )
        basis, _ = compute_family_constraint_basis(
            constrained_families=[("industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )
        with pytest.raises(ValueError, match="factor dimension"):
            basis.reduce_exposures(exposures[:, :, :-1])
        with pytest.raises(ValueError, match="factor dimension"):
            basis.reduce_loading_matrix(exposures[-1][:, :-1])
        with pytest.raises(ValueError, match="factor dimension"):
            basis.expand_factor_returns(
                rng.standard_normal((20, basis.n_reduced_factors + 1))
            )
        with pytest.raises(ValueError, match="factor dimension"):
            basis.reduce_factor_returns(
                rng.standard_normal((20, basis.n_full_factors + 1))
            )
        with pytest.raises(ValueError, match="factor dimension"):
            basis.expand_factor_mu(rng.standard_normal(basis.n_reduced_factors + 1))
        with pytest.raises(ValueError, match="factor_mu must be a 1D array"):
            basis.reduce_factor_mu(rng.standard_normal((20, basis.n_full_factors)))
        with pytest.raises(ValueError, match="factor dimension"):
            basis.reduce_factor_mu(rng.standard_normal(basis.n_full_factors + 1))
        with pytest.raises(ValueError, match="trailing shape"):
            basis.expand_factor_covariance(np.eye(basis.n_reduced_factors + 1))
        with pytest.raises(ValueError, match="2D or 3D array"):
            basis.reduce_factor_covariance(
                rng.standard_normal((2, basis.n_full_factors, basis.n_full_factors, 1))
            )
        with pytest.raises(ValueError, match="trailing shape"):
            basis.reduce_factor_covariance(np.eye(basis.n_full_factors + 1))
        with pytest.raises(ValueError, match="factor dimension"):
            basis.project_factor_coordinates(
                rng.standard_normal((20, basis.n_full_factors + 1))
            )

    def test_direct_basis_rejects_non_finite_ratios(self):
        with pytest.raises(ValueError, match="finite values"):
            FamilyConstraintBasis(
                n_full_factors=2,
                family_constraints=(
                    FamilyConstraint(
                        family_name="industry",
                        full_factor_indices=np.array([0, 1]),
                        dropped_index_in_family=0,
                    ),
                ),
                constraint_ratios=np.array([[np.nan]]),
            )
