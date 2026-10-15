"""
Rigorous tests for compositional mirror maps.

Goal: Verify that forward-backward composition is identity up to numerical error:
      w → grad_psi(w) = z → grad_psi_star(z) = w'  should give w ≈ w'

Test coverage:
1. Diagonal compositions (closed-form quadratic inversion)
2. a_i = 0 coordinates (pure barrier reduction)
3. Coupled compositions (Newton solver with positivity)
4. Nested compositions (verify flattening works)
5. Demonstrate inverse non-additivity
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._mirror_maps import (
    AdaBarronsBarrierMap,
    AdaptiveMahalanobisMap,
    AdaptiveVariationMap,
    CompositeMirrorMap,
    DiagonalQuadraticMap,
    EuclideanMap,
    FullQuadraticMap,
    LogBarrierMap,
    make_ada_barrons_mirror_map,
)


def _rand_positive_simplex(d, rng):
    """Generate random positive weights (not necessarily summing to 1)."""
    x = rng.random(d) + 0.1
    return x


class TestAtomicMapsRoundtrip:
    """Verify each atomic map's inverse is correct."""

    def test_log_barrier_roundtrip(self):
        rng = np.random.default_rng(0)
        d = 5
        barrier = LogBarrierMap(barrier_coef=2.5)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        w = _rand_positive_simplex(d, rng)
        z = barrier.grad_psi(w)
        w_rec = barrier.grad_psi_star(z)
        assert_allclose(w_rec, w, rtol=1e-10, atol=1e-12)

    def test_euclidean_roundtrip(self):
        rng = np.random.default_rng(1)
        d = 5
        euc = EuclideanMap(d_coef=3.0)

        w = rng.random(d) + 0.2
        z = euc.grad_psi(w)
        w_rec = euc.grad_psi_star(z)
        # For Euclidean alone, inverse is simple division
        # Note: EuclideanMap's grad_psi_star only works when used alone!
        # In composition, it will fail; that's tested elsewhere.
        assert_allclose(w_rec, w, rtol=1e-10, atol=1e-12)

    def test_diagonal_quadratic_roundtrip(self):
        rng = np.random.default_rng(2)
        d = 5
        diag = DiagonalQuadraticMap(beta=0.2)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.0

        w = _rand_positive_simplex(d, rng)
        z = diag.grad_psi(w)
        w_rec = diag.grad_psi_star(z)
        # Again, only works alone
        assert_allclose(w_rec, w, rtol=1e-10, atol=1e-12)


class TestDiagonalCompositions:
    """Test closed-form diagonal inversion."""

    def test_diagonal_composite_exact_recovery_simple(self):
        """Test barrier + euclidean + diagonal with fixed coefficients."""
        rng = np.random.default_rng(10)
        d = 7

        barrier = LogBarrierMap(barrier_coef=1.0)
        euclid = EuclideanMap(d_coef=2.0)
        diag = DiagonalQuadraticMap(beta=0.0)  # Static diagonal

        barrier.ensure_dim(d)
        diag.ensure_dim(d)

        # Set coefficients manually
        barrier._D = rng.random(d) + 0.5
        diag._H_diag = rng.random(d) + 2.0

        # Compose
        comp = barrier + euclid + diag

        # Random positive w
        w = _rand_positive_simplex(d, rng) * 3.0
        z = comp.grad_psi(w)
        w_hat = comp.grad_psi_star(z)

        assert np.all(w_hat > 0), "Recovered weights must be positive."
        assert_allclose(w_hat, w, rtol=1e-10, atol=1e-12)

        # Forward-inverse both directions
        z_hat = comp.grad_psi(w_hat)
        assert_allclose(z_hat, z, rtol=1e-10, atol=1e-12)

    def test_nested_compositions_flatten_correctly(self):
        """Verify nested compositions produce identical results after flattening."""
        rng = np.random.default_rng(11)
        d = 6

        barrier = LogBarrierMap(barrier_coef=1.5)
        euclid = EuclideanMap(d_coef=1.0)
        diag = DiagonalQuadraticMap(beta=0.1)

        barrier.ensure_dim(d)
        diag.ensure_dim(d)

        barrier._D = rng.random(d) + 0.3
        diag._H_diag = rng.random(d) + 1.5

        # Different nesting orders
        comp1 = barrier + euclid + diag
        comp2 = (barrier + euclid) + diag
        comp3 = barrier + (euclid + diag)

        w = _rand_positive_simplex(d, rng) * 2.0

        for comp in [comp1, comp2, comp3]:
            z = comp.grad_psi(w)
            w_hat = comp.grad_psi_star(z)
            assert_allclose(
                w_hat, w, rtol=1e-10, atol=1e-12, err_msg=f"Failed for {comp}"
            )

    def test_diagonal_handles_ai_zero_coordinates(self):
        """Test pure barrier case (a_i = 0 for all i)."""
        rng = np.random.default_rng(12)
        d = 5

        barrier = LogBarrierMap(barrier_coef=0.0)
        euclid = EuclideanMap(d_coef=0.0)  # Zero Euclidean
        diag = DiagonalQuadraticMap(beta=0.0)

        barrier.ensure_dim(d)
        diag.ensure_dim(d)

        # Pure barrier: a_i = 0 everywhere
        diag._H_diag = np.zeros(d)
        barrier._D = rng.random(d) + 0.3

        comp = barrier + euclid + diag

        # Construct w > 0 and compute z = -D / w
        w = rng.random(d) + 0.2
        z = -barrier._D / w

        # Inverse should recover w exactly
        w_hat = comp.grad_psi_star(z)
        assert_allclose(w_hat, w, rtol=1e-10, atol=1e-12)


class TestCoupledCompositions:
    """Test Newton solver for coupled (full quadratic) compositions."""

    def test_coupled_composite_newton_recovery(self):
        """Verify Newton solver recovers w from z in coupled system."""
        rng = np.random.default_rng(20)
        d = 6

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        euclid = EuclideanMap(d_coef=2.0)

        diag = DiagonalQuadraticMap(beta=0.5)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.0

        full = FullQuadraticMap(beta=0.3)
        full.ensure_dim(d)
        # Build SPD matrix via B^T B
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = barrier + euclid + diag + full

        # Random positive w
        w = rng.random(d) + 0.5
        z = comp.grad_psi(w)

        # Newton solver should recover w
        w_hat = comp.grad_psi_star(z)
        assert np.all(w_hat > 0), "Newton solver must preserve positivity."
        assert_allclose(w_hat, w, rtol=1e-8, atol=1e-10)

        # Verify dual consistency
        z_hat = comp.grad_psi(w_hat)
        assert_allclose(z_hat, z, rtol=1e-8, atol=1e-10)


class TestInverseNonAdditivity:
    """Demonstrate that naive sum of inverses fails."""

    def test_naive_sum_of_inverses_is_wrong(self):
        """Show that ∇ψ1*(z) + ∇ψ2*(z) ≠ ∇(ψ1+ψ2)*(z)."""
        d = 5
        barrier = LogBarrierMap(barrier_coef=0.0)
        euclid = EuclideanMap(d_coef=5.0)

        barrier.ensure_dim(d)
        barrier._D = np.array([1.0, 0.5, 0.2, 0.8, 0.5])

        w = np.array([0.3, 0.2, 0.15, 0.25, 0.1])

        # Individual forward maps
        z_barrier = barrier.grad_psi(w)
        z_euclid = euclid.grad_psi(w)
        z_sum = z_barrier + z_euclid

        # Naive sum of individual inverses (WRONG!)
        w_wrong = barrier.grad_psi_star(z_barrier) + euclid.grad_psi_star(z_euclid)

        # Correct composite inverse
        comp = barrier + euclid
        w_right = comp.grad_psi_star(z_sum)

        # The naive sum should fail
        assert not np.allclose(w_wrong, w, atol=1e-8), (
            "Naive sum of inverses should fail."
        )

        # The composite inverse should succeed
        assert_allclose(w_right, w, atol=1e-12), "Composite inverse must recover w."

    def test_forward_composes_additively(self):
        """Verify that forward (grad_psi) does compose additively."""
        rng = np.random.default_rng(30)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1.0)
        euclid = EuclideanMap(d_coef=2.0)
        diag = DiagonalQuadraticMap(beta=0.3)

        barrier.ensure_dim(d)
        diag.ensure_dim(d)

        barrier._D = rng.random(d) + 0.5
        diag._H_diag = rng.random(d) + 1.0

        w = _rand_positive_simplex(d, rng)

        # Individual forwards
        z1 = barrier.grad_psi(w)
        z2 = euclid.grad_psi(w)
        z3 = diag.grad_psi(w)
        z_sum_manual = z1 + z2 + z3

        # Composite forward
        comp = barrier + euclid + diag
        z_comp = comp.grad_psi(w)

        # Should be identical
        assert_allclose(z_comp, z_sum_manual, rtol=0, atol=1e-14)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_shape_mismatch_detection(self):
        """Ensure shape mismatches raise appropriate errors."""
        d = 4
        barrier = LogBarrierMap()
        barrier.ensure_dim(d)
        barrier._D = np.ones(d)
        comp = barrier + EuclideanMap(d_coef=1.0)

        z_wrong_shape = np.ones(d + 1)

        # Should fail with shape mismatch
        with pytest.raises((ValueError, IndexError)):
            _ = comp.grad_psi_star(z_wrong_shape)

    def test_zero_barrier_zero_euclidean_pure_diagonal(self):
        """Test composition with zero barrier and zero Euclidean (pure diagonal quadratic)."""
        rng = np.random.default_rng(40)
        d = 4

        barrier = LogBarrierMap(barrier_coef=0.0)
        euclid = EuclideanMap(d_coef=0.0)
        diag = DiagonalQuadraticMap(beta=1.0)

        barrier.ensure_dim(d)
        diag.ensure_dim(d)

        barrier._D = np.zeros(d)  # No barrier
        diag._H_diag = rng.random(d) + 2.0  # Only diagonal quadratic

        comp = barrier + euclid + diag

        w = rng.random(d) + 0.5
        z = comp.grad_psi(w)

        # Pure diagonal: w_i = z_i / (beta * H_ii)
        w_expected = z / (diag.beta * diag._H_diag)
        w_hat = comp.grad_psi_star(z)

        assert_allclose(w_hat, w_expected, rtol=1e-10, atol=1e-12)
        assert_allclose(w_hat, w, rtol=1e-10, atol=1e-12)


class TestAdaBarronsStyleComposition:
    """Test the specific Ada-BARRONS composition: barrier + euclidean + diagonal."""

    def test_adabarrons_diagonal_composition_after_updates(self):
        """Simulate Ada-BARRONS updates and verify inverse consistency."""
        rng = np.random.default_rng(50)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1.0)
        euclid = EuclideanMap(d_coef=1.0)
        diag = DiagonalQuadraticMap(beta=0.1)

        barrier.ensure_dim(d)
        diag.ensure_dim(d)

        # Simulate 3 rounds of gradient updates
        gradients = [
            rng.normal(size=d),
            rng.normal(size=d),
            rng.normal(size=d),
        ]

        for g in gradients:
            barrier.update_state(g)
            diag.update_state(g)

        comp = barrier + euclid + diag

        # After updates, verify inverse works
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_hat = comp.grad_psi_star(z)

        assert np.all(w_hat > 0)
        assert_allclose(w_hat, w, rtol=1e-10, atol=1e-12)


class TestAdaBarronsCorrectImplementation:
    """Test the correct Ada-BARRONS implementation with weight-proximity adaptation."""

    def test_ada_barrons_barrier_weight_proximity_update(self):
        """Test that AdaBarronsBarrierMap updates based on weight proximity, not gradients."""
        np.random.default_rng(100)
        d = 5

        barrier = AdaBarronsBarrierMap(barrier_coef=1.0, alpha=0.5)
        barrier.ensure_dim(d)

        # Initial state: D_i should be barrier_coef (proximity_sum = 0)
        initial_D = barrier._get_D()
        assert_allclose(initial_D, np.ones(d), rtol=1e-14)

        # Update with a weight vector (not gradient!)
        w1 = np.array([0.3, 0.2, 0.15, 0.25, 0.1])
        barrier.update_state(w1)

        # After one update: D_i = 1.0 + 0.5 * (1 - w1_i)
        expected_D = 1.0 + 0.5 * (1.0 - w1)
        assert_allclose(barrier._get_D(), expected_D, rtol=1e-14)

        # Second update
        w2 = np.array([0.35, 0.18, 0.12, 0.28, 0.07])
        barrier.update_state(w2)

        # After two updates: D_i = 1.0 + 0.5 * [(1 - w1_i) + (1 - w2_i)]
        expected_D = 1.0 + 0.5 * (2.0 - w1 - w2)
        assert_allclose(barrier._get_D(), expected_D, rtol=1e-14)

    def test_ada_barrons_barrier_roundtrip(self):
        """Test forward-backward consistency for AdaBarronsBarrierMap."""
        rng = np.random.default_rng(101)
        d = 5

        barrier = AdaBarronsBarrierMap(barrier_coef=1.5, alpha=1.0)
        barrier.ensure_dim(d)

        # Simulate some weight history
        for _ in range(3):
            w_past = rng.random(d) * 0.5 + 0.1
            barrier.update_state(w_past)

        # Test roundtrip
        w = _rand_positive_simplex(d, rng)
        z = barrier.grad_psi(w)
        w_recovered = barrier.grad_psi_star(z)

        assert np.all(w_recovered > 0)
        assert_allclose(w_recovered, w, rtol=1e-12, atol=1e-14)

    def test_make_ada_barrons_mirror_map_construction(self):
        """Test the factory function constructs the correct composite."""
        d = 7
        mirror = make_ada_barrons_mirror_map(
            d=d, barrier_coef=1.0, alpha=1.5, euclidean_coef=1.0, beta=0.2
        )

        # Check it's a composite with 3 components
        assert isinstance(mirror, CompositeMirrorMap)
        assert len(mirror.components_) == 3

        # Check component types
        assert isinstance(mirror.components_[0], AdaBarronsBarrierMap)
        assert isinstance(mirror.components_[1], EuclideanMap)
        assert isinstance(mirror.components_[2], FullQuadraticMap)

        # Check parameters
        assert mirror.components_[0].barrier_coef == 1.0
        assert mirror.components_[0].alpha == 1.5
        assert mirror.components_[1].d_coef == 1.0
        assert mirror.components_[2].beta == 0.2

    def test_ada_barrons_full_composite_roundtrip_no_updates(self):
        """Test Ada-BARRONS composite roundtrip without updates."""
        rng = np.random.default_rng(102)
        d = 6

        mirror = make_ada_barrons_mirror_map(
            d=d, barrier_coef=1.0, alpha=1.0, euclidean_coef=1.0, beta=0.1
        )

        w = _rand_positive_simplex(d, rng)
        z = mirror.grad_psi(w)
        w_recovered = mirror.grad_psi_star(z)

        assert np.all(w_recovered > 0)
        # Note: This is a coupled inverse (requires Newton), so tolerance is slightly larger
        assert_allclose(w_recovered, w, rtol=1e-8, atol=1e-10)

    def test_ada_barrons_full_composite_with_updates(self):
        """Test Ada-BARRONS composite after weight and gradient updates."""
        rng = np.random.default_rng(103)
        d = 5

        mirror = make_ada_barrons_mirror_map(
            d=d, barrier_coef=1.0, alpha=0.5, euclidean_coef=1.0, beta=0.1
        )

        # Simulate online updates:
        # - Barrier updates with weight history
        # - Full quadratic updates with gradient history
        for _ in range(3):
            w_past = rng.random(d) * 0.5 + 0.1
            g_past = rng.normal(size=d)

            # Update barrier with weights (Ada-BARRONS style)
            mirror.components_[0].update_state(w_past)

            # Update full quadratic with gradients
            mirror.components_[2].update_state(g_past)

        # Test roundtrip after updates
        w = _rand_positive_simplex(d, rng)
        z = mirror.grad_psi(w)
        w_recovered = mirror.grad_psi_star(z)

        assert np.all(w_recovered > 0)
        # Coupled inverse, slightly larger tolerance
        assert_allclose(w_recovered, w, rtol=1e-7, atol=1e-9)

    def test_ada_barrons_vs_paper_specification(self):
        """
        Verify Ada-BARRONS matches the paper's specification:
        - Barrier adapts based on Σ(1 - w_i), not gradient magnitude
        - Euclidean scales as d * coef
        - Full quadratic accumulates g g^T
        """
        np.random.default_rng(104)
        d = 4

        barrier_coef = 1.0
        alpha = 1.0
        euclidean_coef = 1.0
        beta = 0.1

        mirror = make_ada_barrons_mirror_map(
            d=d,
            barrier_coef=barrier_coef,
            alpha=alpha,
            euclidean_coef=euclidean_coef,
            beta=beta,
        )

        # Simulate 2 rounds
        w1 = np.array([0.4, 0.3, 0.2, 0.1])
        g1 = np.array([0.5, -0.3, 0.2, -0.1])

        w2 = np.array([0.35, 0.35, 0.2, 0.1])
        g2 = np.array([0.3, -0.2, 0.4, -0.05])

        # Round 1 updates
        mirror.components_[0].update_state(w1)
        mirror.components_[2].update_state(g1)

        # Round 2 updates
        mirror.components_[0].update_state(w2)
        mirror.components_[2].update_state(g2)

        # Check barrier coefficients
        expected_D = barrier_coef + alpha * ((1.0 - w1) + (1.0 - w2))
        actual_D = mirror.components_[0]._get_D()
        assert_allclose(actual_D, expected_D, rtol=1e-14)

        # Check full quadratic matrix
        expected_A = np.outer(g1, g1) + np.outer(g2, g2)
        actual_A = mirror.components_[2]._A
        assert_allclose(actual_A, expected_A, rtol=1e-14)

        # Check forward map components
        w_test = np.array([0.3, 0.25, 0.25, 0.2])

        # Barrier contribution: -D / w
        z_barrier = -expected_D / w_test

        # Euclidean contribution: d * euclidean_coef * w
        z_euclid = d * euclidean_coef * w_test

        # Full quadratic contribution: beta * A @ w
        z_quad = beta * expected_A @ w_test

        # Total
        z_expected = z_barrier + z_euclid + z_quad
        z_actual = mirror.grad_psi(w_test)

        assert_allclose(z_actual, z_expected, rtol=1e-12)


class TestTwoComponentCompositions:
    """Test all pairs of basic mirror map compositions."""

    def test_barrier_plus_euclidean(self):
        """LogBarrier + Euclidean composition."""
        rng = np.random.default_rng(200)
        d = 6

        barrier = LogBarrierMap(barrier_coef=2.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 1.0

        euclid = EuclideanMap(d_coef=1.5)

        comp = barrier + euclid
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-10, atol=1e-12)

    def test_barrier_plus_diagonal(self):
        """LogBarrier + DiagonalQuadratic composition."""
        rng = np.random.default_rng(201)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        diag = DiagonalQuadraticMap(beta=0.3)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 2.0

        comp = barrier + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-10, atol=1e-12)

    def test_euclidean_plus_diagonal(self):
        """Euclidean + DiagonalQuadratic composition."""
        rng = np.random.default_rng(202)
        d = 7

        euclid = EuclideanMap(d_coef=2.0)

        diag = DiagonalQuadraticMap(beta=0.5)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.0

        comp = euclid + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-10, atol=1e-12)

    def test_barrier_plus_full_quadratic(self):
        """LogBarrier + FullQuadratic composition (requires Newton)."""
        rng = np.random.default_rng(203)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1.5)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.8

        full = FullQuadraticMap(beta=0.2)
        full.ensure_dim(d)
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = barrier + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Newton solver: slightly larger tolerance
        assert_allclose(w_rec, w, rtol=1e-8, atol=1e-10)

    def test_euclidean_plus_full_quadratic(self):
        """Euclidean + FullQuadratic composition (requires Newton)."""
        rng = np.random.default_rng(204)
        d = 6

        euclid = EuclideanMap(d_coef=1.0)

        full = FullQuadraticMap(beta=0.1)
        full.ensure_dim(d)
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = euclid + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Newton solver
        assert_allclose(w_rec, w, rtol=1e-8, atol=1e-10)

    def test_diagonal_plus_full_quadratic(self):
        """DiagonalQuadratic + FullQuadratic composition (requires Newton)."""
        rng = np.random.default_rng(205)
        d = 5

        diag = DiagonalQuadraticMap(beta=0.4)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.5

        full = FullQuadraticMap(beta=0.15)
        full.ensure_dim(d)
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = diag + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Newton solver
        assert_allclose(w_rec, w, rtol=1e-8, atol=1e-10)


class TestThreeComponentVariations:
    """Test various orderings and coefficient choices for 3-component compositions."""

    def test_all_positive_coefficients(self):
        """All components with substantial positive coefficients."""
        rng = np.random.default_rng(300)
        d = 5

        barrier = LogBarrierMap(barrier_coef=3.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 2.0

        euclid = EuclideanMap(d_coef=5.0)

        diag = DiagonalQuadraticMap(beta=1.0)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 3.0

        comp = barrier + euclid + diag
        w = _rand_positive_simplex(d, rng) * 2.0
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)

    def test_small_coefficients(self):
        """All components with small coefficients."""
        rng = np.random.default_rng(301)
        d = 6

        barrier = LogBarrierMap(barrier_coef=0.01)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) * 0.05 + 0.01

        euclid = EuclideanMap(d_coef=0.01)

        diag = DiagonalQuadraticMap(beta=0.01)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) * 0.1 + 0.05

        comp = barrier + euclid + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)

    def test_mixed_magnitude_coefficients(self):
        """One large, one medium, one small coefficient."""
        rng = np.random.default_rng(302)
        d = 5

        # Large barrier
        barrier = LogBarrierMap(barrier_coef=10.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 5.0

        # Medium Euclidean
        euclid = EuclideanMap(d_coef=1.0)

        # Small diagonal
        diag = DiagonalQuadraticMap(beta=0.01)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) * 0.1 + 0.05

        comp = barrier + euclid + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)

    def test_barrier_dominant(self):
        """Barrier term dominates (Euclidean and diagonal are tiny)."""
        rng = np.random.default_rng(303)
        d = 4

        barrier = LogBarrierMap(barrier_coef=5.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 3.0

        euclid = EuclideanMap(d_coef=1e-6)
        diag = DiagonalQuadraticMap(beta=1e-6)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.0

        comp = barrier + euclid + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)

    def test_quadratic_dominant(self):
        """Quadratic terms dominate (barrier is tiny)."""
        rng = np.random.default_rng(304)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1e-6)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) * 1e-5 + 1e-6

        euclid = EuclideanMap(d_coef=10.0)

        diag = DiagonalQuadraticMap(beta=5.0)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 5.0

        comp = barrier + euclid + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)


class TestAdaptiveMapsCompositions:
    """Test compositions involving adaptive mirror maps."""

    def test_adaptive_mahalanobis_plus_barrier(self):
        """AdaptiveMahalanobis + LogBarrier composition."""
        rng = np.random.default_rng(400)
        d = 5

        adaptive = AdaptiveMahalanobisMap(eps=1e-8)
        adaptive.ensure_dim(d)
        # Simulate gradient accumulation
        for _ in range(3):
            g = rng.normal(size=d)
            adaptive.update_state(g)

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        comp = adaptive + barrier
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)

    def test_adaptive_variation_plus_euclidean(self):
        """AdaptiveVariation + Euclidean composition."""
        rng = np.random.default_rng(401)
        d = 6

        adaptive = AdaptiveVariationMap(eps=1e-12)
        adaptive.ensure_dim(d)
        # Simulate gradient variation accumulation
        gradients = [rng.normal(size=d) for _ in range(4)]
        for g in gradients:
            adaptive.update_state(g)

        euclid = EuclideanMap(d_coef=2.0)

        comp = adaptive + euclid
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)

    def test_ada_barrons_barrier_plus_euclidean_plus_diagonal(self):
        """AdaBarronsBarrier + Euclidean + Diagonal (no full quadratic)."""
        rng = np.random.default_rng(402)
        d = 5

        barrier = AdaBarronsBarrierMap(barrier_coef=1.0, alpha=1.0)
        barrier.ensure_dim(d)
        # Simulate weight history
        for _ in range(3):
            w_past = rng.random(d) * 0.6 + 0.1
            barrier.update_state(w_past)

        euclid = EuclideanMap(d_coef=1.5)

        diag = DiagonalQuadraticMap(beta=0.2)
        diag.ensure_dim(d)
        # Simulate gradient accumulation
        for _ in range(3):
            g = rng.normal(size=d)
            diag.update_state(g)

        comp = barrier + euclid + diag
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-9, atol=1e-11)


class TestFourComponentCompositions:
    """Test compositions with all four component types (requires Newton)."""

    def test_full_stack_barrier_euclidean_diagonal_full(self):
        """LogBarrier + Euclidean + Diagonal + FullQuadratic."""
        rng = np.random.default_rng(500)
        d = 6

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        euclid = EuclideanMap(d_coef=1.0)

        diag = DiagonalQuadraticMap(beta=0.3)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.0

        full = FullQuadraticMap(beta=0.1)
        full.ensure_dim(d)
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = barrier + euclid + diag + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Newton solver with full coupling
        assert_allclose(w_rec, w, rtol=1e-7, atol=1e-9)

    def test_full_stack_weak_coupling(self):
        """Full stack with weak full-quadratic coupling."""
        rng = np.random.default_rng(501)
        d = 5

        barrier = LogBarrierMap(barrier_coef=2.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 1.0

        euclid = EuclideanMap(d_coef=2.0)

        diag = DiagonalQuadraticMap(beta=0.5)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 2.0

        full = FullQuadraticMap(beta=0.01)  # Weak coupling
        full.ensure_dim(d)
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = barrier + euclid + diag + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-7, atol=1e-9)

    def test_full_stack_strong_coupling(self):
        """Full stack with strong full-quadratic coupling."""
        rng = np.random.default_rng(502)
        d = 4

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        euclid = EuclideanMap(d_coef=1.0)

        diag = DiagonalQuadraticMap(beta=0.2)
        diag.ensure_dim(d)
        diag._H_diag = rng.random(d) + 1.0

        full = FullQuadraticMap(beta=1.0)  # Strong coupling
        full.ensure_dim(d)
        B = rng.normal(size=(d, d))
        full._A = (B.T @ B) / d

        comp = barrier + euclid + diag + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Strong coupling may require slightly looser tolerance
        assert_allclose(w_rec, w, rtol=1e-6, atol=1e-8)


class TestExtremeConditions:
    """Test edge cases and extreme parameter values."""

    def test_very_small_weights(self):
        """Test with weights very close to boundary."""
        rng = np.random.default_rng(600)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        euclid = EuclideanMap(d_coef=1.0)

        comp = barrier + euclid

        # Very small weights (close to 0)
        w = rng.random(d) * 0.01 + 1e-4
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-8, atol=1e-10)

    def test_imbalanced_weights(self):
        """Test with very imbalanced weights (one dominant)."""
        rng = np.random.default_rng(601)
        d = 6

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = np.ones(d)

        euclid = EuclideanMap(d_coef=1.0)

        diag = DiagonalQuadraticMap(beta=0.2)
        diag.ensure_dim(d)
        diag._H_diag = np.ones(d)

        comp = barrier + euclid + diag

        # One weight is 0.9, others share 0.1
        w = np.concatenate([np.array([0.9]), rng.random(d - 1) * 0.1])
        w = w / np.sum(w)  # Normalize
        w = np.maximum(w, 1e-8)  # Ensure positivity

        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        assert_allclose(w_rec, w, rtol=1e-7, atol=1e-9)

    def test_high_dimension(self):
        """Test with higher dimension (stress Newton solver)."""
        rng = np.random.default_rng(602)
        d = 20  # Larger dimension

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = rng.random(d) + 0.5

        euclid = EuclideanMap(d_coef=1.0)

        full = FullQuadraticMap(beta=0.05)
        full.ensure_dim(d)
        B = rng.normal(size=(d, d)) / np.sqrt(d)
        full._A = B.T @ B

        comp = barrier + euclid + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Higher dimension may need slightly looser tolerance
        assert_allclose(w_rec, w, rtol=1e-6, atol=1e-8)

    def test_near_singular_coupling(self):
        """Test with near-singular coupling matrix."""
        rng = np.random.default_rng(603)
        d = 5

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = np.ones(d)

        euclid = EuclideanMap(d_coef=1.0)

        full = FullQuadraticMap(beta=0.1)
        full.ensure_dim(d)
        # Create low-rank matrix (near singular)
        v = rng.normal(size=d)
        full._A = np.outer(v, v) / (d * d)  # Rank-1 matrix

        comp = barrier + euclid + full
        w = _rand_positive_simplex(d, rng)
        z = comp.grad_psi(w)
        w_rec = comp.grad_psi_star(z)

        assert np.all(w_rec > 0)
        # Near-singular may need looser tolerance
        assert_allclose(w_rec, w, rtol=1e-6, atol=1e-8)


class TestConsistencyAcrossUpdates:
    """Test that compositions remain consistent as adaptive maps evolve."""

    def test_adaptive_mahalanobis_multiple_updates(self):
        """Test AdaptiveMahalanobis consistency across multiple gradient updates."""
        rng = np.random.default_rng(700)
        d = 5

        adaptive = AdaptiveMahalanobisMap(eps=1e-8)
        adaptive.ensure_dim(d)

        barrier = LogBarrierMap(barrier_coef=1.0)
        barrier.ensure_dim(d)
        barrier._D = np.ones(d)

        comp = adaptive + barrier

        # Test roundtrip after each update
        for iteration in range(5):
            # Update adaptive map
            g = rng.normal(size=d)
            adaptive.update_state(g)

            # Test roundtrip
            w = _rand_positive_simplex(d, rng)
            z = comp.grad_psi(w)
            w_rec = comp.grad_psi_star(z)

            assert np.all(w_rec > 0), f"Failed at iteration {iteration}"
            assert_allclose(
                w_rec,
                w,
                rtol=1e-9,
                atol=1e-11,
                err_msg=f"Failed at iteration {iteration}",
            )

    def test_ada_barrons_barrier_multiple_updates(self):
        """Test AdaBarronsBarrier consistency across weight updates."""
        rng = np.random.default_rng(701)
        d = 6

        barrier = AdaBarronsBarrierMap(barrier_coef=1.0, alpha=0.5)
        barrier.ensure_dim(d)

        euclid = EuclideanMap(d_coef=1.0)

        comp = barrier + euclid

        # Test roundtrip after each weight update
        for iteration in range(5):
            # Update barrier with weight history
            w_history = rng.random(d) * 0.6 + 0.2
            barrier.update_state(w_history)

            # Test roundtrip
            w = _rand_positive_simplex(d, rng)
            z = comp.grad_psi(w)
            w_rec = comp.grad_psi_star(z)

            assert np.all(w_rec > 0), f"Failed at iteration {iteration}"
            assert_allclose(
                w_rec,
                w,
                rtol=1e-9,
                atol=1e-11,
                err_msg=f"Failed at iteration {iteration}",
            )
