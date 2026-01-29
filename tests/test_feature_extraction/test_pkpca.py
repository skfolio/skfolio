"""Tests for PKPCA wrapper."""

# Copyright (c) 2026
# Author: Ahmed Nabil Atwa
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csc_matrix, csr_matrix

# Conditional import for optional dependency
pytest.importorskip("gen_fex")
pytest.importorskip("jax")

from skfolio.feature_extraction import PKPCA


class TestPKPCA:
    """Test suite for PKPCA wrapper."""

    def test_basic_fit_transform(self):
        """Test basic fit and transform."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=10)
        pkpca.fit(X)
        X_transformed = pkpca.transform(X)

        assert X_transformed.shape == (10, 50)  # (q, timesteps)
        assert isinstance(X_transformed, np.ndarray)

    def test_fitted_attributes(self):
        """Test that fitted attributes are properly set."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=10)
        pkpca.fit(X)

        assert hasattr(pkpca, "load_matrix_")
        assert hasattr(pkpca, "mean_vector_")
        assert hasattr(pkpca, "noise_variance_")
        assert hasattr(pkpca, "n_features_in_")
        assert pkpca.load_matrix_.shape == (100, 10)  # (n_observations, q)
        assert pkpca.mean_vector_.shape == (100, 1)  # (n_observations, 1)
        assert isinstance(pkpca.noise_variance_, float)
        assert pkpca.n_features_in_ == 50

    def test_high_dimensional(self):
        """Test with more features than observations."""
        X = np.random.randn(50, 2000)  # n_assets << timesteps
        pkpca = PKPCA(q=20)
        pkpca.fit(X)

        # load_matrix_ is (n_observations, q) from gen_fex
        assert pkpca.load_matrix_.shape == (50, 20)
        assert pkpca.n_features_in_ == 2000
        X_transformed = pkpca.transform(X)

        # Transform output is (q, timesteps) - captures temporal latents per asset
        assert X_transformed.shape == (20, 2000)

    def test_pandas_dataframe_support(self):
        """Test with pandas DataFrame (feature names preservation)."""
        asset_names = [f"Asset_{i}" for i in range(50)]
        X = pd.DataFrame(np.random.randn(100, 50), columns=asset_names)

        pkpca = PKPCA(q=10)
        pkpca.fit(X)

        assert hasattr(pkpca, "feature_names_in_")
        assert list(pkpca.feature_names_in_) == asset_names

    def test_sparse_matrix_csr(self):
        """Test PKPCA with sparse CSR matrix (data with many zeros)."""
        # Create sparse data (60% zeros)
        X_dense = np.random.randn(100, 200)
        X_dense[np.random.rand(100, 200) < 0.6] = 0
        X_sparse = csr_matrix(X_dense)

        pkpca = PKPCA(q=20)
        pkpca.fit(X_sparse)
        X_transformed = pkpca.transform(X_sparse)

        assert X_transformed.shape == (20, 200)  # (q, timesteps)
        assert hasattr(pkpca, "load_matrix_")
        assert pkpca.n_features_in_ == 200

    def test_sparse_matrix_csc(self):
        """Test PKPCA with sparse CSC matrix."""
        X_dense = np.random.randn(100, 200)
        X_dense[np.random.rand(100, 200) < 0.9] = 0
        X_sparse = csc_matrix(X_dense)

        pkpca = PKPCA(q=20)
        pkpca.fit(X_sparse)
        X_transformed = pkpca.transform(X_sparse)

        assert X_transformed.shape == (20, 200)  # (q, timesteps)

    def test_high_dimensional_sparse(self):
        """Test PKPCA with high-dimensional sparse data (n << d)."""
        # Many features, few observations (typical in finance)
        X_dense = np.random.randn(50, 500)
        X_dense[np.random.rand(50, 500) < 0.95] = 0  # 95% sparse
        X_sparse = csr_matrix(X_dense)

        pkpca = PKPCA(q=30)
        pkpca.fit(X_sparse)
        X_transformed = pkpca.transform(X_sparse)

        assert X_transformed.shape == (30, 500)  # (q, timesteps)
        assert pkpca.load_matrix_.shape == (50, 30)  # (N, q) from gen_fex

    def test_inverse_transform_reconstruction(self):
        """Test reconstruction (approximate pre-image)."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=40)
        pkpca.fit(X)

        X_transformed = pkpca.transform(X)
        X_reconstructed = pkpca.inverse_transform(X_transformed)

        # Kernel methods give approximate reconstruction
        assert X_reconstructed.shape == X.shape
        assert not np.any(np.isnan(X_reconstructed))

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.random.randn(100, 50)

        # Test with use_em=True (stores ell in attribute, returns X_transformed)
        pkpca_em = PKPCA(q=10, random_state=42, use_em=True)
        X_transformed_em = pkpca_em.fit_transform(X)

        assert hasattr(pkpca_em, "ell_")  # Negative log-likelihood stored as attribute
        assert isinstance(pkpca_em.ell_, np.ndarray)
        assert X_transformed_em.shape == (10, 50)  # (q, D) for standard data

        # Test with use_em=False (no ell attribute)
        pkpca_ml = PKPCA(q=10, random_state=42, use_em=False)
        X_transformed_ml = pkpca_ml.fit_transform(X)

        assert not hasattr(pkpca_ml, "ell_")  # No ell for ML estimation
        assert X_transformed_ml.shape == (10, 50)  # (q, D) for standard data

        # Verify fit_transform is consistent with fit().transform()
        pkpca_check = PKPCA(q=10, random_state=42, use_em=True)
        pkpca_check.fit(X)
        X_transformed_check = pkpca_check.transform(X)

        assert_array_almost_equal(X_transformed_em, X_transformed_check, decimal=5)

    def test_transform_without_x(self):
        """Test transform without X (returns training transformation)."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=10)
        pkpca.fit(X)

        # Transform without X should return training latent features
        X_transformed = pkpca.transform()
        assert X_transformed.shape == (10, 50)  # (q, n_assets)

    def test_inverse_transform_without_z(self):
        """Test inverse transform without z (reconstructs training data)."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=40)
        pkpca.fit(X)

        # Inverse transform without z should reconstruct training data
        X_reconstructed = pkpca.inverse_transform()
        assert X_reconstructed.shape == X.shape

    def test_random_state_reproducibility(self):
        """Test random_state provides reproducible results."""
        X = np.random.randn(100, 50)

        pkpca1 = PKPCA(q=10, random_state=42)
        pkpca1.fit(X)

        pkpca2 = PKPCA(q=10, random_state=42)
        pkpca2.fit(X)

        assert_array_almost_equal(pkpca1.load_matrix_, pkpca2.load_matrix_, decimal=4)
        assert_array_almost_equal(pkpca1.mean_vector_, pkpca2.mean_vector_, decimal=4)

    def test_different_random_states(self):
        """Test different random_state gives different results."""
        X = np.random.randn(100, 50)

        pkpca1 = PKPCA(q=10, random_state=42)
        pkpca1.fit(X)

        pkpca2 = PKPCA(q=10, random_state=99)
        pkpca2.fit(X)

        # Should be different (with very high probability)
        assert not np.allclose(pkpca1.load_matrix_, pkpca2.load_matrix_, atol=1e-3)

    def test_add_noise_parameter(self):
        """Test inverse_transform with add_noise parameter."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=10)
        pkpca.fit(X)

        X_transformed = pkpca.transform(X)

        # Without noise
        X_recon_1 = pkpca.inverse_transform(X_transformed, add_noise=False)
        X_recon_2 = pkpca.inverse_transform(X_transformed, add_noise=False)

        # Should be identical without noise
        assert_array_almost_equal(X_recon_1, X_recon_2, decimal=5)

        # With noise - gen_fex uses fixed seed so noise will be reproducible
        X_recon_with_noise = pkpca.inverse_transform(X_transformed, add_noise=True)

        # Reconstruction with noise should differ from without noise
        assert not np.allclose(X_recon_1, X_recon_with_noise, atol=1e-5)

    def test_use_em_parameter(self):
        """Test use_em parameter affects fitting."""
        X = np.random.randn(100, 50)

        pkpca_em = PKPCA(q=10, use_em=True, random_state=42)
        pkpca_em.fit(X)

        pkpca_ml = PKPCA(q=10, use_em=False, random_state=42)
        pkpca_ml.fit(X)

        # Both should fit successfully
        assert hasattr(pkpca_em, "load_matrix_")
        assert hasattr(pkpca_ml, "load_matrix_")

        # Results should be different
        assert not np.allclose(pkpca_em.load_matrix_, pkpca_ml.load_matrix_, atol=1e-3)

    def test_prior_sigma_parameter(self):
        """Test prior_sigma parameter."""
        X = np.random.randn(100, 50)

        pkpca1 = PKPCA(q=10, prior_sigma=1.0, random_state=42)
        pkpca1.fit(X)

        pkpca2 = PKPCA(q=10, prior_sigma=2.0, random_state=42)
        pkpca2.fit(X)

        # Different priors should give different results
        assert not np.allclose(pkpca1.load_matrix_, pkpca2.load_matrix_, atol=1e-3)

    def test_max_iter_parameter(self):
        """Test max_iter parameter."""
        X = np.random.randn(100, 50)

        # Should work with different max_iter values
        pkpca = PKPCA(q=10, max_iter=5, use_em=True, random_state=42)
        pkpca.fit(X)

        assert hasattr(pkpca, "load_matrix_")

    def test_small_q(self):
        """Test with very small q."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=2)
        pkpca.fit(X)

        X_transformed = pkpca.transform(X)
        assert X_transformed.shape == (2, 50)  # (q, n_assets)

    def test_large_q(self):
        """Test with q close to min(n, d)."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=45)
        pkpca.fit(X)

        X_transformed = pkpca.transform(X)
        assert X_transformed.shape == (45, 50)  # (q, n_assets)

    def test_nonlinear_structure_capture(self):
        """Test that PKPCA captures non-linear structure with RBF kernel."""
        # Create non-linear data
        t = np.linspace(0, 2 * np.pi, 100)
        X = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
        X = np.tile(X, (1, 12))  # Make it 48-dimensional

        # RBF kernel (fixed in gen_fex) should work
        pkpca = PKPCA(q=5, random_state=42)
        pkpca.fit(X)

        X_transformed = pkpca.transform(X)
        assert X_transformed.shape == (5, 48)  # (q, features) - 48 features from tiling
        assert not np.any(np.isnan(X_transformed))

        # Should capture structure
        assert hasattr(pkpca, "load_matrix_")

    def test_high_dimensional_temporal_latents(self):
        """Test that high-dimensional data (N < D) captures temporal dependencies."""
        # High-dimensional case: few observations, many assets
        X = np.random.randn(50, 2000)  # 50 assets, 2000 timesteps
        pkpca = PKPCA(q=10)
        pkpca.fit(X)
        X_transformed = pkpca.transform(X)

        # Output: (10 temporal latent features per asset, 2000 timesteps)
        assert X_transformed.shape == (10, 2000)

        # Reconstruction should match original shape
        X_reconstructed = pkpca.inverse_transform(X_transformed)
        assert X_reconstructed.shape == (50, 2000)

        # Model attributes (W and mu) are (N, q) and (N, 1) from gen_fex
        assert pkpca.load_matrix_.shape == (50, 10)
        assert pkpca.mean_vector_.shape == (50, 1)

    def test_sample(self):
        """Test sample method for generating synthetic data."""
        X = np.random.randn(100, 50)
        pkpca = PKPCA(q=10, random_state=42)
        pkpca.fit(X)

        # Generate samples
        # gen_fex returns (n_samples, N) where N is first dimension of training data
        samples = pkpca.sample(n_samples=20, add_noise=True)

        assert samples.shape == (20, 100)  # (n_samples, n_observations)
        assert isinstance(samples, np.ndarray)
        assert not np.any(np.isnan(samples))

        # Test without noise
        samples_no_noise = pkpca.sample(n_samples=10, add_noise=False)
        assert samples_no_noise.shape == (10, 100)

        # Test reproducibility with seed
        samples1 = pkpca.sample(n_samples=5, seed=123, add_noise=True)
        samples2 = pkpca.sample(n_samples=5, seed=123, add_noise=True)
        assert_array_almost_equal(samples1, samples2, decimal=5)

    def test_sklearn_compatibility(self):
        """Test sklearn estimator compatibility."""
        # Basic compatibility check
        pkpca = PKPCA(q=2)
        assert hasattr(pkpca, "fit")
        assert hasattr(pkpca, "transform")
        assert hasattr(pkpca, "inverse_transform")
        assert hasattr(pkpca, "get_params")
        assert hasattr(pkpca, "set_params")

    def test_no_gen_fex_error_message(self, monkeypatch):
        """Test helpful error when gen_fex not installed."""
        # Mock missing gen_fex
        monkeypatch.setattr("skfolio.feature_extraction._pkpca._HAS_GEN_FEX", False)

        with pytest.raises(ImportError, match="gen_fex is required"):
            PKPCA(q=10)
