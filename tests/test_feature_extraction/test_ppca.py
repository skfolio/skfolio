"""Tests for PPCA wrapper."""

# Copyright (c) 2026
# Author: Ahmed Nabil Atwa
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.pipeline import Pipeline

from skfolio.datasets import load_sp500_dataset
from skfolio.feature_extraction import PPCA
from skfolio.preprocessing import prices_to_returns

# Conditional import for optional dependency
pytest.importorskip("gen_fex")
pytest.importorskip("jax")


class TestPPCA:
    """Test suite for PPCA wrapper."""

    def test_basic_fit_transform(self):
        """Test basic fit and transform."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=10)
        ppca.fit(X)
        X_transformed = ppca.transform(X)

        assert X_transformed.shape == (10, 50)  # (q, D) format from gen_fex
        assert isinstance(X_transformed, np.ndarray)

    def test_fitted_attributes(self):
        """Test that fitted attributes are properly set."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=10)
        ppca.fit(X)

        assert hasattr(ppca, "load_matrix_")
        assert hasattr(ppca, "mean_vector_")
        assert hasattr(ppca, "noise_variance_")
        assert hasattr(ppca, "n_features_in_")
        assert ppca.load_matrix_.shape == (100, 10)  # (n_observations, q)
        assert ppca.mean_vector_.shape == (100, 1)  # (n_observations, 1)
        assert isinstance(ppca.noise_variance_, float)
        assert ppca.n_features_in_ == 50

    def test_high_dimensional(self):
        """Test with more features than observations (dual formulation)."""
        X = np.random.randn(50, 2000)  # n_assets << timesteps
        ppca = PPCA(q=20)
        ppca.fit(X)

        # load_matrix_ is (q, timesteps) from gen_fex
        assert ppca.load_matrix_.shape == (50, 20)
        assert ppca.n_features_in_ == 2000
        X_transformed = ppca.transform(X)
        # Transform output is (q, timesteps) - captures temporal latents per asset
        assert X_transformed.shape == (20, 2000)

    def test_sparse_matrix_csr(self):
        """Test PPCA with sparse CSR matrix (data with many zeros)."""
        # Create sparse data (60% zeros)
        X_dense = np.random.randn(100, 200)
        X_dense[np.random.rand(100, 200) < 0.6] = 0
        X_sparse = csr_matrix(X_dense)

        ppca = PPCA(q=20)
        ppca.fit(X_sparse)
        X_transformed = ppca.transform(X_sparse)

        assert X_transformed.shape == (20, 200)  # (q, D)
        assert hasattr(ppca, "load_matrix_")
        assert ppca.n_features_in_ == 200

    def test_sparse_matrix_csc(self):
        """Test PPCA with sparse CSC matrix."""
        X_dense = np.random.randn(100, 200)
        X_dense[np.random.rand(100, 200) < 0.6] = 0
        X_sparse = csc_matrix(X_dense)

        ppca = PPCA(q=20)
        ppca.fit(X_sparse)
        X_transformed = ppca.transform(X_sparse)

        assert X_transformed.shape == (20, 200)  # (q, D)

    def test_high_dimensional_sparse(self):
        """Test PPCA with high-dimensional sparse data (n << d)."""
        # Many features, few observations (typical in finance)
        X_dense = np.random.randn(50, 500)
        X_dense[np.random.rand(50, 500) < 0.95] = 0  # 95% sparse
        X_sparse = csr_matrix(X_dense)

        ppca = PPCA(q=30)
        ppca.fit(X_sparse)
        X_transformed = ppca.transform(X_sparse)

        assert X_transformed.shape == (30, 500)  # (q, D)
        assert ppca.load_matrix_.shape == (50, 30)  # (N, q) from gen_fex

    def test_missing_data_handling(self):
        """Test PPCA can fit with NaN values via EM algorithm."""
        X = np.random.randn(100, 50)
        nan_mask = np.random.rand(100, 50) < 0.1
        X[nan_mask] = np.nan

        ppca = PPCA(q=10, use_em=True)
        # Should be able to fit with NaN values
        ppca.fit(X)

        # Model should be fitted
        assert hasattr(ppca, "load_matrix_")
        assert hasattr(ppca, "mean_vector_")

        # Note: gen_fex preserves NaN locations in reconstruction
        # For full imputation, use sklearn.impute before fitting

    def test_pandas_dataframe_support(self):
        """Test with pandas DataFrame (feature names preservation)."""
        asset_names = [f"Asset_{i}" for i in range(50)]
        X = pd.DataFrame(np.random.randn(100, 50), columns=asset_names)

        ppca = PPCA(q=10)
        ppca.fit(X)

        assert hasattr(ppca, "feature_names_in_")
        assert list(ppca.feature_names_in_) == asset_names

    def test_pipeline_integration(self):
        """Test PPCA can be used in sklearn Pipeline for feature extraction."""
        prices = load_sp500_dataset()
        X = prices_to_returns(prices)

        # PPCA transforms (n_observations, n_assets) → (q, timesteps)
        # This is useful for feature extraction but not directly with MeanRisk
        # which expects (n_observations, n_assets) format

        ppca = PPCA(q=10)
        ppca.fit(X)
        X_transformed = ppca.transform(X)

        # Verify the output shape
        assert X_transformed.shape == (10, X.shape[1])  # (q, timesteps)

        # Can use in pipeline with other sklearn transformers
        import warnings

        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            [
                ("ppca", PPCA(q=10)),
                ("scaler", StandardScaler()),  # Works on (q, timesteps) data
            ]
        )

        # Suppress sklearn RuntimeWarnings about zero variance features
        # (expected when PPCA output has constant features)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )
            pipeline.fit(X)
            result = pipeline.transform(X)

        # PPCA outputs (q, timesteps), StandardScaler standardizes along features (axis=0)
        # So output is still (q, timesteps) = (10, 20)
        assert result.shape == (10, X.shape[1])

    def test_inverse_transform_reconstruction(self):
        """Test reconstruction quality."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=40)  # High q for good reconstruction
        ppca.fit(X)

        X_transformed = ppca.transform(X)
        X_reconstructed = ppca.inverse_transform(X_transformed)

        # Should be close to original
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        assert reconstruction_error < 0.5

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.random.randn(100, 50)

        # Test with use_em=True (stores ell in attribute, returns X_transformed)
        ppca_em = PPCA(q=10, random_state=42, use_em=True)
        X_transformed_em = ppca_em.fit_transform(X)

        assert hasattr(ppca_em, "ell_")  # Negative log-likelihood stored as attribute
        assert isinstance(ppca_em.ell_, np.ndarray)
        assert X_transformed_em.shape == (10, 50)  # (q, D) for standard data

        # Test with use_em=False (no ell attribute)
        ppca_ml = PPCA(q=10, random_state=42, use_em=False)
        X_transformed_ml = ppca_ml.fit_transform(X)

        assert not hasattr(ppca_ml, "ell_")  # No ell for ML estimation
        assert X_transformed_ml.shape == (10, 50)  # (q, D) for standard data

        # Verify fit_transform is consistent with fit().transform()
        ppca_check = PPCA(q=10, random_state=42, use_em=True)
        ppca_check.fit(X)
        X_transformed_check = ppca_check.transform(X)

        assert_array_almost_equal(X_transformed_em, X_transformed_check, decimal=5)

    def test_transform_without_x(self):
        """Test transform without X (returns training transformation)."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=10)
        ppca.fit(X)

        # Transform without X should return training latent features
        X_transformed_1 = ppca.transform()
        X_transformed_2 = ppca.transform(X)

        # Should be close
        assert_array_almost_equal(X_transformed_1, X_transformed_2, decimal=5)

    def test_inverse_transform_without_z(self):
        """Test inverse transform without z (reconstructs training data)."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=40)
        ppca.fit(X)

        # Inverse transform without z should reconstruct training data
        X_reconstructed = ppca.inverse_transform()

        # Should be close to original
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        assert reconstruction_error < 0.5

    def test_random_state_reproducibility(self):
        """Test random_state provides reproducible results."""
        X = np.random.randn(100, 50)

        ppca1 = PPCA(q=10, random_state=42)
        ppca1.fit(X)

        ppca2 = PPCA(q=10, random_state=42)
        ppca2.fit(X)

        assert_array_almost_equal(ppca1.load_matrix_, ppca2.load_matrix_, decimal=5)
        assert_array_almost_equal(ppca1.mean_vector_, ppca2.mean_vector_, decimal=5)

    def test_different_random_states(self):
        """Test different random_state gives different results."""
        X = np.random.randn(100, 50)

        ppca1 = PPCA(q=10, random_state=42)
        ppca1.fit(X)

        ppca2 = PPCA(q=10, random_state=99)
        ppca2.fit(X)

        # Should be different (with very high probability)
        assert not np.allclose(ppca1.load_matrix_, ppca2.load_matrix_)

    def test_add_noise_parameter(self):
        """Test inverse_transform with add_noise parameter."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=10)
        ppca.fit(X)

        X_transformed = ppca.transform(X)

        # Without noise
        X_recon_1 = ppca.inverse_transform(X_transformed, add_noise=False)
        X_recon_2 = ppca.inverse_transform(X_transformed, add_noise=False)

        # Should be identical without noise
        assert_array_almost_equal(X_recon_1, X_recon_2, decimal=10)

        # With noise - gen_fex uses fixed seed so noise will be reproducible
        X_recon_with_noise = ppca.inverse_transform(X_transformed, add_noise=True)

        # Reconstruction with noise should differ from without noise
        assert not np.allclose(X_recon_1, X_recon_with_noise, atol=1e-5)

    def test_use_em_parameter(self):
        """Test use_em parameter affects fitting."""
        X = np.random.randn(100, 50)

        ppca_em = PPCA(q=10, use_em=True)
        ppca_em.fit(X)

        ppca_ml = PPCA(q=10, use_em=False)
        ppca_ml.fit(X)

        # Both should fit successfully
        assert hasattr(ppca_em, "load_matrix_")
        assert hasattr(ppca_ml, "load_matrix_")

    def test_prior_sigma_parameter(self):
        """Test prior_sigma parameter."""
        X = np.random.randn(100, 50)

        ppca1 = PPCA(q=10, prior_sigma=1.0)
        ppca1.fit(X)

        ppca2 = PPCA(q=10, prior_sigma=2.0)
        ppca2.fit(X)

        # Different priors should give different results
        assert not np.allclose(ppca1.load_matrix_, ppca2.load_matrix_)

    def test_max_iter_parameter(self):
        """Test max_iter parameter."""
        X = np.random.randn(100, 50)

        # Should work with different max_iter values
        ppca = PPCA(q=10, max_iter=5, use_em=True)
        ppca.fit(X)

        assert hasattr(ppca, "load_matrix_")

    def test_small_q(self):
        """Test with very small q."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=2)
        ppca.fit(X)

        X_transformed = ppca.transform(X)
        assert X_transformed.shape == (2, 50)  # (q, timesteps)

    def test_large_q(self):
        """Test with q close to min(n, d)."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=45)
        ppca.fit(X)

        X_transformed = ppca.transform(X)
        assert X_transformed.shape == (45, 50)  # (q, timesteps)

    def test_high_dimensional_temporal_latents(self):
        """Test that high-dimensional data (N < D) captures temporal dependencies."""
        # High-dimensional case: few observations, many assets
        X = np.random.randn(50, 200)  # 50 observations, 200 assets
        ppca = PPCA(q=10)
        ppca.fit(X)
        X_transformed = ppca.transform(X)

        # Output: (q, D) where D=200 (n_assets)
        assert X_transformed.shape == (10, 200)

        # Reconstruction should match original shape
        X_reconstructed = ppca.inverse_transform(X_transformed)
        assert X_reconstructed.shape == (50, 200)

        # Model attributes (W and mu) are (N, q) and (N, 1) from gen_fex
        assert ppca.load_matrix_.shape == (50, 10)
        assert ppca.mean_vector_.shape == (50, 1)

    def test_sample(self):
        """Test sample method for generating synthetic data."""
        X = np.random.randn(100, 50)
        ppca = PPCA(q=10, random_state=42)
        ppca.fit(X)

        # Generate samples
        # gen_fex returns (n_samples, D) where D is the second dimension after data_reshape
        # For standard data, this is the number of features in the reshaped space
        samples = ppca.sample(n_samples=20, add_noise=True)

        assert samples.shape == (20, 100)  # (n_samples, D) from gen_fex
        assert isinstance(samples, np.ndarray)
        assert not np.any(np.isnan(samples))

        # Test without noise
        samples_no_noise = ppca.sample(n_samples=10, add_noise=False)
        assert samples_no_noise.shape == (10, 100)

        # Test reproducibility with seed
        samples1 = ppca.sample(n_samples=5, seed=123, add_noise=True)
        samples2 = ppca.sample(n_samples=5, seed=123, add_noise=True)
        assert_array_almost_equal(samples1, samples2, decimal=5)

    def test_sklearn_compatibility(self):
        """Test sklearn estimator compatibility."""

        # Note: This will skip if gen_fex is not installed
        # We just test that PPCA follows sklearn conventions
        ppca = PPCA(q=2)
        assert hasattr(ppca, "fit")
        assert hasattr(ppca, "transform")
