"""Probabilistic Kernel PCA Wrapper."""

# Copyright (c) 2026
# Author: Ahmed Nabil Atwa
# SPDX-License-Identifier: BSD-3-Clause
# Wraps gen_fex.PKPCA (Apache-2.0) - https://github.com/AI-Ahmed/gen_fex
# Citation: Atwa, A. N., Kholief, M., & Sedky, A. (2026).
#          Generative modeling for high-dimensional sparse data: Probabilistic feature
#          extraction in high-risk financial regimes.
#          Engineering Applications of Artificial Intelligence, 164, 113376.
#          https://doi.org/10.1016/j.engappai.2025.113376

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

from skfolio.feature_extraction._base import BaseFeatureExtractor

# Import gen_fex with validation
try:
    import jax
    from gen_fex import PKPCA as GenFexPKPCA

    _HAS_GEN_FEX = True
except ImportError:
    _HAS_GEN_FEX = False
    GenFexPKPCA = None


class PKPCA(BaseFeatureExtractor):
    r"""
    Probabilistic Kernel Principal Component Analysis.

    Wraps gen_fex.PKPCA with automatic JAX-NumPy conversion for seamless
    integration into skfolio's NumPy-based ecosystem.

    PKPCA extends PPCA with Wishart process priors and an RBF kernel,
    enabling non-linear dimensionality reduction while maintaining
    probabilistic uncertainty quantification.

    Parameters
    ----------
    q : int, default=2
        Number of latent components (dimensions).

    prior_sigma : float, default=1.0
        Prior variance for the Gaussian prior on the latent variables.

    use_em : bool, default=True
        Whether to use Expectation-Maximization for fitting. If False, uses
        Maximum Likelihood estimation.

    max_iter : int, default=20
        Maximum number of EM iterations.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    load_matrix_ : ndarray of shape (n_observations, q)
        Kernel principal components (loading matrix W from gen_fex).

    mean_vector_ : ndarray of shape (n_observations, 1)
        Estimated mean vector in feature space.

    noise_variance_ : float
        Estimated noise variance (sigma).

    n_features_in_ : int
        Number of assets seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of assets seen during fit (when X has feature names).

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.feature_extraction import PKPCA
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> print(f"Input shape: {X.shape}")  # e.g., (1000, 20)
    >>>
    >>> # Non-linear latent feature extraction with RBF kernel
    >>> pkpca = PKPCA(q=5)
    >>> pkpca.fit(X)
    >>> X_reduced = pkpca.transform(X)
    >>> print(f"Output shape: {X_reduced.shape}")  # (20, 5) - 5 latent features per asset

    Notes
    -----
    - Requires gen_fex package: pip install 'skfolio[feature_extraction]'
    - Efficiently handles sparse matrices (data with many zeros)
    - Uses fixed RBF kernel with gamma automatically derived from noise variance
    - Wishart process prior provides flexible covariance modeling
    - Computationally more expensive than linear PPCA
    - Can work with missing values (NaN) via EM, but NaN locations are preserved

    **Automatic Dimension Detection and Latent Factor Interpretation:**

    The transform output is always **(q, n_assets)** - q latent features for each asset.
    However, the interpretation of these latent features depends on your data shape:

    - **Standard Data (n_observations >= n_assets)**:
      Example: (10000 observations, 10 assets)
      Captures non-linear correlations **between assets** (cross-sectional dependencies)

    - **High-Dimensional Data (n_assets << timesteps)**:
      Example: (10 assets, 10_000 timesteps)
      Captures temporal dependencies **within each asset's time series**

    **Why This Matters**: In high-dimensional sparse regimes, attempting to discover latent
    space between assets introduces noise and leads to poor estimates. Instead, modeling
    temporal dependencies discovers hidden events and provides better missing value
    imputation. This adaptive behavior follows the methodology in [1].

    The PKPCA model uses an RBF kernel matrix:

    .. math:: K_{ij} = \exp\left(-\frac{1}{2\sigma}||x_i - x_j||^2\right)

    where :math:`\sigma` is the learned noise variance.

    References
    ----------
    .. [1] "Generative modeling for high-dimensional sparse data: Probabilistic feature
        extraction in high-risk financial regimes"
        Atwa, A. N., Kholief, M., & Sedky, A. (2026).
        Engineering Applications of Artificial Intelligence, 164, 113376.
        https://doi.org/10.1016/j.engappai.2025.113376
    """

    def __init__(
        self,
        q: int = 2,
        prior_sigma: float = 1.0,
        use_em: bool = True,
        max_iter: int = 20,
        random_state: int = 42,
    ):
        if not _HAS_GEN_FEX:
            raise ImportError(
                "gen_fex is required for PKPCA. "
                "Install with: pip install 'skfolio[feature_extraction]'"
            )

        self.q = q
        self.prior_sigma = prior_sigma
        self.use_em = use_em
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y=None) -> "PKPCA":
        """
        Fit the PKPCA model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PKPCA
            Fitted estimator.
        """
        # Validate and convert to numpy (allow NaN and sparse for EM algorithm)
        X = skv.validate_data(
            self,
            X,
            dtype=np.float64,
            ensure_all_finite=False,  # Allow NaN (EM will impute)
            accept_sparse=True,  # Accept sparse matrices
        )
        # Convert sparse to dense if needed (gen_fex expects dense arrays)
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Store shape for reference (gen_fex handles primal/dual internally)
        n_observations, n_assets = X.shape
        self._n_observations_fit = n_observations
        self._n_assets_fit = n_assets

        # Convert to JAX array (pass as-is, no transpose)
        X_jax = jax.device_put(X)

        # Initialize gen_fex model with proper seed handling
        seed_value = self.random_state if self.random_state is not None else 42
        self._gen_fex_model = GenFexPKPCA(
            q=self.q,
            prior_sigma=self.prior_sigma,
            seed=seed_value,
        )

        # Fit with JAX arrays
        self._gen_fex_model.fit(
            X_jax, use_em=self.use_em, max_iter=self.max_iter, verbose=0
        )

        # Convert fitted attributes back to NumPy
        # gen_fex's W is (n_observations, q) and mu is (n_observations, 1)
        self.load_matrix_ = jax.device_get(self._gen_fex_model.W)
        self.mean_vector_ = jax.device_get(self._gen_fex_model.mu)

        self.noise_variance_ = float(jax.device_get(self._gen_fex_model.sigma))

        return self

    def fit_transform(self, X: npt.ArrayLike, y=None):
        """
        Fit the model and apply dimensionality reduction.

        This method is more efficient than calling fit() then transform() separately,
        as it uses gen_fex's optimized fit_transform implementation.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        X_transformed : ndarray of shape (q, D)
            Latent features. See `transform` for interpretation details.

        Notes
        -----
        For sklearn compatibility, this method always returns only the transformed data.

        If use_em=True, the negative log-likelihood is stored in the `ell_` attribute
        after fitting. This allows the method to work seamlessly in sklearn pipelines while
        still providing access to the training loss for those who need it.
        """
        # Validate and convert to numpy (allow NaN and sparse for EM algorithm)
        X_input = X  # Keep reference for feature names
        X = skv.check_array(
            X,
            dtype=np.float64,
            ensure_all_finite=False,  # Allow NaN (EM will impute)
            accept_sparse=True,  # Accept sparse matrices
            ensure_2d=True,
        )
        # Convert sparse to dense if needed (gen_fex expects dense arrays)
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Set attributes for sklearn compatibility
        self.n_features_in_ = X.shape[1]
        if hasattr(X_input, "columns"):
            self.feature_names_in_ = np.array(X_input.columns, dtype=object)

        # Store shape for reference
        n_observations, n_assets = X.shape
        self._n_observations_fit = n_observations
        self._n_assets_fit = n_assets

        # Convert to JAX array
        X_jax = jax.device_put(X)

        # Initialize gen_fex model
        seed_value = self.random_state if self.random_state is not None else 42
        self._gen_fex_model = GenFexPKPCA(
            q=self.q, prior_sigma=self.prior_sigma, seed=seed_value
        )

        # Call gen_fex's fit_transform
        result = self._gen_fex_model.fit_transform(
            X_jax, use_em=self.use_em, max_iter=self.max_iter, verbose=0
        )

        # Store fitted attributes
        self.load_matrix_ = jax.device_get(self._gen_fex_model.W)
        self.mean_vector_ = jax.device_get(self._gen_fex_model.mu)
        self.noise_variance_ = float(jax.device_get(self._gen_fex_model.sigma))

        # Handle return format: (ell, z) if use_em else z
        # For sklearn compatibility, always return only z, but store ell as attribute
        if self.use_em:
            ell_jax, z_jax = result
            self.ell_ = jax.device_get(ell_jax)  # Store as attribute
            z = jax.device_get(z_jax)
        else:
            z_jax = result
            z = jax.device_get(z_jax)

        return z

    def transform(self, X: npt.ArrayLike | None = None) -> np.ndarray:
        r"""
        Apply non-linear dimensionality reduction to latent space.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets), optional
            Asset returns. If None, transforms the training data.

        Returns
        -------
        X_transformed : ndarray of shape (q, D)
            Latent features. Shape is (q, n_assets) for standard data or
            (q, n_timesteps) for high-dimensional data. Interpretation depends
            on data dimensionality:

            - **Standard shape** (n_observations >= n_assets):
              Returns (q, n_assets) - latent factors capture non-linear cross-sectional
              correlations between assets

            - **High-dimensional** (n_assets << n_observations):
              Returns (q, n_timesteps) - latent factors capture non-linear temporal
              dependencies within time series

        Notes
        -----
        The transformation computes the latent representation using the RBF kernel trick
        with gamma derived from the learned noise variance.
        """
        skv.check_is_fitted(self, ["load_matrix_", "_gen_fex_model"])

        # Convert to JAX and transform
        if X is None:
            # Transform training data
            z_jax = self._gen_fex_model.transform()
        else:
            # Validate input
            X = skv.validate_data(
                self,
                X,
                reset=False,
                dtype=np.float64,
                ensure_all_finite=False,
                accept_sparse=True,
            )
            # Convert sparse to dense if needed
            if hasattr(X, "toarray"):
                X = X.toarray()

            X_jax = jax.device_put(X)
            z_jax = self._gen_fex_model.transform(X_jax)

        # gen_fex returns (q, D) - return as-is
        # Output shape: (q, n_assets) or (q, n_timesteps) depending on data dimensionality
        z = jax.device_get(z_jax)

        return z

    def inverse_transform(
        self, X_transformed: npt.ArrayLike | None = None, add_noise: bool = False
    ) -> np.ndarray:
        """
        Transform latent variables back to original space.

        Parameters
        ----------
        X_transformed : array-like of shape (q, D), optional
            Latent features from transform method. Shape is (q, n_assets) for standard data
            or (q, n_timesteps) for high-dimensional data.
            If None, uses the latent representation from training data.

        add_noise : bool, default=False
            Whether to add Gaussian noise sampled from N(0, sigma) to the reconstruction.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_observations, n_assets)
            Reconstructed data in original space (approximate pre-image).

        Notes
        -----
        The reconstruction involves computing the approximate pre-image from the
        kernel feature space back to the original space. This is an approximation
        since the exact pre-image may not exist.
        """
        skv.check_is_fitted(self, ["load_matrix_", "_gen_fex_model"])

        # Convert to JAX and inverse transform
        if X_transformed is None:
            # Reconstruct from training latent variables
            X_reconstructed_jax = self._gen_fex_model.inverse_transform(
                is_add_noise=add_noise
            )
        else:
            # Validate input: (q, D) format from transform
            z = skv.check_array(X_transformed, dtype=np.float64)

            # gen_fex expects (q, D) - already in correct format
            z_jax = jax.device_put(z)
            X_reconstructed_jax = self._gen_fex_model.inverse_transform(
                z_jax, is_add_noise=add_noise
            )

        # Convert back to NumPy
        # gen_fex returns (n_observations, n_assets) - original shape
        X_reconstructed = jax.device_get(X_reconstructed_jax)

        return X_reconstructed

    def sample(
        self, n_samples: int = 1, seed: int | None = None, add_noise: bool = True
    ) -> np.ndarray:
        """
        Generate synthetic samples using the PKPCA model with Wishart-derived covariance.

        Samples are drawn from a multivariate normal distribution parameterized by
        covariance matrix sampled from Wishart process using the RBF kernel, the learned
        mean vector, and optional observation noise.

        Parameters
        ----------
        n_samples : int, default=1
            Number of synthetic samples to generate.

        seed : int, optional
            Random seed for reproducibility. If None, uses the model's random_state.

        add_noise : bool, default=True
            Whether to add observation noise (sigma) to generated samples.

        Returns
        -------
        samples : ndarray of shape (n_samples, D)
            Generated synthetic data where D is determined by gen_fex's internal
            data representation (typically matches the first dimension of training data).

        Notes
        -----
        The sampling process:

        1. Computes RBF kernel matrix from training data
        2. Samples covariance from Wishart distribution for uncertainty quantification
        3. Samples from multivariate normal with Wishart covariance
        4. Adds heteroskedastic noise if enabled

        This approach maintains temporal structure and captures covariance uncertainty
        for risk-aware generation.

        Examples
        --------
        >>> pkpca = PKPCA(q=5)
        >>> pkpca.fit(X_train)
        >>> synthetic_data = pkpca.sample(n_samples=100, add_noise=True)
        """
        skv.check_is_fitted(self, ["load_matrix_", "_gen_fex_model"])

        # Use model's seed if not provided
        if seed is None:
            seed = self.random_state if self.random_state is not None else 42

        # Call gen_fex's sample method
        samples_jax = self._gen_fex_model.sample(
            n_samples=n_samples, seed=seed, add_noise=add_noise
        )

        # Convert to NumPy
        samples = jax.device_get(samples_jax)

        return samples
