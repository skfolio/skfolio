"""Probabilistic PCA Wrapper."""

# Copyright (c) 2026
# Author: Ahmed Nabil Atwa
# SPDX-License-Identifier: BSD-3-Clause
# Wraps gen_fex.PPCA (Apache-2.0) - https://github.com/AI-Ahmed/gen_fex
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
    from gen_fex import PPCA as GenFexPPCA

    _HAS_GEN_FEX = True
except ImportError:
    _HAS_GEN_FEX = False
    GenFexPPCA = None


class PPCA(BaseFeatureExtractor):
    r"""
    Probabilistic Principal Component Analysis.

    Wraps gen_fex.PPCA with automatic JAX-NumPy conversion for seamless
    integration into skfolio's NumPy-based ecosystem.

    PPCA extends traditional PCA with a probabilistic framework, making it
    robust to noise and capable of handling missing data. It automatically
    uses the dual formulation when n_assets > n_observations.

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
        Principal components (loading matrix W from gen_fex).

    mean_vector_ : ndarray of shape (n_observations, 1)
        Estimated mean vector.

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
    >>> from skfolio.feature_extraction import PPCA
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> print(f"Input shape: {X.shape}")  # e.g., (1000, 20)
    >>>
    >>> # Extract latent features
    >>> ppca = PPCA(q=5)
    >>> ppca.fit(X)
    >>> X_reduced = ppca.transform(X)
    >>> print(f"Output shape: {X_reduced.shape}")  # (20, 5) - 5 latent features per asset

    Notes
    -----
    - Requires gen_fex package: pip install 'skfolio[feature_extraction]'
    - Efficiently handles sparse matrices (data with many zeros)
    - Can work with missing values (NaN) via EM, but NaN locations are preserved

    **Automatic Dimension Detection and Latent Factor Interpretation:**

    The transform output is always **(n_assets, q)** - q latent features for each asset.
    However, the interpretation of these latent features depends on your data shape:

    - **Standard Data (n_observations >= n_assets)**:
      Example: (10000 observations, 10 assets)
      Captures latent correlations **between assets** (cross-sectional dependencies)

    - **High-Dimensional Data (n_assets <<  timesteps)**:
      Example: (10000 observations, 50 assets)
      Captures temporal dependencies **within each asset's time series**

    **Why This Matters**: In high-dimensional sparse regimes, attempting to discover latent
    space between assets introduces noise and leads to poor estimates. Instead, modeling
    temporal dependencies discovers hidden events and provides better missing value
    imputation. This adaptive behavior follows the methodology in [2].

    The PPCA model assumes:

    .. math:: x = W z + \mu + \epsilon

    where:
        - :math:`x` is the observed data (n_observations, n_assets)
        - :math:`W` is the loading matrix
        - :math:`z` is the latent variable (q dimensions)
        - :math:`\mu` is the mean vector
        - :math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I)` is Gaussian noise

    References
    ----------
    .. [1] "Probabilistic Principal Component Analysis"
        Tipping, M. E., & Bishop, C. M. (1999).
        Journal of the Royal Statistical Society: Series B (Statistical Methodology),
        61(3), 611-622.

    .. [2] "Generative modeling for high-dimensional sparse data: Probabilistic feature
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
                "gen_fex is required for PPCA. "
                "Install with: pip install 'skfolio[feature_extraction]'"
            )

        self.q = q
        self.prior_sigma = prior_sigma
        self.use_em = use_em
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y=None) -> "PPCA":
        """
        Fit the PPCA model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PPCA
            Fitted estimator.
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

        # Store shape for reference (gen_fex handles primal/dual internally)
        n_observations, n_assets = X.shape
        self._n_observations_fit = n_observations
        self._n_assets_fit = n_assets

        # Convert to JAX array (pass as-is, no transpose)
        X_jax = jax.device_put(X)

        # Initialize gen_fex model with proper seed handling
        seed_value = self.random_state if self.random_state is not None else 42
        self._gen_fex_model = GenFexPPCA(
            q=self.q, prior_sigma=self.prior_sigma, seed=seed_value
        )

        # Fit with JAX arrays
        self._gen_fex_model.fit(
            X_jax, use_em=self.use_em, max_iter=self.max_iter, verbose=0
        )

        # Convert fitted attributes back to NumPy
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
        self._gen_fex_model = GenFexPPCA(
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
        Apply dimensionality reduction to latent space.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets), optional
            Asset returns. If None, transforms the training data.

        Returns
        -------
        X_transformed : ndarray of shape (q, n_assets) or (q, n_timesteps)
            Latent features. Shape is (q, n_assets) for standard data or
            (q, n_timesteps) for high-dimensional data. Interpretation depends
            on data dimensionality:

            - **Standard shape** (n_observations >= n_assets):
              Returns (q, n_assets) - latent factors capture cross-sectional
              correlations between assets

            - **High-dimensional** (n_assets << n_observations):
              Returns (q, n_timesteps) - latent factors capture temporal
              dependencies within time series

        Notes
        -----
        The transformation computes:

        .. math:: z = (W^T W + \sigma^2 I_q)^{-1} W^T (X - \mu)
        """
        skv.check_is_fitted(self, ["load_matrix_", "_gen_fex_model"])

        # Convert to JAX and transform
        if X is None:
            # Transform training data (use stored P from model)
            z_jax = self._gen_fex_model.transform()
        else:
            # Validate input
            X = skv.check_array(
                X,
                dtype=np.float64,
                ensure_all_finite=False,
                accept_sparse=True,
                ensure_2d=True,
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
        r"""
        Transform latent variables back to original space.

        Parameters
        ----------
        X_transformed : array-like of shape (n_assets, q) or (q, timesteps) for high-dimensional data, optional
            Latent features for each asset. If None, uses the latent representation from training data.

        add_noise : bool, default=False
            Whether to add Gaussian noise sampled from N(0, sigma) to the reconstruction.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_observations, n_assets)
            Reconstructed data.

        Notes
        -----
        The reconstruction computes:

        .. math:: X\_{reconstructed} = W z + \mu (+ \epsilon \text{ if add_noise=True})

        where :math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I\_q)`.
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

            # gen_fex expects (q, D) - already in correct format, no transpose needed
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
        r"""
        Generate synthetic samples from the fitted PPCA model.

        The method leverages the generative process defined by the PPCA model parameters.
        Latent variables are sampled from a standard normal distribution and transformed
        into the observed space using the learned loading matrix and mean.

        Parameters
        ----------
        n_samples : int, default=1
            Number of synthetic samples to generate.

        seed : int, optional
            Random seed for reproducibility. If None, uses the model's random_state.

        add_noise : bool, default=True
            Whether to include observation noise (sigma) in the generated samples.

        Returns
        -------
        samples : ndarray of shape (n_samples, D)
            Generated synthetic data where D is determined by gen_fex's internal
            data representation (typically matches the first dimension of training data).

        Notes
        -----
        The generative process follows:

        1. Sample :math:`z \sim \mathcal{N}(0, I_q)` where q is the latent dimension
        2. Compute :math:`x\_{mean} = W z + \mu` (project to observed space)
        3. Add noise: :math:`x = x_{mean} + \epsilon`, where :math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I)` if add_noise=True

        Examples
        --------
        >>> ppca = PPCA(q=5)
        >>> ppca.fit(X_train)
        >>> synthetic_data = ppca.sample(n_samples=100, add_noise=True)
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
