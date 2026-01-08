.. _feature_extraction:

.. currentmodule:: skfolio.feature_extraction

***************************
Feature Extraction
***************************

Feature extraction transformers perform dimensionality reduction while
preserving statistical properties important for portfolio optimization.

Installation
============

The feature extraction module requires the optional gen_fex package::

    pip install 'skfolio[feature_extraction]'

Or using uv::

    uv pip install 'skfolio[feature_extraction]'

Available Transformers
======================

* :class:`PPCA` - Probabilistic Principal Component Analysis
* :class:`PKPCA` - Probabilistic Kernel PCA

Key Benefits
============

1. **Automatic Dimension Detection**: Adapts to data shape
   - Standard data (many observations): Captures asset dependencies
   - High-dimensional data (n_assets << timesteps), where timesteps is the column dimension: Captures temporal patterns
2. **Handles Missing Data**: Native support for incomplete observations
3. **Sparse Matrix Support**: Efficient processing of sparse data
4. **Pipeline Compatible**: Full scikit-learn API support

Understanding Output Dimensions
================================

The output shape is always **(n_assets, q)**, representing q latent features
for each asset. However, the **interpretation** of these latent features depends
on your data dimensionality:

**Standard Data (n_observations >= n_assets)**
  Example: (10000 observations, 10 assets)
  
  - Output: (10 assets, q latent features)
  - Captures: **Cross-sectional dependencies between assets**
  - Each latent feature represents patterns shared across multiple assets

**High-Dimensional Data (n_assets << timesteps)**
  Example: (10 n_assets, 1000 timesteps)
  
  - Output: (q latent features, 10000 timesteps)
  - Captures: **Temporal dependencies within each asset's time series**
  - Each latent feature represents temporal patterns for individual assets

**Why This Matters:**

In standard-sized datasets, we discover latent information between assets through
their cross-sectional dependencies. However, when data dimensionality grows
(especially with sparse data), relying on cross-asset latent space introduces
noise and leads to poor estimates. 

Instead, modeling temporal dependencies discovers hidden events in time series
and provides better missing value imputation. This is because in high-dimensional
regimes, the signal in cross-asset relationships becomes overwhelmed by noise,
while temporal structure within each asset remains more robust.

This adaptive behavior follows the methodology in [1].

Example: Basic Usage
====================

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.feature_extraction import PPCA

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    # Reduce dimensionality
    ppca = PPCA(q=50)
    ppca.fit(X)
    X_reduced = ppca.transform(X)

Example: Pipeline Integration
==============================

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from skfolio.feature_extraction import PPCA
    from skfolio.optimization import MeanRisk

    pipeline = Pipeline([
        ('feature_extraction', PPCA(q=50)),
        ('optimization', MeanRisk())
    ])

    pipeline.fit(X_train)
    portfolio = pipeline.predict(X_test)

Example: Non-linear Reduction with PKPCA
=========================================

.. code-block:: python

    from skfolio.feature_extraction import PKPCA

    # Use kernel trick for non-linear dimensionality reduction
    pkpca = PKPCA(q=30)
    pkpca.fit(X)
    X_reduced = pkpca.transform(X)

    # Reconstruct data
    X_reconstructed = pkpca.inverse_transform(X_reduced)

Mathematical Background
=======================

Probabilistic PCA (PPCA)
-------------------------

PPCA extends traditional PCA with a probabilistic framework. The model assumes:

.. math::

    x = W z + \\mu + \\epsilon

where:

* :math:`x \\in \\mathbb{R}^d` is the observed data
* :math:`W \\in \\mathbb{R}^{d \\times q}` is the loading matrix
* :math:`z \\sim \\mathcal{N}(0, I_q)` is the latent variable
* :math:`\\mu \\in \\mathbb{R}^d` is the mean vector
* :math:`\\epsilon \\sim \\mathcal{N}(0, \\sigma^2 I_d)` is Gaussian noise

The latent variables are computed as:

.. math::

    z = (W^T W + \\sigma^2 I_q)^{-1} W^T (x - \\mu)

Probabilistic Kernel PCA (PKPCA)
---------------------------------

PKPCA extends PPCA to capture non-linear relationships using the kernel trick.
The model works in a high-dimensional feature space :math:`\\phi(x)` defined by
a kernel function :math:`k(x_i, x_j) = \\phi(x_i)^T \\phi(x_j)`.

Common kernels include:

* **RBF**: :math:`k(x_i, x_j) = \\exp(-\\gamma \\|x_i - x_j\\|^2)`
* **Polynomial**: :math:`k(x_i, x_j) = (\\gamma x_i^T x_j + c_0)^d`
* **Linear**: :math:`k(x_i, x_j) = x_i^T x_j`

Performance Considerations
==========================

JAX Compilation
---------------

The first call to any method will be slower due to JAX's Just-In-Time (JIT)
compilation. Subsequent calls with similar input shapes will be much faster.

Memory Usage
------------

JAX arrays consume GPU/CPU memory differently than NumPy arrays. Monitor memory
usage when working with very large datasets.

Dual Formulation
----------------

When the number of assets exceeds the number of observations (d > n), PPCA
automatically uses a more efficient dual formulation, making it ideal for
high-dimensional portfolio optimization problems.

References
==========

.. [1] Atwa, A. N., Kholief, M., & Sedky, A. (2026). "Generative modeling for
   high-dimensional sparse data: Probabilistic feature extraction in high-risk
   financial regimes". Engineering Applications of Artificial Intelligence,
   164, 113376. https://doi.org/10.1016/j.engappai.2025.113376

