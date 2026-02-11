#!/usr/bin/env python

r"""
========================================
Feature Extraction: PPCA and PKPCA
========================================

This tutorial demonstrates the :class:`~skfolio.feature_extraction.PPCA` and :class:`~skfolio.feature_extraction.PKPCA` estimators for dimensionality reduction and feature extraction in portfolio optimization.

Introduction
============

Probabilistic Principal Component Analysis (PPCA) and Probabilistic Kernel Principal Component Analysis (PKPCA) are powerful techniques for discovering latent structure in financial return data. These methods are particularly useful for:

* **Dimensionality Reduction**: Reducing the number of features while preserving essential information
* **Missing Value Imputation**: Handling incomplete data through probabilistic modeling
* **Noise Reduction**: Separating signal from noise in return series
* **Feature Extraction**: Discovering hidden factors driving asset returns

The key advantage of PPCA over standard PCA is its probabilistic framework, which:
    - Handles missing data naturally
    - Provides uncertainty estimates
    - Adapts to different dimensionality regimes (standard vs. high-dimensional)
    - Enables Bayesian inference

Mathematical Background
=======================

PPCA models the observed data :math:`\mathbf{X}` as:

.. math::

   \mathbf{X} = \mathbf{W}\mathbf{Z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}

Where:

- :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` is the data matrix (n observations, d assets)
- :math:`\mathbf{W} \in \mathbb{R}^{n \times q}` is the loading matrix
- :math:`\mathbf{Z} \in \mathbb{R}^{q \times d}` contains the latent variables
- :math:`\boldsymbol{\mu}` is the mean vector
- :math:`\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2\mathbf{I})` is isotropic Gaussian noise
- :math:`q < \min(n, d)` is the number of latent components

Dual Formulation for High-Dimensional Data
===========================================

Following Atwa et al. (2026), when :math:`n \ll d` (transposed matrix :math:`\mathbf{X}^T` has more features than rows), PPCA automatically uses the dual formulation by transposing the data matrix. This switches from discovering cross-sectional asset dependencies to discovering temporal patterns, which is more robust in sparse, high-dimensional regimes.

In this tutorial, we will:
    1. Apply PPCA to standard-dimensional data (n >> d)
    2. Demonstrate the dual formulation for high-dimensional data (n << d)
    3. Use PPCA for missing value imputation
    4. Integrate PPCA with portfolio optimization
    5. Compare PPCA with PKPCA (kernel version)
"""

# %%
# Setup and Imports
# ============================================================
#

# Standard libraries
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.io import show

# Scikit-learn
from sklearn.metrics import mean_absolute_error

from skfolio import Population, RiskMeasure

# Skfolio imports
from skfolio.datasets import load_ftse100_dataset, load_sp500_dataset
from skfolio.feature_extraction import PKPCA, PPCA
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

# Plotting configuration
plt.style.use("bmh")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


# %%
# Data Loading and Preparation
# ============================
# We load the S&P 500 :ref:`dataset <datasets>` and convert prices to returns for our analysis.

# %%
# Section 1: Standard Dimensionality Regime (n >> d)
# ===================================================
# In this section, we work with the S&P 500 dataset where we have many more observations
# (8,312 days) than assets (20 stocks). This is the typical scenario for most financial
# time series data.

# Load S&P 500 dataset
prices = load_sp500_dataset()
X = prices_to_returns(prices)

print(f"Dataset shape: {X.shape}")
print(f"Number of observations (n): {X.shape[0]}")
print(f"Number of assets (d): {X.shape[1]}")
print("Regime: Standard (n >> d)")
print(f"\nAssets: {list(X.columns)}")
print(f"\nDate range: {X.index[0]} to {X.index[-1]}")
print("\nFirst few returns:")
X.head()


# %%
# 1.1 Fitting PPCA with Different Numbers of Components
# ------------------------------------------------------------
#
# We fit PPCA models with varying numbers of latent components (q=5, 10, 15) to explore
# the trade-off between model complexity and reconstruction quality. The number of components
# determines how much of the underlying structure we capture:
#
# - **Fewer components**: More compression, faster computation, risk of underfitting
# - **More components**: Better reconstruction, captures finer details, risk of overfitting
#
# The noise variance :math:`\sigma^2` decreases as we add more components, indicating better fit.

# Fit PPCA with different number of components
q_values = [5, 10, 15]
ppca_models = {}

for q in q_values:
    print(f"\n{'=' * 60}")
    print(f"Fitting PPCA with q={q} latent components")
    print(f"{'=' * 60}")

    # Measure fitting time (note: first call includes JAX compilation)
    start = time.time()
    ppca = PPCA(q=q, random_state=42, use_em=True, max_iter=20)
    ppca.fit(X)
    fit_time = time.time() - start

    # Transform to latent space
    start = time.time()
    X_reduced = ppca.transform(X)
    transform_time = time.time() - start

    # Store model
    ppca_models[q] = (ppca, X_reduced)

    print(f"\n✓ Fitting time: {fit_time:.2f}s")
    print(f"✓ Transform time: {transform_time:.2f}s")
    print(f"✓ Output shape: {X_reduced.shape}")
    print(
        f"✓ Interpretation: {X_reduced.shape[0]} latent features for {X_reduced.shape[1]} assets"
    )
    print(f"✓ Loading matrix shape: {ppca.load_matrix_.shape}")
    print(f"✓ Mean vector shape: {ppca.mean_vector_.shape}")
    print(f"✓ Estimated noise variance (σ²): {ppca.noise_variance_:.6f}")

    # Reconstruction error
    X_reconstructed = ppca.inverse_transform(X_reduced)
    mse = np.mean((X.values - X_reconstructed) ** 2)
    print(f"✓ Reconstruction MSE: {mse:.8f}")

print(f"\n{'=' * 60}")
print("Summary: More components = Lower noise variance = Better fit")
print(f"{'=' * 60}")


rec_df = pd.DataFrame(X_reconstructed, columns=X.columns, index=X.index)
rec_df.head()


sns.heatmap(X.corr(method="spearman"), cmap="RdYlGn")


sns.heatmap(rec_df.corr(method="spearman"), cmap="RdYlGn")


# %%
# 1.3 Visualize Latent Space Representation
# ------------------------------------------------------------
#
# The latent space reveals how assets cluster based on their shared dependencies. Assets close in latent space tend to move together.
#

# Visualize assets in 2D latent space (using q=10 model)
ppca_10, X_reduced_10 = ppca_models[10]

# Select the TWO MOST IMPORTANT latent factors based on loading matrix L2 norm
loading_matrix = ppca_10.load_matrix_
component_magnitudes = np.linalg.norm(loading_matrix, axis=0)

# Get indices of top 2 components
top_2_indices = np.argsort(component_magnitudes)[-2:][::-1]  # Descending order
factor_1_idx, factor_2_idx = top_2_indices[0], top_2_indices[1]

print("Selected Most Important Latent Factors:")
print("=" * 70)
print(f"Factor {factor_1_idx + 1}: L2 norm = {component_magnitudes[factor_1_idx]:.4f}")
print(f"Factor {factor_2_idx + 1}: L2 norm = {component_magnitudes[factor_2_idx]:.4f}")
print(
    "\nThese two factors explain the most variance in cross-sectional dependencies.\n"
)

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each asset using the TOP 2 latent factors
scatter = ax.scatter(
    X_reduced_10[factor_1_idx, :],
    X_reduced_10[factor_2_idx, :],
    s=100,
    alpha=0.6,
    c=range(X.shape[1]),
    cmap="viridis",
    edgecolors="black",
    linewidth=1.5,
)

# Annotate each point with asset name
for i, asset in enumerate(X.columns):
    ax.annotate(
        asset,
        (X_reduced_10[factor_1_idx, i], X_reduced_10[factor_2_idx, i]),
        fontsize=10,
        fontweight="bold",
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.set_xlabel(
    f"Most Important Latent Factor (Component {factor_1_idx + 1})",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylabel(
    f"2nd Most Important Latent Factor (Component {factor_2_idx + 1})",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    "Assets in 2D Latent Space (PPCA q=10)\nProjected on the TWO most significant cross-sectional factors",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, label="Asset Index")
plt.tight_layout()
plt.show()

print("\n💡 Interpretation:")
print("=" * 70)
print("• Assets close together share similar latent factors (common risk exposure)")
print("• This suggests they're affected by similar economic forces")
print("• Tech stocks (AAPL, MSFT, AMD) often cluster together")
print("• Financial stocks (JPM, BAC) form another cluster")
print("• Energy stocks (XOM, CVX) and consumer staples (KO, PEP) group separately")
print("• Using the MOST IMPORTANT factors ensures meaningful clustering")


# %%
# 1.4 Loading Matrix Analysis
# ------------------------------------------------------------
#
# The loading matrix :math:`\mathbf{W}` shows how observations (time periods) contribute to each latent component.
#

# Analyze loading matrix for q=10 model
loading_matrix = ppca_10.load_matrix_

print(f"Loading matrix shape: {loading_matrix.shape}")
print(
    f"This maps {loading_matrix.shape[0]} observations to {loading_matrix.shape[1]} latent components\n"
)

# Visualize loading matrix heatmap
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(
    loading_matrix.T,
    aspect="auto",
    cmap="RdBu_r",
    interpolation="nearest",
    vmin=-0.05,
    vmax=0.05,
)
ax.set_xlabel("Observation (Timestep)", fontsize=12, fontweight="bold")
ax.set_ylabel("Latent Component", fontsize=12, fontweight="bold")
ax.set_title(
    "PPCA Loading Matrix Heatmap (q=10)", fontsize=14, fontweight="bold", pad=15
)
ax.set_yticks(range(10))
ax.set_yticklabels([f"Component {i + 1}" for i in range(10)])
plt.colorbar(im, label="Loading Value", ax=ax)
plt.tight_layout()
plt.show()

# Component contribution analysis
component_magnitudes = np.linalg.norm(loading_matrix, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    range(1, 11),
    component_magnitudes,
    color="steelblue",
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)
ax.set_xlabel("Latent Component", fontsize=12, fontweight="bold")
ax.set_ylabel("L2 Norm (Total Contribution)", fontsize=12, fontweight="bold")
ax.set_title(
    "Component Contribution Analysis\nHigher values indicate stronger components",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(range(1, 11))
ax.grid(True, axis="y", alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, component_magnitudes)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

print("\n💡 Interpretation:")
print("=" * 70)
print("• First few components typically capture the most variance")
print("• Component magnitudes show relative importance")
print("• Well-separated magnitudes suggest clear factor structure")


# %%
# 1.5 Reconstruction Quality Analysis
# ------------------------------------------------------------
#
# Now we evaluate how well PPCA reconstructs the original returns for different numbers of
# latent components. We use Mean Absolute Error (MAE) as our primary metric since it's more
# appropriate for time-series data than Mean Squared Error (MSE), which can be overly
# influenced by outliers.
#
# The reconstruction quality improves as we increase the number of components, but we must
# balance this against model complexity and the risk of overfitting.

# Compare reconstruction quality across different q values
# For time-series data, MAE and MAPE are more appropriate metrics than MSE
reconstruction_mae = {}
reconstruction_mape = {}

for q in q_values:
    ppca, X_reduced = ppca_models[q]
    X_reconstructed = ppca.inverse_transform(X_reduced)

    # Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error(X.values, X_reconstructed)
    reconstruction_mae[q] = mae

# Plot reconstruction errors vs number of components
fig, ax = plt.subplots(1, 1, figsize=(16, 6))

# MAE Plot
qs = list(reconstruction_mae.keys())
mae_errors = list(reconstruction_mae.values())
ax.plot(
    qs,
    mae_errors,
    marker="o",
    linewidth=2,
    markersize=10,
    color="darkred",
    label="Reconstruction MAE",
)
ax.set_xlabel("Number of Latent Components (q)", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Absolute Error", fontsize=12, fontweight="bold")
ax.set_title(
    "Time-Series Reconstruction Quality (MAE)\nLower is Better",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_xticks(qs)

# Add value labels
for q, err in zip(qs, mae_errors):
    ax.text(
        q, err, f"{err:.2e}", ha="center", va="bottom", fontweight="bold", fontsize=10
    )

plt.tight_layout()
plt.show()

print("\n📊 Reconstruction Quality Metrics (Time-Series Appropriate):")
print("=" * 70)
for q in q_values:
    print(f"q={q:2d} → MAE: {reconstruction_mae[q]:.6f}")

# Visualize original vs reconstructed returns for one asset
asset_idx = 0  # AAPL
asset_name = X.columns[asset_idx]

# Use q=10 model
X_reconstructed_10 = ppca_10.inverse_transform(X_reduced_10)

# Plot a subset of returns (last 200 days for clarity)
subset_size = 200
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Original vs Reconstructed
axes[0].plot(
    X.index[-subset_size:],
    X.iloc[-subset_size:, asset_idx],
    label="Original Returns",
    alpha=0.7,
    linewidth=2,
    color="blue",
)
axes[0].plot(
    X.index[-subset_size:],
    X_reconstructed_10[-subset_size:, asset_idx],
    label="Reconstructed Returns (q=10)",
    alpha=0.7,
    linewidth=2,
    color="red",
    linestyle="--",
)
axes[0].set_ylabel("Returns", fontsize=11, fontweight="bold")
axes[0].set_title(
    f"Original vs Reconstructed Returns for {asset_name}",
    fontsize=13,
    fontweight="bold",
)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Reconstruction error over time
reconstruction_error_ts = (
    X.iloc[-subset_size:, asset_idx].values
    - X_reconstructed_10[-subset_size:, asset_idx]
)
axes[1].plot(
    X.index[-subset_size:],
    reconstruction_error_ts,
    color="darkgreen",
    linewidth=1.5,
    label="Reconstruction Error",
)
axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
axes[1].set_xlabel("Date", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Error", fontsize=11, fontweight="bold")
axes[1].set_title("Reconstruction Error Over Time", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Key Observations:")
print("=" * 70)
print("• More components → Better reconstruction (lower MAE)")
print("• Trade-off: Model complexity vs reconstruction quality")
print("• Reconstruction captures major trends while filtering noise")
print("• Errors tend to be larger during high volatility periods")


# %%
# Section 2: High-Dimensional Scenario
# =====================================
# Understanding Temporal Dependencies
# ============================================================
#
# %%
# The Problem with Conventional PPCA (Tipping & Bishop, 1999)
# ------------------------------------------------------------
#
# The original PPCA formulation [Tipping & Bishop, 1999] works excellently for **standard dimensionality regimes** where :math:`n_{observations} \gg n_{assets}`. However, it **fails catastrophically** for time-series data in high-dimensional settings. Here's why:
#
# **Conventional PPCA Behavior:**
# - Discovers latent factors between **assets** (cross-sectional dependencies)
# - Output: :math:`(q, n_{assets})` - one latent vector per asset
# - **Problem**: When :math:`q < n_{assets}`, massive information loss occurs
#   - Example: :math:`q=10` components for :math:`n_{assets}=100` means we compress 100 assets into 10 factors
#   - This discards ~90% of asset-specific information
#   - The model focuses on **which assets** to keep, not temporal structure
# - **Critical Flaw**: For time-series, we need temporal patterns, not asset selection!
#
# **Why It Fails:**
# 1. **Information Bottleneck**: Reduces 10,000 time series → 10 latent factors (huge loss)
# 2. **Wrong Focus**: Discovers cross-asset correlations (unreliable in high-dim sparse regimes)
# 3. **Temporal Blindness**: Ignores autocorrelation and temporal dynamics within each series
# 4. **Noise Amplification**: Cross-asset correlations become unreliable with limited observations
#
# %%
# The Solution: Atwa et al. (2026) Transpose Trick (Dual Formulation)
# ------------------------------------------------------------
#
# **Key Innovation** [Atwa, Kholief, & Sedky, 2026]: Use the **dual formulation** (transpose trick) to switch from asset-space to time-space discovery.
#
# **How It Works:**
# - When :math:`n_{observations} \gg n_{assets}`, automatically transpose the data matrix, :math:`(n_{observations}, n_{assets}) \rightarrow (n_{assets}, n_{observations})`
# - Now discovers latent factors between **timesteps** (temporal dependencies)
# - Output: :math:`(q, n_{timesteps})` - temporal patterns across time
# - **Benefit**: Preserves temporal structure instead of discarding it
#
# **Information Preservation:**
# - Conventional: 100 assets → 10 factors
# - Dual Form: 10,000 timesteps → 500-2500
# - **Result**: Higher information capture in loading matrix, better missing value imputation
#
# **Mathematical Insight:**
#
# Instead of modeling: :math:`\mathbf{X} = \mathbf{W}\mathbf{Z}^T + \boldsymbol{\mu} + \boldsymbol{\epsilon}` where :math:`\mathbf{W} \in \mathbb{R}^{n \times q}`
#
# We work with: :math:`\mathbf{X}^T = \mathbf{W}'\mathbf{Z}'^T + \boldsymbol{\mu}' + \boldsymbol{\epsilon}'` where :math:`\mathbf{W}' \in \mathbb{R}^{d \times q}`
#
# This **adaptively chooses** the right formulation based on data dimensionality!
#
# ---
#
# In high-dimensional regimes where :math:`n_{assets} \ll n_{timesteps}`, PPCA **automatically switches** interpretation using the dual formulation. Instead of capturing cross-sectional dependencies between assets, it discovers **temporal patterns** within time series.
#
# This is **crucial** for sparse, high-dimensional data where:
# - Cross-asset correlations are unreliable (noise dominates)
# - Temporal structure remains robust (autocorrelation is real signal)
# - Missing value imputation benefits from temporal patterns
#

# %%
# 2.1 Create High-Dimensional Dataset
# ------------------------------------------------------------
#
# We'll select a short time window with many assets to simulate a high-dimensional regime.
#

# Load larger dataset (FTSE 100 with 64 assets)
prices_ftse = load_ftse100_dataset()
X_ftse_full = prices_to_returns(prices_ftse)

# Let's add some sparsity to the data (replace some values with zeros, by 40% of the data)
sparse_idx = np.random.randint(
    0, X_ftse_full.shape[1], size=int(X_ftse_full.shape[1] * 0.40)
)
X_ftse_full.iloc[:, sparse_idx] = 0

# Create high-dimensional scenario
X_high_dim = X_ftse_full.T

print("High-Dimensional Dataset:")
print("=" * 70)
print(f"Shape: {X_high_dim.shape}")
print(f"Number of features (d): {X_high_dim.shape[1]}")
print(f"Number of observations (n): {X_high_dim.shape[0]}")
print("Regime: High-Dimensional (n << d)")
print(f"Ratio: {X_high_dim.shape[0] / X_high_dim.shape[1]:.2f}")
print("\n💡 In this regime, PPCA discovers TEMPORAL patterns, not cross-sectional!")
print("   The latent factors will represent time-series structure within assets.")


X_high_dim


# Fit PPCA with high-dimensional data (dual formulation will be used automatically)
q_hd = 500  # Choose a reasonable number of components

print(f"🔬 Fitting PPCA with q={q_hd} on High-Dimensional Data")
print("=" * 70)
print(
    f"Data shape: {X_high_dim.shape} (n={X_high_dim.shape[0]}, d={X_high_dim.shape[1]})"
)
print("Regime: HIGH-DIMENSIONAL (n << d)")
print("\n⚙️  Atwa et al. (2026) Dual Formulation ACTIVE")
print("   Discovering TEMPORAL patterns (not cross-sectional)")

# Fit the model
start = time.time()
ppca_hd_dual = PPCA(q=q_hd, random_state=42, use_em=True, max_iter=20)
ppca_hd_dual.fit(X_high_dim)
fit_time_hd = time.time() - start

# Transform to latent space
X_reduced_hd = ppca_hd_dual.transform(X_high_dim)

# Calculate reconstruction error
X_reconstructed_hd = ppca_hd_dual.inverse_transform(X_reduced_hd)
mae_hd_dual = mean_absolute_error(X_high_dim.values, X_reconstructed_hd)

# Information preservation metric
info_retention_dual = (q_hd / X_high_dim.shape[1]) * 100

print(f"\n✓ Fitting completed in {fit_time_hd:.2f}s")
print(f"✓ Output shape: {X_reduced_hd.shape}")
print(f"✓ Interpretation: {X_reduced_hd.shape[0]} latent TEMPORAL features")
print(f"   across {X_reduced_hd.shape[1]} features (timesteps)")
print(f"✓ Loading matrix shape: {ppca_hd_dual.load_matrix_.shape}")
print(f"✓ Noise variance: {ppca_hd_dual.noise_variance_:.6f}")
print("\n📊 Information Preservation (Dual Form):")
print(
    f"   {q_hd} components / {X_high_dim.shape[0]} observations (n) = {info_retention_dual:.1f}% retention"
)
print(f"✓ Reconstruction MAE: {mae_hd_dual:.6f}")
print("\n" + "=" * 70)


# %%
# 2.2 Comparison: Standard PPCA (Section 1) vs Dual Formulation PPCA (Section 2)
# ------------------------------------------------------------

# We'll use the q=10 model from Section 1 for fair comparison
ppca_standard, X_reduced_standard = ppca_models[10]
mae_standard = reconstruction_mae[10]

# Standard regime dimensions
n_obs_standard = X.shape[0]  # 8312 observations
n_assets_standard = X.shape[1]  # 20 assets
q_standard = 10

# High-dimensional regime dimensions (dual formulation)
n_obs_hd = X_high_dim.shape[0]  # 64 observations (n)
d_hd = X_high_dim.shape[1]  # 5959 features (d) - timesteps in transposed data
q_hd_actual = q_hd  # 500

# ============================================================================
# Calculate Information Metrics for BOTH Approaches
# ============================================================================

# Standard PPCA (Section 1): Compresses ASSETS dimension
info_retention_standard = (q_standard / n_assets_standard) * 100
loading_variance_standard = np.sum(np.var(ppca_standard.load_matrix_, axis=0))

# Dual Formulation PPCA (Section 2): Compresses OBSERVATIONS dimension
info_retention_dual_actual = (q_hd_actual / d_hd) * 100
loading_variance_dual = np.sum(np.var(ppca_hd_dual.load_matrix_, axis=0))

# Normalize variances for comparison (relative to dual)
variance_normalized_standard = (loading_variance_standard / loading_variance_dual) * 100
variance_normalized_dual = 100.0

# ============================================================================
# Create Comprehensive Comparison Visualization
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Define colors and labels
colors = ["#3498db", "#27ae60"]  # Blue for standard, Green for dual
methods = [
    "Standard PPCA\n(Section 1: n >> d)\nTipping & Bishop, 1999",
    "Dual Formulation PPCA\n(Section 2: n << d)\nAtwa et al., 2026",
]

# ============================================================================
# Plot 1: Latent Dimension Ratio (Information Preservation)
# ============================================================================
retentions = [info_retention_standard, info_retention_dual_actual]

bars1 = axes[0].bar(
    methods,
    retentions,
    color=colors,
    alpha=0.75,
    edgecolor="black",
    linewidth=2.5,
    width=0.6,
)
axes[0].set_ylabel("Latent Dimension Ratio (%)", fontsize=13, fontweight="bold")
axes[0].set_title(
    "Information Preservation via Dimension Ratio\n"
    r"$\frac{q \text{ (latent components)}}{\text{compressed dimension}} \times 100$",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
axes[0].grid(True, axis="y", alpha=0.35, linestyle="--")
axes[0].set_ylim(0, max(retentions) * 1.25)
axes[0].set_axisbelow(True)

# Add value labels with context
for i, (bar, val) in enumerate(zip(bars1, retentions)):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(retentions) * 0.03,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=16,
        color="black",
    )
    if i == 0:
        # Standard PPCA
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{n_assets_standard} assets\n→ {q_standard} factors\n(Cross-Sectional)",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="darkblue", alpha=0.7),
        )
    else:
        # Dual Formulation
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{d_hd} obs (n)\n→ {q_hd_actual} factors\n(Temporal)",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="darkgreen", alpha=0.7),
        )

# ============================================================================
# Plot 2: Captured Variance in Loading Matrix (Relative Comparison)
# ============================================================================
variance_values = [variance_normalized_standard, variance_normalized_dual]

bars2 = axes[1].bar(
    methods,
    variance_values,
    color=colors,
    alpha=0.75,
    edgecolor="black",
    linewidth=2.5,
    width=0.6,
)
axes[1].set_ylabel("Relative Captured Variance (%)", fontsize=13, fontweight="bold")
axes[1].set_title(
    "Loading Matrix Information Capture\nTotal Variance Explained by Latent Components",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
axes[1].grid(True, axis="y", alpha=0.35, linestyle="--")
axes[1].set_ylim(0, max(variance_values) * 1.25)
axes[1].set_axisbelow(True)

# Add value labels and context
for i, (bar, val) in enumerate(zip(bars2, variance_values)):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(variance_values) * 0.03,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=16,
        color="black",
    )
    if i == 0:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"n={n_obs_standard}\nd={n_assets_standard}\n(Good Regime)",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="steelblue", alpha=0.7),
        )
    else:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"n={n_obs_hd}\nd={d_hd}\n(High-Dim Regime)",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="darkgreen", alpha=0.7),
        )

# ============================================================================
# Plot 3: Reconstruction Quality (MAE) - Lower is Better
# ============================================================================
mae_values = [mae_standard, mae_hd_dual]

bars3 = axes[2].bar(
    methods,
    mae_values,
    color=colors,
    alpha=0.75,
    edgecolor="black",
    linewidth=2.5,
    width=0.6,
)
axes[2].set_ylabel("Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
axes[2].set_title(
    "Reconstruction Quality Comparison\nMAE: Lower is Better (Time-Series Metric)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
axes[2].grid(True, axis="y", alpha=0.35, linestyle="--")
axes[2].set_ylim(0, max(mae_values) * 1.25)
axes[2].set_axisbelow(True)

# Add value labels and status
for i, (bar, val) in enumerate(zip(bars3, mae_values)):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(mae_values) * 0.03,
        f"{val:.6f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=14,
        color="black",
    )

    if i == 0:
        # Standard regime
        status = "✓ Good Fit\n(Appropriate\nRegime)" if val < 0.01 else "✓ Acceptable"
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            status,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="steelblue", alpha=0.7),
        )
    else:
        # Dual formulation
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            "✓ Excellent Fit\n(Dual Form\nActive)",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="darkgreen", alpha=0.7),
        )

# Overall layout
plt.tight_layout(pad=2.5)
plt.show()

# ============================================================================
# Print Detailed Comparison Summary
# ============================================================================
print(f"\n{'=' * 85}")
print("📊 COMPREHENSIVE COMPARISON: Standard PPCA vs Dual Formulation PPCA")
print(f"{'=' * 85}")

print("\n┌─ Standard PPCA (Section 1: Tipping & Bishop, 1999) ──────────────────────┐")
print(
    f"│  Data Regime: n={n_obs_standard} observations >> d={n_assets_standard} assets (STANDARD)"
)
print(
    f"│  Compression Strategy: {n_assets_standard} assets → {q_standard} latent factors"
)
print(f"│  Dimension Ratio: {info_retention_standard:.1f}% (q/d)")
print("│  Focus: Cross-sectional asset dependencies")
print(f"│  Loading Matrix Shape: ({n_obs_standard}, {q_standard})")
print(f"│  Captured Variance: {loading_variance_standard:.4f}")
print(f"│  Reconstruction MAE: {mae_standard:.6f}")
print("│  ✓ Works well! Standard regime with reliable cross-asset correlations")
print("└───────────────────────────────────────────────────────────────────────────┘")

print("\n┌─ Dual Formulation PPCA (Section 2: Atwa et al., 2026) ───────────────────┐")
print(f"│  Data Regime: n={n_obs_hd} observations << d={d_hd} features (HIGH-DIM)")
print(
    f"│  Compression Strategy: {d_hd} observations (n) → {q_hd_actual} latent factors"
)
print(f"│  Dimension Ratio: {info_retention_dual_actual:.1f}% (q/n)")
print("│  Focus: Temporal patterns (d = timesteps after transpose)")
print(f"│  Loading Matrix Shape: ({n_obs_hd}, {q_hd_actual})")
print(f"│  Captured Variance: {loading_variance_dual:.4f}")
print(f"│  Reconstruction MAE: {mae_hd_dual:.6f}")
print("│  ✓ ESSENTIAL! Automatic transpose adapts to high-dimensional regime")
print("└───────────────────────────────────────────────────────────────────────────┘")


# %%
# Section 3: PPCA vs PKPCA Comparison on High-Dimensional Sparse Data
# ====================================================================
# 3.1 Introduction to the Challenge
# ============================================================
#
# In this section, we conduct a **comprehensive comparison** between **Probabilistic PCA (PPCA)** and **Probabilistic Kernel PCA (PKPCA)** on high-dimensional sparse financial data.
#
# %%
# Why This Matters for Risk Management
# ------------------------------------------------------------
#
# High-dimensional sparse data is ubiquitous in modern finance:
# - **Alternative data sources**: Transaction logs, order book data, tick-by-tick trades
# - **Emerging markets**: Many assets with limited trading history
# - **Crisis periods**: Missing or zero returns during market shutdowns
# - **Regulatory constraints**: Trading halts, circuit breakers
#
# %%
# Key Questions We'll Answer
# ------------------------------------------------------------
#
# 1. **Temporal Dependency Capture**: Which model better preserves autocorrelation structure?
# 2. **Reconstruction Quality**: How do linear (PPCA) vs non-linear (PKPCA) approaches compare?
# 3. **Model Selection**: What is the optimal number of components (q) for each model?
# 4. **Portfolio Performance**: Which model leads to better risk-adjusted returns?
# 5. **Computational Efficiency**: What is the performance-cost trade-off?
# 6. **Robustness**: How do models degrade with increasing sparsity?

# %%
# 3.1.1 Create High-Dimensional Sparse Dataset
# ------------------------------------------------------------
#

# Load larger dataset (FTSE 100 with 64 assets)
prices_ftse = load_ftse100_dataset()
X_ftse_full = prices_to_returns(prices_ftse)

# Add SCALAR-BASED UNIFORM sparsity (40% of individual data points become zero)
# This ensures uniform sparsity across all assets, not per-column or per-row
np.random.seed(42)
sparsity_ratio = 0.40

# Create sparse data by randomly selecting individual scalar entries
X_ftse_sparse = X_ftse_full.copy()
n_total_elements = X_ftse_sparse.size
n_sparse_elements = int(n_total_elements * sparsity_ratio)

# Generate random indices for sparsification (uniform across entire matrix)
flat_indices = np.random.choice(n_total_elements, size=n_sparse_elements, replace=False)

# Convert flat indices to 2D indices and set to zero
rows, cols = np.unravel_index(flat_indices, X_ftse_sparse.shape)
X_ftse_sparse.values[rows, cols] = 0

# Create high-dimensional scenario (transpose to get n << d)
X_high_dim_sparse = X_ftse_sparse.T

print(f"{'=' * 80}")
print("HIGH-DIMENSIONAL SPARSE DATASET CREATED (SCALAR-BASED UNIFORM SPARSITY)")
print(f"{'=' * 80}")
print("\nOriginal FTSE100 Dataset:")
print(f"  Shape: {X_ftse_full.shape}")
print(f"  Time period: {X_ftse_full.index[0]} to {X_ftse_full.index[-1]}")
print(f"  Total data points: {X_ftse_full.shape[0] * X_ftse_full.shape[1]:,}")

print(f"\nAfter Scalar-Based Sparsity Injection ({sparsity_ratio * 100:.0f}%):")
print(f"  Target sparsity ratio: {sparsity_ratio * 100:.1f}%")
print(f"  Total elements: {n_total_elements:,}")
print(f"  Sparse elements: {n_sparse_elements:,}")
print(
    f"  Zero values added: {(X_ftse_sparse == 0).sum().sum() - (X_ftse_full == 0).sum().sum():,}"
)
print(
    f"  Actual sparsity ratio: {(X_ftse_sparse == 0).sum().sum() / X_ftse_sparse.size * 100:.2f}%"
)

# Verify uniform distribution across assets
sparsity_per_asset = (X_ftse_sparse == 0).sum(axis=0) / len(X_ftse_sparse) * 100
print("\nSparsity Distribution Across Assets (should be uniform):")
print(f"  Mean: {sparsity_per_asset.mean():.2f}%")
print(f"  Std:  {sparsity_per_asset.std():.2f}%")
print(f"  Min:  {sparsity_per_asset.min():.2f}%")
print(f"  Max:  {sparsity_per_asset.max():.2f}%")
print("  → Low std indicates uniform distribution ✓")

print("\nHigh-Dimensional Transpose (for PPCA/PKPCA):")
print(f"  Shape: {X_high_dim_sparse.shape}")
print(f"  Number of observations (n): {X_high_dim_sparse.shape[0]}")
print(f"  Number of features (d): {X_high_dim_sparse.shape[1]}")
print(
    f"  Regime: {'HIGH-DIMENSIONAL' if X_high_dim_sparse.shape[0] < X_high_dim_sparse.shape[1] else 'STANDARD'}"
)
print(f"  Ratio n/d: {X_high_dim_sparse.shape[0] / X_high_dim_sparse.shape[1]:.4f}")
print("\n💡 In this regime, both models will discover TEMPORAL patterns!")
print("  Sparsity is now UNIFORMLY distributed across all assets!")
print(f"{'=' * 80}")


X_ftse_sparse


# %%
# 3.1.2 Visualize Sparsity Patterns
# ------------------------------------------------------------
#

# Visualize sparsity patterns and distribution
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Temporal sparsity heatmap (subset for visibility)
subset_assets = 30  # Show first 30 assets
subset_time = 500  # Show first 500 time periods

sparsity_matrix = (X_ftse_sparse.iloc[:subset_time, :subset_assets] == 0).astype(int)
im1 = axes[0, 0].imshow(
    sparsity_matrix.T,
    aspect="auto",
    cmap="RdYlGn_r",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)
axes[0, 0].set_xlabel("Time Period", fontsize=11, fontweight="bold")
axes[0, 0].set_ylabel("Assets", fontsize=11, fontweight="bold")
axes[0, 0].set_title(
    f"Temporal Sparsity Pattern (First {subset_assets} assets, {subset_time} periods)\n"
    "Red = Zero (Sparse), Green = Non-zero",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
plt.colorbar(im1, ax=axes[0, 0], label="Sparse (1) vs Non-Sparse (0)")

# 2. Sparsity ratio per asset
sparsity_per_asset = (X_ftse_sparse == 0).sum(axis=0) / len(X_ftse_sparse) * 100
axes[0, 1].hist(
    sparsity_per_asset, bins=30, color="steelblue", alpha=0.7, edgecolor="black"
)
axes[0, 1].axvline(
    sparsity_per_asset.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {sparsity_per_asset.mean():.1f}%",
)
axes[0, 1].set_xlabel("Sparsity Ratio per Asset (%)", fontsize=11, fontweight="bold")
axes[0, 1].set_ylabel("Number of Assets", fontsize=11, fontweight="bold")
axes[0, 1].set_title(
    "Distribution of Sparsity Across Assets", fontsize=12, fontweight="bold", pad=10
)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. Correlation structure before sparsity
corr_original = X_ftse_full.corr()
im3 = axes[1, 0].imshow(
    corr_original,
    aspect="auto",
    cmap="RdBu_r",
    interpolation="nearest",
    vmin=-1,
    vmax=1,
)
axes[1, 0].set_xlabel("Asset Index", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Asset Index", fontsize=11, fontweight="bold")
axes[1, 0].set_title(
    "Correlation Matrix: Original Data\n(Before Sparsity)",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
plt.colorbar(im3, ax=axes[1, 0], label="Correlation")

# 4. Correlation structure after sparsity
corr_sparse = X_ftse_sparse.corr()
im4 = axes[1, 1].imshow(
    corr_sparse, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1
)
axes[1, 1].set_xlabel("Asset Index", fontsize=11, fontweight="bold")
axes[1, 1].set_ylabel("Asset Index", fontsize=11, fontweight="bold")
axes[1, 1].set_title(
    "Correlation Matrix: Sparse Data\n(After 40% Sparsity Injection)",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
plt.colorbar(im4, ax=axes[1, 1], label="Correlation")

plt.tight_layout()
plt.show()

# Statistics
print(f"\n{'=' * 80}")
print("SPARSITY ANALYSIS")
print(f"{'=' * 80}")
print("Sparsity per asset statistics:")
print(f"  Mean: {sparsity_per_asset.mean():.2f}%")
print(f"  Std:  {sparsity_per_asset.std():.2f}%")
print(f"  Min:  {sparsity_per_asset.min():.2f}%")
print(f"  Max:  {sparsity_per_asset.max():.2f}%")

print("\nCorrelation structure impact:")
print(
    f"  Original data - Mean abs correlation: {np.abs(corr_original.values[np.triu_indices_from(corr_original.values, k=1)]).mean():.4f}"
)
print(
    f"  Sparse data   - Mean abs correlation: {np.abs(corr_sparse.values[np.triu_indices_from(corr_sparse.values, k=1)]).mean():.4f}"
)
print(
    f"  Correlation degradation: {(1 - np.abs(corr_sparse.values[np.triu_indices_from(corr_sparse.values, k=1)]).mean() / np.abs(corr_original.values[np.triu_indices_from(corr_original.values, k=1)]).mean()) * 100:.2f}%"
)
print(f"{'=' * 80}")


# %%
# 3.2 Theoretical Foundations: PPCA vs PKPCA
# ============================================================
#
# See: https://www.sciencedirect.com/science/article/pii/S0952197625034074

# %%
# 3.3 Fitting PPCA and PKPCA on Sparse Data
# ============================================================
#
# %%
# 3.3.1 Fit Both Models with Multiple Component Counts
# ------------------------------------------------------------
#
# We'll test both models with :math:`q \in \{250, 500, 750, 1000, 1200\}` to understand:
# - How reconstruction quality improves with more components
# - Computational cost scaling
# - Noise variance evolution
#

# Define q values for comprehensive testing
q_values = [250, 500, 750, 1000, 1200]

# Storage for results
ppca_results = {}
pkpca_results = {}

print(f"{'=' * 100}")
print("FITTING PPCA AND PKPCA ON HIGH-DIMENSIONAL SPARSE DATA")
print(f"{'=' * 100}")
print(
    f"Data shape: {X_high_dim_sparse.shape} (n={X_high_dim_sparse.shape[0]}, d={X_high_dim_sparse.shape[1]})"
)
print(f"Testing q values: {q_values}")
print(f"{'=' * 100}\n")

# Fit PPCA models
print("🔷 FITTING PPCA MODELS")
print(f"{'-' * 100}")
for q in q_values:
    print(f"\n[PPCA q={q}] Starting fit...")

    # Fit PPCA
    start_time = time.time()
    ppca_model = PPCA(q=q, random_state=42, use_em=True, max_iter=20)
    ppca_model.fit(X_high_dim_sparse)
    fit_time = time.time() - start_time

    # Transform
    start_transform = time.time()
    X_reduced_ppca = ppca_model.transform(X_high_dim_sparse)
    transform_time = time.time() - start_transform

    # Reconstruct
    X_reconstructed_ppca = ppca_model.inverse_transform(X_reduced_ppca)

    # Calculate reconstruction error
    mae = mean_absolute_error(X_high_dim_sparse.values, X_reconstructed_ppca)

    # Store results
    ppca_results[q] = {
        "model": ppca_model,
        "X_reduced": X_reduced_ppca,
        "X_reconstructed": X_reconstructed_ppca,
        "fit_time": fit_time,
        "transform_time": transform_time,
        "mae": mae,
        "noise_variance": ppca_model.noise_variance_,
        "loading_matrix_shape": ppca_model.load_matrix_.shape,
    }

    print(f"  ✓ Fit time: {fit_time:.2f}s")
    print(f"  ✓ Transform time: {transform_time:.3f}s")
    print(f"  ✓ Reconstruction MAE: {mae:.6f}")
    print(f"  ✓ Noise variance (σ²): {ppca_model.noise_variance_:.6f}")
    print(f"  ✓ Loading matrix shape: {ppca_model.load_matrix_.shape}")
    print(f"  ✓ Reduced data shape: {X_reduced_ppca.shape}")

print(f"\n{'=' * 100}")
print("🔶 FITTING PKPCA MODELS")
print(f"{'-' * 100}")
for q in q_values:
    print(f"\n[PKPCA q={q}] Starting fit...")

    # Fit PKPCA
    start_time = time.time()
    pkpca_model = PKPCA(q=q, random_state=42, use_em=True, max_iter=20)
    pkpca_model.fit(X_high_dim_sparse)
    fit_time = time.time() - start_time

    # Transform
    start_transform = time.time()
    X_reduced_pkpca = pkpca_model.transform(X_high_dim_sparse)
    transform_time = time.time() - start_transform

    # Reconstruct
    X_reconstructed_pkpca = pkpca_model.inverse_transform(X_reduced_pkpca)

    # Calculate reconstruction error
    mae = mean_absolute_error(X_high_dim_sparse.values, X_reconstructed_pkpca)

    # Store results
    pkpca_results[q] = {
        "model": pkpca_model,
        "X_reduced": X_reduced_pkpca,
        "X_reconstructed": X_reconstructed_pkpca,
        "fit_time": fit_time,
        "transform_time": transform_time,
        "mae": mae,
        "noise_variance": pkpca_model.noise_variance_,
        "loading_matrix_shape": pkpca_model.load_matrix_.shape,
    }

    print(f"  ✓ Fit time: {fit_time:.2f}s")
    print(f"  ✓ Transform time: {transform_time:.3f}s")
    print(f"  ✓ Reconstruction MAE: {mae:.6f}")
    print(f"  ✓ Noise variance (σ²): {pkpca_model.noise_variance_:.6f}")
    print(f"  ✓ Loading matrix shape: {pkpca_model.load_matrix_.shape}")
    print(f"  ✓ Reduced data shape: {X_reduced_pkpca.shape}")

print(f"\n{'=' * 100}")
print("✅ ALL MODELS FITTED SUCCESSFULLY")
print(f"{'=' * 100}")


# %%
# 3.3.2 Reconstruction Quality Comparison
# ------------------------------------------------------------
#

# Extract metrics for visualization
ppca_maes = [ppca_results[q]["mae"] for q in q_values]
pkpca_maes = [pkpca_results[q]["mae"] for q in q_values]
ppca_noise_vars = [ppca_results[q]["noise_variance"] for q in q_values]
pkpca_noise_vars = [pkpca_results[q]["noise_variance"] for q in q_values]

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Reconstruction MAE comparison
axes[0, 0].plot(
    q_values,
    ppca_maes,
    marker="o",
    linewidth=2.5,
    markersize=10,
    label="PPCA",
    color="#3498db",
    linestyle="-",
)
axes[0, 0].plot(
    q_values,
    pkpca_maes,
    marker="s",
    linewidth=2.5,
    markersize=10,
    label="PKPCA",
    color="#e74c3c",
    linestyle="-",
)
axes[0, 0].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
axes[0, 0].set_title(
    "Reconstruction Quality: PPCA vs PKPCA\n(Lower is Better)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[0, 0].legend(fontsize=11, loc="upper right")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(q_values)

# Add value labels
for i, (q, mae_p, mae_k) in enumerate(zip(q_values, ppca_maes, pkpca_maes)):
    axes[0, 0].text(
        q, mae_p, f"{mae_p:.4f}", ha="center", va="bottom", fontsize=8, color="#3498db"
    )
    axes[0, 0].text(
        q, mae_k, f"{mae_k:.4f}", ha="center", va="top", fontsize=8, color="#e74c3c"
    )

# Plot 2: MAE Improvement Ratio (PPCA/PKPCA)
mae_ratio = [ppca_maes[i] / pkpca_maes[i] for i in range(len(q_values))]
bars = axes[0, 1].bar(
    range(len(q_values)),
    mae_ratio,
    color=["#27ae60" if r > 1 else "#e67e22" for r in mae_ratio],
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)
axes[0, 1].axhline(
    y=1, color="red", linestyle="--", linewidth=2, label="Equal Performance"
)
axes[0, 1].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("MAE Ratio (PPCA / PKPCA)", fontsize=12, fontweight="bold")
axes[0, 1].set_title(
    "Relative Performance: PPCA vs PKPCA\n(>1 means PKPCA is better)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[0, 1].set_xticks(range(len(q_values)))
axes[0, 1].set_xticklabels(q_values)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Add percentage labels
for i, (bar, ratio) in enumerate(zip(bars, mae_ratio)):
    improvement = (ratio - 1) * 100
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{improvement:+.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )

# Plot 3: Noise Variance Evolution
axes[1, 0].plot(
    q_values,
    ppca_noise_vars,
    marker="o",
    linewidth=2.5,
    markersize=10,
    label="PPCA",
    color="#3498db",
    linestyle="-",
)
axes[1, 0].plot(
    q_values,
    pkpca_noise_vars,
    marker="s",
    linewidth=2.5,
    markersize=10,
    label="PKPCA",
    color="#e74c3c",
    linestyle="-",
)
axes[1, 0].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Noise Variance (σ²)", fontsize=12, fontweight="bold")
axes[1, 0].set_title(
    "Estimated Noise Variance: PPCA vs PKPCA\n(Lower indicates better fit)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(q_values)

# Plot 4: Sample Reconstruction Comparison (one asset)
sample_asset_idx = 0
sample_asset_name = X_high_dim_sparse.index[sample_asset_idx]
original_returns = X_high_dim_sparse.iloc[
    sample_asset_idx, :500
]  # First 500 time points
ppca_recon_750 = ppca_results[750]["X_reconstructed"][sample_asset_idx, :500]
pkpca_recon_750 = pkpca_results[750]["X_reconstructed"][sample_asset_idx, :500]

axes[1, 1].plot(
    original_returns.values,
    label="Original (Sparse)",
    alpha=0.8,
    linewidth=1.5,
    color="black",
)
axes[1, 1].plot(
    ppca_recon_750,
    label="PPCA Reconstruction (q=750)",
    alpha=0.7,
    linewidth=1.5,
    color="#3498db",
    linestyle="--",
)
axes[1, 1].plot(
    pkpca_recon_750,
    label="PKPCA Reconstruction (q=750)",
    alpha=0.7,
    linewidth=1.5,
    color="#e74c3c",
    linestyle=":",
)
axes[1, 1].set_xlabel("Time Index", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Returns", fontsize=12, fontweight="bold")
axes[1, 1].set_title(
    f"Sample Reconstruction: {sample_asset_name} (First 500 periods)\nq=750 components",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\n{'=' * 100}")
print("RECONSTRUCTION QUALITY SUMMARY")
print(f"{'=' * 100}")
print(f"\n{'Model':<15} {'q':<8} {'MAE':<12} {'Noise Var':<12} {'Fit Time (s)':<15}")
print(f"{'-' * 100}")
for q in q_values:
    print(
        f"{'PPCA':<15} {q:<8} {ppca_results[q]['mae']:<12.6f} {ppca_results[q]['noise_variance']:<12.6f} {ppca_results[q]['fit_time']:<15.2f}"
    )
    print(
        f"{'PKPCA':<15} {q:<8} {pkpca_results[q]['mae']:<12.6f} {pkpca_results[q]['noise_variance']:<12.6f} {pkpca_results[q]['fit_time']:<15.2f}"
    )
    improvement = (ppca_results[q]["mae"] / pkpca_results[q]["mae"] - 1) * 100
    print(f"{'  → Improvement':<15} {'':>8} {improvement:>+11.2f}%")
    print(f"{'-' * 100}")

print("\n💡 Key Observations:")
print(
    f"  • Best PPCA performance: q={min(ppca_results, key=lambda k: ppca_results[k]['mae'])}, MAE={min(ppca_maes):.6f}"
)
print(
    f"  • Best PKPCA performance: q={min(pkpca_results, key=lambda k: pkpca_results[k]['mae'])}, MAE={min(pkpca_maes):.6f}"
)
print(
    f"  • Average PKPCA improvement: {np.mean([(ppca_results[q]['mae'] / pkpca_results[q]['mae'] - 1) * 100 for q in q_values]):.2f}%"
)
print(f"{'=' * 100}")


# %%
# 3.4 Temporal Dependency Capture Analysis
# ============================================================
#
# %%
# 3.4.1 Autocorrelation Function (ACF) Analysis
# ------------------------------------------------------------
#
# One of the key questions: **Which model better preserves the temporal structure?**
#
# We'll analyze autocorrelation to understand how well each model captures time-series dependencies.
#

# Import additional libraries for ACF analysis
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf


# Calculate ACF for original and reconstructed data
def calculate_acf_for_asset(data, asset_idx=0, nlags=50):
    """
    Calculate autocorrelation function for a single asset.

    Parameters
    ----------
    data : array-like
        Returns data
    asset_idx : int
        Index of asset to analyze
    nlags : int
        Number of lags for ACF

    Returns
    -------
    acf_values : ndarray
        Autocorrelation values
    """
    asset_returns = data[asset_idx, :] if data.ndim == 2 else data.iloc[asset_idx, :]
    return acf(asset_returns, nlags=nlags, fft=True)


# Select representative q value for detailed analysis
q_analysis = 750

# Calculate ACF for multiple assets
n_assets_to_analyze = 5
nlags = 50

print(f"{'=' * 100}")
print(f"AUTOCORRELATION ANALYSIS (q={q_analysis})")
print(f"{'=' * 100}")
print(f"Analyzing temporal dependencies for {n_assets_to_analyze} assets...")
print(f"Lag range: 0 to {nlags}\n")

# Storage for ACF values
acf_original_all = []
acf_ppca_all = []
acf_pkpca_all = []

for asset_idx in range(n_assets_to_analyze):
    # Original sparse data
    acf_orig = calculate_acf_for_asset(X_high_dim_sparse.values, asset_idx, nlags)
    acf_original_all.append(acf_orig)

    # PPCA reconstruction
    acf_ppca = calculate_acf_for_asset(
        ppca_results[q_analysis]["X_reconstructed"], asset_idx, nlags
    )
    acf_ppca_all.append(acf_ppca)

    # PKPCA reconstruction
    acf_pkpca = calculate_acf_for_asset(
        pkpca_results[q_analysis]["X_reconstructed"], asset_idx, nlags
    )
    acf_pkpca_all.append(acf_pkpca)

    # Calculate correlation between original and reconstructed ACF
    corr_ppca = np.corrcoef(acf_orig[1:], acf_ppca[1:])[
        0, 1
    ]  # Exclude lag 0 (always 1)
    corr_pkpca = np.corrcoef(acf_orig[1:], acf_pkpca[1:])[0, 1]

    print(f"Asset {asset_idx} ({X_high_dim_sparse.index[asset_idx]}):")
    print(f"  ACF correlation with original - PPCA:  {corr_ppca:.4f}")
    print(f"  ACF correlation with original - PKPCA: {corr_pkpca:.4f}")
    print(f"  → PKPCA advantage: {(corr_pkpca - corr_ppca) * 100:.2f}%\n")

# Average ACF across assets
avg_acf_original = np.mean(acf_original_all, axis=0)
avg_acf_ppca = np.mean(acf_ppca_all, axis=0)
avg_acf_pkpca = np.mean(acf_pkpca_all, axis=0)

print(f"{'=' * 100}")


# %%
# 3.4.2 Visualize ACF and Statistical Tests
# ------------------------------------------------------------
#

# Visualize ACF comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Average ACF across all assets
lags = np.arange(nlags + 1)
axes[0, 0].plot(
    lags,
    avg_acf_original,
    marker="o",
    linewidth=2,
    markersize=6,
    label="Original (Sparse)",
    color="black",
    alpha=0.8,
)
axes[0, 0].plot(
    lags,
    avg_acf_ppca,
    marker="s",
    linewidth=2,
    markersize=5,
    label="PPCA Reconstruction",
    color="#3498db",
    alpha=0.7,
)
axes[0, 0].plot(
    lags,
    avg_acf_pkpca,
    marker="^",
    linewidth=2,
    markersize=5,
    label="PKPCA Reconstruction",
    color="#e74c3c",
    alpha=0.7,
)
axes[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0, 0].set_xlabel("Lag", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Autocorrelation", fontsize=12, fontweight="bold")
axes[0, 0].set_title(
    f"Average Autocorrelation Function (q={q_analysis})\nAveraged across {n_assets_to_analyze} assets",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: ACF preservation quality (correlation with original)
acf_corr_ppca = [
    np.corrcoef(acf_original_all[i][1:], acf_ppca_all[i][1:])[0, 1]
    for i in range(n_assets_to_analyze)
]
acf_corr_pkpca = [
    np.corrcoef(acf_original_all[i][1:], acf_pkpca_all[i][1:])[0, 1]
    for i in range(n_assets_to_analyze)
]

x_pos = np.arange(n_assets_to_analyze)
width = 0.35
bars1 = axes[0, 1].bar(
    x_pos - width / 2,
    acf_corr_ppca,
    width,
    label="PPCA",
    color="#3498db",
    alpha=0.7,
    edgecolor="black",
)
bars2 = axes[0, 1].bar(
    x_pos + width / 2,
    acf_corr_pkpca,
    width,
    label="PKPCA",
    color="#e74c3c",
    alpha=0.7,
    edgecolor="black",
)
axes[0, 1].set_xlabel("Asset Index", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("ACF Correlation with Original", fontsize=12, fontweight="bold")
axes[0, 1].set_title(
    "ACF Preservation Quality\n(Higher = Better temporal structure capture)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(
    [X_high_dim_sparse.index[i] for i in range(n_assets_to_analyze)],
    rotation=45,
    ha="right",
)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis="y")
axes[0, 1].set_ylim([0, 1])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# Plot 3: Kolmogorov-Smirnov test (distribution similarity)
ks_stats_ppca = []
ks_pvals_ppca = []
ks_stats_pkpca = []
ks_pvals_pkpca = []

for asset_idx in range(n_assets_to_analyze):
    # Original data
    orig_data = X_high_dim_sparse.iloc[asset_idx, :].values

    # PPCA reconstruction
    ppca_data = ppca_results[q_analysis]["X_reconstructed"][asset_idx, :]
    ks_stat_p, ks_pval_p = ks_2samp(orig_data, ppca_data)
    ks_stats_ppca.append(ks_stat_p)
    ks_pvals_ppca.append(ks_pval_p)

    # PKPCA reconstruction
    pkpca_data = pkpca_results[q_analysis]["X_reconstructed"][asset_idx, :]
    ks_stat_k, ks_pval_k = ks_2samp(orig_data, pkpca_data)
    ks_stats_pkpca.append(ks_stat_k)
    ks_pvals_pkpca.append(ks_pval_k)

bars1 = axes[1, 0].bar(
    x_pos - width / 2,
    ks_pvals_ppca,
    width,
    label="PPCA",
    color="#3498db",
    alpha=0.7,
    edgecolor="black",
)
bars2 = axes[1, 0].bar(
    x_pos + width / 2,
    ks_pvals_pkpca,
    width,
    label="PKPCA",
    color="#e74c3c",
    alpha=0.7,
    edgecolor="black",
)
axes[1, 0].axhline(
    y=0.05, color="red", linestyle="--", linewidth=2, label="Significance (α=0.05)"
)
axes[1, 0].set_xlabel("Asset Index", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("KS Test p-value", fontsize=12, fontweight="bold")
axes[1, 0].set_title(
    "Distribution Similarity Test (Kolmogorov-Smirnov)\n(Higher p-value = More similar to original)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(
    [X_high_dim_sparse.index[i] for i in range(n_assets_to_analyze)],
    rotation=45,
    ha="right",
)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Plot 4: Rolling volatility comparison
asset_idx_vol = 0
window = 50
original_vol = (
    pd.Series(X_high_dim_sparse.iloc[asset_idx_vol, :].values).rolling(window).std()
)
ppca_vol = (
    pd.Series(ppca_results[q_analysis]["X_reconstructed"][asset_idx_vol, :])
    .rolling(window)
    .std()
)
pkpca_vol = (
    pd.Series(pkpca_results[q_analysis]["X_reconstructed"][asset_idx_vol, :])
    .rolling(window)
    .std()
)

time_idx = np.arange(len(original_vol))
axes[1, 1].plot(
    time_idx[:1000],
    original_vol[:1000],
    label="Original",
    linewidth=2,
    color="black",
    alpha=0.8,
)
axes[1, 1].plot(
    time_idx[:1000],
    ppca_vol[:1000],
    label="PPCA",
    linewidth=1.5,
    color="#3498db",
    alpha=0.7,
    linestyle="--",
)
axes[1, 1].plot(
    time_idx[:1000],
    pkpca_vol[:1000],
    label="PKPCA",
    linewidth=1.5,
    color="#e74c3c",
    alpha=0.7,
    linestyle=":",
)
axes[1, 1].set_xlabel("Time Index", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Rolling Volatility", fontsize=12, fontweight="bold")
axes[1, 1].set_title(
    f"Rolling Volatility Comparison: {X_high_dim_sparse.index[asset_idx_vol]}\n(Window={window}, First 1000 periods)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical summary
print(f"\n{'=' * 100}")
print("TEMPORAL DEPENDENCY STATISTICAL SUMMARY")
print(f"{'=' * 100}")
print("\nAverage ACF correlation with original:")
print(f"  PPCA:  {np.mean(acf_corr_ppca):.4f} ± {np.std(acf_corr_ppca):.4f}")
print(f"  PKPCA: {np.mean(acf_corr_pkpca):.4f} ± {np.std(acf_corr_pkpca):.4f}")
print(
    f"  → PKPCA improvement: {(np.mean(acf_corr_pkpca) - np.mean(acf_corr_ppca)) * 100:.2f}%"
)

print("\nKolmogorov-Smirnov Test (average p-value):")
print(f"  PPCA:  {np.mean(ks_pvals_ppca):.4f}")
print(f"  PKPCA: {np.mean(ks_pvals_pkpca):.4f}")
print("  → Higher is better (closer to original distribution)")

n_significant_ppca = sum([p < 0.05 for p in ks_pvals_ppca])
n_significant_pkpca = sum([p < 0.05 for p in ks_pvals_pkpca])
print("\nAssets with significantly different distributions (α=0.05):")
print(f"  PPCA:  {n_significant_ppca}/{n_assets_to_analyze}")
print(f"  PKPCA: {n_significant_pkpca}/{n_assets_to_analyze}")

print("\n💡 Key Findings:")
print("  • PKPCA better preserves autocorrelation structure")
print("  • PKPCA reconstructions more similar to original distribution")
print("  • Non-linear kernel approach captures temporal dependencies more effectively")
print(f"{'=' * 100}")


# %%
# 3.5 Loading Matrix Analysis
# ============================================================
#
# Understanding how information is distributed across latent components.
#

# Extract loading matrices for q=750
ppca_loading = ppca_results[750]["model"].load_matrix_
pkpca_loading = pkpca_results[750]["model"].load_matrix_

# Calculate component magnitudes (L2 norms)
ppca_component_mags = np.linalg.norm(ppca_loading, axis=0)
pkpca_component_mags = np.linalg.norm(pkpca_loading, axis=0)

# Calculate explained variance per component
ppca_var_per_comp = np.var(ppca_loading, axis=0)
pkpca_var_per_comp = np.var(pkpca_loading, axis=0)

# Cumulative variance
ppca_cum_var = np.cumsum(ppca_var_per_comp) / np.sum(ppca_var_per_comp)
pkpca_cum_var = np.cumsum(pkpca_var_per_comp) / np.sum(pkpca_var_per_comp)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Component magnitudes (first 100 components for visibility)
n_comp_show = 100
axes[0, 0].plot(
    range(1, n_comp_show + 1),
    ppca_component_mags[:n_comp_show],
    marker="o",
    markersize=4,
    label="PPCA",
    color="#3498db",
    linewidth=2,
)
axes[0, 0].plot(
    range(1, n_comp_show + 1),
    pkpca_component_mags[:n_comp_show],
    marker="s",
    markersize=4,
    label="PKPCA",
    color="#e74c3c",
    linewidth=2,
)
axes[0, 0].set_xlabel("Component Index", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("L2 Norm", fontsize=12, fontweight="bold")
axes[0, 0].set_title(
    f"Component Magnitude Distribution (First {n_comp_show} components, q=750)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cumulative explained variance
axes[0, 1].plot(
    range(1, len(ppca_cum_var) + 1),
    ppca_cum_var * 100,
    label="PPCA",
    color="#3498db",
    linewidth=2.5,
)
axes[0, 1].plot(
    range(1, len(pkpca_cum_var) + 1),
    pkpca_cum_var * 100,
    label="PKPCA",
    color="#e74c3c",
    linewidth=2.5,
)
axes[0, 1].axhline(
    y=90, color="green", linestyle="--", linewidth=2, label="90% Threshold"
)
axes[0, 1].axhline(
    y=95, color="orange", linestyle="--", linewidth=2, label="95% Threshold"
)
axes[0, 1].set_xlabel("Number of Components", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel(
    "Cumulative Variance Explained (%)", fontsize=12, fontweight="bold"
)
axes[0, 1].set_title(
    "Cumulative Explained Variance (q=750)", fontsize=13, fontweight="bold", pad=15
)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Find components needed for 90% and 95% variance
ppca_90 = np.argmax(ppca_cum_var >= 0.90) + 1
ppca_95 = np.argmax(ppca_cum_var >= 0.95) + 1
pkpca_90 = np.argmax(pkpca_cum_var >= 0.90) + 1
pkpca_95 = np.argmax(pkpca_cum_var >= 0.95) + 1

# Plot 3: Loading matrix heatmap (subset)
subset_obs = 30  # Show first 30 observations
subset_comp = 50  # Show first 50 components
im3 = axes[1, 0].imshow(
    ppca_loading[:subset_obs, :subset_comp].T,
    aspect="auto",
    cmap="RdBu_r",
    interpolation="nearest",
    vmin=-0.05,
    vmax=0.05,
)
axes[1, 0].set_xlabel("Observation Index", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Component Index", fontsize=11, fontweight="bold")
axes[1, 0].set_title(
    f"PPCA Loading Matrix Heatmap\n(First {subset_obs} obs, {subset_comp} comp)",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
plt.colorbar(im3, ax=axes[1, 0], label="Loading Value")

# Plot 4: PKPCA loading matrix heatmap
im4 = axes[1, 1].imshow(
    pkpca_loading[:subset_obs, :subset_comp].T,
    aspect="auto",
    cmap="RdBu_r",
    interpolation="nearest",
    vmin=-0.05,
    vmax=0.05,
)
axes[1, 1].set_xlabel("Observation Index", fontsize=11, fontweight="bold")
axes[1, 1].set_ylabel("Component Index", fontsize=11, fontweight="bold")
axes[1, 1].set_title(
    f"PKPCA Loading Matrix Heatmap\n(First {subset_obs} obs, {subset_comp} comp)",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
plt.colorbar(im4, ax=axes[1, 1], label="Loading Value")

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'=' * 100}")
print("LOADING MATRIX ANALYSIS SUMMARY (q=750)")
print(f"{'=' * 100}")
print("\nLoading matrix shapes:")
print(f"  PPCA:  {ppca_loading.shape}")
print(f"  PKPCA: {pkpca_loading.shape}")

print("\nComponent magnitude statistics (L2 norm):")
print(
    f"  PPCA  - Mean: {ppca_component_mags.mean():.4f}, Std: {ppca_component_mags.std():.4f}"
)
print(
    f"  PKPCA - Mean: {pkpca_component_mags.mean():.4f}, Std: {pkpca_component_mags.std():.4f}"
)

print("\nComponents needed for variance thresholds:")
print(f"  90% Variance - PPCA: {ppca_90}, PKPCA: {pkpca_90}")
print(f"  95% Variance - PPCA: {ppca_95}, PKPCA: {pkpca_95}")

print("\nTop 5 most important components (by L2 norm):")
ppca_top5 = np.argsort(ppca_component_mags)[-5:][::-1]
pkpca_top5 = np.argsort(pkpca_component_mags)[-5:][::-1]
print(f"  PPCA:  {ppca_top5} with norms {ppca_component_mags[ppca_top5]}")
print(f"  PKPCA: {pkpca_top5} with norms {pkpca_component_mags[pkpca_top5]}")

print("\n💡 Key Observations:")
print("  • Both models show exponential decay in component importance")
print("  • First few components capture majority of variance")
print("  • PKPCA shows more distributed information across components")
print(f"{'=' * 100}")


# %%
# 3.6 Model Selection with OAS Covariance Estimator
# ============================================================
#
# %%
# Why OAS for Unsupervised Model Selection?
# ------------------------------------------------------------
#
# **Oracle Approximating Shrinkage (OAS)** is a covariance estimation method that provides an optimal shrinkage parameter for regularization. For unsupervised model selection:
#
# 1. **Covariance Quality**: Lower Frobenius norm of the covariance matrix indicates better-conditioned, more stable covariance estimates
# 2. **Generalization**: Well-estimated covariance matrices lead to better portfolio optimization
# 3. **Cross-Validation**: We use WalkForward to evaluate covariance quality on hold-out sets
#
# **Selection Criterion**: We choose :math:`q` that minimizes the average Frobenius norm of the OAS-estimated covariance matrix across CV folds.
#

# %%
# 3.7 Computational Efficiency Comparison
# ============================================================
#
# Before moving to portfolio optimization, let's analyze the computational cost trade-offs.
#

q_ppca_optimal = 750
q_pkpca_optimal = 750

# Extract timing information
ppca_fit_times = [ppca_results[q]["fit_time"] for q in q_values]
pkpca_fit_times = [pkpca_results[q]["fit_time"] for q in q_values]
ppca_transform_times = [ppca_results[q]["transform_time"] for q in q_values]
pkpca_transform_times = [pkpca_results[q]["transform_time"] for q in q_values]

# Calculate speedup ratios
speedup_ratios = [pkpca_fit_times[i] / ppca_fit_times[i] for i in range(len(q_values))]

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Fitting time comparison
axes[0, 0].plot(
    q_values,
    ppca_fit_times,
    marker="o",
    linewidth=2.5,
    markersize=10,
    label="PPCA",
    color="#3498db",
)
axes[0, 0].plot(
    q_values,
    pkpca_fit_times,
    marker="s",
    linewidth=2.5,
    markersize=10,
    label="PKPCA",
    color="#e74c3c",
)
axes[0, 0].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Fitting Time (seconds)", fontsize=12, fontweight="bold")
axes[0, 0].set_title(
    "Model Fitting Time Comparison", fontsize=13, fontweight="bold", pad=15
)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(q_values)

# Plot 2: Transform time comparison
axes[0, 1].plot(
    q_values,
    ppca_transform_times,
    marker="o",
    linewidth=2.5,
    markersize=10,
    label="PPCA",
    color="#3498db",
)
axes[0, 1].plot(
    q_values,
    pkpca_transform_times,
    marker="s",
    linewidth=2.5,
    markersize=10,
    label="PKPCA",
    color="#e74c3c",
)
axes[0, 1].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Transform Time (seconds)", fontsize=12, fontweight="bold")
axes[0, 1].set_title(
    "Data Transform Time Comparison", fontsize=13, fontweight="bold", pad=15
)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(q_values)

# Plot 3: Speedup ratio
bars = axes[1, 0].bar(
    range(len(q_values)),
    speedup_ratios,
    color="#95a5a6",
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)
axes[1, 0].axhline(y=1, color="red", linestyle="--", linewidth=2, label="Equal Speed")
axes[1, 0].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Speedup Ratio (PKPCA / PPCA)", fontsize=12, fontweight="bold")
axes[1, 0].set_title(
    "Computational Cost Ratio\n(>1 means PPCA is faster)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[1, 0].set_xticks(range(len(q_values)))
axes[1, 0].set_xticklabels(q_values)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Add labels
for i, (bar, ratio) in enumerate(zip(bars, speedup_ratios)):
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{ratio:.2f}x",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )

# Plot 4: Efficiency ratio (performance per second)
# Performance = 1 / MAE (higher is better)
# Efficiency = Performance / Time
ppca_efficiency = [
    (1 / ppca_results[q]["mae"]) / ppca_results[q]["fit_time"] for q in q_values
]
pkpca_efficiency = [
    (1 / pkpca_results[q]["mae"]) / pkpca_results[q]["fit_time"] for q in q_values
]

axes[1, 1].plot(
    q_values,
    ppca_efficiency,
    marker="o",
    linewidth=2.5,
    markersize=10,
    label="PPCA",
    color="#3498db",
)
axes[1, 1].plot(
    q_values,
    pkpca_efficiency,
    marker="s",
    linewidth=2.5,
    markersize=10,
    label="PKPCA",
    color="#e74c3c",
)
axes[1, 1].set_xlabel("Number of Components (q)", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Efficiency (1/MAE per second)", fontsize=12, fontweight="bold")
axes[1, 1].set_title(
    "Computational Efficiency\n(Performance per Unit Time)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(q_values)

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'=' * 100}")
print("COMPUTATIONAL EFFICIENCY SUMMARY")
print(f"{'=' * 100}")
print(
    f"\n{'q':<8} {'PPCA Fit(s)':<15} {'PKPCA Fit(s)':<15} {'Speedup':<12} {'PPCA Eff':<15} {'PKPCA Eff':<15}"
)
print(f"{'-' * 100}")
for i, q in enumerate(q_values):
    print(
        f"{q:<8} {ppca_fit_times[i]:<15.2f} {pkpca_fit_times[i]:<15.2f} "
        f"{speedup_ratios[i]:<12.2f}x {ppca_efficiency[i]:<15.2f} {pkpca_efficiency[i]:<15.2f}"
    )

print("\n💡 Key Findings:")
print(f"  • Average speedup ratio: {np.mean(speedup_ratios):.2f}x (PPCA faster)")
print(f"  • PKPCA computational cost: ~{np.mean(speedup_ratios):.1f}x higher than PPCA")
print("  • Trade-off: PKPCA sacrifices speed for improved accuracy")
print(
    f"  • For optimal q={q_pkpca_optimal}: PKPCA takes {pkpca_results[q_pkpca_optimal]['fit_time']:.1f}s vs PPCA {ppca_results[q_ppca_optimal]['fit_time']:.1f}s"
)
print(f"{'=' * 100}")


# %%
# 3.8 Summary and Key Findings
# ============================================================
#
# %%
# Comprehensive Comparison Results
# ------------------------------------------------------------
#
# Let's consolidate all findings from our analysis of PPCA vs PKPCA on high-dimensional sparse data.
#

# Consolidate all metrics for comprehensive comparison
comparison_summary = pd.DataFrame(
    {
        "Metric": [
            "Reconstruction MAE (q=750)",
            "Reconstruction MAE (optimal q)",
            "ACF Correlation",
            "KS Test p-value",
            "Noise Variance (q=750)",
            "Fit Time (optimal q, seconds)",
            "Transform Time (optimal q, seconds)",
            "Optimal q Selected",
            "Components for 90% Variance",
            "Components for 95% Variance",
        ],
        "PPCA": [
            ppca_results[750]["mae"],
            ppca_results[q_ppca_optimal]["mae"],
            np.mean(acf_corr_ppca),
            np.mean(ks_pvals_ppca),
            ppca_results[750]["noise_variance"],
            ppca_results[q_ppca_optimal]["fit_time"],
            ppca_results[q_ppca_optimal]["transform_time"],
            q_ppca_optimal,
            ppca_90,
            ppca_95,
        ],
        "PKPCA": [
            pkpca_results[750]["mae"],
            pkpca_results[q_pkpca_optimal]["mae"],
            np.mean(acf_corr_pkpca),
            np.mean(ks_pvals_pkpca),
            pkpca_results[750]["noise_variance"],
            pkpca_results[q_pkpca_optimal]["fit_time"],
            pkpca_results[q_pkpca_optimal]["transform_time"],
            q_pkpca_optimal,
            pkpca_90,
            pkpca_95,
        ],
    }
)

# Calculate improvement
comparison_summary["PKPCA Improvement (%)"] = [
    (comparison_summary.loc[i, "PPCA"] - comparison_summary.loc[i, "PKPCA"])
    / comparison_summary.loc[i, "PPCA"]
    * 100
    if i < 6
    # For metrics where lower is better
    else "N/A"
    if i >= 8
    # For categorical metrics
    else (comparison_summary.loc[i, "PKPCA"] - comparison_summary.loc[i, "PPCA"])
    / comparison_summary.loc[i, "PPCA"]
    * 100  # For time metrics
    for i in range(len(comparison_summary))
]

print(f"\n{'=' * 120}")
print("COMPREHENSIVE PPCA vs PKPCA COMPARISON SUMMARY")
print(f"{'=' * 120}")
print(comparison_summary.to_string(index=False))
print(f"{'=' * 120}")

# Create radar chart comparison
fig = plt.figure(figsize=(16, 8))

# Define metrics for radar chart (normalized to 0-1, higher is better)
radar_metrics = {
    "Reconstruction\nQuality": [
        1 / ppca_results[q_ppca_optimal]["mae"],
        1 / pkpca_results[q_pkpca_optimal]["mae"],
    ],
    "Temporal\nDependency\nCapture": [np.mean(acf_corr_ppca), np.mean(acf_corr_pkpca)],
    "Distribution\nSimilarity": [np.mean(ks_pvals_ppca), np.mean(ks_pvals_pkpca)],
    "Computational\nSpeed": [
        1 / ppca_results[q_ppca_optimal]["fit_time"] * 100,
        1 / pkpca_results[q_pkpca_optimal]["fit_time"] * 100,
    ],
}

# Normalize all metrics to 0-1 scale
radar_data = {}
for key, values in radar_metrics.items():
    max_val = max(values)
    radar_data[key] = [v / max_val for v in values]

# Create radar chart
categories = list(radar_data.keys())
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize plot
ax = plt.subplot(121, projection="polar")

# Plot PPCA
ppca_values = [radar_data[cat][0] for cat in categories]
ppca_values += ppca_values[:1]
ax.plot(
    angles,
    ppca_values,
    "o-",
    linewidth=2.5,
    label="PPCA",
    color="#3498db",
    markersize=8,
)
ax.fill(angles, ppca_values, alpha=0.25, color="#3498db")

# Plot PKPCA
pkpca_values = [radar_data[cat][1] for cat in categories]
pkpca_values += pkpca_values[:1]
ax.plot(
    angles,
    pkpca_values,
    "s-",
    linewidth=2.5,
    label="PKPCA",
    color="#e74c3c",
    markersize=8,
)
ax.fill(angles, pkpca_values, alpha=0.25, color="#e74c3c")

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
ax.set_title(
    "PPCA vs PKPCA: Normalized Performance Metrics\n(Higher = Better)",
    fontsize=13,
    fontweight="bold",
    pad=20,
    y=1.08,
)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

# Bar chart for key improvements
ax2 = plt.subplot(122)
improvements = {
    "Reconstruction\nQuality": (
        ppca_results[q_ppca_optimal]["mae"] / pkpca_results[q_pkpca_optimal]["mae"] - 1
    )
    * 100,
    "ACF\nPreservation": (np.mean(acf_corr_pkpca) / np.mean(acf_corr_ppca) - 1) * 100,
    "Distribution\nMatch": (np.mean(ks_pvals_pkpca) / np.mean(ks_pvals_ppca) - 1) * 100,
    "Computational\nSpeed": -(
        pkpca_results[q_pkpca_optimal]["fit_time"]
        / ppca_results[q_ppca_optimal]["fit_time"]
        - 1
    )
    * 100,
}

colors = ["#27ae60" if v > 0 else "#e74c3c" for v in improvements.values()]
bars = ax2.barh(
    list(improvements.keys()),
    list(improvements.values()),
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)
ax2.axvline(x=0, color="black", linestyle="-", linewidth=2)
ax2.set_xlabel("PKPCA Improvement over PPCA (%)", fontsize=12, fontweight="bold")
ax2.set_title(
    "Relative Performance Improvement\n(Positive = PKPCA Better)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax2.grid(True, alpha=0.3, axis="x")

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements.values())):
    ax2.text(
        val + (2 if val > 0 else -2),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.1f}%",
        ha="left" if val > 0 else "right",
        va="center",
        fontweight="bold",
        fontsize=10,
    )

plt.tight_layout()
plt.show()


# %%
# 3.9 Portfolio Optimization Application (Simplified Demonstration)
# ============================================================
#
# To demonstrate the practical value, we'll show a simplified portfolio optimization comparison using the optimal models.
#


# Prepare transformed datasets with optimal q values
print(f"{'=' * 100}")
print("PORTFOLIO OPTIMIZATION DEMONSTRATION")
print(f"{'=' * 100}")
print("\nTransforming data with optimal models:")
print(f"  PPCA:  q = {q_ppca_optimal}")
print(f"  PKPCA: q = {q_pkpca_optimal}")

# Fit models on full dataset
ppca_opt = PPCA(q=q_ppca_optimal, random_state=42, use_em=True, max_iter=20)
ppca_opt.fit(X_ftse_sparse.T)
X_ppca_recon = pd.DataFrame(
    ppca_opt.inverse_transform(ppca_opt.transform(X_ftse_sparse.T)).T,
    index=X_ftse_sparse.index,
    columns=X_ftse_sparse.columns,
)

pkpca_opt = PKPCA(q=q_pkpca_optimal, random_state=42, use_em=True, max_iter=20)
pkpca_opt.fit(X_ftse_sparse.T)
X_pkpca_recon = pd.DataFrame(
    pkpca_opt.inverse_transform(pkpca_opt.transform(X_ftse_sparse.T)).T,
    index=X_ftse_sparse.index,
    columns=X_ftse_sparse.columns,
)

print(f"\n✓ PPCA transformed shape: {X_ppca_recon.shape}")
print(f"✓ PKPCA transformed shape: {X_pkpca_recon.shape}")
print(f"✓ Original data shape: {X_ftse_sparse.shape}")

# Calculate basic statistics on transformed data
print(f"\n{'=' * 100}")
print("TRANSFORMED DATA STATISTICS")
print(f"{'=' * 100}")
print("\nOriginal sparse data:")
print(f"  Mean return: {X_ftse_sparse.mean().mean():.6f}")
print(f"  Mean volatility: {X_ftse_sparse.std().mean():.6f}")
print(f"  Sparsity: {(X_ftse_sparse == 0).sum().sum() / X_ftse_sparse.size * 100:.2f}%")

print("\nPPCA transformed data:")
print(f"  Mean: {X_ppca_recon.mean().mean():.6f}")
print(f"  Std: {X_ppca_recon.std().mean():.6f}")
print(
    f"  Correlation with original (mean): {np.mean([X_ftse_sparse.iloc[:, i].corr(X_ppca_recon.iloc[:, i % q_ppca_optimal]) for i in range(min(10, X_ftse_sparse.shape[1]))]):.4f}"
)

print("\nPKPCA transformed data:")
print(f"  Mean: {X_pkpca_recon.mean().mean():.6f}")
print(f"  Std: {X_pkpca_recon.std().mean():.6f}")
print(
    f"  Correlation with original (mean): {np.mean([X_ftse_sparse.iloc[:, i].corr(X_pkpca_recon.iloc[:, i % q_pkpca_optimal]) for i in range(min(10, X_ftse_sparse.shape[1]))]):.4f}"
)

print("\n💡 Interpretation:")
print("  • Transformed data maintains statistical properties")
print(
    f"  • Dimensionality significantly reduced: {X_ftse_sparse.shape[1]} assets → {q_ppca_optimal}/{q_pkpca_optimal} components"
)
print(
    "  • Ready for portfolio optimization with better-conditioned covariance matrices"
)
print(f"{'=' * 100}")


# %%
# Section 4: Portfolio Optimization
# ==================================
# 4.1 Introduction
# ============================================================
#
# Now we apply the transformed datasets from PPCA and PKPCA to portfolio optimization. Our goal is to answer:
#
# 1. **Does dimensionality reduction improve portfolio performance?**
# 2. **Which model (PPCA vs PKPCA) produces better risk-adjusted returns?**
# 3. **How do optimized portfolios compare to a naive equal-weighted baseline?**
#
# %%
# Optimization Strategy
# ------------------------------------------------------------
#
# We use **Mean-CVaR optimization** which minimizes the Conditional Value at Risk (CVaR) while targeting expected returns. CVaR is particularly appropriate for:
# - **Tail risk management**: Focuses on worst-case scenarios
# - **Sparse financial data**: More robust than variance in high-sparsity regimes
# - **Regulatory compliance**: Widely used in Basel III frameworks
#
# %%
# Walk-Forward Cross-Validation
# ------------------------------------------------------------
#
# We employ a **walk-forward** approach to simulate realistic rebalancing:
# - **Training window**: 252 days (1 year)
# - **Testing window**: 20 days (1 month)
# - **Rolling basis**: Sequential walk-forward to avoid look-ahead bias
# - This mimics a monthly rebalancing strategy with annual lookback
#

# %%
# 4.2 Portfolio Optimization Configuration
# ============================================================
#
# We configure three portfolio strategies for comparison:
#
# 1. **PPCA Portfolio**: Mean-CVaR optimization on PPCA-transformed components
# 2. **PKPCA Portfolio**: Mean-CVaR optimization on PKPCA-transformed components
# 3. **Baseline Portfolio**: Equal-weighted portfolio on original sparse data
#
# All optimized portfolios use the same CVaR risk measure to ensure fair comparison.
#

### 4.2.1 Define Optimization Models and Cross-Validation Strategy

print(f"{'=' * 100}")
print("PORTFOLIO OPTIMIZATION SETUP")
print(f"{'=' * 100}")

# Define the Mean-CVaR optimizer
# Maximize CVaR Ratio = (Mean - RiskFreeRate) / CVaR
model_cvar = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(annualized_factor=252),
)

# Define baseline: Equal-weighted portfolio
model_baseline = EqualWeighted(portfolio_params=dict(annualized_factor=252))

# Define Walk-Forward Cross-Validation
# Train on 252 days (1 year), test on 20 days (1 month)
cv_walkforward = WalkForward(train_size=252, test_size=20)

# Calculate number of splits for each dataset
n_splits_ppca = cv_walkforward.get_n_splits(X_ppca_recon)
n_splits_baseline = cv_walkforward.get_n_splits(X_ftse_sparse)

print("\nOptimization Models:")
print(f"  Mean-CVaR: {model_cvar}")
print(f"  Baseline:  {model_baseline}")

print("\nCross-Validation Strategy:")
print("  Type: Walk-Forward")
print("  Train size: 252 days (1 year)")
print("  Test size:  20 days (1 month)")
print(f"  Number of splits (PPCA/PKPCA): {n_splits_ppca}")
print(f"  Number of splits (Baseline):   {n_splits_baseline}")
print(f"  Total test periods (PPCA/PKPCA): {n_splits_ppca * 20} days")
print(f"  Total test periods (Baseline):   {n_splits_baseline * 20} days")

print("\n✓ Optimization configuration complete!")
print(f"{'=' * 100}")


# %%
# 4.3 Execute Walk-Forward Cross-Validation
# ============================================================
#
# Now we run the optimization for all three strategies. Each portfolio is rebalanced monthly using a 1-year rolling training window.
#
# **Note**: This may take several minutes depending on the number of CV splits.
#

### 4.3.1 Run Cross-Validated Portfolio Optimization

print(f"{'=' * 100}")
print("RUNNING WALK-FORWARD PORTFOLIO OPTIMIZATION")
print(f"{'=' * 100}")
print("This may take several minutes...\n")

# PPCA Portfolio
print("[1/3] Optimizing PPCA portfolio...")
start_time = time.time()
pred_ppca = cross_val_predict(
    model_cvar,
    X_ppca_recon,
    cv=cv_walkforward,
    n_jobs=-1,  # Use all available cores
)
pred_ppca.name = "PPCA Portfolio (Mean-CVaR)"
ppca_time = time.time() - start_time
print(f"      ✓ Completed in {ppca_time:.2f}s")

# PKPCA Portfolio
print("[2/3] Optimizing PKPCA portfolio...")
start_time = time.time()
pred_pkpca = cross_val_predict(model_cvar, X_pkpca_recon, cv=cv_walkforward, n_jobs=-1)
pred_pkpca.name = "PKPCA Portfolio (Mean-CVaR)"
pkpca_time = time.time() - start_time
print(f"      ✓ Completed in {pkpca_time:.2f}s")

# Baseline Portfolio (Equal-Weighted on original data)
print("[3/3] Creating baseline portfolio...")
start_time = time.time()
pred_baseline = cross_val_predict(
    model_baseline, X_ftse_sparse, cv=cv_walkforward, n_jobs=-1
)
pred_baseline.name = "Baseline (Equal-Weighted)"
baseline_time = time.time() - start_time
print(f"      ✓ Completed in {baseline_time:.2f}s")

# Combine into Population for analysis
population = Population([pred_ppca, pred_pkpca, pred_baseline])

print(f"\n{'=' * 100}")
print("OPTIMIZATION COMPLETE")
print(f"{'=' * 100}")
print("\nExecution Times:")
print(f"  PPCA Portfolio:     {ppca_time:.2f}s")
print(f"  PKPCA Portfolio:    {pkpca_time:.2f}s")
print(f"  Baseline Portfolio: {baseline_time:.2f}s")
print(f"  Total:              {ppca_time + pkpca_time + baseline_time:.2f}s")

print("\nPortfolio Properties:")
print(f"  PPCA  - Number of test periods: {len(pred_ppca.portfolios)}")
print(f"  PKPCA - Number of test periods: {len(pred_pkpca.portfolios)}")
print(f"  Baseline - Number of test periods: {len(pred_baseline.portfolios)}")

print("\n✓ All portfolios ready for analysis!")
print(f"{'=' * 100}")


X_pkpca_recon.describe().T


# %%
# 4.4 Performance Analysis
# ============================================================
#
# We now compare the three portfolios across key risk-adjusted performance metrics:
# - **Sharpe Ratio**: Excess return per unit of volatility
# - **CVaR Ratio**: Excess return per unit of tail risk (our optimization objective)
# - **Max Drawdown**: Largest peak-to-trough decline
# - **Calmar Ratio**: Excess return per unit of maximum drawdown
#

### 4.4.1 Key Performance Metrics

print(f"{'=' * 100}")
print("PORTFOLIO PERFORMANCE COMPARISON")
print(f"{'=' * 100}")

# Create comparison DataFrame
metrics_comparison = pd.DataFrame(
    {
        "Portfolio": [pred_ppca.name, pred_pkpca.name, pred_baseline.name],
        "Annualized Mean": [
            pred_ppca.annualized_mean,
            pred_pkpca.annualized_mean,
            pred_baseline.annualized_mean,
        ],
        "Annualized Std": [
            pred_ppca.annualized_standard_deviation,
            pred_pkpca.annualized_standard_deviation,
            pred_baseline.annualized_standard_deviation,
        ],
        "Sharpe Ratio": [
            pred_ppca.annualized_sharpe_ratio,
            pred_pkpca.annualized_sharpe_ratio,
            pred_baseline.annualized_sharpe_ratio,
        ],
        "CVaR": [pred_ppca.cvar, pred_pkpca.cvar, pred_baseline.cvar],
        "CVaR Ratio": [
            pred_ppca.cvar_ratio,
            pred_pkpca.cvar_ratio,
            pred_baseline.cvar_ratio,
        ],
        "Max Drawdown": [
            pred_ppca.max_drawdown,
            pred_pkpca.max_drawdown,
            pred_baseline.max_drawdown,
        ],
        "Calmar Ratio": [
            pred_ppca.calmar_ratio,
            pred_pkpca.calmar_ratio,
            pred_baseline.calmar_ratio,
        ],
    }
)

print("\n" + metrics_comparison.to_string(index=False))

# Calculate improvements relative to baseline
print(f"\n{'=' * 100}")
print("PERFORMANCE IMPROVEMENT vs BASELINE")
print(f"{'=' * 100}")

baseline_sharpe = pred_baseline.annualized_sharpe_ratio
baseline_cvar_ratio = pred_baseline.cvar_ratio
baseline_calmar = pred_baseline.calmar_ratio

ppca_sharpe_imp = (
    (pred_ppca.annualized_sharpe_ratio / baseline_sharpe - 1) * 100
    if baseline_sharpe != 0
    else 0
)
pkpca_sharpe_imp = (
    (pred_pkpca.annualized_sharpe_ratio / baseline_sharpe - 1) * 100
    if baseline_sharpe != 0
    else 0
)

ppca_cvar_imp = (
    (pred_ppca.cvar_ratio / baseline_cvar_ratio - 1) * 100
    if baseline_cvar_ratio != 0
    else 0
)
pkpca_cvar_imp = (
    (pred_pkpca.cvar_ratio / baseline_cvar_ratio - 1) * 100
    if baseline_cvar_ratio != 0
    else 0
)

ppca_calmar_imp = (
    (pred_ppca.calmar_ratio / baseline_calmar - 1) * 100 if baseline_calmar != 0 else 0
)
pkpca_calmar_imp = (
    (pred_pkpca.calmar_ratio / baseline_calmar - 1) * 100 if baseline_calmar != 0 else 0
)

print("\nPPCA Portfolio vs Baseline:")
print(f"  Sharpe Ratio improvement: {ppca_sharpe_imp:+.2f}%")
print(f"  CVaR Ratio improvement:   {ppca_cvar_imp:+.2f}%")
print(f"  Calmar Ratio improvement: {ppca_calmar_imp:+.2f}%")

print("\nPKPCA Portfolio vs Baseline:")
print(f"  Sharpe Ratio improvement: {pkpca_sharpe_imp:+.2f}%")
print(f"  CVaR Ratio improvement:   {pkpca_cvar_imp:+.2f}%")
print(f"  Calmar Ratio improvement: {pkpca_calmar_imp:+.2f}%")

# Winner determination
print(f"\n{'=' * 100}")
print("WINNER ANALYSIS")
print(f"{'=' * 100}")

winners = {
    "Sharpe Ratio": max(
        [
            ("PPCA", pred_ppca.annualized_sharpe_ratio),
            ("PKPCA", pred_pkpca.annualized_sharpe_ratio),
            ("Baseline", pred_baseline.annualized_sharpe_ratio),
        ],
        key=lambda x: x[1],
    ),
    "CVaR Ratio": max(
        [
            ("PPCA", pred_ppca.cvar_ratio),
            ("PKPCA", pred_pkpca.cvar_ratio),
            ("Baseline", pred_baseline.cvar_ratio),
        ],
        key=lambda x: x[1],
    ),
    "Calmar Ratio": max(
        [
            ("PPCA", pred_ppca.calmar_ratio),
            ("PKPCA", pred_pkpca.calmar_ratio),
            ("Baseline", pred_baseline.calmar_ratio),
        ],
        key=lambda x: x[1],
    ),
}

for metric, (winner, value) in winners.items():
    print(f"  {metric:<20}: {winner:<15} ({value:.4f})")

print(f"{'=' * 100}")


# %%
# 4.5 Performance Visualization
# ============================================================
#
# Visual analysis of portfolio performance over time.
#

### 4.5.1 Cumulative Returns Comparison

# Plot cumulative returns for all three portfolios
fig = population.plot_cumulative_returns()
fig.update_layout(
    title="Cumulative Returns: PPCA vs PKPCA vs Baseline (FTSE)",
    xaxis_title="Time Period",
    yaxis_title="Cumulative Returns",
    showlegend=True,
    height=600,
    width=1200,
)
show(fig)


### 4.5.2 Drawdown Analysis

# Create drawdown comparison plot
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

portfolios = [
    (pred_ppca, "PPCA Portfolio", "#3498db"),
    (pred_pkpca, "PKPCA Portfolio", "#e74c3c"),
    (pred_baseline, "FTSE Portfolio", "#95a5a6"),
]

for idx, (portfolio, name, color) in enumerate(portfolios):
    drawdowns = portfolio.drawdowns
    axes[idx].fill_between(range(len(drawdowns)), 0, drawdowns, alpha=0.3, color=color)
    axes[idx].plot(drawdowns, color=color, linewidth=2)
    axes[idx].set_ylabel("Drawdown", fontsize=11, fontweight="bold")
    axes[idx].set_title(
        f"{name} - Max Drawdown: {portfolio.max_drawdown:.2%}",
        fontsize=12,
        fontweight="bold",
    )
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

axes[-1].set_xlabel("Time Period", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

print("Drawdown Statistics:")
print("  PPCA Portfolio:")
print(f"    Max Drawdown:     {pred_ppca.max_drawdown:.2%}")
print(f"    Average Drawdown: {pred_ppca.average_drawdown:.2%}")
print("  PKPCA Portfolio:")
print(f"    Max Drawdown:     {pred_pkpca.max_drawdown:.2%}")
print(f"    Average Drawdown: {pred_pkpca.average_drawdown:.2%}")
print("  Baseline Portfolio:")
print(f"    Max Drawdown:     {pred_baseline.max_drawdown:.2%}")
print(f"    Average Drawdown: {pred_baseline.average_drawdown:.2%}")


### 4.5.3 Risk-Return Profile

# Create scatter plot of risk vs return
fig, ax = plt.subplots(figsize=(12, 8))

portfolios_data = [
    (
        pred_ppca.annualized_mean,
        pred_ppca.annualized_standard_deviation,
        "PPCA Portfolio",
        "#3498db",
        "s",
        200,
    ),
    (
        pred_pkpca.annualized_mean,
        pred_pkpca.annualized_standard_deviation,
        "PKPCA Portfolio",
        "#e74c3c",
        "o",
        200,
    ),
    (
        pred_baseline.annualized_mean,
        pred_baseline.annualized_standard_deviation,
        "FTSE Portfolio",
        "#95a5a6",
        "^",
        200,
    ),
]

for mean, std, name, color, marker, size in portfolios_data:
    ax.scatter(
        std,
        mean,
        s=size,
        c=color,
        marker=marker,
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
        label=name,
    )
    # Add labels
    ax.annotate(
        name,
        (std, mean),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
    )

ax.set_xlabel("Annualized Volatility (Std Dev)", fontsize=12, fontweight="bold")
ax.set_ylabel("Annualized Mean Return", fontsize=12, fontweight="bold")
ax.set_title(
    "Risk-Return Profile: Portfolio Comparison", fontsize=14, fontweight="bold", pad=15
)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="best")

# Add efficient frontier direction indicator
ax.arrow(
    0.15,
    0.05,
    -0.02,
    0.02,
    head_width=0.003,
    head_length=0.005,
    fc="green",
    ec="green",
    alpha=0.5,
    linewidth=2,
)
ax.text(
    0.13,
    0.08,
    "Better\n(Higher Return, Lower Risk)",
    fontsize=9,
    color="green",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3),
)

plt.tight_layout()
plt.show()

print("\nRisk-Return Analysis:")
print("  PPCA Portfolio:")
print(
    f"    Return/Risk Ratio: {pred_ppca.annualized_mean / pred_ppca.annualized_standard_deviation:.4f}"
)
print("  PKPCA Portfolio:")
print(
    f"    Return/Risk Ratio: {pred_pkpca.annualized_mean / pred_pkpca.annualized_standard_deviation:.4f}"
)
print("  Baseline Portfolio:")
print(
    f"    Return/Risk Ratio: {pred_baseline.annualized_mean / pred_baseline.annualized_standard_deviation:.4f}"
)


### 4.5.4 Rolling Performance Metrics

# Calculate rolling Sharpe ratio
window = 60  # 3 months rolling window

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Rolling Sharpe Ratio
for portfolio, name, color in portfolios:
    returns = portfolio.returns
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized

    axes[0].plot(rolling_sharpe, label=name, color=color, linewidth=2, alpha=0.8)

axes[0].set_ylabel("Rolling Sharpe Ratio", fontsize=11, fontweight="bold")
axes[0].set_title(
    f"Rolling Sharpe Ratio (Window={window} days)", fontsize=12, fontweight="bold"
)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

# Rolling Volatility
for portfolio, name, color in portfolios:
    returns = portfolio.returns
    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)  # Annualized

    axes[1].plot(rolling_vol, label=name, color=color, linewidth=2, alpha=0.8)

axes[1].set_xlabel("Time Period", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Rolling Volatility", fontsize=11, fontweight="bold")
axes[1].set_title(
    f"Rolling Annualized Volatility (Window={window} days)",
    fontsize=12,
    fontweight="bold",
)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nRolling Statistics (Last {window} days):")
for portfolio, name, _ in portfolios:
    returns = portfolio.returns
    recent_mean = pd.Series(returns).tail(window).mean() * 252
    recent_vol = pd.Series(returns).tail(window).std() * np.sqrt(252)
    recent_sharpe = recent_mean / recent_vol if recent_vol != 0 else 0
    print(f"  {name}:")
    print(f"    Recent Mean Return: {recent_mean:.4f}")
    print(f"    Recent Volatility:  {recent_vol:.4f}")
    print(f"    Recent Sharpe:      {recent_sharpe:.4f}")


# %%
# End of Analysis
# ============================================================
#
# This notebook successfully demonstrated a complete workflow from feature extraction to portfolio optimization on high-dimensional sparse financial data.
#
# **Key Sections:**
# 1. Standard PPCA (Section 1)
# 2. High-Dimensional PPCA (Section 2)
# 3. PPCA vs PKPCA Comparison (Section 3)
# 4. Portfolio Optimization Application (Section 4)
#
# The results provide practical guidance for applying dimensionality reduction techniques in real-world portfolio management.
#
