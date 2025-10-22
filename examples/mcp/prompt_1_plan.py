"""
prompt 1 plan mode:generate a portfolio with 10% annual return and at least 0.8 sharpe 
ratio in examples/mcp/prompt_1_plan.py

Begin generated code:

Portfolio Optimization: 10% Annual Return with Sharpe ≥ 0.8
==========================================================

This script generates portfolios targeting 10% annual return with at least 0.8 Sharpe ratio
using the efficient frontier approach with the S&P 500 dataset.
"""

import numpy as np
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

# %%
# Data Loading and Preparation
# ============================
# Load the S&P 500 dataset and convert to returns
prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# %%
# Generate Efficient Frontier
# ===========================
# Create MeanRisk model to generate 50 portfolios across the efficient frontier
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    efficient_frontier_size=50,
    portfolio_params=dict(name="Efficient Frontier"),
)

# Fit on training data and predict on both train and test sets
population_train = model.fit_predict(X_train)
population_test = model.predict(X_test)

print(f"Generated {len(population_train)} portfolios on training set")
print(f"Generated {len(population_test)} portfolios on test set")

# %%
# Filter for Sharpe Ratio ≥ 0.8
# =============================
# Get Sharpe ratios and returns for all test portfolios
sharpe_ratios = population_test.measures(RatioMeasure.ANNUALIZED_SHARPE_RATIO)
returns = population_test.measures(PerfMeasure.ANNUALIZED_MEAN)

# Filter portfolios with Sharpe ratio >= 0.8
mask = sharpe_ratios >= 0.8
filtered_population = Population([p for i, p in enumerate(population_test) if mask[i]])

print(f"Portfolios with Sharpe ≥ 0.8: {len(filtered_population)} out of {len(population_test)}")

if len(filtered_population) > 0:
    filtered_returns = returns[mask]
    filtered_sharpes = sharpe_ratios[mask]
    
    # Find portfolio closest to 10% annual return
    target_return = 0.10
    idx = np.argmin(np.abs(filtered_returns - target_return))
    closest_portfolio = filtered_population[idx]
    
    print(f"\nClosest portfolio to 10% return:")
    print(f"  Annual Return: {closest_portfolio.annualized_mean:.3f}")
    print(f"  Sharpe Ratio: {closest_portfolio.annualized_sharpe_ratio:.3f}")
    print(f"  Annual Volatility: {closest_portfolio.annualized_standard_deviation:.3f}")

# %%
# Benchmark Strategies
# =====================
# Create Maximum Sharpe Ratio portfolio
max_sharpe_model = MeanRisk(
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Max Sharpe"),
)
max_sharpe_portfolio = max_sharpe_model.fit_predict(X_test)

# Create Inverse Volatility benchmark
inverse_vol_model = InverseVolatility(portfolio_params=dict(name="Inverse Volatility"))
inverse_vol_portfolio = inverse_vol_model.fit_predict(X_test)

print(f"\nBenchmark portfolios:")
print(f"Max Sharpe - Return: {max_sharpe_portfolio.annualized_mean:.3f}, Sharpe: {max_sharpe_portfolio.annualized_sharpe_ratio:.3f}")
print(f"Inverse Vol - Return: {inverse_vol_portfolio.annualized_mean:.3f}, Sharpe: {inverse_vol_portfolio.annualized_sharpe_ratio:.3f}")

# %%
# Create Comparison Population
# ===========================
# Combine all portfolios for comparison
comparison_portfolios = [max_sharpe_portfolio, inverse_vol_portfolio]

if len(filtered_population) > 0:
    # Add the closest portfolio to 10% return
    comparison_portfolios.append(closest_portfolio)
    
    # Add a few more qualifying portfolios for comparison
    if len(filtered_population) > 1:
        # Sort by how close they are to 10% return
        distances = np.abs(filtered_returns - target_return)
        sorted_indices = np.argsort(distances)
        
        # Add up to 2 more closest portfolios
        for i in range(1, min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            portfolio = filtered_population[idx]
            portfolio.name = f"Target Portfolio {i}"
            comparison_portfolios.append(portfolio)

comparison_population = Population(comparison_portfolios)

# %%
# Visualizations
# ==============
# 1. Efficient Frontier with Sharpe Ratio Coloring
print("\nGenerating efficient frontier visualization...")
fig_frontier = population_test.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
    title="Efficient Frontier - All Portfolios"
)
show(fig_frontier)

# 2. Portfolio Composition
print("\nGenerating portfolio composition visualization...")
if len(filtered_population) > 0:
    # Show composition of qualifying portfolios
    composition_portfolios = [closest_portfolio]
    if len(filtered_population) > 1:
        # Add one more qualifying portfolio for comparison
        distances = np.abs(filtered_returns - target_return)
        second_best_idx = np.argsort(distances)[1]
        second_best = filtered_population[second_best_idx]
        second_best.name = "Second Best Target"
        composition_portfolios.append(second_best)
    
    composition_population = Population(composition_portfolios)
    fig_composition = composition_population.plot_composition()
    show(fig_composition)

# 3. Cumulative Returns Comparison
print("\nGenerating cumulative returns comparison...")
fig_returns = comparison_population.plot_cumulative_returns()
show(fig_returns)

# %%
# Results Summary
# ===============
print("\n" + "="*60)
print("PORTFOLIO OPTIMIZATION RESULTS")
print("="*60)

print(f"\nDataset: S&P 500 ({X.shape[1]} assets, {X.shape[0]} observations)")
print(f"Training period: {X_train.shape[0]} observations")
print(f"Test period: {X_test.shape[0]} observations")

print(f"\nEfficient Frontier Analysis:")
print(f"  Total portfolios generated: {len(population_test)}")
print(f"  Portfolios with Sharpe ≥ 0.8: {len(filtered_population)}")

if len(filtered_population) > 0:
    print(f"\nQualifying Portfolios (Sharpe ≥ 0.8):")
    for i, portfolio in enumerate(filtered_population[:5]):  # Show top 5
        print(f"  Portfolio {i+1}: Return={portfolio.annualized_mean:.3f}, "
              f"Sharpe={portfolio.annualized_sharpe_ratio:.3f}, "
              f"Vol={portfolio.annualized_standard_deviation:.3f}")
    
    # Find best portfolio meeting both constraints
    best_idx = np.argmin(np.abs(filtered_returns - target_return))
    best_portfolio = filtered_population[best_idx]
    
    print(f"\nBest Portfolio Meeting Constraints:")
    print(f"  Name: {best_portfolio.name}")
    print(f"  Annual Return: {best_portfolio.annualized_mean:.3f} ({best_portfolio.annualized_mean*100:.1f}%)")
    print(f"  Sharpe Ratio: {best_portfolio.annualized_sharpe_ratio:.3f}")
    print(f"  Annual Volatility: {best_portfolio.annualized_standard_deviation:.3f} ({best_portfolio.annualized_standard_deviation*100:.1f}%)")
    print(f"  Max Drawdown: {best_portfolio.max_drawdown:.3f} ({best_portfolio.max_drawdown*100:.1f}%)")
    
    print(f"\nPortfolio Weights:")
    weights = best_portfolio.weights
    asset_names = X.columns
    for i, (asset, weight) in enumerate(zip(asset_names, weights)):
        if weight > 0.01:  # Only show weights > 1%
            print(f"  {asset}: {weight:.3f} ({weight*100:.1f}%)")
else:
    print(f"\n⚠️  No portfolios found with Sharpe ratio ≥ 0.8")
    print(f"   Maximum Sharpe ratio achieved: {np.max(sharpe_ratios):.3f}")
    print(f"   This suggests the constraints may be too restrictive for this dataset")

print(f"\nBenchmark Comparison:")
print(f"  Max Sharpe Portfolio:")
print(f"    Return: {max_sharpe_portfolio.annualized_mean:.3f}, Sharpe: {max_sharpe_portfolio.annualized_sharpe_ratio:.3f}")
print(f"  Inverse Volatility Portfolio:")
print(f"    Return: {inverse_vol_portfolio.annualized_mean:.3f}, Sharpe: {inverse_vol_portfolio.annualized_sharpe_ratio:.3f}")

# %%
# Detailed Summary Table
# ======================
print(f"\nDetailed Summary Table:")
comparison_population.summary()

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
