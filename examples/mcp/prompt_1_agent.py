"""
prompt 1 agent mode:generate a portfolio with 10% annual return and at least 0.8 sharpe 
ratio in examples/mcp/prompt_1_agent.py

Begin generated code:

==========================================
Portfolio with Target Return and Sharpe Ratio
==========================================

This example demonstrates how to create a portfolio with a target annual return of 10%
and a minimum Sharpe ratio of 0.8 using skfolio's MeanRisk optimization.

The optimization uses:
- Target return constraint: min_return=0.10 (10% annual)
- Sharpe ratio optimization: MAXIMIZE_RATIO objective
- Risk measure: Standard deviation for Sharpe ratio calculation
"""

# %%
# Data
# ====
# We load the S&P 500 dataset composed of daily prices of 20 assets from the S&P 500 Index
# starting from 1990-01-02 up to 2022-12-28. Prices are transformed into linear returns
# and split into training and test sets.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

# Load and prepare data
prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Assets: {list(X_train.columns)}")

# %%
# Model Configuration
# ===================
# We create a MeanRisk model with the following parameters:
# - objective_function=MAXIMIZE_RATIO: Optimizes for maximum Sharpe ratio
# - risk_measure=STANDARD_DEVIATION: Uses standard deviation for Sharpe ratio calculation
# - min_return=0.10/252: Sets minimum daily return constraint (10% annualized)
# - portfolio_params: Names the portfolio for identification

target_annual_return = 0.10  # 10% annual return
min_daily_return = target_annual_return / 252  # Convert to daily return

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    min_return=min_daily_return,
    portfolio_params=dict(name="Target 10% Return + Max Sharpe"),
)

# %%
# Training
# ========
# Fit the model on the training data
model.fit(X_train)

print("Portfolio weights:")
weights_df = model.weights_
print(weights_df)

print(f"\nSum of weights: {weights_df.sum():.6f}")

# %%
# Prediction and Analysis
# =======================
# Predict the portfolio performance on the test set
portfolio = model.predict(X_test)

print(f"\nPortfolio Performance on Test Set:")
print(f"Annualized Return: {portfolio.annualized_mean:.4f} ({portfolio.annualized_mean*100:.2f}%)")
print(f"Annualized Volatility: {portfolio.annualized_standard_deviation:.4f} ({portfolio.annualized_standard_deviation*100:.2f}%)")
print(f"Annualized Sharpe Ratio: {portfolio.annualized_sharpe_ratio:.4f}")

# Check if we meet the requirements
target_met = portfolio.annualized_mean >= target_annual_return
sharpe_met = portfolio.annualized_sharpe_ratio >= 0.8

print(f"\nRequirements Check:")
print(f"✓ Target Return (≥10%): {target_met} ({portfolio.annualized_mean*100:.2f}%)")
print(f"✓ Sharpe Ratio (≥0.8): {sharpe_met} ({portfolio.annualized_sharpe_ratio:.4f})")

# %%
# Additional Metrics
# ==================
# Display additional risk and performance metrics
print(f"\nAdditional Portfolio Metrics:")
print(f"Maximum Drawdown: {portfolio.max_drawdown:.4f} ({portfolio.max_drawdown*100:.2f}%)")
print(f"Sortino Ratio: {portfolio.annualized_sortino_ratio:.4f}")
print(f"CVaR (95%): {portfolio.cvar:.4f} ({portfolio.cvar*100:.2f}%)")

# %%
# Portfolio Composition
# =====================
# Show the top 10 holdings by weight
print(f"\nTop 10 Holdings:")
# Convert weights to pandas Series for easier manipulation
weights_series = pd.Series(weights_df, index=X_train.columns)
top_holdings = weights_series.nlargest(10)
for asset, weight in top_holdings.items():
    print(f"{asset}: {weight:.4f} ({weight*100:.2f}%)")

# %%
# Summary
# =======
# Display a comprehensive summary of the portfolio
print(f"\n" + "="*50)
print(f"PORTFOLIO SUMMARY")
print(f"="*50)
print(f"Strategy: Target 10% Return + Maximum Sharpe Ratio")
print(f"Assets: {len(weights_df)}")
print(f"Training Period: {X_train.index[0].strftime('%Y-%m-%d')} to {X_train.index[-1].strftime('%Y-%m-%d')}")
print(f"Test Period: {X_test.index[0].strftime('%Y-%m-%d')} to {X_test.index[-1].strftime('%Y-%m-%d')}")
print(f"")
print(f"Performance Metrics:")
print(f"  Annualized Return: {portfolio.annualized_mean*100:.2f}%")
print(f"  Annualized Volatility: {portfolio.annualized_standard_deviation*100:.2f}%")
print(f"  Sharpe Ratio: {portfolio.annualized_sharpe_ratio:.4f}")
print(f"  Sortino Ratio: {portfolio.annualized_sortino_ratio:.4f}")
print(f"  Maximum Drawdown: {portfolio.max_drawdown*100:.2f}%")
print(f"")
print(f"Requirements:")
print(f"  ✓ Target Return (≥10%): {'PASS' if target_met else 'FAIL'}")
print(f"  ✓ Sharpe Ratio (≥0.8): {'PASS' if sharpe_met else 'FAIL'}")
print(f"="*50)