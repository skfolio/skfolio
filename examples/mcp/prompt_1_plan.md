# Portfolio Optimization: 10% Annual Return with Sharpe ≥ 0.8

## Approach

Create `examples/mcp/prompt_1_plan.py` that:

1. **Load S&P 500 dataset** (20 assets, 1990-2022) and split into train/test sets

   - Use `load_sp500_dataset()` and `prices_to_returns()`
   - Standard 67/33 train/test split

2. **Generate efficient frontier portfolios** near 10% annual return target

   - Use `MeanRisk` with `efficient_frontier_size=50` to get portfolios across the full frontier
   - Daily return equivalent: 10% annual ≈ 0.10/252 daily

3. **Filter for Sharpe ratio ≥ 0.8**

   - Evaluate all portfolios on test set
   - Use `population.measures(RatioMeasure.ANNUALIZED_SHARPE_RATIO)` to get Sharpe ratios
   - Filter portfolios with `annualized_sharpe_ratio >= 0.8`
   - From filtered set, identify portfolios closest to 10% annual return

4. **Compare multiple strategies** as benchmarks

   - Maximum Sharpe Ratio portfolio
   - Inverse Volatility portfolio
   - Selected portfolios meeting both constraints

5. **Visualizations**

   - Efficient frontier plot with return vs risk, colored by Sharpe ratio
   - Portfolio composition for qualifying portfolios
   - Cumulative returns comparison
   - Summary table showing key metrics

## Key Code Snippets

```python
# Generate efficient frontier
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    efficient_frontier_size=50,
)
population_train = model.fit_predict(X_train)
population_test = model.predict(X_test)

# Filter by Sharpe >= 0.8
sharpe_ratios = population_test.measures(RatioMeasure.ANNUALIZED_SHARPE_RATIO)
returns = population_test.measures(PerfMeasure.ANNUALIZED_MEAN)
mask = sharpe_ratios >= 0.8
filtered_population = Population([p for i, p in enumerate(population_test) if mask[i]])

# Find closest to 10% return
target_return = 0.10
filtered_returns = returns[mask]
idx = np.argmin(np.abs(filtered_returns - target_return))
```

## Expected Output

- Portfolio weights and metrics for qualifying portfolios
- Visual confirmation of constraints (Sharpe ≥ 0.8, return ≈ 10%)
- Comparison showing if constraints are achievable with this dataset