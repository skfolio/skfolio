Historical workflow validation
==============================

Run command (repository root)::

    PYTHONPATH=src python integration/issue_262/historical_walk_forward.py --output-dir /tmp/skfolio-supervised-historical

Configuration
-------------

* Bundled adjusted-price example dataset; fixed eight-asset universe:
  AAPL, CVX, GE, JNJ, JPM, KO, MSFT and WMT.
* Three yearly refits, each using 756 preceding training observations.
* Evaluation years 2019, 2020 and 2021: 757 return observations, 37 rebalances.
* Characteristics: trailing 20/60-observation price changes and 20-observation
  volatility, computed without centered windows or future data.
* Labels: InverseVolatility portfolios over subsequent 21-observation training
  windows. Regressor: random forest with 50 trees, minimum leaf size 10, seed 7.
* Baseline: constrained equal-weight policy using constant zero predicted tilts.
* Long-only, 25% maximum target weight, 30% total absolute turnover per rebalance.
* Daily variance cap 0.6**2 / 252, with trailing 60-observation LedoitWolf priors.
* Rebalance every 21 observations; schedule restarts at each annual refit.
* Transaction costs 0.001 times absolute traded weight (10 basis points).
* Holdings are carried across annual refits, rather than reset to equal weight.

Observed results
----------------

===========================  =============  ===================
Metric                       Random forest  Equal-weight policy
===========================  =============  ===================
Net cumulative return        97.12%         105.42%
Annualized volatility        22.10%         23.07%
Maximum drawdown             -33.69%        -34.07%
Sum of absolute turnover     5.000          1.769
Sum of cost fractions        0.005000       0.001769
===========================  =============  ===================

The learned policy underperformed the baseline on total return and traded more.
Cost fractions above are sums of per-period charges, not a compounded terminal
wealth difference. Results may differ slightly with solver/library versions.

What this verifies
------------------

The complete historical-data path runs with chronological refits, past-only risk
estimation, stateful inference, non-daily rebalancing, drifted holdings and costs.
The script checks finite returns, budget, rebalance weight/turnover limits and
zero turnover/cost between rebalances. Unit tests separately cover risk bounds,
prior timing, combined infeasibility and failure isolation.

It does not establish profitability or production readiness. The bundled data
are a fixed historical sample, with no point-in-time constituent membership.
The example has no cash ledger, slippage, liquidity limits or corporate-action
simulation. Cost deductions follow Portfolio's existing return convention.
Risk caps apply to estimates at rebalance time, not realized future volatility.
Model/constraint parameters were fixed for a functional example, not validated
through nested model selection. New research should preserve untouched test data.

Outputs
-------

``summary.json`` records fold boundaries, settings and metrics.
``daily_execution.csv`` records targets, returns, turnover and costs for both
strategies, allowing independent reconciliation of the evaluation.
