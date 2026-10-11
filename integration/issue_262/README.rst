Issue #262: supervised-policy starting point
===========================================

Branch: ``supervised-portfolio-policy``.

Based on Cashubski's open PR #341, commit
``45a69384948836f0624dcd2dc9cdd6378829e5d0``. This branch is stacked on that
unmerged work; its full diff against main also contains PR #341. Review this
prototype against that commit, not against main.

Implemented
-----------

* Separate public ``SupervisedPortfolioPolicy`` estimator with a configurable shared
  regressor (Ridge by default) of benchmark-relative portfolio weights on characteristics.
* Independent clones of a user-supplied optimizer generate labels from complete,
  non-overlapping forward training windows. No label extends outside fit data.
* Shared characteristic processing, benchmark construction and weight prediction
  from PR #341; no duplicate implementation of those components.
* Auditable label weights and window dates, clone/refit behavior, finite-data
  validation, new-characteristic prediction and Euclidean projection for budget, long-only, bounds and group equations.
* Tests for exact coefficient recovery, label boundaries, unused observations,
  real label optimizers, asset ordering, invalid data and failed refits.

Next design decisions
---------------------

1. Extract the shared characteristic/policy helpers into a private base or mixin
   with the PR #341 author. This prototype subclasses ParametricPortfolioPolicy
   solely for reuse; it does not call its utility solver. A production hierarchy
   should avoid carrying the unused utility-fitting configuration.
2. The current API supports arbitrary compatible regressors and pipelines on
   asset/date rows. Decide whether to add joint whole-portfolio regression later.
   This remains a baseline, not a reproduction of the full published method.
3. Agree on training windows, weighting of labels and time-series validation.
   Forward labels belong entirely inside each training fold. Characteristic
   estimators must themselves be causal; this class cannot enforce that for
   arbitrary user-supplied estimators.
4. Expand integration tests for additional metadata-dependent priors. Dated
   metadata now reaches the exact rolling fit window. Rolling causal
   covariance priors are available in sequential evaluation. Explicit covariance
   and pre-trade holdings now support risk/turnover limits. Sequential evaluation
   uses existing Portfolio drift, explicit initial holdings and configurable
   observation-count rebalance intervals. Label constraints alone remain insufficient.
5. Expand the historical validation with point-in-time data and independently
   chosen evaluation periods. A reproducible 2019-2021 example now compares a
   learned policy with a constrained equal-weight baseline; it is not evidence
   of investment performance.

Deferred
--------

Unpriced delistings and trading halts,
joint whole-portfolio regression, cost-aware supervised training, cash borrowing, calendar-based
rebalance schedules,
and full Chevalier--Coqueret--Raffinot methodology.

References
----------

* https://github.com/skfolio/skfolio/issues/262
* https://github.com/skfolio/skfolio/issues/262#issuecomment-5854891297
* https://github.com/skfolio/skfolio/issues/262#issuecomment-5856846048
* https://github.com/skfolio/skfolio/pull/341

Validation example
------------------

Run ``PYTHONPATH=src python integration/issue_262/example.py`` from this worktree.
The example fits a random forest to subperiod portfolios and checks bounds on
subsequent predictions. It illustrates API behavior, not investment performance.

Risk and turnover implementation
-------------------------------

Four incremental commits add validated external inputs, variance limits,
turnover limits, and sequential evaluation. Limits apply at prediction/evaluation,
not to static training diagnostics. Covariance must be causal and PSD; holdings
must be fully invested. Turnover is total absolute traded weight. Evaluation is
isolated from fitted state and rebuilds solver objects instead of copying native
solver caches. See the user guide for timing and remaining execution limitations.

Transaction-cost evaluation
---------------------------

Sequential evaluation now accepts scalar, per-asset or named transaction costs,
using Portfolio accounting with the same drifted-holdings anchor as turnover.
The explicit argument overrides portfolio_params; zero disables costs. Costs
reduce returns but do not alter Portfolio ending-weight drift or train the
regressor to avoid trading. The separate evaluate_execution method now provides explicitly funded cash/fee accounting.


Practical workflow follow-up
----------------------------

Prediction is now repeatable by default. Streaming callers explicitly opt into
atomic exposure-state updates. Sequential evaluation supports causal rolling
prior fits and rebalance intervals such as 5 or 21 observations, while advancing
characteristics and drifting holdings every observation. Costs are charged only
on actual trades. The historical example and validation report describe a tested
fixed-universe workflow and its remaining modeling limitations.


Cash and eligibility implementation
----------------------------------

``evaluate_execution`` returns an explicit PortfolioExecution ledger for long/short
portfolios, with a post-fee cash target and optional cash returns. It funds fees,
records actual trades, supports known-asset exits/reentry, and moves to all cash
when a separate eligibility schedule excludes everything. Known tradable exits
are assumed; suspended/unpriced positions cannot be silently liquidated. Unknown
asset names require explicit admission with ``expand_universe``. Metadata-dependent covariance priors
receive precisely aligned dated fit inputs.

The combined example ran 100 historical-price periods, 9 rebalances and five
all-cash periods. Exclusion dates are synthetic test scenarios. Final wealth was
0.901168 per initial unit and the sum of cost fractions was 0.003634. These numbers
validate execution and are not evidence of investment performance.

Run from this worktree::

    PYTHONPATH=src python integration/issue_262/cash_execution_example.py --output-dir /tmp/skfolio-cash-execution


Label optimizer metadata
------------------------

Dated ``optimizer_fit_params`` are now sliced independently for each forward
training-label window. A real InverseVolatility / TimeSeriesFactorModel routing
integration test verifies this path, in addition to factor-prior execution tests.
No held-out evaluation metadata enter label fitting.


Short positions and new assets
------------------------------

With ``long_only=False``, cash accounting preserves signed positions and charges
no borrowing fees. Existing proportional transaction costs remain optional.
Cash targets remain between zero and one; margin, collateral, borrow availability
and execution realism are outside scope.

``expanded = fitted.expand_universe(history, characteristics_history)`` returns
an independent policy on the expanded asset registry, without retraining its
regressor. Supply history through the current prediction date, including the old
assets and their unchanged past returns. Descriptors are rebuilt by replaying the
expanded history, so use stable feature definitions and enough warmup data. This
is not a general adaptation protocol for learned factor transformations.
Replace positional bounds and groups explicitly. Historical label diagnostics
retain their original universe. Missing pre-entry returns stay missing; risk
estimation still needs actual historical observations or explicit covariance.
