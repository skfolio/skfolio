Issue #262: supervised-policy starting point
===========================================

Branch: ``supervised-portfolio-policy`` (local only).

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
4. Add metadata routing and automatic causal prior updates. Explicit covariance
   and pre-trade holdings now support risk/turnover limits. Sequential evaluation
   uses existing Portfolio drift, explicit initial holdings and one subsequent
   return observation per decision. Label constraints alone remain insufficient.
5. Add a worked empirical comparison after those API choices. This starting
   implementation has correctness tests, not evidence of investment performance.

Deferred
--------

Dynamic universes, missing-return policies, joint whole-portfolio regression,
transaction costs, cash, automatic causal priors, repeated/idempotent inference for
stateful exposure estimators, larger evaluation windows,
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
