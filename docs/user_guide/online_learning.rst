.. _online_learning:

****************
Online Learning
****************

`skfolio` provides dedicated online utilities for estimators that support
`partial_fit`. By updating a single stateful estimator incrementally rather than
refitting from scratch at every split, online evaluation is significantly faster than
standard cross-validation. These utilities cover stateful walk-forward evaluation,
online covariance forecast diagnostics, and online hyper-parameter tuning.

Examples of supported estimators include
:class:`~skfolio.moments.EWMu`, :class:`~skfolio.moments.EWCovariance`,
:class:`~skfolio.moments.RegimeAdjustedEWCovariance`,
:class:`~skfolio.prior.EmpiricalPrior` and portfolio
optimizers such as :class:`~skfolio.optimization.MeanRisk` when they embed
incremental moment estimators through a prior estimator.

Online learning is also where native NaN-aware estimators are especially useful:
they can update from available observations while preserving estimator state. Pipeline
based pre-selection and imputation are not currently available in `skfolio` online
learning workflows. See :ref:`Missing Data and Changing Universes <missing_data>`
for details.


How Online Evaluation Works
***************************

The online utilities all follow the same stateful evaluation pattern:

1. Clone the estimator once, starting from a clean unfitted state.
2. Initialize it on the first `warmup_size` observations with `partial_fit`.
3. Evaluate on the next test window out-of-sample, with optional purging
   between the data seen by the estimator and the test window.
4. Update the same estimator with the newly observed data.
5. Repeat until the end of the sample.

This differs from standard cross-validation, where each split fits an
independent estimator clone. In the online setting, the estimator state is
carried forward through time.


.. _online_failure_handling:

Updates and Failure Handling
****************************

For portfolio optimizers, `partial_fit` uses new observations to update the
prior and other estimators, then computes portfolio weights. Each call builds
on the estimates from previous calls. Use `fit` to start over.

Solver Failures
===============

A solver failure can occur when portfolio constraints are infeasible. With
`fallback=None` (the default), `raise_on_failure` controls the behavior:

* `raise_on_failure=True` (the default) raises the error. Restart the model
  before calling `partial_fit` or `predict` again, as shown below.
* `raise_on_failure=False` emits a warning and sets `weights_` to `None`.
  `predict` returns a :class:`~skfolio.portfolio.FailedPortfolio`. The same
  model can continue learning through subsequent `partial_fit` calls.

With `fallback="previous_weights"`, the optimizer first tries to reuse the
holdings supplied in `previous_weights`. If this succeeds, the model can
continue with either value of `raise_on_failure`. If the fallback fails,
`raise_on_failure` determines the outcome as described above.

With `raise_on_failure=False` or a successful `fallback="previous_weights"`,
you can continue updating the same model after a solver failure. The prior has
already used that call's data, so pass only new observations. For example,
after a failed rebalance using Monday's returns, the next call should receive
Tuesday's returns.

With `raise_on_failure=True`, the solver error is raised after the prior has
learned from the batch when `fallback=None` or the previous-weights fallback
fails. `weights_` can still hold the previous allocation. Restart the model
before calling `partial_fit` or `predict` again by creating a fresh estimator
and training it on the desired history:

.. code-block:: python

    from sklearn.base import clone

    model = clone(model)
    model.partial_fit(X_history)

`X_history` contains the observations for the new run. Supply any required
targets or metadata as usual. Calling `fit(X_history, ...)` also starts fresh.

Online fitting supports `fallback=None` and `fallback="previous_weights"`.
Estimator fallbacks are available with batch `fit`, where each fallback trains
on the supplied history.

Errors other than solver failures always raise, regardless of
`raise_on_failure` and `fallback`. They can leave the model partially updated.
Restart it as described above.

Non-Predictor Estimators Versus Portfolio Optimizers
****************************************************

The online API distinguishes between non-predictor estimators and portfolio
optimizers.

* **Non-predictor estimators** such as covariance, expected-return, and prior
  estimators do not implement `predict`. Their scores are computed from the
  fitted estimator and the current test window using callables such as
  `scorer(estimator, X_test)`. When using
  :func:`~skfolio.metrics.make_scorer`, the appropriate form is
  `make_scorer(..., response_method=None)`.
* **Portfolio optimization estimators** such as
  :class:`~skfolio.optimization.MeanRisk` are evaluated by collecting the
  out-of-sample predictions into a
  :class:`~skfolio.portfolio.MultiPeriodPortfolio`. Measures are then computed
  on that aggregate portfolio. In that case, scoring uses
  :class:`~skfolio.measures.BaseMeasure` enums directly rather than
  :func:`~skfolio.metrics.make_scorer`.

Accordingly, :func:`~skfolio.model_selection.online_predict` is restricted to
portfolio optimizers, while :func:`~skfolio.model_selection.online_score` and
:class:`~skfolio.model_selection.OnlineGridSearch` /
:class:`~skfolio.model_selection.OnlineRandomizedSearch` accept both categories.


Online Versus Standard Cross-Validation
****************************************

The standard :func:`~skfolio.model_selection.cross_val_predict` and its online
counterpart :func:`~skfolio.model_selection.online_predict` are both designed
exclusively for **portfolio optimization** estimators. The key differences are:

* **Fitting strategy**: standard cross-validation clones and refits the estimator from
  scratch at every fold, while online evaluation maintains a single stateful estimator
  updated incrementally via `partial_fit`, which is significantly faster.
* **Scoring methodology**: standard cross-validation scores each test fold independently
  and averages the results, which can be unreliable when test folds are short (e.g. the
  Sharpe ratio is undefined on a single observation). Online evaluation instead collects
  all out-of-sample predictions into a single
  :class:`~skfolio.portfolio.MultiPeriodPortfolio` and computes the metric on the full
  out-of-sample path, which is generally preferred for short rebalancing horizons.

:func:`~skfolio.model_selection.online_score` extends online evaluation to both
portfolio optimizers and non-predictor estimators (covariance, expected-return, and
prior estimators). For non-predictor estimators, scores are computed per test window
and averaged by default.

Because the scoring methodology differs for portfolio optimizers, the online utilities
are complementary to the existing cross-validation tools rather than replacements.

Online Covariance Forecast Evaluation
*************************************

:func:`~skfolio.model_selection.online_covariance_forecast_evaluation`
evaluates the quality of covariance forecasts out-of-sample. It is intended for
covariance estimators rather than portfolio optimizers, which should instead be
evaluated with :func:`~skfolio.model_selection.online_predict` or
:func:`~skfolio.model_selection.online_score`.

At each step, the covariance forecast produced after `partial_fit` is compared
to the realized returns over the next test window. The resulting
:class:`~skfolio.model_selection.CovarianceForecastEvaluation` provides
diagnostics such as:

* Mahalanobis calibration ratio for the full covariance structure across all
  eigenvalue directions.
* Diagonal calibration ratio for asset-level variance calibration.
* Portfolio standardized returns and the associated bias statistic for
  calibration along one portfolio direction by default, or multiple portfolio
  directions when explicit test portfolios are provided.
* Portfolio QLIKE for portfolio variance forecast quality along one or more
  portfolio directions.

When `portfolio_weights=None`, the portfolio diagnostics use a single dynamic
inverse-volatility portfolio direction by default. Passing explicit portfolio
weights extends the evaluation to multiple selected traded directions.

See the example
:ref:`sphx_glr_auto_examples_online_learning_plot_1_online_covariance_forecast_evaluation.py`
for the complete workflow.


Online Hyper-Parameter Tuning
*****************************

:class:`~skfolio.model_selection.OnlineGridSearch` and
:class:`~skfolio.model_selection.OnlineRandomizedSearch` extend the online
workflow to hyper-parameter selection.

Conceptually, this is the online counterpart of combining
:class:`~sklearn.model_selection.GridSearchCV` or
:class:`~sklearn.model_selection.RandomizedSearchCV` with
:class:`~skfolio.model_selection.WalkForward` using `expand_train=True`.
The key difference is that online search updates each candidate
incrementally via `partial_fit` along one sequential path instead of
refitting it from scratch at every split.

Each candidate parameter configuration is evaluated on one full online
walk-forward path. When `refit=True`, `best_estimator_` exposes the selected
fitted candidate without an additional fit after model selection because it
has already been updated through the full sample during evaluation.

For non-predictor estimators, online tuning typically uses callable scorers
such as QLIKE or calibration losses, typically wrapped with
:func:`~skfolio.metrics.make_scorer` using `response_method=None`. For
multi-metric searches, `refit` should be set explicitly to the name of the
metric used to select the best candidate.

See the example
:ref:`sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py`
for covariance tuning with both
:class:`~skfolio.model_selection.OnlineGridSearch` and
:class:`~skfolio.model_selection.OnlineRandomizedSearch`.


Online Evaluation of Portfolio Optimization
*******************************************

For portfolio optimizers, the main entry points are
:func:`~skfolio.model_selection.online_predict` and
:func:`~skfolio.model_selection.online_score`.

* :func:`~skfolio.model_selection.online_predict` returns a
  :class:`~skfolio.portfolio.MultiPeriodPortfolio` built from the sequence of
  out-of-sample portfolio predictions.
* :func:`~skfolio.model_selection.online_score` returns a scalar measure, or a
  dict of measures, computed on the aggregate online evaluation.

This is useful when a portfolio estimator embeds incremental moment estimators such as
:class:`~skfolio.moments.EWMu` and
:class:`~skfolio.moments.RegimeAdjustedEWCovariance`.

During online portfolio evaluation, `online_predict` records a suppressed solver
failure as a :class:`~skfolio.portfolio.FailedPortfolio` and continues with the
next window. Raised errors interrupt evaluation. See
:ref:`Updates and Failure Handling <online_failure_handling>` for data
consumption, fallback, and restart rules.

Pass `portfolio_params={"weight_drift": True, "compounded": True}` to
`online_predict` to evaluate the path with drifted weights and compounded returns. The
`portfolio_params` of `online_predict` follow the same rules as those of
:func:`~skfolio.model_selection.cross_val_predict`: the parameters shared by
:class:`~skfolio.portfolio.Portfolio` and
:class:`~skfolio.portfolio.MultiPeriodPortfolio` are applied to the returned
`MultiPeriodPortfolio` and to each `Portfolio` it contains, take precedence over the
optimizer's `portfolio_params` and are inherited from it when omitted. `weight_drift`
applies to each `Portfolio` of the path. See :ref:`cross_validation` for the complete
rules.

When the estimator needs previous weights, the `ending_weights` of the last successful
portfolio become the `previous_weights` of the next rebalancing. They equal its target
weights with `weight_drift=False` and its weights after the final observation with
`weight_drift=True`.

`online_score`, `OnlineGridSearch` and `OnlineRandomizedSearch` accept the same
`portfolio_params` when evaluating a portfolio optimizer. Because they score the
resulting `MultiPeriodPortfolio`, these parameters can change the scores and the
ranking of the parameter sets. When refitting is enabled, `weight_drift` is retained in
`best_estimator_` because prediction requires it. The other parameters are not.

To use those previous holdings instead of producing a failed rebalance after a
solver failure, configure `fallback="previous_weights"`.

See the example
:ref:`sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py`
for an end-to-end online evaluation of
:class:`~skfolio.optimization.MeanRisk`.

Sequential portfolio policies
-----------------------------

:class:`~skfolio.optimization.ExponentiatedGradient` learns target weights directly
from returns, without an expected-return or covariance estimator. ``fit`` resets
its state; ``partial_fit`` continues and processes a block one observation at a
time. After observing return at time ``t``, ``weights_`` is the target for the
next period. ``initial_weights_`` records the feasible allocation before learning.

.. code-block:: python

    from skfolio.model_selection import online_predict
    from skfolio.optimization import ExponentiatedGradient

    model = ExponentiatedGradient(
        learning_rate=0.05,
        max_weights=0.4,  # Requires at least three assets.
        portfolio_params={"weight_drift": True},
    )
    portfolio = online_predict(model, X, warmup_size=252, test_size=1)

The estimator uses entropy mirror descent with a fixed rate or a pure callable
schedule indexed by the number of observations processed, including warmup.
Weight bounds are enforced with KL geometry and may be scalars, arrays or
asset-name dictionaries. Bounds are fixed for a stream; call ``fit`` after
changing them. The initial reference allocation is also projected into the bounds.

Algorithm target weights are distinct from actual holdings: portfolio evaluation
handles drift and passes held weights through ``previous_weights``. These holdings
do not replace the algorithm's previous target. No trades are simulated during
warmup. Online evaluation requires exactly one observation per test window;
calendar windows that contain several observations are rejected too.

This first implementation supports a fixed asset universe and finite simple
returns strictly greater than -1. It has no prior estimator, risk constraints,
turnover limit or fallback recovery. A failed numerical update leaves earlier
successful observations in the block committed. Inherited ``predict`` holds the
last target, and ``fit_predict`` is not a causal backtest: use ``online_predict``.

The numerical reference weights are floored at ``1e-16`` before taking logarithms,
matching the entropy-map convention used in Carlo Nicolini's online portfolio
implementation. The mode is mirror descent, rather than his default FTRL mode;
these need not agree for nonuniform initial allocations or varying learning rates.
