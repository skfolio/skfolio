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

See the example
:ref:`sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py`
for an end-to-end online evaluation of
:class:`~skfolio.optimization.MeanRisk`.

