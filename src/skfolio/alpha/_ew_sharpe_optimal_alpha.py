"""Exponentially weighted least-squares Sharpe-optimal alpha estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn.utils.metadata_routing as skm

import skfolio.typing as skt
from skfolio._constants import (
    _DESCRIPTOR_SCORES,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
)
from skfolio.alpha import ForecastUnit
from skfolio.alpha._base import BaseAlpha, BaseAlphaDescriptorComposition
from skfolio.containers import AssetPanel
from skfolio.descriptor import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import _forward_mean_return, safe_divide
from skfolio.utils.tools import (
    _validate_bool,
    _validate_non_negative_real,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "alpha_"


class EWSharpeOptimalAlpha(BaseAlphaDescriptorComposition, BaseAlpha):
    r"""Exponentially weighted least-squares Sharpe-optimal alpha estimator.

    This estimator aggregates multiple cross-sectional signals from descriptors into a
    single alpha forecast by estimating their joint contribution to forward
    idiosyncratic returns. Coefficients are estimated with exponentially weighted
    least squares.

    The estimator supports two forecast units. With the default
    `forecast_unit=ForecastUnit.IDIO_RETURN`, descriptors are fitted directly to forward
    idiosyncratic returns. When descriptor scores linearly forecast idiosyncratic
    returns and residual noise is proportional to idiosyncratic variance, the learned
    signal blend is Sharpe-optimal in idiosyncratic return space for an unconstrained
    long-short strategy.

    With `forecast_unit=ForecastUnit.IDIO_SHARPE`, descriptors are fitted to forward
    idiosyncratic return divided by idiosyncratic volatility, with unit regression
    weights. Dividing the target by :math:`\sigma_i` transforms the inverse-variance
    GLS objective in return units into OLS in idiosyncratic-Sharpe units.

    Signals are first transformed into cross-sectional scores (e.g., z-scores, ranks),
    then optionally neutralized against factors and re-transformed into cross-sectional
    scores and finally combined linearly:

    .. math::

        \alpha_i = \sum_{k=1}^{K} \beta_k \, S_{k,i}

    where :math:`S_{k,i}` denotes the cross-sectional score of signal :math:`k` for
    asset :math:`i` and :math:`\beta_k` is the estimated signal coefficient.

    By default, coefficients map descriptor scores directly into expected return units.
    With `forecast_unit=ForecastUnit.IDIO_SHARPE`, coefficients map descriptor scores into
    idiosyncratic-Sharpe units and the final forecast is multiplied by current
    idiosyncratic volatility so `alpha_` remains in expected return units.

    This generalizes IC-based signal weighting by:

    - accounting for cross-signal correlations (multivariate estimation)
    - incorporating asset-specific risk, either through inverse idiosyncratic variance
      weights or through volatility-scaled targets
    - producing an alpha forecast in expected return units, which is required whenever
      the optimizer is trading off alpha against real costs and constraints (e.g.
      transaction costs, market impact, borrow costs, turnover constraints).

    For an individual signal with constant idiosyncratic variance, the estimator reduces
    to a scaled IC-like weighting.

    The estimator uses the following regression target:

    .. math::

        y_t =
        \begin{cases}
        \epsilon_t, & \text{if } \texttt{forecast\_unit=ForecastUnit.IDIO\_RETURN} \\
        \epsilon_t / \sigma_t, &
        \text{if } \texttt{forecast\_unit=ForecastUnit.IDIO\_SHARPE}
        \end{cases}

    and regression weights:

    .. math::

        W_t =
        \begin{cases}
        \operatorname{diag}(1 / \sigma_{t,i}^2), &
        \text{if } \texttt{forecast\_unit=ForecastUnit.IDIO\_RETURN} \\
        I, & \text{if } \texttt{forecast\_unit=ForecastUnit.IDIO\_SHARPE}
        \end{cases}

    where:

    - :math:`\epsilon_{t,i}` is the forward mean idiosyncratic return over the chosen
      horizon
    - :math:`S_{t,i} \in \mathbb{R}^K` is the vector of cross-sectional scores
    - :math:`\sigma_{t,i}^2` is the forecast idiosyncratic variance

    If `normalize_weights=True`, the positive diagonal entries of :math:`W_t` are
    divided by their cross-sectional average before computing the normal-equation
    statistics.

    With `forecast_unit=ForecastUnit.IDIO_SHARPE`, the return model is instead:

    .. math::

        \epsilon_{t,i} = \sigma_{t,i} S_{t,i}^\top \beta + \eta_{t,i},
        \quad \operatorname{Var}(\eta_{t,i}) \propto \sigma_{t,i}^2

    The corresponding inverse-variance GLS objective is:

    .. math::

        \beta_t = \arg\min_\beta \sum_i
            \frac{(\epsilon_{t,i} - \sigma_{t,i} S_{t,i}^\top \beta)^2}
            {\sigma_{t,i}^2}

    which is equivalent to ordinary least squares on the volatility-scaled target
    :math:`\epsilon_{t,i}/\sigma_{t,i}`:

    .. math::

        \beta_t = \arg\min_\beta \sum_i
          \left(\frac{\epsilon_{t,i}}{\sigma_{t,i}}
          - S_{t,i}^\top \beta\right)^2

    The final forecast is converted back to expected return units:

    .. math::

        \alpha_i = \sigma_i S_i^\top \beta

    This is useful when signals are assumed to forecast idiosyncratic Sharpe rather than
    raw idiosyncratic return. For the same scaled signal forecast, higher-volatility
    assets receive larger return alpha because the forecast is converted back from
    idiosyncratic-Sharpe units to return units.

    To reduce estimation noise and turnover, the estimator maintains exponentially
    weighted least-squares statistics:

    .. math::

        A_t^{EW} = \lambda A_{t-1}^{EW} + (1 - \lambda) S_t^\top W_t S_t

    .. math::

        b_t^{EW} = \lambda b_{t-1}^{EW} + (1 - \lambda) S_t^\top W_t y_t,
        \quad \lambda = 2^{-1/\text{half-life}}

    Coefficients are obtained by ridge-stabilized normal equations:

    .. math::

        \beta_t = (A_t^{EW} + \rho_t I)^{-1} b_t^{EW}

    With `forecast_unit=ForecastUnit.IDIO_RETURN`, the final alpha forecast is:

    .. math::

        \alpha_i = S_i^\top \beta

    With `forecast_unit=ForecastUnit.IDIO_SHARPE`, the forecast is:

    .. math::

        \alpha_i = \sigma_i S_i^\top \beta

    No intercept is included to avoid absorbing cross-sectional means, making the
    resulting alpha suitable for long-short strategies.

    The estimator supports latest-alpha fitting with :meth:`fit` and :meth:`partial_fit`,
    and historical alpha forecasts with :meth:`fit_transform` and
    :meth:`partial_fit_transform`. Historical rows are computed as-of each observation:
    for horizon :math:`h` and signal lag :math:`\ell`, alpha at observation :math:`t`
    uses coefficient updates from signal observations up to :math:`t - \ell - h + 1`.

    Parameters
    ----------
    descriptors : list of (name, estimator) tuples
        List of descriptors that compute signals from characteristics. Each tuple
        contains a string name and a descriptor estimator. Multiple descriptors are
        aggregated into a single alpha using multivariate regression. The descriptors
        are evaluated in parallel if `n_jobs > 1`.

    half_life : float, default=20
        Half-life of the exponential weights in number of observations.

        * Larger half-life: More stable alpha estimates, slower adaptation
        * Smaller half-life: More responsive estimates, faster adaptation

    horizon : int, default=1
        Number of forward periods to average for the target idiosyncratic return.
        Must be >= 1. The target for observation :math:`t` is
        `mean(idio_returns[t+signal_lag : t+signal_lag+horizon])`.

        * `horizon=1`: Predicts one-period idiosyncratic return starting after `signal_lag`
        * `horizon>1`: Predicts the mean of `horizon` idiosyncratic returns starting
          after `signal_lag`.

    signal_lag : int, default=1
        Number of periods between the signal observation and the first return in the
        target window. Must be >= 1. Under skfolio's as-of time-indexing convention,
        `signal_lag=0` would use information observed at the end of :math:`t` to predict
        return at :math:`t`, which is look-ahead. Values larger than 1 can model
        conservative data availability or execution delays.

    neutralize_against : list of str, optional
        Factor names or families to neutralize scores against. If provided, scores are
        orthogonalized with respect to the specified factor exposures before regression.

    outlier_transformer : BaseCSTransformer or "passthrough", optional
         Cross-sectional transformer for descriptor outlier handling. If None, defaults
         to `CSWinsorizer()`. Use "passthrough" to skip.

    scoring_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for descriptor scoring applied after outlier
        handling. If None, defaults to `CSStandardScaler()`. Use "passthrough" to skip.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations. If provided, outlier and scoring transformations are applied
        within each group separately.

    forecast_unit : ForecastUnit, default=ForecastUnit.IDIO_RETURN
        Unit of the intermediate forecast learned from descriptor scores. With
        `ForecastUnit.IDIO_RETURN`, the target is the forward mean idiosyncratic return
        and WLS weights are inverse idiosyncratic variance. With
        `ForecastUnit.IDIO_SHARPE`, the target is divided by forecast idiosyncratic
        volatility and fitted with unit weights. The resulting idiosyncratic-Sharpe
        forecast is converted back to return units by multiplying by current
        idiosyncratic volatility.

    forecast_scale : float, default=1.0
        Multiplicative scale applied to the final alpha forecast after the learned
        coefficients have been converted to expected return units. This controls alpha
        strength without changing the EWLS coefficient estimates.

    normalize_weights : bool, default=True
        If `True`, regression weights are normalized within each observation to have an
        average of one across valid assets. This removes changes in aggregate weight
        caused by the scale of idiosyncratic variances, while preserving the greater
        statistical weight of observations with more valid assets. In practice, this
        prevents calm, low-volatility regimes from mechanically dominating the EWLS
        statistics just because inverse-variance weights are larger in those regimes.
        Set `normalize_weights=False` for the unnormalized GLS estimator (which is
        BLUE under the usual assumptions).

    ridge_scale : float, default=1e-6
        Relative ridge penalty applied to the exponentially weighted normal matrix.

    n_jobs : int, default=1
        Number of parallel jobs for descriptor computation. Use `-1` for all available
        cores.

    Attributes
    ----------
    alpha_ : ndarray of shape (n_assets,) or None
        Estimated alpha (expected idiosyncratic return) for each asset. This is the
        aggregated prediction from all signals. Returns `None` during warmup phase
        (fewer than `signal_lag + horizon` observations).

    coef_ : ndarray of shape (n_descriptors,)
        Estimated descriptor coefficients.

    descriptors_ : list of BaseDescriptor
        Fitted descriptor estimators.

    named_descriptors_ : dict of {str: BaseDescriptor}
        Dictionary mapping descriptor names to fitted estimators.

    outlier_transformer_ : BaseCSTransformer or str
        The fitted outlier transformer.

    scoring_transformer_ : BaseCSTransformer or str
        The fitted scoring transformer.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.alpha import EWSharpeOptimalAlpha, ForecastUnit
    >>> from skfolio.descriptor import EWMomentum, BookToPrice, Reversal, Passthrough
    >>>
    >>> X = make_synthetic_characteristics()
    >>> rng = np.random.default_rng(0)
    >>>
    >>> # Alpha models regress forward idiosyncratic returns. In production these
    >>> # come from a fitted CharacteristicsFactorModel.
    >>> idio_returns = rng.standard_normal((X.n_observations, X.n_assets))
    >>> idio_returns[~X.active_mask] = np.nan
    >>> X["idio_returns"] = idio_returns
    >>>
    >>> # Required when forecast_unit=ForecastUnit.IDIO_SHARPE to scale targets and alphas.
    >>> idio_variances = rng.uniform(0.01, 0.05, (X.n_observations, X.n_assets))
    >>> idio_variances[~X.active_mask] = np.nan
    >>> X["idio_variances"] = idio_variances
    >>>
    >>> # Required when neutralize_against is set. In production these are factor
    >>> # exposures from the characteristics factor model.
    >>> exposures = rng.standard_normal((X.n_observations, X.n_assets, 3))
    >>> exposures[~X.active_mask] = np.nan
    >>> X.add_3d_field(
    ...     "exposures",
    ...     exposures,
    ...     third_axis_name="factors",
    ...     third_axis_labels=["market", "beta", "size"],
    ... )
    >>>
    >>> alpha_model = EWSharpeOptimalAlpha(
    ...     descriptors=[
    ...         ("momentum", EWMomentum()),
    ...         ("book_to_price", BookToPrice()),
    ...         ("reversal", Reversal()),
    ...         ("eps_ntm", Passthrough("eps_ntm")),
    ...     ],
    ...     horizon=5,      # one-week forward idiosyncratic return
    ...     half_life=21,    # one-month EWLS half-life
    ...     neutralize_against=["market", "beta", "size"],
    ...     forecast_unit=ForecastUnit.IDIO_SHARPE,
    ... )
    >>>
    >>> # Latest alpha forecast for the current rebalance.
    >>> alpha_model.fit(X)
    >>> print(alpha_model.alpha_)
    >>>
    >>> # Online learning with partial_fit
    >>> alpha_model.partial_fit(X[-5:])
    >>> print(alpha_model.alpha_)
    >>>
    >>> # Historical as-of alpha forecasts with fit_transform
    >>> alphas = alpha_model.fit_transform(X)

    Notes
    -----
    The Information Ratio (IR) of a strategy is approximately [1]_:

    .. math::

        \text{IR} \approx \text{IC} \times \sqrt{\text{Breadth}}

    This estimator generalizes single-signal IC weighting by estimating multivariate,
    risk-weighted signal payoffs. The exponential weighting and ridge stabilization
    reduce turnover and estimation noise in the coefficients.

    References
    ----------
    .. [1] "Active Portfolio Management: A Quantitative Approach for Producing Superior
       Returns and Controlling Risk", McGraw-Hill, Grinold & Kahn (1999).
    """

    def __init__(
        self,
        *,
        descriptors: list[tuple[str, BaseDescriptor]],
        half_life: float = 20,
        ridge_scale: float = 1e-6,
        horizon: int = 1,
        signal_lag: int = 1,
        neutralize_against: list[str] | None = None,
        outlier_transformer: skt.CSTransformer = None,
        scoring_transformer: skt.CSTransformer = None,
        transform_by_group: str | None = None,
        forecast_unit: ForecastUnit = ForecastUnit.IDIO_RETURN,
        forecast_scale: float = 1.0,
        normalize_weights: bool = True,
        n_jobs: int = 1,
    ):
        self.descriptors = descriptors
        self.half_life = half_life
        self.ridge_scale = ridge_scale
        self.horizon = horizon
        self.signal_lag = signal_lag
        self.neutralize_against = neutralize_against
        self.outlier_transformer = outlier_transformer
        self.scoring_transformer = scoring_transformer
        self.transform_by_group = transform_by_group
        self.forecast_unit = forecast_unit
        self.forecast_scale = forecast_scale
        self.normalize_weights = normalize_weights
        self.n_jobs = n_jobs

    def fit(self, X: AssetPanel, y=None, **fit_params) -> EWSharpeOptimalAlpha:
        """Fit the alpha model.

        Resets all internal state, processes the provided panel and stores the latest
        alpha forecast in `alpha_`.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "idio_returns", "idio_variances", descriptor
            fields and optionally "exposures" for score neutralization.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors through metadata routing.

        Returns
        -------
        self : EWSharpeOptimalAlpha
            Fitted estimator.
        """
        self._reset()
        self._fit(X, y, method="fit", **fit_params)
        return self

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Fit the alpha model and return historical alpha forecasts.

        The returned alpha at observation :math:`t` only uses coefficient updates whose
        forward-return target is observable by :math:`t`. Warmup rows are `NaN`.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "idio_returns", "idio_variances", descriptor
            fields and optionally "exposures" for score neutralization.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors through metadata routing.

        Returns
        -------
        alphas : ndarray of shape (n_observations, n_assets)
            Historical alpha forecasts for the input panel.
        """
        self._reset()
        return self._fit(X, y, method="fit", transform=True, **fit_params)

    def partial_fit(self, X: AssetPanel, y=None, **fit_params) -> EWSharpeOptimalAlpha:
        """Incrementally fit the alpha model with new observations.

        This method supports streaming/online updates. It maintains internal buffers to
        compute forward returns across partial_fit calls.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "idio_returns", "idio_variances", descriptor
            fields and optionally "exposures" for score neutralization.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors through metadata routing.

        Returns
        -------
        self : EWSharpeOptimalAlpha
            Fitted estimator.
        """
        self._fit(X, y, method="partial_fit", **fit_params)
        return self

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Incrementally fit the alpha model and return new historical alpha forecasts.

        Only rows corresponding to the newly supplied observations are returned.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "idio_returns", "idio_variances", descriptor
            fields and optionally "exposures" for score neutralization.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors through metadata routing.

        Returns
        -------
        alphas : ndarray of shape (n_observations, n_assets)
            Historical alpha forecasts for the new observations.
        """
        return self._fit(X, y, method="partial_fit", transform=True, **fit_params)

    def _fit(
        self,
        X: AssetPanel,
        y=None,
        *,
        method: str,
        transform: bool = False,
        **fit_params,
    ) -> FloatArray | None:
        """Fit the model state and optionally return historical alpha forecasts."""
        routed_params = skm.process_routing(self, method, **fit_params)

        first_call = not hasattr(self, _FITTED_ATTR)

        required_fields = [_IDIO_RETURNS, _IDIO_VARIANCES]
        if self.neutralize_against is not None:
            required_fields.append(_EXPOSURES)
        if self.transform_by_group is not None:
            required_fields.append(self.transform_by_group)

        validate_asset_panel(
            self,
            X,
            required_fields=required_fields,
            strictly_positive_or_nan=[_IDIO_VARIANCES],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        scores = self._compute_scores(X=X, method=method, routed_params=routed_params)

        X = self._make_training_panel(
            X, scores=scores, fields=[_IDIO_RETURNS, _IDIO_VARIANCES]
        )
        X = self._prepend_buffer(X)

        n_combined_obs = X.n_observations
        n_buffered_obs = n_combined_obs - n_observations
        n_trainable_obs = max(0, n_combined_obs - self._target_gap)
        historical_alphas = (
            np.full((n_observations, n_assets), np.nan, dtype=float)
            if transform
            else None
        )

        if n_trainable_obs == 0:
            self.alpha_ = None
            self._update_buffers(X)
            return historical_alphas

        forward_returns = _forward_mean_return(
            X[_IDIO_RETURNS], horizon=self.horizon, lag=self.signal_lag
        )

        historical_coefficients = self._update_ewls(
            X[:n_trainable_obs],
            forward_returns=forward_returns[:n_trainable_obs],
            return_historical=transform,
        )

        if transform:
            for forecast_idx in range(n_observations):
                combined_obs_idx = n_buffered_obs + forecast_idx
                coef_idx = combined_obs_idx - self._target_gap
                if 0 <= coef_idx < n_trainable_obs:
                    historical_alphas[forecast_idx] = self._compute_alpha(
                        scores=X[_DESCRIPTOR_SCORES][combined_obs_idx],
                        coefficient=historical_coefficients[coef_idx],
                        idio_variances=X[_IDIO_VARIANCES][combined_obs_idx],
                    )

        if self._n_valid_regression_obs == 0:
            self.alpha_ = None
        else:
            self.alpha_ = self._compute_alpha(
                scores=scores[-1],
                coefficient=self.coef_,
                idio_variances=X[_IDIO_VARIANCES][-1],
            )

        self._update_buffers(X)
        return historical_alphas

    def _update_ewls(
        self, X: AssetPanel, forward_returns: FloatArray, return_historical: bool
    ) -> FloatArray | None:
        """Update EWLS normal equations and optionally return coefficient history."""
        idio_variances = X[_IDIO_VARIANCES]
        scores = X[_DESCRIPTOR_SCORES]
        n_observations = X.n_observations
        valid_var = np.isfinite(idio_variances)
        valid_target = np.isfinite(forward_returns) & valid_var & X.estimation_mask

        weights = X.estimation_mask.astype(float)
        if self.forecast_unit is ForecastUnit.IDIO_SHARPE:
            target = safe_divide(
                forward_returns, np.sqrt(idio_variances), fill_value=np.nan
            )
            weights = np.where(valid_target, weights, 0.0)
        else:
            target = forward_returns
            inv_var = safe_divide(1.0, idio_variances, fill_value=0.0)
            weights = np.where(valid_target, weights * inv_var, 0.0)

        historical_coefficients = (
            np.full((n_observations, len(self.descriptors_)), np.nan, dtype=float)
            if return_historical
            else None
        )

        valid = np.all(np.isfinite(scores), axis=2) & valid_target

        for t in range(n_observations):
            valid_t = valid[t]
            if not np.any(valid_t):
                coefficient = self.coef_ if self._n_valid_regression_obs > 0 else None
            else:
                coefficient = self._process_ewls_observation(
                    scores=scores[t, valid_t],
                    target=target[t, valid_t],
                    weights=weights[t, valid_t],
                )
            if return_historical and coefficient is not None:
                historical_coefficients[t] = coefficient

        return historical_coefficients

    def _process_ewls_observation(
        self, scores: FloatArray, target: FloatArray, weights: FloatArray
    ) -> FloatArray:
        """Update EWLS state with one cross-sectional regression observation."""
        if self.normalize_weights:
            weight_scale = float(np.mean(weights))
            if np.isfinite(weight_scale) and weight_scale > 0:
                weights = weights / weight_scale

        sqrt_weights = np.sqrt(weights)

        weighted_scores = scores * sqrt_weights[:, None]
        weighted_target = target * sqrt_weights

        obs_normal_matrix = weighted_scores.T @ weighted_scores
        obs_target_cross_product = weighted_scores.T @ weighted_target

        self._ew_normal_matrix *= self._decay
        self._ew_normal_matrix += (1.0 - self._decay) * obs_normal_matrix
        self._ew_target_cross_product *= self._decay
        self._ew_target_cross_product += (1.0 - self._decay) * obs_target_cross_product
        self._n_valid_regression_obs += 1
        self.coef_ = self._solve_ewls_coefficients()
        return self.coef_

    def _solve_ewls_coefficients(self) -> FloatArray:
        """Solve the ridge-stabilized EWLS normal equations."""
        regularized_normal_matrix = self._ew_normal_matrix.copy()
        if self.ridge_scale > 0:
            diagonal_scale = float(np.mean(np.abs(np.diag(regularized_normal_matrix))))
            diagonal_scale = max(diagonal_scale, np.finfo(float).eps)
            ridge = self.ridge_scale * diagonal_scale
            regularized_normal_matrix[
                np.diag_indices_from(regularized_normal_matrix)
            ] += ridge

        try:
            return np.linalg.solve(
                regularized_normal_matrix, self._ew_target_cross_product
            )
        except np.linalg.LinAlgError:
            return np.linalg.pinv(regularized_normal_matrix) @ (
                self._ew_target_cross_product
            )

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        self._validate_common_params()
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_real(self.forecast_scale, "forecast_scale")
        _validate_non_negative_real(self.ridge_scale, "ridge_scale")
        _validate_bool(self.normalize_weights, "normalize_weights")

    def _initialize(self) -> None:
        """Initialize internal state on first call."""
        self._initialize_common_state()

        n_descriptors = len(self.descriptors_)
        self._decay = half_life_to_decay_factor(self.half_life)
        self._ew_normal_matrix = np.zeros((n_descriptors, n_descriptors))
        self._ew_target_cross_product = np.zeros(n_descriptors)
        self._n_valid_regression_obs = 0
        self.coef_ = np.full(n_descriptors, np.nan)

        self._initialize_buffers()

    def _compute_alpha(
        self,
        scores: FloatArray,
        coefficient: FloatArray,
        idio_variances: FloatArray,
    ) -> FloatArray:
        """Compute one cross-sectional alpha forecast from descriptor scores."""
        alpha = scores @ coefficient
        if self.forecast_unit is ForecastUnit.IDIO_SHARPE:
            alpha = alpha * np.sqrt(idio_variances)
        return self.forecast_scale * alpha
