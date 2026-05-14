"""Exponentially weighted least-squares Sharpe-optimal alpha estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn as sk
import sklearn.utils.metadata_routing as skm
import sklearn.utils.parallel as skp

import skfolio.typing as skt
from skfolio._constants import (
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _PASSTHROUGH,
)
from skfolio.containers import AssetPanel
from skfolio.factor_model._utils import _forward_mean_return, _neutralize_scores
from skfolio.factor_model.alpha._base import BaseAlpha
from skfolio.factor_model.descriptor import BaseDescriptor
from skfolio.factor_model.descriptor._composition import DescriptorCompositionMixin
from skfolio.preprocessing import (
    BaseCSTransformer,
    CSStandardScaler,
    CSWinsorizer,
)
from skfolio.typing import FloatArray
from skfolio.utils.stats import safe_divide
from skfolio.utils.tools import (
    call_asset_panel_transform,
    check_estimator,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "alpha_"


class EWSharpeOptimalAlpha(BaseAlpha, DescriptorCompositionMixin):
    r"""Exponentially weighted least-squares Sharpe-optimal alpha estimator.

    This estimator aggregates multiple cross-sectional signals from descriptors into a
    single alpha forecast by estimating their joint contribution to forward
    idiosyncratic returns. Coefficients are estimated with exponentially weighted
    least squares.

    The estimator supports two target models. With the default
    `scale_target_by_idio_vol=False`, descriptors are fitted directly to forward
    idiosyncratic returns. When descriptor scores linearly forecast idiosyncratic
    returns and residual noise is proportional to idiosyncratic variance, the learned
    signal blend is Sharpe-optimal in idiosyncratic return space for an unconstrained
    long-short strategy.

    With `scale_target_by_idio_vol=True`, descriptors are fitted to forward
    idiosyncratic return divided by idiosyncratic volatility (i.e. idiosyncratic sharpe)
    with unit regression weights (dividing the target by :math:`\sigma_i` transforms the
    inverse-variance GLS objective in return units into OLS in idiosyncratic-sharpe
    units).

    Signals are first transformed into cross-sectional scores (e.g., z-scores, ranks),
    then optionally neutralized against factors and re-transformed into cross-sectional
    scores and finally combined linearly:

    .. math::

        \alpha_i = \sum_{k=1}^{K} \beta_k \, S_{k,i}

    where :math:`S_{k,i}` denotes the cross-sectional score of signal :math:`k` for
    asset :math:`i` and :math:`\beta_k` is the estimated signal coefficient.

    By default, coefficients map descriptor scores directly into expected return units.
    With `scale_target_by_idio_vol=True`, coefficients map descriptor scores into
    idiosyncratic-sharpe units and the final forecast is multiplied by current
    idiosyncratic volatility so `alpha_` remains in expected return units.

    This generalizes IC-based signal weighting by:

    - accounting for cross-signal correlations (multivariate estimation)
    - incorporating asset-specific risk, either through inverse idiosyncratic variance
      weights or through volatility-scaled targets
    - producing an alpha forecast in expected return units, which is required whenever
      the optimizer is trading off alpha against real costs and constraints (e.g.,
      transaction costs, market impact, borrow costs, turnover constraints).

    For a single signal with constant idiosyncratic variance, the estimator reduces
    to a scaled IC-like weighting.

    The estimator uses the following regression target:

    .. math::

        y_t =
        \begin{cases}
        u_t, & \text{if } \texttt{scale\_target\_by\_idio\_vol=False} \\
        u_t / \sigma_t, &
        \text{if } \texttt{scale\_target\_by\_idio\_vol=True}
        \end{cases}

    and regression weights:

    .. math::

        W_t =
        \begin{cases}
        \operatorname{diag}(1 / \sigma_{t,i}^2), &
        \text{if } \texttt{scale\_target\_by\_idio\_vol=False} \\
        I, & \text{if } \texttt{scale\_target\_by\_idio\_vol=True}
        \end{cases}

    where:

    - :math:`u_{t,i}` is the forward mean idiosyncratic return over the chosen horizon
    - :math:`S_{t,i} \in \mathbb{R}^K` is the vector of cross-sectional scores
    - :math:`\sigma_{t,i}^2` is the forecast idiosyncratic variance

    If `normalize_weights=True`, the positive diagonal entries of :math:`W_t` are
    divided by their cross-sectional average before computing the normal-equation
    statistics.

    With `scale_target_by_idio_vol=True`, the return model is instead:

    .. math::

        u_{t,i} = \sigma_{t,i} S_{t,i}^\top \beta + \epsilon_{t,i},
        \quad \operatorname{Var}(\epsilon_{t,i}) \propto \sigma_{t,i}^2

    The corresponding inverse-variance GLS objective is:

    .. math::

        \beta_t = \arg\min_\beta \sum_i
            \frac{(u_{t,i} - \sigma_{t,i} S_{t,i}^\top \beta)^2}{\sigma_{t,i}^2}

    which is equivalent to ordinary least squares on the volatility-scaled target
    :math:`u_{t,i}/\sigma_{t,i}`:

    .. math::

        \beta_t = \arg\min_\beta \sum_i
          \left(\frac{u_{t,i}}{\sigma_{t,i}} - S_{t,i}^\top \beta\right)^2

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

    With `scale_target_by_idio_vol=False`, the final alpha forecast is:

    .. math::

        \alpha_i = S_i^\top \beta

    With `scale_target_by_idio_vol=True`, the forecast is:

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
        target window. Must be >= 1. Under skfolio's as-of convention, `signal_lag=0`
        would use information observed at the end of :math:`t` to predict return at
        :math:`t`, which is look-ahead. Values larger than 1 can model conservative data
        availability or execution delays.

    neutralize_against : list of str, optional
        Factor names or families to neutralize scores against. If provided, scores are
        orthogonalized with respect to the specified factor exposures before regression.

    outlier_transformer : BaseCSTransformer or "passthrough", optional
         Cross-sectional transformer for outlier handling. If None, defaults to
        `CSWinsorizer()`. Use "passthrough" to skip.

    scoring_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for scoring applied after outlier handling.
        If None, defaults to `CSStandardScaler()`. Use "passthrough" to skip.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations. If provided, outlier and scoring transformations are applied
        within each group separately.

    scale_target_by_idio_vol : bool, default=False
        If `False`, the target is the forward mean idiosyncratic return and WLS weights
        are inverse idiosyncratic variance. If `True`, the target is divided by
        forecast idiosyncratic volatility and fitted with unit weights. The resulting
        idiosyncratic-Sharpe forecast is converted back to return units by multiplying
        by current idiosyncratic volatility.

    normalize_weights : bool, default=True
        If `True`, regression weights are normalized within each observation to have
        average one across valid assets. This removes changes in aggregate weight caused
        by the scale of idiosyncratic variances, while preserving the greater
        statistical weight of observations with more valid assets. In practice, this
        prevents calm, low-volatility regimes from mechanically dominating the EWLS
        statistics just because inverse-variance weights are larger in those regimes.
        Set `normalize_weights=False` for the unnormalized GLS / BLUE estimator.

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
    >>> from skfolio.factor_model.alpha import EWSharpeOptimalAlpha
    >>> from skfolio.factor_model.descriptor import EWMomentum, BookToPrice, Reversal, Passthrough
    >>>
    >>> # X is an AssetPanel with point-in-time descriptor fields, plus
    >>> # "idio_returns" and "idio_variances".
    >>>
    >>> alpha_model = EWSharpeOptimalAlpha(
    ...     descriptors=[
    ...         ("momentum", EWMomentum()),
    ...         ("book_to_price", BookToPrice()),
    ...         ("reversal", Reversal()),
    ...         ("earnings_revision", Passthrough("earnings_revision")),
    ...     ],
    ...     horizon=5,      # one-week forward idiosyncratic return
    ...     half_life=21,    # one-month EWLS half-life
    ...     neutralize_against=["sector"],
    ...     scale_target_by_idio_vol=True,
    ... )
    >>>
    >>> # Latest alpha forecast for the current rebalance.
    >>> alpha_model.fit(X)
    >>> alpha = alpha_model.alpha_
    >>>
    >>> # Historical as-of alpha forecasts.
    >>> alphas = alpha_model.fit_transform(X)
    >>>
    >>> # Online update with newly arrived observations.
    >>> alpha_model.partial_fit(X_new)
    >>> next_alpha = alpha_model.alpha_

    Notes
    -----
    The Information Ratio (IR) of a strategy is approximately:

    .. math::

        \text{IR} \approx \text{IC} \times \sqrt{\text{Breadth}}

    This estimator generalizes single-signal IC weighting by estimating multivariate,
    risk-weighted signal payoffs. The exponential weighting and ridge stabilization
    reduce turnover and estimation noise in the coefficients.

    References
    ----------
    .. [1] Active Portfolio Management
        McGraw-Hill. Grinold, R. C., & Kahn, R. N. (1999).
    """

    descriptors_: list[BaseDescriptor]
    named_descriptors_: dict[str, BaseDescriptor]
    outlier_transformer_: skt.CSTransformer
    scoring_transformer_: skt.CSTransformer

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
        scale_target_by_idio_vol: bool = False,
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
        self.scale_target_by_idio_vol = scale_target_by_idio_vol
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

        required_characteristics = [_IDIO_RETURNS, _IDIO_VARIANCES]
        if self.neutralize_against is not None:
            required_characteristics.append(_EXPOSURES)
        if self.transform_by_group is not None:
            required_characteristics.append(self.transform_by_group)

        validate_asset_panel(
            self, X, required_fields=required_characteristics, reset=first_call
        )

        if first_call:
            self._validate_params()
            self._initialize()

        scores = self._compute_scores(X=X, method=method, routed_params=routed_params)

        idio_returns = X[_IDIO_RETURNS]
        idio_var = X[_IDIO_VARIANCES]

        if self._buffer_scores is not None:
            combined_scores = np.concatenate([self._buffer_scores, scores], axis=1)
            combined_returns = np.concatenate(
                [self._buffer_returns, idio_returns], axis=0
            )
            combined_var = np.concatenate([self._buffer_var, idio_var], axis=0)
        else:
            combined_scores = scores
            combined_returns = idio_returns
            combined_var = idio_var

        self._update_buffer(
            combined_scores=combined_scores,
            combined_returns=combined_returns,
            combined_var=combined_var,
        )

        n_assets = X.n_assets
        n_new_obs = scores.shape[1]
        n_combined_obs = combined_scores.shape[1]
        n_buffered_obs = n_combined_obs - n_new_obs
        target_gap = self._target_gap
        n_trainable_obs = max(0, n_combined_obs - target_gap)
        historical_alphas = (
            np.full((n_new_obs, n_assets), np.nan, dtype=float) if transform else None
        )

        if n_trainable_obs == 0:
            self.alpha_ = None
            return historical_alphas

        regression_scores = combined_scores[:, :n_trainable_obs, :].transpose(1, 2, 0)
        forward_returns = _forward_mean_return(
            combined_returns, horizon=self.horizon, lag=self.signal_lag
        )[:n_trainable_obs]

        historical_coefficients = self._update_ewls(
            forward_returns=forward_returns,
            idio_var=combined_var[:n_trainable_obs],
            scores=regression_scores,
            return_historical=transform,
        )

        if transform:
            for forecast_idx in range(n_new_obs):
                combined_obs_idx = n_buffered_obs + forecast_idx
                coef_idx = combined_obs_idx - target_gap
                if 0 <= coef_idx < n_trainable_obs:
                    historical_alphas[forecast_idx] = _compute_alpha(
                        scores=combined_scores[:, combined_obs_idx, :],
                        coefficient=historical_coefficients[coef_idx],
                        idio_var=combined_var[combined_obs_idx],
                        scale_target_by_idio_vol=self.scale_target_by_idio_vol,
                    )

        if self._n_valid_regression_obs == 0:
            self.alpha_ = None
        else:
            self.alpha_ = _compute_alpha(
                scores=scores[:, -1, :],
                coefficient=self.coef_,
                idio_var=idio_var[-1],
                scale_target_by_idio_vol=self.scale_target_by_idio_vol,
            )

        return historical_alphas

    def _compute_scores(
        self, *, X: AssetPanel, method: str, routed_params
    ) -> FloatArray:
        """Compute descriptor scores from the input panel."""
        cs_weights = X.estimation_mask.astype(float)
        cs_groups = (
            X[self.transform_by_group] if self.transform_by_group is not None else None
        )

        n_descriptors = len(self.descriptors_)

        # Threading avoids copying the (potentially large) AssetPanel
        # to each worker. Workers only read from it, so shared memory
        # is safe. Descriptor computations are NumPy-dominated and
        # release the GIL, giving true parallelism with threads.
        scores = skp.Parallel(n_jobs=self.n_jobs, prefer="threads")(
            skp.delayed(call_asset_panel_transform)(
                des,
                X=X,
                fit_params=routed_params[name][method],
                method=f"{method}_transform",
            )
            for name, des in self.named_descriptors_.items()
        )
        scores = np.stack(scores, axis=0)

        for i in range(n_descriptors):
            if self._outlier_transformer != _PASSTHROUGH:
                scores[i] = self._outlier_transformer.fit_transform(
                    scores[i], cs_weights=cs_weights, cs_groups=cs_groups
                )

            if self._scoring_transformer != _PASSTHROUGH:
                scores[i] = self._scoring_transformer.fit_transform(
                    scores[i], cs_weights=cs_weights, cs_groups=cs_groups
                )

        # Score neutralization
        if self.neutralize_against is not None:
            field = X.fields[_EXPOSURES]
            exposures = field.values
            factor_names = field.third_axis_labels
            factor_families = field.third_axis_groups

            scores = _neutralize_scores(
                neutralize_against=self.neutralize_against,
                scores=scores,
                exposures=exposures,
                cs_weights=cs_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )
            # Re-apply scoring after neutralization
            for i in range(n_descriptors):
                if self._scoring_transformer != _PASSTHROUGH:
                    scores[i] = self._scoring_transformer.fit_transform(
                        scores[i],
                        cs_weights=cs_weights,
                        cs_groups=cs_groups,
                    )

        return scores

    def _update_ewls(
        self,
        forward_returns: FloatArray,
        idio_var: FloatArray,
        scores: FloatArray,
        return_historical: bool,
    ) -> FloatArray | None:
        """Update EWLS normal equations and optionally return coefficient history."""
        valid_var = np.isfinite(idio_var) & (idio_var > 0)
        valid_target = np.isfinite(forward_returns) & valid_var

        if self.scale_target_by_idio_vol:
            idio_vol = np.sqrt(np.where(valid_var, idio_var, np.nan))
            target = safe_divide(forward_returns, idio_vol, fill_value=np.nan)
            weights = np.where(valid_target, 1.0, 0.0)
        else:
            target = forward_returns
            inv_var = safe_divide(1.0, idio_var, fill_value=0.0)
            weights = np.where(valid_target, inv_var, 0.0)

        n_trainable_obs, _, n_descriptors = scores.shape
        historical_coefficients = (
            np.full((n_trainable_obs, n_descriptors), np.nan, dtype=float)
            if return_historical
            else None
        )

        valid = np.all(np.isfinite(scores), axis=2) & valid_target

        for t in range(n_trainable_obs):
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

    def _update_buffer(
        self,
        *,
        combined_scores: FloatArray,
        combined_returns: FloatArray,
        combined_var: FloatArray,
    ) -> None:
        """Keep only rows needed to mature future forward-return targets."""
        n_combined_obs = combined_scores.shape[1]
        buffer_start = max(0, n_combined_obs - self._target_gap)
        self._buffer_scores = combined_scores[:, buffer_start:, :].copy()
        self._buffer_returns = combined_returns[buffer_start:, :].copy()
        self._buffer_var = combined_var[buffer_start:, :].copy()

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self):
        """Validate hyperparameters."""
        if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError(
                f"horizon must be a positive integer (>= 1), got {self.horizon}"
            )
        if not isinstance(self.signal_lag, (int, np.integer)) or self.signal_lag < 1:
            raise ValueError(
                f"signal_lag must be a positive integer (>= 1), got {self.signal_lag}"
            )
        if (
            not isinstance(self.half_life, (int, float, np.number))
            or self.half_life <= 0
        ):
            raise ValueError(
                f"half_life must be a positive number, got {self.half_life}"
            )
        if (
            not isinstance(self.ridge_scale, (int, float, np.number))
            or self.ridge_scale < 0
        ):
            raise ValueError(
                f"ridge_scale must be a non-negative number, got {self.ridge_scale}"
            )
        if not self.descriptors:
            raise ValueError("descriptors cannot be empty")
        if not isinstance(self.scale_target_by_idio_vol, (bool, np.bool_)):
            raise ValueError(
                "scale_target_by_idio_vol must be a boolean, "
                f"got {self.scale_target_by_idio_vol!r}"
            )
        if not isinstance(self.normalize_weights, (bool, np.bool_)):
            raise ValueError(
                f"normalize_weights must be a boolean, got {self.normalize_weights!r}"
            )

    def _initialize(self):
        """Initialize internal state on first call."""
        names, descriptors = self._validate_descriptors()

        self.descriptors_ = [sk.clone(des) for des in descriptors]
        self.named_descriptors_ = {
            name: estimator
            for name, estimator in zip(names, self.descriptors_, strict=True)
        }

        self._outlier_transformer = check_estimator(
            self.outlier_transformer,
            default=CSWinsorizer(),
            check_type=BaseCSTransformer,
        )

        self._scoring_transformer = check_estimator(
            self.scoring_transformer,
            default=CSStandardScaler(),
            check_type=BaseCSTransformer,
        )

        n_descriptors = len(self.descriptors_)
        self._decay = half_life_to_decay_factor(self.half_life)
        self._ew_normal_matrix = np.zeros((n_descriptors, n_descriptors))
        self._ew_target_cross_product = np.zeros(n_descriptors)
        self._n_valid_regression_obs = 0
        self.coef_ = np.full(n_descriptors, np.nan)

        self._buffer_scores = None
        self._buffer_returns = None
        self._buffer_var = None

    @property
    def _target_gap(self) -> int:
        """Number of future rows required before a signal observation matures."""
        return self.signal_lag + self.horizon - 1


def _compute_alpha(
    scores: FloatArray,
    coefficient: FloatArray,
    idio_var: FloatArray,
    scale_target_by_idio_vol: bool,
) -> FloatArray:
    """Compute alpha for one observation."""
    alpha = coefficient @ scores
    if scale_target_by_idio_vol:
        valid_var = np.isfinite(idio_var) & (idio_var > 0)
        idio_vol = np.sqrt(np.where(valid_var, idio_var, np.nan))
        alpha = alpha * idio_vol
    return alpha
