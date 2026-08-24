"""Fixed-weighted descriptor alpha estimator."""
# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn.utils.metadata_routing as skm

import skfolio.typing as skt
from skfolio._constants import _EXPOSURES, _IDIO_VARIANCES, _PASSTHROUGH
from skfolio.alpha import ForecastUnit
from skfolio.alpha._base import BaseAlpha, BaseAlphaDescriptorComposition
from skfolio.containers import AssetPanel
from skfolio.descriptor import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import safe_divide
from skfolio.utils.tools import (
    _validate_positive_real,
    _validate_unit_interval,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "alpha_"


class FixedWeightedAlpha(BaseAlphaDescriptorComposition, BaseAlpha):
    r"""Fixed-weighted descriptor alpha estimator.

    This estimator converts descriptors into cross-sectional scores, optionally
    neutralizes the scores against factor exposures, and combines them with fixed
    weights to produce an alpha forecast in expected return units.

    Unlike :class:`EWSharpeOptimalAlpha`, the descriptor weights and forecast scale are
    not estimated from realized returns. They are fixed hyperparameters of the
    estimator.

    For descriptor score :math:`s_{k,i}` and signed fixed weight :math:`w_k`, the
    composite score for asset :math:`i` is:

    .. math::

        z_i = \frac{\sum_{k \in V_i} w_k s_{k,i}}
                   {\sum_{k \in V_i} |w_k|}

    where :math:`V_i` is the set of descriptors with finite scores for asset
    :math:`i`. The forecast in the selected `forecast_unit` is:

    .. math::

        \hat y_i = \text{forecast\_scale} \, z_i

    With `forecast_unit=ForecastUnit.IDIO_RETURN`, the alpha forecast is:

    .. math::

        \alpha_i = \hat y_i

    With `forecast_unit=ForecastUnit.IDIO_SHARPE`, the forecast is converted to
    expected return units:

    .. math::

        \alpha_i = \sigma_i \hat y_i

    where :math:`\sigma_i` is the forecast idiosyncratic volatility.

    Parameters
    ----------
    descriptors : list of (name, estimator) tuples
        List of descriptors that compute signals from characteristics.

    weights : array-like of shape (n_descriptors,), optional
        Signed descriptor weights. If `None`, equal positive weights are used. Weights
        are normalized by their absolute sum. When some descriptor scores are missing,
        the composite score is renormalized over the available absolute weight.

    forecast_scale : float
        Multiplicative scale applied to the composite score in `forecast_unit`. With
        `ForecastUnit.IDIO_RETURN`, this is expected idiosyncratic return per score
        unit. With `ForecastUnit.IDIO_SHARPE`, this is idiosyncratic Sharpe per score
        unit.

    forecast_unit : ForecastUnit, default=ForecastUnit.IDIO_RETURN
        Unit of the fixed forecast before conversion to `alpha_`. The `alpha_`
        attribute is always returned in expected return units. With
        `ForecastUnit.IDIO_SHARPE`, `idio_variances` are required and the forecast is
        multiplied by current idiosyncratic volatility.

    min_coverage : float, default=0.0
        Minimum fraction of absolute descriptor weight that must be finite for the
        composite score to be computed. Values where available absolute weight is below
        this threshold are set to `NaN`. Must be in `[0, 1]`.

    neutralize_against : list of str, optional
        Factor names or families to neutralize scores against.

    outlier_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for descriptor outlier handling. If `None`,
        defaults to `CSWinsorizer()`. Use `"passthrough"` to skip.

    scoring_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for descriptor scoring applied after outlier
        handling. If `None`, defaults to `CSStandardScaler()`. Use `"passthrough"` to
        skip.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations.

    n_jobs : int, default=1
        Number of parallel jobs for descriptor computation.
    """

    descriptors_: list[BaseDescriptor]
    named_descriptors_: dict[str, BaseDescriptor]

    def __init__(
        self,
        *,
        descriptors: list[tuple[str, BaseDescriptor]],
        forecast_scale: float,
        weights: FloatArray | None = None,
        forecast_unit: ForecastUnit = ForecastUnit.IDIO_RETURN,
        min_coverage: float = 0.0,
        neutralize_against: list[str] | None = None,
        outlier_transformer: skt.CSTransformer = None,
        scoring_transformer: skt.CSTransformer = None,
        transform_by_group: str | None = None,
        n_jobs: int = 1,
    ):
        self.descriptors = descriptors
        self.weights = weights
        self.forecast_scale = forecast_scale
        self.forecast_unit = forecast_unit
        self.min_coverage = min_coverage
        self.neutralize_against = neutralize_against
        self.outlier_transformer = outlier_transformer
        self.scoring_transformer = scoring_transformer
        self.transform_by_group = transform_by_group
        self.n_jobs = n_jobs

    def fit(self, X: AssetPanel, y=None, **fit_params) -> FixedWeightedAlpha:
        """Fit descriptors and store the latest alpha forecast in `alpha_`."""
        self._reset()
        self._fit(X, y, method="fit", **fit_params)
        return self

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Fit descriptors and return historical alpha forecasts."""
        self._reset()
        return self._fit(X, y, method="fit", transform=True, **fit_params)

    def partial_fit(self, X: AssetPanel, y=None, **fit_params) -> FixedWeightedAlpha:
        """Incrementally update descriptors and store the latest alpha forecast."""
        self._fit(X, y, method="partial_fit", **fit_params)
        return self

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Incrementally update descriptors and return new alpha forecasts."""
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
        """Fit descriptor state and optionally return alpha history."""
        routed_params = skm.process_routing(self, method, **fit_params)

        first_call = not hasattr(self, _FITTED_ATTR)

        required_fields = []
        strictly_positive_or_nan = []
        if self.forecast_unit is ForecastUnit.IDIO_SHARPE:
            required_fields.append(_IDIO_VARIANCES)
            strictly_positive_or_nan.append(_IDIO_VARIANCES)
        if self.neutralize_against is not None:
            required_fields.append(_EXPOSURES)
        if self.transform_by_group is not None:
            required_fields.append(self.transform_by_group)

        validate_asset_panel(
            self,
            X,
            required_fields=required_fields,
            strictly_positive_or_nan=strictly_positive_or_nan,
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        scores = self._compute_scores(X=X, method=method, routed_params=routed_params)
        self.composite_score_ = self._combine_scores(scores)
        if len(self.descriptors_) > 1 and self.scoring_transformer_ != _PASSTHROUGH:
            cs_weights = X.estimation_mask.astype(float)
            cs_groups = (
                X[self.transform_by_group]
                if self.transform_by_group is not None
                else None
            )
            self.composite_score_ = self.scoring_transformer_.fit_transform(
                self.composite_score_, cs_weights=cs_weights, cs_groups=cs_groups
            )

        idio_variances = (
            X[_IDIO_VARIANCES]
            if self.forecast_unit is ForecastUnit.IDIO_SHARPE
            else None
        )
        alphas = self._compute_alpha(
            composite_score=self.composite_score_, idio_variances=idio_variances
        )
        self.alpha_ = alphas[-1]
        return alphas if transform else None

    def _combine_scores(self, scores: FloatArray) -> FloatArray:
        """Return finite-aware signed fixed-weight descriptor composite."""
        abs_weights = np.abs(self.descriptor_weights_)
        weighted_scores = np.zeros(scores.shape[:2], dtype=float)
        available_abs_weight = np.zeros(scores.shape[:2], dtype=float)

        for weight, abs_weight, score in zip(
            self.descriptor_weights_,
            abs_weights,
            np.moveaxis(scores, 2, 0),
            strict=True,
        ):
            valid = np.isfinite(score)
            weighted_scores += np.where(valid, weight * score, 0.0)
            available_abs_weight += np.where(valid, abs_weight, 0.0)

        composite = safe_divide(
            weighted_scores, available_abs_weight, fill_value=np.nan
        )
        total_abs_weight = float(abs_weights.sum())
        coverage = safe_divide(available_abs_weight, total_abs_weight, fill_value=0.0)
        composite[coverage < self.min_coverage] = np.nan
        return composite

    def _compute_alpha(
        self, composite_score: FloatArray, idio_variances: FloatArray | None
    ) -> FloatArray:
        """Convert composite scores to alpha in expected return units."""
        forecast = self.forecast_scale * composite_score
        if self.forecast_unit is not ForecastUnit.IDIO_SHARPE:
            return forecast
        return forecast * np.sqrt(idio_variances)

    def _reset(self) -> None:
        """Reset fitted descriptor state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        self._validate_descriptor_params()
        _validate_positive_real(self.forecast_scale, "forecast_scale")
        _validate_unit_interval(self.min_coverage, "min_coverage")

    def _initialize(self) -> None:
        """Initialize descriptor state and fixed weights."""
        self._initialize_common_state()

        if self.weights is None:
            self.descriptor_weights_ = np.ones(len(self.descriptors_)) / len(
                self.descriptors_
            )
        else:
            self.descriptor_weights_ = np.asarray(self.weights, dtype=float)
            if self.descriptor_weights_.ndim != 1:
                raise ValueError("weights should be a 1D array")
            if self.descriptor_weights_.size != len(self.descriptors_):
                raise ValueError("weights should have the same length as descriptors")
            if not np.all(np.isfinite(self.descriptor_weights_)):
                raise ValueError("weights should contain only finite values")
            abs_sum = float(np.sum(np.abs(self.descriptor_weights_)))
            _validate_positive_real(abs_sum, "sum(abs(weights))")
            self.descriptor_weights_ = self.descriptor_weights_ / abs_sum
