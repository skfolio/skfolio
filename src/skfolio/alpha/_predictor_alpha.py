"""Cross-sectional predictor alpha estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import sklearn as sk
import sklearn.model_selection as sks
import sklearn.utils.metadata_routing as skm

import skfolio.typing as skt
from skfolio._constants import (
    _DESCRIPTOR_SCORES,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _PASSTHROUGH,
)
from skfolio.alpha import ForecastUnit
from skfolio.alpha._base import BaseAlpha, BaseAlphaDescriptorComposition
from skfolio.containers import AssetPanel
from skfolio.descriptor import BaseDescriptor
from skfolio.preprocessing import BaseCSTransformer, CSWinsorizer
from skfolio.typing import BoolArray, FloatArray
from skfolio.utils.stats import _forward_mean_return, safe_divide
from skfolio.utils.tools import (
    _validate_bool,
    _validate_positive_real,
    check_estimator,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "alpha_"

_MIN_SAMPLES_PER_FOLD = 2
_MIN_VALID_ASSETS_FOR_CALIBRATION = 2
_CALIBRATION_RIDGE_SCALE = 1e-6


class PredictorAlpha(BaseAlphaDescriptorComposition, BaseAlpha):
    r"""Predictor alpha estimator using a user-provided regressor.

    This estimator converts descriptors into cross-sectional scores, optionally
    neutralizes those scores against factor exposures and fits a scikit-learn compatible
    regressor where each observation-asset pair is one training sample. It supports
    nonlinear signal combinations while keeping the final forecast in expected
    idiosyncratic return units when calibration is enabled [1]_.

    The predictor supports two forecast units. With
    `forecast_unit=ForecastUnit.IDIO_RETURN`, it is fitted to the forward mean
    idiosyncratic return :math:`\epsilon_{t,i}`. With
    `forecast_unit=ForecastUnit.IDIO_SHARPE`, it is fitted to
    :math:`\epsilon_{t,i} / \sigma_{t,i}` and the forecast is multiplied by current
    idiosyncratic volatility so `alpha_` remains in expected return units. This is
    useful when signals are assumed to forecast idiosyncratic Sharpe rather than raw
    idiosyncratic return.

    When `calibrate_to_return_units=True`, the raw predictor output
    :math:`\hat a_{t,i}` is calibrated to expected-return units with scalar
    exponentially weighted least squares:

    .. math::

        \beta_t = \arg\min_\beta \sum_i
        \frac{(\epsilon_{t,i} - \hat a_{t,i}\beta)^2}{\sigma_{t,i}^2}

    The calibration coefficient is estimated with exponentially weighted least squares:

    .. math::

        A_t^{EW} = \lambda A_{t-1}^{EW}
        + (1 - \lambda)\hat a_t^\top W_t \hat a_t

    .. math::

        b_t^{EW} = \lambda b_{t-1}^{EW}
        + (1 - \lambda)\hat a_t^\top W_t \epsilon_t,
        \quad \lambda = 2^{-1/\text{half-life}}

    and the ridge-stabilized coefficient is:

    .. math::

        \beta_t = b_t^{EW} / (A_t^{EW} + \rho_t)

    This produces alpha in expected return units, which is required whenever the
    optimizer is trading off alpha against real costs and constraints such as
    transaction costs, market impact, borrow costs, or turnover constraints.

    In batch mode, calibration uses predictions from held-out CV folds when enough
    samples are available. This reduces the in-sample scale inflation from fitting
    and calibrating on the same predictions. These predictions are used only for
    scale calibration, not as a time-series performance estimate. The default splitter
    treats valid observation-asset pairs as approximately exchangeable. Pass a
    date-aware or purged splitter through `cv` when the calibration itself should
    enforce stricter temporal separation.

    The estimator supports :meth:`fit` and :meth:`partial_fit`. In online mode,
    samples are trained when their forward-return targets become observable,
    including rows carried in the target-maturity buffer. The predictor must
    support `partial_fit` after the first fitted update.

    Parameters
    ----------
    predictor : estimator
        Regressor that implements `fit` and `predict`. For online mode, the predictor
        must also implement `partial_fit`. The predictor receives one sample per valid
        observation-asset pair, with shape `(n_observations * n_assets, n_descriptors)`,
        and predicts the transformed target.

    descriptors : list of (name, estimator) tuples
        List of descriptors that compute signals from characteristics. Each tuple
        contains a string name and a descriptor estimator. Multiple descriptors are
        aggregated into a single alpha using multivariate regression. The descriptors
        are evaluated in parallel if `n_jobs > 1`.

    half_life : float, default=20
        Half-life of the EWLS calibration statistics in number of observations. Only
        used when `calibrate_to_return_units=True`.

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
        `signal_lag=0` would use information observed at the end of :math:`t` to
        predict return at :math:`t`, which is look-ahead. Values larger than 1 can
        model conservative data availability or execution delays.

    neutralize_against : list of str, optional
        Factor names or families to neutralize scores against. If provided, scores are
        orthogonalized with respect to the specified factor exposures before prediction.

    outlier_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for descriptor outlier handling. If `None`,
        defaults to `CSWinsorizer()`. Use `"passthrough"` to skip.

    scoring_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for descriptor scoring applied after outlier
        handling. If None, defaults to `CSStandardScaler()`. Use "passthrough" to skip.

    target_outlier_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for target outlier handling. If `None`, defaults
        to `CSWinsorizer()`. Use `"passthrough"` to skip.

    target_scoring_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for target scoring. If `None`, defaults to
        `"passthrough"`. The calibration stage calibrates the predictor output back
        to expected-return units when `calibrate_to_return_units=True`.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations. If provided, cross-sectional transformations are applied
        within each group separately.

    forecast_unit : ForecastUnit, default=ForecastUnit.IDIO_RETURN
        Unit of the intermediate forecast learned by the predictor. With
        `ForecastUnit.IDIO_RETURN`, the predictor is trained on forward mean
        idiosyncratic return. With `ForecastUnit.IDIO_SHARPE`, the target is divided by
        forecast idiosyncratic volatility and the resulting idiosyncratic-Sharpe
        forecast is converted back to return units by multiplying by current
        idiosyncratic volatility.

    calibrate_to_return_units : bool, default=True
        If `True`, calibrate raw predictor output to expected return units using
        scalar EWLS. If `False`, return the predictor output after any volatility
        conversion implied by `forecast_unit`.

    forecast_scale : float, default=1.0
        Multiplicative scale applied to the final alpha forecast after optional
        return-unit calibration. This controls alpha strength without changing the
        predictor or calibration coefficient estimates.

    cv : cross-validator, int, optional
        Cross-validation strategy used to obtain predictions from held-out folds for
        return-unit calibration. When `None`, uses `KFold(5)`. CV is used only in
        batch mode when `calibrate_to_return_units=True` and there are enough
        samples.

    n_jobs : int, default=1
        Number of parallel jobs for descriptor computation and cross-validation.

    Attributes
    ----------
    alpha_ : ndarray of shape (n_assets,) or None
        Estimated alpha for each asset, after applying `forecast_scale`. If
        `calibrate_to_return_units=True`, this is in expected return units. Otherwise,
        it is the predictor output after any volatility conversion. Returns `None`
        during warmup.

    predictor_ : estimator
        Fitted predictor instance.

    descriptors_ : list of BaseDescriptor
        Fitted descriptor estimators.

    named_descriptors_ : dict of {str: BaseDescriptor}
        Dictionary mapping descriptor names to fitted estimators.

    outlier_transformer_ : BaseCSTransformer or str
        Fitted descriptor outlier transformer.

    scoring_transformer_ : BaseCSTransformer or str
        Fitted descriptor scoring transformer.

    target_outlier_transformer_ : BaseCSTransformer or str
        Fitted target outlier transformer.

    target_scoring_transformer_ : BaseCSTransformer or str
        Fitted target scoring transformer.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray
        Names of assets in the coverage universe.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDRegressor
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.alpha import ForecastUnit, PredictorAlpha
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
    >>> alpha_model = PredictorAlpha(
    ...     predictor=SGDRegressor(),
    ...     descriptors=[
    ...         ("momentum", EWMomentum()),
    ...         ("book_to_price", BookToPrice()),
    ...         ("reversal", Reversal()),
    ...         ("eps_ntm", Passthrough("eps_ntm")),
    ...     ],
    ...     horizon=5,
    ...     half_life=21,
    ...     neutralize_against=["market", "beta", "size"],
    ...     forecast_unit=ForecastUnit.IDIO_SHARPE,
    ... )
    >>>
    >>> alpha_model.fit(X)
    >>> print(alpha_model.alpha_)
    >>>
    >>> # Online learning (requires predictor with partial_fit)
    >>> alpha_model.partial_fit(X[-5:])
    >>> print(alpha_model.alpha_)

    See Also
    --------
    EWSharpeOptimalAlpha : Linear signal aggregation with Sharpe-optimal WLS weighting.

    References
    ----------
    .. [1] "Active Portfolio Management: A Quantitative Approach for Producing Superior
       Returns and Controlling Risk", McGraw-Hill, Grinold & Kahn (1999).
    """

    descriptors_: list[BaseDescriptor]
    named_descriptors_: dict[str, BaseDescriptor]

    def __init__(
        self,
        *,
        predictor: Any,
        descriptors: list[tuple[str, BaseDescriptor]],
        horizon: int = 1,
        signal_lag: int = 1,
        neutralize_against: list[str] | None = None,
        outlier_transformer: skt.CSTransformer = None,
        scoring_transformer: skt.CSTransformer = None,
        target_outlier_transformer: skt.CSTransformer = None,
        target_scoring_transformer: skt.CSTransformer = None,
        transform_by_group: str | None = None,
        forecast_unit: ForecastUnit = ForecastUnit.IDIO_RETURN,
        calibrate_to_return_units: bool = True,
        forecast_scale: float = 1.0,
        half_life: float = 20,
        cv: sks.BaseCrossValidator | int | None = None,
        n_jobs: int = 1,
    ):
        self.predictor = predictor
        self.descriptors = descriptors
        self.horizon = horizon
        self.signal_lag = signal_lag
        self.neutralize_against = neutralize_against
        self.outlier_transformer = outlier_transformer
        self.scoring_transformer = scoring_transformer
        self.target_outlier_transformer = target_outlier_transformer
        self.target_scoring_transformer = target_scoring_transformer
        self.transform_by_group = transform_by_group
        self.forecast_unit = forecast_unit
        self.calibrate_to_return_units = calibrate_to_return_units
        self.forecast_scale = forecast_scale
        self.half_life = half_life
        self.cv = cv
        self.n_jobs = n_jobs

    def get_metadata_routing(self):
        """Return metadata routing for descriptors and the predictor."""
        router = super().get_metadata_routing()
        router.add(
            predictor=self.predictor,
            method_mapping=skm.MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="partial_fit", callee="fit")
            .add(caller="partial_fit", callee="partial_fit"),
        )
        return router

    def fit(self, X: AssetPanel, y=None, **fit_params) -> PredictorAlpha:
        """Fit the alpha model from scratch (batch mode).

        This method works with any sklearn-compatible predictor. It resets all
        internal state and fits on the provided data. When calibration is enabled,
        predictions from held-out CV folds can be used to calibrate the return-unit
        scale.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "idio_returns", descriptor fields and optionally
            "idio_variances" for calibration or volatility-scaled targets, and
            "exposures" for score neutralization.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors and predictor.

        Returns
        -------
        self : PredictorAlpha
            Fitted estimator.
        """
        self._reset()
        return self._fit(X, y, method="fit", **fit_params)

    def partial_fit(self, X: AssetPanel, y=None, **fit_params) -> PredictorAlpha:
        """Incrementally fit the alpha model with new observations (online mode).

        This method supports streaming/online updates. It maintains internal
        buffers to compute forward returns across partial_fit calls. Only samples
        whose targets have newly matured are used for training, avoiding
        double-counting while still training buffered rows once their labels are
        observable.

        On the first call, the predictor is trained using `fit()`. On subsequent
        calls, the predictor is updated using `partial_fit()`, which requires
        the predictor to support this method.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "idio_returns", descriptor fields and optionally
            "idio_variances" for calibration or volatility-scaled targets, and
            "exposures" for score neutralization.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors and predictor.

        Returns
        -------
        self : PredictorAlpha
            Fitted estimator.

        Raises
        ------
        TypeError
            If the predictor does not support `partial_fit` (raised on second
            or subsequent calls).
        """
        return self._fit(X, y, method="partial_fit", **fit_params)

    def _fit(
        self,
        X: AssetPanel,
        y=None,
        *,
        method: str,
        **fit_params,
    ) -> PredictorAlpha:
        """Fit predictor and calibration state from one batch."""
        routed_params = skm.process_routing(self, method, **fit_params)

        first_call = not hasattr(self, _FITTED_ATTR)

        required_fields = [_IDIO_RETURNS]
        strictly_positive_or_nan = []
        if self._needs_idio_variances:
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
        else:
            if self._predictor_fitted and not hasattr(self.predictor_, "partial_fit"):
                raise TypeError(
                    f"predictor {type(self.predictor_).__name__} does not support "
                    "partial_fit. For online mode use a predictor with partial_fit. "
                    "For batch mode, use fit() instead."
                )

        scores = self._compute_scores(X=X, method=method, routed_params=routed_params)

        training_fields = [_IDIO_RETURNS]
        if self._needs_idio_variances:
            training_fields.append(_IDIO_VARIANCES)
        if self.transform_by_group is not None:
            training_fields.append(self.transform_by_group)

        X = self._make_training_panel(X, scores=scores, fields=training_fields)
        X = self._prepend_buffer(X)

        n_trainable = max(0, X.n_observations - self._target_gap)
        latest_scores = X[_DESCRIPTOR_SCORES][-1]
        latest_idio_variances = (
            X[_IDIO_VARIANCES][-1] if self._needs_idio_variances else None
        )

        if n_trainable > 0:
            forward_return, predictor_target = self._compute_targets(X)
            scores_flat, predictor_target_flat, train_mask = (
                self._prepare_training_data(
                    X, predictor_target=predictor_target, n_trainable=n_trainable
                )
            )
            has_training_samples = np.any(train_mask)
        else:
            has_training_samples = False

        # Rows inside the target gap cannot train yet. Invalid targets are also
        # skipped. If the predictor is already fitted, still publish a forecast
        # for the latest scores.
        if not has_training_samples:
            if self._predictor_fitted:
                uncalibrated_alpha = self._predict_uncalibrated_alpha(
                    scores=latest_scores, idio_variances=latest_idio_variances
                )
                self._set_alpha(uncalibrated_alpha)
            else:
                self.alpha_ = None
            self._update_buffers(X)
            return self

        train_slice = slice(0, n_trainable)

        if self.calibrate_to_return_units:
            # Calibration uses pre-update predictions when possible. This avoids
            # estimating the calibration coefficient on targets already consumed
            # by the predictor, except on the first incremental update.
            uncalibrated_alpha_for_calibration = (
                self._predict_uncalibrated_alpha_before_update(
                    scores=X[_DESCRIPTOR_SCORES][train_slice],
                    scores_flat=scores_flat,
                    predictor_target_flat=predictor_target_flat,
                    train_mask=train_mask,
                    idio_variances=X[_IDIO_VARIANCES][train_slice],
                    method=method,
                    routed_params=routed_params,
                )
            )

        self._fit_predictor(
            scores_flat=scores_flat,
            predictor_target_flat=predictor_target_flat,
            train_mask=train_mask,
            routed_params=routed_params,
        )

        if self.calibrate_to_return_units:
            if uncalibrated_alpha_for_calibration is None:
                # The first incremental update has no previous predictor and no
                # CV path, so calibration starts from newly fitted predictions.
                uncalibrated_alpha_for_calibration = self._predict_uncalibrated_alpha(
                    scores=X[_DESCRIPTOR_SCORES][train_slice],
                    idio_variances=X[_IDIO_VARIANCES][train_slice],
                )

            self._update_calibration(
                uncalibrated_alpha=uncalibrated_alpha_for_calibration,
                forward_return=forward_return[train_slice],
                idio_variances=X[_IDIO_VARIANCES][train_slice],
                estimation_weights=X.estimation_mask[train_slice].astype(float),
            )

        # The published alpha always uses the updated predictor and the latest
        # available calibration coefficient.
        uncalibrated_alpha = self._predict_uncalibrated_alpha(
            scores=latest_scores, idio_variances=latest_idio_variances
        )
        self._set_alpha(uncalibrated_alpha)

        self._update_buffers(X)
        return self

    def _compute_targets(self, X: AssetPanel) -> tuple[FloatArray, FloatArray]:
        """Compute targets for calibration and predictor fitting.

        Returns
        -------
        forward_return : ndarray
            Raw forward mean returns used for return-unit calibration.
        predictor_target : ndarray
            Target used to train the predictor.
        """
        forward_return = _forward_mean_return(
            X[_IDIO_RETURNS], horizon=self.horizon, lag=self.signal_lag
        )
        if self.forecast_unit is ForecastUnit.IDIO_SHARPE:
            predictor_target = safe_divide(
                forward_return,
                np.sqrt(X[_IDIO_VARIANCES]),
                fill_value=np.nan,
            )
        else:
            predictor_target = forward_return

        cs_weights = X.estimation_mask.astype(float)
        cs_groups = (
            X[self.transform_by_group] if self.transform_by_group is not None else None
        )

        if self.target_outlier_transformer_ != _PASSTHROUGH:
            predictor_target = self.target_outlier_transformer_.fit_transform(
                predictor_target, cs_weights=cs_weights, cs_groups=cs_groups
            )

        if self.target_scoring_transformer_ != _PASSTHROUGH:
            predictor_target = self.target_scoring_transformer_.fit_transform(
                predictor_target, cs_weights=cs_weights, cs_groups=cs_groups
            )

        return forward_return, predictor_target

    def _prepare_training_data(
        self, X: AssetPanel, predictor_target: FloatArray, n_trainable: int
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Build flattened predictor inputs for trainable dates."""
        scores = X[_DESCRIPTOR_SCORES][:n_trainable].reshape(
            n_trainable * self.n_assets_, len(self.descriptors_)
        )
        predictor_target_flat = predictor_target[:n_trainable].reshape(
            n_trainable * self.n_assets_
        )
        weights = (
            X.estimation_mask[:n_trainable]
            .astype(float)
            .reshape(n_trainable * self.n_assets_)
        )
        train_mask = (
            np.isfinite(predictor_target_flat)
            & np.all(np.isfinite(scores), axis=1)
            & (weights > 0)
        )

        return scores, predictor_target_flat, train_mask

    def _fit_predictor(
        self,
        scores_flat: FloatArray,
        predictor_target_flat: FloatArray,
        train_mask: BoolArray,
        routed_params,
    ) -> None:
        """Fit or update the user-provided predictor on new valid samples."""
        if not self._predictor_fitted:
            self.predictor_.fit(
                scores_flat[train_mask],
                predictor_target_flat[train_mask],
                **routed_params["predictor"]["fit"],
            )
        else:
            self.predictor_.partial_fit(
                scores_flat[train_mask],
                predictor_target_flat[train_mask],
                **routed_params["predictor"]["partial_fit"],
            )
        self._predictor_fitted = True

    def _predict_uncalibrated_alpha_before_update(
        self,
        *,
        scores: FloatArray,
        scores_flat: FloatArray,
        predictor_target_flat: FloatArray,
        train_mask: BoolArray,
        idio_variances: FloatArray,
        method: str,
        routed_params,
    ) -> FloatArray | None:
        """Predict uncalibrated alpha before the predictor consumes new targets."""
        if self._predictor_fitted:
            return self._predict_uncalibrated_alpha(
                scores=scores, idio_variances=idio_variances
            )

        if method == "fit":
            cv = sks.check_cv(self.cv)
            n_splits = cv.get_n_splits()
            n_valid_samples = int(np.sum(train_mask))
            if n_valid_samples >= n_splits * _MIN_SAMPLES_PER_FOLD:
                alpha_flat = np.full_like(predictor_target_flat, np.nan, dtype=float)
                oos_pred = sks.cross_val_predict(
                    sk.clone(self.predictor_),
                    scores_flat[train_mask],
                    predictor_target_flat[train_mask],
                    cv=deepcopy(cv),
                    method="predict",
                    n_jobs=self.n_jobs,
                    params=routed_params["predictor"]["fit"],
                )
                alpha_flat[train_mask] = oos_pred
                alpha = alpha_flat.reshape(scores.shape[:2])
                if self.forecast_unit is not ForecastUnit.IDIO_SHARPE:
                    return alpha
                return alpha * np.sqrt(idio_variances)

            if self.cv is not None:
                raise ValueError(
                    "cv requires at least "
                    f"{n_splits * _MIN_SAMPLES_PER_FOLD} valid training samples, "
                    f"got {n_valid_samples}"
                )

        return None

    def _predict_uncalibrated_alpha(
        self, scores: FloatArray, idio_variances: FloatArray | None = None
    ) -> FloatArray:
        """Predict alpha before applying the calibration coefficient."""
        valid_mask = np.all(np.isfinite(scores), axis=-1)
        alpha = np.full(valid_mask.shape, np.nan)
        if np.any(valid_mask):
            alpha[valid_mask] = self.predictor_.predict(scores[valid_mask])

        if self.forecast_unit is not ForecastUnit.IDIO_SHARPE:
            return alpha
        if idio_variances is None:
            raise ValueError(
                "idio_variances are required when "
                "forecast_unit=ForecastUnit.IDIO_SHARPE"
            )
        return alpha * np.sqrt(idio_variances)

    def _update_calibration(
        self,
        *,
        uncalibrated_alpha: FloatArray,
        forward_return: FloatArray,
        idio_variances: FloatArray,
        estimation_weights: FloatArray,
    ) -> None:
        """Update scalar EWLS statistics for return-unit calibration."""
        valid_var = np.isfinite(idio_variances)
        valid_estimation = np.asarray(estimation_weights) > 0
        valid = (
            valid_estimation
            & valid_var
            & np.isfinite(uncalibrated_alpha)
            & np.isfinite(forward_return)
        )
        weights = np.where(
            valid,
            estimation_weights * safe_divide(1.0, idio_variances, fill_value=0.0),
            0.0,
        )

        for alpha_t, target_t, weights_t, valid_t in zip(
            uncalibrated_alpha, forward_return, weights, valid, strict=True
        ):
            if np.count_nonzero(valid_t) < _MIN_VALID_ASSETS_FOR_CALIBRATION:
                continue

            alpha_valid = alpha_t[valid_t]
            target_valid = target_t[valid_t]
            weights_valid = weights_t[valid_t]

            weight_scale = float(np.mean(weights_valid))
            if np.isfinite(weight_scale) and weight_scale > 0:
                weights_valid = weights_valid / weight_scale

            obs_normal = float(np.dot(weights_valid, alpha_valid * alpha_valid))
            obs_cross = float(np.dot(weights_valid, alpha_valid * target_valid))
            if (
                not np.isfinite(obs_normal)
                or not np.isfinite(obs_cross)
                or obs_normal <= np.finfo(float).eps
            ):
                continue

            self._calibration_normal *= self._calibration_decay
            self._calibration_normal += (1.0 - self._calibration_decay) * obs_normal
            self._calibration_cross *= self._calibration_decay
            self._calibration_cross += (1.0 - self._calibration_decay) * obs_cross
            self._n_valid_calibration_obs += 1
            ridge = _CALIBRATION_RIDGE_SCALE * max(
                abs(self._calibration_normal), np.finfo(float).eps
            )
            self.calibration_coef_ = self._calibration_cross / (
                self._calibration_normal + ridge
            )

    def _set_alpha(self, uncalibrated_alpha: FloatArray) -> None:
        """Set `alpha_` from an uncalibrated alpha prediction."""
        if not self.calibrate_to_return_units:
            self.alpha_ = self.forecast_scale * uncalibrated_alpha
        elif np.isfinite(self.calibration_coef_):
            self.alpha_ = (
                self.forecast_scale * self.calibration_coef_ * uncalibrated_alpha
            )
        else:
            self.alpha_ = None

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        self._validate_common_params()
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_real(self.forecast_scale, "forecast_scale")
        _validate_bool(self.calibrate_to_return_units, "calibrate_to_return_units")
        if self.predictor is None:
            raise ValueError("predictor cannot be None")

    def _initialize(self) -> None:
        """Initialize internal state."""
        self._initialize_common_state()

        self.predictor_ = sk.clone(self.predictor)

        # Target transformers (user-configurable)
        self.target_outlier_transformer_ = check_estimator(
            self.target_outlier_transformer,
            default=CSWinsorizer(),
            check_type=BaseCSTransformer,
        )

        self.target_scoring_transformer_ = check_estimator(
            self.target_scoring_transformer,
            default=_PASSTHROUGH,  # Default: predict winsorized returns
            check_type=BaseCSTransformer,
        )

        self._initialize_buffers()
        self._predictor_fitted = False

        if self.calibrate_to_return_units:
            self._calibration_decay = half_life_to_decay_factor(self.half_life)
            self._calibration_normal = 0.0
            self._calibration_cross = 0.0
            self._n_valid_calibration_obs = 0
            self.calibration_coef_ = np.nan

    @property
    def _needs_idio_variances(self) -> bool:
        return (
            self.calibrate_to_return_units
            or self.forecast_unit is ForecastUnit.IDIO_SHARPE
        )
