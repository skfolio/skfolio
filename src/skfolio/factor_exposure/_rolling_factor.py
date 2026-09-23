"""Rolling-window factor exposures computed from a panel field."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pandas as pd

import skfolio.typing as skt
from skfolio._constants import _BENCHMARK_WEIGHTS, _PASSTHROUGH
from skfolio.containers import AssetPanel
from skfolio.factor_exposure._base import BaseFactorExposure
from skfolio.preprocessing import BaseCSTransformer, CSStandardScaler, CSWinsorizer
from skfolio.typing import BoolArray, FloatArray, ObjArray
from skfolio.utils.tools import _validate_positive_integer, check_estimator
from skfolio.utils.validation import validate_asset_panel

__all__ = ["RollingFactor"]

_FITTED_ATTR = "factor_names_"

_LAG = "lag"
_ROLLING_AGGREGATIONS = ("mean", "std", "min", "max", "median", "sum")
_AGGREGATIONS = (*_ROLLING_AGGREGATIONS, _LAG)


class RollingFactor(BaseFactorExposure):
    r"""Rolling-window factor exposures computed from a panel field.

    Computes a small library of trailing-window statistics (and lags) of a single
    numeric field and returns them as a multi-factor exposure tensor of shape
    `(n_observations, n_assets, n_factors)`, following the multi-output contract of
    :class:`~skfolio.factor_exposure.OneHotCategoricalFactors`. This replaces the
    repetitive pattern of one hand-written
    :class:`~skfolio.factor_exposure.DerivedFactor` per window and aggregation.

    For each observation :math:`t`, asset :math:`i`, aggregation :math:`f` and window
    :math:`w`, the raw exposure is

    .. math::

        x_{t,i,f,w} = f\left(z_{t-w+1,i}, \ldots, z_{t,i}\right)

    where :math:`z` is the source field. For the special aggregation `"lag"` with
    lag :math:`k`, the raw exposure is :math:`z_{t-k,i}`. Each output column is then
    optionally passed through the cross-sectional outlier and scoring transformers.

    Windows follow the as-of time-indexing convention: the window of size :math:`w`
    uses the :math:`w` observations ending at :math:`t` inclusive. Output is NaN until
    the asset has a full active lookback window (:math:`w` observations for a rolling
    statistic, :math:`k + 1` for a lag). A missing (NaN) source value inside the window
    propagates to the output, as in pandas.

    Parameters
    ----------
    source : str
        Name of the numeric 2D field in the :class:`~skfolio.containers.AssetPanel`
        to aggregate (for example `"returns"` or `"adj_volume"`).

    windows : dict[str, list[int]]
        Mapping from an aggregation name to the window sizes to compute it over.
        Supported aggregations are `"mean"`, `"std"`, `"min"`, `"max"`, `"median"`
        and `"sum"`, computed over trailing windows, plus `"lag"`, which returns the
        source value :math:`k` observations earlier. `"std"` uses one delta degree of
        freedom, as in pandas. Window sizes (and lags) must be positive integers.
        For example ``{"mean": [5, 21], "std": [21], "lag": [1]}`` produces the four
        factors `<source>_mean_5`, `<source>_mean_21`, `<source>_std_21` and
        `<source>_lag_1`, in that order.

    family : str, default="style"
        The factor family this exposure belongs to (e.g., "market", "style", "industry",
        "country"). Factor families group related factors for basket-neutral constraints,
        neutralization, attribution and reporting. The default is `"style"`.

    outlier_transformer : BaseCSTransformer or "passthrough" or None, default="passthrough"
        Cross-sectional transformer for outlier handling applied to each output column.
        If None, defaults to `CSWinsorizer()`. Use "passthrough" to skip.

    scoring_transformer : BaseCSTransformer or "passthrough", optional
        Cross-sectional transformer for scoring applied to each output column after
        outlier handling. If None, defaults to `CSStandardScaler()`. Use "passthrough"
        to skip and keep the raw rolling statistics.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations. If provided, outlier and scoring transformations are applied
        within each group separately.

    Attributes
    ----------
    factor_names_ : ndarray of shape (n_factors,)
        Names of the output columns, formatted as `<source>_<aggregation>_<window>`.

    outlier_transformer_ : BaseCSTransformer or str
        The fitted outlier transformer.

    scoring_transformer_ : BaseCSTransformer or str
        The fitted scoring transformer.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    Notes
    -----
    The estimator supports online computation. `fit_transform` starts from a clean
    state and `partial_fit_transform` continues from the current state: the trailing
    rows needed by the longest lookback are carried across calls, so chunked
    computation reproduces the full-batch result exactly, including at chunk
    boundaries.

    See Also
    --------
    DerivedFactor : Single factor derived from another factor's exposure.
    OneHotCategoricalFactors : Multi-factor exposures from a categorical field.

    Examples
    --------
    >>> from skfolio.descriptor import LogMarketCap
    >>> from skfolio.factor_exposure import FixedWeightedFactor, RollingFactor
    >>> from skfolio.prior import CharacteristicsFactorModel
    >>>
    >>> # Trailing mean and volatility of returns over one and three months, plus
    >>> # the one-period lag, as six style factors named "returns_mean_21", ...,
    >>> # "returns_lag_1"
    >>> rolling_returns = RollingFactor(
    ...     source="returns",
    ...     windows={"mean": [21, 63], "std": [21, 63], "lag": [1]},
    ...     transform_by_group="industry",
    ... )
    >>>
    >>> model = CharacteristicsFactorModel(
    ...     factors=[
    ...         ("size", FixedWeightedFactor(descriptors=[("log_mcap", LogMarketCap())])),
    ...         ("rolling_returns", rolling_returns),
    ...     ]
    ... )
    """

    factor_names_: ObjArray
    outlier_transformer_: BaseCSTransformer | str
    scoring_transformer_: BaseCSTransformer | str

    def __init__(
        self,
        *,
        source: str,
        windows: dict[str, list[int]],
        family: str = "style",
        outlier_transformer: skt.CSTransformer = "passthrough",
        scoring_transformer: skt.CSTransformer = None,
        transform_by_group: str | None = None,
    ):
        super().__init__(family=family)
        self.source = source
        self.windows = windows
        self.outlier_transformer = outlier_transformer
        self.scoring_transformer = scoring_transformer
        self.transform_by_group = transform_by_group

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute the rolling factor exposures from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the source field, benchmark weights and optional
            grouping field.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. They are ignored.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_assets, n_factors)
            Rolling factor exposures. Column order matches `factor_names_`.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update state and compute the rolling factor exposures.

        This method supports online updates by continuing from the current fitted
        state. Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the source field, benchmark weights and optional
            grouping field.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. They are ignored.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_assets, n_factors)
            Rolling factor exposures for the new observations. Column order matches
            `factor_names_`.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        required_fields = [self.source, _BENCHMARK_WEIGHTS]
        if self.transform_by_group is not None:
            required_fields.append(self.transform_by_group)

        validate_asset_panel(
            self,
            X,
            required_fields=required_fields,
            finite_or_nan=[self.source],
            reset=first_call,
        )

        field = X.fields[self.source]
        if field.is_categorical or field.is_3d:
            raise ValueError(
                f"Field '{self.source}' must be a numeric 2D field; "
                "categorical and 3D fields are not supported."
            )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets
        values = np.asarray(X[self.source], dtype=float)
        active_mask = X.active_mask

        # Prepend the rows carried over from previous calls so that windows spanning
        # a chunk boundary are computed on the full history.
        n_history = self._history_values.shape[0]
        values_ext = np.concatenate([self._history_values, values], axis=0)
        active_ext = np.concatenate([self._history_active_mask, active_mask], axis=0)

        exposures = np.empty((n_observations, n_assets, self.n_factors_), dtype=float)
        for j, (aggregation, window) in enumerate(self._specs):
            raw = _rolling_aggregate(values_ext, aggregation, window)
            lookback = window + 1 if aggregation == _LAG else window
            ready = _full_active_window(active_ext, lookback)
            exposures[:, :, j] = np.where(ready, raw, np.nan)[n_history:]

        # Carry over the trailing rows needed by the longest lookback
        keep = max(0, values_ext.shape[0] - (self._max_lookback - 1))
        self._history_values = values_ext[keep:].copy()
        self._history_active_mask = active_ext[keep:].copy()

        # Cross-sectional transformations, applied column by column
        cs_weight = X[_BENCHMARK_WEIGHTS]
        cs_group = (
            X[self.transform_by_group] if self.transform_by_group is not None else None
        )
        for j in range(self.n_factors_):
            column = exposures[:, :, j]
            if self.outlier_transformer_ != _PASSTHROUGH:
                column = self.outlier_transformer_.fit_transform(
                    column, cs_weights=cs_weight, cs_groups=cs_group
                )
            if self.scoring_transformer_ != _PASSTHROUGH:
                column = self.scoring_transformer_.fit_transform(
                    column, cs_weights=cs_weight, cs_groups=cs_group
                )
            exposures[:, :, j] = column

        # Mask inactive assets
        exposures[~active_mask] = np.nan

        return exposures

    def _reset(self) -> None:
        """Reset the fitted state so the next call behaves like a fresh fit."""
        for attr in (
            _FITTED_ATTR,
            "n_factors_",
            "outlier_transformer_",
            "scoring_transformer_",
            "_specs",
            "_max_lookback",
            "_history_values",
            "_history_active_mask",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if not isinstance(self.windows, dict) or len(self.windows) == 0:
            raise ValueError(
                "`windows` must be a non-empty dict mapping an aggregation name to a "
                f"list of window sizes, got {self.windows!r}"
            )
        for aggregation, sizes in self.windows.items():
            if aggregation not in _AGGREGATIONS:
                raise ValueError(
                    f"Unsupported aggregation {aggregation!r}. "
                    f"Supported aggregations are {list(_AGGREGATIONS)}"
                )
            if isinstance(sizes, str | bytes) or not hasattr(sizes, "__iter__"):
                raise ValueError(
                    f"`windows[{aggregation!r}]` must be a list of positive integers, "
                    f"got {sizes!r}"
                )
            sizes = list(sizes)
            if len(sizes) == 0:
                raise ValueError(
                    f"`windows[{aggregation!r}]` must contain at least one window size"
                )
            for size in sizes:
                _validate_positive_integer(size, f"windows[{aggregation!r}]")
            if len(set(sizes)) != len(sizes):
                raise ValueError(
                    f"`windows[{aggregation!r}]` contains duplicate window sizes: "
                    f"{sizes!r}"
                )

    def _initialize(self) -> None:
        """Initialize the output specification, transformers and history buffers."""
        self._specs = [
            (aggregation, int(size))
            for aggregation, sizes in self.windows.items()
            for size in sizes
        ]
        self.n_factors_ = len(self._specs)
        self.factor_names_ = np.array(
            [
                f"{self.source}_{aggregation}_{size}"
                for aggregation, size in self._specs
            ],
            dtype=object,
        )
        self._max_lookback = max(
            size + 1 if aggregation == _LAG else size
            for aggregation, size in self._specs
        )
        self._history_values = np.empty((0, self.n_assets_), dtype=float)
        self._history_active_mask = np.empty((0, self.n_assets_), dtype=bool)

        self.outlier_transformer_ = check_estimator(
            self.outlier_transformer,
            default=CSWinsorizer(),
            check_type=BaseCSTransformer,
        )
        self.scoring_transformer_ = check_estimator(
            self.scoring_transformer,
            default=CSStandardScaler(),
            check_type=BaseCSTransformer,
        )


def _rolling_aggregate(values: FloatArray, aggregation: str, window: int) -> FloatArray:
    """Apply a trailing-window aggregation (or lag) along axis 0.

    Parameters
    ----------
    values : ndarray of shape (n_observations, n_assets)
        Source values.

    aggregation : str
        One of the supported aggregation names.

    window : int
        Window size for a rolling statistic, or the number of periods for a lag.

    Returns
    -------
    result : ndarray of shape (n_observations, n_assets)
        Aggregated values. The first `window - 1` rows (`window` rows for a lag) and
        any window containing a NaN are NaN.
    """
    df = pd.DataFrame(values)
    if aggregation == _LAG:
        return df.shift(window).to_numpy(dtype=float)
    rolling = df.rolling(window=window, min_periods=window)
    return getattr(rolling, aggregation)().to_numpy(dtype=float)


def _full_active_window(active_mask: BoolArray, lookback: int) -> BoolArray:
    """Return True where the asset is active on all `lookback` observations ending
    at the current one.
    """
    n_observations, n_assets = active_mask.shape
    active_cumsum = np.zeros((n_observations + 1, n_assets), dtype=int)
    active_cumsum[1:] = np.cumsum(active_mask, axis=0)
    ready = np.zeros((n_observations, n_assets), dtype=bool)
    if n_observations >= lookback:
        end = np.arange(lookback, n_observations + 1)
        ready[lookback - 1 :] = (
            active_cumsum[end] - active_cumsum[end - lookback]
        ) == lookback
    return ready
