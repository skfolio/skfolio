"""Cross-sectional standardization."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn.utils.validation as skv
from sklearn.utils.validation import FLOAT_DTYPES

from skfolio.preprocessing._transformer._cross_sectional._base import BaseCSTransformer
from skfolio.preprocessing._transformer._cross_sectional._utils import (
    _cs_equal_weighted_std,
    _cs_group_keys,
    _cs_recenter_rescale,
    _cs_weighted_mean,
    _prepare_cs_estimation_inputs,
    _safe_divide,
    _validate_and_normalize_groups,
    _validate_cs_weights,
)
from skfolio.typing import ArrayLike, FloatArray

__all__ = ["CSStandardScaler"]


class CSStandardScaler(BaseCSTransformer):
    r"""Cross-sectional standardization.

    Standardizes each finite value within an observation's cross-section to have
    weighted mean zero and unit equal-weighted standard deviation over the estimation
    universe.

    When `cs_weights` is provided, weighted means and unbiased equal-weighted standard
    deviations are estimated only on the estimation universe, defined by
    `cs_weights > 0`. Assets outside that universe still receive standardized values
    relative to the estimation universe. For this estimator, `cs_weights` is used to
    define the estimation universe and to compute the cross-sectional mean, while the
    standard deviation remains equal-weighted over the selected assets.

    NaNs are treated as missing values. They are ignored when computing cross-sectional
    statistics and are preserved in the output.

    When `cs_groups` is `None`, standardization is performed globally within each
    observation. For observation :math:`t`, the standardized value :math:`z_{t,i}` is
    defined by:

    .. math::

        z_{t,i} = \frac{x_{t,i} - \mu_t}{\sigma_t}

    where :math:`\mu_t` is the weighted mean,
    :math:`\sigma_t` is the unbiased equal-weighted standard deviation,
    :math:`\mathcal{E}_t` is the estimation universe, and
    :math:`N_{\mathcal{E}_t}` is its number of assets:

    .. math::

        \mu_t = \frac{\sum_{i \in \mathcal{E}_t} w_{t,i} x_{t,i}}
                     {\sum_{i \in \mathcal{E}_t} w_{t,i}},
        \quad
        \sigma_t = \sqrt{\frac{1}{N_{\mathcal{E}_t} - 1}
                   \sum_{i \in \mathcal{E}_t} (x_{t,i} - \mu_t)^2}

    When `cs_groups` is provided, the same centering and scaling scheme is first applied
    within each group. Groups with fewer than `min_group_size` estimation assets, and
    missing groups (`cs_groups == -1`), fall back to global cross-sectional statistics.
    The grouped result is then globally recentered to weighted mean zero and globally
    rescaled to unit equal-weighted standard deviation over the estimation universe.

    This transformer is stateless.

    Parameters
    ----------
    min_group_size : int, default=8
        Minimum number of estimation assets required in a group. Smaller groups fall
        back to global cross-sectional statistics.

    atol : float, default=1e-12
        Absolute tolerance below which the cross-sectional standard deviation is treated
        as zero. When `cs_groups` is `None`, this means that the observation has no
        measurable cross-sectional dispersion on its estimation universe, so finite
        outputs are set to zero rather than `NaN` and the row is treated as a neutral
        exposure. When `cs_groups` is provided, the same convention applies to the
        within-group standardization step and to the final global rescaling step.

    See Also
    --------
    CSPercentileRankScaler
    CSGaussianRankScaler

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.preprocessing import CSStandardScaler
    >>>
    >>> X = np.array([[1.0, np.nan, 3.0, 4.0],
    ...               [4.0, 3.0, 2.0, 1.0],
    ...               [10.0, 20.0, np.nan, 40.0]])
    >>>
    >>> transformer = CSStandardScaler()
    >>> transformer.fit_transform(X)
    array([[-1.09108945,         nan,  0.21821789,  0.87287156],
           [ 1.161895  ,  0.38729833, -0.38729833, -1.161895  ],
           [-0.87287156, -0.21821789,         nan,  1.09108945]])
    >>>
    >>> # Use cs_weights for the estimation universe and weighted means, then standardize within groups.
    >>> cs_weights = np.array([[3.0, 0.0, 1.0, 2.0],
    ...                        [4.0, 0.0, 2.0, 3.0],
    ...                        [2.0, 3.0, 0.0, 5.0]])
    >>> cs_groups = np.array([[0, 0, 1, 1],
    ...                       [0, 0, 1, 1],
    ...                       [0, 0, 1, 1]])
    >>>
    >>> transformer = CSStandardScaler(min_group_size=2)
    >>> transformer.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
    array([[-0.55454325,         nan, -0.62182063,  1.1427252 ],
           [ 0.62254586, -0.15324206,  0.5035012 , -1.16572861],
           [-1.33736075,  0.20821245,         nan,  0.41001683]])
    """

    def __init__(self, *, min_group_size: int = 8, atol: float = 1e-12):
        self.min_group_size = min_group_size
        self.atol = atol

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if self.min_group_size < 1:
            raise ValueError(
                f"`min_group_size` must be >= 1; got {self.min_group_size}."
            )
        if self.atol < 0.0:
            raise ValueError(f"`atol` must be >= 0; got {self.atol}.")

    def transform(
        self,
        X: ArrayLike,
        cs_weights: ArrayLike | None = None,
        cs_groups: ArrayLike | None = None,
    ) -> FloatArray:
        r"""Standardize each observation into cross-sectional z-scores.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row represents an observation and
            each column represents an asset. NaNs are allowed and
            preserved.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional non-negative cross-sectional weights. Positive
            weights define the estimation universe and are used to
            compute weighted means. The standard deviation remains
            equal-weighted over the selected assets. If `None`, all
            finite assets are included in the estimation universe with
            unit weight.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Integer group labels >= -1. Missing groups (`-1`) and
            groups with fewer than `min_group_size` estimation assets
            fall back to global cross-sectional statistics. If `None`,
            standardization is performed globally within each
            observation.

        Returns
        -------
        Z : ndarray of shape (n_observations, n_assets)
            Standardized values with weighted mean zero and unit
            equal-weighted standard deviation over the estimation
            universe.

        Raises
        ------
        ValueError
            If `min_group_size < 1`, `atol < 0`, `X` is not a non-empty
            2D array, `cs_weights` is invalid, `cs_groups` is invalid,
            or any observation has no estimation asset.
        """
        self._validate_params()
        X = skv.validate_data(
            self, X, reset=False, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan"
        )
        cs_weights = _validate_cs_weights(X, cs_weights=cs_weights)
        cs_weights, finite_mask, estimation_mask = _prepare_cs_estimation_inputs(
            X, cs_weights=cs_weights
        )

        # Single weighted mean and equal-weighted std per observations.
        if cs_groups is None:
            mean_global = _cs_weighted_mean(X, cs_weights, estimation_mask)
            std_global = _cs_equal_weighted_std(
                X, mean=mean_global, estimation_mask=estimation_mask
            )
            Z = np.zeros_like(X, dtype=np.float64)
            np.divide(
                X - mean_global,
                std_global,
                out=Z,
                where=finite_mask & (std_global > self.atol),
            )
            return np.where(finite_mask, Z, np.nan)

        # bincount requires an explicit weight array
        bincount_weights = (
            finite_mask.astype(np.float64) if cs_weights is None else cs_weights
        )

        # Encode each (row, group) pair as a single integer group key so bincount
        # can aggregate per row-group in one pass.
        group_ids, missing_group_mask, n_groups = _validate_and_normalize_groups(
            X=X, cs_groups=cs_groups
        )
        group_keys = _cs_group_keys(group_ids=group_ids, n_groups=n_groups)
        estimation_group_mask = estimation_mask & ~missing_group_mask
        group_key_flat = group_keys.ravel()
        n_group_keys = n_groups * X.shape[0]
        bincount_weights_flat = bincount_weights.ravel()
        estimation_group_flat = estimation_group_mask.ravel()

        # Weighted group sums on the estimation universe. `np.multiply(out=, where=)`
        # avoids materializing the full `cs_weights * X` product and dodges any
        # `0 * NaN = NaN` artifact at non-estimation cells.
        weighted_X_contrib = np.zeros_like(bincount_weights_flat)
        np.multiply(
            bincount_weights_flat,
            X.ravel(),
            out=weighted_X_contrib,
            where=estimation_group_flat,
        )
        weighted_X_sum = np.bincount(
            group_key_flat,
            weights=weighted_X_contrib,
            minlength=n_group_keys,
        )
        weight_sum = np.bincount(
            group_key_flat,
            weights=np.where(estimation_group_flat, bincount_weights_flat, 0.0),
            minlength=n_group_keys,
        )
        mean_group = _safe_divide(weighted_X_sum, weight_sum)
        mean_per_cell = mean_group[group_keys]

        # Unbiased equal-weighted group std on the estimation universe.
        residuals_flat = np.where(estimation_group_mask, X - mean_per_cell, 0.0).ravel()
        sum_squared_residuals = np.bincount(
            group_key_flat,
            weights=residuals_flat * residuals_flat,
            minlength=n_group_keys,
        )
        group_counts = np.bincount(
            group_key_flat[estimation_group_flat],
            minlength=n_group_keys,
        )

        # Clip ddof so empty or single-element group keys do not produce a negative
        # denominator under `_safe_divide`.
        ddof_group = np.maximum(group_counts - 1.0, 0.0)
        std_group = np.sqrt(_safe_divide(sum_squared_residuals, ddof_group))
        std_per_cell = std_group[group_keys]

        # Decide fallback at (row, group) level and broadcast once.
        small_group = group_counts < self.min_group_size
        fallback_mask = small_group[group_keys] | missing_group_mask
        if np.any(fallback_mask):
            mean_global = _cs_weighted_mean(X, cs_weights, estimation_mask)
            std_global = _cs_equal_weighted_std(X, mean_global, estimation_mask)
            mean_per_cell = np.where(fallback_mask, mean_global, mean_per_cell)
            std_per_cell = np.where(fallback_mask, std_global, std_per_cell)

        group_standardized = np.zeros_like(X, dtype=np.float64)
        np.divide(
            X - mean_per_cell,
            std_per_cell,
            out=group_standardized,
            where=finite_mask & (std_per_cell > self.atol),
        )
        group_standardized = np.where(finite_mask, group_standardized, np.nan)

        # Renormalize the grouped result so the full cross-section has weighted mean
        # zero and unit equal-weighted std.
        return _cs_recenter_rescale(
            X=group_standardized,
            finite_mask=finite_mask,
            cs_weights=cs_weights,
            estimation_mask=estimation_mask,
            atol=self.atol,
            scale=True,
        )
