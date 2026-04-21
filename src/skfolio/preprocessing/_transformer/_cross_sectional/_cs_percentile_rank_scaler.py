"""Cross-sectional percentile rank."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers

import sklearn.utils.validation as skv
from sklearn.utils.validation import FLOAT_DTYPES

from skfolio.preprocessing._transformer._cross_sectional._base import BaseCSTransformer
from skfolio.preprocessing._transformer._cross_sectional._utils import (
    _cs_percentile_rank,
    _prepare_cs_estimation_inputs,
    _validate_cs_weights,
)
from skfolio.typing import ArrayLike, FloatArray

__all__ = ["CSPercentileRankScaler"]


class CSPercentileRankScaler(BaseCSTransformer):
    r"""Cross-sectional percentile rank.

    Computes the percentile rank of each finite value within an observation's
    cross-section.

    When `cs_weights` is provided, percentile ranks are estimated only on the estimation
    universe, defined by `cs_weights > 0`. Assets outside that universe still receive
    percentile ranks relative to it. For this estimator, `cs_weights` is used only to
    define the estimation universe; percentile estimation itself remains equal-weighted
    over the selected assets.

    NaNs are treated as missing values. They are ignored when computing cross-sectional
    ranks and are preserved in the output.

    When `cs_groups` is `None`, ranks are computed globally within each observation
    using the formula:

    .. math::

        p_{t,i} = \frac{r_{t,i} - 0.5}{N_{\mathcal{E}_t}}

    where :math:`\mathcal{E}_t` is the estimation universe at observation :math:`t`,
    :math:`N_{\mathcal{E}_t}` its size, and :math:`r_{t,i} \in
    [1, N_{\mathcal{E}_t}]` is the rank of asset :math:`i` within that universe. Tied
    values share the average of the ranks they would otherwise occupy (equivalent to
    `scipy.stats.rankdata(method="average")`).

    The :math:`-0.5` shift centers the rank inside its bin, so percentiles sit strictly
    in :math:`(0, 1)`, on the closed interval
    :math:`[0.5 / N_{\mathcal{E}_t},\, 1 - 0.5 / N_{\mathcal{E}_t}]`. This keeps
    downstream inverse-normal mappings always finite.

    When `cs_groups` is provided, the same ranking scheme is applied within each group.
    Groups with fewer than `min_group_size` estimation assets, and missing groups
    (`cs_groups == -1`), fall back to the global cross-section.

    This transformer is stateless.

    Parameters
    ----------
    min_group_size : int, default=8
        Minimum number of estimation assets required in a group. Smaller groups fall
        back to the global cross-section.

    See Also
    --------
    CSGaussianRankScaler
    CSStandardScaler

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.preprocessing import CSPercentileRankScaler
    >>>
    >>> X = np.array([[1.0, np.nan, 3.0, 4.0],
    ...               [4.0, 3.0, 2.0, 1.0],
    ...               [10.0, 20.0, np.nan, 40.0]])
    >>>
    >>> transformer = CSPercentileRankScaler()
    >>> transformer.fit_transform(X)
    array([[0.16666667,        nan, 0.5       , 0.83333333],
           [0.875     , 0.625     , 0.375     , 0.125     ],
           [0.16666667, 0.5       ,        nan, 0.83333333]])
    >>>
    >>> # Restrict the estimation universe with cs_weights and rank within groups.
    >>> cs_weights = np.array([[1.0, 0.0, 1.0, 1.0],
    ...                        [1.0, 0.0, 1.0, 1.0],
    ...                        [1.0, 1.0, 0.0, 1.0]])
    >>> cs_groups = np.array([[0, 0, 1, 1],
    ...                       [0, 0, 1, 1],
    ...                       [0, 0, 1, 1]])
    >>>
    >>> transformer = CSPercentileRankScaler(min_group_size=2)
    >>> transformer.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
    array([[0.16666667,        nan, 0.25      , 0.75      ],
           [0.83333333, 0.66666667, 0.75      , 0.25      ],
           [0.25      , 0.75      ,        nan, 0.83333333]])
    """

    def __init__(self, *, min_group_size: int = 8):
        self.min_group_size = min_group_size

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if (
            isinstance(self.min_group_size, bool)
            or not isinstance(self.min_group_size, numbers.Integral)
            or self.min_group_size < 1
        ):
            raise ValueError(
                f"`min_group_size` must be an integer >= 1; got {self.min_group_size}."
            )

    def transform(
        self,
        X: ArrayLike,
        cs_weights: ArrayLike | None = None,
        cs_groups: ArrayLike | None = None,
    ) -> FloatArray:
        r"""Transform values into cross-sectional percentile ranks.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.
            NaNs are allowed and preserved.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional non-negative cross-sectional weights used only to define the
            estimation universe through the convention `cs_weights > 0`. Percentile
            ranks are then estimated in an equal-weighted way over the selected
            assets. Non-estimation assets still receive percentile ranks relative to
            that universe. If `None`, all finite assets are included in the
            estimation universe.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Integer group labels >= -1. Missing groups (`-1`) and groups with fewer
            than `min_group_size` estimation assets fall back to the global
            cross-section. If `None`, ranking is performed globally within each
            observation.

        Returns
        -------
        P : ndarray of shape (n_observations, n_assets)
            Percentile ranks in
            :math:`[0.5 / N_{\mathcal{E}_t},\, 1 - 0.5 / N_{\mathcal{E}_t}]`, where
            :math:`N_{\mathcal{E}_t}` is the size of the cross-section or group fallback
            used at observation :math:`t`. NaNs from `X` are preserved.

        Raises
        ------
        ValueError
            If `min_group_size` is not an integer `>= 1`, `X` is not a non-empty 2D
            array, `cs_weights` is invalid, `cs_groups` is invalid, or any observation
            has no estimation asset.
        """
        self._validate_params()

        X = skv.validate_data(
            self, X, reset=False, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan"
        )
        cs_weights = _validate_cs_weights(X, cs_weights=cs_weights)
        _, finite_mask, estimation_mask = _prepare_cs_estimation_inputs(
            X, cs_weights=cs_weights
        )

        return _cs_percentile_rank(
            X=X,
            finite_mask=finite_mask,
            estimation_mask=estimation_mask,
            cs_groups=cs_groups,
            min_group_size=self.min_group_size,
        )
