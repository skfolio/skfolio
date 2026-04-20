"""Cross-sectional rank Gaussianization."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt
import scipy.special as scs
import sklearn.utils.validation as skv
from sklearn.utils.validation import FLOAT_DTYPES

from skfolio.preprocessing._transformer._cross_sectional._base import BaseCSTransformer
from skfolio.preprocessing._transformer._cross_sectional._utils import (
    _cs_percentile_rank,
    _cs_recenter_rescale,
    _prepare_cs_estimation_inputs,
    _validate_cs_weights,
)

__all__ = ["CSGaussianRankScaler"]


class CSGaussianRankScaler(BaseCSTransformer):
    r"""Cross-sectional rank Gaussianization.

    Computes percentile ranks within each cross-section (see :class:`CSPercentileRankScaler`),
    maps them through the inverse standard normal CDF :math:`\Phi^{-1}`, and recenters
    to weighted mean zero over the estimation universe. When `scale=True`, the result is
    also rescaled to unit equal-weighted standard deviation.

    When `cs_weights` is provided, the estimation universe is defined by
    `cs_weights > 0`. Assets outside that universe still receive Gaussianized scores
    relative to it. `cs_weights` is used to define the estimation universe and to
    compute the final weighted recentering; ranking itself remains equal-weighted over
    the selected assets.

    NaNs are treated as missing values. They are ignored when computing cross-sectional
    ranks and are preserved in the output.

    For observation :math:`t`, the Gaussianized value of asset :math:`i` is:

    .. math::

        z_{t,i} = \frac{\Phi^{-1}(p_{t,i}) - \mu_t}{\sigma_t}

    where :math:`p_{t,i}` is the percentile rank, :math:`\mu_t` the weighted mean of
    :math:`\Phi^{-1}(p_{t,\cdot})` over the estimation universe, and
    :math:`\sigma_t` its unbiased equal-weighted standard deviation. When `scale=False`,
    the rescaling step is skipped and only weighted recentering is applied.

    When `cs_groups` is provided, the same scheme is applied within each group. Groups
    with fewer than `min_group_size` estimation assets, and missing groups
    (`cs_groups == -1`), fall back to the global cross-section. Recentering and
    rescaling are always performed over the full cross-section, not within groups.

    This transformer is stateless.

    Parameters
    ----------
    min_group_size : int, default=8
        Minimum number of estimation assets required in a group. Smaller groups fall
        back to the global cross-section.

    scale : bool, default=True
        If True, rescale final exposures to unit equal-weighted standard deviation over
        the estimation universe. If False, only weighted recentering is applied.
        Use this when feeding the output to a scale-invariant downstream model
        (e.g. gradient-boosted trees) and you want to avoid injecting per-cross-section
        noise from the unbiased standard-deviation estimate.

    atol : float, default=1e-12
        Absolute tolerance used to guard against division by a near-zero equal-weighted
        standard deviation. Must be finite and non-negative.

    See Also
    --------
    CSPercentileRankScaler
    CSStandardScaler

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.preprocessing import CSGaussianRankScaler
    >>>
    >>> X = np.array([[1.0, np.nan, 3.0, 4.0],
    ...               [4.0, 3.0, 2.0, 1.0],
    ...               [10.0, 20.0, np.nan, 40.0]])
    >>>
    >>> transformer = CSGaussianRankScaler()
    >>> transformer.fit_transform(X)
    array([[-1.        ,         nan,  0.        ,  1.        ],
           [ 1.180302  ,  0.32693605, -0.32693605, -1.180302  ],
           [-1.        ,  0.        ,         nan,  1.        ]])
    >>>
    >>> # Use cs_weights for the estimation universe and weighted recentering, and rank within groups.
    >>> cs_weights = np.array([[3.0, 0.0, 1.0, 2.0],
    ...                        [4.0, 0.0, 2.0, 3.0],
    ...                        [2.0, 3.0, 0.0, 5.0]])
    >>> cs_groups = np.array([[0, 0, 1, 1],
    ...                       [0, 0, 1, 1],
    ...                       [0, 0, 1, 1]])
    >>>
    >>> transformer = CSGaussianRankScaler(min_group_size=2)
    >>> transformer.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
    array([[-0.6791367 ,         nan, -0.34541391,  1.19141201],
           [ 0.69857792,  0.0863589 ,  0.36442412, -1.17438663],
           [-1.33305449,  0.13413753,         nan,  0.45273928]])
    """

    def __init__(
        self, *, min_group_size: int = 8, scale: bool = True, atol: float = 1e-12
    ):
        self.min_group_size = min_group_size
        self.scale = scale
        self.atol = atol

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
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError(f"`atol` must be >= 0; got {self.atol}.")

    def transform(
        self,
        X: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
        cs_groups: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        r"""Transform values into cross-sectional Gaussianized exposures.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.
            NaNs are allowed and preserved.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional non-negative cross-sectional weights. They define the estimation
            universe through `cs_weights > 0` and drive the final weighted recentering.
            Ranking itself remains equal-weighted over the selected assets. If `None`,
            all finite assets are included in the estimation universe.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Integer group labels >= -1. Missing groups (`-1`) and groups with fewer than
            `min_group_size` estimation assets fall back to the global cross-section.
            If `None`, ranking is performed on the full cross-section of each
            observation.

        Returns
        -------
        Z : ndarray of shape (n_observations, n_assets)
            Gaussianized exposures. Each cross-section has weighted mean zero over its
            estimation universe and, when `scale=True`, unit equal-weighted standard
            deviation. NaNs from `X` are preserved.

        Raises
        ------
        ValueError
            If `min_group_size` is not an integer `>= 1`, `atol` is not finite or `< 0`,
             `X` is not a non-empty 2D array, `cs_weights` is invalid, `cs_groups` is
             invalid, or any observation has no estimation asset.
        """
        self._validate_params()

        X = skv.validate_data(
            self, X, reset=False, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan"
        )
        cs_weights = _validate_cs_weights(X, cs_weights=cs_weights)
        cs_weights, finite_mask, estimation_mask = _prepare_cs_estimation_inputs(
            X, cs_weights=cs_weights
        )
        percentile = _cs_percentile_rank(
            X=X,
            finite_mask=finite_mask,
            estimation_mask=estimation_mask,
            cs_groups=cs_groups,
            min_group_size=self.min_group_size,
        )

        return _cs_recenter_rescale(
            X=scs.ndtri(percentile),
            finite_mask=finite_mask,
            cs_weights=cs_weights,
            estimation_mask=estimation_mask,
            atol=self.atol,
            scale=self.scale,
        )
