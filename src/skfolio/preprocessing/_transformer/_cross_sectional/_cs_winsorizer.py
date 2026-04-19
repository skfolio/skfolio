"""Cross-sectional winsorization."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv
from sklearn.utils.validation import FLOAT_DTYPES

from skfolio.preprocessing._transformer._cross_sectional._base import BaseCSTransformer
from skfolio.preprocessing._transformer._cross_sectional._utils import (
    _mask_non_estimation_values,
    _validate_cs_weights,
)

__all__ = ["CSWinsorizer"]


class CSWinsorizer(BaseCSTransformer):
    r"""Cross-sectional winsorization.

    Clips each finite value within an observation to the interval between the `low` and
    `high` percentiles of that observation's cross-section.

    NaNs are treated as missing values. They are ignored when computing cross-sectional
    percentiles and are preserved in the output.

    When `cs_weights` is provided, percentile boundaries are computed on the estimation
    universe, defined by `cs_weights > 0`. Assets outside the estimation universe still
    receive clipped values using those boundaries. For this estimator, `cs_weights` is
    used only to define the estimation universe; percentile estimation itself remains
    equal-weighted over the selected assets.

    This transformer is stateless.

    Parameters
    ----------
    low : float, default=0.01
        Lower percentile used for clipping.
        Must satisfy :math:`0 \le \text{low} < \text{high} \le 1`.

    high : float, default=0.99
        Upper percentile used for clipping.
        Must satisfy :math:`0 \le \text{low} < \text{high} \le 1`.

    See Also
    --------
    CSTanhShrinker : Smoothly shrinks extreme values.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.preprocessing import CSWinsorizer
    >>>
    >>> X = np.array([[1.0, np.nan, 3.0, 4.0],
    ...               [4.0, 3.0, 2.0, 1.0],
    ...               [10.0, 20.0, np.nan, 40.0]])
    >>>
    >>> transformer = CSWinsorizer(low=0.1, high=0.9)
    >>> transformer.fit_transform(X)
    array([[ 1.4,  nan,  3. ,  3.8],
           [ 3.7,  3. ,  2. ,  1.3],
           [12. , 20. ,  nan, 36. ]])
    >>>
    >>> # Use cs_weights for the estimation universe before computing the clip bounds.
    >>> cs_weights = np.array([[1.0, 0.0, 1.0, 1.0],
    ...                        [1.0, 0.0, 1.0, 1.0],
    ...                        [1.0, 1.0, 0.0, 1.0]])
    >>>
    >>> transformer.fit_transform(X, cs_weights=cs_weights)
    array([[ 1.4,  nan,  3. ,  3.8],
           [ 3.6,  3. ,  2. ,  1.2],
           [12. , 20. ,  nan, 36. ]])
    """

    def __init__(self, *, low: float = 0.01, high: float = 0.99):
        self.low = low
        self.high = high

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if not (0.0 <= self.low < self.high <= 1.0):
            raise ValueError(
                "`low` and `high` must satisfy 0.0 <= low < high <= 1.0; "
                f"got low={self.low}, high={self.high}."
            )

    def transform(
        self,
        X: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
        cs_groups: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        r"""Winsorize each observation to low/high percentiles.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.
            NaNs are allowed and preserved.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional non-negative cross-sectional weights used only to define the
            estimation universe through the convention `cs_weights > 0`. Percentile
            boundaries are then estimated in an equal-weighted way over the selected
            assets. Non-estimation assets still receive clipped values using those
            boundaries. If `None`, all finite assets are used to compute percentiles.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Not used, present for API consistency by convention.

        Returns
        -------
        X_clipped : ndarray of shape (n_observations, n_assets)
            Winsorized values. NaN values from the input are preserved.

        Raises
        ------
        ValueError
            If `low` / `high` are invalid, `X` is not a non-empty 2D array, `cs_weights`
            is invalid, or any observation has no estimation asset.
        """
        self._validate_params()
        X = skv.validate_data(
            self,
            X,
            reset=False,
            dtype=FLOAT_DTYPES,
            copy=True,
            ensure_all_finite="allow-nan",
        )
        cs_weights = _validate_cs_weights(X=X, cs_weights=cs_weights)

        X_estimation = _mask_non_estimation_values(X=X, cs_weights=cs_weights)

        q_lo, q_hi = np.nanpercentile(
            X_estimation, [self.low * 100, self.high * 100], axis=1, keepdims=True
        )

        # Numpy clip preserves NaNs from X
        np.clip(X, q_lo, q_hi, out=X)

        return X
