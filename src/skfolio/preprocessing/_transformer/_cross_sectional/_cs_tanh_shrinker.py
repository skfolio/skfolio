"""Cross-sectional tanh outlier shrinker."""

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
    _MAD_CONSISTENCY,
    _mask_non_estimation_values,
    _validate_cs_weights,
)

__all__ = ["CSTanhShrinker"]


class CSTanhShrinker(BaseCSTransformer):
    r"""Cross-sectional tanh outlier shrinker.

    Smoothly shrinks extreme values within an observation toward the cross-sectional
    center while preserving the original scale and units of the input values. Values
    near the center are left nearly unchanged, while extreme values are compressed
    inward.

    NaNs are treated as missing values. They are ignored when computing the
    cross-sectional median and MAD and are preserved in the output.

    Compared to winsorization (:class:`CSWinsorizer`):

    * No hard threshold. The mapping is smooth, so small data changes do not create
      discontinuous jumps at a clipping boundary.
    * Strict monotonicity. Tail ordering is preserved because distinct inputs remain
      distinct after transformation.
    * Smooth transformed values. This can lead to better-conditioned cross-sectional
      regressions and more stable coefficient estimates in downstream models.

    For observation :math:`t` with cross-section :math:`\mathbf{x}_t`, the
    transformation is

    .. math::

        x_{t,i}' = m_t + h_t \cdot \tanh\!\left(\frac{x_{t,i} - m_t}{h_t}\right),
        \quad
        h_t = c \cdot s_t

    where :math:`m_t = \operatorname{median}(\mathbf{x}_t)`,
    :math:`s_t = 1.4826 \cdot \operatorname{MAD}(\mathbf{x}_t)` is a robust scale
    estimator consistent for the standard deviation under normality, and :math:`c` is
    the knee parameter (see `knee`). The quantity :math:`h_t = c \cdot s_t` is the half-width
    of the near-linear region for observation :math:`t`.

    When `cs_weights` is provided, median and MAD are computed from the estimation
    universe, defined by `cs_weights > 0`. Assets outside the estimation universe still
    receive shrunk values using those statistics. For this estimator, `cs_weights` is
    used only to define the estimation universe; the median and MAD remain
    equal-weighted over the selected assets.

    This transformer is stateless.

    Parameters
    ----------
    knee : float, default=3.0
        Compression knee in robust standard deviations. It controls the width of the
        near-linear region around the median. Larger values reduce shrinkage. Must be
        finite and strictly positive.

    atol : float, default=1e-12
        Absolute tolerance for the robust scale. If :math:`s` is below `atol`, the
        observation is returned unchanged. Must be finite and non-negative.

    See Also
    --------
    CSWinsorizer : Hard percentile-based clipping.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.preprocessing import CSTanhShrinker
    >>>
    >>> X = np.array([[1.0, np.nan, 3.0, 4.0],
    ...               [4.0, 3.0, 2.0, 1.0],
    ...               [10.0, 20.0, np.nan, 40.0]])
    >>>
    >>> transformer = CSTanhShrinker()
    >>> transformer.fit_transform(X)
    array([[ 1.12471866,         nan,  3.        ,  3.98348436],
           [ 3.94560619,  2.99790441,  2.00209559,  1.05439381],
           [10.16515641, 20.        ,         nan, 38.75281341]])
    >>>
    >>> # Use cs_weights for the estimation universe before computing the median and MAD.
    >>> cs_weights = np.array([[1.0, 0.0, 1.0, 1.0],
    ...                        [1.0, 0.0, 1.0, 1.0],
    ...                        [1.0, 1.0, 0.0, 1.0]])
    >>>
    >>> transformer.fit_transform(X, cs_weights=cs_weights)
    array([[ 1.12471866,         nan,  3.        ,  3.98348436],
           [ 3.87528134,  2.98348436,  2.        ,  1.01651564],
           [10.16515641, 20.        ,         nan, 38.75281341]])
    """

    def __init__(
        self,
        *,
        knee: float = 3.0,
        atol: float = 1e-12,
    ):
        self.knee = knee
        self.atol = atol

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if not np.isfinite(self.knee) or self.knee <= 0.0:
            raise ValueError(f"`knee` must be strictly positive; got {self.knee}.")
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError(f"`atol` must be >= 0; got {self.atol}.")

    def transform(
        self,
        X: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
        cs_groups: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        r"""Shrink outliers within each observation using a tanh mapping.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.
            NaNs are allowed and preserved.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional non-negative cross-sectional weights used only to define the
            estimation universe through the convention `cs_weights > 0`. The median and
            MAD are then computed in an equal-weighted way over the selected assets.
            Non-estimation assets still receive shrunk values using those statistics.
            If `None`, all finite assets are used.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Not used, present for API consistency by convention.

        Returns
        -------
        X_shrunk : ndarray of shape (n_observations, n_assets)
            Shrunk values in the original scale. NaN values from the input are preserved.

        Raises
        ------
        ValueError
            If `knee` is not finite or `<= 0`, `atol` is not finite or `< 0`, `X` is not
            a non-empty 2D array, `cs_weights` is invalid, or any observation has no
            estimation asset.
        """
        self._validate_params()
        X = skv.validate_data(
            self, X, reset=False, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan"
        )
        cs_weights = _validate_cs_weights(X=X, cs_weights=cs_weights)

        X_estimation = _mask_non_estimation_values(X=X, cs_weights=cs_weights)

        median = np.nanmedian(X_estimation, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(X_estimation - median), axis=1, keepdims=True)

        scale = _MAD_CONSISTENCY * mad
        half_width = scale * self.knee
        shrinkable = scale > self.atol

        # On non-shrinkable rows (effectively constant), substitute a dummy scale to
        # avoid 0/0 and then discards it with np.where.
        # NaNs in X propagate through tanh and are preserved.
        safe_hw = np.where(shrinkable, half_width, 1.0)
        z = (X - median) / safe_hw
        return np.where(shrinkable, median + half_width * np.tanh(z), X)
