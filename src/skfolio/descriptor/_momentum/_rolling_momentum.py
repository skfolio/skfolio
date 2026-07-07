"""Fixed-window (rolling) momentum descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.descriptor._base import _BaseRollingLogReturn
from skfolio.typing import FloatArray


class RollingMomentum(_BaseRollingLogReturn):
    r"""Fixed-window momentum descriptor.

    Computes the sum of log returns over a trailing window with an optional skip period
    to exclude the most recent observations:

    The skip period separates medium-term momentum from short-term reversal. The classic
    "12-1" momentum signal uses a skip of approximately one month [1]_.

    .. math::

        x(k) = \log(1 + r(k))

        S(t) = \sum_{k=t-\text{skip}-\text{window}+1}^{t-\text{skip}} x(k)

        \text{momentum}(t) =
        \begin{cases}
            \exp(S(t)) - 1 & \text{if exponentiate} \\
            S(t) & \text{otherwise}
        \end{cases}

    The window uses the :math:`\text{window}` observations ending at
    :math:`t - \text{skip}`. Output is NaN until the asset has a full active lookback
    window.

    By default, the descriptor is returned in log-return space. Log cumulative returns
    are more symmetric than simple cumulative returns, which makes them better suited to
    cross-sectional standardization. Because the logarithm is monotonic, log-space and
    simple cumulative returns produce the same cross-sectional rankings when returns
    are finite and greater than `-1`.

    Parameters
    ----------
    window : int, default=252
        Number of observations in the lookback window.

    skip : int, default=21
        Number of most recent observations excluded from the window. The last
        observation included is at :math:`t - \text{skip}`. Classic 12-1 momentum uses a
        skip of about one month (21 daily obs). Set to 0 for no skip.

    exponentiate : bool, default=False
        If True, output is :math:`\exp(S(t)) - 1` (return units). If False, output is
        :math:`S(t)` (log space). Cross-sectional ranking is unchanged and only the 
        scale differs.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    momentum_ : ndarray of shape (n_assets,)
        Last rolling momentum value for each asset.

    Notes
    -----
    Two code paths are used depending on context:

    - Batch (first call with sufficient data): vectorized cumsum over
      the full panel. Time :math:`O(T \cdot n)`, space :math:`O(T \cdot n)`.

    - Online (subsequent calls or streaming): ring buffer of size
      :math:`L = \text{skip} + \text{window}` with a running sum. Per observation: one
      subtract (value leaving the window), one add (value entering), one write.
      Time :math:`O(n)` per step, space :math:`O(L \cdot n)`, zero allocation.

    After a batch computation, the ring buffer state is populated for subsequent online
    calls.

    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite and greater than `-1`, so :math:`\log(1 + r)` is finite. Active assets with
    NaN returns (e.g. holidays) contribute 0 to the sum. Inactive asset outputs are
    set to NaN.

    References
    ----------
    .. [1] "Returns to buying winners and selling losers: Implications for stock market
       efficiency" The Journal of Finance. Jegadeesh, N., & Titman, S. (1993).

    See Also
    --------
    EWMomentum : Exponentially weighted momentum.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import RollingMomentum
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # 12-1 momentum
    >>> descriptor = RollingMomentum(window=252, skip=21)
    >>> momentum = descriptor.fit_transform(X)
    >>>
    >>> # Log-space output
    >>> descriptor = RollingMomentum(window=252, skip=21, exponentiate=False)
    >>> momentum_log = descriptor.fit_transform(X)
    """

    _FITTED_ATTR = "momentum_"
    _TRANSFORM_SIGN = 1.0

    momentum_: FloatArray

    def __init__(self, window: int = 252, skip: int = 21, exponentiate: bool = False):
        super().__init__(window=window, skip=skip, exponentiate=exponentiate)
