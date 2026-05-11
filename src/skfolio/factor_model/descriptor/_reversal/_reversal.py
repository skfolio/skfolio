"""Fixed-window short-term reversal descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.factor_model.descriptor._base import _BaseRollingLogReturn
from skfolio.typing import FloatArray


class Reversal(_BaseRollingLogReturn):
    r"""Fixed-window short-term reversal descriptor.

    Computes the negated cumulative log return over a trailing window:

    .. math::

        \text{reversal}(t) = -\sum_{k=t-w+1}^{t} \log(1 + r_k)

    where :math:`w` is the `window` size and :math:`r_k` is the asset return at
    observation :math:`k`. High values indicate recent poor performance (reversal
    candidates).

    Short-term reversal captures mean reversion in returns driven by temporary price
    pressure, liquidity provision and microstructure effects [1]_ [2]_.

    The output is NaN until an asset has a full active lookback window. Active assets
    with missing returns contribute zero to the log-return sum. Non-missing `returns`
    values must be finite and greater than `-1`.

    The descriptor is returned in log-return space. Log cumulative returns are more
    symmetric than simple cumulative returns, which makes them better suited to
    cross-sectional standardization. Because the logarithm is monotonic, log-space and
    simple cumulative returns produce the same cross-sectional rankings when returns
    are finite and greater than `-1`.

    Parameters
    ----------
    window : int, default=21
        Number of trailing observations for the cumulative return.
        Common choices for daily data:

        - `window=1`: 1-day reversal
        - `window=5`: 1-week reversal
        - `window=21`: 1-month reversal (default)

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    reversal_ : ndarray of shape (n_assets,)
        Last short-term reversal value for each asset.

    Notes
    -----
    Two code paths are used depending on context:

    - Batch (first call with sufficient data): vectorized cumsum over
      the full panel. Time :math:`O(T \cdot n)`, space :math:`O(T \cdot n)`.

    - Online (subsequent calls or streaming): ring buffer of size :math:`w` (the window)
      with a running sum. Per observation: one subtract (value leaving the window), one
      add (current value), one write. Time :math:`O(n)` per step, space
      :math:`O(w \cdot n)`, zero allocation.

    After a batch computation, the ring buffer state is populated for subsequent online
    calls.

    References
    ----------
    .. [1] "Evidence of predictable behavior of security returns"
        The Journal of Finance. Jegadeesh, N. (1990).

    .. [2] "Fads, martingales, and market efficiency"
        The Quarterly Journal of Economics. Lehmann, B. N. (1990).

    See Also
    --------
    EWMomentum : Exponentially weighted momentum (medium/long-term).
    RollingMomentum : Fixed-window momentum with optional skip.

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import Reversal
    >>> # 1-month reversal (default)
    >>> descriptor = Reversal()
    >>> rev = descriptor.fit_transform(X)
    >>>
    >>> # 1-day reversal
    >>> descriptor = Reversal(window=1)
    >>> rev_1d = descriptor.fit_transform(X)
    >>>
    >>> # 1-week reversal
    >>> descriptor = Reversal(window=5)
    >>> rev_5d = descriptor.fit_transform(X)
    """

    _FITTED_ATTR = "reversal_"
    _TRANSFORM_SIGN = -1.0

    reversal_: FloatArray

    def __init__(self, window: int = 21):
        super().__init__(window=window, skip=0, exponentiate=False)
