"""Capex-to-assets change-in-intensity descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.factor_model.descriptor._growth._base._change_in_intensity import (
    ChangeInIntensity,
)


class CapexToAssetsChangeInIntensity(ChangeInIntensity):
    r"""Lagged change in capex-to-assets intensity.

    Computes the change in the capex-to-assets ratio over a fixed lag:

    .. math::

        \text{capex\_to\_assets\_change\_in\_intensity}(t)
        = \frac{\text{capex\_ttm}(t)}{\text{total\_assets}(t)}
        - \frac{\text{capex\_ttm}(t - \text{lag})}
               {\text{total\_assets}(t - \text{lag})}

    The first `lag` observations are NaN because no lagged history is available.

    NaNs are allowed as missing observations and propagate when the current, lagged or
    scale value is missing. Non-missing `capex_ttm` values must be finite.
    Non-missing `total_assets` values must be finite and strictly positive.

    A positive value indicates that capex intensity increased relative to total assets
    and a negative value indicates that it decreased.

    This is a convenience subclass of :class:`ChangeInIntensity` with
    `field="capex_ttm"` and `scale_field="total_assets"`.

    Parameters
    ----------
    lag : int, default=252
        Number of observations to look back. The interpretation depends on the data
        frequency: `lag=12` means 1 year for monthly data, `lag=252` for daily data,
        `lag=4` for quarterly data.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    change_in_intensity_ : ndarray of shape (n_assets,)
        Last capex-to-assets intensity change for each asset.

    See Also
    --------
    ChangeInIntensity : Generic field-to-scale intensity change descriptor.
    GrowthRate : Period-over-period growth rate for non-negative fields.

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import CapexToAssetsChangeInIntensity
    >>> descriptor = CapexToAssetsChangeInIntensity(lag=252)
    >>> capex_intensity_change = descriptor.fit_transform(X)
    """

    def __init__(self, lag: int = 252):
        super().__init__(field="capex_ttm", scale_field="total_assets", lag=lag)
