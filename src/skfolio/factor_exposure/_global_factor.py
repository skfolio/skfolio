"""Global factor exposure."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.factor_exposure._base import BaseFactorExposure
from skfolio.typing import FloatArray
from skfolio.utils.validation import validate_asset_panel

__all__ = ["GlobalFactor"]


class GlobalFactor(BaseFactorExposure, stateless=True):
    r"""Constant factor exposure equal to one for every asset.

    `GlobalFactor` represents an intercept or broad market factor in a characteristics
    factor model. It does not depend on any characteristic field. It uses the panel
    dimensions to return an exposure matrix of ones with shape
    `(n_observations, n_assets)`.

    For each observation :math:`t` and asset :math:`i`, the exposure is:

    .. math::

        x_{t,i} = 1

    This factor is typically used when the cross-sectional regression should estimate
    a common return component in addition to characteristic-based style industry or
    country factors.

    Parameters
    ----------
    family : str, default="market"
        The factor family this exposure belongs to. Factor families group related
        factors for basket-neutral constraints, neutralization, attribution and
        reporting. The default is `"market"`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    Examples
    --------
    >>> from skfolio.factor_exposure import GlobalFactor
    >>> from skfolio.prior import CharacteristicsFactorModel
    >>>
    >>> model = CharacteristicsFactorModel(
    ...     factors=[("market", GlobalFactor())]
    ... )
    """

    def __init__(self, *, family: str = "market") -> None:
        super().__init__(family=family)

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Return a constant exposure matrix.

        Parameters
        ----------
        X : AssetPanel
            Input panel used to determine the number of observations and assets.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. They are ignored.

        Returns
        -------
        exposure : ndarray of shape (n_observations, n_assets)
            Constant factor exposure equal to one for every asset in every observation.
        """
        validate_asset_panel(self, X)
        return np.ones((X.n_observations, X.n_assets))
