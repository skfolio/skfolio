"""One-hot categorical factor exposure."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import MISSING_CATEGORY_CODE, AssetPanel
from skfolio.factor_exposure._base import BaseFactorExposure
from skfolio.typing import FloatArray, ObjArray
from skfolio.utils.validation import validate_asset_panel


class OneHotCategoricalFactors(BaseFactorExposure, stateless=True):
    r"""One-hot factor exposures from a categorical field.

    Expands a categorical field into one factor per category level. The result is an
    exposure tensor with shape `(n_observations, n_assets, n_factors)`, where
    `n_factors` is the number of category levels.

    For each observation :math:`t`, asset :math:`i` and category factor :math:`k`, the
    exposure is:

    .. math::

        x_{t,i,k} =
        \begin{cases}
        1 & \text{if asset } i \text{ belongs to category } k \\
        0 & \text{otherwise}
        \end{cases}

    Missing category codes produce NaN exposures for all category factors of that
    asset-observation pair.

    Parameters
    ----------
    category : str
        Name of the categorical field in the AssetPanel to one-hot encode. The field
        must be a `FieldCategorical`.

    family : str
        The factor family this exposure belongs to (e.g., "industry", "country").
        Factor families group related factors for basket-neutral constraints,
        neutralization, attribution and reporting.

    Attributes
    ----------
    factor_names_ : ndarray of shape (n_factors,)
        The category labels corresponding to each one-hot column.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.
    """

    factor_names_: ObjArray

    def __init__(self, category: str, *, family: str) -> None:
        super().__init__(family=family)
        self.category = category

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """One-hot encode the categorical field.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the categorical field as integer codes.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. They are ignored.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_assets, n_factors)
            One-hot encoded exposures. Column order matches `X.fields[category].levels`.
            Entries with missing codes (MISSING_CATEGORY_CODE == -1) are filled with
            NaN.

        Raises
        ------
        IndexError
            If any valid code is >= n_levels (indicates data corruption).
        """
        validate_asset_panel(self, X, required_fields=[self.category])

        try:
            field = X.fields[self.category]
        except KeyError as err:
            raise ValueError(
                f"Field '{self.category}' is not in the AssetPanel."
            ) from err

        if not field.is_categorical:
            raise ValueError(f"Field '{self.category}' must be a CategoricalField.")

        codes = field.values
        factor_names = field.levels
        n_factors = len(factor_names)
        n_observations, n_assets = codes.shape

        # Flatten for vectorized one-hot encoding
        flat_codes = codes.ravel()

        # Identify missing values
        valid_mask = flat_codes != MISSING_CATEGORY_CODE

        # FieldCategorical validates that valid codes are in [0, n_factors)
        exposures = np.full((flat_codes.size, n_factors), np.nan, dtype=np.float64)
        valid_indices = np.nonzero(valid_mask)[0]
        valid_codes = flat_codes[valid_mask]
        exposures[valid_indices, :] = 0.0
        exposures[valid_indices, valid_codes] = 1.0

        # Reshape to (n_observations, n_assets, n_factors)
        exposures = exposures.reshape(n_observations, n_assets, n_factors)

        self.factor_names_ = factor_names
        return exposures
