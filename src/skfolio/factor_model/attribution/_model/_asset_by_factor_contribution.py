"""Asset-by-factor contribution dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from skfolio.factor_model.attribution._utils import _format_percent
from skfolio.typing import FloatArray, ObjArray

__all__ = ["AssetByFactorContribution"]


@dataclass(frozen=True)
class AssetByFactorContribution:
    r"""Asset-by-factor contribution breakdown.

    Breaks down factor contributions by asset. Each cell is the contribution of one
    asset to one factor's total contribution.

    Summing over assets gives per-factor contributions. Summing over factors gives each
    asset's systematic contribution.

    For single-point attribution, arrays have shape `(n_assets, n_factors)`. For rolling
    attribution, arrays have shape `(n_windows, n_assets, n_factors)`.

    Attributes
    ----------
    asset_names : ndarray of shape (n_assets,)
        Asset names.

    factor_names : ndarray of shape (n_factors,)
        Factor names.

    vol_contrib : ndarray of shape (n_assets, n_factors) or (n_windows, n_assets, n_factors)
        Volatility contribution for each asset-factor pair.

    mu_contrib : ndarray of shape (n_assets, n_factors) or (n_windows, n_assets, n_factors)
        Return contribution for each asset-factor pair.
    """

    asset_names: ObjArray
    factor_names: ObjArray
    vol_contrib: FloatArray
    mu_contrib: FloatArray

    def _to_df(
        self,
        metric: str = "vol_contrib",
        formatted: bool = False,
        observation_idx: int | None = None,
    ) -> pd.DataFrame:
        """Return the selected asset-by-factor contribution as a DataFrame.

        Parameters
        ----------
        metric : str, default="vol_contrib"
            Contribution metric to display: "vol_contrib" or "mu_contrib".

        formatted : bool, default=False
            If True, format values as percentages.

        observation_idx : int or None, default=None
            Observation index. Required for rolling attribution.

        Returns
        -------
        df : DataFrame
            Matrix with assets as rows and factors as columns.
        """
        if metric not in ("vol_contrib", "mu_contrib"):
            raise ValueError("`metric` must be 'vol_contrib' or 'mu_contrib'.")

        data = getattr(self, metric)

        if data.ndim == 3:
            if observation_idx is None:
                raise ValueError(
                    "For rolling attribution, must specify `observation_idx`."
                )
            if not 0 <= observation_idx < data.shape[0]:
                raise IndexError(
                    f"`observation_idx` {observation_idx} is out of range "
                    f"[0, {data.shape[0]})."
                )
            data = data[observation_idx]

        df = pd.DataFrame(data, index=self.asset_names, columns=self.factor_names)
        df.index.name = "Asset"

        if formatted:
            df = df.map(_format_percent)

        return df
