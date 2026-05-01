"""Base Asset Panel."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

import pandas as pd

from skfolio.containers._asset_panel._fields import BaseField, FieldCategorical
from skfolio.containers._asset_panel._utils import _to_dataframe
from skfolio.typing import StrArray


class _BaseAssetPanel(ABC):
    """Base AssetPanel class to share read utilities between `AssetPanel` and
    `AssetPanelView`.
    """

    def __contains__(self, name: str) -> bool:
        """Check whether a field exists."""
        return name in self.keys()

    @property
    @abstractmethod
    def n_observations(self) -> int:
        """Number of observations."""

    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Number of assets."""

    @abstractmethod
    def keys(self) -> Iterable[str]:
        """Return field names."""

    @abstractmethod
    def get_field(self, name: str) -> BaseField:
        """Return a field object."""

    @property
    def shape(self) -> tuple[int]:
        """Shape tuple used by scikit-learn sample indexing."""
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Number of dimensions used by scikit-learn sample indexing."""
        return 1

    def decode_categorical_field(
        self, name: str, *, missing_label: str = "MISSING"
    ) -> StrArray:
        """Decode a categorical field to labels.

        Parameters
        ----------
        name : str
            Name of a `FieldCategorical` field.

        missing_label : str, default="MISSING"
            Label assigned to missing or out-of-bound codes.

        Returns
        -------
        decoded : ndarray
            Decoded labels with shape (n_observations, n_assets).
        """
        field = self.get_field(name)
        if not isinstance(field, FieldCategorical):
            raise TypeError(f"Field '{name}' is not categorical.")
        return field.decode(missing_label=missing_label)

    def to_dataframe(
        self,
        *,
        fields: str | Iterable[str] | None = None,
        assets: str | Iterable[str] | None = None,
        output_format: Literal["long", "wide"] = "long",
        decode_categoricals: bool = True,
    ) -> pd.DataFrame:
        """Convert 2D fields to a pandas DataFrame.

        Parameters
        ----------
        fields : str, iterable of str, or None, optional
            Field names to include. If a single string is passed, the result is a simple
            field DataFrame with observations as index and assets as columns. If `None`,
            all 2D fields are included.

        assets : str, iterable of str, or None, optional
            Asset labels to include. If `None`, all assets are included.

        output_format : {"long", "wide"}, default="long"
            Output format used when `fields` is not a single string. In long format,
            rows are indexed by `(observation, asset)` and filtered by `active_mask`.
            In wide format, columns are indexed by `(field, asset)`.

        decode_categoricals : bool, default=True
            If `True`, categorical codes are decoded to labels.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame representation of the selected 2D fields.
        """
        return _to_dataframe(
            self,
            fields=fields,
            assets=assets,
            output_format=output_format,
            decode_categoricals=decode_categoricals,
        )
