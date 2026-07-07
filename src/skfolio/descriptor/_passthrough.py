"""Passthrough descriptor exposing a raw `AssetPanel` field."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray, IntArray
from skfolio.utils.validation import validate_asset_panel


class Passthrough(BaseDescriptor, stateless=True):
    """Passthrough descriptor for an `AssetPanel` field.

    Returns the selected panel field without numerical transformation. This is
    useful for raw vendor fields or for values computed upstream that should
    enter a factor exposure model unchanged.

    Parameters
    ----------
    field : str
        Name of the field to read from the input :class:`AssetPanel`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import Passthrough
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = Passthrough("eps_ntm")
    >>> eps_ntm = descriptor.fit_transform(X)
    """

    def __init__(self, field: str) -> None:
        self.field = field

    def fit_transform(
        self, X: AssetPanel, y=None, **fit_params
    ) -> FloatArray | IntArray:
        """Return the configured panel field.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the `field` characteristic configured at
            construction.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        values : ndarray of shape (n_observations, n_assets)
            Raw values of `field` for each observation and asset.
        """
        validate_asset_panel(self, X, required_fields=[self.field])
        return X[self.field]
