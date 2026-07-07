"""Abstract base classes for factor exposure estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC

from skfolio.base import BaseAssetPanelTransformer

__all__ = ["BaseFactorExposure"]


class BaseFactorExposure(BaseAssetPanelTransformer, ABC):
    """Base class for factor exposure estimators.

    A factor exposure estimator takes an :class:`AssetPanel` and returns asset exposures
    to one or more factors. Single-factor estimators return values with shape
    `(n_observations, n_assets)`. Multi-factor estimators return values with shape
    `(n_observations, n_assets, n_factors)`.

    Factor exposures follow the :class:`~skfolio.base.BaseAssetPanelTransformer`
    protocol:

    - Batch-only estimators implement only `fit_transform`.
    - Stateless estimators declare `stateless=True` and implement only `fit_transform`.
      The base class adds `partial_fit_transform` by delegating to `fit_transform`.
    - Online estimators implement both `fit_transform` and `partial_fit_transform`.
      `fit_transform` starts from a clean state and `partial_fit_transform` continues
      from the current state.

    Parameters
    ----------
    family : str
        The factor family this exposure belongs to (e.g., "market", "style", "industry",
        "country"). Factor families group related factors for basket-neutral
        constraints, neutralization, attribution and reporting.

    See Also
    --------
    BaseAssetPanelTransformer : Shared `AssetPanel` transformer contract.
    BaseDescriptor : Computes raw descriptor values.
    """

    def __init__(self, *, family: str) -> None:
        self.family = family
