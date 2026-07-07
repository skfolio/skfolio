"""Base classes for all estimators and various utility functions."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress

import sklearn.base as skb

from skfolio.containers import AssetPanel
from skfolio.typing import FloatArray, StrArray

__all__ = ["BaseAssetPanelTransformer", "BaseComposition"]


class BaseAssetPanelTransformer(skb.BaseEstimator, ABC):
    """Base class for estimators that transform :class:`AssetPanel` data.

    Descriptors and factor exposure estimators take an :class:`AssetPanel` and
    return transformed values indexed by observation and asset. Most transformers return
    an array with shape `(n_observations, n_assets)`. Transformers that produce multiple
    values per asset, such as :class:`OneHotCategoricalFactors`, return an array with
    shape `(n_observations, n_assets, n_categories)`.

    In scikit-learn, `fit` and `partial_fit` update fitted state, stored in trailing
    underscore attributes, while `transform` returns transformed input data using that
    state. This separation is not suitable for every :class:`AssetPanel` transformer.
    For some estimators, the transformed value is produced by the same state transition
    that updates the estimator. A separate `transform` method would either need to
    mutate state or depend on a preceding `partial_fit` call, so the API exposes the
    combined operation directly. For example, the exponentially weighted momentum
    descriptor :class:`EWMomentum` needs to update its internal EWMA state to compute
    the transformed value on each observation.

    Other transformers are independent across observations. For example, the
    :class:`DividendToPrice` descriptor depends only on the current `dividends_ttm`
    and `market_cap` values and can therefore be declared stateless.

    Accordingly, `fit_transform` is used for full-batch computation and
    `partial_fit_transform` for online computation. Subclasses must implement
    `fit_transform`. Downstream meta-estimators use the presence of
    `partial_fit_transform` to determine whether a transformer supports online
    transformation.

    Supported implementation patterns are:

    - Batch-only transformers implement only `fit_transform`.
    - Stateless transformers declare `stateless=True` and implement only `fit_transform`.
      The base class adds `partial_fit_transform` as a direct delegation to `fit_transform`.
    - Online transformers implement both `fit_transform` and `partial_fit_transform`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    See Also
    --------
    BaseDescriptor : Computes a raw descriptor from characteristics.
    BaseFactorExposure : Computes factor exposures from characteristics.
    """

    n_assets_: int
    asset_names_: StrArray

    stateless: bool = False

    def __init_subclass__(cls, *, stateless: bool | None = None, **kwargs):
        """When `stateless=True`, the subclass declares that `fit_transform` is
        independent across observations. In this case, the base class injects
        `partial_fit_transform` as a delegation to `fit_transform`, so downstream
        meta-estimators can detect online support by checking for that method. When
        `stateless` is omitted, the value is inherited from the parent class.
        """
        super().__init_subclass__(**kwargs)
        if stateless is None:
            stateless = getattr(cls, "stateless", False)
        cls.stateless = stateless

        if stateless and "partial_fit_transform" in cls.__dict__:
            raise TypeError(
                "Classes declared with stateless=True must not define "
                "partial_fit_transform."
            )

        if stateless:

            def partial_fit_transform(
                self, X: AssetPanel, y=None, **fit_params
            ) -> FloatArray:
                """Stateless class delegation to `fit_transform`."""
                return self.fit_transform(X, y, **fit_params)

            cls.partial_fit_transform = partial_fit_transform

    @abstractmethod
    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Fit the transformer if needed and return transformed values.

        Parameters
        ----------
        X : AssetPanel
            Input panel data.

        y : None
            Ignored. Present for API consistency.

        **fit_params : dict
            Additional fit parameters. Metadata routing may pass these parameters to
            sub-estimators when applicable.

        Returns
        -------
        values : ndarray of shape (n_observations, n_assets) or (n_observations, n_assets, n_components)
            Transformed values.
        """


class BaseComposition(skb.BaseEstimator, ABC):
    """Handles parameter management for ensemble estimators."""

    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out

        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            # Ignore TypeError for cases where estimators is not a list of
            # (name, estimator) and ignore ValueError when the list is not
            # formatted correctly. This is to prevent errors when calling
            # `set_params`. `BaseEstimator.set_params` calls `get_params` which
            # can error for invalid values for `estimators`.
            return out

        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value
        return out

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Replace items with estimators in params
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            # Get item names used to identify valid names in params
            # `zip` raises a TypeError when `items` does not contains
            # elements of length 2
            with suppress(TypeError):
                item_names, _ = zip(*items, strict=True)
                for name in params:
                    if "__" not in name and name in item_names:
                        self._replace_estimator(attr, name, params.pop(name))

        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError(f"Names provided are not unique: {list(names)!r}")
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                f"Estimator names conflict with constructor arguments: {sorted(invalid_names)!r}"
            )
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                f"Estimator names must not contain __: got {invalid_names!r}"
            )
