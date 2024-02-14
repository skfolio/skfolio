"""Base Composition estimator.
Follow same implementation as Base composition from sklearn
"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from abc import ABC, abstractmethod
from contextlib import suppress

import sklearn.base as skb


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
