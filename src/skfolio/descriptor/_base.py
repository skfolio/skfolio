"""Base class for descriptor transformers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC

import numpy as np

from skfolio.base import BaseAssetPanelTransformer, BaseComposition
from skfolio.containers import AssetPanel
from skfolio.typing import BoolArray, FloatArray
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_positive_integer,
)
from skfolio.utils.validation import validate_asset_panel

__all__ = ["BaseDescriptor"]

from sklearn import utils as sku
from sklearn.utils import metadata_routing as skm


class BaseDescriptor(BaseAssetPanelTransformer, ABC):
    """Base class for all descriptor transformers.

    A descriptor takes an :class:`AssetPanel` and returns one raw descriptor
    value per observation and asset, usually with shape
    `(n_observations, n_assets)`. Descriptors are the inputs used by factor
    exposure estimators such as
    :class:`~skfolio.factor_exposure.FixedWeightedFactor`.

    Descriptors follow the :class:`~skfolio.base.BaseAssetPanelTransformer`
    protocol:

    - Batch-only descriptors implement only `fit_transform`.
    - Stateless descriptors declare `stateless=True` and implement only
      `fit_transform`. The base class adds `partial_fit_transform` by
      delegating to `fit_transform`.
    - Online descriptors implement both `fit_transform` and
      `partial_fit_transform`. `fit_transform` starts from a clean state;
      `partial_fit_transform` continues from the current state.

    Examples include direct field access (:class:`Passthrough`), point-in-time
    ratios (:class:`BookToPrice`), and history-dependent descriptors
    (:class:`EWMomentum`).

    See Also
    --------
    BaseAssetPanelTransformer : Shared `AssetPanel` transformer contract.
    BaseFactorExposure : Combines descriptors into factor exposures.
    """


class BaseDescriptorComposition(BaseComposition, ABC):
    """Base class for all descriptor composition estimators in skfolio.

    This mixin provides `get_params` / `set_params` / metadata routing
    for the `descriptors` parameter, following the scikit-learn named
    estimator convention (similar to `Pipeline` or `ColumnTransformer`).

    Descriptors are specified as `(name, estimator)` tuples.
    """

    descriptors: list[tuple[str, BaseDescriptor]]

    @property
    def named_descriptors(self):
        """Dictionary to access any fitted factors by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return sku.Bunch(**dict(self.descriptors))

    def set_params(self, **params):
        """Set the parameters of an factor from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("descriptors", **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("descriptors", deep=deep)

    def get_metadata_routing(self):
        """Return metadata routing for descriptor estimators."""
        router = skm.MetadataRouter(owner=self.__class__.__name__)
        names, descriptors = self._validate_descriptors()
        for name, descriptor in zip(names, descriptors, strict=True):
            router.add(
                **{name: descriptor},
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        return router

    def _validate_descriptors(self) -> tuple[list[str], list[BaseDescriptor]]:
        """Validate the `descriptors` parameter.

        Returns
        -------
        names : list[str]
            The list of descriptor names.
        descriptors : list[BaseDescriptor]
            The list of descriptor estimators.
        """
        if self.descriptors is None or len(self.descriptors) == 0:
            raise ValueError(
                "Invalid 'descriptors' attribute, 'descriptors' should be a "
                "list of (name, descriptor) tuples."
            )
        names, descriptors = zip(*self.descriptors, strict=True)

        self._validate_names(names)
        for descriptor in descriptors:
            if not isinstance(descriptor, BaseDescriptor):
                raise TypeError(
                    f"Expected descriptor to be a BaseDescriptor, got {type(descriptor)}"
                )

        return list(names), list(descriptors)


class _BaseRollingLogReturn(BaseDescriptor):
    """Private base class for fixed-window log-return descriptors.

    Used by `RollingMomentum` and `Reversal`.
    """

    _FITTED_ATTR: str
    _TRANSFORM_SIGN: float = 1.0

    def __init__(self, window: int, skip: int = 0, exponentiate: bool = False):
        self.window = window
        self.skip = skip
        self.exponentiate = exponentiate

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute the rolling log-return descriptor from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        descriptor : ndarray of shape (n_observations, n_assets)
            Rolling log-return descriptor for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update state and compute the rolling log-return descriptor.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        descriptor : ndarray of shape (n_observations, n_assets)
            Rolling log-return descriptor for each observation and asset.
        """
        first_call = not hasattr(self, self._FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["returns"],
            finite_or_nan=["returns"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        returns = X["returns"]
        active_mask = X.active_mask

        buffer_length = self._rolling_buffer_length
        first_valid = self.skip + self.window - 1

        estimator_name = self.__class__.__name__
        non_missing = ~np.isnan(returns)
        if np.any(returns[non_missing] <= -1):
            raise ValueError(
                'Field "returns" contains values less than or equal to -1. '
                f"{estimator_name} requires returns greater than -1 because it uses "
                "log returns."
            )

        # Active assets with missing returns contribute zero to the log-return sum.
        contrib = np.zeros_like(returns, dtype=float)
        contrib[non_missing] = np.log1p(returns[non_missing])

        result = np.full((n_observations, n_assets), np.nan, dtype=float)

        if self._rolling_n_seen == 0 and n_observations > first_valid:
            # Vectorized batch path.
            cumsum = np.zeros((n_observations + 1, n_assets), dtype=float)
            cumsum[1:] = np.cumsum(contrib, axis=0)
            active_cumsum = np.zeros((n_observations + 1, n_assets), dtype=int)
            active_cumsum[1:] = np.cumsum(active_mask, axis=0)

            valid_t = np.arange(first_valid, n_observations)
            end = valid_t - self.skip + 1
            start = end - self.window
            rolling = cumsum[end] - cumsum[start]
            active_count = active_cumsum[end] - active_cumsum[start]

            transformed = self._transform_rolling_sum(rolling)
            result[first_valid:] = np.where(
                active_count == self.window, transformed, np.nan
            )

            self._sync_rolling_ring(contrib, active_mask, n_observations)
        else:
            # Online path: ring buffer with running sum and active count.
            for t in range(n_observations):
                pos = self._rolling_n_seen % buffer_length

                if self._rolling_n_seen >= first_valid:
                    if self._rolling_n_seen == first_valid:
                        self._rolling_contrib_buffer[pos] = contrib[t]
                        self._rolling_active_mask_buffer[pos] = active_mask[t]
                        self._rolling_sum[:] = np.sum(
                            self._rolling_contrib_buffer[: self.window], axis=0
                        )
                        self._rolling_active_count[:] = np.sum(
                            self._rolling_active_mask_buffer[: self.window], axis=0
                        )
                    else:
                        self._rolling_sum -= self._rolling_contrib_buffer[pos]
                        self._rolling_active_count -= self._rolling_active_mask_buffer[
                            pos
                        ]
                        self._rolling_contrib_buffer[pos] = contrib[t]
                        self._rolling_active_mask_buffer[pos] = active_mask[t]
                        entering = (self._rolling_n_seen - self.skip) % buffer_length
                        self._rolling_sum += self._rolling_contrib_buffer[entering]
                        self._rolling_active_count += self._rolling_active_mask_buffer[
                            entering
                        ]

                    transformed = self._transform_rolling_sum(self._rolling_sum)
                    result[t] = np.where(
                        self._rolling_active_count == self.window,
                        transformed,
                        np.nan,
                    )
                else:
                    self._rolling_contrib_buffer[pos] = contrib[t]
                    self._rolling_active_mask_buffer[pos] = active_mask[t]

                self._rolling_n_seen += 1

        # Mask for inactive assets.
        result = np.where(X.active_mask, result, np.nan)

        last = result[-1].copy() if n_observations > 1 else result[-1]
        setattr(self, self._FITTED_ATTR, last)

        return result

    def _transform_rolling_sum(self, rolling_sum: FloatArray) -> FloatArray:
        """Apply the descriptor-specific sign and output scale."""
        signed = self._TRANSFORM_SIGN * rolling_sum
        if self.exponentiate:
            return np.expm1(signed)
        return signed

    def _sync_rolling_ring(
        self, contrib: FloatArray, active_mask: BoolArray, n_observations: int
    ) -> None:
        """Populate ring buffer and running aggregates after a vectorized batch."""
        tail_start = max(0, n_observations - self._rolling_buffer_length)
        indices = np.arange(tail_start, n_observations)
        self._rolling_contrib_buffer[indices % self._rolling_buffer_length] = contrib[
            indices
        ]
        self._rolling_active_mask_buffer[indices % self._rolling_buffer_length] = (
            active_mask[indices]
        )
        start = n_observations - self.skip - self.window
        end = n_observations - self.skip
        self._rolling_sum[:] = np.sum(contrib[start:end], axis=0)
        self._rolling_active_count[:] = np.sum(active_mask[start:end], axis=0)
        self._rolling_n_seen = n_observations

    def _reset(self):
        if hasattr(self, self._FITTED_ATTR):
            delattr(self, self._FITTED_ATTR)
        for attr in (
            "_rolling_buffer_length",
            "_rolling_contrib_buffer",
            "_rolling_active_mask_buffer",
            "_rolling_sum",
            "_rolling_active_count",
            "_rolling_n_seen",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _validate_params(self) -> None:
        _validate_positive_integer(self.window, "window")
        _validate_non_negative_integer(self.skip, "skip")

    def _initialize(self) -> None:
        n_assets = self.n_assets_
        buffer_length = self.skip + self.window
        self._rolling_buffer_length = buffer_length
        self._rolling_contrib_buffer = np.zeros((buffer_length, n_assets), dtype=float)
        self._rolling_active_mask_buffer = np.zeros(
            (buffer_length, n_assets), dtype=bool
        )
        self._rolling_sum = np.zeros(n_assets, dtype=float)
        self._rolling_active_count = np.zeros(n_assets, dtype=int)
        self._rolling_n_seen = 0
