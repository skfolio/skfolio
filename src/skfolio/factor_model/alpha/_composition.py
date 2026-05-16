"""Base alpha descriptor composition."""
# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC

import numpy as np
import sklearn as sk
import sklearn.utils.parallel as skp

import skfolio.typing as skt
from skfolio._constants import _DESCRIPTOR_SCORES, _EXPOSURES, _PASSTHROUGH
from skfolio.containers import (
    AssetPanel,
    AssetPanelView,
    Field3D,
    InactivePolicy,
    concat,
)
from skfolio.factor_model._utils import _neutralize_scores
from skfolio.factor_model.descriptor import BaseDescriptor
from skfolio.factor_model.descriptor._composition import BaseDescriptorComposition
from skfolio.preprocessing import BaseCSTransformer, CSStandardScaler, CSWinsorizer
from skfolio.typing import FloatArray
from skfolio.utils.tools import call_asset_panel_transform, check_estimator


class BaseAlphaDescriptorComposition(BaseDescriptorComposition, ABC):
    """Base class for alpha estimators built from composing descriptor estimators."""

    descriptors: list[tuple[str, BaseDescriptor]]
    horizon: int
    signal_lag: int
    neutralize_against: list[str] | None
    outlier_transformer: skt.CSTransformer
    scoring_transformer: skt.CSTransformer
    transform_by_group: str | None
    scale_target_by_idio_vol: bool
    n_jobs: int

    descriptors_: list[BaseDescriptor]
    named_descriptors_: dict[str, BaseDescriptor]
    outlier_transformer_: skt.CSTransformer
    scoring_transformer_: skt.CSTransformer

    def _validate_common_params(self) -> None:
        """Validate common descriptor-alpha hyperparameters."""
        if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError(
                f"horizon must be a positive integer (>= 1), got {self.horizon}"
            )
        if not isinstance(self.signal_lag, (int, np.integer)) or self.signal_lag < 1:
            raise ValueError(
                f"signal_lag must be a positive integer (>= 1), got {self.signal_lag}"
            )
        if not self.descriptors:
            raise ValueError("descriptors cannot be empty")
        if not isinstance(self.scale_target_by_idio_vol, (bool, np.bool_)):
            raise ValueError(
                "scale_target_by_idio_vol must be a boolean, "
                f"got {self.scale_target_by_idio_vol!r}"
            )

    def _initialize_common_state(self) -> None:
        """Initialize common descriptor state and cross-sectional transformers."""
        names, descriptors = self._validate_descriptors()

        self.descriptors_ = [sk.clone(des) for des in descriptors]
        self.named_descriptors_ = {
            name: estimator
            for name, estimator in zip(names, self.descriptors_, strict=True)
        }

        self.outlier_transformer_ = check_estimator(
            self.outlier_transformer,
            default=CSWinsorizer(),
            check_type=BaseCSTransformer,
        )

        self.scoring_transformer_ = check_estimator(
            self.scoring_transformer,
            default=CSStandardScaler(),
            check_type=BaseCSTransformer,
        )

    def _compute_scores(self, X: AssetPanel, method: str, routed_params) -> FloatArray:
        """Compute transformed descriptor scores from the input panel."""
        cs_weights = X.estimation_mask.astype(float)
        cs_groups = (
            X[self.transform_by_group] if self.transform_by_group is not None else None
        )

        # Threading avoids copying the (potentially large) AssetPanel
        # to each worker. Workers only read from it, so shared memory
        # is safe. Descriptor computations are NumPy-dominated and
        # release the GIL, giving true parallelism with threads.
        scores = skp.Parallel(n_jobs=self.n_jobs, prefer="threads")(
            skp.delayed(call_asset_panel_transform)(
                descriptor,
                X=X,
                fit_params=routed_params[name][method],
                method=f"{method}_transform",
            )
            for name, descriptor in self.named_descriptors_.items()
        )

        for i, score in enumerate(scores):
            if self.outlier_transformer_ != _PASSTHROUGH:
                score = self.outlier_transformer_.fit_transform(
                    score, cs_weights=cs_weights, cs_groups=cs_groups
                )
            if self.scoring_transformer_ != _PASSTHROUGH:
                score = self.scoring_transformer_.fit_transform(
                    score, cs_weights=cs_weights, cs_groups=cs_groups
                )
            scores[i] = score

        scores = np.stack(scores, axis=2)

        # Score neutralization
        if self.neutralize_against is not None:
            field = X.fields[_EXPOSURES]
            scores = _neutralize_scores(
                neutralize_against=self.neutralize_against,
                scores=scores,
                exposures=field.values,
                cs_weights=cs_weights,
                factor_names=field.third_axis_labels,
                factor_families=field.third_axis_groups,
            )
            # Re-apply scoring after neutralization
            if self.scoring_transformer_ != _PASSTHROUGH:
                for i in range(len(self.descriptors_)):
                    scores[:, :, i] = self.scoring_transformer_.fit_transform(
                        scores[:, :, i], cs_weights=cs_weights, cs_groups=cs_groups
                    )

        return scores

    def _make_training_panel(
        self,
        X: AssetPanel | AssetPanelView,
        *,
        scores: FloatArray,
        fields: list[str],
    ) -> AssetPanel:
        """Build the compact panel used for online target maturation."""
        panel = (
            X.to_panel(fields=fields, deep=False)
            if isinstance(X, AssetPanelView)
            else X.sel(fields=fields)
        )
        panel[_DESCRIPTOR_SCORES] = Field3D(
            scores,
            third_axis_name="descriptor",
            third_axis_labels=list(self.named_descriptors_),
            inactive_policy=InactivePolicy.IGNORE,
        )
        return panel

    def _prepend_buffer(self, current: AssetPanel) -> AssetPanel:
        """Prepend pending rows to the current compact training panel."""
        if self._buffer is None:
            return current
        return concat([self._buffer, current])

    def _update_buffers(self, combined: AssetPanel) -> None:
        """Keep only rows needed to mature future forward-return targets."""
        buffer_start = max(0, combined.n_observations - self._target_gap)
        self._buffer = combined[buffer_start:].to_panel(deep=True)

    def _initialize_buffers(self) -> None:
        """Initialize online buffers."""
        self._buffer = None

    @property
    def _target_gap(self) -> int:
        """Number of future rows required before a signal observation matures."""
        return self.signal_lag + self.horizon - 1
