"""Fixed-weighted factor exposure."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn as sk
import sklearn.utils.metadata_routing as skm
import sklearn.utils.parallel as skp

import skfolio.typing as skt
from skfolio._constants import _BENCHMARK_WEIGHTS, _PASSTHROUGH
from skfolio.containers import AssetPanel
from skfolio.factor_model.descriptor import BaseDescriptor
from skfolio.factor_model.descriptor._composition import BaseDescriptorComposition
from skfolio.factor_model.factor_exposure._base import BaseFactorExposure
from skfolio.preprocessing import (
    BaseCSTransformer,
    CSStandardScaler,
    CSWinsorizer,
)
from skfolio.typing import FloatArray
from skfolio.utils.tools import call_asset_panel_transform, check_estimator
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "descriptors_"


class FixedWeightedFactor(BaseFactorExposure, BaseDescriptorComposition):
    r"""Factor exposure as a fixed weighted combination of descriptors.

    Computes descriptor values, applies cross-sectional outlier and scoring transforms
    to each descriptor, then combines the resulting scores into a single factor exposure
    matrix with shape `(n_observations, n_assets)`.

    For descriptor :math:`i`, let :math:`s_{i,t,j}` be its score for observation
    :math:`t` and asset :math:`j` after the cross-sectional transforms, :math:`w_i` its
    fixed non-negative weight and :math:`V_{t,j}` the set of descriptors with a valid
    (non-NaN) score. The weighted composite is:

    .. math::

        c_{t,j} = \frac{\sum_{i \in V_{t,j}} w_i \, s_{i,t,j}}
                       {\sum_{i \in V_{t,j}} w_i}

    Weights are renormalized over available scores for each asset-observation pair.
    This allows assets with structurally unavailable descriptor values (e.g., Gross
    Margin for financial firms, which do not report cost of goods sold) to receive a
    composite score from the remaining descriptors. When all descriptor scores are NaN
    for a pair, the composite is NaN.

    The `min_coverage` parameter controls the minimum fraction of total descriptor
    weight that must be valid for the composite to be computed. If the valid weight
    fraction falls below this threshold, the composite is set to NaN instead. This
    guards against low-quality exposures based on too few descriptors:

    - A composite from 1 out of 7 descriptors is qualitatively different
      from the full factor and may load on a different latent signal.
    - Assets sharing the same "partial" descriptor coverage (e.g., an
      entire sector) can introduce systematic bias in factor return
      estimates.

    The default `min_coverage=0.0` uses any available descriptor (no threshold), which
    maximizes coverage.  A value of `0.5` requires at least half the descriptor weight
    to be valid.

    When multiple descriptors are combined and `scoring_transformer` is not
    `"passthrough"`, the composite is scored again cross-sectionally so assets with
    different descriptor coverage are on the same scale. The final exposure is this
    re-scored composite, or the weighted composite when scoring is skipped.

    `weights` are fixed inputs and are not learned by this estimator. They can be set
    from economic priors or selected by hyperparameters tuning.

    Parameters
    ----------
    descriptors : list of tuple (str, BaseDescriptor)
        List of `(name, descriptor)` pairs. Each descriptor computes values from the
        :class:`AssetPanel`.

    family : str, default="style"
        The factor family this exposure belongs to (e.g., "market", "style", "industry",
        "country"). Factor families group related factors for basket-neutral constraints,
        neutralization, attribution and reporting. The default is `"style"`.

    weights : array-like of shape (n_descriptors,), optional
        Non-negative descriptor combination weights. Must sum to 1. If `None` (default),
        equal weights are used.

    min_coverage : float, default=0.0
        Minimum fraction of total descriptor weight that must be valid (non-NaN) for the
        composite to be computed. Values where the valid weight fraction is below this
        threshold are set to NaN. Must be in `[0, 1]`.

        - `0.0` (default): use any available descriptor. Maximizes cross-sectional coverage.
        - `0.5`: require at least half the descriptor weight to be valid.

        The threshold is weight-based, not count-based. If descriptor weights are
        `[0.8, 0.2]` and only the first descriptor is valid, the valid weight fraction
        is 0.8, so a `min_coverage=0.5` threshold is satisfied even though only 1 out of
        2 descriptors is present.

    outlier_transformer : BaseCSTransformer or "passthrough" or None, default=None
        Cross-sectional transformer for outlier handling. If None, defaults to
        `CSWinsorizer()`. Use "passthrough" to skip.

    scoring_transformer : BaseCSTransformer or "passthrough" or None, default=None
        Cross-sectional transformer for scoring applied after outlier handling.
        If None, defaults to `CSStandardScaler()`. Use "passthrough" to skip.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations. If provided, outlier and scoring transformations are applied
        within each group separately.

    n_jobs : int, default=1
        Number of parallel jobs for descriptor computation.

    Attributes
    ----------
    descriptors_ : list of BaseDescriptor
        Fitted descriptor estimators.

    named_descriptors_ : dict of {str: BaseDescriptor}
        Dictionary mapping descriptor names to fitted estimators.

    outlier_transformer_ : BaseCSTransformer or str
        The fitted outlier transformer.

    scoring_transformer_ : BaseCSTransformer or str
        The fitted scoring transformer.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.
    """

    descriptors_: list[BaseDescriptor]
    named_descriptors_: dict[str, BaseDescriptor]
    outlier_transformer_: skt.CSTransformer
    scoring_transformer_: skt.CSTransformer

    def __init__(
        self,
        *,
        descriptors: list[tuple[str, BaseDescriptor]],
        family: str = "style",
        weights: FloatArray | None = None,
        min_coverage: float = 0.0,
        outlier_transformer: skt.CSTransformer = None,
        scoring_transformer: skt.CSTransformer = None,
        transform_by_group: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(family=family)
        self.descriptors = descriptors
        self.weights = weights
        self.min_coverage = min_coverage
        self.outlier_transformer = outlier_transformer
        self.scoring_transformer = scoring_transformer
        self.transform_by_group = transform_by_group
        self.n_jobs = n_jobs

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute factor exposure from a clean descriptor state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing "benchmark_weights", descriptor fields and optional
            grouping fields.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors through metadata routing.

        Returns
        -------
        exposure : ndarray of shape (n_observations, n_assets)
            Fixed-weighted factor exposure.
        """
        self._reset()
        return self._fit_transform(X, method="fit_transform", **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update descriptor state and compute factor exposure.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing benchmark weights, descriptor fields and optional
            grouping fields.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters passed to descriptors through metadata routing.

        Returns
        -------
        exposure : ndarray of shape (n_observations, n_assets)
            Fixed-weighted factor exposure for the new observations.
        """
        return self._fit_transform(
            X,
            method="partial_fit_transform",
            **fit_params,
        )

    def _fit_transform(
        self,
        X: AssetPanel,
        *,
        method: str,
        **fit_params,
    ) -> FloatArray:
        """Compute factor exposure using the requested descriptor transform method."""
        routing_method = method.removesuffix("_transform")
        routed_params = skm.process_routing(self, routing_method, **fit_params)

        first_call = not hasattr(self, _FITTED_ATTR)

        required_fields = [_BENCHMARK_WEIGHTS]
        if self.transform_by_group is not None:
            required_fields.append(self.transform_by_group)
        validate_asset_panel(self, X, required_fields=required_fields, reset=first_call)

        if first_call:
            self._initialize()

        n_descriptors = len(self.descriptors_)

        # Threading avoids copying the (potentially large) AssetPanel
        # to each worker. Workers only read from it, so shared memory
        # is safe. Descriptor computations are NumPy-dominated and
        # release the GIL, giving true parallelism with threads.
        scores = skp.Parallel(n_jobs=self.n_jobs, prefer="threads")(
            skp.delayed(call_asset_panel_transform)(
                des,
                X=X,
                fit_params=routed_params[name][routing_method],
                method=method,
            )
            for name, des in self.named_descriptors_.items()
        )
        # (n_descriptors, n_observations, n_assets)
        scores = np.stack(scores, axis=0)

        cs_weight = X[_BENCHMARK_WEIGHTS]
        cs_group = (
            X[self.transform_by_group] if self.transform_by_group is not None else None
        )

        for i in range(n_descriptors):
            if self.outlier_transformer_ != _PASSTHROUGH:
                scores[i] = self.outlier_transformer_.fit_transform(
                    scores[i], cs_weights=cs_weight, cs_groups=cs_group
                )

            if self.scoring_transformer_ != _PASSTHROUGH:
                scores[i] = self.scoring_transformer_.fit_transform(
                    scores[i], cs_weights=cs_weight, cs_groups=cs_group
                )

        # NaN-aware weighted combination: for each (t, j) use only the
        # descriptors that produced a valid score, renormalizing weights so
        # they sum to one over the available subset.  This avoids dropping
        # entire assets when some descriptors are structurally undefined
        # (e.g., COGS-based descriptors for financial firms).
        w = self.descriptor_weights_[:, np.newaxis, np.newaxis]  # (D, 1, 1)
        valid = ~np.isnan(scores)  # (D, T, N)
        w_valid = w * valid  # zero-out weights of NaN descriptors
        w_sum = w_valid.sum(axis=0)  # (T, N)

        # Replace NaN with 0 for the summation, then divide by valid weight.
        # Apply min_coverage: cells where the valid weight fraction is below
        # the threshold are set to NaN.
        scores_filled = np.where(valid, scores, 0.0)
        sufficient = w_sum >= self.min_coverage_
        # Guard against division by zero (masked out by `sufficient` anyway)
        w_sum_safe = np.where(sufficient, w_sum, 1.0)
        scores = np.where(
            sufficient,
            (w * scores_filled).sum(axis=0) / w_sum_safe,
            np.nan,
        )

        # Re-standardize the composite so that assets with different
        # numbers of contributing descriptors are on the same scale.
        if n_descriptors > 1 and self.scoring_transformer_ != _PASSTHROUGH:
            scores = self.scoring_transformer_.fit_transform(
                scores, cs_weights=cs_weight, cs_groups=cs_group
            )

        return scores

    def _reset(self):
        """Reset fitted descriptor state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _initialize(self):
        """Initialize descriptors, transformers and descriptor weights."""
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

        if self.weights is None:
            self.descriptor_weights_ = np.ones(len(descriptors)) / len(descriptors)
        else:
            self.descriptor_weights_ = np.asarray(self.weights)
            if self.descriptor_weights_.ndim != 1:
                raise ValueError("Descriptor weights should be a 1D array")
            if self.descriptor_weights_.size != len(descriptors):
                raise ValueError(
                    "Descriptor weights should have the same length of descriptors"
                )
            if np.any(self.descriptor_weights_ < 0):
                raise ValueError("Descriptor weights should be non-negative")
            if not np.isclose(self.descriptor_weights_.sum(), 1):
                raise ValueError("Descriptor weights should sum to 1")

        if not 0 <= self.min_coverage <= 1:
            raise ValueError(f"min_coverage must be in [0, 1], got {self.min_coverage}")
        # Store as a tiny positive number when 0 so the comparison
        # `w_sum >= threshold` naturally passes when any descriptor is valid,
        # while still producing NaN when w_sum == 0 (all descriptors NaN).
        self.min_coverage_ = max(self.min_coverage, 1e-12)
