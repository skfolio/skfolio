"""Hierarchical asset seriation."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import sklearn.utils.metadata_routing as skm

from skfolio.cluster import HierarchicalClustering
from skfolio.seriation._base import BaseSeriation
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.tools import _validate_bool, check_estimator


class HierarchicalSeriation(BaseSeriation):
    """Order assets by the leaves of a hierarchical clustering tree.

    Parameters
    ----------
    hierarchical_clustering_estimator : HierarchicalClustering, optional
        Hierarchical clustering estimator, including compatible subclasses.
        Each `fit` trains a fresh clone when at least two assets are investable.
        The default (`None`) uses :class:`~skfolio.cluster.HierarchicalClustering`
        with Ward linkage.

    optimal_ordering : bool, default=True
        If True, minimize distances between adjacent leaves without changing
        the tree [1]_. If False, use the leaf order produced by the clustering tree.

    Attributes
    ----------
    ordering_ : ndarray of shape (n_investable_assets,)
        Positions in the original input matrix, listed in the computed order.
        Each investable asset appears exactly once. Empty when no assets are
        investable. A single investable asset produces its original position.

    investable_mask_ : ndarray of shape (n_assets,)
        Boolean mask selecting investable assets for the current ordering.

    hierarchical_clustering_estimator_ : HierarchicalClustering or None
        Fitted clustering estimator, or None with fewer than two investable assets.

    ordered_linkage_matrix_ : ndarray of shape (max(n_investable_assets - 1, 0), 4)
        Linkage matrix with the selected leaf ordering. Leaf indices refer to
        the compact input order, given by `np.flatnonzero(investable_mask_)`.

    n_features_in_ : int
        Number of assets in the full schema.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Asset names, defined when the input names are all strings.

    References
    ----------
    .. [1] "Fast optimal leaf ordering for hierarchical clustering".
        Ziv Bar-Joseph, David K. Gifford and Tommi S. Jaakkola,
        Bioinformatics (2001).
    """

    hierarchical_clustering_estimator_: HierarchicalClustering | None
    ordered_linkage_matrix_: FloatArray

    def __init__(
        self,
        *,
        hierarchical_clustering_estimator: HierarchicalClustering | None = None,
        optimal_ordering: bool = True,
    ) -> None:
        self.hierarchical_clustering_estimator = hierarchical_clustering_estimator
        self.optimal_ordering = optimal_ordering

    def get_metadata_routing(self) -> skm.MetadataRouter:
        """Route fitting metadata to the clustering estimator."""
        return skm.MetadataRouter(owner=type(self).__name__).add(
            hierarchical_clustering_estimator=self.hierarchical_clustering_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )

    def fit(
        self, X: ArrayLike, y: None = None, **fit_params: Any
    ) -> HierarchicalSeriation:
        """Start a new hierarchical ordering from a distance snapshot.

        Parameters
        ----------
        X : array-like of shape (n_assets, n_assets)
            Distance snapshot. NaN diagonal entries mark non-investable assets.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the clustering estimator.
            Only available if `enable_metadata_routing=True`, which can be
            set by using `sklearn.set_config(enable_metadata_routing=True)`.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : HierarchicalSeriation
            Fitted estimator.
        """
        _validate_bool(self.optimal_ordering, "optimal_ordering")
        clustering = check_estimator(
            self.hierarchical_clustering_estimator,
            default=HierarchicalClustering(),
            check_type=HierarchicalClustering,
        )
        routed = skm.process_routing(self, "fit", **fit_params)
        distance, investable_mask = self._validate_distance(X, reset=True)
        indices = np.flatnonzero(investable_mask)
        linkage = np.empty((0, 4))
        ordering = indices
        if len(indices) > 1:
            if hasattr(self, "feature_names_in_"):
                names = self.feature_names_in_[investable_mask]
                distance = pd.DataFrame(distance, index=names, columns=names)
            clustering.fit(distance, **routed.hierarchical_clustering_estimator.fit)
            linkage = clustering.linkage_matrix_
            if self.optimal_ordering:
                linkage = sch.optimal_leaf_ordering(
                    linkage, clustering.condensed_distance_
                )
            ordering = indices[sch.leaves_list(linkage)]
        else:
            clustering = None
        self.hierarchical_clustering_estimator_ = clustering
        self.ordered_linkage_matrix_ = linkage
        self.investable_mask_ = investable_mask
        self.ordering_ = ordering
        return self
