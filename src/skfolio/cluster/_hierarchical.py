"""Hierarchical Clustering estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from enum import auto

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
import sklearn.base as skb
import sklearn.utils.validation as skv

from skfolio.utils.fixes import create_dendrogram
from skfolio.utils.stats import assert_is_distance, compute_optimal_n_clusters
from skfolio.utils.tools import AutoEnum, default_asset_names


class LinkageMethod(AutoEnum):
    r"""Methods for calculating the distance between clusters in the linkage matrix.
    See the `Linkage Methods` section of `scipy.cluster.hierarchy.linkage`
    for full descriptions.

    Parameters
    ----------
    SINGLE : str
        Assigns

        .. math:: d(u,v) = \min(dist(u[i],v[j]))

        for all points :math:`i` in cluster :math:`u` and
        :math:`j` in cluster :math:`v`. This is also known as the
        Nearest Point Algorithm.

    COMPLETE : str
        Assigns

        .. math:: d(u, v) = \max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.

    AVERAGE : str
        Assigns

        .. math:: d(u,v) = \sum_{ij} \frac{d(u[i], v[j])}{(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm.

    WEIGHTED : str
        Assigns

        .. math:: d(u,v) = (dist(s,v) + dist(t,v))/2

        where cluster u was formed with cluster s and t and v
        is a remaining cluster in the forest (also called WPGMA).

    CENTROID : str
        Assigns

        .. math::
           dist(s,t) = ||c_s-c_t||_2

        where :math:`c_s` and :math:`c_t` are the centroids of
        clusters :math:`s` and :math:`t`, respectively.
        This is also known as the UPGMC
        algorithm.

    MEDIAN : str
    assigns :math:`d(s,t)` like the ``centroid`` method.
    This is also known as the WPGMC algorithm.

    WARD : str
        Uses the Ward variance minimization algorithm.
        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

           d(u,v) = \sqrt{\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \frac{|v|}
                               {T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.
    """

    SINGLE = auto()
    COMPLETE = auto()
    AVERAGE = auto()
    WEIGHTED = auto()
    CENTROID = auto()
    MEDIAN = auto()
    WARD = auto()


class HierarchicalClustering(skb.ClusterMixin, skb.BaseEstimator):
    r"""Hierarchical Clustering.

    Parameters
    ----------
    max_clusters : int, optional
        For coherent clustering, the algorithm finds a minimum threshold ``r`` so that
        the cophenetic distance between any two original observations in the same flat
        cluster is no more than ``r`` and no more than `max_clusters` flat clusters are
        formed. The default (`None`) is to estimate the maximal number of clusters
        based on the Two-Order Difference to Gap Statistic [1]_.

    linkage_method : LinkageMethod, default=LinkageMethod.WARD
        Methods for calculating the distance between clusters in the linkage matrix.
        See the `Linkage Methods` section of `scipy.cluster.hierarchy.linkage` for
        the full descriptions.
        The default is the Ward variance minimization algorithm `LinkageMethod.WARD`.

    Attributes
    ----------
    n_clusters_ : int
        Number of formed clusters.

    labels_ : ndarray of shape (n_assets,)
        Labels of each asset.

    linkage_matrix_ : ndarray of shape (n_assets - 1, 4)
        Linkage matrix computed from the distance matrix of the `distance_estimator`.

    condensed_distance_ : ndarray of shape (\\binom{n_assets}{2}, )
        The 1-D condensed distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Application of two-order difference to gap statistic".
        Yue, Wang & Wei (2009)
    """

    n_clusters_: int
    labels_: np.ndarray
    linkage_matrix_: np.ndarray
    condensed_distance_: np.ndarray

    def __init__(
        self,
        max_clusters: int | None = None,
        linkage_method: LinkageMethod = LinkageMethod.WARD,
    ):
        self.max_clusters = max_clusters
        self.linkage_method = linkage_method

    def fit(self, X: npt.ArrayLike, y: None = None) -> "HierarchicalClustering":
        """Fit the Hierarchical Equal Risk Contribution estimator.

        Parameters
        ----------
        X : array-like of shape (n_assets, n_assets)
            Distance matrix of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : HierarchicalClustering
            Fitted estimator.
        """
        X = self._validate_data(X)
        assert_is_distance(X)
        self.condensed_distance_ = scd.squareform(X, checks=False)
        self.linkage_matrix_ = sch.linkage(
            self.condensed_distance_,
            method=str(self.linkage_method.value),
            # Not needed for clustering, only for
            # visualization and can be slow. So we perform the optimal ordering only
            # in `plot_dendrogram`.
            optimal_ordering=False,
        )
        max_clusters = self.max_clusters
        if max_clusters is None:
            max_clusters = compute_optimal_n_clusters(
                distance=X,
                linkage_matrix=self.linkage_matrix_,
            )
        # Get the clusters from the linkage matrix
        labels_ = sch.fcluster(
            self.linkage_matrix_, t=max_clusters, criterion="maxclust"
        )
        labels_ -= 1  # Start at 0
        self.n_clusters_ = len(set(labels_))
        assert self.n_clusters_ == max(labels_) + 1 <= max_clusters
        self.labels_ = labels_
        return self

    def plot_dendrogram(self, heatmap: bool = True) -> go.Figure:
        """Plot the dendrogram.

        The blue lines represent distinct clusters composed of a single asset.
        The remaining colors represent clusters of more than one asset.

        When `heatmap` is set to True, the heatmap of the reordered distance matrix is
        displayed below the dendrogram and clusters are outlined with yellow squares.

        The number of clusters used in the plot is the same as the `n_clusters_`
        attribute if it exists, otherwise a default number is used corresponding to the
        number of cluster with a distance above 70% of the maximum cluster distance.

        Parameters
        ----------
        heatmap : bool, default=True
            If this is set to True, the distance heatmap is returned with the clustered
            outlined in yellow.

        Returns
        -------
        fig : Figure
            The dendrogram figure.
        """
        skv.check_is_fitted(self, "linkage_matrix_")
        linkage_matrix = sch.optimal_leaf_ordering(
            self.linkage_matrix_, self.condensed_distance_
        )

        n_assets = linkage_matrix.shape[0] + 1
        cophenetic_distance_threshold = linkage_matrix[-(self.n_clusters_ - 1), 2]

        if hasattr(self, "feature_names_in_"):
            asset_names = self.feature_names_in_
        else:
            asset_names = default_asset_names(n_assets=n_assets)

        if not heatmap:
            fig = create_dendrogram(
                np.ones(1),
                distfun=lambda x: None,
                linkagefun=lambda x: linkage_matrix,
                color_threshold=cophenetic_distance_threshold,
                labels=asset_names,
            )
            fig.update_layout(
                title="Dendrogram",
                width=800,
                height=400,
                showlegend=False,
                hovermode="closest",
                xaxis={"title": "Assets"},
                yaxis={"title": "Distance"},
            )
            return fig

        # Initialize figure by creating upper dendrogram
        fig = create_dendrogram(
            np.ones(1),
            orientation="bottom",
            distfun=lambda x: None,
            linkagefun=lambda x: linkage_matrix,
            color_threshold=cophenetic_distance_threshold,
            labels=asset_names,
        )

        for i in range(len(fig["data"])):
            fig["data"][i]["yaxis"] = "y2"

        # Create Side Dendrogram
        side_dendrogram = create_dendrogram(
            np.ones(1),
            orientation="right",
            distfun=lambda x: None,
            linkagefun=lambda x: linkage_matrix,
            color_threshold=cophenetic_distance_threshold,
            labels=asset_names,
        )
        for i in range(len(side_dendrogram["data"])):
            side_dendrogram["data"][i]["xaxis"] = "x2"

        # Add Side Dendrogram Data to Figure
        for data in side_dendrogram["data"]:
            fig.add_trace(data)

        # Create Heatmap
        ordered_asset_names = side_dendrogram["layout"]["yaxis"]["ticktext"]
        ordered_asset_names_idx = np.array(
            [np.argwhere(x == asset_names)[0][0] for x in ordered_asset_names]
        )
        assert np.array_equal(asset_names[ordered_asset_names_idx], ordered_asset_names)

        distance = scd.squareform(self.condensed_distance_, checks=False)
        heat_data = distance[ordered_asset_names_idx, :][:, ordered_asset_names_idx]

        heatmap = [
            go.Heatmap(
                x=ordered_asset_names,
                y=ordered_asset_names,
                z=heat_data,
                colorscale="Blues",
                name="",
            )
        ]

        heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
        heatmap[0]["y"] = side_dendrogram["layout"]["yaxis"]["tickvals"]

        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)

        # Outline clusters
        delta = heatmap[0]["x"][1] - heatmap[0]["x"][0]

        clusters_ids = self.labels_[ordered_asset_names_idx]

        for i in range(max(clusters_ids) + 1):
            c_ids = np.argwhere(clusters_ids == i).ravel()
            a = c_ids[0] * delta
            b = (c_ids[-1] + 1) * delta
            fig.add_shape(
                type="rect",
                x0=a,
                y0=a,
                x1=b,
                y1=b,
                line=dict(
                    color="gold",
                    width=2,
                ),
            )
        fig.update_layout(
            title="Dendrogram",
            width=800,
            height=800,
            showlegend=False,
            hovermode="closest",
            xaxis={
                "title": "Assets",
                "domain": [0.15, 1],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "ticks": "",
            },
            xaxis2={
                "domain": [0, 0.15],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            },
            yaxis={
                "title": "Assets",
                "domain": [0, 0.85],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
                "tickvals": fig["layout"]["xaxis"]["tickvals"],
                "ticktext": fig["layout"]["xaxis"]["ticktext"],
            },
            yaxis2={
                "domain": [0.825, 0.975],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            },
        )

        return fig
