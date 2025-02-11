"""
Vine Copula Estimation
----------------------
"""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

# This is a novel implementation of R-vine that is fully OOP with new features to also
# capture C-like and clustered dependency structures.
#
# Unlike previous implementations that relied on a procedural paradigm, this version
# adopts an object-oriented design for constructing vine copulas.
#
# The validity of a vine tree is based on the following principle:
#   - In tree T_{k-1}, each edge represents a subset of k variables.
#   - Two such edges can form a new edge in tree T_k if and only if they share exactly
#     k-1 variables.
#
# In an OOP approach, we can use the below equivalent graphical adjacency rule:
# Two edges in T_{k-1} (which are “nodes” when building T_k) are connected by an edge
# in T_k if and only if they share exactly one node in T_{k-1}.
#
#  By using the concept of central assets in the MST, this novel implementation is able
#  to capture clustered or C-like dependency structures, allowing for more nuanced
#  representation of hierarchical relationships among assets and improving conditional
#  sampling.

from collections import deque

import numpy as np
import numpy.typing as npt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.parallel as skp
import sklearn.utils.validation as skv

from skfolio.distribution import ClaytonCopula, GaussianCopula, GumbelCopula, JoeCopula
from skfolio.distribution.copula import (
    BaseBivariateCopula,
    StudentTCopula,
    select_bivariate_copula,
)
from skfolio.distribution.copula._base import _UNIFORM_MARGINAL_EPSILON
from skfolio.distribution.multivariate._utils import DependenceMethod, Edge, Node, Tree
from skfolio.distribution.univariate import (
    BaseUnivariate,
    Gaussian,
    NormalInverseGaussian,
    StudentT,
    find_best_and_fit_univariate_dist,
)


class VineCopula(skb.BaseEstimator):
    """
    Regular Vine Copula Estimator.

    This model first fits the best univariate distribution for each asset, transforming
    the data to uniform marginals via the fitted CDFs. Then, it constructs a regular
    vine copula by sequentially selecting the best bivariate copula from a list of
    candidates for each edge in the vine using a maximum spanning tree algorithm based
    on a given dependence measure [1]_.

    Regular vines captures complex, fat-tailed dependencies and tail co-movements
    between asset returns.

    It also supports conditional sampling, enabling stress testing and scenario analysis
    by generating samples under specified conditions.

    Moreover, by marking some assets as central, this novel implementation is able to
    capture clustered or C-like dependency structures, allowing for more nuanced
    representation of hierarchical relationships among assets and improving conditional
    sampling.

    Parameters
    ----------
    fit_marginals : bool, default=True
        Whether to fit marginal distributions to each asset before constructing the
        vine. If True, the data will be transformed to uniform marginals using the
        fitted CDFs.

    marginal_candidates : list[BaseUnivariate], optional
        Candidate univariate distribution estimators to fit the marginals.
        If None, defaults to `[Gaussian(), StudentT(), NormalInverseGaussian()]`.

    copula_candidates : list[BaseBivariateCopula], optional
        Candidate bivariate copula estimators. If None, defaults to
        `[GaussianCopula(), StudentTCopula(), ClaytonCopula(), GumbelCopula(), JoeCopula()]`.

    max_depth : int, default=5
        Maximum vine tree depth to build. Must be greater than 1.

    central_assets : list[bool], optional
        A list indicating which assets should be centrally placed during vine
        construction. If None, no asset is forced to the center. Assets marked as
        central are forced to occupy central positions in the vine, leading to C-like
        or clustered structure. This is especially useful for conditional sampling,
        where the conditioning variables should be central nodes.

        For example:
          1) If only asset 1 is marked as central, it will be connected to all other
             assets in the first tree (yielding a C-like structure for the initial
             tree), with subsequent trees following the standard R-vine pattern.
          2) If asset 1 and asset 2 are marked as central, they will be connected
             together and the remaining assets will connect to either asset 1 or asset 2
             (forming a clustered structure in the initial trees). In the next tree,
             the edge between asset 1 and asset 2 becomes the central node, with
             subsequent trees following the standard R-vine structure.
          3) This logic extends naturally to more than two central assets.

       Marking conditioning assets as central during vine construction ensures that they
       are placed centrally, which is critical for effective conditional sampling.

    dependence_method : DependenceMethod, default=DependenceMethod.KENDALL_TAU
        The dependence measure used to compute edge weights for the MST.

    aic : bool, default=True
        If True, use AIC for univariate and copula selection; otherwise, use BIC.

    independence_level : float, default=0.05
        Significance level used for the Kendall tau independence test during copula
        fitting. If the p-value exceeds this threshold, the null hypothesis of
        independence is accepted, and the pair copula is modeled using the
        `IndependentCopula()` class.

    n_jobs : int, optional
        The number of jobs to run in parallel for `fit` of all `estimators`.
        The value `-1` means using all processors.
        The default (`None`) means 1 unless in a `joblib.parallel_backend` context.

    Attributes
    ----------
    trees_ : list[Tree]
        List of constructed vine trees.

    marginal_distributions_ : list[BaseUnivariate]
        List of fitted marginal distributions (if fit_marginals is True).

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Selecting and estimating regular vine copulae and application to financial
        returns" Dißmann, Brechmann, Czado, and Kurowicka (2013).

    .. [2] "Growing simplified vine copula trees: improving Dißmann's algorithm"
        Krausa and Czado (2017).

    .. [3] "Pair-Copula Constructions for Financial Applications: A Review"
        Aas and Czado (2016).
    """

    trees_: list[Tree]
    marginal_distributions_: list[BaseUnivariate]
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        fit_marginals: bool = True,
        marginal_candidates: list[BaseUnivariate] | None = None,
        copula_candidates: list[BaseBivariateCopula] | None = None,
        max_depth: int = 5,
        central_assets: list[bool] | None = None,
        dependence_method: DependenceMethod = DependenceMethod.KENDALL_TAU,
        aic: bool = True,
        independence_level: float = 0.05,
        n_jobs: int | None = None,
    ):
        self.fit_marginals = fit_marginals
        self.marginal_candidates = marginal_candidates
        self.copula_candidates = copula_candidates
        self.max_depth = max_depth
        self.central_assets = central_assets
        self.dependence_method = dependence_method
        self.aic = aic
        self.independence_level = independence_level
        self.n_jobs = n_jobs

    def fit(self, X: npt.ArrayLike, y=None) -> "VineCopula":
        """
        Fit the Vine Copula model to the data.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : VineCopula
            The fitted VineCopula instance.

        Raises
        ------
        ValueError
            If the number of assets is less than or equal to 2, or if max_depth <= 1.
        """
        X = skv.validate_data(self, X, dtype=np.float64)
        n_observations, n_assets = X.shape

        if n_assets <= 2:
            raise ValueError(
                f"The number of assets must be higher than 2, got {n_assets}"
            )
        if self.max_depth <= 1:
            raise ValueError(f"`max_depth` must be higher than 1, got {self.max_depth}")

        depth = min(self.max_depth, n_assets - 1)

        central_assets = (
            self.central_assets
            if self.central_assets is not None
            else [False] * n_assets
        )

        marginal_candidates = (
            self.marginal_candidates
            if self.marginal_candidates is not None
            else [
                Gaussian(),
                StudentT(),
                NormalInverseGaussian(),
            ]
        )

        copula_candidates = (
            self.copula_candidates
            if self.copula_candidates is not None
            else [
                GaussianCopula(),
                StudentTCopula(),
                ClaytonCopula(),
                GumbelCopula(),
                JoeCopula(),
            ]
        )

        if self.fit_marginals:
            # Fit marginal distributions
            # noinspection PyCallingNonCallable
            self.marginal_distributions_ = skp.Parallel(n_jobs=self.n_jobs)(
                skp.delayed(find_best_and_fit_univariate_dist)(
                    X=X[:, [i]], distribution_candidates=marginal_candidates
                )
                for i in range(n_assets)
            )
            # Transform data to uniform marginals using the fitted CDF
            X = np.hstack(
                [
                    dist.cdf(X[:, [i]])
                    for i, dist in enumerate(self.marginal_distributions_)
                ]
            )

        # Ensure values are within [0, 1].
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError(
                "X must be in the interval `[0, 1]`, typically obtained via marginal "
                "CDF transformation."
            )
        # Handle potential numerical issues by ensuring X doesn't contain exact 0 or 1.
        X = np.clip(X, _UNIFORM_MARGINAL_EPSILON, 1 - _UNIFORM_MARGINAL_EPSILON)

        trees: list[Tree] = []
        for level in range(depth):
            if level == 0:
                # Initial tree: each asset is a node.
                tree = Tree(
                    level=level,
                    nodes=[
                        Node(ref=i, pseudo_values=X[:, i], central=central_assets[i])
                        for i in range(n_assets)
                    ],
                )
            else:
                # Subsequent trees: nodes represent edges from the previous tree.
                tree = Tree(
                    level=level, nodes=[Node(ref=edge) for edge in trees[-1].edges]
                )
            # Set edges via Maximum Spanning Tree using the specified dependence method.
            tree.set_edges_from_mst(dependence_method=self.dependence_method)
            assert len(tree.edges) == n_assets - level - 1

            # Fit bivariate copulas for each edge.
            # noinspection PyCallingNonCallable
            copulas = skp.Parallel(n_jobs=self.n_jobs)(
                skp.delayed(select_bivariate_copula)(
                    X=edge.get_X(),
                    copula_candidates=copula_candidates,
                    aic=self.aic,
                    independence_level=self.independence_level,
                )
                for edge in tree.edges
            )
            for edge, copula in zip(tree.edges, copulas, strict=True):
                edge.copula = copula

            # Clear previous tree cache to free memory.
            if level > 0:
                trees[-1].clear_cache()

            trees.append(tree)

        # Attach a node to each terminal edge (used for sampling).
        for edge in trees[-1].edges:
            Node(ref=edge)

        self.trees_ = trees
        return self

    def sample(
        self,
        n_samples: int = 1,
        random_state: int | None = None,
        conditioning_samples: dict[int | str : npt.ArrayLike] | None = None,
    ):
        """Generate random samples from the vine copula.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the sample generation.

        conditioning_samples : dict, optional
            Dictionary mapping asset indices or names to 1D arrays of length n_samples
            to condition on specific values.

        Returns
        -------
        X : array-like of shape (n_samples, n_assets)
            An array where each row represents a multivariate observation.

        Raises
        ------
        ValueError
            If conditioning_samples contains invalid keys or arrays with improper
            dimensions.
        """
        skv.check_is_fitted(self)
        self.clear_cache()
        n_assets = self.n_features_in_

        if conditioning_samples is None:
            conditioning_vars = set()
        else:
            conditioning_samples = conditioning_samples.copy()
            conditioning_vars = set(conditioning_samples.keys())
            if not conditioning_vars.issubset(set(range(n_assets))):
                raise ValueError(
                    "The keys of `conditioning_samples` must be asset indices or names "
                    "from the input X."
                )
            if len(conditioning_vars) >= n_assets:
                raise ValueError(
                    "Conditioning samples must be provided for strictly fewer assets "
                    "than the total."
                )

            for var, sample in conditioning_samples.items():
                sample = np.asarray(sample)
                if sample.ndim != 1:
                    raise ValueError("Each conditioning_samples should be a 1D array")
                if sample.shape[0] != n_samples:
                    raise ValueError(
                        f"Each conditioning_samples should be of length "
                        f"n_samples={n_samples}"
                    )
                if self.fit_marginals:
                    # Transform conditioning samples using the fitted marginal CDF.
                    sample = self.marginal_distributions_[var].cdf(sample)
                conditioning_samples[var] = sample

        # Determine sampling order based on vine structure.
        sampling_order = self._sampling_order(conditioning_vars=conditioning_vars)

        # Generate independent Uniform(0,1) samples for non-conditioned nodes.
        rng = sku.check_random_state(random_state)
        X_rand = iter(rng.random(size=(n_samples, n_assets - len(conditioning_vars))).T)

        # Initialize samples for each node according to the sampling order.
        queue: deque[tuple[Node, bool]] = deque()
        for node, is_left in sampling_order:
            node_var = node.get_var(is_left)
            if node_var in conditioning_vars:
                init_samples = conditioning_samples[node_var]
            else:
                init_samples = next(X_rand)

            # For the root node, is_left is None.
            if is_left is None:
                node.u = init_samples
            else:
                queue.append((node, is_left))
                if is_left:
                    node.u = init_samples
                else:
                    node.v = init_samples

        # Propagate samples through the vine structure.
        while queue:
            node, is_left = queue.popleft()
            edge = node.ref
            if edge.node1.is_root_node:
                if is_left:
                    x = np.stack([node.u, edge.node2.u]).T
                    edge.node1.u = edge.copula.inverse_partial_derivative(x)
                else:
                    x = np.stack([node.v, edge.node1.u]).T
                    edge.node2.u = edge.copula.inverse_partial_derivative(x)
            else:
                is_left1, is_left2 = edge.node1.ref.shared_node_is_left(edge.node2.ref)
                if is_left:
                    x = np.stack([node.u, edge.node2.v if is_left2 else edge.node2.u]).T
                    u = edge.copula.inverse_partial_derivative(x)
                    if is_left1:
                        edge.node1.v = u
                    else:
                        edge.node1.u = u
                    queue.appendleft((edge.node1, not is_left1))
                else:
                    x = np.stack([node.v, edge.node1.v if is_left1 else edge.node1.u]).T
                    u = edge.copula.inverse_partial_derivative(x)
                    if is_left2:
                        edge.node2.v = u
                    else:
                        edge.node2.u = u
                    queue.appendleft((edge.node2, not is_left2))

        # Collect samples from the root tree.
        samples = np.stack([node.u for node in self.trees_[0].nodes]).T
        self.clear_cache()

        # Transform samples back to the original scale using inverse CDF
        # (if marginals were fitted).
        if self.fit_marginals:
            samples = np.hstack(
                [
                    dist.ppf(samples[:, [i]])
                    for i, dist in enumerate(self.marginal_distributions_)
                ]
            )

        return samples

    def clear_cache(self):
        """Clear cached intermediate results in the vine trees."""
        for tree in self.trees_:
            tree.clear_cache()
        for edge in self.trees_[-1].edges:
            edge.ref_node.clear_cache()

    def _conditioning_count(self) -> np.ndarray:
        """Compute cumulative counts of conditioning set occurrences in the vine.

        Returns
        -------
        conditioning_counts : ndarray of shape (n_trees, n_assets)
            Array of cumulative conditioning counts.
        """
        n_assets = self.n_features_in_
        conditioning_counts = [
            [s for edge in tree.edges for s in edge.cond_sets.conditioning]
            for tree in self.trees_
        ]
        conditioning_counts = np.stack(
            [np.bincount(cond, minlength=n_assets) for cond in conditioning_counts]
        ).cumsum(axis=0)
        return conditioning_counts

    def _sampling_order(
        self, conditioning_vars: set[int] | None = None
    ) -> list[tuple[Node, bool]]:
        """
        Determine the optimal sampling order for the vine copula.

        The sampling order is derived using a top-down elimination strategy that is
        analogous to finding a perfect elimination ordering for chordal graphs. In our
        vine copula, each conditional density is expressed as a product of bivariate
        copula densities and univariate margins. The algorithm starts with the deepest
        tree (i.e., the tree with the largest conditioning sets) and selects a variable
        from its conditioned set. This variable is then marginalized out, effectively
        generating a new sub-vine with a new deepest node. This process is repeated
        until the first tree level is reached and all n variables have been ordered.

        Choosing the optimal sampling order in this manner simplifies the inversion of
        conditional CDFs, thereby improving numerical stability. At each elimination
        step, among the candidate variables, the one that appears least frequently in
        the conditioning sets of shallower trees is chosen, ensuring that variables
        critical for conditional sampling occupy central roles.

        Parameters
        ----------
        conditioning_vars : set[int], optional
            A set of asset indices for which conditioning samples are provided.
            If specified, these assets will be prioritized during the sampling order
            determination.

        Returns
        -------
        sampling_order : list[tuple(Node, bool | None)]
            A list of tuples representing the optimal sampling order. Each tuple
            contains:
              - A Node object corresponding to an asset or an edge in the vine.
              - A boolean flag indicating whether the left branch is used for sampling
               at that node, or None if the node is the root.
        """
        conditioning_vars = conditioning_vars or set()
        sampling_order: list[tuple[Node, bool | None]] = []
        n_assets = self.n_features_in_
        conditioning_counts = self._conditioning_count()

        edges = self.trees_[-1].edges
        if len(edges) == 1:
            edge = edges[0]
            is_left = _is_left(
                edge=edge,
                conditioning_vars=conditioning_vars,
                conditioning_counts=conditioning_counts[-1],
            )
            sampling_order.append((edge.ref_node, is_left))
        else:
            # For truncated trees, select edges minimizing conditioning counts.
            remaining = edges
            visited: set[Edge] = set()
            prev_visited: set[Edge] = set()
            edge, is_left = None, None
            while remaining:
                selected = []
                costs = []
                for edge in remaining:
                    c1 = len(edge.node1.edges - prev_visited) == 1
                    c2 = len(edge.node2.edges - prev_visited) == 1
                    if c1 or c2:
                        if c1 and c2:
                            # Last node
                            is_left = _is_left(
                                edge=edge,
                                conditioning_vars=conditioning_vars,
                                conditioning_counts=conditioning_counts[-1],
                            )
                        else:
                            is_left = c1
                        costs.append(
                            conditioning_counts[
                                -1, edge.cond_sets.conditioned[0 if is_left else 1]
                            ]
                        )
                        selected.append((edge.ref_node, is_left))
                        visited.add(edge)
                remaining = [x for x in remaining if x not in visited]
                # Sort selected nodes by cost and add to sampling order.
                ordered_selected = sorted(
                    zip(costs, selected, strict=True), key=lambda pair: pair[0]
                )
                sampling_order.extend(node for _, node in ordered_selected)
                prev_visited = visited.copy()

        # Marginalization: Peeling off the tree level by level.
        # We have two valid choices (two conditioned vars: left or right),
        # we chose the conditioned var with the less conditioning cost
        i = 2
        while True:
            node = edge.node2 if is_left else edge.node1
            if node.is_root_node:
                sampling_order.append((node, None))
                break
            else:
                edge = node.ref
                is_left = _is_left(
                    edge=edge,
                    conditioning_vars=conditioning_vars,
                    conditioning_counts=conditioning_counts[-i],
                )
                sampling_order.append((node, is_left))
            i += 1

        # Replace nodes with conditioning samples where applicable.
        if conditioning_vars:
            for i, (node, is_left) in enumerate(sampling_order):
                node_var = node.get_var(is_left)
                if node_var in conditioning_vars:
                    sampling_order[i] = (self.trees_[0].nodes[node_var], None)

        sampling_order = sampling_order[::-1]
        if not (len(set(sampling_order)) == len(sampling_order) == n_assets):
            raise ValueError(
                "Sampling order computation failed: ordering is not unique or complete."
            )
        return sampling_order

    def display_vine_structure(self):
        """Display the vine trees and fitted copulas.
        Prints the structure of each tree and the details of each edge.
        """
        skv.check_is_fitted(self)

        for tree in self.trees_:
            print(tree)
            for edge in tree.edges:
                print(edge)
            print()

    def plot_scatter_matrix(
        self,
        X: npt.ArrayLike | None = None,
        n_samples: int | None = None,
        conditioning_samples=None,
    ) -> go.Figure:
        """
        Plot a scatter matrix comparing historical and generated samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_assets), optional
            Historical data for assets; each column corresponds to an asset.

        n_samples : int, optional
            Number of samples to generate if historical data is not provided.

        conditioning_samples : dict, optional
            Conditioning samples as provided in the `sample` method.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure object containing the scatter matrix.
        """
        n_assets = self.n_features_in_
        traces = []
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X should be an 2D array")
            if X.shape[1] != n_assets:
                raise ValueError(f"X should have {n_assets} columns")
            # noinspection PyTypeChecker
            traces.append(
                go.Splom(
                    dimensions=[
                        {"label": self.feature_names_in_[i], "values": X[:, i]}
                        for i in range(n_assets)
                    ],
                    showupperhalf=False,
                    diagonal_visible=False,
                    marker=dict(
                        size=5,
                        color="rgb(85,168,104)",
                        line=dict(width=0.2, color="white"),
                    ),
                    name="Historical",
                    showlegend=True,
                )
            )

        if n_samples is None:
            n_samples = 5000 if X is None else X.shape[0]

        sample = self.sample(
            n_samples=n_samples, conditioning_samples=conditioning_samples
        )

        # noinspection PyTypeChecker
        traces.append(
            go.Splom(
                dimensions=[
                    {"label": self.feature_names_in_[i], "values": sample[:, i]}
                    for i in range(n_assets)
                ],
                showupperhalf=False,
                diagonal_visible=False,
                marker=dict(
                    size=5, color="rgb(221,132,82)", line=dict(width=0.2, color="white")
                ),
                name="Generated",
                showlegend=True,
            )
        )
        fig = go.Figure(data=traces)
        fig.update_layout(title="Scatter Matrix")
        return fig

    def plot_univariate_distributions(
        self,
        X: npt.ArrayLike | None = None,
        n_samples: int | None = None,
        conditioning_samples=None,
        subset: list[int | str] | None = None,
    ) -> go.Figure:
        """
        Plot overlaid univariate distributions for all assets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_assets), optional
            Historical data where each column corresponds to an asset.

        n_samples : int, optional
            Number of samples to generate if historical data is not provided.

        conditioning_samples : dict, optional
            Conditioning samples as in `sample`.

        subset : list[int | str], optional
            Indices or names of assets to include in the plot. If None, all assets are used.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure with overlaid univariate distributions for each asset.
        """
        n_assets = self.n_features_in_
        subset = subset or list(range(n_assets))

        # Process Historical data if provided.
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X should be a 2D array")
            if X.shape[1] != n_assets:
                raise ValueError(f"X should have {n_assets} columns")

        # Determine n_samples for Generated data.
        if n_samples is None:
            n_samples = X.shape[0] if X is not None else 5000

        samples = self.sample(
            n_samples=n_samples, conditioning_samples=conditioning_samples
        )

        # Prepare lists to hold data arrays and labels.
        dist_data = []
        group_labels = []

        # If Historical data exists, add one distribution per asset.
        if X is not None:
            for i in subset:
                dist_data.append(X[:, i])
                group_labels.append(f"{self.feature_names_in_[i]} Historical")

        # Add the Generated data for each asset.
        for i in subset:
            dist_data.append(samples[:, i])
            group_labels.append(f"{self.feature_names_in_[i]} Generated")

        # Create the distplot. All distributions will be plotted on the same axes.
        fig = ff.create_distplot(
            dist_data, group_labels, show_hist=False, show_curve=True, show_rug=True
        )

        # Update layout.
        fig.update_layout(
            title={
                "text": "Univariate Distributions (All Assets)",
                "x": 0.5,
                "xanchor": "center",
            },
        )
        return fig


def _is_left(
    edge: Edge, conditioning_vars: set[int], conditioning_counts: np.ndarray
) -> bool:
    """
    Determine whether the left branch should be used for an edge based on conditioning.

    Parameters
    ----------
    edge : Edge
        The edge for which to decide the branch.

    conditioning_vars : set[int]
        Set of asset indices with conditioning samples.

    conditioning_counts : np.ndarray
        Array of conditioning counts for each asset.

    Returns
    -------
    bool
        True if the left branch is preferred, False otherwise.
    """
    v1, v2 = edge.cond_sets.conditioned
    if v1 in conditioning_vars and v2 not in conditioning_vars:
        return False
    if v1 not in conditioning_vars and v2 in conditioning_vars:
        return True
    # noinspection PyTypeChecker
    return conditioning_counts[v1] <= conditioning_counts[v2]
