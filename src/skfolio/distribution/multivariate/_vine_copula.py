"""Vine Copula Estimation."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

# This is a novel implementation of R-vine that is fully OOP based on graph theory with
# new features to also capture C-like and clustered dependency structures.
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

import warnings
from collections import deque

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.parallel as skp
import sklearn.utils.validation as skv

from skfolio.distribution.copula import (
    BaseBivariateCopula,
    ClaytonCopula,
    GaussianCopula,
    GumbelCopula,
    JoeCopula,
    StudentTCopula,
    select_bivariate_copula,
)
from skfolio.distribution.copula._base import _UNIFORM_MARGINAL_EPSILON
from skfolio.distribution.multivariate._utils import DependenceMethod, Edge, Node, Tree
from skfolio.distribution.univariate import (
    BaseUnivariateDist,
    Gaussian,
    JohnsonSU,
    StudentT,
    select_univariate_dist,
)
from skfolio.utils.tools import validate_input_list

_UNIFORM_SAMPLE_EPSILON = 1e-14


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
    sampling and stress testing.

    Parameters
    ----------
    fit_marginals : bool, default=True
        Whether to fit marginal distributions to each asset before constructing the
        vine. If True, the data will be transformed to uniform marginals using the
        fitted CDFs.

    marginal_candidates : list[BaseUnivariateDist], optional
        Candidate univariate distribution estimators to fit the marginals.
        If None, defaults to `[Gaussian(), StudentT(), JohnsonSU()]`.

    copula_candidates : list[BaseBivariateCopula], optional
        Candidate bivariate copula estimators. If None, defaults to
        `[GaussianCopula(), StudentTCopula(), ClaytonCopula(), GumbelCopula(), JoeCopula()]`.

    max_depth : int, default=5
        Maximum vine tree depth (truncated level). Must be greater than 1.

    log_transform : bool, default=False
        If True, the simple returns provided as input will be transformed to log returns
        before fitting the vine copula. That is, each return R is transformed via
        r = log(1+R). After sampling, the generated log returns are converted back to
        simple returns using R = exp(r) - 1.

    central_assets : array-like of asset names or asset positions, optional
        Assets that should be centrally placed during vine construction.
        If None, no asset is forced to the center.
        If an array-like of **integer** is provided, its values must be asset positions.
        If an array-like of **string** is provided, its values must be asset names and
        the input `X` of the `fit` method must be a DataFrame with the assets names in
        columns. Assets marked as central are forced to occupy central positions in the
        vine, leading to C-like or clustered structure. This is needed for conditional
        sampling, where the conditioning assets should be central nodes.

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

    dependence_method : DependenceMethod, default=DependenceMethod.KENDALL_TAU
        The dependence measure used to compute edge weights for the MST.
        Possible values are:
            - KENDALL_TAU
            - MUTUAL_INFORMATION
            - WASSERSTEIN_DISTANCE

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

    marginal_distributions_ : list[BaseUnivariateDist]
        List of fitted marginal distributions (if fit_marginals is True).

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    Examples
    --------
    >>> from skfolio.datasets import load_factors_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution import VineCopula
    >>>
    >>> # Load historical prices and convert them to returns
    >>> prices = load_factors_dataset()
    >>> X = prices_to_returns(prices)
    >>>
    >>> # Instanciate the VineCopula model
    >>> vine = VineCopula()
    >>> # Fit the model
    >>> vine.fit(X)
    >>> # Display the vine trees and fitted copulas
    >>> vine.display_vine()
    >>> # Generate 10 samples from the fitted vine copula
    >>> samples = vine.sample(n_samples=10)
    >>>
    >>> # Set QUAL, SIZE and MTUM as central
    >>> vine = VineCopula(central_assets=["QUAL", "SIZE", "MTUM"])
    >>> vine.fit(X)
    >>> # Sample by conditioning on QUAL and SIZE returns
    >>> samples = vine.sample(
    ...    n_samples=4,
    ...    conditioning={
    ...        "QUAL": [-0.1, -0.2, -0.3, -0.4],
    ...        "SIZE": -0.2,
    ...        "MTUM": (None, -0.3) # MTUM sampled between -Inf and -30%
    ...    },
    ...)
    >>> # Plots Scatter matrix of sampled returns vs historical X
    >>> fig = vine.plot_scatter_matrix(X=X)
    >>> fig.show()
    >>>
    >>> # Plots univariate distributions of sampled returns vs historical X
    >>> fig = vine.plot_marginal_distributions(X=X)
    >>> fig.show()

    References
    ----------
    .. [1] "Selecting and estimating regular vine copulae and application to financial
        returns" Dißmann, Brechmann, Czado, and Kurowicka (2013).

    .. [2] "Growing simplified vine copula trees: improving Dißmann's algorithm"
        Krausa and Czado (2017).

    .. [3] "Pair-copula constructions of multiple dependence" Aas, Czado, Frigessi,
        Bakken (2009).

    .. [4] "Pair-Copula Constructions for Financial Applications: A Review"
        Aas and Czado (2016).

    .. [5] "Conditional copula simulation for systemic risk stress testing"
        Brechmann, Hendrich, Czado (2013)
    """

    trees_: list[Tree]
    marginal_distributions_: list[BaseUnivariateDist]
    n_features_in_: int
    feature_names_in_: np.ndarray
    central_assets_: set[int | str]

    def __init__(
        self,
        fit_marginals: bool = True,
        marginal_candidates: list[BaseUnivariateDist] | None = None,
        copula_candidates: list[BaseBivariateCopula] | None = None,
        max_depth: int = 5,
        log_transform: bool = False,
        central_assets: list[int | str] | None = None,
        dependence_method: DependenceMethod = DependenceMethod.KENDALL_TAU,
        aic: bool = True,
        independence_level: float = 0.05,
        n_jobs: int | None = None,
    ):
        self.fit_marginals = fit_marginals
        self.marginal_candidates = marginal_candidates
        self.copula_candidates = copula_candidates
        self.max_depth = max_depth
        self.log_transform = log_transform
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

        if self.log_transform:
            X = np.log(1 + X)

        depth = min(self.max_depth, n_assets - 1)

        self.central_assets_ = set()
        if self.central_assets is not None:
            self.central_assets_ = set(
                validate_input_list(
                    items=self.central_assets,
                    n_assets=n_assets,
                    assets_names=(
                        self.feature_names_in_
                        if hasattr(self, "feature_names_in_")
                        else None
                    ),
                    name="central_assets",
                    raise_if_string_missing=False,
                )
            )

        marginal_candidates = (
            self.marginal_candidates
            if self.marginal_candidates is not None
            else [
                Gaussian(),
                StudentT(),
                JohnsonSU(),
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
                skp.delayed(select_univariate_dist)(
                    X=X[:, [i]],
                    distribution_candidates=marginal_candidates,
                    aic=self.aic,
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
                        Node(
                            ref=i,
                            pseudo_values=X[:, i],
                            central=i in self.central_assets_,
                        )
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
        conditioning: dict[int | str : float | tuple[float, float] | npt.ArrayLike]
        | None = None,
        random_state: int | None = None,
    ):
        """Generate random samples from the vine copula.

        This method generates `n_samples` from the fitted vine copula model. The
        resulting samples represent multivariate observations drawn according to the
        dependence structure captured by the vine copula.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        conditioning : dict[int | str, float | tuple[float, float] | array-like], optional
            A dictionary specifying conditioning information for one or more assets.
            The dictionary keys are asset indices or names, and the values define how
            the samples are conditioned for that asset. Three types of conditioning
            values are supported:

            1. **Fixed value (float):**
               If a float is provided, all samples are generated under the condition
               that the asset takes exactly that value.

            2. **Bounds (tuple of two floats):**
               If a tuple `(min_value, max_value)` is provided, samples are generated
               under the condition that the asset's value falls within the specified
               bounds. Use `-np.Inf` for no lower bound or `np.Inf` for no upper bound.

            3. **Array-like (1D array):**
               If an array-like of length `n_samples` is provided, each sample is
               conditioned on the corresponding value in the array for that asset.

            **Important:** When using conditional sampling, it is recommended that the
            assets you condition on are set as central during the vine copula
            construction. This can be specified via the `central_assets` parameter in
            the vine copula instantiation.


        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the sample generation.

        Returns
        -------
        X : array-like of shape (n_samples, n_assets)
            A two-dimensional array where each row is a multivariate observation sampled
            from the vine copula.
        """
        skv.check_is_fitted(self)
        self.clear_cache()
        n_assets = self.n_features_in_
        rng = sku.check_random_state(random_state)

        uniform_cond_samples = {}
        initial_cond = {}
        if conditioning is None:
            conditioning_vars = set()
        else:
            conditioning_vars = validate_input_list(
                items=list(conditioning.keys()),
                n_assets=n_assets,
                assets_names=(
                    self.feature_names_in_
                    if hasattr(self, "feature_names_in_")
                    else None
                ),
                name="conditioning",
            )

            missing_central_vars = set(conditioning_vars).difference(
                self.central_assets_
            )

            if not set(conditioning_vars).issubset(set(range(n_assets))):
                raise ValueError(
                    "The keys of `conditioning` must be asset indices or names "
                    "from the input X."
                )
            if len(conditioning_vars) >= n_assets:
                raise ValueError(
                    "`conditioning` must be provided for strictly fewer assets "
                    "than the total."
                )

            if missing_central_vars:
                warnings.warn(
                    "When performing conditional sampling, it is recommended to set "
                    "conditioning assets as central during Vine Copula construction. "
                    "The following conditioning assets were not set as central: "
                    f"{missing_central_vars}. "
                    "This can be achieved by using the `central_assets` parameter.",
                    stacklevel=2,
                )

            for var, value in zip(
                conditioning_vars, conditioning.values(), strict=True
            ):
                if isinstance(value, tuple):
                    if len(value) != 2:
                        raise ValueError(
                            "When a tuple is provided in `conditioning`, it must be"
                            "of length 2, representing the conditioning bounds: "
                            "(min_value, max_value)"
                        )
                    min_value, max_value = value
                    if min_value is None:
                        min_value = -np.inf
                    if max_value is None:
                        max_value = np.inf

                    if min_value >= max_value:
                        raise ValueError(
                            "The conditioning tuple lower bound must be lower than "
                            "its upper bound."
                        )
                    initial_cond[var] = (min_value, max_value)

                    if self.log_transform:
                        if not np.isneginf(min_value):
                            min_value = np.log(1 + min_value)
                        if not np.isposinf(max_value):
                            max_value = np.log(1 + max_value)
                    if self.fit_marginals:
                        # Transform the bounds using the marginal CDF
                        dist = self.marginal_distributions_[var]
                        u_min = 0.0 if np.isneginf(min_value) else dist.cdf(min_value)
                        u_max = 1.0 if np.isposinf(max_value) else dist.cdf(max_value)
                    else:
                        u_min, u_max = min_value, max_value

                    # Sample uniformly in the transformed interval.
                    samples = rng.uniform(low=u_min, high=u_max, size=n_samples)

                elif np.isscalar(value):
                    initial_cond[var] = value
                    if self.log_transform:
                        value = np.log(1 + value)
                    if self.fit_marginals:
                        # Transform conditioning value using the fitted marginal CDF.
                        value = self.marginal_distributions_[var].cdf(value)
                    samples = np.full(n_samples, value)
                else:
                    samples = np.asarray(value)
                    initial_cond[var] = samples
                    if samples.ndim != 1 or samples.shape[0] != n_samples:
                        raise ValueError(
                            "When an array is provided in `conditioning`, it must be a "
                            f"1D array of length n_samples={n_samples}, got {samples.ndim}D of "
                            f"length {samples.shape[0]}"
                        )
                    if self.log_transform:
                        samples = np.log(1 + samples)
                    if self.fit_marginals:
                        # Transform conditioning samples using the fitted marginal CDF.
                        samples = self.marginal_distributions_[var].cdf(samples)
                uniform_cond_samples[var] = samples

        # Determine sampling order based on vine structure.
        sampling_order = self._sampling_order(conditioning_vars=conditioning_vars)

        # Generate independent Uniform(0,1) samples for non-conditioned nodes.
        X_rand = iter(rng.random(size=(n_samples, n_assets - len(conditioning_vars))).T)

        # Initialize samples for each node according to the sampling order.
        queue: deque[tuple[Node, bool]] = deque()
        for node, is_left in sampling_order:
            node_var = node.get_var(is_left)
            if node_var in conditioning_vars:
                init_samples = uniform_cond_samples[node_var]
            else:
                init_samples = next(X_rand)

            # Avoid Inf
            init_samples = np.clip(
                init_samples,
                a_min=_UNIFORM_SAMPLE_EPSILON,
                a_max=1 - _UNIFORM_SAMPLE_EPSILON,
            )

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

        # Avoid Inf
        samples = np.clip(
            samples,
            a_min=_UNIFORM_SAMPLE_EPSILON,
            a_max=1 - _UNIFORM_SAMPLE_EPSILON,
        )

        # Transform samples back to the original scale using inverse CDF
        # (if marginals were fitted).
        if self.fit_marginals:
            samples = np.hstack(
                [
                    dist.ppf(samples[:, [i]])
                    for i, dist in enumerate(self.marginal_distributions_)
                ]
            )

        # Reverse the log-return transformation if log_transform is True.
        if self.log_transform:
            # A known effect when sampling with log transform is that large sampling
            # log returns explode after applying the reverse log transform into
            # unrealistic simple returns.
            # A common approach is to apply Box-Cox transform above some threshold.
            # For financial returns, log returns above 200% are transformed linearly
            # instead of exponentially.
            box_cox_threshold = np.log(3)
            samples = np.where(
                samples <= box_cox_threshold,
                np.exp(samples) - 1,
                np.exp(box_cox_threshold) + (samples - box_cox_threshold) - 1,
            )

        # To avoid Inf values and numerical instability, conditional values converted to
        # uniforms are clipped to [1e-14 , 1-1e-14]. This means that if the conditional
        # value has an extremely low probability, its final value will be bounded by
        # [ppf(1e-14) , ppf(1-1e-14)]. To keep conditional values accurate even for
        # extremely low probability, we force them back in the final samples.
        for var, cond in initial_cond.items():
            if isinstance(cond, tuple):
                samples[:, var] = np.clip(samples[:, var], a_min=cond[0], a_max=cond[1])
            else:
                samples[:, var] = cond

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

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted Vine Copula."""
        skv.check_is_fitted(self)
        lines = []
        if self.fit_marginals:
            lines.append("Root Nodes")
            lines.append("-" * 10)
            for node, dist in zip(
                self.trees_[0].nodes, self.marginal_distributions_, strict=True
            ):
                lines.append(f"{node}: {dist.fitted_repr}")
            lines.append("")

        for tree in self.trees_:
            lines.append(str(tree))
            lines.append("-" * len(str(tree)))
            for edge in tree.edges:
                lines.append(str(edge))
            lines.append("")

        result_string = "\n".join(lines)
        return result_string

    def display_vine(self):
        """Display the vine trees and fitted copulas.
        Prints the structure of each tree and the details of each edge.
        """
        print(self.fitted_repr)

    def plot_scatter_matrix(
        self,
        X: npt.ArrayLike | None = None,
        n_samples: int | None = None,
        conditioning: dict[int | str : float | tuple[float, float] | npt.ArrayLike]
        | None = None,
        random_state: int | None = None,
        title: str = "Vine Copula Scatter Matrix",
    ) -> go.Figure:
        """
        Plot the vine copula scatter matrix by generating samples from the vine copula
        and comparing it versus the empirical distribution of `X` if provided.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_assets), optional
            If provided, it is used to plot the empirical scatter matrix for
            comparison versus the vine copula scatter matrix.

        n_samples : int, optional
            Number of samples to generate if historical data is not provided.

        conditioning : dict[int | str, float | tuple[float, float] | array-like], optional
            A dictionary specifying conditioning information for one or more assets.
            The dictionary keys are asset indices or names, and the values define how
            the samples are conditioned for that asset. Three types of conditioning
            values are supported:

            1. **Fixed value (float):**
               If a float is provided, all samples are generated under the condition
               that the asset takes exactly that value.

            2. **Bounds (tuple of two floats):**
               If a tuple `(min_value, max_value)` is provided, samples are generated
               under the condition that the asset's value falls within the specified
               bounds. Use `-np.Inf` for no lower bound or `np.Inf` for no upper bound.

            3. **Array-like (1D array):**
               If an array-like of length `n_samples` is provided, each sample is
               conditioned on the corresponding value in the array for that asset.

            **Important:** When using conditional sampling, it is recommended that the
            assets you condition on are set as central during the vine copula
            construction. This can be specified via the `central_assets` parameter in
            the vine copula instantiation.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the sample generation.

        title : str, default="Vine Copula Scatter Matrix"
            The title for the plot.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure object containing the scatter matrix.
        """
        n_assets = self.n_features_in_
        traces = []

        if n_samples is None:
            n_samples = 5000 if X is None else X.shape[0]

        sample = self.sample(
            n_samples=n_samples, conditioning=conditioning, random_state=random_state
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
                    size=5,
                    color="rgb(221,132,82)",
                    line=dict(width=0.2, color="white"),
                    opacity=0.7,
                ),
                name="Generated",
                showlegend=True,
            )
        )
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
                        opacity=0.6,
                    ),
                    name="Historical",
                    showlegend=True,
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(title=title)
        return fig

    def plot_marginal_distributions(
        self,
        X: npt.ArrayLike | None = None,
        n_samples: int | None = None,
        conditioning: dict[int | str : float | tuple[float, float] | npt.ArrayLike]
        | None = None,
        subset: list[int | str] | None = None,
        random_state: int | None = None,
        title: str = "Vine Copula Marginal Distributions",
    ) -> go.Figure:
        """
        Plot overlaid marginal distributions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_assets), optional
            Historical data where each column corresponds to an asset.

        n_samples : int, optional
            Number of samples to generate if historical data is not provided.

        conditioning : dict[int | str, float | tuple[float, float] | array-like], optional
            A dictionary specifying conditioning information for one or more assets.
            The dictionary keys are asset indices or names, and the values define how
            the samples are conditioned for that asset. Three types of conditioning
            values are supported:

            1. **Fixed value (float):**
               If a float is provided, all samples are generated under the condition
               that the asset takes exactly that value.

            2. **Bounds (tuple of two floats):**
               If a tuple `(min_value, max_value)` is provided, samples are generated
               under the condition that the asset's value falls within the specified
               bounds. Use `-np.Inf` for no lower bound or `np.Inf` for no upper bound.

            3. **Array-like (1D array):**
               If an array-like of length `n_samples` is provided, each sample is
               conditioned on the corresponding value in the array for that asset.

            **Important:** When using conditional sampling, it is recommended that the
            assets you condition on are set as central during the vine copula
            construction. This can be specified via the `central_assets` parameter in
            the vine copula instantiation.

        subset : list[int | str], optional
            Indices or names of assets to include in the plot. If None, all assets are
            used.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the sample generation.

        title : str, default="Vine Copula Marginal Distributions"
            The title for the plot.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure with overlaid univariate distributions for each asset.
        """
        n_assets = self.n_features_in_
        subset = subset or list(range(n_assets))
        colors = px.colors.qualitative.Plotly

        def _kde_trace(
            x: np.ndarray, opacity: float, color: str, name: str
        ) -> go.Scatter:
            kde = st.gaussian_kde(x)
            x = np.linspace(min(x), max(x), 500)
            return go.Scatter(
                x=x,
                y=kde(x),
                mode="lines",
                name=name,
                line=dict(color=color),
                fill="tozeroy",
                opacity=opacity,
                visible=True if i == 0 else "legendonly",
            )

        traces = []
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X should be a 2D array")
            if X.shape[1] != n_assets:
                raise ValueError(f"X should have {n_assets} columns")
            for i, s in enumerate(subset):
                traces.append(
                    _kde_trace(
                        x=X[:, s],
                        opacity=0.6,
                        color=colors[i % len(colors)],
                        name=f"{self.feature_names_in_[s]} Empirical",
                    )
                )
        if n_samples is None:
            n_samples = X.shape[0] if X is not None else 5000

        samples = self.sample(
            n_samples=n_samples, conditioning=conditioning, random_state=random_state
        )
        for i, s in enumerate(subset):
            traces.append(
                _kde_trace(
                    x=samples[:, s],
                    opacity=1.0,
                    color=colors[i % len(colors)],
                    name=f"{self.feature_names_in_[s]} Generated",
                )
            )
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="Probability Density",
        )
        fig.update_xaxes(
            tickformat=".0%",
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
