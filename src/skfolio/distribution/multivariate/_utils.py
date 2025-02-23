"""Utils module for multivariate distribution"""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from enum import auto
from functools import cached_property
from itertools import combinations
from typing import Union

import numpy as np
import scipy.sparse.csgraph as ssc
import scipy.stats as st
import sklearn.feature_selection as sf

from skfolio.utils.tools import AutoEnum


class DependenceMethod(AutoEnum):
    """
    Enumeration of methods to measure bivariate dependence.

    Attributes
    ----------
    KENDALL_TAU
       Use Kendall's tau correlation coefficient.

    MUTUAL_INFORMATION
       Use mutual information estimated via a k-nearest neighbors method.

    WASSERSTEIN_DISTANCE
       Use the Wasserstein (Earth Mover's) distance.
    """

    KENDALL_TAU = auto()
    MUTUAL_INFORMATION = auto()
    WASSERSTEIN_DISTANCE = auto()


@dataclass
class EdgeCondSets:
    """
    Container for conditioning sets associated with an edge in an R-vine.

    Attributes
    ----------
    conditioned : tuple[int, int]
      A tuple of conditioned variable indices.

    conditioning : set[int]
      A set of conditioning variable indices.
    """

    conditioned: tuple[int, int]
    conditioning: set[int]

    def to_set(self) -> set[int]:
        """Union of conditioned and conditioning sets"""
        return set(self.conditioned) | self.conditioning

    def __add__(self, other: "EdgeCondSets") -> "EdgeCondSets":
        """Combine two EdgeCondSets, merging conditioned and conditioning sets"""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot add a EdgeCondSets with an object of type {type(other)}"
            )
        s1 = self.to_set()
        s2 = other.to_set()
        conditioning = s1 & s2
        conditioned = tuple(s1 ^ s2)
        # maintain order
        if conditioned[0] in other.conditioned:
            conditioned = conditioned[::-1]
        # noinspection PyArgumentList
        return self.__class__(conditioned=conditioned, conditioning=conditioning)

    def __repr__(self) -> str:
        """string representation of the EdgeCondSets"""
        if self.conditioning:
            return f"{self.conditioned} | {self.conditioning}"
        return str(self.conditioned)


class Node:
    """
    Represents a node in an R-vine tree.

    For k=1 (first tree), `ref` is an integer representing a variable index.
    For k > 1, `ref` is a reference to an Edge from the previous tree.

    Attributes
    ----------
    ref : int or Edge
        The reference to either a variable index or an edge in the previous tree.

    edges : set[Edge]
        The set of edges attached to this node.
    """

    def __init__(
        self,
        ref: Union[int, "Edge"],
        pseudo_values: np.ndarray | None = None,
        central: bool | None = None,
    ):
        self.ref = ref  # variable index OR a reference to an edge in the previous tree
        self.edges = set()
        self._central = central
        self._u = pseudo_values
        self._v = None

    @property
    def is_root_node(self) -> bool:
        """Determine if this node is a root node"""
        return not isinstance(self.ref, Edge)

    @property
    def ref(self) -> Union[int, "Edge"]:
        """Return the reference of this node."""
        return self._ref

    @ref.setter
    def ref(self, value: Union[int, "Edge"]) -> None:
        if isinstance(value, Edge):
            # pointer from Edge to Node
            value.ref_node = self
        self._ref = value

    @property
    def u(self) -> np.ndarray:
        """Get the first margin value (u) for the node.

        For non-root nodes, if u is not already computed, it is obtained by computing
        the partial derivative of the copula with respect to v.

        Returns
        -------
        u : ndarray
            The u values for this node.

        Raises
        ------
        ValueError
            If u is requested for a root node but is not provided.
        """
        if self._u is None:
            if self.is_root_node:
                raise ValueError("u must be provided for root Nodes")
            X = self.ref.get_X()
            self._u = self.ref.copula.partial_derivative(X, first_margin=False)
        return self._u

    @u.setter
    def u(self, value: np.ndarray) -> None:
        self._u = value

    @property
    def v(self) -> np.ndarray:
        """Get the second margin value (v) for the node.

        For non-root nodes, if v is not already computed, it is obtained by computing
        the partial derivative of the copula with respect to u.

        Returns
        -------
        ndarray
           The v values for this node.

        Raises
        ------
        ValueError
           If v is requested for a root node.
        """
        if self._v is None:
            if self.is_root_node:
                raise ValueError("v doesn't exist for root Nodes")
            X = self.ref.get_X()
            self._v = self.ref.copula.partial_derivative(X, first_margin=True)
        return self._v

    @v.setter
    def v(self, value: np.ndarray):
        self._v = value

    @property
    def central(self) -> bool:
        """Determine whether this node is considered central.

        For root nodes, it uses the central flag explicitly set.
        For non-root nodes, it is inherited from the associated edge's centrality.

        Returns
        -------
        central: bool
           True if the node is central; otherwise, False.

        Raises
        ------
        ValueError
           If centrality is not provided for a root node.
        """
        if self._central is None:
            if self.is_root_node:
                raise ValueError("central must be provided for root Nodes")
            self._central = self.ref.strongly_central
        return self._central

    def get_var(self, is_left: bool | None) -> int:
        """Return the variable index associated with this node.

        For a root node, the variable is given directly by ref.
        For non-root nodes, the variable is determined by the conditioned set of the
        edge.

        Parameters
        ----------
        is_left : bool or None
            For non-root nodes, indicates whether to select the left or right node.

        Returns
        -------
        var : int
            The variable index corresponding to this node.

        Raises
        ------
        ValueError
            If the input `is_left` is inconsistent with the node type.
        """
        if self.is_root_node:
            if is_left is not None:
                raise ValueError("is_left must be None for root Nodes")
            var = self.ref
        else:
            if is_left is None:
                raise ValueError("is_left cannot be None for non-starting Nodes")
            var = self.ref.cond_sets.conditioned[0 if is_left else 1]
        return var

    def clear_cache(self):
        """Clear the cached margin values (u and v)."""
        self._u = None
        self._v = None

    def __repr__(self) -> str:
        """String representation of the node"""
        return f"Node({self.ref})"


class Edge:
    """
    Represents an edge in an R-vine tree connecting two nodes.

    This class encapsulates the information for an edge between two nodes in an R-vine,
    including the associated copula, the dependence measure, and the conditioning sets.

    Attributes
    ----------
    node1 : Node
       The first node in the edge.

    node2 : Node
       The second node in the edge.

    dependence_method : DependenceMethod
       The method used to measure dependence between the two nodes.

    copula : object or None
       The fitted copula for this edge (if available).

    ref_node : Node or None
       A pointer to the node in the next tree constructed from this edge.
    """

    def __init__(
        self,
        node1: Node,
        node2: Node,
        dependence_method: DependenceMethod = DependenceMethod.KENDALL_TAU,
    ):
        self.node1 = node1
        self.node2 = node2
        self.dependence_method = dependence_method
        self.copula = None
        self.ref_node = None  # Pointer to the next tree Node

    @cached_property
    def weakly_central(self) -> bool:
        """Determine if the edge is weakly central.
        An edge is weakly central if at least one of its two nodes is central.
        """
        return self.node1.central or self.node2.central

    @cached_property
    def strongly_central(self) -> bool:
        """Determine if the edge is strongly central.
        An edge is strongly central if both of its nodes are central.
        """
        return self.node1.central and self.node2.central

    @cached_property
    def dependence(self) -> float:
        """Dependence measure between the two nodes.
        This is computed on the data from the edge using the specified dependence
        method.
        """
        X = self.get_X()
        dep = _dependence(X, dependence_method=self.dependence_method)
        return dep

    @cached_property
    def cond_sets(self) -> EdgeCondSets:
        """Compute the conditioning sets for the edge.
        For a root node edge, the conditioned set consists of the two variable indices.
        For non-root nodes, the conditioning sets are obtained by combining the
        conditioning sets of the two edges from the previous tree.
        """
        if self.node1.is_root_node:
            return EdgeCondSets(
                conditioned=(self.node1.ref, self.node2.ref), conditioning=set()
            )
        return self.node1.ref.cond_sets + self.node2.ref.cond_sets

    def ref_to_nodes(self):
        """Connect this edge to its two nodes"""
        self.node1.edges.add(self)
        self.node2.edges.add(self)

    def get_X(self) -> np.ndarray:
        """Retrieve the bivariate pseudo-observation data associated with the edge.

        For a root edge, this returns the pseudo-values from node1 and node2.
        For non-root edges, the appropriate margins (u or v) are selected
        based on the shared node order.

        Returns
        -------
        X : ndarray of shape (n_observations, 2)
            The bivariate pseudo-observation data corresponding to this edge.
        """
        if self.node1.is_root_node:
            u = self.node1.u
            v = self.node2.u
        else:
            is_left1, is_left2 = self.node1.ref.shared_node_is_left(self.node2.ref)
            u = self.node1.v if is_left1 else self.node1.u
            v = self.node2.v if is_left2 else self.node2.u
        X = np.stack([u, v]).T
        return X

    def shared_node_is_left(self, other: "Edge") -> tuple[bool, bool]:
        """Determine the ordering of shared nodes between this edge and another edge.

        If the two edges share one node, this method indicates for each edge whether the
        shared node is the left node.

        Parameters
        ----------
        other : Edge
            Another edge to compare with.

        Returns
        -------
        is_left1, is_left2 : tuple[bool, bool]
            A tuple (is_left1, is_left2) where is_left1 is True if the shared node is
            the left node of self and is_left2 is True if the shared node is the left
            node of other.

        Raises
        ------
        ValueError
            If the edges do not share exactly one node.
        """
        if self.node1 == other.node1:
            return True, True
        if self.node2 == other.node1:
            return False, True
        if self.node2 == other.node2:
            return False, False
        # self.node1 == other.node2
        raise ValueError("Edges are not correctly ordered")

    def share_one_node(self, other: "Edge") -> bool:
        """Check whether two edges share exactly one node.

        Parameters
        ----------
        other : Edge
            Another edge to compare with.

        Returns
        -------
        bool
            True if the two edges share exactly one node; otherwise, False.
        """
        return len({self.node1, self.node2} & {other.node1, other.node2}) == 1

    def __repr__(self) -> str:
        """String representation of the edge"""
        if self.copula is None:
            return f"Edge({self.cond_sets})"
        return f"Edge({self.cond_sets}, {self.copula.fitted_repr})"


class Tree:
    """
    Represents an R-vine tree at level k.

    A Tree consists of a set of nodes and the edges connecting them. It represents one
    level (k) in the R-vine structure.

    Parameters
    ----------
    level : int
        The tree level (k) in the R-vine.

    nodes : list[Node]
        A list of Node objects representing the nodes in this tree.
    """

    def __init__(self, level: int, nodes: list[Node]):
        self.level = level
        self.nodes = nodes
        self._edges = None

    @property
    def edges(self) -> list[Edge] | None:
        return self._edges

    @edges.setter
    def edges(self, values: list[Edge]) -> None:
        """List of edges in the tree"""
        if values is not None:
            if (
                not isinstance(values, list)
                or not values
                or not isinstance(values[0], Edge)
            ):
                raise ValueError("edges must be a non-empty list of Edges.")
            # connect Nodes to Edges
            for edge in values:
                edge.ref_to_nodes()
        self._edges = values

    def set_edges_from_mst(self, dependence_method: DependenceMethod) -> None:
        """Construct the Maximum Spanning Tree (MST) from the current nodes using
        the specified dependence method.

        The MST is built based on pairwise dependence measures computed between nodes.
        If any edge is (weakly) central, a central factor is added to the dependence
        measure to favor edges connected to central nodes.

        Parameters
        ----------
        dependence_method : DependenceMethod
            The method used to compute the dependence measure between nodes (e.g.,
            Kendall's tau).

        Returns
        -------
        None
        """
        n = len(self.nodes)
        dependence_matrix = np.zeros((n, n))
        eligible_edges = {}
        central = False
        for i, j in combinations(range(n), 2):
            node1 = self.nodes[i]
            node2 = self.nodes[j]
            if self.level == 0 or node1.ref.share_one_node(node2.ref):
                edge = Edge(
                    node1=node1, node2=node2, dependence_method=dependence_method
                )
                if not central and edge.weakly_central:
                    central = True
                # Negate the matrix to use minimum_spanning_tree for maximum spanning
                # Add a cst to ensure that even if dep is 0, we still build a valid MST
                dep = abs(edge.dependence) + 1e-5
                dependence_matrix[i, j], dependence_matrix[j, i] = dep, dep
                eligible_edges[(i, j)] = edge

        if np.any(np.isnan(dependence_matrix)):
            raise RuntimeError("dependence_matrix contains NaNs")

        if central:
            max_dep = np.max(dependence_matrix)
            for (i, j), edge in eligible_edges.items():
                if edge.weakly_central:
                    if edge.strongly_central:
                        central_factor = 3 * max_dep
                    else:
                        central_factor = 2 * max_dep
                    dep = dependence_matrix[i, j] + central_factor
                    dependence_matrix[i, j], dependence_matrix[j, i] = dep, dep

        # Compute the minimum spanning tree
        mst = ssc.minimum_spanning_tree(-dependence_matrix).toarray()
        # Extract the indices of the non-zero entries (edges)
        rows, cols = np.where(mst != 0)
        selected_edges = [
            eligible_edges[(i, j)] for i, j in zip(rows, cols, strict=True)
        ]
        self.edges = selected_edges

    def clear_cache(self):
        """Clear cached values for all nodes in the tree"""
        for node in self.nodes:
            node.clear_cache()

    def __repr__(self):
        """String representation of the tree"""
        return f"Tree(level {self.level}): {len(self.nodes)} nodes"


def _dependence(X, dependence_method: DependenceMethod) -> float:
    """Compute the dependence between two variables in X using the specified method.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        A 2D array of bivariate inputs (u, v), where u and v are assumed to lie in
        [0, 1].

    dependence_method : DependenceMethod
        The method to use for measuring dependence. Options are:
        - DependenceMethod.KENDALL_TAU
        - DependenceMethod.MUTUAL_INFORMATION
        - DependenceMethod.WASSERSTEIN_DISTANCE

    Returns
    -------
    dependence : float
        The computed dependence measure.

    Raises
    ------
    ValueError
        If X does not have exactly 2 columns or if an unsupported dependence method is
        provided.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be a 2D array with exactly 2 columns.")
    match dependence_method:
        case DependenceMethod.KENDALL_TAU:
            dep = st.kendalltau(X[:, 0], X[:, 1]).statistic
        case DependenceMethod.MUTUAL_INFORMATION:
            dep = sf.mutual_info_regression(X[:, 0].reshape(-1, 1), X[:, 1])[0]
        case DependenceMethod.WASSERSTEIN_DISTANCE:
            # noinspection PyTypeChecker
            dep = st.wasserstein_distance(X[:, 0], X[:, 1])
        case _:
            raise ValueError(f"Dependence method {dependence_method} not valid")
    return dep
