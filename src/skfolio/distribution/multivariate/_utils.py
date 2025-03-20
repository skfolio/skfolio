"""Utils module for multivariate distribution."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
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
        """Union of conditioned and conditioning sets."""
        return set(self.conditioned) | self.conditioning

    def __add__(self, other: "EdgeCondSets") -> "EdgeCondSets":
        """Combine two EdgeCondSets, merging conditioned and conditioning sets."""
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
        return self.__class__(conditioned=conditioned, conditioning=conditioning)

    def __repr__(self) -> str:
        """String representation of the EdgeCondSets."""
        if self.conditioning:
            return f"{self.conditioned} | {self.conditioning}"
        return str(self.conditioned)


class BaseNode(ABC):
    """Base class for Nodes of the R-vine tree.

    Parameters
    ----------
    ref : int or Edge
        For RootNode: reference of the variable index.
        For ChildNode: reference of the edge in the previous tree.

    Attributes
    ----------
    edges : set[Edge]
        The set of edges attached to this node.

    tree : Tree
        The Tree containing this Node.
    """

    def __init__(self, ref: Union[int, "Edge"]):
        self._ref = ref
        self.edges: set[Edge] = set()
        self.tree: Tree | None = None  # Reference to the Tree containing this Node

    @property
    def ref(self) -> Union[int, "Edge"]:
        """Return the reference of this node (read-only)."""
        return self._ref

    @abstractmethod
    def clear_cache(self, **kwargs):
        """Clear the cached pseudo-values and margin values (u and v)."""
        pass

    def __repr__(self) -> str:
        """String representation of the node."""
        return f"Node({self.ref})"


class RootNode(BaseNode):
    """Root Node of the R-vine tree.

    Parameters
    ----------
    ref : int
        The reference variable index.

    central : bool
        True if the node is central; otherwise, False.

    pseudo_values : ndarray, optional
        The pseudo-values of the Root Node.

    Attributes
    ----------
    edges : set[Edge]
        The set of edges attached to this node.

    tree : Tree
        The Tree containing this Node.
    """

    def __init__(
        self, ref: int, central: bool, pseudo_values: np.ndarray | None = None
    ):
        super().__init__(ref=ref)
        self.central = central
        self.pseudo_values = pseudo_values

    def clear_cache(self, **kwargs):
        """Clear the cached margin values (u and v)."""
        self.pseudo_values = None


class ChildNode(BaseNode):
    """Child Node of the R-vine tree.
    A child node is an edge from the previous tree.

    Parameters
    ----------
    ref : Edge
        The reference edge in the previous tree.

    Attributes
    ----------
    edges : set[Edge]
        The set of edges attached to this node.

    tree : Tree
        The Tree containing this Node.
    """

    def __init__(self, ref: "Edge"):
        super().__init__(ref=ref)
        # pointer from Edge to Node
        ref.ref_node = self
        self._central: bool | None = None
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._u_count: int = 0
        self._v_count: int = 0
        self._u_count_total: int = 0
        self._v_count_total: int = 0

    @property
    def central(self) -> bool:
        """Determine whether this node is considered central.
        It is inherited from the associated edge's centrality.

        Returns
        -------
        central: bool
           True if the node is central; otherwise, False.
        """
        if self._central is None:
            self._central = self.ref.strongly_central
        return self._central

    @property
    def u(self) -> np.ndarray:
        """Get the first margin value (u) for the node.

        It is obtained by computing the partial derivative of the copula with respect
        to v.

        Returns
        -------
        u : ndarray
            The u values for this node.
        """
        is_count = self.tree is not None and self.tree.is_count_visits

        if is_count:
            self._u_count_total += 1
        else:
            self._u_count += 1

        if self._u is None:
            X = self.ref.get_X()
            if is_count:
                self._u = np.array([np.nan])
            else:
                self._u = self.ref.copula.partial_derivative(X, first_margin=False)

        value = self._u

        # Clear cache
        if (
            not is_count
            and self._u_count_total != 0
            and self._u_count == self._u_count_total
        ):
            self._u = None
            self._u_count = 0

        return value

    @u.setter
    def u(self, value: np.ndarray) -> None:
        self._u = value

    @property
    def v(self) -> np.ndarray:
        """Get the second margin value (v) for the node.

        It is obtained by computing the partial derivative of the copula with respect
        to u.

        Returns
        -------
        v : ndarray
           The v values for this node.
        """
        is_count = self.tree is not None and self.tree.is_count_visits

        if is_count:
            self._v_count_total += 1
        else:
            self._v_count += 1

        if self._v is None:
            X = self.ref.get_X()
            if is_count:
                self._v = np.array([np.nan])
            else:
                self._v = self.ref.copula.partial_derivative(X, first_margin=True)

        value = self._v

        # Clear cache
        if (
            not is_count
            and self._v_count_total != 0
            and self._v_count == self._v_count_total
        ):
            self._v = None
            self._v_count = 0

        return value

    @v.setter
    def v(self, value: np.ndarray):
        self._v = value

    def get_var(self, is_left: bool) -> int:
        """Return the variable index associated with this node.

        The variable is determined by the conditioned set of the edge.

        Parameters
        ----------
        is_left : bool
            Indicates whether to select the left or right node.

        Returns
        -------
        var : int
            The variable index corresponding to this node.
        """
        if is_left is None:
            raise ValueError("is_left cannot be None for Child Nodes")
        var = self.ref.cond_sets.conditioned[0 if is_left else 1]
        return var

    def clear_cache(self, clear_count: bool):
        """Clear the cached margin values (u and v) and counts.

        Parameters
        ----------
        clear_count : bool
            If True, the visit counts are also reset.
        """
        self._u = None
        self._v = None
        if clear_count:
            self._u_count = 0
            self._v_count = 0
            self._u_count_total = 0
            self._v_count_total = 0


class Edge:
    """
    Represents an edge in an R-vine tree connecting two nodes.

    This class encapsulates the information for an edge between two nodes in an R-vine,
    including the associated copula, the dependence measure, and the conditioning sets.

    Attributes
    ----------
    node1 : RootNode | ChildNode
       The first node in the edge.

    node2 : RootNode | ChildNode
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
        node1: RootNode | ChildNode,
        node2: RootNode | ChildNode,
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
        if isinstance(self.node1, RootNode):
            return EdgeCondSets(
                conditioned=(self.node1.ref, self.node2.ref), conditioning=set()
            )
        return self.node1.ref.cond_sets + self.node2.ref.cond_sets

    def ref_to_nodes(self):
        """Connect this edge to its two nodes."""
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
        if isinstance(self.node1, RootNode):
            u = self.node1.pseudo_values
            v = self.node2.pseudo_values
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
        """String representation of the edge."""
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

    Attributes
    ----------
    edges : list[Edge]
        The list of edges in the Tree.

    is_count_visits : bool
        Whether to count the number of visit of each Node during sampling.
    """

    def __init__(self, level: int, nodes: list[RootNode | ChildNode]):
        self.level = level
        self._nodes = nodes
        for node in nodes:
            # pointer from Node to Tree
            node.tree = self
        self.edges = None
        self.is_count_visits: bool = False

    @property
    def nodes(self) -> list[RootNode | ChildNode]:
        """Return the tree nodes (read-only)."""
        return self._nodes

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
                dependence_matrix[i, j] = dep
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
                    dependence_matrix[i, j] = dep

        # Compute the minimum spanning tree
        mst = ssc.minimum_spanning_tree(-dependence_matrix, overwrite=True)

        edges = []
        # Extract the indices of the non-zero entries (edges)
        for i, j in zip(*mst.nonzero(), strict=True):
            edge = eligible_edges[(i, j)]
            # connect Nodes to Edges
            edge.ref_to_nodes()
            edges.append(edge)

        self.edges = edges

    def clear_cache(self, clear_count: bool = True):
        """Clear cached values for all nodes in the tree."""
        for node in self.nodes:
            node.clear_cache(clear_count=clear_count)

    def __repr__(self):
        """String representation of the tree."""
        return f"Tree(level {self.level})"


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
            dep = st.wasserstein_distance(X[:, 0], X[:, 1])
        case _:
            raise ValueError(f"Dependence method {dependence_method} not valid")
    return dep
