"""Utils module for multivariate distribution"""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from enum import auto
from functools import cached_property
from itertools import combinations
from typing import Union

import numpy as np
import scipy.sparse.csgraph as ssc
import scipy.stats as st

from skfolio.utils.tools import AutoEnum


class DependenceMethod(AutoEnum):
    KENDALL_TAU = auto()
    MUTUAL_INFORMATION = auto()


@dataclass
class EdgeCondSets:
    conditioned: tuple[int, int]
    conditioning: set[int]

    def to_set(self) -> set[int]:
        return set(self.conditioned) | self.conditioning

    def __add__(self, other: "EdgeCondSets") -> "EdgeCondSets":
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Cannot add a EdgeCondSets with an object of type {type(other)}"
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
        if self.conditioning:
            return f"{self.conditioned} | {self.conditioning}"
        return str(self.conditioned)


class Node:
    """
    A node in T_k.
    - If k=1, 'ref' is an integer variable index.
    - If k>1, 'ref' is a reference to an edge from T_{k-1}'s list_of_edges.
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
        return not isinstance(self.ref, Edge)

    @property
    def ref(self) -> Union[int, "Edge"]:
        return self._ref

    @ref.setter
    def ref(self, value: Union[int, "Edge"]) -> None:
        if isinstance(value, Edge):
            # pointer from Edge to Node
            value.ref_node = self
        self._ref = value

    @property
    def u(self) -> np.ndarray:
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
        if self._central is None:
            if self.is_root_node:
                raise ValueError("central must be provided for root Nodes")
            self._central = self.ref.strongly_central
        return self._central

    def get_var(self, is_left: bool | None) -> int:
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
        self._u = None
        self._v = None

    def __repr__(self) -> str:
        return f"Node({self.ref})"


class Edge:
    """
    An edge in T_k, connecting two nodes in T_k.
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
        return self.node1.central or self.node2.central

    @cached_property
    def strongly_central(self) -> bool:
        return self.node1.central and self.node2.central

    @cached_property
    def dependence(self) -> float:
        X = self.get_X()
        match self.dependence_method:
            case DependenceMethod.KENDALL_TAU:
                dep = st.kendalltau(X[:, 0], X[:, 1]).statistic
            case _:
                raise ValueError(
                    f"Dependence method {self.dependence_method} not valid"
                )
        return dep

    @cached_property
    def cond_sets(self) -> EdgeCondSets:
        if self.node1.is_root_node:
            return EdgeCondSets(
                conditioned=(self.node1.ref, self.node2.ref), conditioning=set()
            )
        return self.node1.ref.cond_sets + self.node2.ref.cond_sets

    def ref_to_nodes(self):
        self.node1.edges.add(self)
        self.node2.edges.add(self)

    def get_X(self) -> np.ndarray:
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
        if self.node1 == other.node1:
            return True, True
        if self.node2 == other.node1:
            return False, True
        if self.node2 == other.node2:
            return False, False
        # self.node1 == other.node2
        raise ValueError("Edges are not correctly ordered")

    def share_one_node(self, other: "Edge") -> bool:
        """
        Returns True if e1 and e2 (from T_{k-1}) share exactly one node.
        e1 = (node1, node2); e2 = (node3, node4).
        We check intersection of {node1,node2} & {node3,node4}.
        """
        return len({self.node1, self.node2} & {other.node1, other.node2}) == 1

    def __repr__(self) -> str:
        if self.copula is None:
            return f"Edge({self.cond_sets})"
        return f"Edge({self.cond_sets}, {self.copula.fitted_repr()})"


class Tree:
    """
    Represents T_k in the R-vine: a tree structure with `nodes` and `edges`.
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
        """
        Constructs a Maximum Spanning Tree (MST).
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
                dep = abs(edge.dependence)
                dependence_matrix[i, j], dependence_matrix[j, i] = dep, dep
                eligible_edges[(i, j)] = edge

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
        for node in self.nodes:
            node.clear_cache()

    def __repr__(self):
        return f"Tree(level {self.level}): {len(self.nodes)} nodes"
