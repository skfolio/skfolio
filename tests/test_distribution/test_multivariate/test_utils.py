from __future__ import annotations

import numpy as np
import pytest

from skfolio.distribution.multivariate._utils import (
    ChildNode,
    DependenceMethod,
    Edge,
    EdgeCondSets,
    RootNode,
    Tree,
    _dependence,
)


def test_edge_cond_sets_add():
    ecs1 = EdgeCondSets(conditioned=(1, 2), conditioning={0})
    ecs2 = EdgeCondSets(conditioned=(1, 3), conditioning={0})
    ecs3 = ecs1 + ecs2
    assert ecs3.conditioning == {0, 1}
    assert ecs3.conditioned == (2, 3)


def test_node_root_and_nonroot():
    # For a root node, create a node with pseudo_values.
    pseudo_vals = np.array([0.1, 0.2, 0.3])
    root1 = RootNode(ref=0, pseudo_values=pseudo_vals, central=True)
    root2 = RootNode(ref=1, pseudo_values=pseudo_vals, central=True)
    np.testing.assert_array_equal(root1.pseudo_values, pseudo_vals)
    edge = Edge(root1, root2)
    _ = ChildNode(ref=edge)


def test_edge_get_X():
    # Create two root nodes with dummy pseudo-observations.
    node1 = RootNode(ref=0, pseudo_values=np.array([0.1, 0.2]), central=True)
    node2 = RootNode(ref=1, pseudo_values=np.array([0.3, 0.4]), central=True)
    edge = Edge(node1=node1, node2=node2)
    X = edge.get_X()
    # Expected: first column is node1.u, second column is node2.u.
    np.testing.assert_array_equal(X[:, 0], node1.pseudo_values)
    np.testing.assert_array_equal(X[:, 1], node2.pseudo_values)


def test_tree_set_edges_from_mst():
    # Create a simple tree with 3 nodes (root nodes with integer refs).
    nodes = [
        RootNode(
            ref=i, pseudo_values=np.array([0.1 * (i + 1), 0.2 * (i + 1)]), central=True
        )
        for i in range(3)
    ]
    tree = Tree(level=0, nodes=nodes)
    # Set edges using Kendall's tau.
    tree.set_edges_from_mst(DependenceMethod.KENDALL_TAU)
    # Check that the tree has the correct number of edges.
    assert tree.edges is not None
    # For a tree with 3 nodes, there should be 2 edges.
    assert len(tree.edges) == 2


def test_dependence(X):
    X = X[["AAPL", "AMD"]]
    assert np.isclose(_dependence(X, DependenceMethod.KENDALL_TAU), 0.3077480329721)
    assert np.isclose(
        _dependence(X, DependenceMethod.MUTUAL_INFORMATION), 0.13061948, atol=1e-2
    )
    assert np.isclose(
        _dependence(X, DependenceMethod.WASSERSTEIN_DISTANCE), 0.012640552370723
    )


def test_edge_cond_sets_add_wrong_type():
    ecs = EdgeCondSets(conditioned=(1, 2), conditioning={0})
    with pytest.raises(TypeError, match="Cannot add a EdgeCondSets"):
        _ = ecs + 5


def test_edge_cond_sets_repr():
    assert repr(EdgeCondSets(conditioned=(1, 2), conditioning={0})) == "(1, 2) | {0}"
    assert repr(EdgeCondSets(conditioned=(1, 2), conditioning=set())) == "(1, 2)"


def test_root_node_repr():
    node = RootNode(ref=0, central=False)
    assert repr(node) == "Node(0)"


def test_child_node_get_var_none():
    edge = Edge(RootNode(ref=0, central=False), RootNode(ref=1, central=False))
    child = ChildNode(ref=edge)
    with pytest.raises(ValueError, match="is_left cannot be None for Child Nodes"):
        child.get_var(None)
    assert child.get_var(True) == 0
    assert child.get_var(False) == 1


def test_edge_shared_node_is_left_wrong_order():
    a, b, c = (RootNode(ref=i, central=False) for i in range(3))
    edge1 = Edge(a, b)
    edge2 = Edge(c, a)
    with pytest.raises(ValueError, match="Edges are not correctly ordered"):
        edge1.shared_node_is_left(edge2)
    assert edge1.share_one_node(edge2)


def test_edge_repr_without_copula():
    edge = Edge(RootNode(ref=0, central=False), RootNode(ref=1, central=False))
    assert repr(edge) == "Edge((0, 1))"


def test_tree_set_edges_from_mst_nan_dependence():
    # A constant margin makes Kendall's tau undefined (NaN).
    nodes = [
        RootNode(ref=0, central=False, pseudo_values=np.full(5, 0.5)),
        RootNode(ref=1, central=False, pseudo_values=np.linspace(0.1, 0.9, 5)),
    ]
    tree = Tree(level=0, nodes=nodes)
    with pytest.raises(RuntimeError, match="dependence_matrix contains NaNs"):
        tree.set_edges_from_mst(DependenceMethod.KENDALL_TAU)


def test_dependence_raise():
    with pytest.raises(ValueError, match="X must be a 2D array with exactly 2 columns"):
        _dependence(np.zeros((5, 3)), DependenceMethod.KENDALL_TAU)
    with pytest.raises(ValueError, match="Dependence method foo not valid"):
        _dependence(np.zeros((5, 2)), "foo")
