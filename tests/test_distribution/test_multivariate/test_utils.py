import numpy as np

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
