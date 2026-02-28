import numpy as np

from lapspec.graph import from_edge_list
from lapspec.laplacian import build_laplacian


def test_neumann_path_graph_laplacian():
    graph = from_edge_list(
        num_nodes=4,
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        boundary_nodes=np.array([0, 3], dtype=np.int64),
    )
    result = build_laplacian(graph, boundary="neumann")
    got = result.matrix.toarray()
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            [1.0, -inv_sqrt2, 0.0, 0.0],
            [-inv_sqrt2, 1.0, -0.5, 0.0],
            [0.0, -0.5, 1.0, -inv_sqrt2],
            [0.0, 0.0, -inv_sqrt2, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(got, expected)
    assert np.array_equal(result.active_nodes, np.array([0, 1, 2, 3], dtype=np.int64))


def test_dirichlet_reduces_to_interior_nodes():
    graph = from_edge_list(
        num_nodes=4,
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        boundary_nodes=np.array([0, 3], dtype=np.int64),
    )
    result = build_laplacian(graph, boundary="dirichlet")
    got = result.matrix.toarray()
    expected = np.array([[1.0, -0.5], [-0.5, 1.0]], dtype=np.float64)
    assert np.allclose(got, expected)
    assert np.array_equal(result.active_nodes, np.array([1, 2], dtype=np.int64))


def test_duplicate_edges_are_merged():
    graph = from_edge_list(
        num_nodes=3,
        edges=np.array([[0, 1], [1, 0], [1, 2]], dtype=np.int64),
        weights=np.array([1.0, 2.0, 1.0], dtype=np.float64),
        boundary_nodes=np.array([], dtype=np.int64),
    )
    result = build_laplacian(graph, boundary="neumann")
    got = result.matrix.toarray()
    off01 = -3.0 / np.sqrt(3.0 * 4.0)
    expected = np.array(
        [[1.0, off01, 0.0], [off01, 1.0, -0.5], [0.0, -0.5, 1.0]],
        dtype=np.float64,
    )
    assert np.allclose(got, expected)
