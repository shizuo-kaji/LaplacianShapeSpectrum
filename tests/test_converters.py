import numpy as np

from lapspec.converters import image_to_graph, pointcloud_to_graph


def test_image_to_graph_2x2_connectivity4():
    mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    graph = image_to_graph(mask, connectivity=4, weight_mode="unit")
    assert graph.num_nodes == 4
    assert graph.edges.shape[0] == 4
    assert graph.boundary_nodes.shape[0] == 4


def test_pointcloud_knn_graph():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    graph = pointcloud_to_graph(points, method="knn", k=2, weight_mode="inverse_distance")
    assert graph.num_nodes == 4
    assert graph.edges.shape[0] > 0
    assert np.all(graph.weights > 0)

