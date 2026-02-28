import numpy as np
from lapspec.batch import batch_features
from lapspec.graph import from_edge_list


def test_batch_feature_shape():
    g1 = from_edge_list(
        num_nodes=4,
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        boundary_nodes=np.array([0, 3], dtype=np.int64),
    )
    g2 = from_edge_list(
        num_nodes=4,
        edges=np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int64),
        boundary_nodes=np.array([0, 3], dtype=np.int64),
    )
    features = batch_features(
        [g1, g2],
        spectrum_k=5,
        hist_bins=7,
        boundary="neumann",
    )
    assert features.shape == (2, 12)
    assert np.isfinite(features).all()


def test_batch_feature_allows_custom_hist_range():
    g = from_edge_list(
        num_nodes=3,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        boundary_nodes=np.array([], dtype=np.int64),
    )
    f_default = batch_features(
        [g],
        spectrum_k=3,
        hist_bins=4,
        boundary="neumann",
        hist_range=(0.0, 2.0),
        density=False,
    )
    f_custom = batch_features(
        [g],
        spectrum_k=3,
        hist_bins=4,
        boundary="neumann",
        hist_range=(0.0, 1.0),
        density=False,
    )
    assert f_default.shape == (1, 7)
    assert f_custom.shape == (1, 7)
    assert np.isfinite(f_custom).all()
    assert not np.allclose(f_default[:, 3:], f_custom[:, 3:])
