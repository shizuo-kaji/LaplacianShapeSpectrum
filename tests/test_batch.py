import numpy as np

from lapspec.batch import (
    FeatureConfig,
    batch_spectrum,
    batch_spectrum_histogram,
    spectrum_histogram,
)
from lapspec.graph import from_edge_list
from lapspec.metrics import histogram_distance_matrix, spectrum_distance_matrix


def _sample_graphs():
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
    g3 = from_edge_list(
        num_nodes=4,
        edges=np.array([[0, 1], [1, 3]], dtype=np.int64),
        boundary_nodes=np.array([0, 3], dtype=np.int64),
    )
    return [g1, g2, g3]


def test_batch_spectrum_shape():
    spectra = batch_spectrum(_sample_graphs()[:2], spectrum_k=5, boundary="neumann")
    assert spectra.shape == (2, 5)
    assert np.isfinite(spectra).all()


def test_batch_spectrum_histogram_supports_range_and_quantile():
    spectra = np.array(
        [
            [0.0, 0.2, 0.5, 0.7],
            [0.0, 0.8, 1.1, 1.6],
        ],
        dtype=np.float64,
    )

    h_range = batch_spectrum_histogram(
        spectra,
        bins=4,
        value_range=(0.0, 2.0),
        density=False,
    )
    h_quantile = batch_spectrum_histogram(
        spectra,
        bins=4,
        quantile_range=(0.0, 0.75),
        density=False,
    )

    assert h_range.shape == (2, 4)
    assert h_quantile.shape == (2, 4)
    assert np.isfinite(h_quantile).all()
    assert not np.allclose(h_range, h_quantile)


def test_spectrum_histogram_per_graph_vector_shape():
    graph = _sample_graphs()[0]
    config = FeatureConfig(
        spectrum_k=3,
        hist_bins=4,
        hist_range=(0.0, 2.0),
        density=False,
        boundary="neumann",
    )
    features = spectrum_histogram(graph, config)
    assert features.shape == (7,)
    assert np.isfinite(features).all()


def test_batch_outputs_can_be_used_with_metrics_module():
    graphs = _sample_graphs()

    spectra = batch_spectrum(graphs, spectrum_k=5, boundary="neumann")
    d_spec = spectrum_distance_matrix(spectra, metric="l2")

    histograms = batch_spectrum_histogram(
        spectra,
        bins=6,
        value_range=(0.0, 2.0),
        density=False,
    )
    d_hist = histogram_distance_matrix(histograms, metric="l1", normalize=False)

    assert d_spec.shape == (3, 3)
    assert np.allclose(d_spec, d_spec.T)
    assert np.allclose(np.diag(d_spec), 0.0)

    assert d_hist.shape == (3, 3)
    assert np.allclose(d_hist, d_hist.T)
    assert np.allclose(np.diag(d_hist), 0.0)
