import numpy as np

from lapspec.graph import from_edge_list
from lapspec.laplacian import build_laplacian
from lapspec.spectrum import compute_spectrum, fixed_length_spectrum, spectrum_histogram


def test_path3_spectrum_matches_known_values():
    graph = from_edge_list(
        num_nodes=3,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        boundary_nodes=np.array([], dtype=np.int64),
    )
    lap = build_laplacian(graph, boundary="neumann").matrix
    eigvals = compute_spectrum(lap, k=None)
    assert np.allclose(eigvals, np.array([0.0, 1.0, 2.0]), atol=1e-7)
    assert eigvals.min() >= -1e-10
    assert eigvals.max() <= 2.0 + 1e-10


def test_histogram_and_fixed_length():
    eigvals = np.array([0.0, 0.5, 1.5, 2.0], dtype=np.float64)
    hist, edges = spectrum_histogram(eigvals, bins=2, value_range=(0.0, 2.0), density=False)
    assert hist.shape == (2,)
    assert edges.shape == (3,)
    padded = fixed_length_spectrum(eigvals[:2], length=5, fill_value=0.0)
    assert padded.shape == (5,)
    assert np.allclose(padded[:2], np.array([0.0, 0.5]))
    assert np.allclose(padded[2:], 0.0)
