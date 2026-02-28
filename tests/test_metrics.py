import numpy as np

from lapspec.metrics import histogram_distance_matrix, spectrum_distance_matrix


def test_spectrum_distance_matrix_l2_and_wasserstein():
    spectra = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    d_l2 = spectrum_distance_matrix(spectra, metric="l2")
    d_w1 = spectrum_distance_matrix(spectra, metric="wasserstein")

    assert d_l2.shape == (3, 3)
    assert np.allclose(np.diag(d_l2), 0.0)
    assert np.allclose(d_l2, d_l2.T)
    assert np.isclose(d_l2[0, 2], 1.0)

    assert d_w1.shape == (3, 3)
    assert np.allclose(np.diag(d_w1), 0.0)
    assert np.allclose(d_w1, d_w1.T)
    assert np.isclose(d_w1[0, 2], 1.0 / 3.0, atol=1e-8)


def test_histogram_distance_matrix_l1_and_js():
    hist = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    d_l1 = histogram_distance_matrix(hist, metric="l1", normalize=True)
    d_js = histogram_distance_matrix(hist, metric="js", normalize=True)

    assert d_l1.shape == (3, 3)
    assert np.allclose(np.diag(d_l1), 0.0)
    assert np.allclose(d_l1, d_l1.T)
    assert np.isclose(d_l1[0, 1], 2.0, atol=1e-8)
    assert np.isclose(d_l1[0, 2], 1.0, atol=1e-8)

    assert d_js.shape == (3, 3)
    assert np.allclose(np.diag(d_js), 0.0)
    assert np.allclose(d_js, d_js.T)
    assert np.isclose(d_js[0, 1], np.log(2.0), atol=1e-8)

