import numpy as np

from lapspec.visualization import mds_projection


def test_mds_projection_from_distance_matrix():
    d = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )
    coords, mds = mds_projection(d, n_components=2, random_state=0)
    assert coords.shape == (3, 2)
    assert hasattr(mds, "stress_")
    assert np.isfinite(coords).all()

