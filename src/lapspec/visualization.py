from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


def pca_projection(
    features: NDArray[np.float64],
    n_components: int = 2,
    standardize: bool = True,
    random_state: int = 0,
) -> tuple[NDArray[np.float64], PCA]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] < 2:
        raise ValueError("PCA needs at least two samples")
    if n_components <= 0 or n_components > min(x.shape):
        raise ValueError("n_components is out of valid range")

    if standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(x)
    return coords, pca


def plot_pca(
    features: NDArray[np.float64],
    labels: NDArray[np.float64] | None = None,
    save_path: str | Path | None = None,
    title: str = "PCA of Laplacian Features",
) -> tuple[NDArray[np.float64], PCA]:
    coords, pca = pca_projection(features, n_components=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.9)
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="label")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)

    return coords, pca


def mds_projection(
    distance_matrix: NDArray[np.float64],
    n_components: int = 2,
    random_state: int = 0,
) -> tuple[NDArray[np.float64], MDS]:
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if d.shape[0] < 2:
        raise ValueError("MDS needs at least two samples")
    if n_components <= 0 or n_components > d.shape[0]:
        raise ValueError("n_components is out of valid range")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix contains non-finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be non-negative")
    if not np.allclose(d, d.T, atol=1e-10):
        raise ValueError("distance_matrix must be symmetric")

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=300,
    )
    coords = mds.fit_transform(d)
    return coords, mds
