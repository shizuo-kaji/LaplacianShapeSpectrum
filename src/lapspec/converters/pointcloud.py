from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from ..graph import from_edge_list
from ..types import WeightedGraph

GraphMethod = Literal["knn", "radius"]
WeightMode = Literal["unit", "inverse_distance", "gaussian"]
BoundaryMode = Literal["outer_percentile", "none"]


def _pair_weight(
    distance: float,
    weight_mode: WeightMode,
    sigma: float,
    eps: float,
) -> float:
    if weight_mode == "unit":
        return 1.0
    if weight_mode == "inverse_distance":
        return 1.0 / max(distance, eps)
    if weight_mode == "gaussian":
        return float(np.exp(-((distance**2) / (2.0 * sigma**2))))
    raise ValueError("unknown weight_mode")


def _outer_boundary_indices(
    points: NDArray[np.float64], boundary_percentile: float
) -> NDArray[np.int64]:
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    threshold = np.percentile(distances, boundary_percentile)
    return np.flatnonzero(distances >= threshold).astype(np.int64)


def pointcloud_to_graph(
    points: ArrayLike,
    method: GraphMethod = "knn",
    k: int = 10,
    radius: float | None = None,
    weight_mode: WeightMode = "inverse_distance",
    sigma: float | None = None,
    boundary_mode: BoundaryMode = "outer_percentile",
    boundary_percentile: float = 90.0,
    boundary_indices: ArrayLike | None = None,
    eps: float = 1e-9,
) -> WeightedGraph:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must have shape (n, dim)")
    n = pts.shape[0]
    if n == 0:
        raise ValueError("point cloud is empty")

    tree = cKDTree(pts)
    pair_to_dist: dict[tuple[int, int], float] = {}
    sampled_distances: list[float] = []

    if method == "knn":
        if k <= 0:
            raise ValueError("k must be positive")
        k_query = min(k + 1, n)
        dists, inds = tree.query(pts, k=k_query)
        if k_query == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        for i in range(n):
            for d, j in zip(dists[i], inds[i]):
                j = int(j)
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                distance = float(d)
                sampled_distances.append(distance)
                prev = pair_to_dist.get((a, b))
                if prev is None or distance < prev:
                    pair_to_dist[(a, b)] = distance
    elif method == "radius":
        if radius is None or radius <= 0:
            raise ValueError("radius must be provided and positive for radius graph")
        for a, b in tree.query_pairs(radius):
            distance = float(np.linalg.norm(pts[a] - pts[b]))
            pair_to_dist[(a, b)] = distance
            sampled_distances.append(distance)
    else:
        raise ValueError("method must be 'knn' or 'radius'")

    nonzero_distances = np.asarray([d for d in sampled_distances if d > eps])
    if sigma is None:
        sigma_value = float(np.median(nonzero_distances)) if nonzero_distances.size else 1.0
    else:
        sigma_value = float(sigma)
        if sigma_value <= 0:
            raise ValueError("sigma must be positive")

    if pair_to_dist:
        edges = np.asarray(list(pair_to_dist.keys()), dtype=np.int64)
        weights = np.asarray(
            [
                _pair_weight(d, weight_mode=weight_mode, sigma=sigma_value, eps=eps)
                for d in pair_to_dist.values()
            ],
            dtype=np.float64,
        )
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        weights = np.zeros((0,), dtype=np.float64)

    if boundary_indices is not None:
        boundary = np.unique(np.asarray(boundary_indices, dtype=np.int64))
    elif boundary_mode == "outer_percentile":
        boundary = _outer_boundary_indices(pts, boundary_percentile)
    elif boundary_mode == "none":
        boundary = np.empty(0, dtype=np.int64)
    else:
        raise ValueError("boundary_mode must be 'outer_percentile' or 'none'")

    return from_edge_list(
        num_nodes=n,
        edges=edges,
        weights=weights,
        boundary_nodes=boundary,
        node_positions=pts,
    )

