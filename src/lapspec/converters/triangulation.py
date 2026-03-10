from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay, QhullError

from ..graph import from_edge_list
from ..types import WeightedGraph

BoundaryMode2D = Literal["convex_hull", "none"]


def _cotangent(u: NDArray[np.float64], v: NDArray[np.float64], eps: float) -> float:
    cross = float(u[0] * v[1] - u[1] * v[0])
    denom = max(abs(cross), eps)
    dot = float(np.dot(u, v))
    return dot / denom


def pointcloud2d_to_cotan_graph(
    points: ArrayLike,
    boundary_mode: BoundaryMode2D = "convex_hull",
    boundary_indices: ArrayLike | None = None,
    min_weight: float = 1e-12,
    clip_nonpositive: bool = True,
    qhull_options: str | None = None,
) -> WeightedGraph:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    n = pts.shape[0]
    if n < 3:
        raise ValueError("at least 3 points are required for triangulation")
    if min_weight <= 0:
        raise ValueError("min_weight must be positive")

    try:
        tri = Delaunay(pts, qhull_options=qhull_options)
    except QhullError as exc:
        raise ValueError("failed to triangulate points") from exc

    simplices = np.asarray(tri.simplices, dtype=np.int64)
    if simplices.size == 0:
        raise ValueError("triangulation returned no simplices")

    edge_to_weight: dict[tuple[int, int], float] = {}

    for i, j, k in simplices:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]

        cot_i = _cotangent(pj - pi, pk - pi, eps=min_weight)
        cot_j = _cotangent(pi - pj, pk - pj, eps=min_weight)
        cot_k = _cotangent(pi - pk, pj - pk, eps=min_weight)

        e_jk = (int(min(j, k)), int(max(j, k)))
        e_ik = (int(min(i, k)), int(max(i, k)))
        e_ij = (int(min(i, j)), int(max(i, j)))

        edge_to_weight[e_jk] = edge_to_weight.get(e_jk, 0.0) + 0.5 * cot_i
        edge_to_weight[e_ik] = edge_to_weight.get(e_ik, 0.0) + 0.5 * cot_j
        edge_to_weight[e_ij] = edge_to_weight.get(e_ij, 0.0) + 0.5 * cot_k

    edges = np.asarray(list(edge_to_weight.keys()), dtype=np.int64)
    weights = np.asarray(list(edge_to_weight.values()), dtype=np.float64)

    if clip_nonpositive:
        weights = np.maximum(weights, min_weight)
    if np.any(~np.isfinite(weights)):
        raise ValueError("computed cotan weights are non-finite")
    if np.any(weights <= 0):
        raise ValueError("computed cotan weights must be positive")

    if boundary_indices is not None:
        boundary = np.unique(np.asarray(boundary_indices, dtype=np.int64))
    elif boundary_mode == "convex_hull":
        boundary = np.unique(np.asarray(tri.convex_hull, dtype=np.int64).ravel())
    elif boundary_mode == "none":
        boundary = np.empty(0, dtype=np.int64)
    else:
        raise ValueError("boundary_mode must be 'convex_hull' or 'none'")

    return from_edge_list(
        num_nodes=n,
        edges=edges,
        weights=weights,
        boundary_nodes=boundary,
        node_positions=pts,
    )
