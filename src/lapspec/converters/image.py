from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..graph import from_edge_list
from ..types import WeightedGraph

WeightMode = Literal["unit", "distance"]


def _neighbor_offsets(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        return [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    raise ValueError("connectivity must be 4 or 8")


def image_to_graph(
    mask: ArrayLike,
    connectivity: int = 4,
    weight_mode: WeightMode = "unit",
) -> WeightedGraph:
    grid = np.asarray(mask, dtype=bool)
    if grid.ndim != 2:
        raise ValueError("mask must be a 2D array")

    coords = np.argwhere(grid)
    n = coords.shape[0]
    if n == 0:
        raise ValueError("mask has no foreground pixels")

    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords)}
    offsets = _neighbor_offsets(connectivity)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    boundary: set[int] = set()
    h, w = grid.shape

    # Boundary detection uses 4-neighborhood for stability.
    boundary_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for idx, (r, c) in enumerate(coords):
        for dr, dc in boundary_offsets:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or not grid[rr, cc]:
                boundary.add(idx)
                break

        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or not grid[rr, cc]:
                continue
            j = coord_to_index[(rr, cc)]
            if idx < j:
                if weight_mode == "unit":
                    weight = 1.0
                elif weight_mode == "distance":
                    distance = float(np.hypot(dr, dc))
                    weight = 1.0 / distance
                else:
                    raise ValueError("weight_mode must be 'unit' or 'distance'")
                edges.append((idx, j))
                weights.append(weight)

    edge_arr: NDArray[np.int64]
    if edges:
        edge_arr = np.asarray(edges, dtype=np.int64)
        weight_arr = np.asarray(weights, dtype=np.float64)
    else:
        edge_arr = np.zeros((0, 2), dtype=np.int64)
        weight_arr = np.zeros((0,), dtype=np.float64)

    position_arr = coords.astype(np.float64)
    return from_edge_list(
        num_nodes=n,
        edges=edge_arr,
        weights=weight_arr,
        boundary_nodes=np.fromiter(sorted(boundary), dtype=np.int64),
        node_positions=position_arr,
    )

