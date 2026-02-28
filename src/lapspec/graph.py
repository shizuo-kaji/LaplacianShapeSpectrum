from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray, IntArray, WeightedGraph


def _as_int_array(values: ArrayLike, shape: tuple[int, ...] | None = None) -> IntArray:
    arr = np.asarray(values, dtype=np.int64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


def _as_float_array(values: ArrayLike, shape: tuple[int, ...] | None = None) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


def from_edge_list(
    num_nodes: int,
    edges: ArrayLike,
    weights: ArrayLike | None = None,
    boundary_nodes: ArrayLike | None = None,
    node_positions: ArrayLike | None = None,
) -> WeightedGraph:
    edge_arr = _as_int_array(edges)
    if edge_arr.ndim != 2 or edge_arr.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2)")

    if weights is None:
        weight_arr = np.ones(edge_arr.shape[0], dtype=np.float64)
    else:
        weight_arr = _as_float_array(weights, shape=(edge_arr.shape[0],))

    if edge_arr.size > 0:
        normalized = np.sort(edge_arr, axis=1)
        unique_edges, inverse = np.unique(normalized, axis=0, return_inverse=True)
        merged_weights = np.zeros(unique_edges.shape[0], dtype=np.float64)
        np.add.at(merged_weights, inverse, weight_arr)
        edge_arr = unique_edges
        weight_arr = merged_weights

    if boundary_nodes is None:
        boundary_arr = np.empty(0, dtype=np.int64)
    else:
        boundary_arr = np.unique(_as_int_array(boundary_nodes).reshape(-1))

    if node_positions is None:
        position_arr = None
    else:
        position_arr = _as_float_array(node_positions)

    graph = WeightedGraph(
        num_nodes=int(num_nodes),
        edges=edge_arr,
        weights=weight_arr,
        boundary_nodes=boundary_arr,
        node_positions=position_arr,
    )
    graph.validate()
    return graph
