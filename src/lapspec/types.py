from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WeightedGraph:
    """Undirected weighted graph with a boundary node subset."""

    num_nodes: int
    edges: IntArray
    weights: FloatArray
    boundary_nodes: IntArray
    node_positions: FloatArray | None = None

    def validate(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape (m, 2)")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.edges.shape[0]:
            raise ValueError("weights must have shape (m,) and match edges")
        if self.boundary_nodes.ndim != 1:
            raise ValueError("boundary_nodes must be a 1D array")
        if self.edges.size > 0:
            if self.edges.min() < 0 or self.edges.max() >= self.num_nodes:
                raise ValueError("edge indices are out of range")
            if np.any(self.edges[:, 0] == self.edges[:, 1]):
                raise ValueError("self-loop edges are not supported")
        if self.boundary_nodes.size > 0:
            if self.boundary_nodes.min() < 0 or self.boundary_nodes.max() >= self.num_nodes:
                raise ValueError("boundary node index out of range")
        if np.any(self.weights <= 0):
            raise ValueError("edge weights must be positive")
        if self.node_positions is not None and self.node_positions.shape[0] != self.num_nodes:
            raise ValueError("node_positions must have shape (num_nodes, dim)")

