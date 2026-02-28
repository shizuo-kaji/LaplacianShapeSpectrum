from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .types import WeightedGraph

BoundaryCondition = Literal["dirichlet", "neumann"]


@dataclass(frozen=True, slots=True)
class LaplacianResult:
    matrix: sp.csr_matrix
    active_nodes: NDArray[np.int64]
    boundary_condition: BoundaryCondition


def build_adjacency(graph: WeightedGraph) -> sp.csr_matrix:
    graph.validate()
    n = graph.num_nodes
    if graph.edges.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)

    src = graph.edges[:, 0]
    dst = graph.edges[:, 1]
    data = graph.weights
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    val = np.concatenate([data, data])
    adjacency = sp.coo_matrix((val, (row, col)), shape=(n, n), dtype=np.float64).tocsr()
    adjacency.sum_duplicates()
    adjacency.setdiag(0.0)
    return adjacency


def build_laplacian(
    graph: WeightedGraph, boundary: BoundaryCondition = "neumann"
) -> LaplacianResult:
    adjacency = build_adjacency(graph)
    laplacian = sp.csgraph.laplacian(adjacency, normed=True)
    laplacian = laplacian.tocsr().astype(np.float64)

    if boundary == "neumann":
        active = np.arange(graph.num_nodes, dtype=np.int64)
        return LaplacianResult(
            matrix=laplacian, active_nodes=active, boundary_condition=boundary
        )

    if boundary != "dirichlet":
        raise ValueError("boundary must be either 'dirichlet' or 'neumann'")

    mask = np.ones(graph.num_nodes, dtype=bool)
    mask[graph.boundary_nodes] = False
    interior = np.flatnonzero(mask).astype(np.int64)
    if interior.size == 0:
        raise ValueError("Dirichlet boundary leaves no interior nodes to solve")

    reduced = laplacian[interior][:, interior].tocsr()
    return LaplacianResult(
        matrix=reduced, active_nodes=interior, boundary_condition=boundary
    )
