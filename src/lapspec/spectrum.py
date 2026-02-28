from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray


def _dense_eigvals(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    eigvals = np.linalg.eigvalsh(matrix)
    return np.sort(np.real(eigvals))


def compute_spectrum(
    matrix: sp.spmatrix | NDArray[np.float64],
    k: int | None = None,
    tol: float = 1e-10,
    eigsh_kwargs: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    if eigsh_kwargs is None:
        eigsh_kwargs = {}

    if sp.issparse(matrix):
        lap = matrix.tocsr()
        n = lap.shape[0]
    else:
        lap = np.asarray(matrix, dtype=np.float64)
        n = lap.shape[0]

    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([float(lap[0, 0])], dtype=np.float64)

    if k is None or k >= n:
        dense = lap.toarray() if sp.issparse(lap) else lap
        eigvals = _dense_eigvals(dense)
        eigvals[np.abs(eigvals) < tol] = 0.0
        return eigvals

    if k <= 0:
        raise ValueError("k must be positive")

    if not sp.issparse(lap):
        eigvals = _dense_eigvals(lap)[:k]
        eigvals[np.abs(eigvals) < tol] = 0.0
        return eigvals

    eigvals = None
    tries = [dict(which="SM"), dict(which="SA")]
    for attempt in tries:
        try:
            eigvals = spla.eigsh(
                lap,
                k=k,
                return_eigenvectors=False,
                **attempt,
                **eigsh_kwargs,
            )
            break
        except Exception:
            continue

    if eigvals is None:
        dense = lap.toarray()
        eigvals = _dense_eigvals(dense)[:k]
    else:
        eigvals = np.sort(np.real(eigvals))

    eigvals[np.abs(eigvals) < tol] = 0.0
    return eigvals


def spectrum_histogram(
    eigenvalues: NDArray[np.float64],
    bins: int | NDArray[np.float64] = 32,
    value_range: tuple[float, float] | None = None,
    density: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hist, edges = np.histogram(
        eigenvalues, bins=bins, range=value_range, density=density
    )
    return hist.astype(np.float64), edges.astype(np.float64)


def fixed_length_spectrum(
    eigenvalues: NDArray[np.float64],
    length: int,
    fill_value: float = 0.0,
) -> NDArray[np.float64]:
    if length <= 0:
        raise ValueError("length must be positive")
    out = np.full(length, fill_value, dtype=np.float64)
    end = min(length, eigenvalues.shape[0])
    out[:end] = eigenvalues[:end]
    return out

