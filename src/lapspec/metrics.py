from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

SpectrumMetric = Literal["l2", "wasserstein"]
HistogramMetric = Literal["l1", "js"]


def _validate_feature_matrix(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[0] == 0:
        return np.zeros((0, arr.shape[1]), dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _pairwise_distance_matrix(
    values: NDArray[np.float64],
    dist_fn,
) -> NDArray[np.float64]:
    n = values.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_fn(values[i], values[j]))
            out[i, j] = d
            out[j, i] = d
    return out


def spectrum_distance_matrix(
    spectra: NDArray[np.float64],
    metric: SpectrumMetric = "l2",
) -> NDArray[np.float64]:
    x = _validate_feature_matrix(spectra, name="spectra")

    if metric == "l2":
        return _pairwise_distance_matrix(x, lambda a, b: np.linalg.norm(a - b))
    if metric == "wasserstein":
        return _pairwise_distance_matrix(x, lambda a, b: wasserstein_distance(a, b))
    raise ValueError("metric must be 'l2' or 'wasserstein'")


def _to_probability_mass(hist_row: NDArray[np.float64], eps: float = 1e-15) -> NDArray[np.float64]:
    p = np.clip(hist_row, 0.0, None)
    total = float(np.sum(p))
    if total <= eps:
        return np.full_like(p, 1.0 / p.shape[0], dtype=np.float64)
    return p / total


def _js_divergence(p: NDArray[np.float64], q: NDArray[np.float64], eps: float = 1e-15) -> float:
    m = 0.5 * (p + q)

    def _kl(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (_kl(p, m) + _kl(q, m))


def histogram_distance_matrix(
    histograms: NDArray[np.float64],
    metric: HistogramMetric = "l1",
    normalize: bool = True,
) -> NDArray[np.float64]:
    x = _validate_feature_matrix(histograms, name="histograms")
    if x.shape[1] == 0 and x.shape[0] > 0:
        raise ValueError("histograms must have at least one bin")

    if normalize:
        h = np.vstack([_to_probability_mass(row) for row in x]) if x.size else x
    else:
        h = x

    if metric == "l1":
        return _pairwise_distance_matrix(h, lambda a, b: np.sum(np.abs(a - b)))
    if metric == "js":
        return _pairwise_distance_matrix(h, _js_divergence)
    raise ValueError("metric must be 'l1' or 'js'")

