from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .laplacian import BoundaryCondition, build_laplacian
from .spectrum import (
    compute_spectrum,
    fixed_length_spectrum,
    spectrum_histogram as _spectrum_histogram,
)
from .types import WeightedGraph

SpectrumNormalization = Literal["none", "max"]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    spectrum_k: int = 32
    hist_bins: int = 32
    hist_range: tuple[float, float] | None = None
    hist_quantile_range: tuple[float, float] | None = None
    density: bool = True
    boundary: BoundaryCondition = "neumann"
    normalize_spectrum: SpectrumNormalization = "none"


def _normalize_spectrum_values(
    eigenvalues: NDArray[np.float64],
    normalize_spectrum: SpectrumNormalization,
) -> NDArray[np.float64]:
    if normalize_spectrum == "none":
        return eigenvalues
    if normalize_spectrum == "max":
        if eigenvalues.size == 0:
            return eigenvalues
        denom = float(np.max(np.abs(eigenvalues)))
        if denom > 0:
            return eigenvalues / denom
        return eigenvalues
    raise ValueError("normalize_spectrum must be 'none' or 'max'")


def _validate_hist_range(
    value_range: tuple[float, float],
    *,
    name: str,
) -> tuple[float, float]:
    lo = float(value_range[0])
    hi = float(value_range[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} must contain finite values")
    if lo >= hi:
        raise ValueError(f"{name} must satisfy min < max")
    return lo, hi


def _resolve_hist_range(
    spectrums: NDArray[np.float64],
    value_range: tuple[float, float] | None,
    quantile_range: tuple[float, float] | None,
) -> tuple[float, float]:
    if value_range is not None and quantile_range is not None:
        raise ValueError("value_range and quantile_range are mutually exclusive")

    if value_range is not None:
        return _validate_hist_range(value_range, name="value_range")

    if quantile_range is None:
        return (0.0, 2.0)

    q_lo = float(quantile_range[0])
    q_hi = float(quantile_range[1])
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError("quantile_range must satisfy 0 <= q_lo < q_hi <= 1")

    finite_values = spectrums[np.isfinite(spectrums)]
    if finite_values.size == 0:
        raise ValueError("cannot infer value_range from empty spectra")

    lo, hi = np.quantile(finite_values, [q_lo, q_hi]).astype(np.float64)
    if lo >= hi:
        center = float(lo)
        eps = max(1e-12, abs(center) * 1e-12)
        lo = center - eps
        hi = center + eps
    lo = max(lo, 0.0)
    return _validate_hist_range((float(lo), float(hi)), name="resolved value_range")


def batch_spectrum(
    graphs: Sequence[WeightedGraph],
    spectrum_k: int = 32,
    boundary: BoundaryCondition = "neumann",
    normalize_spectrum: SpectrumNormalization = "none",
) -> NDArray[np.float64]:
    if spectrum_k <= 0:
        raise ValueError("spectrum_k must be positive")

    spectra: list[NDArray[np.float64]] = []
    for graph in graphs:
        lap_result = build_laplacian(graph, boundary=boundary)
        eigenvalues = compute_spectrum(lap_result.matrix, k=spectrum_k)
        eigenvalues = _normalize_spectrum_values(eigenvalues, normalize_spectrum)
        spectra.append(fixed_length_spectrum(eigenvalues, spectrum_k, fill_value=0.0))

    if not spectra:
        return np.zeros((0, spectrum_k), dtype=np.float64)
    return np.vstack(spectra)


def batch_spectrum_histogram(
    spectrums: NDArray[np.float64],
    bins: int = 32,
    value_range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] | None = None,
    density: bool = True,
) -> NDArray[np.float64]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    spectra = np.asarray(spectrums, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError("spectrums must be a 2D array")
    if not np.all(np.isfinite(spectra)):
        raise ValueError("spectrums contains non-finite values")

    if spectra.shape[0] == 0:
        return np.zeros((0, bins), dtype=np.float64)

    resolved_range = _resolve_hist_range(
        spectra,
        value_range=value_range,
        quantile_range=quantile_range,
    )

    histograms: list[NDArray[np.float64]] = []
    for row in spectra:
        hist, _ = _spectrum_histogram(
            row,
            bins=bins,
            value_range=resolved_range,
            density=density,
        )
        if not np.all(np.isfinite(hist)):
            hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
        histograms.append(hist)
    return np.vstack(histograms)


def spectrum_histogram(
    graph: WeightedGraph,
    config: FeatureConfig,
) -> NDArray[np.float64]:
    spectra = batch_spectrum(
        [graph],
        spectrum_k=config.spectrum_k,
        boundary=config.boundary,
        normalize_spectrum=config.normalize_spectrum,
    )
    histograms = batch_spectrum_histogram(
        spectra,
        bins=config.hist_bins,
        value_range=config.hist_range,
        quantile_range=config.hist_quantile_range,
        density=config.density,
    )
    return np.concatenate([spectra[0], histograms[0]])
