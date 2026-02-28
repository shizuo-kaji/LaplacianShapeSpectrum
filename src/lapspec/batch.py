from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .laplacian import BoundaryCondition, build_laplacian
from .spectrum import compute_spectrum, fixed_length_spectrum, spectrum_histogram
from .types import WeightedGraph

SpectrumNormalization = Literal["none", "max"]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    spectrum_k: int = 32
    hist_bins: int = 32
    hist_range: tuple[float, float] | None = None
    density: bool = True
    boundary: BoundaryCondition = "neumann"
    normalize_spectrum: SpectrumNormalization = "none"


def graph_feature_vector(
    graph: WeightedGraph,
    config: FeatureConfig,
) -> NDArray[np.float64]:
    lap_result = build_laplacian(graph, boundary=config.boundary)
    eigenvalues = compute_spectrum(lap_result.matrix, k=config.spectrum_k)

    if config.normalize_spectrum == "max" and eigenvalues.size > 0:
        denom = float(np.max(np.abs(eigenvalues)))
        if denom > 0:
            eigenvalues = eigenvalues / denom
    elif config.normalize_spectrum != "none":
        raise ValueError("normalize_spectrum must be 'none' or 'max'")
    selected_hist_range = config.hist_range if config.hist_range is not None else (0.0, 2.0)
    if selected_hist_range[0] >= selected_hist_range[1]:
        raise ValueError("hist_range must satisfy min < max")

    spectrum_part = fixed_length_spectrum(eigenvalues, config.spectrum_k, fill_value=0.0)
    hist_part, _ = spectrum_histogram(
        eigenvalues,
        bins=config.hist_bins,
        value_range=selected_hist_range,
        density=config.density,
    )
    return np.concatenate([spectrum_part, hist_part])


def batch_features(
    graphs: Sequence[WeightedGraph],
    spectrum_k: int = 32,
    hist_bins: int = 32,
    boundary: BoundaryCondition = "neumann",
    hist_range: tuple[float, float] | None = None,
    density: bool = True,
    normalize_spectrum: SpectrumNormalization = "none",
) -> NDArray[np.float64]:
    config = FeatureConfig(
        spectrum_k=spectrum_k,
        hist_bins=hist_bins,
        hist_range=hist_range,
        density=density,
        boundary=boundary,
        normalize_spectrum=normalize_spectrum,
    )
    features = [graph_feature_vector(graph, config=config) for graph in graphs]
    if not features:
        return np.zeros((0, spectrum_k + hist_bins), dtype=np.float64)
    return np.vstack(features)
