from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable, Sequence

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lapspec.laplacian import build_laplacian
from lapspec.spectrum import compute_spectrum, fixed_length_spectrum, spectrum_histogram
from lapspec.types import WeightedGraph


@dataclass(frozen=True, slots=True)
class BoundaryOutputs:
    features: np.ndarray
    hist_matrix: np.ndarray
    step_distances: np.ndarray
    hist_rows: list[list[object]]
    eig_rows: list[list[object]]
    hist_range: tuple[float, float]


def clean_legacy_npy(out_dir: Path) -> None:
    for legacy_npy in out_dir.glob("*.npy"):
        legacy_npy.unlink()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def to_repo_relative(path: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (ROOT / p).resolve()
    try:
        rel = resolved.relative_to(ROOT)
    except ValueError:
        rel = Path(os.path.relpath(resolved, ROOT))
    return rel.as_posix()


def _sample_indices(size: int, max_items: int) -> np.ndarray:
    if size <= max_items:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, max_items, dtype=np.int64)


def save_overlay_test_images(
    masks: Sequence[np.ndarray],
    graphs: Sequence[WeightedGraph],
    sample_ids: Sequence[int],
    out_dir: Path,
    title_builder: Callable[[int], str],
    max_overlay_edges: int = 5000,
    max_overlay_boundary_nodes: int = 4000,
) -> list[Path]:
    image_dir = out_dir / "test_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for mask, graph, sample_id in zip(masks, graphs, sample_ids):
        image_path = image_dir / f"test_image_{sample_id:03d}.png"
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.imshow(mask.astype(float), cmap="gray", vmin=0.0, vmax=1.0, origin="upper")

        if graph.node_positions is not None:
            pos = graph.node_positions
            if graph.edges.shape[0] > 0:
                edge_idx = _sample_indices(graph.edges.shape[0], max_overlay_edges)
                edges = graph.edges[edge_idx]
                segments = pos[edges][:, :, ::-1]
                lines = LineCollection(
                    segments,
                    colors="deepskyblue",
                    linewidths=0.22,
                    alpha=0.38,
                )
                ax.add_collection(lines)

            if graph.boundary_nodes.shape[0] > 0:
                boundary_idx = _sample_indices(
                    graph.boundary_nodes.shape[0],
                    max_overlay_boundary_nodes,
                )
                nodes = graph.boundary_nodes[boundary_idx]
                bpos = pos[nodes]
                ax.scatter(
                    bpos[:, 1],
                    bpos[:, 0],
                    s=1.2,
                    c="red",
                    alpha=0.88,
                    linewidths=0.0,
                )

        ax.set_title(title_builder(sample_id), fontsize=8)
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        fig.savefig(image_path, dpi=170)
        plt.close(fig)
        image_paths.append(image_path)

    return image_paths


def save_pointcloud_overlay_images(
    pointclouds: Sequence[np.ndarray],
    graphs: Sequence[WeightedGraph],
    sample_ids: Sequence[int],
    out_dir: Path,
    title_builder: Callable[[int], str],
    max_points: int = 1400,
    max_overlay_edges: int = 2600,
    max_overlay_boundary_nodes: int = 600,
    elev: float = 22.0,
    azim: float = 35.0,
) -> list[Path]:
    image_dir = out_dir / "test_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for points, graph, sample_id in zip(pointclouds, graphs, sample_ids):
        image_path = image_dir / f"test_image_{sample_id:03d}.png"
        fig = plt.figure(figsize=(4.0, 4.0))
        ax = fig.add_subplot(111, projection="3d")

        point_idx = _sample_indices(points.shape[0], max_points)
        p = points[point_idx]
        ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=2.0,
            c="lightgray",
            alpha=0.65,
            depthshade=False,
        )

        if graph.node_positions is not None:
            pos = graph.node_positions
            if graph.edges.shape[0] > 0:
                edge_idx = _sample_indices(graph.edges.shape[0], max_overlay_edges)
                edges = graph.edges[edge_idx]
                segments = pos[edges]
                lines = Line3DCollection(
                    segments,
                    colors="deepskyblue",
                    linewidths=0.23,
                    alpha=0.32,
                )
                ax.add_collection3d(lines)
            if graph.boundary_nodes.shape[0] > 0:
                boundary_idx = _sample_indices(
                    graph.boundary_nodes.shape[0],
                    max_overlay_boundary_nodes,
                )
                nodes = graph.boundary_nodes[boundary_idx]
                b = pos[nodes]
                ax.scatter(
                    b[:, 0],
                    b[:, 1],
                    b[:, 2],
                    s=4.0,
                    c="red",
                    alpha=0.88,
                    depthshade=False,
                )

        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        center = 0.5 * (mins + maxs)
        radius = float(np.max(maxs - mins) * 0.58)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title_builder(sample_id), fontsize=8)
        ax.set_axis_off()
        fig.tight_layout(pad=0.1)
        fig.savefig(image_path, dpi=180)
        plt.close(fig)
        image_paths.append(image_path)

    return image_paths


def save_test_images_overview(
    image_paths: Sequence[Path],
    sample_ids: Sequence[int],
    out_path: Path,
    title: str,
    cols: int = 8,
) -> None:
    n = len(image_paths)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes_array = np.atleast_2d(axes)
    for i, ax in enumerate(axes_array.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(plt.imread(image_paths[i]))
        ax.set_title(f"ID {int(sample_ids[i]):03d}", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_weighted_graph_lists(
    graphs: Sequence[WeightedGraph],
    out_dir: Path,
) -> None:
    graph_dir = out_dir / "weighted_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[list[object]] = []
    index_rows: list[list[object]] = []

    for sample_id, graph in enumerate(graphs):
        edges_path = graph_dir / f"graph_{sample_id:03d}_edges.csv"
        boundary_path = graph_dir / f"graph_{sample_id:03d}_boundary_nodes.csv"

        with edges_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["u", "v", "weight"])
            for (u, v), w in zip(graph.edges, graph.weights):
                writer.writerow([int(u), int(v), float(w)])

        with boundary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id"])
            for node_id in graph.boundary_nodes:
                writer.writerow([int(node_id)])

        n = graph.num_nodes
        m = graph.edges.shape[0]
        boundary_count = graph.boundary_nodes.shape[0]
        density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
        min_w = float(np.min(graph.weights)) if m > 0 else 0.0
        mean_w = float(np.mean(graph.weights)) if m > 0 else 0.0
        max_w = float(np.max(graph.weights)) if m > 0 else 0.0
        summary_rows.append(
            [
                sample_id,
                n,
                m,
                boundary_count,
                density,
                min_w,
                mean_w,
                max_w,
            ]
        )
        index_rows.append(
            [sample_id, to_repo_relative(edges_path), to_repo_relative(boundary_path)]
        )

    write_csv(
        out_dir / "weighted_graph_summary.csv",
        [
            "sample_id",
            "num_nodes",
            "num_edges",
            "num_boundary_nodes",
            "edge_density",
            "weight_min",
            "weight_mean",
            "weight_max",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / "weighted_graph_index.csv",
        ["sample_id", "edges_csv", "boundary_nodes_csv"],
        index_rows,
    )


def save_pointcloud_lists(
    pointclouds: Sequence[np.ndarray],
    out_dir: Path,
) -> None:
    point_dir = out_dir / "pointclouds"
    point_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[list[object]] = []
    summary_rows: list[list[object]] = []

    for sample_id, points in enumerate(pointclouds):
        points_path = point_dir / f"pointcloud_{sample_id:03d}.csv"
        rows = [[float(x), float(y), float(z)] for x, y, z in points]
        write_csv(points_path, ["x", "y", "z"], rows)

        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        extent = maxs - mins
        index_rows.append([sample_id, to_repo_relative(points_path), points.shape[0]])
        summary_rows.append(
            [
                sample_id,
                points.shape[0],
                float(np.mean(points[:, 0])),
                float(np.mean(points[:, 1])),
                float(np.mean(points[:, 2])),
                float(extent[0]),
                float(extent[1]),
                float(extent[2]),
            ]
        )

    write_csv(
        out_dir / "pointcloud_index.csv",
        ["sample_id", "points_csv", "num_points"],
        index_rows,
    )
    write_csv(
        out_dir / "pointcloud_summary.csv",
        [
            "sample_id",
            "num_points",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "extent_x",
            "extent_y",
            "extent_z",
        ],
        summary_rows,
    )


def save_matrix_csv(path: Path, matrix: np.ndarray, prefix: str) -> None:
    rows: list[list[object]] = []
    n_cols = matrix.shape[1]
    for i in range(matrix.shape[0]):
        rows.append([i] + [float(v) for v in matrix[i]])
    write_csv(path, ["sample_id"] + [f"{prefix}_{j:03d}" for j in range(n_cols)], rows)


def save_distance_matrix_csv(
    path: Path,
    distance_matrix: np.ndarray,
    sample_ids: Sequence[int] | None = None,
) -> None:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    n = matrix.shape[0]
    if sample_ids is None:
        ids = np.arange(n, dtype=np.int64)
    else:
        ids = np.asarray(sample_ids, dtype=np.int64)
        if ids.shape[0] != n:
            raise ValueError("sample_ids length must match distance_matrix size")

    rows: list[list[object]] = []
    for i in range(n):
        rows.append([int(ids[i])] + [float(v) for v in matrix[i]])
    header = ["sample_id"] + [f"id_{int(sid):03d}" for sid in ids]
    write_csv(path, header, rows)


def save_step_distances_csv(path: Path, distances: np.ndarray) -> None:
    rows = [[i, i + 1, float(d)] for i, d in enumerate(distances)]
    write_csv(path, ["from_sample_id", "to_sample_id", "distance_l2"], rows)


def consecutive_distances(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(features, axis=0), axis=1)


def boundary_outputs(
    graphs: Sequence[WeightedGraph],
    boundary: str,
    spectrum_k: int,
    hist_bins: int,
    hist_range: tuple[float, float] | None = (0.0, 2.0),
    range_quantile: float = 0.995,
    range_padding: float = 1.05,
    range_min_upper: float = 0.05,
    range_max_upper: float = 2.0,
) -> BoundaryOutputs:
    eigs_list: list[np.ndarray] = []
    for graph in graphs:
        lap = build_laplacian(graph, boundary=boundary).matrix
        eigs = compute_spectrum(lap, k=spectrum_k)
        eigs_list.append(eigs)

    if hist_range is None:
        flat_parts = [e for e in eigs_list if e.size > 0]
        if flat_parts:
            flat = np.concatenate(flat_parts)
            finite = flat[np.isfinite(flat)]
            finite = finite[finite >= 0.0]
        else:
            finite = np.empty(0, dtype=np.float64)
        if finite.size == 0:
            selected_hist_range = (0.0, range_max_upper)
        else:
            q = float(np.quantile(finite, range_quantile))
            upper = max(range_min_upper, q * range_padding)
            upper = min(range_max_upper, upper)
            if upper <= 0.0:
                upper = range_min_upper
            selected_hist_range = (0.0, float(upper))
    else:
        selected_hist_range = (float(hist_range[0]), float(hist_range[1]))
        if not np.isfinite(selected_hist_range[0]) or not np.isfinite(selected_hist_range[1]):
            raise ValueError("hist_range values must be finite")
        if selected_hist_range[0] >= selected_hist_range[1]:
            raise ValueError("hist_range must satisfy min < max")

    hist_rows: list[list[object]] = []
    eig_rows: list[list[object]] = []
    hist_matrix_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []

    for sample_id, eigs in enumerate(eigs_list):
        eigs_fixed = fixed_length_spectrum(eigs, length=spectrum_k, fill_value=0.0)
        hist, edges = spectrum_histogram(
            eigs,
            bins=hist_bins,
            value_range=selected_hist_range,
            density=True,
        )
        feature_rows.append(np.concatenate([eigs_fixed, hist]))
        hist_matrix_rows.append(hist)

        for j, eig in enumerate(eigs):
            eig_rows.append([sample_id, j, float(eig)])
        for b, value in enumerate(hist):
            hist_rows.append([sample_id, b, float(edges[b]), float(edges[b + 1]), float(value)])

    features = np.vstack(feature_rows) if feature_rows else np.zeros((0, spectrum_k + hist_bins))
    hist_matrix = np.vstack(hist_matrix_rows) if hist_matrix_rows else np.zeros((0, hist_bins))
    step_distances = consecutive_distances(features)
    return BoundaryOutputs(
        features=features,
        hist_matrix=hist_matrix,
        step_distances=step_distances,
        hist_rows=hist_rows,
        eig_rows=eig_rows,
        hist_range=selected_hist_range,
    )


def plot_pca_with_ids(
    dir_coords: np.ndarray,
    neu_coords: np.ndarray,
    sample_ids: Sequence[int],
    color_values: np.ndarray,
    color_label: str,
    out_path: Path,
    cmap: str = "viridis",
    sizes: np.ndarray | None = None,
    annotate_fontsize: int = 5,
    case_ids: np.ndarray | None = None,
) -> None:
    plot_embedding_with_ids(
        dir_coords=dir_coords,
        neu_coords=neu_coords,
        sample_ids=sample_ids,
        color_values=color_values,
        color_label=color_label,
        out_path=out_path,
        method_name="PCA",
        cmap=cmap,
        sizes=sizes,
        annotate_fontsize=annotate_fontsize,
        case_ids=case_ids,
    )


def _case_marker_map(case_ids: np.ndarray) -> dict[int, str]:
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p"]
    unique_case_ids = sorted(int(v) for v in np.unique(case_ids).tolist())
    return {cid: markers[i % len(markers)] for i, cid in enumerate(unique_case_ids)}


def _scatter_embedding_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    sample_ids: Sequence[int],
    color_values: np.ndarray,
    cmap: str,
    sizes: np.ndarray,
    annotate_fontsize: int,
    case_ids: np.ndarray | None,
) -> plt.cm.ScalarMappable:
    vmin = float(np.min(color_values))
    vmax = float(np.max(color_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    marker_map = _case_marker_map(case_ids) if case_ids is not None else None

    if case_ids is None:
        colors = cmap_obj(norm(color_values))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, alpha=0.9, marker="o")
    else:
        unique_case_ids = sorted(int(v) for v in np.unique(case_ids).tolist())
        for cid in unique_case_ids:
            mask = case_ids == cid
            colors = cmap_obj(norm(color_values[mask]))
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors,
                s=sizes[mask],
                alpha=0.9,
                marker=marker_map[cid],
                label=f"case {cid}",
            )
        ax.legend(title="case", fontsize=7, title_fontsize=8, loc="best", framealpha=0.85)

    for sid, (x, y) in zip(sample_ids, coords):
        ax.text(x, y, f"{int(sid)}", fontsize=annotate_fontsize, alpha=0.8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    return sm


def plot_embedding_with_ids(
    dir_coords: np.ndarray,
    neu_coords: np.ndarray,
    sample_ids: Sequence[int],
    color_values: np.ndarray,
    color_label: str,
    out_path: Path,
    method_name: str,
    cmap: str = "viridis",
    sizes: np.ndarray | None = None,
    annotate_fontsize: int = 5,
    case_ids: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_samples = dir_coords.shape[0]
    if sizes is None:
        sizes_arr = np.full(n_samples, 20.0, dtype=np.float64)
    else:
        sizes_arr = np.asarray(sizes, dtype=np.float64)
        if sizes_arr.ndim == 0:
            sizes_arr = np.full(n_samples, float(sizes_arr), dtype=np.float64)
        elif sizes_arr.shape[0] != n_samples:
            raise ValueError("sizes must be scalar or have length equal to number of samples")
    case_arr = np.asarray(case_ids, dtype=np.int64) if case_ids is not None else None
    if case_arr is not None and case_arr.shape[0] != n_samples:
        raise ValueError("case_ids length must match number of samples")

    color_arr = np.asarray(color_values, dtype=np.float64)
    if color_arr.shape[0] != n_samples:
        raise ValueError("color_values length must match number of samples")

    for ax, coords, title in [(axes[0], dir_coords, "Dirichlet"), (axes[1], neu_coords, "Neumann")]:
        sm = _scatter_embedding_panel(
            ax=ax,
            coords=coords,
            sample_ids=sample_ids,
            color_values=color_arr,
            cmap=cmap,
            sizes=sizes_arr,
            annotate_fontsize=annotate_fontsize,
            case_ids=case_arr,
        )
        ax.set_title(f"{method_name} Trajectory ({title})")
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")
        ax.grid(alpha=0.25)
        fig.colorbar(sm, ax=ax, label=color_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_mds_with_ids(
    dir_coords: np.ndarray,
    neu_coords: np.ndarray,
    sample_ids: Sequence[int],
    color_values: np.ndarray,
    color_label: str,
    out_path: Path,
    cmap: str = "viridis",
    sizes: np.ndarray | None = None,
    annotate_fontsize: int = 5,
    case_ids: np.ndarray | None = None,
) -> None:
    plot_embedding_with_ids(
        dir_coords=dir_coords,
        neu_coords=neu_coords,
        sample_ids=sample_ids,
        color_values=color_values,
        color_label=color_label,
        out_path=out_path,
        method_name="MDS",
        cmap=cmap,
        sizes=sizes,
        annotate_fontsize=annotate_fontsize,
        case_ids=case_ids,
    )


def _draw_case_brackets(ax: plt.Axes, case_ids: np.ndarray) -> None:
    if case_ids.size == 0:
        return
    x0 = -2.4
    x1 = -1.9
    start = 0
    n = case_ids.shape[0]
    while start < n:
        cid = int(case_ids[start])
        end = start
        while end + 1 < n and int(case_ids[end + 1]) == cid:
            end += 1
        y0 = start - 0.5
        y1 = end + 0.5
        ax.plot([x0, x0], [y0, y1], color="white", linewidth=1.4, clip_on=False)
        ax.plot([x0, x1], [y0, y0], color="white", linewidth=1.4, clip_on=False)
        ax.plot([x0, x1], [y1, y1], color="white", linewidth=1.4, clip_on=False)
        ax.text(
            x0 - 0.2,
            0.5 * (y0 + y1),
            f"C{cid}",
            color="white",
            fontsize=7,
            ha="right",
            va="center",
            clip_on=False,
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 0.4},
        )
        start = end + 1


def plot_histogram_matrices(
    dir_hist: np.ndarray,
    neu_hist: np.ndarray,
    out_path: Path,
    case_ids: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(dir_hist, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Dirichlet Histogram List")
    axes[0].set_xlabel("bin index")
    axes[0].set_ylabel("sample ID")
    fig.colorbar(im0, ax=axes[0], label="density")

    im1 = axes[1].imshow(neu_hist, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Neumann Histogram List")
    axes[1].set_xlabel("bin index")
    axes[1].set_ylabel("sample ID")
    fig.colorbar(im1, ax=axes[1], label="density")

    if case_ids is not None:
        case_arr = np.asarray(case_ids, dtype=np.int64)
        axes[0].set_xlim(-3.2, dir_hist.shape[1] - 0.5)
        axes[1].set_xlim(-3.2, neu_hist.shape[1] - 0.5)
        _draw_case_brackets(axes[0], case_arr)
        _draw_case_brackets(axes[1], case_arr)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def compute_h0_h1(
    mask: np.ndarray,
    min_component_pixels: int = 1,
    min_hole_pixels: int = 1,
) -> tuple[int, int]:
    fg = np.asarray(mask, dtype=bool)
    fg_labels, fg_count = ndimage.label(fg)
    if fg_count == 0:
        return 0, 0
    fg_sizes = np.bincount(fg_labels.ravel())
    h0 = int(np.sum(fg_sizes[1:] >= min_component_pixels))

    filled = ndimage.binary_fill_holes(fg)
    holes = np.logical_and(filled, np.logical_not(fg))
    hole_labels, hole_count = ndimage.label(holes)
    if hole_count == 0:
        return h0, 0
    hole_sizes = np.bincount(hole_labels.ravel())
    h1 = int(np.sum(hole_sizes[1:] >= min_hole_pixels))
    return h0, h1


def homology_arrays(
    masks: Sequence[np.ndarray],
    min_component_pixels: int = 1,
    min_hole_pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    h0_vals: list[int] = []
    h1_vals: list[int] = []
    for mask in masks:
        h0, h1 = compute_h0_h1(
            mask,
            min_component_pixels=min_component_pixels,
            min_hole_pixels=min_hole_pixels,
        )
        h0_vals.append(h0)
        h1_vals.append(h1)
    return np.asarray(h0_vals, dtype=np.int64), np.asarray(h1_vals, dtype=np.int64)


def save_homology_csv(
    path: Path,
    sample_ids: Sequence[int],
    h0_values: np.ndarray,
    h1_values: np.ndarray,
) -> None:
    rows: list[list[object]] = []
    for sid, h0, h1 in zip(sample_ids, h0_values, h1_values):
        rows.append([int(sid), int(h0), int(h1)])
    write_csv(path, ["sample_id", "H0", "H1"], rows)


def plot_pca_homology_colors(
    dir_coords: np.ndarray,
    neu_coords: np.ndarray,
    h0_values: np.ndarray,
    h1_values: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        (axes[0, 0], dir_coords, h0_values, "Dirichlet colored by H0"),
        (axes[0, 1], dir_coords, h1_values, "Dirichlet colored by H1"),
        (axes[1, 0], neu_coords, h0_values, "Neumann colored by H0"),
        (axes[1, 1], neu_coords, h1_values, "Neumann colored by H1"),
    ]
    for ax, coords, color_values, title in panels:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_values,
            cmap="plasma",
            s=22,
            alpha=0.92,
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="homology value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
