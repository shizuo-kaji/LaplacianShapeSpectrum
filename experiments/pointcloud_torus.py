from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from shared import (
    boundary_outputs,
    clean_legacy_npy,
    plot_histogram_matrices,
    plot_mds_with_ids,
    plot_pca_homology_colors,
    plot_pca_with_ids,
    save_distance_matrix_csv,
    save_homology_csv,
    save_matrix_csv,
    save_pointcloud_lists,
    save_pointcloud_overlay_images,
    save_step_distances_csv,
    save_test_images_overview,
    save_weighted_graph_lists,
    to_repo_relative,
    write_csv,
)
from lapspec.converters import pointcloud_to_graph
from lapspec.metrics import histogram_distance_matrix, spectrum_distance_matrix
from lapspec.types import WeightedGraph
from lapspec.visualization import mds_projection, pca_projection


@dataclass(frozen=True, slots=True)
class TorusCase:
    case_id: int
    case_name: str
    axis_a: float
    axis_b: float
    axis_c: float
    dent_rx: float
    dent_ry: float
    dent_rz: float
    dent_offset: float
    dent_travel: float
    depth_gamma: float


@dataclass(frozen=True, slots=True)
class SampleMeta:
    sample_id: int
    case_id: int
    case_name: str
    local_step: int
    t: float
    dent_center_z: float
    axis_a: float
    axis_b: float
    axis_c: float
    dent_rx: float
    dent_ry: float
    dent_rz: float
    dent_offset: float
    dent_travel: float
    depth_gamma: float
    homology_h0: int
    homology_h1: int


def default_cases() -> list[TorusCase]:
    return [
        TorusCase(
            case_id=0,
            case_name="balanced",
            axis_a=1.00,
            axis_b=0.82,
            axis_c=0.72,
            dent_rx=0.35,
            dent_ry=0.32,
            dent_rz=0.40,
            dent_offset=0.24,
            dent_travel=0.62,
            depth_gamma=1.20,
        ),
        TorusCase(
            case_id=1,
            case_name="x_elongated",
            axis_a=1.15,
            axis_b=0.72,
            axis_c=0.68,
            dent_rx=0.42,
            dent_ry=0.30,
            dent_rz=0.48,
            dent_offset=0.28,
            dent_travel=0.68,
            depth_gamma=1.00,
        ),
        TorusCase(
            case_id=2,
            case_name="near_sphere",
            axis_a=0.92,
            axis_b=0.92,
            axis_c=0.78,
            dent_rx=0.33,
            dent_ry=0.33,
            dent_rz=0.46,
            dent_offset=0.20,
            dent_travel=0.58,
            depth_gamma=1.50,
        ),
        TorusCase(
            case_id=3,
            case_name="flat_z",
            axis_a=1.05,
            axis_b=0.88,
            axis_c=0.62,
            dent_rx=0.40,
            dent_ry=0.34,
            dent_rz=0.36,
            dent_offset=0.20,
            dent_travel=0.54,
            depth_gamma=1.20,
        ),
        TorusCase(
            case_id=4,
            case_name="large_ring",
            axis_a=1.22,
            axis_b=0.78,
            axis_c=0.74,
            dent_rx=0.48,
            dent_ry=0.33,
            dent_rz=0.54,
            dent_offset=0.32,
            dent_travel=0.76,
            depth_gamma=0.90,
        ),
    ]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = int(np.argmax(sizes[1:]) + 1)
    return labels == keep


def torus_transition_volume(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    case: TorusCase,
    t: float,
) -> tuple[np.ndarray, float]:
    t_clamped = min(max(float(t), 0.0), 1.0)
    depth_scale = t_clamped**case.depth_gamma
    dent_center_z = case.axis_c + case.dent_offset - case.dent_travel * depth_scale

    outer = (
        (x / case.axis_a) ** 2
        + (y / case.axis_b) ** 2
        + (z / case.axis_c) ** 2
        <= 1.0
    )
    dent_top = (
        (x / case.dent_rx) ** 2
        + (y / case.dent_ry) ** 2
        + ((z - dent_center_z) / case.dent_rz) ** 2
        <= 1.0
    )
    dent_bottom = (
        (x / case.dent_rx) ** 2
        + (y / case.dent_ry) ** 2
        + ((z + dent_center_z) / case.dent_rz) ** 2
        <= 1.0
    )

    volume = np.logical_and(outer, np.logical_not(np.logical_or(dent_top, dent_bottom)))
    volume = _largest_component(volume)
    return volume.astype(bool), float(dent_center_z)


def volume_homology_proxy(volume: np.ndarray) -> tuple[int, int]:
    _, h0 = ndimage.label(volume)
    z_center = volume.shape[0] // 2
    y_center = volume.shape[1] // 2
    x_center = volume.shape[2] // 2
    center_line = volume[:, y_center, x_center]
    h1 = int(h0 == 1 and not bool(np.any(center_line)))
    return int(h0), int(h1)


def volume_to_surface_points(
    volume: np.ndarray,
    axis_values: np.ndarray,
    rng: np.random.Generator,
    points_per_sample: int,
    jitter_std: float = 0.0025,
) -> np.ndarray:
    eroded = ndimage.binary_erosion(
        volume,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    boundary = np.logical_and(volume, np.logical_not(eroded))
    idx = np.argwhere(boundary)
    if idx.shape[0] == 0:
        idx = np.argwhere(volume)
    if idx.shape[0] == 0:
        raise ValueError("volume has no active voxels")

    if idx.shape[0] > points_per_sample:
        choice = rng.choice(idx.shape[0], size=points_per_sample, replace=False)
        idx = idx[choice]

    points = np.column_stack(
        [
            axis_values[idx[:, 2]],
            axis_values[idx[:, 1]],
            axis_values[idx[:, 0]],
        ]
    ).astype(np.float64)
    points += rng.normal(loc=0.0, scale=jitter_std, size=points.shape)
    points -= np.mean(points, axis=0, keepdims=True)
    return points


def generate_dataset(
    num_steps_per_case: int = 12,
    grid_size: int = 72,
    extent: float = 1.6,
    points_per_sample: int = 900,
    random_seed: int = 0,
) -> tuple[list[np.ndarray], list[WeightedGraph], list[SampleMeta]]:
    if num_steps_per_case < 2:
        raise ValueError("num_steps_per_case must be at least 2")

    axis_values = np.linspace(-extent, extent, grid_size, dtype=np.float64)
    z, y, x = np.meshgrid(axis_values, axis_values, axis_values, indexing="ij")
    rng = np.random.default_rng(random_seed)

    pointclouds: list[np.ndarray] = []
    graphs: list[WeightedGraph] = []
    metas: list[SampleMeta] = []
    sample_id = 0
    for case in default_cases():
        for local_step in range(num_steps_per_case):
            t = local_step / (num_steps_per_case - 1)
            volume, dent_center_z = torus_transition_volume(x=x, y=y, z=z, case=case, t=t)
            h0, h1 = volume_homology_proxy(volume)
            points = volume_to_surface_points(
                volume=volume,
                axis_values=axis_values,
                rng=rng,
                points_per_sample=points_per_sample,
            )
            graph = pointcloud_to_graph(
                points,
                method="knn",
                k=14,
                weight_mode="inverse_distance",
                boundary_mode="none",
            )

            pointclouds.append(points)
            graphs.append(graph)
            metas.append(
                SampleMeta(
                    sample_id=sample_id,
                    case_id=case.case_id,
                    case_name=case.case_name,
                    local_step=local_step,
                    t=float(t),
                    dent_center_z=dent_center_z,
                    axis_a=case.axis_a,
                    axis_b=case.axis_b,
                    axis_c=case.axis_c,
                    dent_rx=case.dent_rx,
                    dent_ry=case.dent_ry,
                    dent_rz=case.dent_rz,
                    dent_offset=case.dent_offset,
                    dent_travel=case.dent_travel,
                    depth_gamma=case.depth_gamma,
                    homology_h0=h0,
                    homology_h1=h1,
                )
            )
            sample_id += 1
    return pointclouds, graphs, metas


def save_pca_csv(path: Path, coords: np.ndarray, metas: list[SampleMeta]) -> None:
    rows: list[list[object]] = []
    for meta, (x, y) in zip(metas, coords):
        rows.append(
            [
                meta.sample_id,
                float(x),
                float(y),
                meta.case_id,
                meta.case_name,
                meta.local_step,
                meta.t,
                meta.dent_center_z,
                meta.homology_h0,
                meta.homology_h1,
                meta.axis_a,
                meta.axis_b,
                meta.axis_c,
                meta.dent_rx,
                meta.dent_ry,
                meta.dent_rz,
                meta.dent_offset,
                meta.dent_travel,
                meta.depth_gamma,
            ]
        )
    write_csv(
        path,
        [
            "sample_id",
            "pc1",
            "pc2",
            "case_id",
            "case_name",
            "local_step",
            "t",
            "dent_center_z",
            "H0",
            "H1",
            "axis_a",
            "axis_b",
            "axis_c",
            "dent_rx",
            "dent_ry",
            "dent_rz",
            "dent_offset",
            "dent_travel",
            "depth_gamma",
        ],
        rows,
    )


def save_mds_csv(path: Path, coords: np.ndarray, metas: list[SampleMeta]) -> None:
    rows: list[list[object]] = []
    for meta, (x, y) in zip(metas, coords):
        rows.append(
            [
                meta.sample_id,
                float(x),
                float(y),
                meta.case_id,
                meta.case_name,
                meta.local_step,
                meta.t,
                meta.dent_center_z,
                meta.homology_h0,
                meta.homology_h1,
                meta.axis_a,
                meta.axis_b,
                meta.axis_c,
                meta.dent_rx,
                meta.dent_ry,
                meta.dent_rz,
                meta.dent_offset,
                meta.dent_travel,
                meta.depth_gamma,
            ]
        )
    write_csv(
        path,
        [
            "sample_id",
            "mds1",
            "mds2",
            "case_id",
            "case_name",
            "local_step",
            "t",
            "dent_center_z",
            "H0",
            "H1",
            "axis_a",
            "axis_b",
            "axis_c",
            "dent_rx",
            "dent_ry",
            "dent_rz",
            "dent_offset",
            "dent_travel",
            "depth_gamma",
        ],
        rows,
    )


def main(output_dir: str = "experiments/output/pointcloud_torus") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_legacy_npy(out)

    pointclouds, graphs, metas = generate_dataset()
    sample_ids = [meta.sample_id for meta in metas]
    h0_values = np.asarray([meta.homology_h0 for meta in metas], dtype=np.int64)
    h1_values = np.asarray([meta.homology_h1 for meta in metas], dtype=np.int64)

    image_paths = save_pointcloud_overlay_images(
        pointclouds=pointclouds,
        graphs=graphs,
        sample_ids=sample_ids,
        out_dir=out,
        title_builder=lambda sid: f"ID {sid:03d} | ellipsoid->torus",
    )
    save_test_images_overview(
        image_paths=image_paths,
        sample_ids=sample_ids,
        out_path=out / "test_images_overview.png",
        title="Point Cloud Torus Transition with Graph Overlay",
        cols=6,
    )
    write_csv(
        out / "test_images_index.csv",
        [
            "sample_id",
            "image_path",
            "case_id",
            "case_name",
            "local_step",
            "t",
            "dent_center_z",
            "axis_a",
            "axis_b",
            "axis_c",
            "dent_rx",
            "dent_ry",
            "dent_rz",
            "dent_offset",
            "dent_travel",
            "depth_gamma",
            "H0",
            "H1",
        ],
        [
            [
                meta.sample_id,
                to_repo_relative(path),
                meta.case_id,
                meta.case_name,
                meta.local_step,
                meta.t,
                meta.dent_center_z,
                meta.axis_a,
                meta.axis_b,
                meta.axis_c,
                meta.dent_rx,
                meta.dent_ry,
                meta.dent_rz,
                meta.dent_offset,
                meta.dent_travel,
                meta.depth_gamma,
                meta.homology_h0,
                meta.homology_h1,
            ]
            for meta, path in zip(metas, image_paths)
        ],
    )
    save_pointcloud_lists(pointclouds, out)
    save_weighted_graph_lists(graphs, out)

    spectrum_k = 48
    hist_bins = 32
    dirichlet = boundary_outputs(
        graphs,
        boundary="dirichlet",
        spectrum_k=spectrum_k,
        hist_bins=hist_bins,
        hist_range=None,
        range_quantile=0.995,
        range_padding=1.08,
        range_min_upper=0.10,
        range_max_upper=0.60,
    )
    neumann = boundary_outputs(
        graphs,
        boundary="neumann",
        spectrum_k=spectrum_k,
        hist_bins=hist_bins,
        hist_range=None,
        range_quantile=0.995,
        range_padding=1.08,
        range_min_upper=0.10,
        range_max_upper=0.60,
    )

    save_matrix_csv(out / "dirichlet_features.csv", dirichlet.features, prefix="feature")
    save_matrix_csv(out / "neumann_features.csv", neumann.features, prefix="feature")
    save_step_distances_csv(out / "dirichlet_step_distances.csv", dirichlet.step_distances)
    save_step_distances_csv(out / "neumann_step_distances.csv", neumann.step_distances)
    write_csv(
        out / "dirichlet_histograms.csv",
        ["sample_id", "bin_index", "bin_left", "bin_right", "density"],
        dirichlet.hist_rows,
    )
    write_csv(
        out / "neumann_histograms.csv",
        ["sample_id", "bin_index", "bin_left", "bin_right", "density"],
        neumann.hist_rows,
    )
    write_csv(
        out / "dirichlet_eigenvalues.csv",
        ["sample_id", "eigen_index", "eigenvalue"],
        dirichlet.eig_rows,
    )
    write_csv(
        out / "neumann_eigenvalues.csv",
        ["sample_id", "eigen_index", "eigenvalue"],
        neumann.eig_rows,
    )
    save_matrix_csv(out / "dirichlet_hist_matrix.csv", dirichlet.hist_matrix, prefix="bin")
    save_matrix_csv(out / "neumann_hist_matrix.csv", neumann.hist_matrix, prefix="bin")

    dir_spectrum = dirichlet.features[:, :spectrum_k]
    neu_spectrum = neumann.features[:, :spectrum_k]
    distance_mats = {
        "dirichlet_spectrum_l2": spectrum_distance_matrix(dir_spectrum, metric="l2"),
        "dirichlet_spectrum_wasserstein": spectrum_distance_matrix(
            dir_spectrum, metric="wasserstein"
        ),
        "dirichlet_hist_l1": histogram_distance_matrix(dirichlet.hist_matrix, metric="l1"),
        "dirichlet_hist_js": histogram_distance_matrix(dirichlet.hist_matrix, metric="js"),
        "neumann_spectrum_l2": spectrum_distance_matrix(neu_spectrum, metric="l2"),
        "neumann_spectrum_wasserstein": spectrum_distance_matrix(
            neu_spectrum, metric="wasserstein"
        ),
        "neumann_hist_l1": histogram_distance_matrix(neumann.hist_matrix, metric="l1"),
        "neumann_hist_js": histogram_distance_matrix(neumann.hist_matrix, metric="js"),
    }
    for name, matrix in distance_mats.items():
        save_distance_matrix_csv(out / f"{name}_distance_matrix.csv", matrix, sample_ids=sample_ids)

    save_homology_csv(out / "homology.csv", sample_ids, h0_values, h1_values)

    dir_coords, dir_pca = pca_projection(dirichlet.features, n_components=2)
    neu_coords, neu_pca = pca_projection(neumann.features, n_components=2)
    save_pca_csv(out / "pca_dirichlet.csv", dir_coords, metas)
    save_pca_csv(out / "pca_neumann.csv", neu_coords, metas)
    plot_pca_with_ids(
        dir_coords=dir_coords,
        neu_coords=neu_coords,
        sample_ids=sample_ids,
        color_values=np.asarray([meta.case_id for meta in metas], dtype=np.float64),
        color_label="case_id",
        out_path=out / "pca_trajectory_id_labeled.png",
        cmap="tab10",
        sizes=18.0 + 10.0 * h1_values.astype(np.float64),
        case_ids=np.asarray([meta.case_id for meta in metas], dtype=np.int64),
    )
    plot_pca_homology_colors(
        dir_coords=dir_coords,
        neu_coords=neu_coords,
        h0_values=h0_values,
        h1_values=h1_values,
        out_path=out / "pca_homology_h0_h1.png",
    )
    plot_histogram_matrices(
        dir_hist=dirichlet.hist_matrix,
        neu_hist=neumann.hist_matrix,
        out_path=out / "spectrum_histogram_list.png",
        case_ids=np.asarray([meta.case_id for meta in metas], dtype=np.int64),
    )

    dir_mds_spec, dir_mds_spec_model = mds_projection(
        distance_mats["dirichlet_spectrum_wasserstein"], n_components=2
    )
    neu_mds_spec, neu_mds_spec_model = mds_projection(
        distance_mats["neumann_spectrum_wasserstein"], n_components=2
    )
    dir_mds_hist, dir_mds_hist_model = mds_projection(
        distance_mats["dirichlet_hist_js"], n_components=2
    )
    neu_mds_hist, neu_mds_hist_model = mds_projection(
        distance_mats["neumann_hist_js"], n_components=2
    )
    save_mds_csv(out / "mds_spectrum_wasserstein_dirichlet.csv", dir_mds_spec, metas)
    save_mds_csv(out / "mds_spectrum_wasserstein_neumann.csv", neu_mds_spec, metas)
    save_mds_csv(out / "mds_histogram_js_dirichlet.csv", dir_mds_hist, metas)
    save_mds_csv(out / "mds_histogram_js_neumann.csv", neu_mds_hist, metas)
    plot_mds_with_ids(
        dir_coords=dir_mds_spec,
        neu_coords=neu_mds_spec,
        sample_ids=sample_ids,
        color_values=np.asarray([meta.case_id for meta in metas], dtype=np.float64),
        color_label="case_id",
        out_path=out / "mds_spectrum_wasserstein_id_labeled.png",
        cmap="tab10",
        sizes=18.0 + 10.0 * h1_values.astype(np.float64),
        case_ids=np.asarray([meta.case_id for meta in metas], dtype=np.int64),
    )
    plot_mds_with_ids(
        dir_coords=dir_mds_hist,
        neu_coords=neu_mds_hist,
        sample_ids=sample_ids,
        color_values=np.asarray([meta.case_id for meta in metas], dtype=np.float64),
        color_label="case_id",
        out_path=out / "mds_histogram_js_id_labeled.png",
        cmap="tab10",
        sizes=18.0 + 10.0 * h1_values.astype(np.float64),
        case_ids=np.asarray([meta.case_id for meta in metas], dtype=np.int64),
    )

    case_ids = np.asarray([meta.case_id for meta in metas], dtype=np.int64)
    unique_case_ids = sorted(set(int(v) for v in case_ids.tolist()))
    case_counts = {cid: int(np.sum(case_ids == cid)) for cid in unique_case_ids}
    saved_paths = [
        out / "test_images_index.csv",
        out / "pointcloud_index.csv",
        out / "weighted_graph_summary.csv",
        out / "dirichlet_features.csv",
        out / "neumann_features.csv",
        out / "homology.csv",
        out / "pca_trajectory_id_labeled.png",
        out / "pca_homology_h0_h1.png",
        out / "mds_spectrum_wasserstein_id_labeled.png",
        out / "mds_histogram_js_id_labeled.png",
        out / "spectrum_histogram_list.png",
    ]
    saved_lines = "".join(f"  - {to_repo_relative(path)}\n" for path in saved_paths)
    summary = (
        "Pointcloud ellipsoid-to-torus experiment summary\n"
        f"num_samples: {len(metas)}\n"
        f"num_cases: {len(unique_case_ids)}\n"
        f"case_sample_counts: {case_counts}\n"
        f"H0_range: [{int(h0_values.min())}, {int(h0_values.max())}]\n"
        f"H1_range: [{int(h1_values.min())}, {int(h1_values.max())}]\n"
        "homology_note: H1 is a topology proxy from center-axis puncture detection\n"
        f"dirichlet_feature_dim: {dirichlet.features.shape[1]}\n"
        f"neumann_feature_dim: {neumann.features.shape[1]}\n"
        f"dirichlet_hist_range: {dirichlet.hist_range}\n"
        f"neumann_hist_range: {neumann.hist_range}\n"
        f"dirichlet_pca_explained_var: {dir_pca.explained_variance_ratio_.tolist()}\n"
        f"neumann_pca_explained_var: {neu_pca.explained_variance_ratio_.tolist()}\n"
        f"dirichlet_mds_spectrum_wasserstein_stress: {float(dir_mds_spec_model.stress_):.6f}\n"
        f"neumann_mds_spectrum_wasserstein_stress: {float(neu_mds_spec_model.stress_):.6f}\n"
        f"dirichlet_mds_histogram_js_stress: {float(dir_mds_hist_model.stress_):.6f}\n"
        f"neumann_mds_histogram_js_stress: {float(neu_mds_hist_model.stress_):.6f}\n"
        f"dirichlet_mean_step_distance: {float(dirichlet.step_distances.mean()) if dirichlet.step_distances.size else 0.0:.6f}\n"
        f"neumann_mean_step_distance: {float(neumann.step_distances.mean()) if neumann.step_distances.size else 0.0:.6f}\n"
        "saved_files:\n"
        f"{saved_lines}"
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
