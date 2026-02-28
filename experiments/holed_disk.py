from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared import (
    boundary_outputs,
    clean_legacy_npy,
    homology_arrays,
    plot_histogram_matrices,
    plot_mds_with_ids,
    plot_pca_homology_colors,
    plot_pca_with_ids,
    save_distance_matrix_csv,
    save_homology_csv,
    save_matrix_csv,
    save_overlay_test_images,
    save_step_distances_csv,
    save_test_images_overview,
    save_weighted_graph_lists,
    to_repo_relative,
    write_csv,
)
from lapspec.converters import image_to_graph
from lapspec.metrics import histogram_distance_matrix, spectrum_distance_matrix
from lapspec.visualization import mds_projection, pca_projection


@dataclass(frozen=True, slots=True)
class Hole:
    offset_rc: tuple[float, float]
    radius: float


@dataclass(frozen=True, slots=True)
class SampleMeta:
    sample_id: int
    t: float
    r1: float
    r2: float
    r3: float
    r4: float
    r5: float
    active_holes: int


def disk_with_holes_mask(
    size: int,
    outer_radius: float,
    holes: list[Hole],
) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= outer_radius**2
    for hole in holes:
        if hole.radius <= 0:
            continue
        hr = center + hole.offset_rc[0]
        hc = center + hole.offset_rc[1]
        hole_region = (yy - hr) ** 2 + (xx - hc) ** 2 <= hole.radius**2
        mask = np.logical_and(mask, ~hole_region)
    return mask


def _periodic_distance(t: float, center: float) -> float:
    return abs(((t - center + 0.5) % 1.0) - 0.5)


def _cosine_pulse(t: float, center: float, width: float) -> float:
    distance = _periodic_distance(t, center)
    if distance >= width:
        return 0.0
    return float(0.5 * (1.0 + np.cos(np.pi * distance / width)))


def radius_schedule(t: float) -> tuple[float, float, float, float, float]:
    centers = [0.22, 0.34, 0.46, 0.58, 0.70]
    amplitudes = [9.0, 8.5, 8.0, 7.5, 7.0]
    width = 0.28
    radii = [
        amplitude * _cosine_pulse(t, center=center, width=width)
        for amplitude, center in zip(amplitudes, centers)
    ]
    return (float(radii[0]), float(radii[1]), float(radii[2]), float(radii[3]), float(radii[4]))


def generate_sequence(num_steps: int = 72) -> tuple[list[np.ndarray], list[SampleMeta]]:
    masks: list[np.ndarray] = []
    metas: list[SampleMeta] = []
    for step in range(num_steps):
        t = step / (num_steps - 1)
        r1, r2, r3, r4, r5 = radius_schedule(t)
        holes = [
            Hole(offset_rc=(-16.0, -8.0), radius=r1),
            Hole(offset_rc=(-14.0, 14.0), radius=r2),
            Hole(offset_rc=(-2.0, -22.0), radius=r3),
            Hole(offset_rc=(14.0, 10.0), radius=r4),
            Hole(offset_rc=(20.0, -2.0), radius=r5),
        ]
        masks.append(disk_with_holes_mask(size=160, outer_radius=60.0, holes=holes))
        active = int(sum(r > 0 for r in (r1, r2, r3, r4, r5)))
        metas.append(
            SampleMeta(
                sample_id=step,
                t=float(t),
                r1=float(r1),
                r2=float(r2),
                r3=float(r3),
                r4=float(r4),
                r5=float(r5),
                active_holes=active,
            )
        )
    return masks, metas


def save_pca_csv(
    path: Path,
    coords: np.ndarray,
    metas: list[SampleMeta],
    h0_values: np.ndarray,
    h1_values: np.ndarray,
) -> None:
    rows: list[list[object]] = []
    for meta, (x, y), h0, h1 in zip(metas, coords, h0_values, h1_values):
        rows.append(
            [
                meta.sample_id,
                float(x),
                float(y),
                meta.t,
                meta.r1,
                meta.r2,
                meta.r3,
                meta.r4,
                meta.r5,
                meta.active_holes,
                int(h0),
                int(h1),
            ]
        )
    write_csv(
        path,
        [
            "sample_id",
            "pc1",
            "pc2",
            "t",
            "r1",
            "r2",
            "r3",
            "r4",
            "r5",
            "active_holes",
            "H0",
            "H1",
        ],
        rows,
    )


def save_mds_csv(
    path: Path,
    coords: np.ndarray,
    metas: list[SampleMeta],
    h0_values: np.ndarray,
    h1_values: np.ndarray,
) -> None:
    rows: list[list[object]] = []
    for meta, (x, y), h0, h1 in zip(metas, coords, h0_values, h1_values):
        rows.append(
            [
                meta.sample_id,
                float(x),
                float(y),
                meta.t,
                meta.r1,
                meta.r2,
                meta.r3,
                meta.r4,
                meta.r5,
                meta.active_holes,
                int(h0),
                int(h1),
            ]
        )
    write_csv(
        path,
        [
            "sample_id",
            "mds1",
            "mds2",
            "t",
            "r1",
            "r2",
            "r3",
            "r4",
            "r5",
            "active_holes",
            "H0",
            "H1",
        ],
        rows,
    )


def main(output_dir: str = "experiments/output/holed_disk") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_legacy_npy(out)

    masks, metas = generate_sequence()
    sample_ids = [meta.sample_id for meta in metas]
    graphs = [image_to_graph(mask, connectivity=8, weight_mode="distance") for mask in masks]

    image_paths = save_overlay_test_images(
        masks=masks,
        graphs=graphs,
        sample_ids=sample_ids,
        out_dir=out,
        title_builder=lambda sid: f"ID {sid:03d}",
    )
    save_test_images_overview(
        image_paths=image_paths,
        sample_ids=sample_ids,
        out_path=out / "test_images_overview.png",
        title="Holed Disk Test Images with Graph Overlay",
    )
    write_csv(
        out / "test_images_index.csv",
        ["sample_id", "image_path", "t", "r1", "r2", "r3", "r4", "r5", "active_holes"],
        [
            [
                meta.sample_id,
                to_repo_relative(path),
                meta.t,
                meta.r1,
                meta.r2,
                meta.r3,
                meta.r4,
                meta.r5,
                meta.active_holes,
            ]
            for meta, path in zip(metas, image_paths)
        ],
    )
    save_weighted_graph_lists(graphs, out)

    spectrum_k = 48
    hist_bins = 32
    dirichlet = boundary_outputs(
        graphs=graphs,
        boundary="dirichlet",
        spectrum_k=spectrum_k,
        hist_bins=hist_bins,
        hist_range=None,
        range_quantile=0.995,
        range_padding=1.08,
        range_min_upper=0.03,
        range_max_upper=0.12,
    )
    neumann = boundary_outputs(
        graphs=graphs,
        boundary="neumann",
        spectrum_k=spectrum_k,
        hist_bins=hist_bins,
        hist_range=None,
        range_quantile=0.995,
        range_padding=1.08,
        range_min_upper=0.03,
        range_max_upper=0.12,
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
        save_distance_matrix_csv(
            out / f"{name}_distance_matrix.csv",
            matrix,
            sample_ids=sample_ids,
        )

    h0_values, h1_values = homology_arrays(masks, min_component_pixels=3, min_hole_pixels=3)
    save_homology_csv(out / "homology.csv", sample_ids, h0_values, h1_values)

    dir_coords, dir_pca = pca_projection(dirichlet.features, n_components=2)
    neu_coords, neu_pca = pca_projection(neumann.features, n_components=2)
    save_pca_csv(out / "pca_dirichlet.csv", dir_coords, metas, h0_values, h1_values)
    save_pca_csv(out / "pca_neumann.csv", neu_coords, metas, h0_values, h1_values)
    plot_pca_with_ids(
        dir_coords=dir_coords,
        neu_coords=neu_coords,
        sample_ids=sample_ids,
        color_values=np.asarray([meta.active_holes for meta in metas], dtype=np.float64),
        color_label="active holes",
        out_path=out / "pca_trajectory_id_labeled.png",
        cmap="viridis",
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
    save_mds_csv(out / "mds_spectrum_wasserstein_dirichlet.csv", dir_mds_spec, metas, h0_values, h1_values)
    save_mds_csv(out / "mds_spectrum_wasserstein_neumann.csv", neu_mds_spec, metas, h0_values, h1_values)
    save_mds_csv(out / "mds_histogram_js_dirichlet.csv", dir_mds_hist, metas, h0_values, h1_values)
    save_mds_csv(out / "mds_histogram_js_neumann.csv", neu_mds_hist, metas, h0_values, h1_values)
    plot_mds_with_ids(
        dir_coords=dir_mds_spec,
        neu_coords=neu_mds_spec,
        sample_ids=sample_ids,
        color_values=np.asarray([meta.active_holes for meta in metas], dtype=np.float64),
        color_label="active holes",
        out_path=out / "mds_spectrum_wasserstein_id_labeled.png",
        cmap="viridis",
    )
    plot_mds_with_ids(
        dir_coords=dir_mds_hist,
        neu_coords=neu_mds_hist,
        sample_ids=sample_ids,
        color_values=np.asarray([meta.active_holes for meta in metas], dtype=np.float64),
        color_label="active holes",
        out_path=out / "mds_histogram_js_id_labeled.png",
        cmap="viridis",
    )

    active_holes_values = np.asarray([meta.active_holes for meta in metas], dtype=np.int64)
    saved_paths = [
        out / "test_images_index.csv",
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
        "Synthetic holes experiment summary\n"
        f"num_samples: {len(metas)}\n"
        f"dirichlet_feature_dim: {dirichlet.features.shape[1]}\n"
        f"neumann_feature_dim: {neumann.features.shape[1]}\n"
        f"dirichlet_hist_range: {dirichlet.hist_range}\n"
        f"neumann_hist_range: {neumann.hist_range}\n"
        f"max_active_holes: {int(active_holes_values.max()) if active_holes_values.size else 0}\n"
        f"H0_range: [{int(h0_values.min())}, {int(h0_values.max())}]\n"
        f"H1_range: [{int(h1_values.min())}, {int(h1_values.max())}]\n"
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
