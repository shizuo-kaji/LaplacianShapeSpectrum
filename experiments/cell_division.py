from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from shared import (
    boundary_outputs,
    clean_legacy_npy,
    compute_h0_h1,
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
class DivisionCase:
    case_id: int
    case_name: str
    axis_a: float
    axis_b: float
    angle_deg: float
    sep_max: float
    threshold: float
    threshold_ramp: float
    profile_gamma: float
    lobe_balance: float
    neck_strength: float


@dataclass(frozen=True, slots=True)
class SampleMeta:
    sample_id: int
    case_id: int
    case_name: str
    local_step: int
    t: float
    sep: float
    axis_a: float
    axis_b: float
    angle_deg: float
    threshold: float
    threshold_ramp: float
    profile_gamma: float
    lobe_balance: float
    neck_strength: float
    homology_h0: int
    homology_h1: int


def default_cases() -> list[DivisionCase]:
    return [
        DivisionCase(
            case_id=0,
            case_name="round_soft",
            axis_a=24.0,
            axis_b=24.0,
            angle_deg=0.0,
            sep_max=44.0,
            threshold=0.64,
            threshold_ramp=0.08,
            profile_gamma=1.0,
            lobe_balance=1.00,
            neck_strength=0.16,
        ),
        DivisionCase(
            case_id=1,
            case_name="elongated_tilt",
            axis_a=32.0,
            axis_b=18.0,
            angle_deg=22.0,
            sep_max=62.0,
            threshold=0.62,
            threshold_ramp=0.14,
            profile_gamma=1.35,
            lobe_balance=1.10,
            neck_strength=0.26,
        ),
        DivisionCase(
            case_id=2,
            case_name="flat_wide",
            axis_a=36.0,
            axis_b=14.5,
            angle_deg=-18.0,
            sep_max=56.0,
            threshold=0.60,
            threshold_ramp=0.16,
            profile_gamma=0.80,
            lobe_balance=0.92,
            neck_strength=0.30,
        ),
        DivisionCase(
            case_id=3,
            case_name="tall_rotated",
            axis_a=19.0,
            axis_b=30.0,
            angle_deg=66.0,
            sep_max=40.0,
            threshold=0.66,
            threshold_ramp=0.08,
            profile_gamma=1.20,
            lobe_balance=1.05,
            neck_strength=0.18,
        ),
        DivisionCase(
            case_id=4,
            case_name="asymmetric_split",
            axis_a=28.0,
            axis_b=20.0,
            angle_deg=-34.0,
            sep_max=52.0,
            threshold=0.63,
            threshold_ramp=0.14,
            profile_gamma=1.55,
            lobe_balance=1.22,
            neck_strength=0.27,
        ),
    ]


def _rotated_coords(size: int, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    center = (size - 1) / 2.0
    dx = xx - center
    dy = yy - center
    theta = np.deg2rad(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    return xr, yr


def division_mask(size: int, case: DivisionCase, t: float) -> tuple[np.ndarray, float]:
    t_clamped = min(max(float(t), 0.0), 1.0)
    sep = case.sep_max * (t_clamped**case.profile_gamma)

    xr, yr = _rotated_coords(size=size, angle_deg=case.angle_deg)
    left = np.exp(
        -((((xr + 0.5 * sep) / case.axis_a) ** 2) + ((yr / case.axis_b) ** 2))
    )
    right = case.lobe_balance * np.exp(
        -((((xr - 0.5 * sep) / case.axis_a) ** 2) + ((yr / case.axis_b) ** 2))
    )
    neck_profile = t_clamped**1.25
    neck = case.neck_strength * neck_profile * np.exp(
        -((((xr) / (0.28 * case.axis_a)) ** 2) + ((yr / (0.95 * case.axis_b)) ** 2))
    )
    field = left + right - neck
    effective_threshold = case.threshold + case.threshold_ramp * (t_clamped**1.3)
    mask = field >= effective_threshold
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool), float(sep)


def generate_dataset(
    num_steps_per_case: int = 18,
    size: int = 180,
) -> tuple[list[np.ndarray], list[SampleMeta]]:
    if num_steps_per_case < 2:
        raise ValueError("num_steps_per_case must be at least 2")

    masks: list[np.ndarray] = []
    metas: list[SampleMeta] = []
    sample_id = 0
    for case in default_cases():
        for local_step in range(num_steps_per_case):
            t = local_step / (num_steps_per_case - 1)
            mask, sep = division_mask(size=size, case=case, t=t)
            h0, h1 = compute_h0_h1(mask, min_component_pixels=40, min_hole_pixels=40)
            masks.append(mask)
            metas.append(
                SampleMeta(
                    sample_id=sample_id,
                    case_id=case.case_id,
                    case_name=case.case_name,
                    local_step=local_step,
                    t=float(t),
                    sep=sep,
                    axis_a=case.axis_a,
                    axis_b=case.axis_b,
                    angle_deg=case.angle_deg,
                    threshold=case.threshold,
                    threshold_ramp=case.threshold_ramp,
                    profile_gamma=case.profile_gamma,
                    lobe_balance=case.lobe_balance,
                    neck_strength=case.neck_strength,
                    homology_h0=h0,
                    homology_h1=h1,
                )
            )
            sample_id += 1
    return masks, metas


def save_pca_csv(
    path: Path,
    coords: np.ndarray,
    metas: list[SampleMeta],
) -> None:
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
                meta.sep,
                meta.homology_h0,
                meta.homology_h1,
                meta.axis_a,
                meta.axis_b,
                meta.angle_deg,
                meta.threshold,
                meta.threshold_ramp,
                meta.profile_gamma,
                meta.lobe_balance,
                meta.neck_strength,
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
            "sep",
            "H0",
            "H1",
            "axis_a",
            "axis_b",
            "angle_deg",
            "threshold",
            "threshold_ramp",
            "profile_gamma",
            "lobe_balance",
            "neck_strength",
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
                meta.sep,
                meta.homology_h0,
                meta.homology_h1,
                meta.axis_a,
                meta.axis_b,
                meta.angle_deg,
                meta.threshold,
                meta.threshold_ramp,
                meta.profile_gamma,
                meta.lobe_balance,
                meta.neck_strength,
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
            "sep",
            "H0",
            "H1",
            "axis_a",
            "axis_b",
            "angle_deg",
            "threshold",
            "threshold_ramp",
            "profile_gamma",
            "lobe_balance",
            "neck_strength",
        ],
        rows,
    )


def main(output_dir: str = "experiments/output/cell_division") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_legacy_npy(out)

    masks, metas = generate_dataset()
    sample_ids = [meta.sample_id for meta in metas]
    graphs = [image_to_graph(mask, connectivity=8, weight_mode="distance") for mask in masks]

    image_paths = save_overlay_test_images(
        masks=masks,
        graphs=graphs,
        sample_ids=sample_ids,
        out_dir=out,
        title_builder=lambda sid: f"ID {sid:03d} | cell division",
    )
    save_test_images_overview(
        image_paths=image_paths,
        sample_ids=sample_ids,
        out_path=out / "test_images_overview.png",
        title="Cell Division Test Images with Graph Overlay",
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
            "sep",
            "axis_a",
            "axis_b",
            "angle_deg",
            "threshold",
            "threshold_ramp",
            "profile_gamma",
            "lobe_balance",
            "neck_strength",
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
                meta.sep,
                meta.axis_a,
                meta.axis_b,
                meta.angle_deg,
                meta.threshold,
                meta.threshold_ramp,
                meta.profile_gamma,
                meta.lobe_balance,
                meta.neck_strength,
                meta.homology_h0,
                meta.homology_h1,
            ]
            for meta, path in zip(metas, image_paths)
        ],
    )
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
        range_min_upper=0.12,
        range_max_upper=0.50,
    )
    neumann = boundary_outputs(
        graphs,
        boundary="neumann",
        spectrum_k=spectrum_k,
        hist_bins=hist_bins,
        hist_range=None,
        range_quantile=0.995,
        range_padding=1.08,
        range_min_upper=0.12,
        range_max_upper=0.50,
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

    h0_values = np.asarray([meta.homology_h0 for meta in metas], dtype=np.int64)
    h1_values = np.asarray([meta.homology_h1 for meta in metas], dtype=np.int64)
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
        sizes=18.0 + 10.0 * np.asarray(h0_values - 1, dtype=np.float64),
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
        sizes=18.0 + 10.0 * np.asarray(h0_values - 1, dtype=np.float64),
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
        sizes=18.0 + 10.0 * np.asarray(h0_values - 1, dtype=np.float64),
        case_ids=np.asarray([meta.case_id for meta in metas], dtype=np.int64),
    )

    case_ids = np.asarray([meta.case_id for meta in metas], dtype=np.int64)
    unique_case_ids = sorted(set(int(v) for v in case_ids.tolist()))
    case_counts = {cid: int(np.sum(case_ids == cid)) for cid in unique_case_ids}
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
        "Cell division experiment summary\n"
        f"num_samples: {len(metas)}\n"
        f"num_cases: {len(unique_case_ids)}\n"
        f"case_sample_counts: {case_counts}\n"
        f"H0_range: [{int(h0_values.min())}, {int(h0_values.max())}]\n"
        f"H1_range: [{int(h1_values.min())}, {int(h1_values.max())}]\n"
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
