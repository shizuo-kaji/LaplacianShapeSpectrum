from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lapspec.converters import pointcloud2d_to_cotan_graph, pointcloud_to_graph
from lapspec.types import WeightedGraph


# -----------------------------
# Holed disk generation
# -----------------------------


@dataclass(frozen=True, slots=True)
class Hole:
    offset_rc: tuple[float, float]
    radius: float


@dataclass(frozen=True, slots=True)
class HoledDiskSampleMeta:
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


def generate_holed_disk_sequence(
    num_steps: int = 72,
) -> tuple[list[np.ndarray], list[HoledDiskSampleMeta]]:
    masks: list[np.ndarray] = []
    metas: list[HoledDiskSampleMeta] = []
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
            HoledDiskSampleMeta(
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


def _mask_to_planar_pointcloud(
    mask: np.ndarray,
    rng: np.random.Generator,
    points_per_sample: int,
    jitter_std: float,
) -> np.ndarray:
    fg_indices = np.argwhere(mask)
    if fg_indices.shape[0] == 0:
        raise ValueError("mask has no foreground pixels")
    if fg_indices.shape[0] > points_per_sample:
        choice = rng.choice(fg_indices.shape[0], size=points_per_sample, replace=False)
        fg_indices = fg_indices[choice]

    # Map image coordinates (row, col) to planar (x, y) points.
    xy = np.column_stack([fg_indices[:, 1], fg_indices[:, 0]]).astype(np.float64)
    xy += rng.normal(loc=0.0, scale=jitter_std, size=xy.shape)
    center = np.mean(xy, axis=0, keepdims=True)
    xy = xy - center
    return xy


def generate_holed_disk_pointcloud_dataset(
    num_steps: int = 72,
    points_per_sample: int = 900,
    random_seed: int = 0,
    graph_mode: Literal["triangulation_cotan", "knn"] = "triangulation_cotan",
    knn_k: int = 14,
    jitter_std: float = 0.08,
) -> tuple[list[np.ndarray], list[np.ndarray], list[WeightedGraph], list[HoledDiskSampleMeta]]:
    masks, metas = generate_holed_disk_sequence(num_steps=num_steps)
    rng = np.random.default_rng(random_seed)

    pointclouds: list[np.ndarray] = []
    graphs: list[WeightedGraph] = []
    for mask in masks:
        points = _mask_to_planar_pointcloud(
            mask=mask,
            rng=rng,
            points_per_sample=points_per_sample,
            jitter_std=jitter_std,
        )
        if graph_mode == "triangulation_cotan":
            graph = pointcloud2d_to_cotan_graph(
                points,
                boundary_mode="convex_hull",
            )
        elif graph_mode == "knn":
            graph = pointcloud_to_graph(
                points,
                method="knn",
                k=knn_k,
                weight_mode="inverse_distance",
                boundary_mode="outer_percentile",
                boundary_percentile=90.0,
            )
        else:
            raise ValueError("graph_mode must be 'triangulation_cotan' or 'knn'")
        pointclouds.append(points)
        graphs.append(graph)

    return masks, pointclouds, graphs, metas


# -----------------------------
# Cell division generation
# -----------------------------


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
class CellDivisionSampleMeta:
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


def default_cell_division_cases() -> list[DivisionCase]:
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


def _compute_h0_h1(
    mask: np.ndarray,
    min_component_pixels: int,
    min_hole_pixels: int,
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


def generate_cell_division_dataset(
    num_steps_per_case: int = 18,
    size: int = 180,
    min_component_pixels: int = 40,
    min_hole_pixels: int = 40,
) -> tuple[list[np.ndarray], list[CellDivisionSampleMeta]]:
    if num_steps_per_case < 2:
        raise ValueError("num_steps_per_case must be at least 2")

    masks: list[np.ndarray] = []
    metas: list[CellDivisionSampleMeta] = []
    sample_id = 0
    for case in default_cell_division_cases():
        for local_step in range(num_steps_per_case):
            t = local_step / (num_steps_per_case - 1)
            mask, sep = division_mask(size=size, case=case, t=t)
            h0, h1 = _compute_h0_h1(
                mask,
                min_component_pixels=min_component_pixels,
                min_hole_pixels=min_hole_pixels,
            )
            masks.append(mask)
            metas.append(
                CellDivisionSampleMeta(
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


# -----------------------------
# Pointcloud torus generation
# -----------------------------


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
class PointcloudTorusSampleMeta:
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


def default_pointcloud_torus_cases() -> list[TorusCase]:
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


def generate_pointcloud_torus_dataset(
    num_steps_per_case: int = 12,
    grid_size: int = 72,
    extent: float = 1.6,
    points_per_sample: int = 900,
    random_seed: int = 0,
) -> tuple[list[np.ndarray], list[WeightedGraph], list[PointcloudTorusSampleMeta]]:
    if num_steps_per_case < 2:
        raise ValueError("num_steps_per_case must be at least 2")

    axis_values = np.linspace(-extent, extent, grid_size, dtype=np.float64)
    z, y, x = np.meshgrid(axis_values, axis_values, axis_values, indexing="ij")
    rng = np.random.default_rng(random_seed)

    pointclouds: list[np.ndarray] = []
    graphs: list[WeightedGraph] = []
    metas: list[PointcloudTorusSampleMeta] = []
    sample_id = 0
    for case in default_pointcloud_torus_cases():
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
                PointcloudTorusSampleMeta(
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


__all__ = [
    "CellDivisionSampleMeta",
    "DivisionCase",
    "HoledDiskSampleMeta",
    "PointcloudTorusSampleMeta",
    "TorusCase",
    "default_cell_division_cases",
    "default_pointcloud_torus_cases",
    "division_mask",
    "disk_with_holes_mask",
    "generate_cell_division_dataset",
    "generate_holed_disk_pointcloud_dataset",
    "generate_holed_disk_sequence",
    "generate_pointcloud_torus_dataset",
    "radius_schedule",
    "torus_transition_volume",
    "volume_homology_proxy",
    "volume_to_surface_points",
]
