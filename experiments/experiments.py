from __future__ import annotations

import argparse
from pathlib import Path

from data_generation import (
    generate_cell_division_dataset,
    generate_holed_disk_sequence,
    generate_holed_disk_pointcloud_dataset,
    generate_pointcloud_torus_dataset,
)
import cell_division
import holed_disk
import pointcloud_holed_disk
import pointcloud_torus


def run_holed_disk(output_dir: Path) -> None:
    masks, metas = generate_holed_disk_sequence()
    holed_disk.main(output_dir=str(output_dir), masks=masks, metas=metas)


def run_cell_division(output_dir: Path) -> None:
    masks, metas = generate_cell_division_dataset()
    cell_division.main(output_dir=str(output_dir), masks=masks, metas=metas)


def run_pointcloud_holed_disk(output_dir: Path) -> None:
    masks, pointclouds, graphs, metas = generate_holed_disk_pointcloud_dataset()
    pointcloud_holed_disk.main(
        output_dir=str(output_dir),
        masks=masks,
        pointclouds=pointclouds,
        graphs=graphs,
        metas=metas,
    )


def run_pointcloud_torus(output_dir: Path) -> None:
    pointclouds, graphs, metas = generate_pointcloud_torus_dataset()
    pointcloud_torus.main(
        output_dir=str(output_dir),
        pointclouds=pointclouds,
        graphs=graphs,
        metas=metas,
    )


def main(target: str = "all", output_root: str = "experiments/output") -> None:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if target in {"all", "holed_disk"}:
        run_holed_disk(out_root / "holed_disk")
    if target in {"all", "cell_division"}:
        run_cell_division(out_root / "cell_division")
    if target in {"all", "pointcloud_holed_disk"}:
        run_pointcloud_holed_disk(out_root / "pointcloud_holed_disk")
    if target in {"all", "pointcloud_torus"}:
        run_pointcloud_torus(out_root / "pointcloud_torus")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument(
        "--target",
        choices=[
            "all",
            "holed_disk",
            "cell_division",
            "pointcloud_holed_disk",
            "pointcloud_torus",
        ],
        default="all",
        help="Experiment target to run",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/output",
        help="Output root directory",
    )
    args = parser.parse_args()
    main(target=args.target, output_root=args.output_root)
