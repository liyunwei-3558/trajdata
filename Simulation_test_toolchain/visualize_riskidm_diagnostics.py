from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPoint

from trajdata.dataset_specific.sind.scene_filters import load_curbstone_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RiskIDM diagnostic trajectory plots.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "Simulation_test_toolchain/batch_outputs/raw_open_loop_250traj/"
            "raw_open_loop_risk_idm/run_manifest.csv"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Simulation_test_toolchain/batch_outputs/raw_open_loop_250traj/diagnostics/risk_idm"),
    )
    parser.add_argument("--run-id", action="append", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(args.manifest)
    selected = _select_rows(rows, args.run_id)
    index_rows: List[Dict[str, Any]] = []
    for row in selected:
        output_path = args.output_dir / f"{_safe_name(row['run_id'])}.png"
        render_case(row, args.data_dir, output_path)
        index_rows.append(
            {
                "run_id": row["run_id"],
                "location": row["location"],
                "ADE": row["ADE"],
                "FDE": row["FDE"],
                "offroad": row["offroad"],
                "collision": row["collision"],
                "path": str(output_path),
            }
        )
        print(output_path)
    pd.DataFrame.from_records(index_rows).to_csv(args.output_dir / "index.csv", index=False)


def render_case(row: Mapping[str, str], data_dir: Path, output_path: Path) -> None:
    result = json.loads(Path(row["trajectory_log"]).read_text(encoding="utf-8"))
    frames = pd.DataFrame.from_records(result["frames"])
    ego = frames[frames["is_ego"]].sort_values("timestep")
    non_ego = frames[~frames["is_ego"]].sort_values("timestep")

    location = row["location"]
    scene_id = row["scene_id"]
    agent_id = row["agent_id"]
    gt = _load_gt(data_dir, location, scene_id, agent_id)
    drivable_polys = _load_curbstone_hull_polygons(location)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{row['location']} | {row['run_id']}\n"
        f"ADE={float(row['ADE']):.2f}, FDE={float(row['FDE']):.2f}, "
        f"offroad={row['offroad']}, collision={row['collision']}",
        fontsize=10,
    )

    for poly in drivable_polys:
        patch = MplPolygon(poly, closed=True, facecolor="#d8ead2", edgecolor="#72a06a", alpha=0.35, linewidth=0.8)
        ax.add_patch(patch)

    for _, agent_df in non_ego.groupby("agent_name"):
        xy = agent_df[["x", "y"]].to_numpy(dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], color="#4e79a7", alpha=0.16, linewidth=0.7)

    if not gt.empty:
        gt_win = gt[
            (gt["frame_id"] >= int(ego["timestep"].min()))
            & (gt["frame_id"] <= int(ego["timestep"].max()))
        ]
        ax.plot(
            gt_win["x"],
            gt_win["y"],
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="GT ego",
        )
        ax.scatter(gt_win["x"].iloc[0], gt_win["y"].iloc[0], color="black", s=42, marker="o", label="GT start")
        ax.scatter(gt_win["x"].iloc[-1], gt_win["y"].iloc[-1], color="black", s=56, marker="x", label="GT end")

    ax.plot(ego["x"], ego["y"], color="#d62728", linewidth=2.3, label="RiskIDM ego")
    ax.scatter(ego["x"].iloc[0], ego["y"].iloc[0], color="#d62728", s=42, marker="o", label="RiskIDM start")
    ax.scatter(ego["x"].iloc[-1], ego["y"].iloc[-1], color="#d62728", s=56, marker="x", label="RiskIDM end")

    pad = 10.0
    xs = list(ego["x"])
    ys = list(ego["y"])
    if not gt.empty:
        xs += list(gt["x"])
        ys += list(gt["y"])
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_rows(rows: Sequence[Dict[str, str]], run_ids: Sequence[str]) -> List[Dict[str, str]]:
    if run_ids:
        by_id = {row["run_id"]: row for row in rows}
        return [by_id[run_id] for run_id in run_ids if run_id in by_id]

    completed = [row for row in rows if row["status"] == "completed"]
    for row in completed:
        row["_ADE"] = float(row["ADE"])
        row["_FDE"] = float(row["FDE"])
    high = sorted(completed, key=lambda row: row["_ADE"], reverse=True)[:3]
    low_offroad = sorted(
        [row for row in completed if row.get("offroad", "").lower() == "true"],
        key=lambda row: row["_ADE"],
    )[:3]
    chosen: List[Dict[str, str]] = []
    seen = set()
    for row in high + low_offroad:
        if row["run_id"] in seen:
            continue
        chosen.append(row)
        seen.add(row["run_id"])
    return chosen


def _load_gt(data_dir: Path, location: str, scene_id: str, agent_id: str) -> pd.DataFrame:
    tp_path = data_dir / location / f"tp_info_{location}.pkl"
    with tp_path.open("rb") as handle:
        tp_info = pickle.load(handle)
    scene_tracks = tp_info[scene_id]
    tp_data = None
    for key, value in scene_tracks.items():
        if str(key) == str(agent_id):
            tp_data = value
            break
    if tp_data is None:
        raise KeyError(f"agent_id={agent_id!r} not found in {location}/{scene_id}")
    return tp_data["State"].sort_values("frame_id").copy()


def _load_curbstone_hull_polygons(location: str) -> List[np.ndarray]:
    try:
        city_points = load_curbstone_points()
        curbstone = city_points[location]
        points = []
        for xs, ys in zip(curbstone["curbston_x"], curbstone["curbston_y"]):
            points.extend((float(x), float(y)) for x, y in zip(xs, ys))
        hull = MultiPoint(points).convex_hull
        if hull.is_empty or hull.geom_type != "Polygon":
            return []
        return [np.asarray(hull.exterior.coords, dtype=float)]
    except Exception:
        return []


def _safe_name(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(".", "_")
    )


if __name__ == "__main__":
    main()
