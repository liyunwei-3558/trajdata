from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from Simulation_test_toolchain.core.batch_metrics import (
    MetricContext,
    _compute_offroad_metrics,
    _frames_to_df,
    load_result_json,
)
from Simulation_test_toolchain.run_raw_open_loop_trajectory_batch import (
    _load_manifest,
    _write_outputs,
)
from Simulation_test_toolchain.summarize_batch_manifest import _write_balanced_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute batch offroad metrics using the current boundary implementation."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--balanced-target", type=int, default=250)
    args = parser.parse_args()

    policy_dir = args.manifest.parent
    rows_by_key = _load_manifest(args.manifest)
    rows = list(rows_by_key.values())
    context = MetricContext(args.data_dir, args.dt)

    updated = 0
    for row in rows:
        if row.get("status") != "completed" or not row.get("trajectory_log"):
            continue
        trajectory_path = Path(row["trajectory_log"])
        metrics_path = Path(row["metrics_json"]) if row.get("metrics_json") else None
        if not trajectory_path.exists():
            continue

        result = load_result_json(trajectory_path)
        frames_df = _frames_to_df(result.frames)
        ego_df = frames_df[frames_df["is_ego"]].sort_values("timestep")
        location = str(result.metadata.get("location") or row.get("location"))
        offroad_metrics = _compute_offroad_metrics(
            ego_df, context.road_boundary(location)
        )
        row.update(offroad_metrics)
        _update_metrics_json(metrics_path, offroad_metrics)
        updated += 1

    _write_outputs(policy_dir, rows)
    _write_balanced_summary(policy_dir, rows, args.balanced_target)
    print(f"updated_offroad_rows={updated}")


def _update_metrics_json(path: Path | None, offroad_metrics: dict) -> None:
    if path is None or not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload.get("metrics") or {})
    metrics.update(offroad_metrics)
    payload["metrics"] = metrics
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
