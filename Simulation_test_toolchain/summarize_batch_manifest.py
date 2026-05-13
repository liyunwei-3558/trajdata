from __future__ import annotations

import argparse
from pathlib import Path

from Simulation_test_toolchain.run_raw_open_loop_trajectory_batch import (
    _load_manifest,
    _summary_row,
    _write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite batch summaries from a manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--balanced-target", type=int, default=250)
    args = parser.parse_args()

    policy_dir = args.manifest.parent
    rows = list(_load_manifest(args.manifest).values())
    _write_outputs(policy_dir, rows)
    _write_balanced_summary(policy_dir, rows, args.balanced_target)


def _write_balanced_summary(policy_dir: Path, rows: list[dict], target: int) -> None:
    import pandas as pd

    df = pd.DataFrame.from_records(rows)
    if df.empty or "intersection" not in df.columns:
        return

    selected_groups = []
    for _, group in df.groupby("intersection"):
        group = group.copy()
        group["_status_order"] = group["status"].map({"completed": 0, "failed": 1}).fillna(2)
        group["_candidate_rank_num"] = pd.to_numeric(
            group.get("candidate_rank"), errors="coerce"
        )
        group = group.sort_values(
            by=[
                "_status_order",
                "_candidate_rank_num",
                "location",
                "scene_name",
                "run_id",
            ],
            ascending=[True, True, True, True, True],
        )
        selected_groups.append(
            group.head(target).drop(columns=["_status_order", "_candidate_rank_num"])
        )
    balanced = pd.concat(selected_groups, ignore_index=True)
    summary = [_summary_row("ALL_BALANCED", balanced)]
    by_intersection = [
        _summary_row(intersection, group)
        for intersection, group in balanced.groupby("intersection")
    ]
    by_intersection_df = pd.DataFrame.from_records(by_intersection)
    if "location" in by_intersection_df.columns:
        by_intersection_df = by_intersection_df.rename(columns={"location": "intersection"})
    by_intersection_df.to_csv(
        policy_dir / "summary_by_intersection_balanced.csv", index=False
    )
    pd.DataFrame.from_records(summary).to_csv(
        policy_dir / "summary_total_balanced.csv", index=False
    )


if __name__ == "__main__":
    main()
