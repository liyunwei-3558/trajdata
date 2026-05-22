from __future__ import annotations

import argparse
from pathlib import Path

from Simulation_test_toolchain.run_narrow_feasible_riskidm_batch import (
    _load_manifest,
    _write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge shard manifests for narrow feasible policy batch runs."
    )
    parser.add_argument(
        "policy_dir",
        type=Path,
        help="Directory containing shard_*/run_manifest.csv files.",
    )
    args = parser.parse_args()

    rows_by_key = {}
    for manifest in sorted(args.policy_dir.glob("shard_*/run_manifest.csv")):
        rows_by_key.update(_load_manifest(manifest))
    if not rows_by_key:
        raise SystemExit(f"No shard manifests found under {args.policy_dir}")
    _write_outputs(args.policy_dir, rows_by_key.values())
    print(
        f"Merged {len(rows_by_key)} rows from shards into {args.policy_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
