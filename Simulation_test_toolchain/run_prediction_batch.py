from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from Simulation_test_toolchain.run_qcnet_prediction_eval import (
    DEFAULT_QCNET_CKPT,
    DEFAULT_QCNET_REPO,
    DEFAULT_TRACE_REPO,
    add_trace_repo_path,
    base_row,
    build_dataset,
    build_diffuser_dataset,
    build_prediction_plans,
    load_qcnet_model,
    run_diffuser_prediction_sample,
    run_prediction_sample,
    save_prediction_html,
    summarize_rows,
    write_csv,
)


DEFAULT_LOCATIONS = ("cc", "tj", "cqIR", "cqNR", "cqR", "xa")
POLICY_OUTPUT_DIRS = {
    "qcnet": "qcnet",
    "diffuser": "trace_diffuser",
}


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    all_summary_rows: List[Dict[str, Any]] = []

    for policy in args.policies:
        policy_dir = output_root / POLICY_OUTPUT_DIRS[policy]
        policy_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[batch-pred] policy={policy} locations={','.join(args.locations)} "
            f"target_per_location={args.target_per_location}",
            flush=True,
        )
        model_bundle = load_policy_bundle(policy, args)
        policy_args = effective_policy_args(args, model_bundle)

        for location in args.locations:
            loc_started = time.time()
            loc_dir = policy_dir / location
            loc_dir.mkdir(parents=True, exist_ok=True)
            rows_path = loc_dir / "prediction_metrics.csv"
            rows = load_rows(rows_path)
            completed = completed_count(rows)
            if completed >= args.target_per_location and not args.force:
                print(
                    f"[batch-pred] skip policy={policy} location={location} completed={completed}",
                    flush=True,
                )
                all_summary_rows.append(location_summary(policy, location, rows))
                continue

            loc_args = argparse.Namespace(**vars(policy_args))
            loc_args.location = location
            loc_args.output_dir = loc_dir
            loc_args.no_html = True
            plans = build_prediction_plans(
                data_dir=loc_args.data_dir.expanduser(),
                location=location,
                history_steps=model_bundle["history_steps"],
                future_steps=model_bundle["future_steps"],
                min_displacement_m=loc_args.min_displacement_m,
                min_mean_speed_mps=loc_args.min_mean_speed_mps,
                dt=loc_args.dt,
            )
            if args.max_candidates is not None:
                plans = plans[: args.max_candidates]
            print(
                f"[batch-pred] policy={policy} location={location} candidates={len(plans)} "
                f"already_completed={completed}",
                flush=True,
            )
            dataset, scenes = (
                build_dataset(loc_args)
                if policy == "qcnet"
                else build_diffuser_dataset(loc_args)
            )
            scene_by_name = {scene.name: scene for scene in scenes}
            predictions = load_predictions(loc_dir / "prediction_metrics.json")
            rows_by_run = {str(row.get("run_id")): row for row in rows}

            for plan in plans:
                if completed >= args.target_per_location:
                    break
                if (
                    plan.run_id in rows_by_run
                    and rows_by_run[plan.run_id].get("status") == "completed"
                    and not args.force
                ):
                    continue
                row = base_row(plan)
                row["policy"] = policy
                row["history_sec"] = loc_args.history_sec
                row["future_sec"] = loc_args.future_sec
                row["history_steps"] = model_bundle["history_steps"]
                row["future_steps"] = model_bundle["future_steps"]
                row["min_displacement_m"] = loc_args.min_displacement_m
                row["min_mean_speed_mps"] = loc_args.min_mean_speed_mps
                sample_started = time.time()
                try:
                    scene = scene_by_name.get(plan.scene_name)
                    if scene is None:
                        if 0 <= plan.scene_index < len(scenes):
                            scene = scenes[plan.scene_index]
                        else:
                            raise IndexError(
                                f"scene not found for {plan.scene_name!r} "
                                f"(scene_index={plan.scene_index}, scenes={len(scenes)})"
                            )
                    sample = run_one_sample(
                        policy=policy,
                        plan=plan,
                        scene=scene,
                        dataset=dataset,
                        model_bundle=model_bundle,
                        args=loc_args,
                    )
                    row.update(sample["metrics"])
                    row["status"] = "completed"
                    row["error"] = ""
                    row["valid_future_steps"] = sample["valid_future_steps"]
                    row["top1_mode_index"] = sample["top1_mode_index"]
                    row["best_mode_index"] = sample["best_mode_index"]
                    row["top1_mode_prob"] = sample["top1_mode_prob"]
                    row["best_mode_prob"] = sample["best_mode_prob"]
                    predictions = upsert_prediction(predictions, sample["prediction"])
                    completed += 1
                    print(
                        f"[batch-pred] {policy} {location} completed={completed}/"
                        f"{args.target_per_location} run={plan.run_id} "
                        f"minFDE={row['minFDE']:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                    print(
                        f"[batch-pred] failed policy={policy} location={location} "
                        f"run={plan.run_id}: {exc}",
                        flush=True,
                    )
                finally:
                    row["elapsed_s"] = round(time.time() - sample_started, 3)
                    rows_by_run[plan.run_id] = row
                    rows = list(rows_by_run.values())
                    write_location_outputs(
                        loc_dir=loc_dir,
                        rows=rows,
                        predictions=predictions,
                        args=loc_args,
                        policy=policy,
                        elapsed_s=round(time.time() - loc_started, 3),
                        html=args.save_html,
                    )

            all_summary_rows.append(location_summary(policy, location, rows))
            write_csv(
                policy_dir / "summary_by_location.csv",
                [row for row in all_summary_rows if row["policy"] == policy],
            )
            write_csv(output_root / "summary_all.csv", all_summary_rows)

    print(f"[batch-pred] output: {output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched one-shot trajectory prediction evaluation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Simulation_test_toolchain/batch_outputs/prediction_1500"),
    )
    parser.add_argument("--locations", nargs="+", default=list(DEFAULT_LOCATIONS))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=sorted(POLICY_OUTPUT_DIRS),
        default=["qcnet", "diffuser"],
    )
    parser.add_argument("--target-per-location", type=int, default=250)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=20.0)
    parser.add_argument("--map-radius", type=float, default=20.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--max-guesses", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--min-displacement-m", type=float, default=5.0)
    parser.add_argument("--min-mean-speed-mps", type=float, default=0.2)
    parser.add_argument("--qcnet-repo-path", type=str, default=DEFAULT_QCNET_REPO)
    parser.add_argument("--qcnet-ckpt-path", type=str, default=DEFAULT_QCNET_CKPT)
    parser.add_argument("--trace-repo-path", type=str, default=DEFAULT_TRACE_REPO)
    parser.add_argument("--diffuser-ckpt-path", type=str, default=None)
    parser.add_argument("--diffuser-num-samples", type=int, default=None)
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_policy_bundle(policy: str, args: argparse.Namespace) -> Dict[str, Any]:
    if policy == "qcnet":
        torch, model = load_qcnet_model(
            repo_path=args.qcnet_repo_path,
            ckpt_path=args.qcnet_ckpt_path,
            device=args.device,
            strict=args.strict,
        )
        return {
            "torch": torch,
            "model": model,
            "history_steps": int(getattr(model, "num_historical_steps")),
            "future_steps": int(getattr(model, "num_future_steps")),
        }
    if policy == "diffuser":
        if not args.diffuser_ckpt_path:
            raise ValueError(
                "TRACE/Diffuser evaluation requires --diffuser-ckpt-path. "
                "No Diffuser checkpoint was found in this repository."
            )
        add_trace_repo_path(args.trace_repo_path)
        from Simulation_test_toolchain.policies.diffuser_policy import DiffuserPolicy

        policy_obj = DiffuserPolicy(
            dt=args.dt,
            ckpt_path=args.diffuser_ckpt_path,
            device=args.device,
        )
        trace_num_samples = int(
            policy_obj.trace_config.get("algo", {})
            .get("diffuser", {})
            .get("num_eval_samples", 10)
        )
        return {
            "policy": policy_obj,
            "history_steps": int(round(args.history_sec / args.dt)) + 1,
            "future_steps": int(getattr(policy_obj.model, "horizon", round(args.future_sec / args.dt))),
            "num_samples": int(args.diffuser_num_samples or trace_num_samples),
        }
    raise ValueError(f"Unsupported policy: {policy}")


def effective_policy_args(
    args: argparse.Namespace, model_bundle: Mapping[str, Any]
) -> argparse.Namespace:
    policy_args = argparse.Namespace(**vars(args))
    policy_args.history_sec = max(
        0.0, (int(model_bundle["history_steps"]) - 1) * float(args.dt)
    )
    policy_args.future_sec = int(model_bundle["future_steps"]) * float(args.dt)
    return policy_args


def run_one_sample(
    *,
    policy: str,
    plan,
    scene,
    dataset,
    model_bundle: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if policy == "qcnet":
        return run_prediction_sample(
            plan=plan,
            scene=scene,
            dataset=dataset,
            model=model_bundle["model"],
            torch=model_bundle["torch"],
            args=args,
            hist_steps=model_bundle["history_steps"],
            fut_steps=model_bundle["future_steps"],
        )
    if policy == "diffuser":
        return run_diffuser_prediction_sample(
            plan=plan,
            scene=scene,
            dataset=dataset,
            policy=model_bundle["policy"],
            args=args,
            fut_steps=model_bundle["future_steps"],
            num_samples=model_bundle["num_samples"],
        )
    raise ValueError(f"Unsupported policy: {policy}")


def write_location_outputs(
    *,
    loc_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    policy: str,
    elapsed_s: float,
    html: bool,
) -> None:
    rows = sorted(rows, key=lambda row: int(row.get("candidate_rank", 0) or 0))
    write_csv(loc_dir / "prediction_metrics.csv", rows)
    summary = summarize_rows(rows)
    write_csv(loc_dir / "summary.csv", [summary])
    payload = {
        "metadata": {
            "experiment_mode": "batch_prediction_eval",
            "policy": policy,
            "location": args.location,
            "target_per_location": args.target_per_location,
            "history_sec": args.history_sec,
            "future_sec": args.future_sec,
            "dt": args.dt,
            "miss_threshold": args.miss_threshold,
            "max_guesses": args.max_guesses,
            "device": args.device,
            "elapsed_s": elapsed_s,
        },
        "summary": summary,
        "rows": list(rows),
        "predictions": list(predictions),
    }
    (loc_dir / "prediction_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if html and predictions:
        save_prediction_html(
            predictions=predictions,
            rows=rows,
            summary=summary,
            path=loc_dir / "prediction_visualization.html",
        )


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(payload.get("predictions", []))


def upsert_prediction(
    predictions: Sequence[Mapping[str, Any]], prediction: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    out = [
        dict(item)
        for item in predictions
        if item.get("run_id") != prediction.get("run_id")
    ]
    out.append(dict(prediction))
    return out


def completed_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("status") == "completed")


def location_summary(
    policy: str, location: str, rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    summary = summarize_rows(rows)
    return {"policy": policy, "location": location, **summary}


if __name__ == "__main__":
    main()
