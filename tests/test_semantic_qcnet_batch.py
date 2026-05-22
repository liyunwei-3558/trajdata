from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from Simulation_test_toolchain.run_semantic_qcnet_batch import (
    DEFAULT_MPRTTC_EXTRA_LABEL_PATH,
    DEFAULT_QCNET_CKPT,
    DEFAULT_QCNET_REPO,
    NarrowCandidate,
    SemanticPlan,
    build_plans,
    _compute_qcnet_command_metrics,
    _label_sort_key,
    _load_labels,
    _make_config,
    _plans_from_narrow_candidate,
    _summary_row,
)
from Simulation_test_toolchain.core.records import AgentFrame, SimulationResult


def _label(
    scenario_id: str,
    *,
    scenario_type: str = "mprttc",
    location: str = "tj",
    scene_name: str = "tj_1",
    ego_id: str = "7",
    start_frame: int = 20,
    end_frame: int = 40,
    semantics: dict | None = None,
) -> dict:
    if semantics is None:
        semantics = {"type": scenario_type}
    else:
        semantics = {"type": scenario_type, **semantics}
    return {
        "scenario_id": scenario_id,
        "location": location,
        "source_scene_name": scene_name,
        "agents": {"ego_id": ego_id},
        "time_window": {"start_frame": start_frame, "end_frame": end_frame},
        "semantics": semantics,
    }


def test_load_labels_deduplicates_mprttc_sources(tmp_path: Path):
    first_path = tmp_path / "scenarios.json"
    second_path = tmp_path / "extra.json"
    duplicate = _label("dup", semantics={"min_mprttc": 1})
    unique = _label("unique", location="cc", semantics={"min_mprttc": 0})
    first_path.write_text(
        '{"scenarios": [%s]}' % json.dumps(duplicate),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps([duplicate, unique]),
        encoding="utf-8",
    )

    labels = _load_labels([first_path, second_path], "mprttc")

    assert [label["scenario_id"] for label in labels] == ["dup", "unique"]


def test_visual_shielding_sort_is_length_then_scene_time_ego():
    long_late = _label(
        "a",
        scenario_type="visual_shielding",
        scene_name="cc_2",
        ego_id="9",
        start_frame=100,
        semantics={"shielding_frame_length": 30},
    )
    long_early_scene = _label(
        "b",
        scenario_type="visual_shielding",
        scene_name="cc_1",
        ego_id="9",
        start_frame=200,
        semantics={"shielding_frame_length": 30},
    )
    short = _label(
        "c",
        scenario_type="visual_shielding",
        scene_name="cc_0",
        start_frame=1,
        semantics={"shielding_frame_length": 5},
    )

    ordered = sorted(
        [short, long_late, long_early_scene],
        key=lambda label: _label_sort_key(label, "visual_shielding"),
    )

    assert [label["scenario_id"] for label in ordered] == ["b", "a", "c"]


def test_visual_shielding_global_target_total_allows_unmet_requirements(
    monkeypatch, tmp_path: Path
):
    labels = [
        _label(
            "high_a",
            scenario_type="visual_shielding",
            location="tj",
            scene_name="tj_1",
            ego_id="9",
            semantics={"shielding_frame_length": 60},
        ),
        _label(
            "high_b",
            scenario_type="visual_shielding",
            location="tj",
            scene_name="tj_2",
            ego_id="8",
            semantics={"shielding_frame_length": 50},
        ),
        _label(
            "low_c",
            scenario_type="visual_shielding",
            location="cqIR",
            scene_name="cqIR_1",
            ego_id="7",
            semantics={"shielding_frame_length": 40},
        ),
    ]

    monkeypatch.setattr(
        "Simulation_test_toolchain.run_semantic_qcnet_batch._load_labels",
        lambda label_paths, scenario_type: labels,
    )
    monkeypatch.setattr(
        "Simulation_test_toolchain.run_semantic_qcnet_batch._scene_index_map",
        lambda data_dir, location: {f"{location}_1": 1, f"{location}_2": 2},
    )
    monkeypatch.setattr(
        "Simulation_test_toolchain.run_semantic_qcnet_batch._load_scene_tracks",
        lambda data_dir, location, scene_id: {},
    )
    monkeypatch.setattr(
        "Simulation_test_toolchain.run_semantic_qcnet_batch._scene_length",
        lambda scene_tracks: 999,
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("window gating should be bypassed")

    monkeypatch.setattr(
        "Simulation_test_toolchain.run_semantic_qcnet_batch._has_valid_window",
        _should_not_run,
    )
    monkeypatch.setattr(
        "Simulation_test_toolchain.run_semantic_qcnet_batch._has_history_window",
        _should_not_run,
    )

    plans_by_intersection = build_plans(
        scenario_type="visual_shielding",
        label_paths=[tmp_path / "labels.json"],
        data_dir=tmp_path,
        intersections=("tj", "cqIR"),
        target_per_intersection=200,
        target_total=2,
        num_steps=150,
        history_steps=20,
        future_steps=60,
        candidate_buffer_factor=2.0,
        strict_min_path_length=8.0,
        strict_min_displacement=5.0,
        strict_min_initial_speed=0.5,
        relaxed_min_path_length=4.0,
        relaxed_min_displacement=2.0,
        relaxed_min_initial_speed=0.2,
        max_original_initial_speed=8.5,
        require_boost_increase=True,
        boost_min_speed=9.0,
        boost_max_speed=17.0,
        boost_scale=2.0,
        progress_interval=0,
        allow_unmet_requirements=True,
    )

    assert sum(len(plans) for plans in plans_by_intersection.values()) == 2
    assert [plan.scenario_id for plan in plans_by_intersection["tj"]] == [
        "high_a",
        "high_b",
    ]
    assert plans_by_intersection["cqIR"] == []


def test_make_config_sets_qcnet_open_loop_contract(tmp_path: Path):
    plan = SemanticPlan(
        policy="qcnet",
        scenario_type="mprttc",
        intersection="xa",
        location="xasl",
        scene_name="xasl_1",
        scene_id="1",
        scene_index=3,
        run_id="run",
        scenario_id="scenario",
        agent_id="42",
        challenger_id=None,
        candidate_rank=1,
        semantic_score=0.0,
        label_start_frame=10,
        label_end_frame=20,
        init_timestep=20,
        num_steps=150,
        speed_mode="",
        pair_id="",
        original_initial_speed_mps=7.5,
        initial_speed_override_mps=9.0,
        speed_boost_factor=1.2,
        gt_path_length_150m=12.0,
        gt_displacement_150m=8.0,
        motion_filter_level="strict",
        semantic_label=_label("scenario", location="xasl"),
    )
    args = Namespace(
        dt=0.1,
        history_sec=2.0,
        future_sec=4.0,
        neighbor_radius=50.0,
        device="cuda:0",
        map_radius=150.0,
        qcnet_prediction_interval_steps=15,
        qcnet_ckpt_path=DEFAULT_QCNET_CKPT,
        qcnet_repo_path=DEFAULT_QCNET_REPO,
    )

    cfg = _make_config(plan, tmp_path / "SinD_dataset", tmp_path / "run", args)

    assert cfg.policies.ego_policy == "qcnet"
    assert cfg.policies.non_ego_policy == "ground_truth"
    assert cfg.policies.ego["prediction_interval_steps"] == 15
    assert cfg.policies.ego["execute_top1_cached_trajectory"] is True
    assert cfg.policies.ego["initial_velocity_override_mps"] == 9.0
    assert cfg.policies.ego["device"] == "cuda:0"
    assert cfg.policies.ego["map_radius"] == 150.0
    assert cfg.checkpoints.qcnet_repo_path == DEFAULT_QCNET_REPO
    assert cfg.checkpoints.qcnet_ckpt_path == DEFAULT_QCNET_CKPT
    assert cfg.dataset.desired_dt == 0.1
    assert cfg.scenario.num_steps == 150
    assert cfg.scenario.allow_ego_fallback is False


def test_qcnet_command_metrics_count_predictions_and_cached_frames():
    result = SimulationResult(
        metadata={},
        frames=[
            AgentFrame(
                timestep=0,
                agent_name="ego",
                agent_type="VEHICLE",
                x=0.0,
                y=0.0,
                heading=0.0,
                speed=0.0,
                length=4.0,
                width=2.0,
                is_ego=True,
                command={
                    "policy": "qcnet",
                    "device": "cuda:0",
                    "prediction_interval_steps": 15,
                    "used_cached_trajectory": False,
                },
            ),
            AgentFrame(
                timestep=1,
                agent_name="ego",
                agent_type="VEHICLE",
                x=1.0,
                y=0.0,
                heading=0.0,
                speed=0.0,
                length=4.0,
                width=2.0,
                is_ego=True,
                command={
                    "policy": "qcnet",
                    "device": "cuda:0",
                    "prediction_interval_steps": 15,
                    "used_cached_trajectory": True,
                },
            ),
        ],
    )

    metrics = _compute_qcnet_command_metrics(result)

    assert metrics["qcnet_prediction_count"] == 1
    assert metrics["qcnet_cached_trajectory_count"] == 1
    assert metrics["prediction_interval_steps"] == 15
    assert metrics["device"] == "cuda:0"


def test_summary_uses_location_label_column_for_intersection_groups():
    row = _summary_row(
        "cc",
        pd.DataFrame.from_records(
            [
                {
                    "status": "completed",
                    "ADE": 1.0,
                    "FDE": 2.0,
                    "collision": False,
                    "offroad": False,
                    "violation": False,
                }
            ]
        ),
    )

    assert row["location"] == "cc"


def test_default_mprttc_extra_label_path_matches_plan():
    assert (
        DEFAULT_MPRTTC_EXTRA_LABEL_PATH
        == "risk_mining/typical_risks_extract/high_risk_mprttc/output/"
        "high_risk_mprttc_non_tj_scenarios.json"
    )


def test_narrow_candidate_expands_to_original_and_boosted_runs():
    candidate = NarrowCandidate(
        label=_label(
            "narrow",
            scenario_type="narrow_feasible_area",
            location="cc",
            scene_name="cc_1",
            ego_id="42",
            start_frame=0,
            end_frame=100,
            semantics={"min_max_area": 4},
        ),
        intersection="cc",
        location="cc",
        scene_name="cc_1",
        scene_id="1",
        scene_index=0,
        agent_id="42",
        init_timestep=20,
        label_start_frame=0,
        label_end_frame=100,
        original_initial_speed_mps=5.0,
        gt_path_length_150m=20.0,
        gt_displacement_150m=15.0,
        motion_filter_level="strict",
    )

    plans = _plans_from_narrow_candidate(
        candidate,
        1,
        150,
        boost_min_speed=9.0,
        boost_max_speed=17.0,
        boost_scale=2.0,
    )

    assert [plan.speed_mode for plan in plans] == ["original_speed", "boosted_speed"]
    assert plans[0].initial_speed_override_mps == 5.0
    assert plans[1].initial_speed_override_mps is not None
    assert plans[0].pair_id == plans[1].pair_id
