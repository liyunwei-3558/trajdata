from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import ToolchainConfig
from .ego_selection import select_ego_from_scene
from .records import AgentFrame, SimulationResult
from .state_utils import (
    get_agent_world_pose,
    get_ground_truth_next_xyh,
    xyh_to_state_array,
)
from Simulation_test_toolchain.policies.registry import build_policy


def run_simulation(cfg: ToolchainConfig) -> SimulationResult:
    from trajdata import AgentType, UnifiedDataset
    from trajdata.simulation import SimulationScene

    print(
        f"[toolchain] loading {cfg.dataset.name}-{cfg.dataset.location} "
        f"scene_index={cfg.scenario.scene_index}",
        flush=True,
    )

    if cfg.simulation.mode == "open_loop":
        ego_policy_name = "ground_truth"
    else:
        ego_policy_name = cfg.policies.ego_policy

    use_raster_map = bool(cfg.policies.ego.get("require_raster_map", False))
    dataset = UnifiedDataset(
        desired_data=[f"sind-{cfg.dataset.location}"],
        data_dirs={"sind": str(Path(cfg.dataset.data_dir))},
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(
            lambda: float(cfg.simulation.neighbor_radius)
        ),
        desired_dt=cfg.dataset.desired_dt,
        centric="agent",
        history_sec=(cfg.simulation.history_sec, cfg.simulation.history_sec),
        future_sec=(cfg.simulation.future_sec, cfg.simulation.future_sec),
        incl_raster_map=use_raster_map,
        raster_map_params=(
            {
                "px_per_m": 12,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
                "use_lanelet2_maps": cfg.dataset.use_lanelet2_maps,
            }
            if use_raster_map
            else None
        ),
        incl_vector_map=False,
        vector_map_params={
            "collate": False,
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        },
        verbose=True,
        num_workers=0,
    )
    scenes = list(dataset.scenes())
    print(
        f"[toolchain] dataset ready: scenes={len(scenes)}, samples={len(dataset)}",
        flush=True,
    )

    if cfg.scenario.scene_name:
        scene = next(
            (candidate for candidate in scenes if candidate.name == cfg.scenario.scene_name),
            None,
        )
        if scene is None:
            available = ", ".join(candidate.name for candidate in scenes[:5])
            raise ValueError(
                f"scenario.scene_name={cfg.scenario.scene_name!r} not found. "
                f"First available scenes: {available}"
            )
    else:
        scene = scenes[cfg.scenario.scene_index]
    ego_agent, _ = select_ego_from_scene(
        scene,
        strategy=cfg.scenario.ego_selection_strategy,
        ego_agent_name=cfg.scenario.ego_agent_name,
    )
    requested_ego_name = ego_agent.name
    init_timestep = cfg.scenario.init_timestep
    if init_timestep is None:
        init_timestep = min(
            int(cfg.simulation.history_sec / cfg.dataset.desired_dt) + 10,
            max(0, scene.length_timesteps // 2),
        )
    windowed_scene = _make_windowed_scene(scene, init_timestep, cfg)

    sim_scene = SimulationScene(
        env_name=f"sind_{cfg.dataset.location}_test",
        scene_name=f"scene_{cfg.scenario.scene_index:03d}_test",
        scene=windowed_scene,
        dataset=dataset,
        init_timestep=init_timestep,
        freeze_agents=True,
    )
    _ensure_sim_cache_defaults(sim_scene)
    obs = sim_scene.reset()
    ego_idx = _find_agent_idx(obs.agent_name, ego_agent.name)
    if ego_idx is None:
        ego_idx = 0
        active_ego_name = str(obs.agent_name[ego_idx])
        print(
            f"[toolchain] requested ego '{requested_ego_name}' not present; "
            f"falling back to present agent '{active_ego_name}' at timestep {init_timestep}",
            flush=True,
        )
    else:
        active_ego_name = ego_agent.name
    print(
        f"[toolchain] simulation reset: scene={scene.name}, init_timestep={init_timestep}, "
        f"agents={len(obs.agent_name)}, ego={active_ego_name}",
        flush=True,
    )

    ego_policy = build_policy(
        ego_policy_name,
        dt=cfg.dataset.desired_dt,
        params=cfg.policies.ego,
        checkpoints=cfg.checkpoints,
    )
    non_ego_policy = build_policy(
        cfg.policies.non_ego_policy
        if cfg.simulation.mode == "multi_agent_closed_loop"
        else "ground_truth",
        dt=cfg.dataset.desired_dt,
        params=cfg.policies.non_ego,
        checkpoints=cfg.checkpoints,
    )
    if ego_policy is not None:
        ego_policy.reset(obs, ego_idx)
    if non_ego_policy is not None:
        non_ego_policy.reset(obs, ego_idx)

    frames: List[AgentFrame] = []
    _append_frames(frames, obs, ego_idx, init_timestep, ego_policy_name, {})

    max_available = max(0, windowed_scene.length_timesteps - init_timestep - 1)
    num_steps = min(cfg.scenario.num_steps, max_available)
    print(
        f"[toolchain] stepping: requested={cfg.scenario.num_steps}, actual={num_steps}, "
        f"ego={active_ego_name}, policy={ego_policy_name}",
        flush=True,
    )
    for offset in range(1, num_steps + 1):
        current_ego_idx = _find_agent_idx(obs.agent_name, active_ego_name)
        if current_ego_idx is None:
            break

        actions = {}
        ego_command = {}
        for idx, agent_name in enumerate(obs.agent_name):
            if idx == current_ego_idx and ego_policy is not None:
                action = ego_policy.get_action(obs, idx)
                xyh = action.xyh
                ego_command = action.command
                if not np.isfinite(xyh).all():
                    xyh = get_ground_truth_next_xyh(obs, idx)
                    ego_command = {
                        **ego_command,
                        "fallback": "ground_truth",
                        "fallback_reason": "policy returned non-finite xyh",
                    }
            elif idx != current_ego_idx and non_ego_policy is not None:
                action = non_ego_policy.get_action(obs, idx)
                xyh = action.xyh
                if not np.isfinite(xyh).all():
                    xyh = get_ground_truth_next_xyh(obs, idx)
            else:
                xyh = get_ground_truth_next_xyh(obs, idx)
            actions[agent_name] = xyh_to_state_array(xyh)

        obs = sim_scene.step(actions)
        next_ego_idx = _find_agent_idx(obs.agent_name, active_ego_name)
        if next_ego_idx is None:
            break
        _append_frames(
            frames,
            obs,
            next_ego_idx,
            init_timestep + offset,
            ego_policy_name,
            ego_command,
        )

    print(f"[toolchain] collected frames={len(frames)}", flush=True)

    metadata = {
        "dataset": "sind",
        "location": cfg.dataset.location,
        "scene_index": cfg.scenario.scene_index,
        "scene_name": scene.name,
        "init_timestep": init_timestep,
        "window_end_timestep": windowed_scene.length_timesteps - 1,
        "num_steps": num_steps,
        "ego_agent": requested_ego_name,
        "active_ego_agent": active_ego_name,
        "ego_policy": ego_policy_name,
        "non_ego_policy": cfg.policies.non_ego_policy,
        "mode": cfg.simulation.mode,
        "cache_path": str(dataset.cache_path),
        "map_name": f"sind:{cfg.dataset.location}",
    }
    if cfg.scenario.semantic_label_id:
        metadata["semantic_label_id"] = cfg.scenario.semantic_label_id
    if cfg.scenario.semantic_label:
        metadata["semantic_label"] = cfg.scenario.semantic_label
    if getattr(obs, "map_names", None) is not None and len(obs.map_names) > 0:
        metadata["map_name"] = obs.map_names[next_ego_idx]
    return SimulationResult(metadata=metadata, frames=frames)


def _make_windowed_scene(scene, init_timestep: int, cfg: ToolchainConfig):
    from trajdata.data_structures.scene_metadata import Scene

    future_steps = int(np.ceil(cfg.simulation.future_sec / cfg.dataset.desired_dt))
    window_end = min(
        scene.length_timesteps,
        init_timestep + cfg.scenario.num_steps + future_steps + 2,
    )
    agent_presence = scene.agent_presence[:window_end]
    agent_names = {
        agent.name
        for present_agents in agent_presence
        for agent in present_agents
    }
    agents = [agent for agent in scene.agents if agent.name in agent_names]

    return Scene(
        env_metadata=scene.env_metadata,
        name=scene.name,
        location=scene.location,
        data_split=scene.data_split,
        length_timesteps=window_end,
        raw_data_idx=scene.raw_data_idx,
        data_access_info=scene.data_access_info,
        description=scene.description,
        agents=agents,
        agent_presence=agent_presence,
    )


def _find_agent_idx(agent_names, target: str) -> Optional[int]:
    for idx, name in enumerate(agent_names):
        if name == target:
            return idx
    return None


def _ensure_sim_cache_defaults(sim_scene) -> None:
    # SimulationDataFrameCache.reset() does not initialize these optional
    # transform attributes, but append_state()->get_value() expects them.
    for attr in ("_transf_mean", "_transf_rotmat"):
        if not hasattr(sim_scene.cache, attr):
            setattr(sim_scene.cache, attr, None)


def _append_frames(
    frames: List[AgentFrame],
    obs,
    ego_idx: int,
    scene_timestep: int,
    ego_policy_name: str,
    ego_command: Dict,
) -> None:
    for idx, name in enumerate(obs.agent_name):
        pose = get_agent_world_pose(obs, idx)
        speed = float(np.linalg.norm(pose["velocity"]))
        extent = pose["extent"]
        agent_type = "UNKNOWN"
        if getattr(obs, "agent_type", None) is not None:
            agent_type = str(obs.agent_type[idx].item())
        is_ego = idx == ego_idx
        frames.append(
            AgentFrame(
                timestep=int(scene_timestep),
                agent_name=str(name),
                agent_type=agent_type,
                x=float(pose["position"][0]),
                y=float(pose["position"][1]),
                heading=float(pose["heading"]),
                speed=speed,
                length=float(extent[0]) if len(extent) > 0 else 4.2,
                width=float(extent[1]) if len(extent) > 1 else 1.8,
                is_ego=is_ego,
                policy=ego_policy_name if is_ego else "ground_truth",
                command=ego_command if is_ego else {},
            )
        )
