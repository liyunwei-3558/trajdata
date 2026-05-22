from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from Simulation_test_toolchain.policies.qcnet_adapter import (
    QcnetPose,
    build_qcnet_action,
    build_qcnet_sample_spec,
)
from Simulation_test_toolchain.policies.qcnet_policy import (
    _apply_initial_speed_override,
)


def _seq(position, heading, velocity=None):
    return SimpleNamespace(
        position=np.asarray(position, dtype=np.float32),
        heading=np.asarray(heading, dtype=np.float32),
        velocity=np.asarray(
            velocity if velocity is not None else np.zeros_like(position),
            dtype=np.float32,
        ),
    )


class _AgentSeqs:
    def __init__(self, *seqs):
        self._seqs = list(seqs)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            agent_idx, time_idx, cols = item
            return self._seqs[agent_idx].position[time_idx, cols]
        return self._seqs[item]

    def __len__(self):
        return len(self._seqs)


def test_build_qcnet_sample_spec_and_action():
    lane = SimpleNamespace(
        id="lane_1",
        center=SimpleNamespace(
            points=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        ),
        prev_lanes=set(),
        next_lanes=set(),
        adj_lanes_left=set(),
        adj_lanes_right=set(),
    )
    vector_map = SimpleNamespace(
        get_lanes_within=lambda xyz, dist: [lane],
        get_areas_within=lambda xy, elem_type, dist: [],
    )

    ego_hist = _seq(
        position=[[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
        heading=[0.0, 0.0, 0.0],
        velocity=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    ego_fut = _seq(
        position=[[1.0, 0.0], [2.0, 0.0]],
        heading=[0.0, 0.0],
        velocity=[[1.0, 0.0], [1.0, 0.0]],
    )
    neigh_hist = _seq(
        position=[[-1.0, -1.0], [-0.5, -1.0], [0.0, -1.0]],
        heading=[0.0, 0.0, 0.0],
        velocity=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    neigh_fut = _seq(
        position=[[0.5, -1.0], [1.0, -1.0]],
        heading=[0.0, 0.0],
        velocity=[[1.0, 0.0], [1.0, 0.0]],
    )

    obs = SimpleNamespace(
        dt=np.asarray([0.1], dtype=np.float32),
        agent_name=["ego", "neighbor_0"],
        agent_type=np.asarray([1, 1], dtype=np.int64),
        num_neigh=np.asarray([1], dtype=np.int64),
        agent_hist=_AgentSeqs(ego_hist),
        agent_hist_len=np.asarray([3], dtype=np.int64),
        agent_fut=_AgentSeqs(ego_fut),
        agent_fut_len=np.asarray([2], dtype=np.int64),
        neigh_hist=np.asarray([[neigh_hist]], dtype=object),
        neigh_hist_len=np.asarray([[3]], dtype=np.int64),
        neigh_fut=np.asarray([[neigh_fut]], dtype=object),
        neigh_fut_len=np.asarray([[2]], dtype=np.int64),
        neigh_types=np.asarray([[1]], dtype=np.int64),
        agents_from_world_tf=np.asarray(
            [np.eye(3, dtype=np.float32)], dtype=np.float32
        ),
        vector_maps=[vector_map],
        history_pad_dir="BEFORE",
    )

    spec = build_qcnet_sample_spec(
        obs, 0, hist_steps=3, fut_steps=2, dt=0.1, map_radius=50.0
    )

    assert spec["agent"]["num_nodes"] == 2
    assert spec["map_polygon"]["num_nodes"] == 1
    assert spec["agent"]["position"].shape == (2, 5, 2)
    assert spec["agent"]["predict_mask"][0, 3:].all()
    assert spec["agent"]["valid_mask"].shape == (2, 5)

    spec["ego_pose"] = QcnetPose(
        position=np.asarray([10.0, 5.0], dtype=np.float32),
        heading=np.pi / 2,
        rotation=np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
        world_from_agent=np.eye(3, dtype=np.float32),
        agent_from_world=np.eye(3, dtype=np.float32),
    )
    pred = {
        "pi": np.asarray([[0.0, 3.0]], dtype=np.float32),
        "loc_refine_pos": np.zeros((1, 2, 2, 2), dtype=np.float32),
        "loc_refine_head": np.zeros((1, 2, 2, 1), dtype=np.float32),
    }
    pred["loc_refine_pos"][0, 1, 0] = np.asarray([2.0, 0.0], dtype=np.float32)

    xyh, command = build_qcnet_action(pred, spec, ego_idx=0, step_index=0)

    assert command["mode_index"] == 1
    assert np.isfinite(xyh).all()
    np.testing.assert_allclose(
        xyh[:2], np.asarray([10.0, 7.0], dtype=np.float32), atol=1e-6
    )


def test_apply_initial_speed_override_scales_first_step_distance():
    world_xy = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=float)
    scaled, scale = _apply_initial_speed_override(
        world_xy,
        np.asarray([0.0, 0.0], dtype=float),
        target_speed_mps=2.0,
        dt=0.1,
    )

    assert scale == 0.2
    np.testing.assert_allclose(scaled[0], np.asarray([0.2, 0.0]), atol=1e-6)
    np.testing.assert_allclose(scaled[1], np.asarray([0.4, 0.0]), atol=1e-6)
