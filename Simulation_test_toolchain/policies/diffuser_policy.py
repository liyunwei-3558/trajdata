from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .base import BasePolicy, PolicyAction, PolicyState


class DiffuserPolicy(BasePolicy):
    policy_name = "diffuser"

    def __init__(
        self,
        dt: float = 0.1,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)
        if ckpt_path is None:
            raise ValueError("Diffuser requires checkpoints.diffuser_ckpt_path.")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Diffuser checkpoint not found: {ckpt_path}")
        try:
            import torch
            from tbsim.models.trace import DiffuserModel
        except Exception as exc:
            raise ImportError(
                "Diffuser/TRACE requires working torch and tbsim installations."
            ) from exc

        self.torch = torch
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(DiffuserModel).to(self.device)
        self.model.eval()

    def _load_model(self, model_cls):
        config = {
            "map_encoder_model_arch": "resnet18",
            "input_image_shape": (4, 224, 224),
            "map_feature_dim": 256,
            "map_grid_feature_dim": 32,
            "diffuser_model_arch": "TemporalMapUnet",
            "horizon": 52,
            "observation_dim": 4,
            "action_dim": 2,
            "output_dim": 2,
            "cond_feature_dim": 256,
            "rasterized_map": True,
            "use_map_feat_global": False,
            "use_map_feat_grid": True,
            "hist_num_frames": 31,
            "hist_feature_dim": 128,
            "n_timesteps": 100,
            "loss_type": "l2",
            "action_weight": 1.0,
            "loss_discount": 1.0,
            "dim_mults": (2, 4, 8),
            "dynamics_type": {
                "Bicycle": {
                    "max_steer": 0.5,
                    "max_yawvel": 3,
                    "acce_bound": [-10, 8],
                    "vbound": [0, 20],
                    "wheelbase": 2.0,
                    "d_f": 1.0,
                    "d_r": 1.0,
                },
                "Unicycle": {
                    "max_steer": 0.3,
                    "max_yawvel": 0.5,
                    "acce_bound": [-5, 5],
                },
            },
            "dynamics_kwargs": {},
            "base_dim": 32,
            "diffuser_input_mode": "state_and_action",
            "use_conditioning": True,
            "cond_fill_value": -1.0,
            "diffuser_norm_info": (
                [-3.538049, 0.004175, -1.360894, 0.001894, 0.015233, 0.000562],
                [2.304491, 0.462847, 0.426683, 0.19193, 0.255089, 0.175583],
            ),
            "agent_hist_norm_info": ([0.0] * 5, [1.0] * 5),
            "neighbor_hist_norm_info": ([0.0] * 5, [1.0] * 5),
            "dt": self.dt,
        }
        model = model_cls(**config)
        checkpoint = self.torch.load(self.ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        cleaned = {
            key[len("nets.policy.") :] if key.startswith("nets.policy.") else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=False)
        return model

    def reset(self, obs=None, ego_idx: int = 0) -> PolicyState:
        self.state = PolicyState(agent_name="", dt=self.dt, initialized=True)
        return self.state

    def get_action(self, obs, ego_idx: int = 0) -> PolicyAction:
        batch = self._build_batch(obs, ego_idx)
        with self.torch.no_grad():
            pred = self.model(
                batch,
                num_samp=1,
                return_diffusion=False,
                return_guidance_losses=False,
                apply_guidance=False,
            )
        pos = pred["predictions"]["positions"][0, 0, 0].detach().cpu().numpy()
        yaw = float(pred["predictions"]["yaws"][0, 0, 0, 0].detach().cpu().item())
        from Simulation_test_toolchain.core.state_utils import get_agent_world_pose

        pose = get_agent_world_pose(obs, ego_idx)
        rot = pose["rotation"]
        world_pos = pose["position"] + pos @ rot.T
        xyh = np.array([world_pos[0], world_pos[1], pose["heading"] + yaw])
        return PolicyAction(
            xyh=xyh,
            command={"policy": self.policy_name, "pred_local_x": float(pos[0]), "pred_local_y": float(pos[1])},
        )

    def _build_batch(self, obs, agent_idx: int) -> Dict:
        from Simulation_test_toolchain.core.state_utils import to_numpy, get_agent_world_pose

        torch = self.torch
        device = self.device
        hist_num_frames = 31
        hist_positions = np.zeros((1, hist_num_frames, 2), dtype=np.float32)
        hist_yaws = np.zeros((1, hist_num_frames, 1), dtype=np.float32)
        hist_avail = np.zeros((1, hist_num_frames), dtype=bool)
        if getattr(obs, "agent_hist", None) is not None:
            hist_len = min(int(obs.agent_hist_len[agent_idx].item()), hist_num_frames)
            hist = to_numpy(obs.agent_hist[agent_idx, -hist_len:])
            hist_positions[0, -hist_len:] = hist[:, :2]
            hist_yaws[0, -hist_len:, 0] = hist[:, 4] if hist.shape[1] > 4 else 0.0
            hist_avail[0, -hist_len:] = True

        maps = np.zeros((1, 4, 224, 224), dtype=np.float32)
        if getattr(obs, "maps", None) is not None and len(obs.maps) > agent_idx:
            map_np = to_numpy(obs.maps[agent_idx])
            channels = min(4, map_np.shape[0])
            h = min(224, map_np.shape[1])
            w = min(224, map_np.shape[2])
            maps[0, :channels, :h, :w] = map_np[:channels, :h, :w]

        pose = get_agent_world_pose(obs, agent_idx)
        world_from_agent = np.eye(3, dtype=np.float32)[None]
        world_from_agent[0, :2, :2] = pose["rotation"]
        world_from_agent[0, :2, 2] = pose["position"]

        extent = np.asarray(pose["extent"], dtype=np.float32)
        return {
            "history_positions": torch.tensor(hist_positions, device=device),
            "history_yaws": torch.tensor(hist_yaws, device=device),
            "history_availabilities": torch.tensor(hist_avail, device=device),
            "history_speeds": torch.zeros(1, hist_num_frames, device=device),
            "extent": torch.tensor(extent[:2][None], device=device),
            "type": torch.ones(1, device=device),
            "curr_speed": torch.tensor([np.linalg.norm(pose["velocity"])], dtype=torch.float32, device=device),
            "image": torch.tensor(maps, device=device),
            "world_from_agent": torch.tensor(world_from_agent, device=device),
            "agent_from_world": torch.linalg.inv(torch.tensor(world_from_agent, device=device)),
            "raster_from_agent": torch.tensor(world_from_agent, device=device),
            "all_other_agents_history_positions": torch.zeros(1, 0, hist_num_frames, 2, device=device),
            "all_other_agents_history_yaws": torch.zeros(1, 0, hist_num_frames, 1, device=device),
            "all_other_agents_history_speeds": torch.zeros(1, 0, hist_num_frames, device=device),
            "all_other_agents_history_availabilities": torch.zeros(1, 0, hist_num_frames, device=device, dtype=torch.bool),
            "all_other_agents_extents": torch.zeros(1, 0, 2, device=device),
            "neigh_types": torch.zeros(1, 0, device=device),
            "scene_index": torch.zeros(1, device=device, dtype=torch.long),
        }
