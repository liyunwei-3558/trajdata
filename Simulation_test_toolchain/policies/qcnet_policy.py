from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from Simulation_test_toolchain.core.state_utils import get_agent_world_pose

from .base import BasePolicy, PolicyAction, PolicyState
from .qcnet_adapter import (
    build_qcnet_action,
    build_qcnet_sample_spec,
    spec_to_heterodata,
)


class QcnetPolicy(BasePolicy):
    policy_name = "qcnet"

    def __init__(
        self,
        dt: float = 0.1,
        ckpt_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        device: Optional[str] = None,
        map_radius: float = 150.0,
        step_index: int = 0,
        strict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)
        if ckpt_path is None:
            raise ValueError(
                "QCNet requires checkpoints.qcnet_ckpt_path or policies.<role>.ckpt_path."
            )
        ckpt = Path(ckpt_path).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"QCNet checkpoint not found: {ckpt}")

        repo = repo_path or os.environ.get("QCNET_REPO_PATH")
        if repo is None:
            raise ValueError(
                "QCNet requires policies.<role>.repo_path, checkpoints.qcnet_repo_path, "
                "or QCNET_REPO_PATH."
            )
        repo_dir = Path(repo).expanduser()
        if not repo_dir.exists():
            raise FileNotFoundError(f"QCNet repo_path not found: {repo_dir}")

        try:
            import torch
        except Exception as exc:
            raise ImportError(
                "QCNet policy requires torch in the active conda environment."
            ) from exc

        self.torch = torch
        self.ckpt_path = str(ckpt)
        self.repo_path = str(repo_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.map_radius = float(map_radius)
        self.step_index = int(step_index)
        self.strict = bool(strict)
        self.model = self._load_model().to(self.device)
        self.model.eval()

        if (
            int(getattr(self.model, "input_dim", 2)) != 2
            or int(getattr(self.model, "output_dim", 2)) != 2
        ):
            raise ValueError(
                "QCNet policy wrapper currently supports 2D input_dim/output_dim checkpoints."
            )

    def _load_model(self):
        repo_path = str(Path(self.repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from predictors import QCNet
        except Exception as exc:
            raise ImportError(
                "Could not import QCNet from repo_path. Activate a QCNet-compatible env "
                "or install QCNet dependencies such as torch_geometric and torch_cluster."
            ) from exc

        try:
            return QCNet.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                strict=self.strict,
                map_location="cpu",
            )
        except TypeError:
            return QCNet.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                strict=self.strict,
            )

    def reset(self, obs, ego_idx: int = 0) -> PolicyState:
        self.state = PolicyState(
            agent_name=str(obs.agent_name[ego_idx]),
            dt=self.dt,
            initialized=True,
        )
        return self.state

    def get_action(self, obs, ego_idx: int = 0) -> PolicyAction:
        if self.state is None:
            self.reset(obs, ego_idx)

        spec = build_qcnet_sample_spec(
            obs,
            ego_idx,
            map_radius=self.map_radius,
            hist_steps=int(getattr(self.model, "num_historical_steps")),
            fut_steps=int(getattr(self.model, "num_future_steps")),
            dt=self.dt,
        )
        data = spec_to_heterodata(spec).to(self.device)

        with self.torch.no_grad():
            pred = self.model(data)

        xyh, command = build_qcnet_action(
            pred, spec, ego_idx=0, step_index=self.step_index
        )
        if not np.isfinite(xyh).all():
            pose = get_agent_world_pose(obs, ego_idx)
            xyh = np.array(
                [pose["position"][0], pose["position"][1], pose["heading"]], dtype=float
            )
            command = {
                **command,
                "fallback": "hold_current_pose",
                "fallback_reason": "QCNet returned non-finite xyh",
            }

        self.last_command = command.copy()
        return PolicyAction(xyh=xyh, command=self.last_command.copy())
