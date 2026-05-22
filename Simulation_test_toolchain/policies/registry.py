from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BasePolicy
from .risk_idm import RiskIDMPolicy


def build_policy(
    name: str,
    dt: float,
    params: Optional[Dict[str, Any]] = None,
    checkpoints=None,
) -> Optional[BasePolicy]:
    params = dict(params or {})
    name = name.lower()
    if name == "ground_truth":
        return None
    if name == "risk_idm":
        return RiskIDMPolicy(dt=dt, **params)
    if name == "asaprl":
        from .asaprl import ASAPRLPolicy

        params.setdefault("ckpt_path", getattr(checkpoints, "asaprl_ckpt_path", None))
        params.setdefault("device", getattr(checkpoints, "asaprl_device", None))
        return ASAPRLPolicy(dt=dt, **params)
    if name == "diffuser":
        from .diffuser_policy import DiffuserPolicy

        params.setdefault("ckpt_path", getattr(checkpoints, "diffuser_ckpt_path", None))
        return DiffuserPolicy(dt=dt, **params)
    if name == "qcnet":
        from .qcnet_policy import QcnetPolicy

        params.setdefault("ckpt_path", getattr(checkpoints, "qcnet_ckpt_path", None))
        params.setdefault("repo_path", getattr(checkpoints, "qcnet_repo_path", None))
        return QcnetPolicy(dt=dt, **params)
    raise ValueError(f"Unknown policy: {name}")
