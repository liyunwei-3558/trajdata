"""
JSON-backed dual library for mined risk scenarios.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.slicer import Episode


class RiskElementLibrary:
    """Store lightweight summaries for retrieval and manual browsing."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.library_dir / "index.json"
        self.index = self._load_index("elements")

    def add_element(
        self,
        episode: Episode,
        event_id: str,
        element_id: Optional[str] = None,
    ) -> str:
        element_id = element_id or f"element_{uuid.uuid4().hex[:8]}"
        payload = {
            "element_id": element_id,
            "event_id": event_id,
            "scene_id": episode.scene_id,
            "scene_name": episode.scene_name,
            "env_name": episode.env_name,
            "ego_agent_id": episode.ego_agent_id,
            "episode_type": episode.episode_type.value,
            "risk_score": episode.risk_score,
            "timeframe": episode.semantic_timesteps,
            "involved_agents": episode.involved_agents,
            "applied_rules": episode.rule_trace,
            "sstg_summary": episode.sstg.get_summary() if episode.sstg is not None else None,
            "created_at": datetime.now().isoformat(),
        }
        self._write_json(self.library_dir / f"{element_id}.json", payload)
        self.index["elements"].append(
            {
                "element_id": element_id,
                "event_id": event_id,
                "scene_name": episode.scene_name,
                "risk_score": episode.risk_score,
                "created_at": payload["created_at"],
            }
        )
        self._save_index("elements")
        return element_id

    def get_stats(self) -> Dict[str, Any]:
        return self._stats()

    def _load_index(self, key: str) -> Dict[str, Any]:
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            key: [],
            "count": 0,
        }

    def _save_index(self, key: str) -> None:
        self.index["last_updated"] = datetime.now().isoformat()
        self.index["count"] = len(self.index[key])
        self._write_json(self.index_path, self.index)

    def _stats(self) -> Dict[str, Any]:
        return {
            "count": self.index["count"],
            "created_at": self.index["created_at"],
            "last_updated": self.index["last_updated"],
        }

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


class RiskEventLibrary:
    """Store complete JSON-serialized events."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.library_dir / "index.json"
        self.index = self._load_index("events")

    def add_event(self, episode: Episode, event_id: Optional[str] = None) -> str:
        event_id = event_id or f"event_{uuid.uuid4().hex[:8]}"
        payload = {
            "event_id": event_id,
            "source_scene": {
                "scene_id": episode.scene_id,
                "scene_name": episode.scene_name,
                "env_name": episode.env_name,
            },
            "episode_window": episode.semantic_timesteps,
            "ego_agent_id": episode.ego_agent_id,
            "involved_agents": episode.involved_agents,
            "episode_type": episode.episode_type.value,
            "risk_score": episode.risk_score,
            "applied_rules": episode.rule_trace,
            "metadata": episode.metadata,
            "sstg": episode.sstg.to_dict() if episode.sstg is not None else None,
            "created_at": datetime.now().isoformat(),
        }
        self._write_json(self.library_dir / f"{event_id}.json", payload)
        self.index["events"].append(
            {
                "event_id": event_id,
                "scene_name": episode.scene_name,
                "risk_score": episode.risk_score,
                "created_at": payload["created_at"],
            }
        )
        self._save_index("events")
        return event_id

    def get_stats(self) -> Dict[str, Any]:
        return self._stats()

    def _load_index(self, key: str) -> Dict[str, Any]:
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            key: [],
            "count": 0,
        }

    def _save_index(self, key: str) -> None:
        self.index["last_updated"] = datetime.now().isoformat()
        self.index["count"] = len(self.index[key])
        self._write_json(self.index_path, self.index)

    def _stats(self) -> Dict[str, Any]:
        return {
            "count": self.index["count"],
            "created_at": self.index["created_at"],
            "last_updated": self.index["last_updated"],
        }

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


class DualLibrary:
    """Combined interface for risk element and risk event storage."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.elements_dir = output_dir / "libraries" / "risk_elements"
        self.events_dir = output_dir / "libraries" / "risk_events"
        self.element_lib = RiskElementLibrary(self.elements_dir)
        self.event_lib = RiskEventLibrary(self.events_dir)

    def add_episode(self, episode: Episode, episode_id: Optional[str] = None) -> Tuple[str, str]:
        episode_suffix = episode_id or uuid.uuid4().hex[:8]
        event_id = self.event_lib.add_event(episode, event_id=f"event_{episode_suffix}")
        element_id = self.element_lib.add_element(
            episode,
            event_id=event_id,
            element_id=f"element_{episode_suffix}",
        )
        return element_id, event_id

    def save_summary(self, additional_stats: Optional[Dict[str, Any]] = None) -> Path:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "risk_element_library": self.element_lib.get_stats(),
            "risk_event_library": self.event_lib.get_stats(),
        }
        if additional_stats:
            payload["additional_stats"] = additional_stats

        summary_path = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return summary_path

    def get_element_library(self) -> RiskElementLibrary:
        return self.element_lib

    def get_event_library(self) -> RiskEventLibrary:
        return self.event_lib
