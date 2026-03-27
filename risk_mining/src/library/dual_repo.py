"""
Dual library system for risk mining pipeline.

Stores:
1. Risk Element Library - Human-readable JSON files (individual risk elements)
2. Risk Event Library - Complete episodes with SSTG objects (Pickle format)
"""

import json
import pickle
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.slicer import Episode
from ..core.scene_graph import Node, Edge


class RiskElementLibrary:
    """
    Library for storing individual risk elements as human-readable JSON.

    Each element is a JSON file with metadata and summary statistics.
    """

    def __init__(self, library_dir: Path):
        """
        Initialize the risk element library.

        Args:
            library_dir: Directory to store element JSON files
        """
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.library_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load or create the library index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "elements": [],
                "count": 0,
            }

    def _save_index(self) -> None:
        """Save the library index."""
        self.index["last_updated"] = datetime.now().isoformat()
        self.index["count"] = len(self.index["elements"])
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_element(self, episode: Episode, element_id: Optional[str] = None) -> str:
        """
        Add a risk element to the library.

        Args:
            episode: The episode to extract element from
            element_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The element ID
        """
        if element_id is None:
            element_id = f"element_{uuid.uuid4().hex[:8]}"

        # Create element data (human-readable summary)
        element_data = {
            "element_id": element_id,
            "created_at": datetime.now().isoformat(),
            "source": {
                "scene_name": episode.metadata.get("scene_name", "unknown"),
                "env_name": episode.metadata.get("env_name", "unknown"),
            },
            "episode": {
                "t_start": episode.t_start,
                "t_peak": episode.t_peak,
                "t_end": episode.t_end,
                "duration_timesteps": episode.duration_timesteps,
                "episode_type": episode.episode_type.value,
            },
            "agents": {
                "involved_agents": episode.involved_agents,
                "agent_count": len(episode.involved_agents),
            },
            "risk": {
                "risk_score": episode.risk_score,
            },
            "sstg_summary": episode.sstg.get_summary() if episode.sstg else None,
        }

        # Save element file
        element_path = self.library_dir / f"{element_id}.json"
        with open(element_path, 'w') as f:
            json.dump(element_data, f, indent=2)

        # Update index
        self.index["elements"].append({
            "element_id": element_id,
            "created_at": element_data["created_at"],
            "scene_name": episode.metadata.get("scene_name", "unknown"),
            "episode_type": episode.episode_type.value,
            "risk_score": episode.risk_score,
        })
        self._save_index()

        return element_id

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get an element by ID."""
        element_path = self.library_dir / f"{element_id}.json"
        if element_path.exists():
            with open(element_path, 'r') as f:
                return json.load(f)
        return None

    def list_elements(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all elements in the library."""
        elements = self.index["elements"]
        if limit is not None:
            return elements[:limit]
        return elements

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            "count": self.index["count"],
            "created_at": self.index["created_at"],
            "last_updated": self.index["last_updated"],
        }


class RiskEventLibrary:
    """
    Library for storing complete episodes with SSTG objects.

    Events are stored as pickle files for efficient serialization of complex objects.
    """

    def __init__(self, library_dir: Path):
        """
        Initialize the risk event library.

        Args:
            library_dir: Directory to store event pickle files
        """
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.library_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load or create the library index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "events": [],
                "count": 0,
            }

    def _save_index(self) -> None:
        """Save the library index."""
        self.index["last_updated"] = datetime.now().isoformat()
        self.index["count"] = len(self.index["events"])
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_event(self, episode: Episode, event_id: Optional[str] = None) -> str:
        """
        Add a risk event to the library.

        Args:
            episode: The episode to store
            event_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The event ID
        """
        if event_id is None:
            event_id = f"event_{uuid.uuid4().hex[:8]}"

        # Save event file (pickle)
        event_path = self.library_dir / f"{event_id}.pkl"
        with open(event_path, 'wb') as f:
            pickle.dump(episode, f)

        # Update index
        self.index["events"].append({
            "event_id": event_id,
            "created_at": datetime.now().isoformat(),
            "scene_name": episode.metadata.get("scene_name", "unknown"),
            "episode_type": episode.episode_type.value,
            "risk_score": episode.risk_score,
            "t_peak": episode.t_peak,
            "agent_count": len(episode.involved_agents),
        })
        self._save_index()

        return event_id

    def get_event(self, event_id: str) -> Optional[Episode]:
        """Get an event by ID."""
        event_path = self.library_dir / f"{event_id}.pkl"
        if event_path.exists():
            with open(event_path, 'rb') as f:
                return pickle.load(f)
        return None

    def list_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all events in the library."""
        events = self.index["events"]
        if limit is not None:
            return events[:limit]
        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            "count": self.index["count"],
            "created_at": self.index["created_at"],
            "last_updated": self.index["last_updated"],
        }


class DualLibrary:
    """
    Combined interface for both risk element and risk event libraries.

    Provides unified access to both libraries.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the dual library system.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = output_dir

        # Create subdirectories
        self.elements_dir = output_dir / "libraries" / "risk_elements"
        self.events_dir = output_dir / "libraries" / "risk_events"

        # Initialize libraries
        self.element_lib = RiskElementLibrary(self.elements_dir)
        self.event_lib = RiskEventLibrary(self.events_dir)

    def add_episode(
        self,
        episode: Episode,
        episode_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Add an episode to both libraries.

        Args:
            episode: The episode to add
            episode_id: Optional custom ID (used for both libraries with suffix)

        Returns:
            (element_id, event_id) tuple
        """
        if episode_id is None:
            episode_id = uuid.uuid4().hex[:8]

        element_id = self.element_lib.add_element(episode, f"element_{episode_id}")
        event_id = self.event_lib.add_event(episode, f"event_{episode_id}")

        return element_id, event_id

    def save_summary(self, additional_stats: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save a summary of the library contents.

        Args:
            additional_stats: Optional additional statistics to include

        Returns:
            Path to the summary file
        """
        summary = {
            "generated_at": datetime.now().isoformat(),
            "risk_element_library": self.element_lib.get_stats(),
            "risk_event_library": self.event_lib.get_stats(),
        }

        if additional_stats:
            summary["additional_stats"] = additional_stats

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_{timestamp}.json"

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def get_element_library(self) -> RiskElementLibrary:
        """Get the risk element library."""
        return self.element_lib

    def get_event_library(self) -> RiskEventLibrary:
        """Get the risk event library."""
        return self.event_lib
