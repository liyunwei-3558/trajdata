"""
Core data structures for Semantic Spatio-Temporal Graphs (SSTG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class EdgeType(str, Enum):
    """Canonical edge categories used across the pipeline."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class Node:
    """Agent state at a semantic timestamp."""

    agent_id: str
    timestamp: str
    type: str
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    position: Optional[Tuple[float, float]] = None
    heading: Optional[float] = None
    extent: Optional[Tuple[float, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(np.asarray(self.velocity, dtype=float)))

    @property
    def acceleration_magnitude(self) -> float:
        return float(np.linalg.norm(np.asarray(self.acceleration, dtype=float)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "velocity": list(self.velocity),
            "acceleration": list(self.acceleration),
            "position": list(self.position) if self.position is not None else None,
            "heading": self.heading,
            "extent": list(self.extent) if self.extent is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            agent_id=data["agent_id"],
            timestamp=data["timestamp"],
            type=data["type"],
            velocity=tuple(data["velocity"]),
            acceleration=tuple(data["acceleration"]),
            position=tuple(data["position"]) if data.get("position") is not None else None,
            heading=data.get("heading"),
            extent=tuple(data["extent"]) if data.get("extent") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Edge:
    """Directed relation between two nodes in the SSTG."""

    source_id: str
    target_id: str
    source_timestamp: str
    target_timestamp: str
    edge_type: EdgeType
    weight: float = 1.0
    relation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_timestamp": self.source_timestamp,
            "target_timestamp": self.target_timestamp,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "relation": self.relation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            source_timestamp=data["source_timestamp"],
            target_timestamp=data["target_timestamp"],
            edge_type=EdgeType(data["edge_type"]),
            weight=float(data.get("weight", 1.0)),
            relation=data.get("relation"),
            metadata=dict(data.get("metadata", {})),
        )


class SSTG:
    """Thin wrapper around ``networkx.DiGraph`` with JSON-safe serialization."""

    _DEFAULT_TIMESTAMP_ORDER: Dict[str, int] = {"T_start": 0, "T_peak": 1, "T_end": 2}

    def __init__(self, scene_id: str, dt: float, metadata: Optional[Dict[str, Any]] = None):
        self.scene_id = scene_id
        self.dt = float(dt)
        self.metadata = dict(metadata or {})
        self.graph = nx.MultiDiGraph()

    @property
    def timestamps(self) -> List[str]:
        labels = {data["timestamp"] for _, data in self.graph.nodes(data=True)}
        return sorted(labels, key=self._timestamp_sort_key)

    def add_node(self, node: Node) -> None:
        self.graph.add_node(self._make_node_key(node.agent_id, node.timestamp), **node.to_dict())

    def has_node(self, agent_id: str, timestamp: str) -> bool:
        return self._make_node_key(agent_id, timestamp) in self.graph

    def get_node(self, agent_id: str, timestamp: str) -> Optional[Node]:
        node_key = self._make_node_key(agent_id, timestamp)
        if node_key not in self.graph:
            return None
        return Node.from_dict(self.graph.nodes[node_key])

    def add_edge(self, edge: Edge) -> None:
        source_key = self._make_node_key(edge.source_id, edge.source_timestamp)
        target_key = self._make_node_key(edge.target_id, edge.target_timestamp)
        if source_key not in self.graph or target_key not in self.graph:
            raise ValueError("Edge endpoints must exist in the graph before adding the edge.")

        self.graph.add_edge(
            source_key,
            target_key,
            key=self._make_edge_key(edge),
            **edge.to_dict(),
        )

    def get_nodes_at_timestamp(self, timestamp: str) -> List[Node]:
        nodes = [
            Node.from_dict(data)
            for _, data in self.graph.nodes(data=True)
            if data["timestamp"] == timestamp
        ]
        return sorted(nodes, key=lambda node: node.agent_id)

    def get_edges_at_timestamp(self, timestamp: str) -> List[Edge]:
        edges: List[Edge] = []
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            if data["source_timestamp"] == timestamp or data["target_timestamp"] == timestamp:
                edges.append(Edge.from_dict(data))
        return edges

    def get_agent_states(self, agent_id: str) -> List[Node]:
        nodes = [
            Node.from_dict(data)
            for _, data in self.graph.nodes(data=True)
            if data["agent_id"] == agent_id
        ]
        return sorted(nodes, key=lambda node: self._timestamp_sort_key(node.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "dt": self.dt,
            "metadata": self.metadata,
            "nodes": [
                {"key": key, "data": dict(data)}
                for key, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, "key": key, "data": dict(data)}
                for source, target, key, data in self.graph.edges(keys=True, data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSTG":
        sstg = cls(scene_id=data["scene_id"], dt=data["dt"], metadata=data.get("metadata"))
        for node_entry in data.get("nodes", []):
            sstg.graph.add_node(node_entry["key"], **node_entry["data"])
        for edge_entry in data.get("edges", []):
            sstg.graph.add_edge(
                edge_entry["source"],
                edge_entry["target"],
                key=edge_entry.get("key"),
                **edge_entry["data"],
            )
        return sstg

    def get_summary(self) -> Dict[str, Any]:
        node_counts_by_type: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_counts_by_type[data["type"]] = node_counts_by_type.get(data["type"], 0) + 1

        edge_counts_by_type: Dict[str, int] = {}
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            edge_type = data["edge_type"]
            edge_counts_by_type[edge_type] = edge_counts_by_type.get(edge_type, 0) + 1

        return {
            "scene_id": self.scene_id,
            "dt": self.dt,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "timestamps": self.timestamps,
            "node_counts_by_type": node_counts_by_type,
            "edge_counts_by_type": edge_counts_by_type,
            "metadata": self.metadata,
        }

    @classmethod
    def _make_node_key(cls, agent_id: str, timestamp: str) -> str:
        return f"{agent_id}@{timestamp}"

    @staticmethod
    def _make_edge_key(edge: Edge) -> str:
        relation = edge.relation or "none"
        return (
            f"{edge.edge_type.value}:"
            f"{edge.source_id}@{edge.source_timestamp}->"
            f"{edge.target_id}@{edge.target_timestamp}:"
            f"{relation}"
        )

    @classmethod
    def _timestamp_sort_key(cls, timestamp: str) -> Tuple[int, str]:
        return (cls._DEFAULT_TIMESTAMP_ORDER.get(timestamp, len(cls._DEFAULT_TIMESTAMP_ORDER)), timestamp)
