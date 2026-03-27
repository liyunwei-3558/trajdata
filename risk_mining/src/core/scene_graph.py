"""
Core data structures for Semantic Spatio-Temporal Graph (SSTG).

The SSTG represents traffic scenarios as directed graphs with nodes (agents)
and edges (relationships) that evolve over time.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx


class EdgeType(Enum):
    """Types of edges in the SSTG."""
    SPATIAL_PROXIMITY = "spatial_proximity"  # Agents within spatial threshold
    LEAD_FOLLOW = "lead_follow"              # One agent following another
    CROSSING_PATH = "crossing_path"          # Paths will intersect
    CONFLICT_LANE = "conflict_lane"          # On conflicting lanes
    INTERACTION = "interaction"              # General interaction (braking, swerving)


@dataclass
class Node:
    """
    Represents an agent at a specific timestep in the SSTG.

    Note: Agent type access should use `agent.type` (NOT `agent.agent_type`)
    when querying from trajdata's AgentMetadata.
    """
    agent_id: str
    agent_type: str  # "VEHICLE", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE", "OTHER"
    timestep: int
    position: Tuple[float, float]  # (x, y) in meters
    velocity: Tuple[float, float]  # (vx, vy) in m/s
    acceleration: Tuple[float, float]  # (ax, ay) in m/s^2
    heading: float  # radians
    extent: Tuple[float, float]  # (length, width) in meters

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "timestep": self.timestep,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "heading": self.heading,
            "extent": self.extent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Create node from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            timestep=data["timestep"],
            position=tuple(data["position"]),
            velocity=tuple(data["velocity"]),
            acceleration=tuple(data["acceleration"]),
            heading=data["heading"],
            extent=tuple(data["extent"]),
        )

    @property
    def speed(self) -> float:
        """Compute speed magnitude from velocity."""
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

    @property
    def acceleration_magnitude(self) -> float:
        """Compute acceleration magnitude."""
        return np.sqrt(self.acceleration[0]**2 + self.acceleration[1]**2)


@dataclass
class Edge:
    """
    Represents a relationship between two agents in the SSTG.
    """
    source_id: str  # Agent ID of source node
    target_id: str  # Agent ID of target node
    edge_type: EdgeType
    weight: float  # Strength of relationship (0-1)
    ttc: Optional[float] = None  # Time-to-collision in seconds (inf if not applicable)
    distance: Optional[float] = None  # Distance in meters

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "ttc": self.ttc if self.ttc is not None else float("inf"),
            "distance": self.distance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """Create edge from dictionary."""
        ttc = data["ttc"]
        if ttc == float("inf") or ttc is None:
            ttc = None
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data["weight"],
            ttc=ttc,
            distance=data["distance"],
        )


class SSTG:
    """
    Semantic Spatio-Temporal Graph representing a traffic scenario.

    The graph is a directed graph where nodes represent agent states at timesteps
    and edges represent relationships between agents.

    Uses networkx.DiGraph for storage and manipulation.
    """

    def __init__(self, scene_id: str, dt: float):
        """
        Initialize SSTG.

        Args:
            scene_id: Identifier for the source scene
            dt: Time delta in seconds between timesteps
        """
        self.scene_id = scene_id
        self.dt = dt
        self.graph = nx.DiGraph()
        self._timesteps: set = set()

    @property
    def timesteps(self) -> List[int]:
        """Get sorted list of timesteps in the graph."""
        return sorted(self._timesteps)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        node_key = self._make_node_key(node.agent_id, node.timestep)
        self.graph.add_node(
            node_key,
            agent_id=node.agent_id,
            agent_type=node.agent_type,
            timestep=node.timestep,
            position=node.position,
            velocity=node.velocity,
            acceleration=node.acceleration,
            heading=node.heading,
            extent=node.extent,
        )
        self._timesteps.add(node.timestep)

    def add_edge(self, edge: Edge, timestep: int) -> None:
        """
        Add an edge to the graph.

        Args:
            edge: The edge to add
            timestep: The timestep this edge exists at
        """
        source_key = self._make_node_key(edge.source_id, timestep)
        target_key = self._make_node_key(edge.target_id, timestep)

        if source_key in self.graph and target_key in self.graph:
            self.graph.add_edge(
                source_key,
                target_key,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                ttc=edge.ttc if edge.ttc is not None else float("inf"),
                distance=edge.distance,
            )
            self._timesteps.add(timestep)

    def get_nodes_at_timestep(self, timestep: int) -> List[Node]:
        """Get all nodes at a specific timestep."""
        nodes = []
        for node_key, node_data in self.graph.nodes(data=True):
            if node_data["timestep"] == timestep:
                nodes.append(Node(
                    agent_id=node_data["agent_id"],
                    agent_type=node_data["agent_type"],
                    timestep=node_data["timestep"],
                    position=node_data["position"],
                    velocity=node_data["velocity"],
                    acceleration=node_data["acceleration"],
                    heading=node_data["heading"],
                    extent=node_data["extent"],
                ))
        return nodes

    def get_subgraph_at_timestep(self, timestep: int) -> nx.DiGraph:
        """
        Get a subgraph containing only nodes and edges at a specific timestep.

        Args:
            timestep: The timestep to extract

        Returns:
            A networkx DiGraph containing only the specified timestep
        """
        node_keys = [
            self._make_node_key(data["agent_id"], timestep)
            for _, data in self.graph.nodes(data=True)
            if data["timestep"] == timestep
        ]
        return self.graph.subgraph(node_keys).copy()

    def get_edges_at_timestep(self, timestep: int) -> List[Edge]:
        """Get all edges at a specific timestep."""
        edges = []
        subgraph = self.get_subgraph_at_timestep(timestep)

        for source, target, edge_data in subgraph.edges(data=True):
            # Extract agent IDs from node keys
            source_agent_id = self.graph.nodes[source]["agent_id"]
            target_agent_id = self.graph.nodes[target]["agent_id"]

            ttc = edge_data.get("ttc", float("inf"))
            edges.append(Edge(
                source_id=source_agent_id,
                target_id=target_agent_id,
                edge_type=EdgeType(edge_data["edge_type"]),
                weight=edge_data["weight"],
                ttc=ttc if ttc != float("inf") else None,
                distance=edge_data.get("distance"),
            ))
        return edges

    def get_agent_states(self, agent_id: str, timesteps: Optional[List[int]] = None) -> List[Node]:
        """
        Get states for a specific agent across timesteps.

        Args:
            agent_id: The agent identifier
            timesteps: Optional list of timesteps to filter by

        Returns:
            List of nodes for the agent
        """
        nodes = []
        for node_key, node_data in self.graph.nodes(data=True):
            if node_data["agent_id"] == agent_id:
                if timesteps is None or node_data["timestep"] in timesteps:
                    nodes.append(Node(
                        agent_id=node_data["agent_id"],
                        agent_type=node_data["agent_type"],
                        timestep=node_data["timestep"],
                        position=node_data["position"],
                        velocity=node_data["velocity"],
                        acceleration=node_data["acceleration"],
                        heading=node_data["heading"],
                        extent=node_data["extent"],
                    ))
        return sorted(nodes, key=lambda n: n.timestep)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SSTG to dictionary for serialization.

        Returns:
            Dictionary containing graph data and metadata
        """
        nodes = []
        for node_data in self.graph.nodes(data=True):
            nodes.append({
                "key": node_data[0],
                "data": node_data[1],
            })

        edges = []
        for source, target, edge_data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "data": edge_data,
            })

        return {
            "scene_id": self.scene_id,
            "dt": self.dt,
            "timesteps": sorted(self._timesteps),
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSTG":
        """Create SSTG from dictionary."""
        sstg = cls(scene_id=data["scene_id"], dt=data["dt"])

        # Add nodes
        for node_entry in data["nodes"]:
            node_key = node_entry["key"]
            node_data = node_entry["data"]
            sstg.graph.add_node(node_key, **node_data)
            sstg._timesteps.add(node_data["timestep"])

        # Add edges
        for edge_entry in data["edges"]:
            sstg.graph.add_edge(
                edge_entry["source"],
                edge_entry["target"],
                **edge_entry["data"]
            )

        return sstg

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the SSTG."""
        node_counts_by_type: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            agent_type = data["agent_type"]
            node_counts_by_type[agent_type] = node_counts_by_type.get(agent_type, 0) + 1

        edge_counts_by_type: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data["edge_type"]
            edge_counts_by_type[edge_type] = edge_counts_by_type.get(edge_type, 0) + 1

        return {
            "scene_id": self.scene_id,
            "num_timesteps": len(self._timesteps),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "timestep_range": (min(self._timesteps) if self._timesteps else None,
                               max(self._timesteps) if self._timesteps else None),
            "node_counts_by_type": node_counts_by_type,
            "edge_counts_by_type": edge_counts_by_type,
        }

    @staticmethod
    def _make_node_key(agent_id: str, timestep: int) -> str:
        """Create a unique key for a node."""
        return f"{agent_id}_t{timestep}"
