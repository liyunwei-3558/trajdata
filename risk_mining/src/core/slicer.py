"""
Episode Slicer for extracting risk scenarios from trajectory data.

The slicer identifies peak risk moments (T_peak) based on combined risk indicators
and extracts episodes with pre/post buffers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from enum import Enum

from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures import Scene, AgentMetadata
from trajdata.data_structures.state import StateArray

from .scene_graph import SSTG, Node, Edge, EdgeType


class EpisodeType(Enum):
    """Types of risk episodes."""
    TTC_CRITICAL = "ttc_critical"           # Low time-to-collision
    DECELERATION = "deceleration"           # Hard braking
    LANE_CONFLICT = "lane_conflict"         # Conflicting lane usage
    INTERACTION = "interaction"             # Complex multi-agent interaction
    CROSSING = "crossing"                   # Crossing path scenarios


@dataclass
class Episode:
    """
    Represents a risk episode extracted from a scene.

    Attributes:
        t_start: Start timestep of the episode
        t_peak: Peak risk timestep (most critical moment)
        t_end: End timestep of the episode
        involved_agents: List of agent IDs involved in the episode
        risk_score: Overall risk score (0-1)
        episode_type: Type of risk episode
        sstg: Semantic Spatio-Temporal Graph for this episode
        metadata: Additional metadata about the episode
    """
    t_start: int
    t_peak: int
    t_end: int
    involved_agents: List[str]
    risk_score: float
    episode_type: EpisodeType
    sstg: Optional[SSTG] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_timesteps(self) -> int:
        """Duration in timesteps."""
        return self.t_end - self.t_start + 1

    @property
    def duration_seconds(self, dt: float) -> float:
        """Duration in seconds."""
        return self.duration_timesteps * dt


class Slicer:
    """
    Extracts risk episodes from scenes using rule-based risk indicators.

    The slicing process:
    1. Identifies T_peak using combined risk indicators
    2. Defines T_start and T_end with buffers
    3. Extracts agent states for involved agents
    4. Builds SSTG for the episode
    """

    def __init__(
        self,
        pre_buffer_sec: float = 2.0,
        post_buffer_sec: float = 2.0,
        ttc_threshold: float = 3.0,
        decel_threshold: float = 3.0,  # m/s^2
        distance_threshold: float = 50.0,  # meters
    ):
        """
        Initialize the slicer.

        Args:
            pre_buffer_sec: Seconds before T_peak to include
            post_buffer_sec: Seconds after T_peak to include
            ttc_threshold: TTC threshold for critical events (seconds)
            decel_threshold: Deceleration threshold for hard braking (m/s^2)
            distance_threshold: Maximum distance for considering interactions (m)
        """
        self.pre_buffer_sec = pre_buffer_sec
        self.post_buffer_sec = post_buffer_sec
        self.ttc_threshold = ttc_threshold
        self.decel_threshold = decel_threshold
        self.distance_threshold = distance_threshold

    def extract_episodes(
        self,
        scene: Scene,
        cache: DataFrameCache,
        center_point: Optional[Tuple[float, float]] = None,
    ) -> List[Episode]:
        """
        Extract risk episodes from a scene.

        Args:
            scene: The scene to extract episodes from
            cache: DataFrameCache for state queries
            center_point: Optional center point (x, y) for ROI filtering

        Returns:
            List of extracted episodes
        """
        episodes = []

        # Get scene timesteps
        timesteps = range(scene.length_timesteps)
        dt = scene.dt

        # Convert buffer times to timesteps
        pre_buffer_ts = int(self.pre_buffer_sec / dt)
        post_buffer_ts = int(self.post_buffer_sec / dt)

        # Find peak risk timesteps
        peak_risks = self._find_peak_risks(scene, cache, timesteps, center_point)

        for t_peak, risk_type, involved_agents, risk_score in peak_risks:
            # Define episode boundaries
            t_start = max(0, t_peak - pre_buffer_ts)
            t_end = min(scene.length_timesteps - 1, t_peak + post_buffer_ts)

            # Build SSTG for the episode
            sstg = self._build_episode_sstg(
                scene, cache, t_start, t_end, involved_agents
            )

            episode = Episode(
                t_start=t_start,
                t_peak=t_peak,
                t_end=t_end,
                involved_agents=list(involved_agents),
                risk_score=risk_score,
                episode_type=risk_type,
                sstg=sstg,
                metadata={
                    "scene_name": scene.name,
                    "env_name": scene.env_name,
                    "dt": dt,
                }
            )
            episodes.append(episode)

        return episodes

    def _find_peak_risks(
        self,
        scene: Scene,
        cache: DataFrameCache,
        timesteps: range,
        center_point: Optional[Tuple[float, float]],
    ) -> List[Tuple[int, EpisodeType, Set[str], float]]:
        """
        Find peak risk timesteps using combined indicators.

        Returns:
            List of (timestep, episode_type, involved_agents, risk_score)
        """
        risk_scores = {}  # timestep -> (type, agents, score)

        for timestep in timesteps:
            # Get agents at this timestep
            agents = scene.agent_presence[timestep]

            if center_point is not None:
                # Filter by ROI
                agents = self._filter_by_roi(agents, cache, timestep, center_point)

            if len(agents) < 2:
                continue

            # Get states for all agents (batched StateArray)
            agent_names = [a.name for a in agents]
            states_batched = cache.get_states(agent_names, timestep)

            # Create a dict mapping agent names to their individual states
            # StateArray is batched, where states_batched[i] corresponds to agents[i]
            states_dict = {
                agents[i].name: states_batched[i]
                for i in range(len(agents))
            }

            # Compute risk indicators
            ttc_min, ttc_agents = self._compute_min_ttc(states_dict, agents)
            decel_max, decel_agents = self._compute_max_deceleration(states_dict, agents)
            conflict_count, conflict_agents = self._compute_lane_conflicts(
                states_dict, agents, timestep
            )

            # Combined risk score
            score = 0.0
            episode_type = EpisodeType.INTERACTION
            involved_agents = set()

            # TTC indicator
            if ttc_min < self.ttc_threshold:
                ttc_score = 1.0 - (ttc_min / self.ttc_threshold)
                score += ttc_score * 0.5
                involved_agents.update(ttc_agents)
                episode_type = EpisodeType.TTC_CRITICAL

            # Deceleration indicator
            if decel_max > self.decel_threshold:
                decel_score = min(1.0, (decel_max - self.decel_threshold) / 3.0)
                score += decel_score * 0.3
                involved_agents.update(decel_agents)
                if episode_type == EpisodeType.INTERACTION:
                    episode_type = EpisodeType.DECELERATION

            # Lane conflict indicator
            if conflict_count > 0:
                conflict_score = min(1.0, conflict_count * 0.3)
                score += conflict_score * 0.2
                involved_agents.update(conflict_agents)
                if episode_type == EpisodeType.INTERACTION:
                    episode_type = EpisodeType.LANE_CONFLICT

            if score > 0.1 and len(involved_agents) >= 2:
                risk_scores[timestep] = (episode_type, involved_agents, score)

        # Sort by risk score and return top peaks
        sorted_risks = sorted(
            risk_scores.items(),
            key=lambda x: x[1][2],
            reverse=True
        )

        # Return top peaks (can be adjusted)
        return [
            (ts, ep_type, agents, score)
            for ts, (ep_type, agents, score) in sorted_risks[:10]
        ]

    def _filter_by_roi(
        self,
        agents: List[AgentMetadata],
        cache: DataFrameCache,
        timestep: int,
        center_point: Tuple[float, float],
    ) -> List[AgentMetadata]:
        """Filter agents by distance from center point."""
        filtered = []
        cx, cy = center_point

        for agent in agents:
            state = cache.get_state(agent.name, timestep)
            if state is None:
                continue

            x, y = state.position[0], state.position[1]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)

            if dist <= self.distance_threshold:
                filtered.append(agent)

        return filtered

    def _compute_min_ttc(
        self,
        states: Dict[str, StateArray],
        agents: List[AgentMetadata],
    ) -> Tuple[float, Set[str]]:
        """
        Compute minimum time-to-collision among all agent pairs.

        Returns:
            (min_ttc, set_of_involved_agent_ids)
        """
        min_ttc = float("inf")
        involved_agents = set()

        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                s1 = states.get(a1.name)
                s2 = states.get(a2.name)

                if s1 is None or s2 is None:
                    continue

                # Relative position and velocity
                pos1 = np.array([s1.position[0], s1.position[1]])
                pos2 = np.array([s2.position[0], s2.position[1]])
                vel1 = np.array([s1.velocity[0], s1.velocity[1]])
                vel2 = np.array([s2.velocity[0], s2.velocity[1]])

                rel_pos = pos2 - pos1
                rel_vel = vel2 - vel1

                distance = np.linalg.norm(rel_pos)
                rel_speed = np.linalg.norm(rel_vel)

                # Check if approaching
                if rel_speed > 0.001:
                    # Project relative position onto relative velocity
                    closing_rate = -np.dot(rel_pos, rel_vel) / (rel_speed * distance)

                    if closing_rate > 0:  # Approaching
                        ttc = distance / rel_speed
                        if ttc < min_ttc:
                            min_ttc = ttc
                            involved_agents = {a1.name, a2.name}

        return min_ttc, involved_agents

    def _compute_max_deceleration(
        self,
        states: Dict[str, StateArray],
        agents: List[AgentMetadata],
    ) -> Tuple[float, Set[str]]:
        """
        Compute maximum deceleration among all agents.

        Returns:
            (max_decel, set_of_agents_with_high_decel)
        """
        max_decel = 0.0
        involved_agents = set()

        for agent in agents:
            state = states.get(agent.name)
            if state is None:
                continue

            vx = state.velocity[0]
            vy = state.velocity[1]
            ax = state.acceleration[0]
            ay = state.acceleration[1]

            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.1:  # Only consider moving agents
                # Deceleration component opposite to velocity
                decel = -(vx * ax + vy * ay) / speed

                if decel > max_decel:
                    max_decel = decel
                    involved_agents = {agent.name}
                elif abs(decel - max_decel) < 0.1 and decel > self.decel_threshold:
                    involved_agents.add(agent.name)

        return max_decel, involved_agents

    def _compute_lane_conflicts(
        self,
        states: Dict[str, StateArray],
        agents: List[AgentMetadata],
        timestep: int,
    ) -> Tuple[int, Set[str]]:
        """
        Compute lane conflicts (dummy implementation).

        Real implementation would use map data to detect conflicting lanes.

        Returns:
            (conflict_count, set_of_involved_agents)
        """
        # Dummy: count close-proximity pairs as potential conflicts
        conflicts = 0
        involved = set()

        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                s1 = states.get(a1.name)
                s2 = states.get(a2.name)

                if s1 is None or s2 is None:
                    continue

                pos1 = np.array([s1.position[0], s1.position[1]])
                pos2 = np.array([s2.position[0], s2.position[1]])
                distance = np.linalg.norm(pos1 - pos2)

                # Close proximity potential conflict
                if distance < 10.0:
                    conflicts += 1
                    involved.update({a1.name, a2.name})

        return conflicts, involved

    def _build_episode_sstg(
        self,
        scene: Scene,
        cache: DataFrameCache,
        t_start: int,
        t_end: int,
        involved_agents: Set[str],
    ) -> SSTG:
        """
        Build SSTG for an episode.

        Creates nodes for each agent at each timestep and edges based on
        proximity and interaction.
        """
        sstg = SSTG(scene_id=scene.name, dt=scene.dt)

        # Create nodes
        for timestep in range(t_start, t_end + 1):
            # Get agents actually present at this timestep
            agents_at_ts = scene.agent_presence[timestep]
            present_agent_ids = {a.name for a in agents_at_ts}

            for agent_id in involved_agents:
                # Skip if agent not present at this timestep
                if agent_id not in present_agent_ids:
                    continue

                agent = next(
                    (a for a in scene.agents if a.name == agent_id),
                    None
                )
                if agent is None:
                    continue

                state = cache.get_state(agent_id, timestep)
                if state is None:
                    continue

                node = Node(
                    agent_id=agent_id,
                    agent_type=agent.type.value,  # Note: use .type not .agent_type
                    timestep=timestep,
                    position=(state.position[0], state.position[1]),
                    velocity=(state.velocity[0], state.velocity[1]),
                    acceleration=(state.acceleration[0], state.acceleration[1]),
                    heading=state.heading[0],
                    extent=(agent.extent.length, agent.extent.width),
                )
                sstg.add_node(node)

        # Create edges (proximity and interaction edges)
        for timestep in range(t_start, t_end + 1):
            # Get agents actually present at this timestep
            agents_at_ts = scene.agent_presence[timestep]
            present_agent_ids = {a.name for a in agents_at_ts}

            # Only consider agents present at this timestep
            agents_list = [aid for aid in involved_agents if aid in present_agent_ids]

            for i, a1_id in enumerate(agents_list):
                for a2_id in agents_list[i+1:]:
                    s1 = cache.get_state(a1_id, timestep)
                    s2 = cache.get_state(a2_id, timestep)

                    if s1 is None or s2 is None:
                        continue

                    pos1 = np.array([s1.position[0], s1.position[1]])
                    pos2 = np.array([s2.position[0], s2.position[1]])
                    distance = np.linalg.norm(pos1 - pos2)

                    # Spatial proximity edge
                    if distance < 30.0:
                        edge = Edge(
                            source_id=a1_id,
                            target_id=a2_id,
                            edge_type=EdgeType.SPATIAL_PROXIMITY,
                            weight=1.0 - (distance / 30.0),
                            distance=distance,
                        )
                        sstg.add_edge(edge, timestep)

                        # Bidirectional
                        edge2 = Edge(
                            source_id=a2_id,
                            target_id=a1_id,
                            edge_type=EdgeType.SPATIAL_PROXIMITY,
                            weight=1.0 - (distance / 30.0),
                            distance=distance,
                        )
                        sstg.add_edge(edge2, timestep)

                    # TTC-based interaction edge
                    vel1 = np.array([s1.velocity[0], s1.velocity[1]])
                    vel2 = np.array([s2.velocity[0], s2.velocity[1]])
                    rel_vel = vel2 - vel1
                    rel_speed = np.linalg.norm(rel_vel)

                    if rel_speed > 0.1:
                        closing_rate = -np.dot(pos2 - pos1, rel_vel) / (rel_speed * distance)
                        if closing_rate > 0:
                            ttc = distance / rel_speed
                            if ttc < self.ttc_threshold:
                                edge = Edge(
                                    source_id=a1_id,
                                    target_id=a2_id,
                                    edge_type=EdgeType.INTERACTION,
                                    weight=1.0 - (ttc / self.ttc_threshold),
                                    ttc=ttc,
                                    distance=distance,
                                )
                                sstg.add_edge(edge, timestep)

        return sstg
