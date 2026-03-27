#!/usr/bin/env python
"""
Visualize extracted risk episodes from the risk mining pipeline.

Usage:
    python visualize_episode.py --event-id event_8a854095
    python visualize_episode.py --list  # List all available events
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import SSTG, Node, Edge, EdgeType, Episode


# Agent type colors
AGENT_COLORS = {
    "VEHICLE": "#3498db",      # Blue
    "PEDESTRIAN": "#e74c3c",   # Red
    "BICYCLE": "#2ecc71",      # Green
    "MOTORCYCLE": "#f39c12",   # Orange
    "OTHER": "#95a5a6",        # Gray
}

# Agent type markers
AGENT_MARKERS = {
    "VEHICLE": "s",           # Square
    "PEDESTRIAN": "o",        # Circle
    "BICYCLE": "^",           # Triangle
    "MOTORCYCLE": "D",        # Diamond
    "OTHER": "x",             # X
}

# Edge type colors
EDGE_COLORS = {
    EdgeType.SPATIAL_PROXIMITY: "#bdc3c7",  # Light gray
    EdgeType.LEAD_FOLLOW: "#3498db",        # Blue
    EdgeType.CROSSING_PATH: "#e74c3c",      # Red
    EdgeType.CONFLICT_LANE: "#f39c12",      # Orange
    EdgeType.INTERACTION: "#9b59b6",        # Purple
}

# Edge type display names
EDGE_TYPE_NAMES = {
    EdgeType.SPATIAL_PROXIMITY: "Spatial Proximity",
    EdgeType.LEAD_FOLLOW: "Lead-Follow",
    EdgeType.CROSSING_PATH: "Crossing Path",
    EdgeType.CONFLICT_LANE: "Conflict Lane",
    EdgeType.INTERACTION: "Interaction",
}


def load_episode(event_dir: Path, event_id: str) -> Episode:
    """Load an episode from the library."""
    event_path = event_dir / f"{event_id}.pkl"
    with open(event_path, 'rb') as f:
        return pickle.load(f)


def list_events(element_dir: Path, event_dir: Path) -> List[Dict]:
    """List all available events."""
    # Load element index
    element_index_path = element_dir / "index.json"
    if element_index_path.exists():
        import json
        with open(element_index_path, 'r') as f:
            element_index = json.load(f)
        return element_index.get("elements", [])
    return []


def filter_stationary_agents(sstg: SSTG, min_avg_speed: float = 0.5, min_displacement: float = 5.0) -> Set[str]:
    """
    Filter out stationary vehicles.

    Args:
        sstg: The SSTG to filter
        min_avg_speed: Minimum average speed threshold (m/s)
        min_displacement: Minimum total displacement threshold (m)

    Returns:
        Set of agent IDs to exclude (stationary agents)
    """
    stationary_agents = set()

    for agent_id in sstg.graph.nodes():
        # Get all nodes for this agent
        agent_nodes = sstg.get_agent_states(agent_id)
        if len(agent_nodes) < 2:
            continue

        # Calculate average speed
        speeds = [node.speed for node in agent_nodes]
        avg_speed = np.mean(speeds)

        # Calculate total displacement
        if len(agent_nodes) >= 2:
            start_pos = np.array(agent_nodes[0].position)
            end_pos = np.array(agent_nodes[-1].position)
            displacement = np.linalg.norm(end_pos - start_pos)
        else:
            displacement = 0

        # Check if stationary (low speed AND low displacement)
        if avg_speed < min_avg_speed and displacement < min_displacement:
            stationary_agents.add(agent_id)

    return stationary_agents


def plot_edge_type_subplots(episode: Episode, stationary_agents: Set[str], save_path: Path = None):
    """
    Create separate subplots for each edge type at t_peak.

    Each subplot shows only the edges of a specific type.
    """
    if episode.sstg is None:
        print("No SSTG data available for visualization")
        return

    sstg = episode.sstg
    nodes_at_peak = sstg.get_nodes_at_timestep(episode.t_peak)
    edges_at_peak = sstg.get_edges_at_timestep(episode.t_peak)

    # Filter out stationary agents
    active_nodes = [n for n in nodes_at_peak if n.agent_id not in stationary_agents]
    active_agent_ids = {n.agent_id for n in active_nodes}

    # Group edges by type
    edges_by_type: Dict[EdgeType, List[Edge]] = {et: [] for et in EdgeType}
    for edge in edges_at_peak:
        # Only include edges involving active agents
        if edge.source_id in active_agent_ids and edge.target_id in active_agent_ids:
            edges_by_type[edge.edge_type].append(edge)

    # Create subplots for each edge type
    num_edge_types = len(EdgeType)
    ncols = 3
    nrows = (num_edge_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    fig.suptitle(
        f"Edge Type Analysis at t_peak={episode.t_peak}\n"
        f"Episode: {episode.metadata.get('scene_name', 'Unknown')} | "
        f"Risk Score: {episode.risk_score:.2f}\n"
        f"Filtered {len(stationary_agents)} stationary agents",
        fontsize=14
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten() if nrows > 1 else ([axes] if ncols == 1 else axes)

    for idx, edge_type in enumerate(EdgeType):
        ax = axes_flat[idx]
        edges = edges_by_type[edge_type]

        ax.set_title(f"{EDGE_TYPE_NAMES[edge_type]} ({len(edges)} edges)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Draw all active nodes
        for node in active_nodes:
            color = AGENT_COLORS.get(node.agent_type, "#95a5a6")
            marker = AGENT_MARKERS.get(node.agent_type, "o")
            ax.plot(
                node.position[0], node.position[1],
                marker=marker, color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=0.5,
                alpha=0.7
            )

        # Draw edges of this type
        edge_color = EDGE_COLORS[edge_type]
        for edge in edges:
            source_node = next((n for n in active_nodes if n.agent_id == edge.source_id), None)
            target_node = next((n for n in active_nodes if n.agent_id == edge.target_id), None)

            if source_node and target_node:
                # Line width based on weight
                lw = max(0.5, edge.weight * 3)
                ax.plot(
                    [source_node.position[0], target_node.position[0]],
                    [source_node.position[1], target_node.position[1]],
                    '-', color=edge_color, alpha=0.7, linewidth=lw
                )

                # Draw arrow if interaction
                if edge_type == EdgeType.INTERACTION or edge_type == EdgeType.LEAD_FOLLOW:
                    mid_x = (source_node.position[0] + target_node.position[0]) / 2
                    mid_y = (source_node.position[1] + target_node.position[1]) / 2
                    ax.annotate(
                        '', xy=(target_node.position[0], target_node.position[1]),
                        xytext=(source_node.position[0], source_node.position[1]),
                        arrowprops=dict(arrowstyle='->', color=edge_color, lw=lw, alpha=0.5)
                    )

    # Hide empty subplots
    for idx in range(num_edge_types, len(axes_flat)):
        axes_flat[idx].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Vehicle',
                  markerfacecolor=AGENT_COLORS['VEHICLE'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Pedestrian',
                  markerfacecolor=AGENT_COLORS['PEDESTRIAN'], markersize=10),
        plt.Line2D([0], [0], marker='^', color='w', label='Bicycle',
                  markerfacecolor=AGENT_COLORS['BICYCLE'], markersize=10),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved edge type subplots to: {save_path}")

    plt.show()


def plot_episode_static(episode: Episode, stationary_agents: Set[str], save_path: Path = None):
    """
    Create a static visualization of the episode.

    Shows:
    - Trajectories of all involved agents (excluding stationary)
    - Agent positions at t_peak
    - Velocity vectors
    """
    if episode.sstg is None:
        print("No SSTG data available for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Risk Episode: {episode.metadata.get('scene_name', 'Unknown')}\n"
        f"Type: {episode.episode_type.value} | Risk Score: {episode.risk_score:.2f}\n"
        f"t_start={episode.t_start} | t_peak={episode.t_peak} | t_end={episode.t_end}\n"
        f"Filtered {len(stationary_agents)} stationary agents",
        fontsize=14
    )

    sstg = episode.sstg

    # Plot 1: Trajectories (top-left)
    ax1 = axes[0, 0]
    ax1.set_title("Agent Trajectories (Non-stationary)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Group nodes by agent (exclude stationary)
    agent_trajectories: Dict[str, List[Tuple[float, float, float]]] = {}
    for node in sstg.get_nodes_at_timestep(episode.t_peak):
        if node.agent_id in stationary_agents:
            continue
        agent_id = node.agent_id
        if agent_id not in agent_trajectories:
            # Get full trajectory
            agent_nodes = sstg.get_agent_states(agent_id)
            agent_trajectories[agent_id] = [
                (n.position[0], n.position[1], float(n.heading))
                for n in agent_nodes
            ]

    # Plot trajectories
    for agent_id, traj in agent_trajectories.items():
        if not traj:
            continue
        xs, ys, hs = zip(*traj)
        agent_node = sstg.get_agent_states(agent_id)[0]
        agent_type = agent_node.agent_type
        color = AGENT_COLORS.get(agent_type, "#95a5a6")
        ax1.plot(xs, ys, '-', color=color, alpha=0.6, linewidth=1)

        # Mark t_peak position
        ax1.plot(xs[-1], ys[-1], marker='o', color=color, markersize=8)

        # Velocity vector at t_peak
        peak_node = sstg.get_agent_states(agent_id)[-1]
        vx, vy = peak_node.velocity
        ax1.arrow(
            xs[-1], ys[-1], vx*0.5, vy*0.5,
            head_width=2, head_length=2, fc=color, ec=color, alpha=0.8
        )

    # Plot 2: SSTG at t_peak (top-right)
    ax2 = axes[0, 1]
    ax2.set_title(f"SSTG at t_peak={episode.t_peak} (All Edge Types)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    nodes_at_peak = [n for n in sstg.get_nodes_at_timestep(episode.t_peak)
                     if n.agent_id not in stationary_agents]
    edges_at_peak = sstg.get_edges_at_timestep(episode.t_peak)

    # Filter edges to only include active agents
    active_ids = {n.agent_id for n in nodes_at_peak}
    edges_at_peak = [e for e in edges_at_peak
                     if e.source_id in active_ids and e.target_id in active_ids]

    # Draw edges first (so nodes are on top)
    for edge in edges_at_peak:
        source_node = next((n for n in nodes_at_peak if n.agent_id == edge.source_id), None)
        target_node = next((n for n in nodes_at_peak if n.agent_id == edge.target_id), None)

        if source_node and target_node:
            edge_color = EDGE_COLORS.get(edge.edge_type, "#bdc3c7")
            alpha = 0.3 if edge.edge_type == EdgeType.SPATIAL_PROXIMITY else 0.6
            ax2.plot(
                [source_node.position[0], target_node.position[0]],
                [source_node.position[1], target_node.position[1]],
                '-', color=edge_color, alpha=alpha, linewidth=1
            )

    # Draw nodes
    for node in nodes_at_peak:
        color = AGENT_COLORS.get(node.agent_type, "#95a5a6")
        ax2.plot(
            node.position[0], node.position[1],
            marker='o', color=color, markersize=10,
            markeredgecolor='black', markeredgewidth=0.5
        )
        ax2.text(
            node.position[0], node.position[1] + 2,
            node.agent_id, fontsize=6, ha='center'
        )

    # Add edge legend
    for edge_type, color in EDGE_COLORS.items():
        ax2.plot([], [], '-', color=color, label=EDGE_TYPE_NAMES[edge_type], linewidth=2)
    ax2.legend(fontsize=6, loc='upper right')

    # Plot 3: Speed profiles over time (bottom-left)
    ax3 = axes[1, 0]
    ax3.set_title("Agent Speeds Over Time (Non-stationary)")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Speed (m/s)")
    ax3.grid(True, alpha=0.3)

    active_agents = [aid for aid in episode.involved_agents if aid not in stationary_agents]
    for agent_id in active_agents[:10]:  # Limit to 10 agents
        agent_nodes = sstg.get_agent_states(agent_id)
        if not agent_nodes:
            continue
        timesteps = [n.timestep for n in agent_nodes]
        speeds = [n.speed for n in agent_nodes]
        agent_node = agent_nodes[0]
        agent_type = agent_node.agent_type
        color = AGENT_COLORS.get(agent_type, "#95a5a6")
        ax3.plot(timesteps, speeds, '-', color=color, alpha=0.7, linewidth=1, label=agent_id)

    ax3.legend(fontsize=6, ncol=2)

    # Plot 4: Edge statistics (bottom-right)
    ax4 = axes[1, 1]
    ax4.set_title("Edge Type Distribution")

    # Count edge types across all timesteps (filtered)
    edge_counts: Dict[str, int] = {}
    for ts in sstg.timesteps:
        for edge in sstg.get_edges_at_timestep(ts):
            if edge.source_id in active_ids and edge.target_id in active_ids:
                edge_type = edge.edge_type.value
                edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

    if edge_counts:
        edge_types = list(edge_counts.keys())
        counts = list(edge_counts.values())
        colors = [EDGE_COLORS.get(EdgeType(et), "#bdc3c7") for et in edge_types]

        bars = ax4.bar(edge_types, counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel("Count")
        ax4.set_xlabel("Edge Type")
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()


def plot_interactions_graph(episode: Episode, stationary_agents: Set[str], save_path: Path = None):
    """Plot the interaction graph at t_peak."""
    if episode.sstg is None:
        print("No SSTG data available for visualization")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(
        f"Interaction Graph at t_peak={episode.t_peak}\n"
        f"Episode: {episode.metadata.get('scene_name', 'Unknown')} | "
        f"Risk Score: {episode.risk_score:.2f}\n"
        f"Filtered {len(stationary_agents)} stationary agents"
    )

    sstg = episode.sstg
    nodes_at_peak = [n for n in sstg.get_nodes_at_timestep(episode.t_peak)
                     if n.agent_id not in stationary_agents]
    edges_at_peak = [e for e in sstg.get_edges_at_timestep(episode.t_peak)
                     if e.source_id not in stationary_agents and e.target_id not in stationary_agents]

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes with positions
    pos = {}
    node_colors = []
    for node in nodes_at_peak:
        G.add_node(node.agent_id)
        pos[node.agent_id] = (node.position[0], node.position[1])
        node_colors.append(AGENT_COLORS.get(node.agent_type, "#95a5a6"))

    # Add edges by type
    for edge_type in EdgeType:
        edge_colors_list = []
        edge_widths = []
        edge_labels = {}

        for edge in edges_at_peak:
            if edge.edge_type == edge_type:
                G.add_edge(edge.source_id, edge.target_id)
                edge_colors_list.append(EDGE_COLORS[edge.edge_type])
                edge_widths.append(max(1, edge.weight * 3))
                edge_labels[(edge.source_id, edge.target_id)] = f"{edge.weight:.2f}"

    # Draw the graph
    if G.edges:
        # Group edges by type for drawing
        for edge_type in EdgeType:
            edge_list = [(e.source_id, e.target_id) for e in edges_at_peak if e.edge_type == edge_type]
            if edge_list:
                edge_colors = [EDGE_COLORS[edge_type]] * len(edge_list)
                edge_widths = [max(1, e.weight * 3) for e in edges_at_peak if e.edge_type == edge_type]
                nx.draw_networkx_edges(
                    G, pos, ax=ax,
                    edgelist=edge_list,
                    edge_color=edge_colors,
                    width=edge_widths,
                    alpha=0.6,
                    arrows=True,
                    arrowsize=20,
                    arrowstyle='->,head_length=0.4,head_width=0.3',
                    label=EDGE_TYPE_NAMES[edge_type]
                )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=500,
        edgecolors='black',
        linewidths=1,
        alpha=0.9
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_weight='bold'
    )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Vehicle',
                  markerfacecolor=AGENT_COLORS['VEHICLE'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Pedestrian',
                  markerfacecolor=AGENT_COLORS['PEDESTRIAN'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Bicycle',
                  markerfacecolor=AGENT_COLORS['BICYCLE'], markersize=10),
    ]
    for edge_type, color in EDGE_COLORS.items():
        legend_elements.append(
            plt.Line2D([0], [0], color=color, label=EDGE_TYPE_NAMES[edge_type], linewidth=2)
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved graph visualization to: {save_path}")

    plt.show()


def create_ttc_heatmap(episode: Episode, stationary_agents: Set[str], save_path: Path = None):
    """Create a heatmap of TTC values between agent pairs at t_peak."""
    if episode.sstg is None:
        print("No SSTG data available for visualization")
        return

    nodes_at_peak = episode.sstg.get_nodes_at_timestep(episode.t_peak)
    nodes_at_peak = [n for n in nodes_at_peak if n.agent_id not in stationary_agents]
    agent_ids = sorted([n.agent_id for n in nodes_at_peak])

    if len(agent_ids) < 2:
        print("Not enough active agents for TTC heatmap")
        return

    # Compute pairwise TTC
    ttc_matrix = np.full((len(agent_ids), len(agent_ids)), np.inf)

    for i, aid1 in enumerate(agent_ids):
        for j, aid2 in enumerate(agent_ids):
            if i == j:
                ttc_matrix[i, j] = 0
                continue

            node1 = next((n for n in nodes_at_peak if n.agent_id == aid1), None)
            node2 = next((n for n in nodes_at_peak if n.agent_id == aid2), None)

            if node1 and node2:
                # Compute TTC
                pos1 = np.array(node1.position)
                pos2 = np.array(node2.position)
                vel1 = np.array(node1.velocity)
                vel2 = np.array(node2.velocity)

                rel_pos = pos2 - pos1
                rel_vel = vel2 - vel1
                distance = np.linalg.norm(rel_pos)
                rel_speed = np.linalg.norm(rel_vel)

                if rel_speed > 0.1:
                    closing_rate = -np.dot(rel_pos, rel_vel) / (rel_speed * distance)
                    if closing_rate > 0:
                        ttc = distance / rel_speed
                        ttc_matrix[i, j] = ttc

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"TTC Matrix at t_peak={episode.t_peak} (seconds)\nFiltered {len(stationary_agents)} stationary agents")

    # Use log scale for better visualization
    ttc_display = ttc_matrix.copy()
    ttc_display[ttc_display > 10] = 10  # Cap at 10 seconds for visualization

    im = ax.imshow(ttc_display, cmap='RdYlGn_r', vmin=0, vmax=10)

    # Labels
    ax.set_xticks(range(len(agent_ids)))
    ax.set_yticks(range(len(agent_ids)))
    ax.set_xticklabels(agent_ids, rotation=90, fontsize=8)
    ax.set_yticklabels(agent_ids, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('TTC (seconds)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(agent_ids)):
        for j in range(len(agent_ids)):
            if i != j and ttc_matrix[i, j] < 10:
                text = ax.text(j, i, f'{ttc_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved TTC heatmap to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize risk episodes")
    parser.add_argument("--event-id", type=str, help="Event ID to visualize")
    parser.add_argument("--list", action="store_true", help="List all available events")
    parser.add_argument("--output-dir", type=str, default="./output/visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--show-graph", action="store_true", help="Show interaction graph")
    parser.add_argument("--show-ttc", action="store_true", help="Show TTC heatmap")
    parser.add_argument("--show-edge-types", action="store_true", help="Show edge type subplots")
    parser.add_argument("--min-speed", type=float, default=0.5,
                       help="Minimum average speed threshold for filtering stationary agents (m/s)")
    parser.add_argument("--min-displacement", type=float, default=5.0,
                       help="Minimum displacement threshold for filtering stationary agents (m)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    element_dir = Path("./output/test/libraries/risk_elements")
    event_dir = Path("./output/test/libraries/risk_events")

    if args.list:
        print("Available events:")
        print("-" * 60)
        events = list_events(element_dir, event_dir)
        for event in events:
            # Extract event_id from element_id (element_f5923ef8 -> event_f5923ef8)
            element_id = event.get('element_id', event.get('event_id', 'unknown'))
            event_id = element_id.replace('element_', 'event_')
            print(f"  {event_id}: "
                  f"Scene={event['scene_name']}, "
                  f"Type={event['episode_type']}, "
                  f"Score={event['risk_score']:.2f}")
        return

    if not args.event_id:
        print("Error: Please specify --event-id or use --list to see available events")
        return

    # Load episode
    print(f"Loading episode: {args.event_id}")
    episode = load_episode(event_dir, args.event_id)

    print(f"  Scene: {episode.metadata.get('scene_name', 'Unknown')}")
    print(f"  Episode type: {episode.episode_type.value}")
    print(f"  Risk score: {episode.risk_score:.2f}")
    print(f"  t_start: {episode.t_start}, t_peak: {episode.t_peak}, t_end: {episode.t_end}")
    print(f"  Involved agents: {len(episode.involved_agents)}")

    if episode.sstg:
        summary = episode.sstg.get_summary()
        print(f"  SSTG: {summary['num_nodes']} nodes, {summary['num_edges']} edges")

    # Filter stationary agents
    print("\nFiltering stationary agents...")
    stationary_agents = filter_stationary_agents(
        episode.sstg,
        min_avg_speed=args.min_speed,
        min_displacement=args.min_displacement
    )
    print(f"  Found {len(stationary_agents)} stationary agents to filter")
    if stationary_agents:
        print(f"  Stationary agents: {list(stationary_agents)[:5]}{'...' if len(stationary_agents) > 5 else ''}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Static plot
    save_path = output_dir / f"{args.event_id}_overview.png"
    plot_episode_static(episode, stationary_agents, save_path)

    # Edge type subplots
    if args.show_edge_types:
        save_path = output_dir / f"{args.event_id}_edge_types.png"
        plot_edge_type_subplots(episode, stationary_agents, save_path)

    # Interaction graph
    if args.show_graph:
        save_path = output_dir / f"{args.event_id}_graph.png"
        plot_interactions_graph(episode, stationary_agents, save_path)

    # TTC heatmap
    if args.show_ttc:
        save_path = output_dir / f"{args.event_id}_ttc.png"
        create_ttc_heatmap(episode, stationary_agents, save_path)


if __name__ == "__main__":
    main()
