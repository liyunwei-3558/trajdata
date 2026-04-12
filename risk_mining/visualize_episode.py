#!/usr/bin/env python
"""
Bokeh-based interactive visualization for mined risk events.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import CheckboxGroup, ColumnDataSource, CustomJS, Div, HoverTool, LabelSet, Select
from bokeh.plotting import figure, output_file, save

sys.path.insert(0, str(Path(__file__).parent))

from trajdata import MapAPI
from trajdata.data_structures.agent import AgentType
from trajdata.utils import vis_utils


EDGE_TYPES = ["spatial", "temporal", "causal"]
MAP_LAYERS = ["road_areas", "road_lanes", "crosswalks", "walkways", "lane_centers"]

EDGE_COLORS = {
    "spatial": "#8d99ae",
    "temporal": "#457b9d",
    "causal": "#d62828",
}

AGENT_COLOR_BY_NAME = {
    "VEHICLE": vis_utils.get_agent_type_color(AgentType.VEHICLE),
    "PEDESTRIAN": vis_utils.get_agent_type_color(AgentType.PEDESTRIAN),
    "BICYCLE": vis_utils.get_agent_type_color(AgentType.BICYCLE),
    "MOTORCYCLE": vis_utils.get_agent_type_color(AgentType.MOTORCYCLE),
    "UNKNOWN": "#7a7a7a",
}


def list_events(element_dir: Path) -> List[Dict]:
    index_path = element_dir / "index.json"
    if not index_path.exists():
        return []
    with open(index_path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("elements", [])


def load_events(event_dir: Path) -> Dict[str, Dict]:
    events: Dict[str, Dict] = {}
    for event_path in sorted(event_dir.glob("event_*.json")):
        with open(event_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        events[payload["event_id"]] = payload
    return events


def _to_list_dict(data: Dict[str, Any]) -> Dict[str, List]:
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in data.items()}


def _normalize_xy_data(data: Dict[str, Any]) -> Dict[str, List]:
    normalized = _to_list_dict(data)
    normalized.setdefault("xs", [])
    normalized.setdefault("ys", [])
    return normalized


def _agent_type_from_name(type_name: str) -> int:
    try:
        return int(AgentType[type_name])
    except Exception:
        return int(AgentType.UNKNOWN)


def _timestamp_sort_key(timestamp: str) -> Tuple[int, str]:
    if timestamp == "T_start":
        return (0, timestamp)
    if timestamp == "T_peak":
        return (1, timestamp)
    if timestamp == "T_end":
        return (100, timestamp)
    mid_match = re.fullmatch(r"T_mid_(\d+)", timestamp)
    if mid_match is not None:
        return (10 + int(mid_match.group(1)), timestamp)
    return (1000, timestamp)


def _extract_timestamp_order(raw_event: Dict[str, Any]) -> List[str]:
    metadata_order = raw_event.get("metadata", {}).get("semantic_timestamp_order")
    if metadata_order:
        return list(metadata_order)
    window = raw_event.get("episode_window", {})
    return sorted(window.keys(), key=_timestamp_sort_key)


def _format_window(window: Dict[str, Any], timestamp_order: List[str]) -> str:
    return ", ".join(
        f"{label}={window[label]}"
        for label in timestamp_order
        if label in window
    )


def _infer_map_id(raw_event: Dict) -> str | None:
    env_name = raw_event["source_scene"]["env_name"]
    scene_name = raw_event["source_scene"]["scene_name"]
    if env_name == "sind":
        location = scene_name.split("_", 1)[0]
        return f"{env_name}:{location}"
    return None


def _compute_bbox(raw_event: Dict, margin: float = 60.0) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for node_entry in raw_event["sstg"]["nodes"]:
        pos = node_entry["data"].get("position")
        if pos is not None:
            xs.append(pos[0])
            ys.append(pos[1])
    if not xs:
        return (-50.0, 50.0, -50.0, 50.0)
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def _build_map_payload(raw_event: Dict, cache_path: Path) -> Dict[str, Dict[str, List]]:
    empty = {layer: {"xs": [], "ys": []} for layer in MAP_LAYERS}
    map_id = _infer_map_id(raw_event)
    if map_id is None:
        return empty

    try:
        map_api = MapAPI(cache_path, keep_in_memory=True)
        vec_map = map_api.get_map(
            map_id,
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
        )
        cds_tuple = vis_utils.get_map_cds(np.eye(3), vec_map, bbox=_compute_bbox(raw_event))
    except Exception:
        return empty

    return {
        "lane_centers": _normalize_xy_data(cds_tuple[0].data),
        "road_lanes": _normalize_xy_data(cds_tuple[1].data),
        "crosswalks": _normalize_xy_data(cds_tuple[2].data),
        "walkways": _normalize_xy_data(cds_tuple[3].data),
        "road_areas": _normalize_xy_data(cds_tuple[4].data),
    }


def _build_agent_geometry(node: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float]]:
    position = node.get("position")
    if position is None:
        return [], [], [], []

    extent = node.get("extent") or [4.3, 1.8, 1.5]
    length = float(extent[0]) if len(extent) > 0 else 4.3
    width = float(extent[1]) if len(extent) > 1 else 1.8
    heading = float(node.get("heading") or 0.0)
    agent_type = _agent_type_from_name(node["type"])

    rect_coords, dir_coords = vis_utils.compute_agent_rect_coords(
        agent_type,
        heading,
        length,
        width,
    )
    rect_xs = (rect_coords[:, 0] + position[0]).tolist()
    rect_ys = (rect_coords[:, 1] + position[1]).tolist()
    dir_xs = (dir_coords[:, 0] + position[0]).tolist()
    dir_ys = (dir_coords[:, 1] + position[1]).tolist()
    return rect_xs, rect_ys, dir_xs, dir_ys


def _build_event_payload(raw_event: Dict) -> Dict[str, Any]:
    timestamp_order = _extract_timestamp_order(raw_event)
    node_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    nodes_by_timestamp: Dict[str, List[Dict[str, Any]]] = {label: [] for label in timestamp_order}
    trajectory_points: Dict[str, Dict[str, Any]] = {}
    agent_positions_by_timestamp: Dict[str, Dict[str, Tuple[float, float]]] = {}
    primary_risk_agent_ids = set(raw_event.get("metadata", {}).get("primary_risk_agent_ids", []))
    trigger_agent_id = raw_event.get("metadata", {}).get("trigger_agent_id")
    if trigger_agent_id is not None:
        primary_risk_agent_ids.add(trigger_agent_id)

    for node_entry in raw_event["sstg"]["nodes"]:
        node = node_entry["data"]
        timestamp = node["timestamp"]
        if node["agent_id"] == raw_event["ego_agent_id"]:
            agent_role = "ego"
        elif node["agent_id"] in primary_risk_agent_ids:
            agent_role = "primary"
        else:
            agent_role = "secondary"
        rect_xs, rect_ys, dir_xs, dir_ys = _build_agent_geometry(node)
        record = {
            "agent_id": node["agent_id"],
            "display_id": str(node["agent_id"]),
            "agent_role": agent_role,
            "type": node["type"],
            "timestamp": timestamp,
            "x": node["position"][0] if node.get("position") else None,
            "y": node["position"][1] if node.get("position") else None,
            "vx": node["velocity"][0],
            "vy": node["velocity"][1],
            "ax": node["acceleration"][0],
            "ay": node["acceleration"][1],
            "heading": node.get("heading"),
            "raw_timestep": node.get("metadata", {}).get("raw_timestep"),
            "fill_color": AGENT_COLOR_BY_NAME.get(node["type"], AGENT_COLOR_BY_NAME["UNKNOWN"]),
            "line_color": "#d62828" if agent_role == "ego" else "#111111" if agent_role == "primary" else "#666666",
            "fill_alpha": 0.95 if agent_role == "ego" else 0.8 if agent_role == "primary" else 0.28,
            "speed_mps": float(np.linalg.norm(np.asarray(node["velocity"], dtype=float))),
            "speed_kph": float(np.linalg.norm(np.asarray(node["velocity"], dtype=float)) * 3.6),
            "xs": rect_xs,
            "ys": rect_ys,
            "dir_xs": dir_xs,
            "dir_ys": dir_ys,
        }
        if node.get("position") is not None:
            nodes_by_timestamp.setdefault(timestamp, []).append(record)
            node_lookup[(node["agent_id"], timestamp)] = record
            agent_positions_by_timestamp.setdefault(node["agent_id"], {})[timestamp] = (
                node["position"][0],
                node["position"][1],
            )

        trajectory = trajectory_points.setdefault(
            node["agent_id"],
            {
                "points_by_timestamp": {},
                "line_color": AGENT_COLOR_BY_NAME.get(node["type"], AGENT_COLOR_BY_NAME["UNKNOWN"]),
                "agent_id": node["agent_id"],
                "agent_role": agent_role,
                "type": node["type"],
            },
        )
        if node.get("position") is not None:
            trajectory["points_by_timestamp"][timestamp] = (
                node["position"][0],
                node["position"][1],
            )

    trajectories: List[Dict[str, Any]] = []
    for trajectory in trajectory_points.values():
        if trajectory["agent_role"] not in {"ego", "primary"}:
            continue
        points_by_timestamp = trajectory["points_by_timestamp"]
        current_segment: List[Tuple[float, float]] = []
        for timestamp in timestamp_order:
            point = points_by_timestamp.get(timestamp)
            if point is None:
                if len(current_segment) >= 2:
                    trajectories.append(
                        {
                            "xs": [pt[0] for pt in current_segment],
                            "ys": [pt[1] for pt in current_segment],
                            "line_color": trajectory["line_color"],
                            "agent_id": trajectory["agent_id"],
                            "agent_role": trajectory["agent_role"],
                            "type": trajectory["type"],
                        }
                    )
                current_segment = []
                continue
            current_segment.append(point)

        if len(current_segment) >= 2:
            trajectories.append(
                {
                    "xs": [pt[0] for pt in current_segment],
                    "ys": [pt[1] for pt in current_segment],
                    "line_color": trajectory["line_color"],
                    "agent_id": trajectory["agent_id"],
                    "agent_role": trajectory["agent_role"],
                    "type": trajectory["type"],
                }
            )

    edges_by_timestamp: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        label: {edge_type: [] for edge_type in EDGE_TYPES} for label in timestamp_order
    }
    for edge_entry in raw_event["sstg"]["edges"]:
        edge = edge_entry["data"]
        source_x, source_y, source_ghost = _resolve_edge_endpoint(
            agent_id=edge["source_id"],
            timestamp=edge["source_timestamp"],
            node_lookup=node_lookup,
            agent_positions_by_timestamp=agent_positions_by_timestamp,
            timestamp_order=timestamp_order,
        )
        target_x, target_y, target_ghost = _resolve_edge_endpoint(
            agent_id=edge["target_id"],
            timestamp=edge["target_timestamp"],
            node_lookup=node_lookup,
            agent_positions_by_timestamp=agent_positions_by_timestamp,
            timestamp_order=timestamp_order,
        )
        if None in (source_x, source_y, target_x, target_y):
            continue
        edge_status = edge.get("metadata", {}).get("edge_status", "active")
        if edge["edge_type"] == "spatial":
            line_alpha = 0.95 if edge_status == "active" else 0.45 if edge_status == "outside_roi" else 0.3
            line_dash = "solid" if edge_status == "active" else "dashed"
        elif edge["edge_type"] == "temporal":
            line_alpha = 0.8
            line_dash = "dashed"
        else:
            line_alpha = 0.95
            line_dash = "solid"
        edge_record = {
            "x0": source_x,
            "y0": source_y,
            "x1": target_x,
            "y1": target_y,
            "source_id": edge["source_id"],
            "target_id": edge["target_id"],
            "relation": edge.get("relation") or "",
            "weight": float(edge.get("weight", 1.0)),
            "color": EDGE_COLORS[edge["edge_type"]],
            "line_width": max(1.5, float(edge.get("weight", 1.0)) * 4.0),
            "edge_type": edge["edge_type"],
            "edge_status": edge_status,
            "line_alpha": line_alpha,
            "line_dash": line_dash,
            "source_is_ghost": source_ghost,
            "target_is_ghost": target_ghost,
            "timestamp_pair": f"{edge['source_timestamp']} -> {edge['target_timestamp']}",
            "metric_type": edge.get("metadata", {}).get("peak_metric_type", ""),
            "metric_value": edge.get("metadata", {}).get("peak_metric_value"),
            "details": json.dumps(edge.get("metadata", {}), ensure_ascii=False),
        }
        for timestamp in {edge["source_timestamp"], edge["target_timestamp"]}:
            if timestamp in edges_by_timestamp:
                edges_by_timestamp[timestamp][edge["edge_type"]].append(edge_record)

    return {
        "event_id": raw_event["event_id"],
        "scene_name": raw_event["source_scene"]["scene_name"],
        "env_name": raw_event["source_scene"]["env_name"],
        "ego_agent_id": raw_event["ego_agent_id"],
        "risk_score": raw_event["risk_score"],
        "episode_type": raw_event["episode_type"],
        "applied_rules": raw_event.get("applied_rules", []),
        "peak_metric_type": raw_event.get("metadata", {}).get("peak_metric_type"),
        "peak_metric_value": raw_event.get("metadata", {}).get("peak_metric_value"),
        "start_rule": raw_event.get("metadata", {}).get("start_rule"),
        "end_rule": raw_event.get("metadata", {}).get("end_rule"),
        "episode_window": raw_event["episode_window"],
        "timestamp_order": timestamp_order,
        "primary_risk_agent_ids": sorted(primary_risk_agent_ids),
        "nodes_by_timestamp": nodes_by_timestamp,
        "edges_by_timestamp": edges_by_timestamp,
        "trajectories": trajectories,
        "bbox": _compute_bbox(raw_event, margin=30.0),
    }


def _columns_from_records(records: List[Dict[str, Any]], fields: List[str]) -> Dict[str, List]:
    return {field: [record.get(field) for record in records] for field in fields}


def _resolve_edge_endpoint(
    agent_id: str,
    timestamp: str,
    node_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    agent_positions_by_timestamp: Dict[str, Dict[str, Tuple[float, float]]],
    timestamp_order: List[str],
) -> Tuple[Optional[float], Optional[float], bool]:
    node = node_lookup.get((agent_id, timestamp))
    if node is not None and node.get("x") is not None and node.get("y") is not None:
        return node["x"], node["y"], False

    positions = agent_positions_by_timestamp.get(agent_id, {})
    if not positions:
        return None, None, False

    if "T_peak" in positions:
        x, y = positions["T_peak"]
        return x, y, True

    for label in timestamp_order:
        if label in positions:
            x, y = positions[label]
            return x, y, True

    return None, None, False


def _build_info_html(event_payload: Dict[str, Any], timestamp: str) -> str:
    window = event_payload["episode_window"]
    timestamp_order = event_payload.get("timestamp_order", [])
    return (
        f"<b>Event</b>: {event_payload['event_id']}<br>"
        f"<b>Scene</b>: {event_payload['scene_name']} ({event_payload['env_name']})<br>"
        f"<b>Ego</b>: {event_payload['ego_agent_id']}<br>"
        f"<b>Primary Risk Agents</b>: {', '.join(event_payload['primary_risk_agent_ids'])}<br>"
        f"<b>Peak Metric</b>: {event_payload.get('peak_metric_type') or 'n/a'}"
        f" ({event_payload.get('peak_metric_value')})<br>"
        f"<b>Start Rule</b>: {event_payload.get('start_rule') or 'n/a'}<br>"
        f"<b>End Rule</b>: {event_payload.get('end_rule') or 'n/a'}<br>"
        f"<b>Episode Type</b>: {event_payload['episode_type']}<br>"
        f"<b>Risk Score</b>: {event_payload['risk_score']:.4f}<br>"
        f"<b>Timestamp</b>: {timestamp}<br>"
        f"<b>Window</b>: {_format_window(window, timestamp_order)}<br>"
        f"<b>Rules</b>: {', '.join(event_payload['applied_rules'])}"
    )


def create_interactive_document(events: Dict[str, Dict], initial_event_id: str, cache_path: Path):
    event_payloads = {event_id: _build_event_payload(payload) for event_id, payload in events.items()}
    map_payloads = {event_id: _build_map_payload(payload, cache_path) for event_id, payload in events.items()}
    current_payload = event_payloads[initial_event_id]
    current_timestamp = "T_peak" if "T_peak" in current_payload["timestamp_order"] else current_payload["timestamp_order"][0]

    rect_fields = [
        "xs",
        "ys",
        "agent_id",
        "display_id",
        "agent_role",
        "type",
        "speed_mps",
        "speed_kph",
        "heading",
        "raw_timestep",
        "fill_color",
        "line_color",
        "fill_alpha",
        "x",
        "y",
    ]
    point_fields = [
        "x",
        "y",
        "agent_id",
        "display_id",
        "agent_role",
        "type",
        "speed_mps",
        "speed_kph",
        "heading",
        "raw_timestep",
        "fill_color",
    ]
    dir_fields = ["xs", "ys", "fill_color", "line_color", "fill_alpha"]
    edge_fields = ["x0", "y0", "x1", "y1", "source_id", "target_id", "relation", "weight", "color", "line_width", "edge_type", "edge_status", "line_alpha", "line_dash", "source_is_ghost", "target_is_ghost", "timestamp_pair", "metric_type", "metric_value", "details"]
    traj_fields = ["xs", "ys", "line_color", "agent_id", "agent_role", "type"]

    rect_source = ColumnDataSource(_columns_from_records(current_payload["nodes_by_timestamp"][current_timestamp], rect_fields))
    point_source = ColumnDataSource(_columns_from_records(current_payload["nodes_by_timestamp"][current_timestamp], point_fields))
    dir_source = ColumnDataSource(
        {
            "xs": [record["dir_xs"] for record in current_payload["nodes_by_timestamp"][current_timestamp]],
            "ys": [record["dir_ys"] for record in current_payload["nodes_by_timestamp"][current_timestamp]],
            "fill_color": [record["fill_color"] for record in current_payload["nodes_by_timestamp"][current_timestamp]],
            "line_color": [record["line_color"] for record in current_payload["nodes_by_timestamp"][current_timestamp]],
            "fill_alpha": [record["fill_alpha"] for record in current_payload["nodes_by_timestamp"][current_timestamp]],
        }
    )
    edge_sources = {
        edge_type: ColumnDataSource(_columns_from_records(current_payload["edges_by_timestamp"][current_timestamp][edge_type], edge_fields))
        for edge_type in EDGE_TYPES
    }
    traj_source = ColumnDataSource(_columns_from_records(current_payload["trajectories"], traj_fields))
    map_sources = {layer: ColumnDataSource(map_payloads[initial_event_id][layer]) for layer in MAP_LAYERS}

    x_min, x_max, y_min, y_max = current_payload["bbox"]
    plot = figure(
        title="Risk Event Graph",
        width=1080,
        height=780,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        match_aspect=True,
        output_backend="canvas",
    )
    vis_utils.apply_default_settings(plot)
    plot.background_fill_color = "#ffffff"

    map_renderers = {
        "road_areas": plot.multi_polygons(
            xs="xs",
            ys="ys",
            source=map_sources["road_areas"],
            line_color="black",
            line_width=0.3,
            fill_alpha=0.1,
            fill_color=vis_utils.get_map_patch_color(2),
        ),
        "road_lanes": plot.patches(
            xs="xs",
            ys="ys",
            source=map_sources["road_lanes"],
            line_color="black",
            line_width=0.3,
            fill_alpha=0.12,
            fill_color=vis_utils.get_map_patch_color(1),
        ),
        "crosswalks": plot.patches(
            xs="xs",
            ys="ys",
            source=map_sources["crosswalks"],
            line_color="black",
            line_width=0.3,
            fill_alpha=0.5,
            fill_color=vis_utils.get_map_patch_color(3),
        ),
        "walkways": plot.patches(
            xs="xs",
            ys="ys",
            source=map_sources["walkways"],
            line_color="black",
            line_width=0.3,
            fill_alpha=0.25,
            fill_color=vis_utils.get_map_patch_color(4),
        ),
        "lane_centers": plot.multi_line(
            xs="xs",
            ys="ys",
            source=map_sources["lane_centers"],
            line_color="gray",
            line_alpha=0.5,
            line_width=1.2,
        ),
    }

    traj_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=traj_source,
        line_color="line_color",
        line_dash="dashed",
        line_width=2,
        line_alpha=0.65,
    )

    edge_renderers = {
        edge_type: plot.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=edge_sources[edge_type],
            line_color="color",
            line_width="line_width",
            line_alpha="line_alpha",
            line_dash="line_dash",
        )
        for edge_type in EDGE_TYPES
    }

    rect_renderer = plot.patches(
        xs="xs",
        ys="ys",
        source=rect_source,
        fill_color="fill_color",
        fill_alpha="fill_alpha",
        line_color="line_color",
        line_width=1.0,
    )
    point_renderer = plot.scatter(
        x="x",
        y="y",
        source=point_source,
        size=22,
        line_alpha=0.0,
        fill_alpha=0.001,
        fill_color="fill_color",
    )
    dir_renderer = plot.patches(
        xs="xs",
        ys="ys",
        source=dir_source,
        fill_color="fill_color",
        fill_alpha="fill_alpha",
        line_color="line_color",
        line_width=0.8,
    )
    label_renderer = LabelSet(
        x="x",
        y="y",
        text="display_id",
        source=point_source,
        text_font_size="8pt",
        text_color="#111111",
        x_offset=5,
        y_offset=5,
        background_fill_color="#ffffff",
        background_fill_alpha=0.65,
    )
    plot.add_layout(label_renderer)

    plot.add_tools(
        HoverTool(
            renderers=[point_renderer],
            tooltips=[
                ("agent", "@agent_id"),
                ("role", "@agent_role"),
                ("type", "@type"),
                ("position", "(@x{0.00}, @y{0.00})"),
                ("speed", "@speed_mps{0.00} m/s (@speed_kph{0.0} km/h)"),
                ("heading", "@heading{0.000}"),
                ("raw_timestep", "@raw_timestep"),
            ],
        )
    )
    plot.add_tools(
        HoverTool(
            renderers=[traj_renderer],
            tooltips=[("agent", "@agent_id"), ("role", "@agent_role"), ("type", "@type")],
        )
    )
    for edge_type, renderer in edge_renderers.items():
        plot.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[
                    ("edge_type", edge_type),
                    ("agents", "@source_id -> @target_id"),
                    ("relation", "@relation"),
                    ("status", "@edge_status"),
                    ("ghost", "@source_is_ghost -> @target_is_ghost"),
                    ("weight", "@weight{0.000}"),
                    ("timestamps", "@timestamp_pair"),
                    ("metric", "@metric_type (@metric_value)"),
                    ("details", "@details"),
                ],
            )
        )

    info_div = Div(text=_build_info_html(current_payload, current_timestamp), width=360, height=180)
    legend_div = Div(
        text=(
            "<b>Map Layers</b><br>"
            "road_areas: drivable polygons<br>"
            "road_lanes: lane surface polygons<br>"
            "crosswalks: pedestrian crossings<br>"
            "walkways: pedestrian areas<br>"
            "lane_centers: lane centerlines<br><br>"
            "<b>Scene Layers</b><br>"
            "trajectories: ego + primary risk only<br>"
            "agents: ego / primary highlighted, secondary faded<br>"
            "spatial: active solid, outside_roi dashed<br>"
            "out_of_scene spatial: kept in JSON, not drawable<br>"
            "temporal / causal: full graph relations"
        ),
        width=360,
        height=220,
    )

    event_select = Select(
        title="Event",
        value=initial_event_id,
        options=[(event_id, f"{event_id} | {payload['source_scene']['scene_name']}") for event_id, payload in events.items()],
        width=360,
    )
    timestamp_select = Select(title="Timestamp", value=current_timestamp, options=current_payload["timestamp_order"], width=180)
    map_checkbox = CheckboxGroup(labels=MAP_LAYERS, active=[0, 1, 2, 3, 4], width=320)
    scene_checkbox = CheckboxGroup(labels=["trajectories", "agents"] + EDGE_TYPES, active=[1, 2, 4], width=320)

    callback = CustomJS(
        args=dict(
            events=event_payloads,
            maps=map_payloads,
            rect_source=rect_source,
            point_source=point_source,
            dir_source=dir_source,
            traj_source=traj_source,
            road_areas_source=map_sources["road_areas"],
            road_lanes_source=map_sources["road_lanes"],
            crosswalks_source=map_sources["crosswalks"],
            walkways_source=map_sources["walkways"],
            lane_centers_source=map_sources["lane_centers"],
            spatial_source=edge_sources["spatial"],
            temporal_source=edge_sources["temporal"],
            causal_source=edge_sources["causal"],
            info_div=info_div,
            event_select=event_select,
            timestamp_select=timestamp_select,
            map_checkbox=map_checkbox,
            scene_checkbox=scene_checkbox,
            road_areas_renderer=map_renderers["road_areas"],
            road_lanes_renderer=map_renderers["road_lanes"],
            crosswalks_renderer=map_renderers["crosswalks"],
            walkways_renderer=map_renderers["walkways"],
            lane_centers_renderer=map_renderers["lane_centers"],
            traj_renderer=traj_renderer,
            rect_renderer=rect_renderer,
            point_renderer=point_renderer,
            dir_renderer=dir_renderer,
            label_renderer=label_renderer,
            spatial_renderer=edge_renderers["spatial"],
            temporal_renderer=edge_renderers["temporal"],
            causal_renderer=edge_renderers["causal"],
            plot=plot,
        ),
        code="""
const eventPayload = events[event_select.value];
const mapPayload = maps[event_select.value];
const timestampOrder = eventPayload.timestamp_order || [];
timestamp_select.options = timestampOrder;
if (!timestampOrder.includes(timestamp_select.value)) {
  timestamp_select.value = timestampOrder.includes("T_peak") ? "T_peak" : (timestampOrder[0] || "");
}
const timestamp = timestamp_select.value;

const setData = (source, data) => {
  source.data = data;
  source.change.emit();
};

const rectRecords = eventPayload.nodes_by_timestamp[timestamp] || [];
setData(rect_source, {
  xs: rectRecords.map(r => r.xs),
  ys: rectRecords.map(r => r.ys),
  agent_id: rectRecords.map(r => r.agent_id),
  agent_role: rectRecords.map(r => r.agent_role),
  type: rectRecords.map(r => r.type),
  speed_mps: rectRecords.map(r => r.speed_mps),
  speed_kph: rectRecords.map(r => r.speed_kph),
  heading: rectRecords.map(r => r.heading),
  raw_timestep: rectRecords.map(r => r.raw_timestep),
  fill_color: rectRecords.map(r => r.fill_color),
  line_color: rectRecords.map(r => r.line_color),
  fill_alpha: rectRecords.map(r => r.fill_alpha),
  x: rectRecords.map(r => r.x),
  y: rectRecords.map(r => r.y),
});
setData(point_source, {
  x: rectRecords.map(r => r.x),
  y: rectRecords.map(r => r.y),
  agent_id: rectRecords.map(r => r.agent_id),
  display_id: rectRecords.map(r => r.display_id),
  agent_role: rectRecords.map(r => r.agent_role),
  type: rectRecords.map(r => r.type),
  speed_mps: rectRecords.map(r => r.speed_mps),
  speed_kph: rectRecords.map(r => r.speed_kph),
  heading: rectRecords.map(r => r.heading),
  raw_timestep: rectRecords.map(r => r.raw_timestep),
  fill_color: rectRecords.map(r => r.fill_color),
});
setData(dir_source, {
  xs: rectRecords.map(r => r.dir_xs),
  ys: rectRecords.map(r => r.dir_ys),
  fill_color: rectRecords.map(r => r.fill_color),
  line_color: rectRecords.map(r => r.line_color),
  fill_alpha: rectRecords.map(r => r.fill_alpha),
});
setData(traj_source, {
  xs: eventPayload.trajectories.map(r => r.xs),
  ys: eventPayload.trajectories.map(r => r.ys),
  line_color: eventPayload.trajectories.map(r => r.line_color),
  agent_id: eventPayload.trajectories.map(r => r.agent_id),
  agent_role: eventPayload.trajectories.map(r => r.agent_role),
  type: eventPayload.trajectories.map(r => r.type),
});
setData(road_areas_source, mapPayload.road_areas || {xs: [], ys: []});
setData(road_lanes_source, mapPayload.road_lanes || {xs: [], ys: []});
setData(crosswalks_source, mapPayload.crosswalks || {xs: [], ys: []});
setData(walkways_source, mapPayload.walkways || {xs: [], ys: []});
setData(lane_centers_source, mapPayload.lane_centers || {xs: [], ys: []});

const updateEdge = (source, records) => {
  setData(source, {
    x0: records.map(r => r.x0),
    y0: records.map(r => r.y0),
    x1: records.map(r => r.x1),
    y1: records.map(r => r.y1),
    source_id: records.map(r => r.source_id),
    target_id: records.map(r => r.target_id),
    relation: records.map(r => r.relation),
    weight: records.map(r => r.weight),
    color: records.map(r => r.color),
    line_width: records.map(r => r.line_width),
    edge_type: records.map(r => r.edge_type),
    edge_status: records.map(r => r.edge_status),
    line_alpha: records.map(r => r.line_alpha),
    line_dash: records.map(r => r.line_dash),
    source_is_ghost: records.map(r => r.source_is_ghost),
    target_is_ghost: records.map(r => r.target_is_ghost),
    timestamp_pair: records.map(r => r.timestamp_pair),
    metric_type: records.map(r => r.metric_type),
    metric_value: records.map(r => r.metric_value),
    details: records.map(r => r.details),
  });
};

updateEdge(spatial_source, eventPayload.edges_by_timestamp[timestamp].spatial || []);
updateEdge(temporal_source, eventPayload.edges_by_timestamp[timestamp].temporal || []);
updateEdge(causal_source, eventPayload.edges_by_timestamp[timestamp].causal || []);

road_areas_renderer.visible = map_checkbox.active.includes(0);
road_lanes_renderer.visible = map_checkbox.active.includes(1);
crosswalks_renderer.visible = map_checkbox.active.includes(2);
walkways_renderer.visible = map_checkbox.active.includes(3);
lane_centers_renderer.visible = map_checkbox.active.includes(4);

traj_renderer.visible = scene_checkbox.active.includes(0);
rect_renderer.visible = scene_checkbox.active.includes(1);
point_renderer.visible = scene_checkbox.active.includes(1);
dir_renderer.visible = scene_checkbox.active.includes(1);
label_renderer.visible = scene_checkbox.active.includes(1);
spatial_renderer.visible = scene_checkbox.active.includes(2);
temporal_renderer.visible = scene_checkbox.active.includes(3);
causal_renderer.visible = scene_checkbox.active.includes(4);

const bbox = eventPayload.bbox;
plot.x_range.start = bbox[0];
plot.x_range.end = bbox[1];
plot.y_range.start = bbox[2];
plot.y_range.end = bbox[3];

const window = eventPayload.episode_window;
const windowText = (eventPayload.timestamp_order || [])
  .filter(label => Object.prototype.hasOwnProperty.call(window, label))
  .map(label => `${label}=${window[label]}`)
  .join(", ");
info_div.text =
  `<b>Event</b>: ${eventPayload.event_id}<br>` +
  `<b>Scene</b>: ${eventPayload.scene_name} (${eventPayload.env_name})<br>` +
  `<b>Ego</b>: ${eventPayload.ego_agent_id}<br>` +
  `<b>Primary Risk Agents</b>: ${eventPayload.primary_risk_agent_ids.join(", ")}<br>` +
  `<b>Peak Metric</b>: ${(eventPayload.peak_metric_type || "n/a")} (${eventPayload.peak_metric_value ?? "n/a"})<br>` +
  `<b>Start Rule</b>: ${(eventPayload.start_rule || "n/a")}<br>` +
  `<b>End Rule</b>: ${(eventPayload.end_rule || "n/a")}<br>` +
  `<b>Episode Type</b>: ${eventPayload.episode_type}<br>` +
  `<b>Risk Score</b>: ${eventPayload.risk_score.toFixed(4)}<br>` +
  `<b>Timestamp</b>: ${timestamp}<br>` +
  `<b>Window</b>: ${windowText}<br>` +
  `<b>Rules</b>: ${eventPayload.applied_rules.join(", ")}`;
""",
    )

    event_select.js_on_change("value", callback)
    timestamp_select.js_on_change("value", callback)
    map_checkbox.js_on_change("active", callback)
    scene_checkbox.js_on_change("active", callback)

    return row(
        plot,
        column(event_select, timestamp_select, map_checkbox, scene_checkbox, info_div, legend_div, width=380),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an interactive Bokeh viewer for mined risk events.")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--html", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    element_dir = output_dir / "libraries" / "risk_elements"
    event_dir = output_dir / "libraries" / "risk_events"

    if args.list:
        for entry in list_events(element_dir):
            print(f"{entry['event_id']} | scene={entry['scene_name']} | risk={entry['risk_score']:.4f}")
        return

    events = load_events(event_dir)
    if not events:
        raise FileNotFoundError(f"No event JSON files found in {event_dir}")

    initial_event_id = args.event_id or next(iter(events.keys()))
    if initial_event_id not in events:
        raise KeyError(f"Unknown event id: {initial_event_id}")

    default_cache_path = Path(".unified_data_cache")
    cache_path = Path(args.cache_path) if args.cache_path else (
        default_cache_path if default_cache_path.exists() else Path.home() / ".unified_data_cache"
    )
    html_path = Path(args.html) if args.html else output_dir / f"{initial_event_id}_interactive.html"
    output_file(html_path, title=f"Risk Event Viewer - {initial_event_id}")
    save(create_interactive_document(events, initial_event_id, cache_path=cache_path))
    print(f"Saved interactive viewer to {html_path}")


if __name__ == "__main__":
    main()
