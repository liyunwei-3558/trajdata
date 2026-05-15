from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from Simulation_test_toolchain.core.state_utils import get_agent_world_pose, to_numpy

try:
    from trajdata.maps.vec_map_elements import MapElementType
except Exception:  # pragma: no cover - optional dependency path
    MapElementType = None


QCNET_POINT_TYPES = {
    "DASH_SOLID_YELLOW": 0,
    "DASH_SOLID_WHITE": 1,
    "DASHED_WHITE": 2,
    "DASHED_YELLOW": 3,
    "DOUBLE_SOLID_YELLOW": 4,
    "DOUBLE_SOLID_WHITE": 5,
    "DOUBLE_DASH_YELLOW": 6,
    "DOUBLE_DASH_WHITE": 7,
    "SOLID_YELLOW": 8,
    "SOLID_WHITE": 9,
    "SOLID_DASH_WHITE": 10,
    "SOLID_DASH_YELLOW": 11,
    "SOLID_BLUE": 12,
    "NONE": 13,
    "UNKNOWN": 14,
    "CROSSWALK": 15,
    "CENTERLINE": 16,
}
QCNET_POINT_SIDES = {"LEFT": 0, "RIGHT": 1, "CENTER": 2}
QCNET_POLYGON_TYPES = {"VEHICLE": 0, "BIKE": 1, "BUS": 2, "PEDESTRIAN": 3}
QCNET_POLYGON_REL_TYPES = {"NONE": 0, "PRED": 1, "SUCC": 2, "LEFT": 3, "RIGHT": 4}
QCNET_INTERSECTION_TYPES = {"TRUE": 0, "FALSE": 1, "NONE": 2}


@dataclass
class QcnetPose:
    position: np.ndarray
    heading: float
    rotation: np.ndarray
    world_from_agent: np.ndarray
    agent_from_world: np.ndarray


def _as_numpy(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    return np.asarray(arr)


def _scalar(value: Any) -> float:
    arr = _as_numpy(value)
    if arr.shape == ():
        return float(arr.item())
    return float(np.asarray(arr).reshape(-1)[0])


def _rotate_xy(coords: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return coords.reshape(-1, 2)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    return (coords[:, :2] @ rotation.T).astype(np.float32, copy=False)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32, copy=False)


def _world_pose(obs, ego_idx: int) -> QcnetPose:
    pose = get_agent_world_pose(obs, ego_idx)
    position = np.asarray(pose["position"], dtype=np.float32)
    heading = float(pose["heading"])
    rotation = np.asarray(pose["rotation"], dtype=np.float32)
    world_from_agent = np.eye(3, dtype=np.float32)
    world_from_agent[:2, :2] = rotation
    world_from_agent[:2, 2] = position
    agent_from_world = np.linalg.inv(world_from_agent).astype(np.float32)
    return QcnetPose(
        position=position,
        heading=heading,
        rotation=rotation,
        world_from_agent=world_from_agent,
        agent_from_world=agent_from_world,
    )


def _sequence_len(seq: Any, fallback: int = 0) -> int:
    if seq is None:
        return fallback
    if hasattr(seq, "shape") and len(seq.shape) > 0:
        return int(seq.shape[0])
    if hasattr(seq, "__len__"):
        return len(seq)
    return fallback


def _extract_series(
    seq: Any, name: str, fallback: Optional[np.ndarray] = None
) -> np.ndarray:
    if seq is None:
        if fallback is None:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(fallback)
    if hasattr(seq, name):
        return _as_numpy(getattr(seq, name))
    if fallback is not None:
        return np.asarray(fallback)
    raise AttributeError(f"State sequence does not expose '{name}'.")


def _compute_velocity(
    position: np.ndarray, heading: np.ndarray, dt: float
) -> np.ndarray:
    if position.shape[0] < 2:
        return np.zeros((position.shape[0], 2), dtype=np.float32)
    diff = np.zeros_like(position, dtype=np.float32)
    diff[1:] = np.diff(position[:, :2], axis=0) / max(dt, 1e-6)
    if heading.size == position.shape[0]:
        c = np.cos(heading).astype(np.float32)
        s = np.sin(heading).astype(np.float32)
        lon = diff[:, 0] * c + diff[:, 1] * s
        lat = -diff[:, 0] * s + diff[:, 1] * c
        return np.stack([lon * c - lat * s, lon * s + lat * c], axis=-1)
    return diff[:, :2]


def _pad_sequence(
    values: np.ndarray,
    valid_len: int,
    target_len: int,
    pad_before: bool,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    valid_len = max(0, min(int(valid_len), values.shape[0] if values.ndim > 0 else 0))
    if valid_len > 0:
        values = values[-valid_len:] if pad_before else values[:valid_len]
    else:
        values = np.zeros(
            (0,) + values.shape[1:], dtype=values.dtype if values.size else np.float32
        )

    if valid_len >= target_len:
        values = values[-target_len:] if pad_before else values[:target_len]
        mask = np.ones((target_len,), dtype=bool)
        return values.astype(np.float32, copy=False), mask

    pad_len = target_len - valid_len
    pad_shape = (pad_len,) + values.shape[1:]
    pad = np.full(
        pad_shape, fill_value, dtype=values.dtype if values.size else np.float32
    )
    if pad_before:
        values = np.concatenate([pad, values], axis=0)
        mask = np.concatenate(
            [np.zeros((pad_len,), dtype=bool), np.ones((valid_len,), dtype=bool)]
        )
    else:
        values = np.concatenate([values, pad], axis=0)
        mask = np.concatenate(
            [np.ones((valid_len,), dtype=bool), np.zeros((pad_len,), dtype=bool)]
        )
    return values.astype(np.float32, copy=False), mask


def _sequence_components(
    seq: Any,
    valid_len: int,
    target_len: int,
    pad_before: bool,
    dt: float,
    observation_frame: Optional[QcnetPose],
) -> Dict[str, np.ndarray]:
    pos = _extract_series(seq, "position")
    heading = _extract_series(seq, "heading")
    velocity = None
    if hasattr(seq, "velocity"):
        velocity = _as_numpy(seq.velocity)
    if velocity is None or velocity.size == 0:
        velocity = _compute_velocity(pos, heading, dt)

    pos = np.asarray(pos[:, :2], dtype=np.float32)
    heading = np.asarray(heading, dtype=np.float32).reshape(-1)
    velocity = np.asarray(velocity[:, :2], dtype=np.float32)
    if observation_frame is not None:
        pos = (
            _rotate_xy(pos, observation_frame.rotation) + observation_frame.position[:2]
        )
        velocity = _rotate_xy(velocity, observation_frame.rotation)
        heading = _wrap_angle(heading + observation_frame.heading)

    pos_aligned, valid_mask = _pad_sequence(
        pos[:, :2], valid_len, target_len, pad_before
    )
    heading_aligned, _ = _pad_sequence(
        heading.reshape(-1, 1), valid_len, target_len, pad_before
    )
    heading_aligned = heading_aligned.reshape(-1)
    velocity_aligned, _ = _pad_sequence(
        velocity[:, :2], valid_len, target_len, pad_before
    )
    return {
        "position": pos_aligned,
        "heading": heading_aligned,
        "velocity": velocity_aligned,
        "valid_mask": valid_mask,
    }


def _agent_entry(
    seq_hist: Any,
    hist_len: int,
    seq_fut: Any,
    fut_len: int,
    hist_steps: int,
    fut_steps: int,
    dt: float,
    pad_before: bool,
    observation_frame: Optional[QcnetPose],
    predict_all_future: bool = True,
) -> Dict[str, np.ndarray]:
    hist = _sequence_components(
        seq_hist, hist_len, hist_steps, pad_before, dt, observation_frame
    )
    fut = (
        _sequence_components(seq_fut, fut_len, fut_steps, False, dt, observation_frame)
        if seq_fut is not None
        else {
            "position": np.zeros((fut_steps, 2), dtype=np.float32),
            "heading": np.zeros((fut_steps,), dtype=np.float32),
            "velocity": np.zeros((fut_steps, 2), dtype=np.float32),
            "valid_mask": np.zeros((fut_steps,), dtype=bool),
        }
    )

    position = np.concatenate([hist["position"], fut["position"]], axis=0)
    heading = np.concatenate([hist["heading"], fut["heading"]], axis=0)
    velocity = np.concatenate([hist["velocity"], fut["velocity"]], axis=0)
    valid_mask = np.concatenate([hist["valid_mask"], fut["valid_mask"]], axis=0)
    predict_mask = np.zeros((hist_steps + fut_steps,), dtype=bool)
    if predict_all_future and fut_steps > 0:
        predict_mask[hist_steps:] = True
    elif fut_steps > 0:
        predict_mask[hist_steps:] = fut["valid_mask"]

    return {
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "valid_mask": valid_mask,
        "predict_mask": predict_mask,
    }


def _polyline_points(
    points: np.ndarray,
    point_type: int,
    point_side: int,
) -> Dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return {
            "position": np.zeros((0, 2), dtype=np.float32),
            "orientation": np.zeros((0,), dtype=np.float32),
            "magnitude": np.zeros((0,), dtype=np.float32),
            "type": np.zeros((0,), dtype=np.int64),
            "side": np.zeros((0,), dtype=np.int64),
        }

    seg = np.diff(points[:, :2], axis=0)
    return {
        "position": points[:-1, :2].astype(np.float32, copy=False),
        "orientation": np.arctan2(seg[:, 1], seg[:, 0]).astype(np.float32, copy=False),
        "magnitude": np.linalg.norm(seg, axis=1).astype(np.float32, copy=False),
        "type": np.full((seg.shape[0],), point_type, dtype=np.int64),
        "side": np.full((seg.shape[0],), point_side, dtype=np.int64),
    }


def _interpolated_points(polyline: Any, max_dist: float = 2.0) -> np.ndarray:
    if polyline is None:
        return np.zeros((0, 2), dtype=np.float32)
    geom = (
        polyline.interpolate(max_dist=max_dist)
        if hasattr(polyline, "interpolate")
        else polyline
    )
    points = getattr(geom, "points", geom)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return points[:, :2]


def _element_center_xy(elem: Any) -> np.ndarray:
    if hasattr(elem, "center"):
        pts = _interpolated_points(elem.center)
    elif hasattr(elem, "polygon"):
        pts = _interpolated_points(elem.polygon)
    elif hasattr(elem, "exterior_polygon"):
        pts = _interpolated_points(elem.exterior_polygon)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
    if pts.size == 0:
        return np.zeros((2,), dtype=np.float32)
    return pts[0]


def _push_polygon(
    polygons: List[Dict[str, Any]],
    polygon_id_to_index: Dict[str, int],
    points: List[Dict[str, np.ndarray]],
    edge_index: List[Tuple[int, int]],
    elem_id: str,
    polygon_type: int,
    is_intersection: int,
    polyline_xy: np.ndarray,
    point_type: int,
    point_side: int,
) -> Optional[int]:
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    if polyline_xy.ndim != 2 or polyline_xy.shape[0] < 2:
        return None

    poly_pos = polyline_xy[0]
    orientation = float(
        np.arctan2(
            polyline_xy[1, 1] - polyline_xy[0, 1], polyline_xy[1, 0] - polyline_xy[0, 0]
        )
    )
    poly_idx = len(polygons)
    polygon_id_to_index[elem_id] = poly_idx
    polygons.append(
        {
            "position": poly_pos.astype(np.float32, copy=False),
            "orientation": np.array(orientation, dtype=np.float32),
            "type": np.array(polygon_type, dtype=np.int64),
            "is_intersection": np.array(is_intersection, dtype=np.int64),
        }
    )

    point_spec = _polyline_points(
        polyline_xy, point_type=point_type, point_side=point_side
    )
    for pos, ori, mag, typ, side in zip(
        point_spec["position"],
        point_spec["orientation"],
        point_spec["magnitude"],
        point_spec["type"],
        point_spec["side"],
    ):
        points.append(
            {
                "position": pos,
                "orientation": np.array(ori, dtype=np.float32),
                "magnitude": np.array(mag, dtype=np.float32),
                "type": np.array(typ, dtype=np.int64),
                "side": np.array(side, dtype=np.int64),
                "polygon_index": np.array(poly_idx, dtype=np.int64),
            }
        )
        edge_index.append((len(points) - 1, poly_idx))
    return poly_idx


def build_qcnet_sample_spec(
    obs: Any,
    ego_idx: int,
    *,
    map_radius: float = 150.0,
    hist_steps: Optional[int] = None,
    fut_steps: Optional[int] = None,
    predict_all_future: bool = True,
    predict_neighbors: bool = False,
    dt: Optional[float] = None,
) -> Dict[str, Any]:
    if hist_steps is None:
        hist_steps = (
            int(getattr(obs, "agent_hist", obs).shape[-2])
            if hasattr(getattr(obs, "agent_hist", obs), "shape")
            else 50
        )
    if fut_steps is None:
        fut_steps = (
            int(getattr(obs, "agent_fut", obs).shape[-2])
            if hasattr(getattr(obs, "agent_fut", obs), "shape")
            else 60
        )
    if dt is None:
        dt = float(_scalar(obs.dt[ego_idx])) if hasattr(obs, "dt") else 0.1

    pose = _world_pose(obs, ego_idx)
    observation_frame = None
    if getattr(obs, "agents_from_world_tf", None) is not None:
        agent_from_world = _as_numpy(obs.agents_from_world_tf[ego_idx])
        if not np.allclose(agent_from_world, np.eye(3, dtype=np.float32)):
            observation_frame = pose
    vector_map = getattr(obs, "vector_maps", None)
    vector_map = (
        vector_map[ego_idx]
        if vector_map is not None and len(vector_map) > ego_idx
        else None
    )
    if vector_map is None:
        raise ValueError(
            "QCNet requires vector maps. Enable incl_vector_map=True in trajdata."
        )

    hist_pad_before = True
    if hasattr(obs, "history_pad_dir"):
        hist_pad_before = str(getattr(obs, "history_pad_dir")).endswith("BEFORE")

    agent_ids: List[str] = [str(obs.agent_name[ego_idx])]
    agent_types: List[int] = [int(_scalar(obs.agent_type[ego_idx]))]
    agent_categories: List[int] = [3]
    agent_entries: List[Dict[str, np.ndarray]] = []

    ego_hist = obs.agent_hist[ego_idx]
    ego_fut = (
        obs.agent_fut[ego_idx] if getattr(obs, "agent_fut", None) is not None else None
    )
    ego_hist_len = (
        int(_scalar(obs.agent_hist_len[ego_idx]))
        if getattr(obs, "agent_hist_len", None) is not None
        else _sequence_len(ego_hist)
    )
    ego_fut_len = (
        int(_scalar(obs.agent_fut_len[ego_idx]))
        if getattr(obs, "agent_fut_len", None) is not None
        else _sequence_len(ego_fut)
    )
    agent_entries.append(
        _agent_entry(
            ego_hist,
            ego_hist_len,
            ego_fut,
            ego_fut_len,
            hist_steps,
            fut_steps,
            dt,
            pad_before=hist_pad_before,
            observation_frame=observation_frame,
            predict_all_future=predict_all_future,
        )
    )

    num_neigh = (
        int(_scalar(obs.num_neigh[ego_idx]))
        if getattr(obs, "num_neigh", None) is not None
        else 0
    )
    neigh_hist = getattr(obs, "neigh_hist", None)
    neigh_fut = getattr(obs, "neigh_fut", None)
    neigh_hist_len = getattr(obs, "neigh_hist_len", None)
    neigh_fut_len = getattr(obs, "neigh_fut_len", None)
    neigh_types = getattr(obs, "neigh_types", None)

    for neigh_idx in range(num_neigh):
        seq_hist = neigh_hist[ego_idx, neigh_idx] if neigh_hist is not None else None
        seq_fut = neigh_fut[ego_idx, neigh_idx] if neigh_fut is not None else None
        hist_len = (
            int(_scalar(neigh_hist_len[ego_idx, neigh_idx]))
            if neigh_hist_len is not None
            else _sequence_len(seq_hist)
        )
        fut_len = (
            int(_scalar(neigh_fut_len[ego_idx, neigh_idx]))
            if neigh_fut_len is not None
            else _sequence_len(seq_fut)
        )
        agent_ids.append(f"neighbor_{neigh_idx}")
        agent_types.append(
            int(_scalar(neigh_types[ego_idx, neigh_idx]))
            if neigh_types is not None
            else 0
        )
        agent_categories.append(2)
        agent_entries.append(
            _agent_entry(
                seq_hist,
                hist_len,
                seq_fut,
                fut_len,
                hist_steps,
                fut_steps,
                dt,
                pad_before=hist_pad_before,
                observation_frame=observation_frame,
                predict_all_future=predict_neighbors,
            )
        )

    position = np.stack([entry["position"] for entry in agent_entries], axis=0)
    heading = np.stack([entry["heading"] for entry in agent_entries], axis=0)
    velocity = np.stack([entry["velocity"] for entry in agent_entries], axis=0)
    valid_mask = np.stack([entry["valid_mask"] for entry in agent_entries], axis=0)
    predict_mask = np.stack([entry["predict_mask"] for entry in agent_entries], axis=0)

    polygons: List[Dict[str, Any]] = []
    points: List[Dict[str, np.ndarray]] = []
    point_to_polygon_edges: List[Tuple[int, int]] = []
    polygon_to_polygon_edges: List[Tuple[int, int]] = []
    polygon_to_polygon_types: List[int] = []
    polygon_id_to_index: Dict[str, int] = {}

    ego_xy = pose.position[:2]
    lanes: Iterable[Any] = vector_map.get_lanes_within(
        np.array([ego_xy[0], ego_xy[1], 0.0], dtype=np.float32), map_radius
    )
    if hasattr(vector_map, "get_areas_within") and MapElementType is not None:
        crosswalks = vector_map.get_areas_within(
            ego_xy, MapElementType.PED_CROSSWALK, map_radius
        )
        walkways = vector_map.get_areas_within(
            ego_xy, MapElementType.PED_WALKWAY, map_radius
        )
    else:
        crosswalks = []
        walkways = []

    def _sort_key(elem: Any) -> float:
        center = _element_center_xy(elem)
        if center.size == 0:
            return float("inf")
        return float(np.linalg.norm(center[:2] - ego_xy))

    map_elems: List[Tuple[str, Any, int, int, int]] = []
    for lane in sorted(list(lanes), key=_sort_key):
        map_elems.append(
            (
                str(lane.id),
                lane,
                QCNET_POLYGON_TYPES["VEHICLE"],
                QCNET_INTERSECTION_TYPES["FALSE"],
                QCNET_POINT_TYPES["CENTERLINE"],
            )
        )
    for elem in sorted(list(crosswalks), key=_sort_key):
        map_elems.append(
            (
                str(elem.id),
                elem,
                QCNET_POLYGON_TYPES["PEDESTRIAN"],
                QCNET_INTERSECTION_TYPES["NONE"],
                QCNET_POINT_TYPES["CROSSWALK"],
            )
        )
    for elem in sorted(list(walkways), key=_sort_key):
        map_elems.append(
            (
                str(elem.id),
                elem,
                QCNET_POLYGON_TYPES["PEDESTRIAN"],
                QCNET_INTERSECTION_TYPES["NONE"],
                QCNET_POINT_TYPES["CROSSWALK"],
            )
        )

    for elem_id, elem, polygon_type, is_intersection, point_type in map_elems:
        if hasattr(elem, "center"):
            pts = _interpolated_points(elem.center)
        elif hasattr(elem, "polygon"):
            pts = _interpolated_points(elem.polygon)
        elif hasattr(elem, "exterior_polygon"):
            pts = _interpolated_points(elem.exterior_polygon)
        else:
            pts = np.zeros((0, 2), dtype=np.float32)
        poly_idx = _push_polygon(
            polygons,
            polygon_id_to_index,
            points,
            point_to_polygon_edges,
            elem_id,
            polygon_type,
            is_intersection,
            pts[:, :2],
            point_type=point_type,
            point_side=QCNET_POINT_SIDES["CENTER"],
        )
        if poly_idx is None:
            continue

        if hasattr(elem, "prev_lanes"):
            for prev_id in getattr(elem, "prev_lanes", set()):
                if prev_id in polygon_id_to_index:
                    polygon_to_polygon_edges.append(
                        (polygon_id_to_index[prev_id], poly_idx)
                    )
                    polygon_to_polygon_types.append(QCNET_POLYGON_REL_TYPES["PRED"])
        if hasattr(elem, "next_lanes"):
            for next_id in getattr(elem, "next_lanes", set()):
                if next_id in polygon_id_to_index:
                    polygon_to_polygon_edges.append(
                        (polygon_id_to_index[next_id], poly_idx)
                    )
                    polygon_to_polygon_types.append(QCNET_POLYGON_REL_TYPES["SUCC"])
        if hasattr(elem, "adj_lanes_left"):
            for left_id in getattr(elem, "adj_lanes_left", set()):
                if left_id in polygon_id_to_index:
                    polygon_to_polygon_edges.append(
                        (polygon_id_to_index[left_id], poly_idx)
                    )
                    polygon_to_polygon_types.append(QCNET_POLYGON_REL_TYPES["LEFT"])
        if hasattr(elem, "adj_lanes_right"):
            for right_id in getattr(elem, "adj_lanes_right", set()):
                if right_id in polygon_id_to_index:
                    polygon_to_polygon_edges.append(
                        (polygon_id_to_index[right_id], poly_idx)
                    )
                    polygon_to_polygon_types.append(QCNET_POLYGON_REL_TYPES["RIGHT"])

    map_polygon_position = (
        np.stack([poly["position"] for poly in polygons], axis=0)
        if polygons
        else np.zeros((0, 2), dtype=np.float32)
    )
    map_polygon_orientation = (
        np.stack([poly["orientation"] for poly in polygons], axis=0)
        if polygons
        else np.zeros((0,), dtype=np.float32)
    )
    map_polygon_type = (
        np.stack([poly["type"] for poly in polygons], axis=0)
        if polygons
        else np.zeros((0,), dtype=np.int64)
    )
    map_polygon_is_intersection = (
        np.stack([poly["is_intersection"] for poly in polygons], axis=0)
        if polygons
        else np.zeros((0,), dtype=np.int64)
    )

    map_point_position = (
        np.stack([pt["position"] for pt in points], axis=0)
        if points
        else np.zeros((0, 2), dtype=np.float32)
    )
    map_point_orientation = (
        np.stack([pt["orientation"] for pt in points], axis=0)
        if points
        else np.zeros((0,), dtype=np.float32)
    )
    map_point_magnitude = (
        np.stack([pt["magnitude"] for pt in points], axis=0)
        if points
        else np.zeros((0,), dtype=np.float32)
    )
    map_point_type = (
        np.stack([pt["type"] for pt in points], axis=0)
        if points
        else np.zeros((0,), dtype=np.int64)
    )
    map_point_side = (
        np.stack([pt["side"] for pt in points], axis=0)
        if points
        else np.zeros((0,), dtype=np.int64)
    )

    point_edge_index = (
        np.asarray(point_to_polygon_edges, dtype=np.int64).T
        if point_to_polygon_edges
        else np.zeros((2, 0), dtype=np.int64)
    )
    polygon_edge_index = (
        np.asarray(polygon_to_polygon_edges, dtype=np.int64).T
        if polygon_to_polygon_edges
        else np.zeros((2, 0), dtype=np.int64)
    )
    polygon_edge_type = (
        np.asarray(polygon_to_polygon_types, dtype=np.int64)
        if polygon_to_polygon_types
        else np.zeros((0,), dtype=np.int64)
    )

    return {
        "agent": {
            "num_nodes": len(agent_entries),
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "valid_mask": valid_mask,
            "predict_mask": predict_mask,
            "av_index": 0,
            "id": agent_ids,
            "type": np.asarray(agent_types, dtype=np.int64),
            "category": np.asarray(agent_categories, dtype=np.int64),
        },
        "map_polygon": {
            "num_nodes": int(map_polygon_position.shape[0]),
            "position": map_polygon_position,
            "orientation": map_polygon_orientation,
            "type": map_polygon_type,
            "is_intersection": map_polygon_is_intersection,
        },
        "map_point": {
            "num_nodes": int(map_point_position.shape[0]),
            "position": map_point_position,
            "orientation": map_point_orientation,
            "magnitude": map_point_magnitude,
            "type": map_point_type,
            "side": map_point_side,
        },
        ("map_point", "to", "map_polygon"): {"edge_index": point_edge_index},
        ("map_polygon", "to", "map_polygon"): {
            "edge_index": polygon_edge_index,
            "type": polygon_edge_type,
        },
        "ego_pose": pose,
    }


def spec_to_heterodata(spec: Dict[str, Any]):
    try:
        import torch
        from torch_geometric.data import HeteroData
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "QCNet requires torch_geometric. Install QCNet's dependencies before running this policy."
        ) from exc

    data = HeteroData()
    for node_type in ("agent", "map_polygon", "map_point"):
        node_spec = spec[node_type]
        for key, value in node_spec.items():
            if key == "num_nodes":
                data[node_type]["num_nodes"] = int(value)
            elif key in {"id"}:
                data[node_type][key] = list(value)
            else:
                data[node_type][key] = torch.as_tensor(value)

    for edge_type in (
        ("map_point", "to", "map_polygon"),
        ("map_polygon", "to", "map_polygon"),
    ):
        edge_spec = spec[edge_type]
        for key, value in edge_spec.items():
            data[edge_type][key] = torch.as_tensor(value)
    return data


def build_qcnet_action(
    pred: Dict[str, Any],
    spec: Dict[str, Any],
    ego_idx: int = 0,
    step_index: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pose: QcnetPose = spec["ego_pose"]
    pi = pred["pi"][ego_idx]
    if hasattr(pi, "detach"):
        pi = pi.detach().cpu().numpy()
    pi = np.asarray(pi, dtype=np.float32)
    mode_idx = int(np.nanargmax(pi)) if np.isfinite(pi).any() else 0

    horizon = int(pred["loc_refine_pos"].shape[2])
    step_index = int(np.clip(step_index, 0, max(0, horizon - 1)))
    traj = pred["loc_refine_pos"][ego_idx, mode_idx, step_index, :2]
    if hasattr(traj, "detach"):
        traj = traj.detach().cpu().numpy()
    traj = np.asarray(traj, dtype=np.float32)

    if "loc_refine_head" in pred and pred["loc_refine_head"] is not None:
        heading_delta = pred["loc_refine_head"][ego_idx, mode_idx, step_index, 0]
        if hasattr(heading_delta, "detach"):
            heading_delta = float(heading_delta.detach().cpu().item())
        else:
            heading_delta = float(heading_delta)
    else:
        heading_delta = (
            float(np.arctan2(traj[1], traj[0])) if np.linalg.norm(traj) > 1e-6 else 0.0
        )

    world_xy = pose.position + traj @ pose.rotation.T
    heading = pose.heading + heading_delta
    if pi.ndim > 0 and pi.size > 0 and np.isfinite(pi).any():
        finite_pi = np.nan_to_num(pi, nan=-np.inf)
        shifted = finite_pi - np.max(finite_pi)
        exp_pi = np.exp(shifted)
        probs = exp_pi / max(float(exp_pi.sum()), 1e-12)
        mode_prob = float(probs[mode_idx])
    else:
        mode_prob = 1.0

    command = {
        "policy": "qcnet",
        "mode_index": mode_idx,
        "mode_prob": mode_prob,
        "step_index": step_index,
        "pred_local_x": float(traj[0]),
        "pred_local_y": float(traj[1]),
        "pred_local_heading": float(heading_delta),
    }
    xyh = np.array([world_xy[0], world_xy[1], heading], dtype=np.float32)
    return xyh, command
