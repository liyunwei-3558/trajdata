"""Traffic-light utilities for the SinD trajdata integration.

The SinD pkl files used by this project do not contain signal phases, while the
original CSV release stores phase change tables in a separate directory.  This
module keeps the integration optional: if no external traffic-light directory is
configured, the existing SinD cache path behaves exactly as before.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trajdata.maps import TrafficLightStatus

TRAFFIC_LIGHT_ENV_VARS: Tuple[str, ...] = (
    "SIND_TRAFFIC_LIGHT_DIR",
    "SIND_TRAFFIC_LIGHT_DATA_DIR",
)

LOCATION_TO_RAW_FOLDER: Dict[str, str] = {
    "cc": "Changchun_Pudong",
    "xa": "Xi'an_shanglin",
    "xasl": "Xi'an_shanglin",
    "tj": "tianjin",
    "cqNR": "Chongqing_NR",
    "cqIR": "Chongqing_IR",
    "cqR": "Chongqing_R",
}

SKIPPED_LOCATIONS: set[str] = set()
TRAFFIC_LIGHT_FILE_PATTERNS: Tuple[str, ...] = (
    "*Traffic*Light*.csv",
    "*Traffic_Lights*.csv",
)

DEFAULT_MAPPING_PATH = Path(__file__).with_name("sind_traffic_light_mapping.json")
DEFAULT_RAW_VIDEO_FPS = 30.0


@dataclass
class TrafficLightBuildReport:
    """Diagnostics emitted while building a SinD traffic-light cache table."""

    scene_name: str
    location: str
    scene_id: str
    status: str
    message: str = ""
    csv_path: Optional[str] = None
    num_lights: int = 0
    num_rows: int = 0
    num_scene_timesteps: int = 0
    alignment: str = ""
    status_counts: Dict[str, int] = field(default_factory=dict)
    unknown_codes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "scene_name": self.scene_name,
            "location": self.location,
            "scene_id": self.scene_id,
            "status": self.status,
            "message": self.message,
            "csv_path": self.csv_path,
            "num_lights": self.num_lights,
            "num_rows": self.num_rows,
            "num_scene_timesteps": self.num_scene_timesteps,
            "alignment": self.alignment,
            "status_counts": self.status_counts,
            "unknown_codes": self.unknown_codes,
        }


def configured_traffic_light_root(
    explicit_root: Optional[os.PathLike[str] | str] = None,
) -> Optional[Path]:
    """Resolve the optional SinD traffic-light CSV root directory."""
    candidates: List[os.PathLike[str] | str] = []
    if explicit_root is not None:
        candidates.append(explicit_root)

    for env_var in TRAFFIC_LIGHT_ENV_VARS:
        env_value = os.environ.get(env_var)
        if env_value:
            candidates.append(env_value)

    for candidate in candidates:
        root = Path(candidate).expanduser()
        if root.exists():
            return root

    return None


def normalize_scene_token(value: str) -> str:
    """Normalize scene/folder names for robust cross-release matching."""
    value = value.lower().replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def scene_id_to_raw_aliases(location: str, scene_id: str) -> List[str]:
    """Return likely raw CSV folder aliases for a trajdata SinD scene id."""
    aliases = [normalize_scene_token(scene_id)]

    if location in {"xa", "xasl"}:
        match = re.search(
            r"(?P<month>\d+)\.(?P<day>\d+)\s+"
            r"(?P<period>morning|night)\s+(?P<idx>\d+)",
            scene_id,
            re.IGNORECASE,
        )
        if match:
            period = "m" if match.group("period").lower() == "morning" else "n"
            aliases.append(
                f"xian_{int(match.group('month'))}{int(match.group('day'))}"
                f"_{period}{int(match.group('idx'))}"
            )

    if location == "cc":
        if normalize_scene_token(scene_id).startswith("xx"):
            aliases.append("changchun_pudong_xx")
        else:
            number_match = re.search(r"\d+", scene_id)
            if number_match:
                aliases.append(f"changchun_pudong_{int(number_match.group(0)):03d}")

    if location == "tj":
        match = re.match(r"(?P<folder>\d+_\d+_\d+)\b", scene_id)
        if match:
            aliases.append(match.group("folder"))

    return list(dict.fromkeys(aliases))


def find_traffic_light_csv(
    root: Path,
    location: str,
    scene_id: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the traffic-light CSV for a SinD scene.

    Returns:
        Tuple of (csv_path, scene_folder). Both values are None when not found.
    """
    if location in SKIPPED_LOCATIONS:
        return None, None

    raw_location = LOCATION_TO_RAW_FOLDER.get(location)
    if raw_location is None:
        return None, None

    location_dir = root / raw_location
    if not location_dir.exists():
        return None, None

    aliases = {normalize_scene_token(alias) for alias in scene_id_to_raw_aliases(location, scene_id)}
    scene_folders = [p for p in location_dir.iterdir() if p.is_dir()]
    folder_by_norm = {normalize_scene_token(p.name): p for p in scene_folders}

    scene_folder = next((folder_by_norm[alias] for alias in aliases if alias in folder_by_norm), None)

    if scene_folder is None and location == "cc":
        number_match = re.search(r"\d+", scene_id)
        if number_match:
            suffix = f"_{int(number_match.group(0)):03d}"
            scene_folder = next(
                (p for norm, p in folder_by_norm.items() if norm.endswith(suffix)),
                None,
            )

    if scene_folder is None:
        return None, None

    for pattern in TRAFFIC_LIGHT_FILE_PATTERNS:
        matches = sorted(scene_folder.glob(pattern))
        if matches:
            return matches[0], scene_folder

    return None, scene_folder


def _load_mapping(mapping_path: Optional[Path] = None) -> Dict[str, Any]:
    path = mapping_path or DEFAULT_MAPPING_PATH
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _light_key_from_column(column: str) -> Tuple[str, int]:
    lower_col = column.lower()
    kind = "general"
    if "vehicle" in lower_col:
        kind = "vehicle"
    elif "pedestrian" in lower_col:
        kind = "pedestrian"

    match = re.search(r"(\d+)\s*$", column)
    light_idx = int(match.group(1)) if match else 0
    return kind, light_idx


def _synthetic_lane_id(location: str, kind: str, light_idx: int) -> str:
    return f"sind_tl:{location}:{kind}:{light_idx}"


def _lane_ids_for_light(
    mapping: Mapping[str, Any],
    location: str,
    scene_id: str,
    kind: str,
    light_idx: int,
) -> List[str]:
    """Resolve optional light-to-lane mapping, falling back to synthetic ids."""
    light_key = f"{kind}:{light_idx}"
    location_mapping = mapping.get("locations", {}).get(location, {})
    scene_mapping = location_mapping.get("scenes", {}).get(scene_id, {})

    lane_ids: Sequence[str] = (
        scene_mapping.get("light_to_lanes", {}).get(light_key)
        or location_mapping.get("light_to_lanes", {}).get(light_key)
        or []
    )
    if lane_ids:
        return [str(lane_id) for lane_id in lane_ids]

    return [_synthetic_lane_id(location, kind, light_idx)]


def translate_sind_light_status(value: Any) -> Tuple[TrafficLightStatus, Optional[str]]:
    """Translate SinD light code to trajdata's limited traffic-light enum."""
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return TrafficLightStatus.UNKNOWN, str(value)

    if code == 0:
        return TrafficLightStatus.RED, None
    if code == 1:
        return TrafficLightStatus.GREEN, None
    if code == 3:
        return TrafficLightStatus.YELLOW, None

    return TrafficLightStatus.UNKNOWN, str(code)


def _traffic_light_columns(columns: Iterable[str]) -> List[str]:
    return [
        col
        for col in columns
        if col not in {"RawFrameID", "timestamp(ms)"}
        and "traffic" in col.lower()
        and "light" in col.lower()
    ]


def _compute_change_scene_ts(
    df: pd.DataFrame,
    scene_dt: float,
) -> Tuple[np.ndarray, str]:
    timestamp_col = "timestamp(ms)"
    if timestamp_col in df:
        timestamp_series = pd.to_numeric(df[timestamp_col], errors="coerce")
        timestamps = timestamp_series.to_numpy(dtype=float)
        finite_timestamps = timestamps[np.isfinite(timestamps)]
        if finite_timestamps.size > 1 and np.ptp(finite_timestamps) > 1e-6:
            timestamp_series = timestamp_series.interpolate().ffill().bfill()
            timestamps = timestamp_series.to_numpy(dtype=float)
            scene_ts = np.rint(timestamps / (scene_dt * 1000.0)).astype(int)
            return scene_ts, "timestamp(ms)"

    raw_frames = pd.to_numeric(df["RawFrameID"], errors="coerce").to_numpy(dtype=float)
    raw_start = np.nanmin(raw_frames)
    denom = DEFAULT_RAW_VIDEO_FPS * scene_dt
    scene_ts = np.rint((raw_frames - raw_start) / denom).astype(int)
    return scene_ts, f"RawFrameID_relative_{DEFAULT_RAW_VIDEO_FPS:g}fps"


def build_traffic_light_dataframe(
    scene_name: str,
    location: str,
    scene_id: str,
    scene_length: int,
    scene_dt: float,
    root: Optional[Path] = None,
    mapping_path: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], TrafficLightBuildReport]:
    """Build a trajdata traffic-light status table for one SinD scene."""
    report = TrafficLightBuildReport(
        scene_name=scene_name,
        location=location,
        scene_id=scene_id,
        status="skipped",
        num_scene_timesteps=scene_length,
    )

    root = configured_traffic_light_root(root)
    if root is None:
        report.message = (
            "No SinD traffic-light CSV root configured. Set "
            "SIND_TRAFFIC_LIGHT_DIR to enable optional signal caching."
        )
        return None, report

    if location in SKIPPED_LOCATIONS:
        report.message = f"Location {location} is intentionally skipped."
        return None, report

    csv_path, scene_folder = find_traffic_light_csv(root, location, scene_id)
    if csv_path is None:
        folder_msg = f" under {scene_folder}" if scene_folder else ""
        report.message = f"No TrafficLight CSV found{folder_msg}."
        return None, report

    report.csv_path = str(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty or "RawFrameID" not in df.columns:
        report.message = "TrafficLight CSV is empty or missing RawFrameID."
        return None, report

    light_columns = _traffic_light_columns(df.columns)
    if not light_columns:
        report.message = "TrafficLight CSV contains no traffic-light columns."
        return None, report

    mapping = _load_mapping(mapping_path)
    change_scene_ts, alignment = _compute_change_scene_ts(df, scene_dt)
    df = df.copy()
    df["_scene_ts"] = change_scene_ts
    df = df.sort_values("_scene_ts").drop_duplicates("_scene_ts", keep="last")

    change_scene_ts = df["_scene_ts"].to_numpy(dtype=int)
    all_scene_ts = np.arange(scene_length, dtype=int)
    records: List[Dict[str, Any]] = []
    unknown_codes: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}

    for column in light_columns:
        kind, light_idx = _light_key_from_column(column)
        lane_ids = _lane_ids_for_light(mapping, location, scene_id, kind, light_idx)

        translated_values: List[int] = []
        for raw_value in df[column].to_numpy():
            status, unknown_code = translate_sind_light_status(raw_value)
            translated_values.append(int(status))
            if unknown_code is not None:
                unknown_codes[unknown_code] = unknown_codes.get(unknown_code, 0) + 1

        translated = np.asarray(translated_values, dtype=int)
        source_idx = np.searchsorted(change_scene_ts, all_scene_ts, side="right") - 1
        scene_status = np.full(scene_length, int(TrafficLightStatus.UNKNOWN), dtype=int)
        valid_mask = source_idx >= 0
        scene_status[valid_mask] = translated[source_idx[valid_mask]]

        for status_value in scene_status:
            status_name = TrafficLightStatus(int(status_value)).name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        for lane_id in lane_ids:
            records.extend(
                {
                    "lane_id": lane_id,
                    "scene_ts": int(scene_ts),
                    "status": int(status_value),
                }
                for scene_ts, status_value in zip(all_scene_ts, scene_status)
            )

    if not records:
        report.message = "No traffic-light records were generated."
        return None, report

    tls_df = pd.DataFrame.from_records(records)
    tls_df.set_index(["lane_id", "scene_ts"], inplace=True)
    tls_df.sort_index(inplace=True)

    report.status = "ok"
    report.message = "Traffic-light table built successfully."
    report.num_lights = len(light_columns)
    report.num_rows = len(tls_df)
    report.alignment = alignment
    report.status_counts = status_counts
    report.unknown_codes = unknown_codes
    return tls_df, report
