"""Load the sample YAML config into a mutable object the services share."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"

JOINTS = (
    "base",
    "shoulder",
    "elbow",
    "wrist_vertical",
    "wrist_rotation",
    "cylinder",
)
ARM_JOINTS = JOINTS[:5]


@dataclass
class Pose:
    base: float
    shoulder: float
    elbow: float
    wrist_vertical: float
    wrist_rotation: float

    def angles(self) -> dict[str, float]:
        return {
            "base": self.base,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist_vertical": self.wrist_vertical,
            "wrist_rotation": self.wrist_rotation,
        }


@dataclass
class Zone:
    id: str
    x: float
    y: float
    reachable: bool
    approach: Pose
    grasp: Pose
    lift: Pose
    cover: Pose | None = None


@dataclass
class SceneObject:
    id: str
    label: str
    kind: str
    color: str
    close_angle: float
    zone: str | None
    localizable: bool
    image: dict | None = None

    @property
    def name(self) -> str:
        if self.kind == "fruit" and self.label == "orange":
            return "orange fruit"
        return f"{self.color} {self.kind}".replace("_", " ")


@dataclass
class CoverTool:
    """Sampling cover. Servo angles and object size stay uncalibrated until measured."""

    name: str = "Cylinder sampling cover"
    inside_diameter_mm: float = 110
    clear_height_mm: float = 100
    body_outside_diameter_mm: float = 112.4
    flange_outside_diameter_mm: float = 118
    wall_thickness_mm: float = 1.2
    roof_thickness_mm: float = 1.6
    sample_bore_mm: float = 6.2
    vent_diameter_mm: float = 1.8
    pivot_x_mm: float = -6
    pivot_y_mm: float = 40.5
    pivot_z_mm: float = 47
    tilt_axis: str = "y"
    park_servo_deg: float = 73
    sample_servo_deg: float = 10
    design_tilt_at_park_deg: float = -63
    design_tilt_at_sample_deg: float = -38
    angles_calibrated: bool = False
    approach_clearance_mm: float = 40
    seal_standoff_mm: float = 0
    assumed_object_diameter_mm: float = 100
    assumed_object_height_mm: float = 80
    placement_uncertainty_mm: float = 5
    dimensions_measured: bool = False
    descent_tolerance_px: float = 48
    surface_calibrated: bool = False
    surface_z_mm: float = 0
    mouth_base_deg: float = 0
    mouth_shoulder_deg: float = 0
    mouth_elbow_deg: float = 0
    mouth_wrist_vertical_deg: float = 0
    mouth_wrist_rotation_deg: float = 0


@dataclass
class Config:
    raw: dict
    server_host: str
    server_port: int
    mode: str
    motion_timeout_s: float
    sim_joint_s: float
    pose_hz: float
    lift_segments: int
    sample_hold_s: float
    enose_step_s: float
    serial_mode: str
    serial_port: str
    baud: int
    ready_token: str
    ready_timeout_s: float
    stop_timeout_s: float
    camera_mode: str
    camera_index: int
    width: int
    height: int
    stale_after_s: float
    llm_enabled: bool
    llm_api: str
    llm_base_url: str
    llm_model: str
    llm_timeout_s: float
    vlm_enabled: bool
    vlm_api: str
    vlm_base_url: str
    vlm_model: str
    vlm_timeout_s: float
    enose_mode: str
    enose_base_url: str
    enose_timeout_s: float
    database_url: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    zone_radius_mm: float
    transit_shoulder: float
    clearance_shoulder_min: float
    clearance_shoulder_max: float
    forbidden_elbow_below: float
    forbidden_shoulder_above: float
    joint_limits: dict[str, tuple[float, float]]
    gripper_open: float
    gripper_closed: float
    park: Pose
    sample: Pose
    scan: list[Pose]
    zones: dict[str, Zone]
    objects: dict[str, SceneObject]
    calibration: dict
    tool: CoverTool
    aim_offset_x: float = 1
    aim_offset_y: float = 1
    aim_offset_z: float = 1
    pose_errors: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or DEFAULT_CONFIG_PATH
        data = yaml.safe_load(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        data = deepcopy(data)
        ex = data["execution"]
        ser = data["serial"]
        cam = data["camera"]
        llm = data["llm"]
        vlm = data["vlm"]
        nose = data["enose"]
        space = data["workspace"]
        limits = {
            name: (float(spec["min"]), float(spec["max"]))
            for name, spec in data["joints"].items()
        }
        poses = data["poses"]
        zones = {}
        for zid, spec in data["zones"].items():
            zones[zid] = Zone(
                id=zid,
                x=float(spec["x"]),
                y=float(spec["y"]),
                reachable=bool(spec["reachable"]),
                approach=_pose(spec["approach"]),
                grasp=_pose(spec["grasp"]),
                lift=_pose(spec["lift"]),
                cover=_pose(spec["cover"]) if spec.get("cover") else None,
            )
        aim = data.get("aim") or {}
        objects = {}
        for oid, spec in data["objects"].items():
            objects[oid] = SceneObject(
                id=oid,
                label=spec["label"],
                kind=spec["kind"],
                color=spec["color"],
                close_angle=float(spec["close_angle"]),
                zone=spec.get("zone"),
                localizable=bool(spec.get("localizable", True)),
                image=spec.get("image"),
            )
        cfg = cls(
            raw=data,
            server_host=data["server"]["host"],
            server_port=int(data["server"]["port"]),
            mode=ex["mode"],
            motion_timeout_s=float(ex["motion_timeout_s"]),
            sim_joint_s=float(ex["sim_joint_s"]),
            pose_hz=float(ex.get("pose_hz", 30)),
            lift_segments=int(ex["lift_segments"]),
            sample_hold_s=float(ex["sample_hold_s"]),
            enose_step_s=float(ex["enose_step_s"]),
            serial_mode=ser["mode"],
            serial_port=ser.get("port") or "",
            baud=int(ser["baud"]),
            ready_token=ser["ready_token"],
            ready_timeout_s=float(ser["ready_timeout_s"]),
            stop_timeout_s=float(ser["stop_timeout_s"]),
            camera_mode=cam["mode"],
            camera_index=int(cam["index"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            stale_after_s=float(cam["stale_after_s"]),
            llm_enabled=bool(llm["enabled"]),
            llm_api=str(llm.get("api") or ("ollama" if ":11434" in str(llm["base_url"]) else "openai")),
            llm_base_url=llm["base_url"],
            llm_model=llm["model"],
            llm_timeout_s=float(llm["timeout_s"]),
            vlm_enabled=bool(vlm["enabled"]),
            vlm_api=str(vlm.get("api") or ("ollama" if ":11434" in str(vlm["base_url"]) else "openai")),
            vlm_base_url=vlm["base_url"],
            vlm_model=vlm["model"],
            vlm_timeout_s=float(vlm["timeout_s"]),
            enose_mode=nose["mode"],
            enose_base_url=nose["base_url"],
            enose_timeout_s=float(nose["timeout_s"]),
            database_url=os.environ.get("DATABASE_URL") or str((data.get("database") or {}).get("url") or ""),
            x_min=float(space["x_min"]),
            x_max=float(space["x_max"]),
            y_min=float(space["y_min"]),
            y_max=float(space["y_max"]),
            zone_radius_mm=float(space["zone_radius_mm"]),
            transit_shoulder=float(space["transit_shoulder"]),
            clearance_shoulder_min=float(space["clearance_shoulder_min"]),
            clearance_shoulder_max=float(space["clearance_shoulder_max"]),
            forbidden_elbow_below=float(space["forbidden_elbow_below"]),
            forbidden_shoulder_above=float(space["forbidden_shoulder_above"]),
            joint_limits=limits,
            gripper_open=float(data["joints"]["cylinder"]["open"]),
            gripper_closed=float(data["joints"]["cylinder"]["closed"]),
            park=_pose(poses["park"]),
            sample=_pose(poses["sample"]),
            scan=[_pose(item) for item in poses["scan"]],
            zones=zones,
            objects=objects,
            calibration=data["calibration"],
            tool=_cover_tool(data.get("tool") or {}),
            aim_offset_x=float(aim.get("offset_x", 1) or 1),
            aim_offset_y=float(aim.get("offset_y", 1) or 1),
            aim_offset_z=float(aim.get("offset_z", 1) or 1),
        )
        cfg.pose_errors = validate_taught_poses(cfg)
        return cfg


def _cover_tool(spec: dict) -> CoverTool:
    defaults = CoverTool()
    values = {}
    for name in defaults.__dataclass_fields__:
        if name in spec and spec[name] is not None:
            current = getattr(defaults, name)
            raw = spec[name]
            if isinstance(current, bool):
                values[name] = bool(raw)
            elif isinstance(current, float):
                values[name] = float(raw)
            else:
                values[name] = raw
    return CoverTool(**values)


def _pose(spec: dict) -> Pose:
    return Pose(
        base=float(spec["base"]),
        shoulder=float(spec["shoulder"]),
        elbow=float(spec["elbow"]),
        wrist_vertical=float(spec["wrist_vertical"]),
        wrist_rotation=float(spec["wrist_rotation"]),
    )


def validate_taught_poses(cfg: Config) -> list[str]:
    """Refuse to boot a motion stack whose taught poses already break the rules."""
    from app.geometry import pose_problem

    errors = []
    named = [("park", cfg.park), ("sample", cfg.sample)]
    named += [(f"scan[{i}]", pose) for i, pose in enumerate(cfg.scan)]
    for zone in cfg.zones.values():
        named.append((f"{zone.id}.approach", zone.approach))
        named.append((f"{zone.id}.grasp", zone.grasp))
        named.append((f"{zone.id}.lift", zone.lift))
        if zone.cover is not None:
            named.append((f"{zone.id}.cover", zone.cover))
    for name, pose in named:
        problem = pose_problem(pose.angles(), cfg, gripper=cfg.gripper_open)
        if problem:
            errors.append(f"{name}: {problem}")
        problem = pose_problem(pose.angles(), cfg, gripper=cfg.gripper_closed)
        if problem:
            errors.append(f"{name} with the cylinder at its other limit: {problem}")
    low, high = cfg.joint_limits["shoulder"]
    if not low <= cfg.transit_shoulder <= high:
        errors.append("transit_shoulder is outside the shoulder limits")
    lo, hi = cfg.joint_limits["cylinder"]
    for label, angle in (("park", cfg.tool.park_servo_deg), ("sample", cfg.tool.sample_servo_deg)):
        if not lo <= angle <= hi:
            errors.append(f"cover {label} servo {angle} is outside {lo}–{hi}")
    for obj in cfg.objects.values():
        lo, hi = cfg.joint_limits["cylinder"]
        if not lo <= obj.close_angle <= hi:
            errors.append(f"{obj.id} close_angle {obj.close_angle} is outside {lo}–{hi}")
        if obj.zone and obj.zone not in cfg.zones:
            errors.append(f"{obj.id} refers to unknown zone {obj.zone}")
    return errors
