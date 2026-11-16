"""Joint limits, intermediate-pose checks, and the starter camera model.

Servo commands come from taught poses. A camera pixel is only a table
coordinate when calibration.complete is true, and even then it is snapped to a
taught zone. It is never turned straight into a joint angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import ARM_JOINTS, JOINTS, Config

class MotionRejected(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class TableFix:
    ok: bool
    x: float | None = None
    y: float | None = None
    reason: str = ""


def _round_deg(value: float) -> int:
    return int(round(value))


def limit_problem(joint: str, angle: float, cfg: Config) -> str | None:
    lo, hi = cfg.joint_limits[joint]
    if angle < lo or angle > hi:
        return f"{joint} {angle:g}° is outside {lo:g}–{hi:g}°"
    return None


def pose_problem(arm: dict, cfg: Config, gripper: float) -> str | None:
    for joint in ARM_JOINTS:
        problem = limit_problem(joint, arm[joint], cfg)
        if problem:
            return problem
    problem = limit_problem("gripper", gripper, cfg)
    if problem:
        return problem
    if (
        arm["elbow"] < cfg.forbidden_elbow_below
        and arm["shoulder"] > cfg.forbidden_shoulder_above
    ):
        return (
            "that pose drops the shoulder while the elbow is folded in, "
            "which is outside the starter clearance rule"
        )
    return None


def _shoulder_clears_base(shoulder: float, cfg: Config) -> bool:
    return cfg.clearance_shoulder_min <= shoulder <= cfg.clearance_shoulder_max


def compile_joint_script(
    current: dict[str, float],
    target_arm: dict[str, float],
    gripper: float,
    cfg: Config,
) -> list[tuple[str, int]]:
    """Validate the whole single-joint path before any serial command is sent.

    The firmware moves one joint per command. The script raises the shoulder
    into the clearance band before rotating the base, opens the gripper before
    a swing, and closes it only after the arm joints have arrived.
    """
    if any(name not in current or current[name] is None for name in JOINTS):
        raise MotionRejected("Commanded joint angles are unknown. Sync status before moving.")

    rounded_arm = {joint: _round_deg(target_arm[joint]) for joint in ARM_JOINTS}
    gripper_i = _round_deg(gripper)
    problem = pose_problem(rounded_arm, cfg, gripper_i)
    if problem:
        raise MotionRejected(problem)

    pose = {joint: _round_deg(current[joint]) for joint in JOINTS}
    script: list[tuple[str, int]] = []

    def commit(joint: str, angle: int) -> None:
        if pose[joint] == angle:
            return
        nxt = dict(pose)
        nxt[joint] = angle
        problem = pose_problem(nxt, cfg, nxt["gripper"])
        if problem:
            raise MotionRejected(f"Intermediate pose after {joint} → {angle}°: {problem}")
        pose[joint] = angle
        script.append((joint, angle))

    opening = gripper_i < pose["gripper"]
    if opening:
        commit("gripper", gripper_i)

    base_delta = abs(rounded_arm["base"] - pose["base"])
    if base_delta > 0 and not _shoulder_clears_base(pose["shoulder"], cfg):
        transit = _round_deg(cfg.transit_shoulder)
        commit("shoulder", transit)
        if not _shoulder_clears_base(pose["shoulder"], cfg):
            raise MotionRejected(
                "The shoulder cannot reach a clearance angle before the base rotates."
            )

    for joint in ("base", "elbow", "wrist_vertical", "wrist_rotation", "shoulder"):
        commit(joint, rounded_arm[joint])
    if not opening:
        commit("gripper", gripper_i)

    final_problem = pose_problem(pose, cfg, pose["gripper"])
    if final_problem:
        raise MotionRejected(final_problem)
    return script


def jog_target(current: dict[str, float], joint: str, delta: float, cfg: Config) -> int:
    if joint not in JOINTS:
        raise MotionRejected(f"{joint} is not a joint on this arm.")
    if abs(delta) > 15:
        raise MotionRejected("Jog steps are limited to 15°.")
    if any(current.get(name) is None for name in JOINTS):
        raise MotionRejected("Commanded joint angles are unknown. Sync status before jogging.")
    angle = _round_deg(current[joint] + delta)
    problem = limit_problem(joint, angle, cfg)
    if problem:
        raise MotionRejected(problem)
    if joint == "base" and angle != _round_deg(current["base"]):
        if not _shoulder_clears_base(current["shoulder"], cfg):
            raise MotionRejected(
                "Raise the shoulder into the clearance band before jogging the base."
            )
    trial = {name: _round_deg(current[name]) for name in JOINTS}
    trial[joint] = angle
    problem = pose_problem(trial, cfg, trial["gripper"])
    if problem:
        raise MotionRejected(problem)
    return angle


def aim_from_bbox(bbox: list[float], cfg: Config) -> dict[str, dict[str, float]]:
    """Point the arm at a box in the image. Angles stay inside the joint limits.

    Left and right in the photo turn the base. Higher in the photo is farther
    from the arm, so the shoulder and elbow reach out. Lower in the photo is
    nearer, so the arm folds in. Tall standing cups are aimed on the mid-body
    with the wrist hanging down so the gripper can close around the glass.
    """
    u, v, w, h = (list(bbox) + [0, 0, 80, 80])[:4]
    width = float(cfg.calibration.get("image_width") or 640) or 640
    height = float(cfg.calibration.get("image_height") or 480) or 480
    across = min(1.0, max(0.0, (float(u) + float(w) / 2) / width))
    # Tall boxes (cups/glasses): aim above the geometric center so the gripper
    # closes around the body instead of scraping the table at the bbox bottom.
    tall = float(h) > float(w) * 1.25
    aim_down = float(v) + float(h) * (0.38 if tall else 0.5)
    down = min(1.0, max(0.0, aim_down / height))
    depth = 1.0 - down
    base_lo, base_hi = cfg.joint_limits["base"]
    # FK: base 90° faces +X, base > 90 turns toward +Y (robot left).
    # Image left (across → 0) must use the high base end so the arm turns left.
    # Stay in the Begin fold (elbow near 180, wrist hanging). Only the shoulder
    # leans forward to the object. Opening the elbow is what points the arm at the sky.
    grasp = {
        "base": base_hi - across * (base_hi - base_lo),
        "shoulder": 115 - depth * 25,
        "elbow": 175 - depth * 15,
        "wrist_vertical": 170,
        "wrist_rotation": 90,
    }
    approach = dict(grasp)
    approach["shoulder"] = max(40.0, grasp["shoulder"] - 12)
    approach["elbow"] = min(180.0, grasp["elbow"] + 5)
    approach["wrist_vertical"] = 160
    lift = dict(grasp)
    lift["shoulder"] = max(40.0, grasp["shoulder"] - 18)
    lift["elbow"] = 180
    lift["wrist_vertical"] = 165
    return {
        "approach": {joint: _clamp_joint(joint, angle, cfg) for joint, angle in approach.items()},
        "grasp": {joint: _clamp_joint(joint, angle, cfg) for joint, angle in grasp.items()},
        "lift": {joint: _clamp_joint(joint, angle, cfg) for joint, angle in lift.items()},
    }


def grip_for_bbox(bbox: list[float], cfg: Config) -> float:
    """Close farther on a narrow object and stay more open on a wide one."""
    width = float((list(bbox) + [0, 0, 70])[2] or 70)
    span = cfg.gripper_closed - cfg.gripper_open
    wide = min(1.0, max(0.0, (width - 40.0) / 200.0))
    return cfg.gripper_closed - wide * span * 0.7


def _clamp_joint(joint: str, angle: float, cfg: Config) -> float:
    lo, hi = cfg.joint_limits[joint]
    return max(lo, min(hi, angle))


def in_workspace(x: float, y: float, cfg: Config) -> bool:
    return cfg.x_min <= x <= cfg.x_max and cfg.y_min <= y <= cfg.y_max


def classify_table_point(x: float, y: float, cfg: Config, zone_id: str | None = None):
    """Return (zone, reason). zone is None when the point must not be picked."""
    if zone_id:
        zone = cfg.zones.get(zone_id)
        if zone is None:
            return None, f"There is no taught zone named {zone_id}."
        if not zone.reachable or not in_workspace(zone.x, zone.y, cfg):
            return None, (
                f"The target is outside the validated pickup area "
                f"(table x={zone.x:g} mm, y={zone.y:g} mm; "
                f"workspace x {cfg.x_min:g}–{cfg.x_max:g} mm, "
                f"y {cfg.y_min:g}–{cfg.y_max:g} mm)."
            )
        return zone, ""
    if not in_workspace(x, y, cfg):
        return None, (
            f"The target is outside the validated pickup area "
            f"(table x={x:g} mm, y={y:g} mm; "
            f"workspace x {cfg.x_min:g}–{cfg.x_max:g} mm, "
            f"y {cfg.y_min:g}–{cfg.y_max:g} mm)."
        )
    best = None
    best_d = cfg.zone_radius_mm
    for zone in cfg.zones.values():
        if not zone.reachable:
            continue
        dist = math.hypot(zone.x - x, zone.y - y)
        if dist <= best_d:
            best = zone
            best_d = dist
    if best is None:
        return None, (
            "The table coordinate does not fall on a taught pickup pose. "
            "Jog a pose for that spot and add it to the workspace before picking."
        )
    return best, ""


def _rot_x(deg: float) -> list[list[float]]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_y(deg: float) -> list[list[float]]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rot_z(deg: float) -> list[list[float]]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _matvec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def _add(a, b):
    return [a[i] + b[i] for i in range(3)]


def forward_wrist(arm: dict, calibration: dict) -> tuple[list[float], list[list[float]]]:
    """Starter forward kinematics. Replace link lengths with measured ones.

    Convention: base 90° faces +X, shoulder 90° holds the upper arm horizontal,
    and the elbow servo is the absolute pitch of the forearm (90° horizontal).
    Positive pitch is below the horizontal.
    """
    links = calibration["links"]
    yaw = math.radians(arm["base"] - 90)
    sh = math.radians(arm["shoulder"] - 90)
    el = math.radians(arm["elbow"] - 90)
    x1 = links["shoulder"] * math.cos(sh)
    z1 = links["base_height"] - links["shoulder"] * math.sin(sh)
    x2 = x1 + links["elbow"] * math.cos(el)
    z2 = z1 - links["elbow"] * math.sin(el)
    x3 = x2 + links["wrist"] * math.cos(el)
    z3 = z2 - links["wrist"] * math.sin(el)
    radial = [x3, 0.0, z3]
    yaw_m = _rot_z(math.degrees(yaw))
    origin = _matvec(yaw_m, radial)
    # Wrist axes: x along the forearm, z up when the forearm is horizontal.
    forearm = _matmul(yaw_m, _rot_y(math.degrees(el)))
    return origin, forearm


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def pixel_to_table(u: float, v: float, arm: dict, cfg: Config) -> TableFix:
    """Intersect a camera ray with the table plane. Pixels alone are not enough."""
    cal = cfg.calibration
    if not cal.get("complete"):
        return TableFix(
            ok=False,
            reason=(
                "The wrist camera is not calibrated. An image coordinate cannot "
                "be turned into a table position or a servo command."
            ),
        )
    intr = cal["intrinsics"]
    mount = cal["camera_in_wrist"]
    pitch = math.radians(mount.get("pitch", 0))
    # Wrist frame: +X forward, +Y left, +Z up. Pitch 0 looks forward; pitch
    # increases downward. Image +X is to the right, image +Y is down.
    forward_w = [math.cos(pitch), 0.0, -math.sin(pitch)]
    right_w = [0.0, -1.0, 0.0]
    down_w = _cross(forward_w, right_w)
    du = (u - intr["cx"]) / intr["fx"]
    dv = (v - intr["cy"]) / intr["fy"]
    direction_w = [
        du * right_w[i] + dv * down_w[i] + forward_w[i]
        for i in range(3)
    ]
    origin, wrist_in_world = forward_wrist(arm, cal)
    offset_w = [mount.get("x", 0.0), mount.get("y", 0.0), mount.get("z", 0.0)]
    cam_origin = _add(origin, _matvec(wrist_in_world, offset_w))
    direction_world = _matvec(wrist_in_world, direction_w)
    dz = direction_world[2]
    if abs(dz) < 1e-6:
        return TableFix(ok=False, reason="The camera ray is parallel to the table.")
    table_z = float(cal.get("table_z_mm", 0))
    scale = (table_z - cam_origin[2]) / dz
    if scale <= 0:
        return TableFix(ok=False, reason="The camera ray does not meet the table in front of the lens.")
    hit = _add(cam_origin, [scale * c for c in direction_world])
    if math.hypot(hit[0], hit[1]) > float(cal.get("max_range_mm", 800)):
        return TableFix(
            ok=False,
            reason="The table intersection is beyond the distance the calibration trusts.",
        )
    return TableFix(ok=True, x=hit[0], y=hit[1])
