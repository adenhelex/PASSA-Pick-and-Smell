"""Cylindrical sampling cover.

Design millimetres are the braccio_wrist_roll.stl vertex frame. The twin
applies one scale of 0.001 and the wrist visual origin (0.006, 0, 0) m.
Print STLs are rotated into a print pose and must be restored first.

Park servo 73° and sampling servo 10° are dry-run assumptions:
servo 10° is design tilt -38° (opening down, rim level on the table) and
servo 73° is design tilt -63° (cup sideways). The horn direction has not been measured.
"""

from __future__ import annotations

import math

from app.config import CoverTool, Config
from app.geometry import _clamp_pose, image_table_point

# Design assembly, millimetres, wrist-mesh vertex frame.
COVER_PIVOT_MM = (-6.0, 40.5, 47.0)
COVER_CENTER_XY_MM = (70.0, 65.0)
MOUTH_PLANE_Z_MM = -70.0
ROOF_PLANE_Z_MM = 30.0
TILT_AXIS = "y"
JAW_TOOLS = frozenset({"grasp", "lift", "place"})
# Bench motion: reach out with the cup still sideways, level the opening,
# then set the rim flat on the table. Tuned on the twin's joint chain.
# The rim sits 10° off the wrist centerline, so the base is shifted to put
# the cup, not the wrist, on the target heading.
SEAL_BASE_OFFSET = 10.0
COVER_HOVER_JOINTS = {
    "shoulder": 130.0,
    "elbow": 138.0,
    "wrist_vertical": 40.0,
    "wrist_rotation": 90.0,
}
COVER_LOW_JOINTS = {
    "shoulder": 165.0,
    "elbow": 140.0,
    "wrist_vertical": 3.0,
    "wrist_rotation": 90.0,
}


def vertical_cover_pose(base: float, low: bool) -> dict:
    """Arm pose for the sealed cover. Only the base heading changes per target."""
    shape = COVER_LOW_JOINTS if low else COVER_HOVER_JOINTS
    return {"base": float(base) + SEAL_BASE_OFFSET, **shape}


def print_cylinder_to_design(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Print-orientation shroud vertex to the design assembly, in millimetres."""
    return (x - 21.0, 124.0 - y, 62.0 - z)


def print_bracket_to_design(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Print-orientation bracket vertex to the design assembly, in millimetres."""
    return (x - 21.0, z - 10.3, 59.0 - y)


def cover_tilt_deg(servo: float, tool: CoverTool) -> float:
    """Map an M6 command to the design tilt about the wrist Y axis."""
    span = tool.sample_servo_deg - tool.park_servo_deg
    if abs(span) < 1e-6:
        return tool.design_tilt_at_park_deg
    share = (servo - tool.park_servo_deg) / span
    return tool.design_tilt_at_park_deg + share * (tool.design_tilt_at_sample_deg - tool.design_tilt_at_park_deg)


def apply_mouth_offset(pose: dict, cfg: Config) -> dict:
    """Shift a pose so the cylinder mouth, not the old jaw, is the aim point.

    The default offset is zero. It is not a measured calibration.
    """
    tool = cfg.tool
    shifted = {
        "base": float(pose["base"]) + tool.mouth_base_deg,
        "shoulder": float(pose["shoulder"]) + tool.mouth_shoulder_deg,
        "elbow": float(pose["elbow"]) + tool.mouth_elbow_deg,
        "wrist_vertical": float(pose["wrist_vertical"]) + tool.mouth_wrist_vertical_deg,
        "wrist_rotation": float(pose["wrist_rotation"]) + tool.mouth_wrist_rotation_deg,
    }
    return _clamp_pose(shifted, cfg)


def assumption_note(tool: CoverTool) -> str:
    return (
        f"Dry-run size assumption: object diameter {tool.assumed_object_diameter_mm:g} mm, "
        f"height {tool.assumed_object_height_mm:g} mm, placement uncertainty {tool.placement_uncertainty_mm:g} mm. "
        f"Approach clearance {tool.approach_clearance_mm:g} mm and seal stand-off {tool.seal_standoff_mm:g} mm are not measured. "
        f"Mouth joint offset is {tool.mouth_base_deg:g}/{tool.mouth_shoulder_deg:g}/{tool.mouth_elbow_deg:g}/"
        f"{tool.mouth_wrist_vertical_deg:g}/{tool.mouth_wrist_rotation_deg:g} degrees and is not calibrated. "
        f"Sampling servo {tool.sample_servo_deg:g}° is design tilt {tool.design_tilt_at_sample_deg:g}° (mouth down). "
        f"Park servo {tool.park_servo_deg:g}° is design tilt {tool.design_tilt_at_park_deg:g}°. "
        "Those servo angles are dry-run assumptions. Bounding-box pixels are not millimetres. "
        "The CAD sweep is reference only and does not clear the physical arm."
    )


def descent_gate(tool: CoverTool, armed: bool) -> tuple[str | None, str]:
    """Return (block reason, assumption note). A block stops a hardware descent."""
    needed = tool.assumed_object_diameter_mm / 2.0 + tool.placement_uncertainty_mm
    inside = tool.inside_diameter_mm / 2.0
    problems: list[str] = []
    if needed > inside + 1e-6:
        problems.append(
            f"the assumed object diameter {tool.assumed_object_diameter_mm:g} mm plus "
            f"{tool.placement_uncertainty_mm:g} mm placement uncertainty does not fit the "
            f"{tool.inside_diameter_mm:g} mm inside diameter"
        )
    if tool.assumed_object_height_mm > tool.clear_height_mm + 1e-6:
        problems.append(
            f"the assumed object height {tool.assumed_object_height_mm:g} mm exceeds the "
            f"cover's {tool.clear_height_mm:g} mm clear height"
        )
    note = assumption_note(tool)
    if problems:
        return (
            "The cover will not lower: " + "; ".join(problems) + ".",
            note,
        )
    if tool.dimensions_measured and tool.angles_calibrated:
        return None, "Object size and cover angles are marked as measured."
    # The rim stops at the taught covering pose. Object size is not the table height.
    return None, note


def descent_block(start_bbox, live_bbox, tolerance_px: float, uncertain: bool) -> str | None:
    """Stop a descent that was using one frozen pose when the target is no longer that target."""
    if uncertain:
        return "The target became uncertain during descent, so the cover stopped."
    if not start_bbox or len(list(start_bbox)) < 4:
        return None
    if not live_bbox or len(list(live_bbox)) < 4:
        return "The target left the camera during descent, so the cover stopped."
    start = list(start_bbox)
    live = list(live_bbox)
    su = float(start[0]) + float(start[2]) / 2.0
    sv = float(start[1]) + float(start[3]) / 2.0
    lu = float(live[0]) + float(live[2]) / 2.0
    lv = float(live[1]) + float(live[3]) / 2.0
    drift = math.hypot(su - lu, sv - lv)
    if drift > float(tolerance_px):
        return (
            f"The target moved {drift:.0f} px during descent, past the {float(tolerance_px):.0f} px tolerance, "
            "so the cover stopped."
        )
    return None


def cover_still_low(current: dict, low: dict, high: dict) -> bool:
    """True when the arm is still at the sampling height rather than the raised pose."""
    if any(current.get(joint) is None for joint in ("shoulder", "elbow")):
        return True
    d_low = abs(float(current["shoulder"]) - float(low["shoulder"])) + abs(float(current["elbow"]) - float(low["elbow"]))
    d_high = abs(float(current["shoulder"]) - float(high["shoulder"])) + abs(float(current["elbow"]) - float(high["elbow"]))
    return d_low + 4 < d_high


def tilt_reference(tool: CoverTool) -> str:
    return (
        f"Tilting the cover from design {tool.design_tilt_at_sample_deg:g}° toward {tool.design_tilt_at_park_deg:g}° "
        f"about the M6 pivot ({tool.pivot_x_mm:g}, {tool.pivot_y_mm:g}, {tool.pivot_z_mm:g}) mm, axis {tool.tilt_axis}. "
        "The CAD check found no intersection above 0.001 mm³ against the bracket, wrist plate, and nominal servo. "
        "That check excludes the rest of the arm, camera, cables, fasteners, and table. "
        "Physical clearance and the real servo angles still need calibration."
    )


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _matvec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _rot(axis: str, deg: float):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _transform(point, rotation, origin):
    turned = _matvec(rotation, point)
    return (turned[0] + origin[0], turned[1] + origin[1], turned[2] + origin[2])


def _z_up():
    """Three.js Euler XYZ (90°, 180°, 0°): URDF Z-up into the twin's table frame."""
    x, y = math.pi / 2, math.pi
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    return ((cy, 0.0, sy), (sx * sy, cx, -sx * cy), (-cx * sy, sx, cx * cy))


_Z_UP = _z_up()
_REACH_GRID: list[tuple] | None = None


def _mouth_metrics(shoulder: float, elbow: float, wrist: float, tilt: float = -38.0):
    """Mouth center radius, lowest rim height, rim span, and opening direction.

    Opening direction is the table-frame Z of the mouth axis. -1 faces the table.
    """
    cx, cy, z = COVER_CENTER_XY_MM[0], COVER_CENTER_XY_MM[1], MOUTH_PLANE_Z_MM
    samples = [(cx, cy, z), (cx, cy, z - 50.0)]
    for step in range(4):
        angle = step * math.pi / 2
        samples.append((cx + 55.0 * math.cos(angle), cy + 55.0 * math.sin(angle), z))
    placed = []
    pivot = (COVER_PIVOT_MM[0] / 1000.0, COVER_PIVOT_MM[1] / 1000.0, COVER_PIVOT_MM[2] / 1000.0)
    for x, y, zz in samples:
        point = ((x - COVER_PIVOT_MM[0]) / 1000.0, (y - COVER_PIVOT_MM[1]) / 1000.0, (zz - COVER_PIVOT_MM[2]) / 1000.0)
        point = _transform(point, _rot("y", tilt), pivot)
        point = _transform(point, _rot("x", 0), (0.006, 0.0, 0.0))
        point = _transform(point, _rot("z", -90), (0.0, 0.0, 0.06))
        point = _transform(point, _matmul(_rot("x", -90), _rot("x", wrist)), (0.0, 0.0, 0.125))
        point = _transform(point, _matmul(_rot("x", -90), _rot("x", elbow)), (0.0, 0.0, 0.125))
        point = _transform(point, _matmul(_rot("x", -90), _rot("x", shoulder)), (0.0, -0.002, 0.072))
        scene = _matvec(_Z_UP, point)
        placed.append((scene[0] * 1000.0, scene[2] * 1000.0, scene[1] * 1000.0))
    center, tip, *ring = placed
    opening = tuple(tip[i] - center[i] for i in range(3))
    norm = math.sqrt(sum(value * value for value in opening)) or 1.0
    heights = [point[2] for point in (center, *ring)]
    return (
        math.hypot(center[0], center[1]),
        min(heights),
        max(heights) - min(heights),
        opening[2] / norm,
    )


def mouth_reach_mm(tool: CoverTool | None = None) -> float:
    """Distance from the base to the mouth center at the fully extended seal.

    Same joint chain as the twin. The shoulder is on its upper limit there.
    Closer targets use other joint angles inside the same limits.
    """
    tilt = -38.0 if tool is None else tool.design_tilt_at_sample_deg
    radius, *_rest = _mouth_metrics(
        COVER_LOW_JOINTS["shoulder"],
        COVER_LOW_JOINTS["elbow"],
        COVER_LOW_JOINTS["wrist_vertical"],
        tilt,
    )
    return radius


def _reach_grid() -> list[tuple]:
    """Legal sample-tilt poses whose opening still faces mostly downward."""
    global _REACH_GRID
    if _REACH_GRID is not None:
        return _REACH_GRID
    rows = []
    samples = [
        (shoulder, elbow, wrist)
        for shoulder in range(35, 166, 5)
        for elbow in range(40, 181, 5)
        for wrist in range(0, 181, 5)
    ]
    samples.append((
        int(COVER_LOW_JOINTS["shoulder"]),
        int(COVER_LOW_JOINTS["elbow"]),
        int(COVER_LOW_JOINTS["wrist_vertical"]),
    ))
    for shoulder, elbow, wrist in samples:
        if elbow < 35 and shoulder > 140:
            continue
        radius, low, span, down = _mouth_metrics(shoulder, elbow, wrist)
        if low < -3 or down > -0.55:
            continue
        rows.append((radius, low, span, down, float(shoulder), float(elbow), float(wrist)))
    _REACH_GRID = rows
    return rows


def joints_for_reach(distance_mm: float, low: bool) -> dict[str, float]:
    """Shoulder, elbow, and wrist that put the mouth near this distance.

    The low pose sets the rim against the table. The hover pose holds it about
    110 mm up at the same distance. Both stay inside the joint limits.
    """
    goal_height = 0.0 if low else 110.0
    best = None
    for radius, height, span, down, shoulder, elbow, wrist in _reach_grid():
        score = (
            abs(radius - distance_mm) * 4
            + abs(height - goal_height) * 1.2
            + span * 0.08
            + (down + 1) * 90
        )
        if height < -1:
            score += 30 * (-1 - height)
        if best is None or score < best[0]:
            best = (score, shoulder, elbow, wrist)
    _score, shoulder, elbow, wrist = best
    return {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist_vertical": wrist,
        "wrist_rotation": COVER_LOW_JOINTS["wrist_rotation"],
    }


def cover_pose_at_distance(base: float, distance_mm: float, low: bool, cfg: Config) -> dict:
    """Cover pose for one distance. Only the base heading is shared with the target."""
    pose = {"base": float(base) + SEAL_BASE_OFFSET, **joints_for_reach(distance_mm, low)}
    return apply_mouth_offset(pose, cfg)


def cover_reach_mm(tool: CoverTool) -> float:
    """Farthest object center the mouth can still sit over."""
    return mouth_reach_mm(tool) + tool.inside_diameter_mm / 2.0


def target_point_mm(bound: dict, cfg: Config) -> tuple[float, float] | None:
    if bound.get("localizable") and bound.get("table_x") is not None and bound.get("table_y") is not None:
        return float(bound["table_x"]), float(bound["table_y"])
    bbox = bound.get("bbox")
    if bbox and len(bbox) >= 4:
        return image_table_point(bbox, cfg)
    zone = cfg.zones.get(bound.get("zone") or "")
    if zone is not None:
        return float(zone.x), float(zone.y)
    return None


def target_reach_problem(bound: dict, cfg: Config) -> str | None:
    """Tell the user when the object is outside the cover's reach. None when it can seal."""
    point = target_point_mm(bound, cfg)
    if point is None:
        return None
    distance = math.hypot(point[0], point[1])
    tool = cfg.tool
    far_limit = cover_reach_mm(tool)
    name = bound.get("name") or "object"
    if distance > far_limit:
        return (
            f"The {name} is too far for the arm. "
            f"It is about {distance:.0f} mm from the base, and the cover only reaches about {far_limit:.0f} mm. "
            "I did not move the cover."
        )
    return None


def jaw_refusal(tool: str) -> str:
    return (
        f"This plan still asks the jaw gripper to {str(tool).replace('_', ' ')}. "
        "Cylinder mode will not grasp, lift, or put the object back. "
        "Send the request again so a cover sequence is planned."
    )


def tool_export(cfg: Config, armed: bool) -> dict:
    tool = cfg.tool
    block, note = descent_gate(tool, armed)
    return {
        "name": tool.name,
        "inside_diameter_mm": tool.inside_diameter_mm,
        "clear_height_mm": tool.clear_height_mm,
        "body_outside_diameter_mm": tool.body_outside_diameter_mm,
        "flange_outside_diameter_mm": tool.flange_outside_diameter_mm,
        "wall_thickness_mm": tool.wall_thickness_mm,
        "roof_thickness_mm": tool.roof_thickness_mm,
        "sample_bore_mm": tool.sample_bore_mm,
        "vent_diameter_mm": tool.vent_diameter_mm,
        "pivot_mm": [tool.pivot_x_mm, tool.pivot_y_mm, tool.pivot_z_mm],
        "tilt_axis": tool.tilt_axis,
        "park_servo_deg": tool.park_servo_deg,
        "sample_servo_deg": tool.sample_servo_deg,
        "design_tilt_at_park_deg": tool.design_tilt_at_park_deg,
        "design_tilt_at_sample_deg": tool.design_tilt_at_sample_deg,
        "angles_calibrated": tool.angles_calibrated,
        "dimensions_measured": tool.dimensions_measured,
        "approach_clearance_mm": tool.approach_clearance_mm,
        "seal_standoff_mm": tool.seal_standoff_mm,
        "assumed_object_diameter_mm": tool.assumed_object_diameter_mm,
        "assumed_object_height_mm": tool.assumed_object_height_mm,
        "placement_uncertainty_mm": tool.placement_uncertainty_mm,
        "descent_tolerance_px": tool.descent_tolerance_px,
        "mouth_offset_deg": {
            "base": tool.mouth_base_deg,
            "shoulder": tool.mouth_shoulder_deg,
            "elbow": tool.mouth_elbow_deg,
            "wrist_vertical": tool.mouth_wrist_vertical_deg,
            "wrist_rotation": tool.mouth_wrist_rotation_deg,
        },
        "assumption": note,
        "descent_block": block or "",
    }

