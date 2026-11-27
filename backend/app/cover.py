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
from app.geometry import _clamp_pose

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
    "shoulder": 164.0,
    "elbow": 144.0,
    "wrist_vertical": 0.0,
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
    if tool.dimensions_measured and tool.angles_calibrated and not problems:
        return None, "Object size and cover angles are marked as measured."
    if armed:
        missing = []
        if not tool.dimensions_measured:
            missing.append("a measured object diameter and height in millimetres")
        if not tool.angles_calibrated:
            missing.append("calibrated park and sampling cover angles")
        missing.extend(problems)
        return (
            "Hardware descent needs "
            + "; ".join(missing)
            + ". The camera bounding box does not provide object size or table height.",
            note,
        )
    if problems:
        return "Dry-run will not lower the cover: " + "; ".join(problems) + ".", note
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

