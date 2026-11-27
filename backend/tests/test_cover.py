"""Cylinder cover: placement, sequence, descent, and stored jaw data."""

import struct
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.cover import (
    COVER_CENTER_XY_MM,
    COVER_PIVOT_MM,
    MOUTH_PLANE_Z_MM,
    apply_mouth_offset,
    cover_still_low,
    cover_tilt_deg,
    descent_block,
    descent_gate,
    print_bracket_to_design,
    print_cylinder_to_design,
)
from app.main import create_app
from app.planner import COVER_SEQUENCE
from test_arm import client_for, fast_config, wait_mission

ROOT = Path(__file__).resolve().parents[2]
STL = ROOT / "frontend" / "public" / "models" / "braccio" / "stl"
SHROUD = STL / "cylindrical-shroud.stl"
ASSEMBLY = STL / "passa-cylindrical-sampling-cover.stl"
CYLINDER = STL / "cover-cylinder.stl"
FIXED = STL / "cover-fixed.stl"
WRIST = STL / "braccio_wrist_roll.stl"


def _stl_vertices(path: Path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    offset = 84
    for _ in range(count):
        base = offset + 12
        for vertex in range(3):
            yield struct.unpack_from("<fff", data, base + vertex * 12)
        offset += 50


def test_the_shroud_mouth_returns_to_the_design_centre():
    mouth = []
    for x, y, z in _stl_vertices(SHROUD):
        design = print_cylinder_to_design(x, y, z)
        if design[2] < -68:
            mouth.append(design)
    assert mouth
    mean_x = sum(point[0] for point in mouth) / len(mouth)
    mean_y = sum(point[1] for point in mouth) / len(mouth)
    assert abs(mean_x - COVER_CENTER_XY_MM[0]) < 0.2
    assert abs(mean_y - COVER_CENTER_XY_MM[1]) < 0.2
    assert abs(min(point[2] for point in mouth) - MOUTH_PLANE_Z_MM) < 0.05
    assert COVER_PIVOT_MM == (-6.0, 40.5, 47.0)
    assert print_bracket_to_design(21, 12, 50.8) == (0.0, 40.5, 47.0)


def _round_mm(point):
    return tuple(round(value, 2) for value in point)


def test_the_twin_cylinder_matches_the_sketchup_stl():
    """The shown cover is the SketchUp assembly, already in the wrist frame."""
    cylinder = list(_stl_vertices(CYLINDER))
    mouth = [point for point in cylinder if point[2] < -68]
    mean_x = sum(point[0] for point in mouth) / len(mouth)
    mean_y = sum(point[1] for point in mouth) / len(mouth)
    assert abs(mean_x - 70) < 0.2
    assert abs(mean_y - 65) < 0.2
    tube = [point for point in cylinder if abs(point[0] - 70) < 8 and abs(point[1] - 65) < 8 and point[2] > 65]
    assert tube
    assert max(point[2] for point in cylinder) > 69
    shown = {_round_mm(point) for point in cylinder}
    shown.update(_round_mm(point) for point in _stl_vertices(FIXED))
    shown.update(_round_mm(point) for point in _stl_vertices(WRIST))
    source = {_round_mm(point) for point in _stl_vertices(ASSEMBLY)}
    assert shown == source


def test_servo_map_is_a_labelled_dry_run_assumption():
    cfg = fast_config()
    assert cfg.objects["orange_fruit"].close_angle == 42
    assert cfg.tool.angles_calibrated is False
    assert cfg.tool.dimensions_measured is False
    assert cover_tilt_deg(cfg.tool.sample_servo_deg, cfg.tool) == -38
    assert cover_tilt_deg(cfg.tool.park_servo_deg, cfg.tool) == -63
    assert cfg.joint_limits["gripper"] == (10, 73)
    pose = {"base": 90, "shoulder": 80, "elbow": 100, "wrist_vertical": 80, "wrist_rotation": 90}
    assert apply_mouth_offset(pose, cfg) == pose
    cfg.tool.mouth_shoulder_deg = 4
    assert apply_mouth_offset(pose, cfg)["shoulder"] == 84


def test_hardware_descent_names_the_missing_measurement_and_dry_run_does_not():
    tool = fast_config().tool
    block, note = descent_gate(tool, armed=True)
    assert block is not None
    assert "millimetres" in block
    assert "bounding box" in block.lower()
    dry_block, dry_note = descent_gate(tool, armed=False)
    assert dry_block is None
    assert "Dry-run size assumption" in dry_note
    assert "not calibrated" in note


def test_descent_stops_for_a_moved_or_uncertain_target_and_tilt_waits_for_clearance():
    assert descent_block([10, 10, 20, 20], [12, 11, 20, 20], 48, False) is None
    moved = descent_block([10, 10, 20, 20], [80, 10, 20, 20], 48, False)
    assert moved is not None and "moved" in moved
    lost = descent_block([10, 10, 20, 20], None, 48, False)
    assert lost is not None and "left the camera" in lost
    unsure = descent_block([10, 10, 20, 20], [10, 10, 20, 20], 48, True)
    assert unsure is not None and "uncertain" in unsure
    low = {"shoulder": 108, "elbow": 90}
    high = {"shoulder": 78, "elbow": 112}
    assert cover_still_low(low, low, high) is True
    assert cover_still_low(high, low, high) is False


def test_find_only_stops_before_the_cover_and_a_grab_request_stays_on_the_table():
    with client_for() as client:
        found = client.post("/api/command", json={"text": "Find the orange fruit."}).json()
        tools = [step["tool"] for step in found["state"]["mission"]["plan"]["steps"]]
        assert tools == ["scan", "inspect", "locate"]
        assert "sample_odor" not in tools
        grabbed = client.post(
            "/api/command",
            json={"text": "Grab the orange fruit and smell it."},
        ).json()
        plan = grabbed["state"]["mission"]["plan"]
        assert plan["targets"][0]["object_id"] == "orange_fruit"
        assert [step["tool"] for step in plan["steps"]] == list(COVER_SEQUENCE)
        assert "stays on the table" in plan["summary"]
        assert "will not be picked up" in plan["summary"]
        assert "grasp" not in [step["tool"] for step in plan["steps"]]


def test_a_stored_jaw_step_is_refused_and_the_object_is_not_carried():
    app = create_app(fast_config())
    with TestClient(app) as client:
        client.post("/api/command", json={"text": "Pick up the orange fruit."})
        mission = app.state.ctx.mission
        mission.plan["steps"] = [
            {
                "tool": "grasp",
                "label": "Grasp",
                "detail": "Grab the orange fruit.",
                "object_index": 0,
                "object_id": "orange_fruit",
            }
        ]
        client.post("/api/run")
        state = wait_mission(client)
        text = " ".join(item["message"] for item in state["mission"]["timeline"])
        assert state["mission"]["status"] == "blocked"
        assert "jaw gripper" in text
        assert "will not grasp" in text
        fruit = next(item for item in state["observation"]["items"] if item["id"] == "orange_fruit")
        assert fruit["held"] is False
        assert fruit["zone"] == "front"


def test_a_cover_run_leaves_the_object_on_the_table_and_writes_no_serial():
    with client_for() as client:
        client.post("/api/command", json={"text": "Grab the orange fruit and smell it."})
        client.post("/api/run")
        state = wait_mission(client, timeout=20)
        assert state["mission"]["status"] == "complete"
        assert state["robot"]["serial_connected"] is False
        fruit = next(item for item in state["observation"]["items"] if item["id"] == "orange_fruit")
        assert fruit["held"] is False
        assert fruit["zone"] == "front"
        text = " ".join(item["message"] for item in state["mission"]["timeline"])
        assert "Mock E-nose" in text
        assert "not a real measurement" in text
        assert "Simulated sampling" in text
        assert "servo stall is not used as contact" in text
        assert "Grab the orange fruit." not in text


def test_descent_stops_when_the_live_target_moves_and_does_not_park():
    app = create_app(fast_config())
    with TestClient(app) as client:
        vision = app.state.ctx.vision
        vision._stop.set()
        time.sleep(0.2)
        vision.capture("locate")
        item = next(obs for obs in vision.observations if obs.id == "orange_fruit")
        mission = app.state.ctx.mission
        bound = mission._bound_from_obs(item)
        bound["bbox"] = [100, 100, 40, 40]
        mission._bounds()[0] = bound
        item.bbox = [400, 300, 40, 40]
        outcome = mission._lower({"tool": "lower", "object_index": 0, "object_id": "orange_fruit"})
        assert outcome == "abort"
        text = " ".join(entry["message"] for entry in mission.timeline)
        assert "moved" in text
        assert "Tilt the cover" not in text
        assert "tilt" not in [entry["message"].lower() for entry in mission.timeline if "park angle" in entry["message"].lower()]
        assert client.get("/api/state").json()["robot"]["serial_connected"] is False


def test_armed_descent_reports_the_missing_size_without_moving():
    app = create_app(fast_config())
    with TestClient(app):
        mission = app.state.ctx.mission
        vision = app.state.ctx.vision
        vision.capture("locate")
        item = next(obs for obs in vision.observations if obs.id == "orange_fruit")
        mission._bounds()[0] = mission._bound_from_obs(item)
        mission.motion.mode = "armed"
        sent_before = len(mission.motion.command_log)
        outcome = mission._lower({"tool": "lower", "object_index": 0, "object_id": "orange_fruit"})
        assert outcome == "abort"
        text = " ".join(entry["message"] for entry in mission.timeline)
        assert "measured object diameter" in text
        assert len(mission.motion.command_log) == sent_before


def test_cover_angles_stay_editable_and_uncalibrated():
    with client_for() as client:
        updated = client.post("/api/settings", json={"cover_park_deg": 0, "cover_sample_deg": 90}).json()
        tool = updated["settings"]["tool"]
        assert tool["park_servo_deg"] == 10
        assert tool["sample_servo_deg"] == 73
        assert tool["angles_calibrated"] is False
        assert "Dry-run" in tool["assumption"]


def test_stop_during_the_cover_sequence_does_not_tilt_or_park():
    cfg = fast_config()
    cfg.sim_joint_s = 0.45
    with client_for(cfg) as client:
        client.post("/api/command", json={"text": "Grab the orange fruit and smell it."})
        client.post("/api/run")
        deadline = time.time() + 3
        while time.time() < deadline:
            state = client.get("/api/state").json()
            if state["robot"]["moving"] or state["robot"]["commands_sent"] > 0:
                break
            time.sleep(0.02)
        stopped = client.post("/api/stop").json()
        assert stopped["uart_sent"] is False
        assert stopped["software_stop"] is True
        time.sleep(0.3)
        later = client.get("/api/state").json()
        assert later["mission"]["status"] == "stopped"
        assert "tilt" not in later["mission"]["record"]["completed"]
        assert "park" not in later["mission"]["record"]["completed"]
        assert later["robot"]["serial_connected"] is False


def test_descent_stops_at_the_taught_covering_pose_and_the_hold_defaults_to_ten_seconds():
    from app.config import Config

    cfg = fast_config()
    assert Config.load().sample_hold_s == 10
    app = create_app(cfg)
    with TestClient(app):
        mission = app.state.ctx.mission
        mission.vision.capture("locate")
        item = next(obs for obs in mission.vision.observations if obs.id == "orange_fruit")
        bound = mission._bound_from_obs(item)
        pose = mission._arm_pose(bound, "lower")
        cover = cfg.zones["front"].cover
        assert cover is not None
        assert pose["base"] == 100
        assert pose["shoulder"] == cover.shoulder == 164
        assert pose["elbow"] == cover.elbow == 144
        assert pose["wrist_vertical"] == 0
        assert pose["wrist_rotation"] == 90
        hover = mission._arm_pose(bound, "approach")
        assert hover["base"] == 100
        assert hover["shoulder"] == 130
        assert hover["elbow"] == 138
        assert mission._arm_pose(bound, "align") == hover
        assert mission._arm_pose(bound, "raise") == hover
        bound["aim"] = {
            "approach": {"base": 70, "shoulder": 40, "elbow": 40, "wrist_vertical": 170, "wrist_rotation": 90},
            "grasp": {"base": 70, "shoulder": 50, "elbow": 50, "wrist_vertical": 160, "wrist_rotation": 90},
        }
        aimed_low = mission._arm_pose(bound, "lower")
        aimed_hover = mission._arm_pose(bound, "approach")
        assert aimed_low["base"] == aimed_hover["base"] == 80
        assert aimed_low["shoulder"] == 164 and aimed_low["elbow"] == 144
        assert aimed_hover["shoulder"] == 130 and aimed_hover["elbow"] == 138
    with client_for() as client:
        updated = client.post("/api/settings", json={"sample_hold_s": 12}).json()
        assert updated["settings"]["sample_hold_s"] == 12
        clamped = client.post("/api/settings", json={"sample_hold_s": 500}).json()
        assert clamped["settings"]["sample_hold_s"] == 120


def test_stop_during_the_cover_hold_cancels_the_timer_and_does_not_raise():
    cfg = fast_config()
    cfg.sample_hold_s = 2
    with client_for(cfg) as client:
        client.post("/api/command", json={"text": "Grab the orange fruit and smell it."})
        client.post("/api/run")
        saw = None
        deadline = time.time() + 8
        while time.time() < deadline:
            state = client.get("/api/state").json()
            if state["mission"].get("hold_remaining_s") is not None:
                saw = state
                break
            time.sleep(0.05)
        assert saw is not None
        assert saw["mission"]["hold_label"] == "Simulated sampling"
        assert 0 < saw["mission"]["hold_remaining_s"] <= 2
        client.post("/api/stop")
        later = client.get("/api/state").json()
        assert later["mission"]["status"] == "stopped"
        assert later["mission"]["hold_remaining_s"] is None
        assert "raise" not in later["mission"]["record"]["completed"]
        assert "park" not in later["mission"]["record"]["completed"]
        sent = later["robot"]["commands_sent"]
        time.sleep(0.4)
        again = client.get("/api/state").json()
        assert again["robot"]["commands_sent"] == sent
        assert again["robot"]["moving"] is False
