import time

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.geometry import MotionRejected, compile_joint_script, forward_wrist, pixel_to_table
from app.main import create_app
from app.motion import FirmwareSim, MotionController


def fast_config() -> Config:
    cfg = Config.load()
    cfg.sim_joint_s = 0.01
    cfg.enose_step_s = 0.01
    cfg.sample_hold_s = 0.01
    cfg.motion_timeout_s = 2
    cfg.llm_enabled = False
    return cfg


def client_for(cfg: Config | None = None) -> TestClient:
    return TestClient(create_app(cfg or fast_config()))


def wait_mission(client: TestClient, timeout: float = 12):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get("/api/state").json()
        mission = last["mission"]
        if not mission["running"] and mission["status"] in {"complete", "blocked", "stopped"}:
            return last
        time.sleep(0.03)
    raise AssertionError(last["mission"] if last else "no state")


def test_taught_poses_fit_the_starter_rules():
    cfg = Config.load()
    assert cfg.pose_errors == []


def test_joint_limits_and_base_clearance():
    cfg = Config.load()
    current = {**cfg.park.angles(), "gripper": 10, "shoulder": 150}
    script = compile_joint_script(
        current,
        {"base": 120, "shoulder": 80, "elbow": 100, "wrist_vertical": 90, "wrist_rotation": 90},
        10,
        cfg,
    )
    assert script[0] == ("shoulder", 75)
    assert ("base", 120) in script
    assert script.index(("shoulder", 75)) < script.index(("base", 120))
    with pytest.raises(MotionRejected):
        compile_joint_script(current, cfg.park.angles(), 5, cfg)
    with pytest.raises(MotionRejected):
        compile_joint_script(
            {**cfg.park.angles(), "gripper": 10, "shoulder": 150},
            {"base": 90, "shoulder": 80, "elbow": 20, "wrist_vertical": 90, "wrist_rotation": 90},
            10,
            cfg,
        )


def test_image_coordinate_without_calibration_is_not_a_pose():
    cfg = Config.load()
    fix = pixel_to_table(320, 240, cfg.park.angles(), cfg)
    assert fix.ok is False
    assert "not calibrated" in fix.reason
    origin, _rotation = forward_wrist(
        {"base": 90, "shoulder": 90, "elbow": 90, "wrist_vertical": 90, "wrist_rotation": 90},
        cfg.calibration,
    )
    assert origin[0] == pytest.approx(310)
    assert origin[1] == pytest.approx(0)
    assert origin[2] == pytest.approx(70)
    cfg.calibration["complete"] = True
    projected = pixel_to_table(320, 240, cfg.park.angles(), cfg)
    assert projected.ok is True
    assert projected.x is not None and projected.y is not None


def test_firmware_noop_is_immediate_done():
    sim = FirmwareSim(move_s=0.05)
    try:
        assert b"READY" in sim.readline()
        sim.submit("gripper rotate 10")
        assert sim.readline() == b"DONE\n"
        sim.submit("base rotate 100")
        assert sim.readline() == b"OK moving\n"
        assert sim.readline() == b"DONE\n"
    finally:
        sim.close()


def test_stop_acks_and_timeout_is_not_replayed():
    cfg = fast_config()
    cfg.sim_joint_s = 0.6
    cfg.motion_timeout_s = 0.4
    cfg.mode = "armed"
    ctrl = MotionController(cfg)
    ctrl.start()
    try:
        ctrl.connect("mock://braccio", confirm_reset=True)
        assert ctrl.export()["angles"]["base"] == 90
        ctrl._link.sim.stall_next = True
        stalled = ctrl.jog("elbow", 10)
        assert stalled.uncertain
        assert ctrl.command_log.count("elbow rotate 100") == 1
        refused = ctrl.jog("elbow", 10)
        assert refused.ok is False
        assert ctrl.command_log.count("elbow rotate 100") == 1
        ctrl.stop()
        assert ctrl.command_log.count("elbow rotate 100") == 1
        synced = ctrl.sync_status()
        assert synced.ok

        worker = __import__("threading").Thread(target=lambda: ctrl.jog("base", 15))
        worker.start()
        deadline = time.time() + 2
        while "base rotate 105" not in ctrl.command_log and time.time() < deadline:
            time.sleep(0.01)
        stopped = ctrl.stop()
        worker.join(timeout=2)
        assert stopped.uart_sent is True
        assert stopped.acknowledged is True
        assert "acknowledged" in stopped.detail
        assert ctrl.command_log.count("base rotate 105") == 1
        assert ctrl.export()["joints_known"] is False
        synced_after = ctrl.sync_status()
        assert synced_after.ok
        assert ctrl.export()["angles"]["base"] == 90
    finally:
        ctrl.shutdown()


def test_lost_serial_stop_is_not_confirmed():
    cfg = fast_config()
    cfg.sim_joint_s = 0.8
    cfg.mode = "armed"
    ctrl = MotionController(cfg)
    ctrl.start()
    try:
        ctrl.connect("mock://braccio", confirm_reset=True)
        worker = __import__("threading").Thread(target=lambda: ctrl.jog("base", 10))
        worker.start()
        deadline = time.time() + 2
        while not any(line.startswith("base rotate") for line in ctrl.command_log) and time.time() < deadline:
            time.sleep(0.01)
        ctrl._link.close()
        stopped = ctrl.stop()
        worker.join(timeout=2)
        assert stopped.acknowledged is False
        assert "could not be confirmed" in stopped.detail
        assert ctrl.export()["joints_known"] is False
    finally:
        ctrl.shutdown()


def test_dry_run_pose_advances_at_about_30hz():
    cfg = fast_config()
    cfg.sim_joint_s = 0.2
    cfg.pose_hz = 30
    ctrl = MotionController(cfg)
    ctrl.start()
    seen: list[float] = []

    def watch() -> None:
        last = None
        deadline = time.time() + 1
        while time.time() < deadline:
            value = ctrl.export()["angles"]["base"]
            if value != last:
                seen.append(float(value))
                last = value
            time.sleep(0.005)

    worker = __import__("threading").Thread(target=watch)
    worker.start()
    try:
        result = ctrl.jog("base", 15)
        assert result.ok
    finally:
        worker.join(timeout=1)
        ctrl.shutdown()
    assert ctrl.export()["angles"]["base"] == 105
    assert any(90 < value < 105 for value in seen)


def test_a_selected_object_stays_when_the_next_caption_drops_it():
    from app.main import create_app

    app = create_app(fast_config())
    with TestClient(app) as client:
        client.post("/api/look")
        client.post("/api/select", json={"object_id": "orange_cup"})
        mission = app.state.ctx.mission
        mission.vision.observations = []
        step = next(item for item in mission.plan["steps"] if item["tool"] == "approach")
        bound = mission._resolve_live(step)
        assert bound["name"] == "orange cup"
        assert bound["object_id"] == "orange_cup"


def test_choosing_a_detection_plans_that_object():
    with client_for() as client:
        looked = client.post("/api/look").json()
        names = {item["id"]: item["name"] for item in looked["observation"]["items"]}
        assert names["orange_cup"] == "orange cup"
        selected = client.post("/api/select", json={"object_id": "orange_cup"}).json()
        plan = selected["state"]["mission"]["plan"]
        assert plan["planner"] == "selected"
        assert plan["targets"][0]["object_id"] == "orange_cup"
        assert "orange cup" in plan["summary"].lower()
        assert selected["state"]["observation"]["selected_id"] == "orange_cup"


def test_plan_pick_smell_and_ambiguous_orange():
    with client_for() as client:
        vague = client.post(
            "/api/command",
            json={"text": "Find the orange, pick it up, smell it, and put it back."},
        ).json()
        assert vague["status"] == "awaiting_clarification"
        assert "Which orange" in vague["state"]["mission"]["clarification"]["question"]
        answered = client.post("/api/clarify", json={"object_id": "orange_fruit"}).json()
        plan = answered["mission"]["plan"]
        assert [step["tool"] for step in plan["steps"]] == [
            "approach",
            "grasp",
            "lift",
            "sample_odor",
            "place",
            "park",
        ]
        assert "front" in plan["final_state"]
        assert "mock" in plan["enose_note"].lower()

        blue = client.post(
            "/api/command",
            json={"text": "Pick up the blue cup and put it on the left."},
        ).json()
        assert blue["state"]["mission"]["record"]["selected"]["id"] == "blue_cup"
        assert blue["state"]["observation"]["selected_id"] == "blue_cup"
        fruit = client.post(
            "/api/command",
            json={"text": "Find the orange fruit, pick it up, smell it, and put it back."},
        ).json()
        assert fruit["state"]["mission"]["record"]["selected"]["id"] == "orange_fruit"
        assert fruit["state"]["observation"]["selected_id"] == "orange_fruit"


def test_the_arm_turns_toward_the_object_instead_of_always_going_left():
    with client_for() as client:
        client.post("/api/command", json={"text": "Pick up the orange cup."})
        client.post("/api/run")
        wait_mission(client)
        commands = _commands(client)
        assert "base rotate 55" in commands
        assert "base rotate 125" not in commands


def test_an_image_target_is_aimed_within_joint_limits():
    from app.geometry import aim_from_bbox

    cfg = fast_config()
    poses = aim_from_bbox([90, 70, 48, 40], cfg)
    for pose in poses.values():
        for joint, angle in pose.items():
            lo, hi = cfg.joint_limits[joint]
            assert lo <= angle <= hi
    with client_for() as client:
        red = client.post("/api/command", json={"text": "Pick up the red block."}).json()
        assert "joint limits" in red["state"]["mission"]["plan"]["summary"]
        client.post("/api/run")
        state = wait_mission(client)
        commands = " ".join(_commands(client))
        assert "gripper rotate 48" in commands
        assert state["mission"]["outcome"] != "skipped"


def test_mock_cycle_labels_enose_and_missing_enose_is_explicit():
    with client_for() as client:
        client.post(
            "/api/command",
            json={"text": "Find the orange fruit, pick it up, smell it, and put it back."},
        )
        client.post("/api/run")
        state = wait_mission(client)
        text = " ".join(item["message"] for item in state["mission"]["timeline"])
        assert state["mission"]["status"] == "complete"
        assert "Mock E-nose" in text
        assert "not a real measurement" in text
        assert "purge" in text.lower()
        assert "Grab the orange fruit" in text or "Closing the gripper" in text

        client.post("/api/settings", json={"enose_mode": "unavailable"})
        client.post("/api/command", json={"text": "Smell the orange fruit."})
        client.post("/api/run")
        state = wait_mission(client)
        text = " ".join(item["message"] for item in state["mission"]["timeline"])
        assert "Sampling could not be performed" in text
        assert "not connected" in text


def test_multi_object_purges_before_the_next_sample():
    with client_for() as client:
        client.post(
            "/api/command",
            json={"text": "Smell the orange fruit and then the blue cup."},
        )
        client.post("/api/run")
        state = wait_mission(client, timeout=20)
        messages = [item["message"] for item in state["mission"]["timeline"]]
        first_purge = next(
            i for i, message in enumerate(messages) if "orange fruit" in message and "purge" in message.lower()
        )
        second_sample = next(
            i for i, message in enumerate(messages) if "blue cup" in message and "synthetic pattern" in message
        )
        assert first_purge < second_sample
        assert state["mission"]["status"] == "complete"


def test_dry_run_stop_interrupts_and_requires_a_new_command():
    cfg = fast_config()
    cfg.sim_joint_s = 0.45
    with client_for(cfg) as client:
        client.post(
            "/api/command",
            json={"text": "Find the orange fruit, pick it up, smell it, and put it back."},
        )
        client.post("/api/run")
        deadline = time.time() + 3
        while time.time() < deadline:
            state = client.get("/api/state").json()
            if state["robot"]["moving"] or state["robot"]["commands_sent"] > 0:
                break
            time.sleep(0.02)
        stopped = client.post("/api/stop").json()
        assert stopped["software_stop"] is True
        assert stopped["uart_sent"] is False
        assert "Dry-run" in stopped["detail"]
        time.sleep(0.3)
        later = client.get("/api/state").json()
        assert later["mission"]["status"] == "stopped"
        assert later["mission"]["plan"] is None
        assert later["robot"]["moving"] is False
        refused = client.post("/api/run")
        assert refused.status_code == 400
        jog = client.post("/api/jog", json={"joint": "base", "delta": 5})
        assert jog.status_code == 200


def test_manual_jog_is_rejected_during_a_mission_and_limits_apply():
    cfg = fast_config()
    cfg.sim_joint_s = 0.4
    with client_for(cfg) as client:
        denied = client.post("/api/jog", json={"joint": "gripper", "delta": -5})
        assert denied.status_code == 400
        client.post("/api/command", json={"text": "Show me what's on the table."})
        client.post("/api/run")
        deadline = time.time() + 3
        while time.time() < deadline:
            if client.get("/api/state").json()["mission"]["running"]:
                break
            time.sleep(0.02)
        blocked = client.post("/api/jog", json={"joint": "base", "delta": 5})
        assert blocked.status_code == 409
        client.post("/api/stop")


def test_missing_capability_and_scene_survey():
    with client_for() as client:
        thrown = client.post("/api/command", json={"text": "Throw the orange fruit."}).json()
        assert thrown["status"] == "idle"
        assert "throwing" in thrown["state"]["mission"]["error"].lower()
        survey = client.post("/api/command", json={"text": "Show me what's on the table."}).json()
        tools = [step["tool"] for step in survey["state"]["mission"]["plan"]["steps"]]
        assert tools == ["scan", "inspect", "report_scene", "park"]
        assert "grasp" not in tools


def test_connect_warns_and_stop_endpoint_does_not_need_a_plan():
    with client_for() as client:
        denied = client.post("/api/serial/connect", json={"port": "mock://braccio", "confirm_reset": False})
        assert denied.status_code == 400
        assert "reset" in denied.json()["detail"].lower()
        connected = client.post("/api/serial/connect", json={"port": "mock://braccio", "confirm_reset": True})
        assert connected.status_code == 200
        assert connected.json()["synced"] is True
        state = client.get("/api/state").json()
        assert state["robot"]["joint_source"] == "firmware"
        assert state["robot"]["angles"]["gripper"] == 10
        client.post("/api/serial/disconnect")
        unknown = client.get("/api/state").json()["robot"]
        assert unknown["joints_known"] is False
        assert unknown["angles"]["base"] is None
        stopped = client.post("/api/stop").json()
        assert stopped["software_stop"] is True


def test_ollama_scan_keeps_local_blocks_small_and_lists_answers(monkeypatch):
    from app.ollama import hosts_in_block, reset_scan

    assert hosts_in_block("127.0.0.1", "255.0.0.0") == ["127.0.0.1"]
    local = hosts_in_block("192.168.1.20", "255.255.255.0")
    assert "192.168.1.1" in local and "192.168.1.254" in local
    assert "192.168.2.1" not in local
    wide = hosts_in_block("10.1.2.9", "255.0.0.0")
    assert len(wide) == 254
    assert "10.1.2.1" in wide and "10.1.3.1" not in wide

    reset_scan()
    monkeypatch.setattr("app.ollama.local_ipv4_targets", lambda: ["127.0.0.1", "10.9.9.9"])
    monkeypatch.setattr(
        "app.ollama.probe_ollama",
        lambda ip: {"ip": ip, "base_url": f"http://{ip}:11434", "models": ["llama3.2:latest"]}
        if ip == "127.0.0.1"
        else None,
    )
    with client_for() as client:
        started = client.get("/api/llm/scan", params={"refresh": True}).json()
        assert started["scanning"] is True
        deadline = time.time() + 3
        found = started
        while time.time() < deadline:
            found = client.get("/api/llm/scan").json()
            if not found["scanning"]:
                break
            time.sleep(0.02)
        assert found["hosts"] == [{"ip": "127.0.0.1", "base_url": "http://127.0.0.1:11434", "models": ["llama3.2:latest"]}]
    reset_scan()


def test_llm_ip_selects_an_ollama_host(monkeypatch):
    from app.ollama import ollama_base

    assert ollama_base("192.168.1.40") == "http://192.168.1.40:11434"
    assert ollama_base("http://10.0.0.8:11434/") == "http://10.0.0.8:11434"

    class _Tags:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llama3.2:latest"}, {"name": "llava:latest"}]}

    monkeypatch.setattr("httpx.get", lambda url, timeout: _Tags())
    with client_for() as client:
        listed = client.get("/api/llm/models", params={"host": "192.168.1.40"}).json()
        assert listed["base_url"] == "http://192.168.1.40:11434"
        assert listed["models"] == ["llama3.2:latest", "llava:latest"]
        saved = client.post(
            "/api/settings",
            json={"llm_base_url": "10.0.0.5", "llm_model": "llama3.2:latest", "llm_enabled": True},
        )
        settings = saved.json()["settings"]
        assert settings["llm_base_url"] == "http://10.0.0.5:11434"
        assert settings["llm_model"] == "llama3.2:latest"
        assert settings["llm_enabled"] is True


def test_a_click_can_pick_an_object_the_detector_did_not_name():
    from PIL import Image

    from app.geometry import grip_for_bbox
    from app.vision import _box_around

    cfg = fast_config()
    assert grip_for_bbox([0, 0, 60, 140], cfg) > grip_for_bbox([0, 0, 240, 80], cfg)
    image = Image.new("RGB", (640, 480), (230, 230, 228))
    for y in range(180, 320):
        for x in range(220, 360):
            image.putpixel((x, y), (40, 70, 50))
    box = _box_around(image, 280, 240)
    assert 200 < box[0] < 250
    assert 160 < box[1] < 210
    assert box[2] < 180
    with client_for() as client:
        chosen = client.post("/api/select", json={"u": 300, "v": 220}).json()
        plan = chosen["state"]["mission"]["plan"]
        assert plan["status"] == "ready"
        assert "Object" in plan["summary"]
        assert any(step["tool"] == "grasp" for step in plan["steps"])


def test_a_moving_detection_keeps_the_same_identity():
    from app.vision import Observation, Vision

    vision = Vision(fast_config())

    def bottle(bbox):
        return Observation(
            id="temp",
            label="Bottle",
            kind="Bottle",
            color="clear",
            name="clear Bottle",
            bbox=bbox,
            confidence=0.8,
            uncertain=False,
            attributes=["Bottle"],
            table_x=None,
            table_y=None,
            zone=None,
            localizable=False,
            localization_note="",
            source="camera",
            held=False,
            frame_id="",
        )

    vision.observations = [bottle([100, 150, 40, 90])]
    vision.observations[0].id = "det-Bottle-1"
    vision._seen_at["det-Bottle-1"] = time.time()
    moved = vision._assign_ids([bottle([128, 168, 44, 96])])
    assert moved[0].id == "det-Bottle-1"
    assert moved[0].bbox == [128, 168, 44, 96]


def test_the_arm_follows_when_the_object_moves_during_the_reach():
    cfg = fast_config()
    motion = MotionController(cfg)
    motion.start()
    shifted = {"on": False}

    def follow():
        base = motion.export()["angles"]["base"]
        if base is not None and base <= 50:
            shifted["on"] = True
        if shifted["on"]:
            return {"base": 130, "shoulder": 100, "elbow": 92, "wrist_vertical": 70, "wrist_rotation": 90}
        return {"base": 40, "shoulder": 158, "elbow": 62, "wrist_vertical": 85, "wrist_rotation": 90}

    try:
        result = motion.move_pose(follow(), cfg.gripper_open, follow=follow)
        assert result.ok
        assert motion.export()["angles"]["base"] >= 120
        assert "base rotate 40" in motion.command_log
        assert "base rotate 130" in motion.command_log
    finally:
        motion.shutdown()


def test_the_arm_aims_at_the_object_wherever_it_sits_in_the_photo():
    from app.geometry import aim_from_bbox, forward_wrist

    cfg = fast_config()
    corner = aim_from_bbox([30, 40, 50, 80], cfg)["grasp"]["base"]
    middle = aim_from_bbox([280, 180, 70, 90], cfg)["grasp"]["base"]
    far_side = aim_from_bbox([540, 300, 40, 70], cfg)["grasp"]["base"]
    # Image-left → higher base (robot left / +Y). Image-right → lower base.
    assert corner > middle > far_side
    near = aim_from_bbox([280, 360, 60, 80], cfg)["grasp"]
    far = aim_from_bbox([280, 40, 60, 80], cfg)["grasp"]
    assert abs(near["base"] - far["base"]) < 1
    # Elbow stays folded so the gripper hangs toward the table, not the sky.
    for pose in (near, far):
        assert pose["elbow"] >= 150
        assert 40 <= pose["shoulder"] <= 140


def test_a_grid_cell_sits_on_the_object_not_the_bottom_edge():
    from PIL import Image

    from app.vision import _cell_bbox, _content_rect

    image = Image.new("RGB", (640, 480), (0, 0, 0))
    for y in range(60, 420):
        for x in range(0, 640):
            image.putpixel((x, y), (180, 180, 180))
    content = _content_rect(image)
    assert content[1] >= 50
    assert content[3] <= 430
    top_left = _cell_bbox((1, 1), content)
    bottom_right = _cell_bbox((4, 4), content)
    assert top_left[1] < 120
    assert bottom_right[0] > top_left[0]
    assert bottom_right[1] > top_left[1]


def test_a_picture_place_puts_the_bottle_on_the_left_and_the_keyboard_on_the_right():
    from app.vision import _cell_bbox, _parse_region

    content = (0, 60, 640, 420)
    bottle = _cell_bbox(_parse_region({"place": "middle-left"}), content, 3, 3)
    keyboard = _cell_bbox(_parse_region({"horizontal": "right", "vertical": "lower"}), content, 3, 3)
    assert bottle[0] < 220
    assert 120 < bottle[1] < 250
    assert keyboard[0] > bottle[0]
    assert keyboard[1] > bottle[1]


def test_a_place_is_used_instead_of_a_low_pixel_box(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"observations":[{"label":"bottle","color":"clear","place":"middle-left",'
                        '"bbox":[0.05,0.8,0.2,0.15],"confidence":0.9,"uncertain":false}]}'
                    )
                }
            }

    monkeypatch.setattr("httpx.post", lambda url, json, timeout: _Response())
    from app.vision import Vision

    items = Vision(fast_config())._ask_vlm(b"\xff\xd8\xff\xd9")
    assert items[0].name == "clear bottle"
    assert items[0].bbox[0] < 220
    assert 100 < items[0].bbox[1] < 250


def test_a_cell_reply_is_not_read_as_the_photo_region(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"observations":[{"label":"bottle","color":"green","cell":[1,1],'
                        '"confidence":0.9,"uncertain":false}]}'
                    )
                }
            }

    monkeypatch.setattr("httpx.post", lambda url, json, timeout: _Response())
    from app.vision import Vision

    items = Vision(fast_config())._ask_vlm(b"\xff\xd8\xff\xd9")
    assert items[0].name == "green bottle"
    assert items[0].bbox[1] < 120


def test_width_and_height_cover_the_object_instead_of_a_nearby_corner():
    from app.vision import _bbox_pixels

    box = _bbox_pixels([0.15, 0.2, 0.28, 0.5], 640, 480)
    assert box == [96, 96, 179, 240]


def test_a_corner_box_is_not_stretched_into_a_line():
    from app.vision import _bbox_pixels

    # The live model returned the opposite corner. Reading it as width and
    # height made a box that ran off the bottom of the frame.
    box = _bbox_pixels([0.39, 0.67, 0.53, 0.86], 336, 252)
    assert box[0] + box[2] <= 336
    assert box[1] + box[3] <= 252
    assert box[2] < 80
    assert box[3] < 80


def test_ollama_llava_request_uses_the_native_chat_api(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"observations":[{"label":"cup","color":"blue","bbox":[0.1,0.2,0.3,0.4],'
                        '"confidence":0.8,"uncertain":false,"attributes":["cup"]}]}'
                    )
                }
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("httpx.post", fake_post)
    cfg = fast_config()
    assert cfg.vlm_api == "ollama"
    assert cfg.vlm_model == "llava:latest"
    from app.vision import Vision

    vision = Vision(cfg)
    items = vision._ask_vlm(b"\xff\xd8\xff\xd9")
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["json"]["model"] == "llava:latest"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["images"]
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_predict"] == 400
    assert items[0].name == "blue cup"
    assert items[0].source == "camera"
    assert items[0].bbox == [64, 96, 192, 192]


def test_the_local_model_can_choose_the_bottle_the_camera_named(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"id":"bottle-1","question":"",'
                        '"steps":[{"tool":"locate","detail":"Use the green bottle of water."},'
                        '{"tool":"grasp","detail":"Close on the bottle."}]}'
                    )
                }
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    item = _seen("bottle-1", "green Bottle of water")
    adopted = Planner(cfg).plan("Pick up the bottle and smell it.", [item], "mock", [])
    assert adopted.planner == "local-llm"
    assert adopted.targets[0]["object_id"] == "bottle-1"
    assert adopted.targets[0]["seen"] is True
    assert [step["tool"] for step in adopted.steps] == [
        "approach",
        "grasp",
        "lift",
        "sample_odor",
        "place",
        "park",
    ]
    assert adopted.steps[-2]["destination"] == "original"
    assert adopted.steps[0]["object_id"] == "bottle-1"


def _seen(object_id: str, name: str):
    from app.vision import Observation

    return Observation(
        id=object_id,
        label=name,
        kind=name,
        color="",
        name=name,
        bbox=[0, 0, 10, 10],
        confidence=0.8,
        uncertain=False,
        attributes=[],
        table_x=None,
        table_y=None,
        zone=None,
        localizable=False,
        localization_note="",
        source="camera",
        held=False,
        frame_id="f",
    )


def test_the_user_command_is_sent_with_the_detected_names(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"summary":"Pick up the green Bag.","object_ids":["bag-1"],'
                        '"steps":[{"tool":"grasp","detail":"Close on the green Bag."}],"question":""}'
                    )
                }
            }

    def fake_post(url, json, timeout):
        captured["body"] = json
        return _Response()

    monkeypatch.setattr("httpx.post", fake_post)
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    Planner(cfg).plan(
        "pick up the bag",
        [_seen("bag-1", "green Bag"), _seen("bottle-1", "clear Bottle")],
        "mock",
        [],
    )
    system = captured["body"]["messages"][0]["content"]
    user = captured["body"]["messages"][1]["content"]
    assert "pick up the bag" in user
    assert "green Bag" in user
    assert "clear Bottle" in user
    assert "allowed_tools" not in user
    assert "fixed sequence" in system


def test_an_echoed_reply_that_names_the_bottle_becomes_a_pick(monkeypatch):
    content = (
        '{"instruction":"pick a bottle","seen":[],"reply":{'
        '"summary":"",'
        '"object_ids":["det-Bottle-1"],'
        '"steps":[{"tool":"locate","detail":"The bottle is in view."},'
        '{"tool":"grasp","detail":"Close on the green Bottle."}],'
        '"question":"Is the green Bottle the one you meant to pick?"}}'
    )

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": content}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan(
        "pick a bottle",
        [_seen("det-Bottle-1", "green Bottle"), _seen("det-Sink-2", "white Sink")],
        "mock",
        [],
    )
    assert result.status == "ready"
    assert result.question == ""
    assert result.targets[0]["object_id"] == "det-Bottle-1"
    assert [step["tool"] for step in result.steps] == [
        "approach",
        "grasp",
        "lift",
        "place",
        "park",
    ]
    assert result.steps[-2]["destination"] == "original"
    assert result.summary.startswith("Pick up the green Bottle")


def test_the_model_can_name_the_bottle_and_its_steps_are_ignored(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"summary":"I will pick the green Bottle.","object_ids":["green Bottle"],'
                        '"steps":[{"tool":"pick","detail":"Pick up the green Bottle."}],"question":""}'
                    )
                }
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    adopted = Planner(cfg).plan(
        "pick a bottle",
        [_seen("det-Bottle-1", "green Bottle"), _seen("det-Sink-2", "white Sink")],
        "mock",
        [],
    )
    assert adopted.status == "ready"
    assert adopted.targets[0]["object_id"] == "det-Bottle-1"
    assert [step["tool"] for step in adopted.steps] == [
        "approach",
        "grasp",
        "lift",
        "place",
        "park",
    ]
    assert adopted.steps[-2]["destination"] == "original"
    assert all(step["detail"] != "Pick up the green Bottle." for step in adopted.steps)


def test_the_model_choice_is_kept_when_the_words_name_the_middle(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"id":"middle-mouse","question":""}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    left = _seen("left-mouse", "gray Mouse")
    left.kind = "Mouse"
    left.label = "Mouse"
    left.bbox = [212, 194, 47, 46]
    middle = _seen("middle-mouse", "gray Mouse 4")
    middle.kind = "Mouse"
    middle.label = "Mouse"
    middle.bbox = [301, 200, 37, 44]
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan(
        "pick a mouse in the middle of the desk",
        [left, middle],
        "mock",
        [],
    )
    assert result.targets[0]["object_id"] == "middle-mouse"
    assert "The model chose" in result.planner_note


def test_a_mouse_without_a_place_stays_the_one_the_model_named(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"id":"left-mouse","question":""}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    left = _seen("left-mouse", "gray Mouse")
    left.kind = "Mouse"
    left.label = "Mouse"
    left.bbox = [212, 194, 47, 46]
    middle = _seen("middle-mouse", "gray Mouse 4")
    middle.kind = "Mouse"
    middle.label = "Mouse"
    middle.bbox = [301, 200, 37, 44]
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("pick a mouse", [left, middle], "mock", [])
    assert result.targets[0]["object_id"] == "left-mouse"


def test_furthest_in_the_command_does_not_replace_the_model_choice(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"id":"right-glass","question":""}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    def box(object_id, name, kind, bbox):
        item = _seen(object_id, name)
        item.kind = kind
        item.label = kind
        item.bbox = bbox
        return item

    far_mouse = box("far-mouse", "gray Mouse", "Mouse", [180, 40, 40, 30])
    near_mouse = box("near-mouse", "gray Mouse 4", "Mouse", [220, 300, 40, 30])
    far_glass = box("far-glass", "white Cup", "Cup", [300, 30, 48, 80])
    right_glass = box("right-glass", "white Cup 2", "Cup", [420, 180, 50, 90])
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan(
        "pick a glass on the right of furthest mouse",
        [far_mouse, near_mouse, far_glass, right_glass],
        "mock",
        [],
    )
    assert result.targets[0]["object_id"] == "right-glass"
    assert "The model chose white Cup 2" in result.planner_note


def test_an_empty_answer_can_still_be_read_from_the_thinking_text(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": "",
                    "thinking": 'The glass nearest that mouse is white Cup 3. {"id":"cup","question":""}',
                }
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cup = _seen("cup", "white Cup 3")
    mouse = _seen("mouse", "gray Mouse")
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("pick a glass the nearest the furthest mouse", [cup, mouse], "mock", [])
    assert result.status == "ready"
    assert result.targets[0]["object_id"] == "cup"


def test_the_model_chooses_the_object_the_word_matcher_would_miss(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"summary":"Use the clear Cup.","object_ids":["cup-1"],'
                        '"steps":[{"tool":"locate","detail":"The clear Cup is the one to use."},'
                        '{"tool":"grasp","detail":"Close on the clear Cup."}],"question":""}'
                    )
                }
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    cup = _seen("cup-1", "clear Cup")
    bottle = _seen("bottle-1", "green Water bottle")
    result = Planner(cfg).plan("Pick up the glass.", [bottle, cup], "mock", [])
    assert result.planner == "local-llm"
    assert result.targets[0]["object_id"] == "cup-1"
    assert "clear Cup" in result.summary
    assert [step["tool"] for step in result.steps][0] == "approach"
    assert "grasp" in [step["tool"] for step in result.steps]
    assert [step["tool"] for step in result.steps][-1] == "park"


def test_thai_do_not_pick_does_not_reuse_a_target(monkeypatch):
    called = {"llm": False}

    def _fake_post(*args, **kwargs):
        called["llm"] = True
        raise AssertionError("noop commands must not call the model")

    monkeypatch.setattr("httpx.post", _fake_post)
    from app.planner import Planner

    cup = _seen("cup", "white Cup")
    cup.kind = "Cup"
    cup.label = "Cup"
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("ไม่ต้องหยิบอะไรเลย", [cup], "mock", [])
    assert result.status == "ready"
    assert result.intent == "noop"
    assert result.targets == []
    assert all(step["tool"] != "grasp" for step in result.steps)
    assert called["llm"] is False


def test_thai_beside_bottle_on_the_right_picks_the_cup(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            # Model wrongly returns the landmark bottle.
            return {"message": {"content": '{"id":"bottle","question":""}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    bottle = _seen("bottle", "blue Bottle")
    bottle.kind = "Bottle"
    bottle.label = "Bottle"
    bottle.bbox = [247, 300, 56, 108]
    cup = _seen("cup", "white Cup")
    cup.kind = "Cup"
    cup.label = "Cup"
    cup.bbox = [376, 298, 96, 121]
    sink = _seen("sink", "white Sink")
    sink.kind = "Sink"
    sink.label = "Sink"
    sink.bbox = [1, 118, 529, 299]
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("หยิบของที่อยู่ข้างขวดทางด้านขวา", [bottle, cup, sink], "mock", [])
    assert result.status == "ready"
    assert result.targets[0]["object_id"] == "cup"


def test_thai_glass_does_not_keep_a_bottle(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"id":"bottle","question":""}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cup = _seen("cup", "white Cup")
    cup.kind = "Cup"
    cup.label = "Cup"
    bottle = _seen("bottle", "blue Bottle")
    bottle.kind = "Bottle"
    bottle.label = "Bottle"
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("หยิบแก้ว", [cup, bottle], "mock", [])
    assert result.status == "ready"
    assert result.targets[0]["object_id"] == "cup"


def test_an_unsure_model_question_asks_which_object(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"id":"cup-1","question":"which white Cup?"}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    left = _seen("cup-1", "white Cup")
    left.kind = "Cup"
    left.label = "Cup"
    right = _seen("cup-2", "white Cup 2")
    right.kind = "Cup"
    right.label = "Cup"
    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("หยิบแก้ว", [left, right], "mock", [])
    assert result.status == "clarification"
    assert result.steps == []
    assert {option["id"] for option in result.options} == {"cup-1", "cup-2"}


def test_an_object_the_camera_cannot_see_asks_the_user_again(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"id":"","question":"I can\'t detect that object. The camera sees white Cup and blue Bottle. '
                        'Please ask for one of these."}'
                    )
                }
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan(
        "pick a banana",
        [_seen("cup", "white Cup"), _seen("bottle", "blue Bottle")],
        "mock",
        [],
    )
    assert result.status == "clarification"
    assert result.steps == []
    assert "can't detect" in result.question.lower()
    assert "Cup" in result.question or "Bottle" in result.question
    assert {option["id"] for option in result.options} == {"cup", "bottle"}


def test_an_empty_camera_asks_the_user_to_look_again(monkeypatch):
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("pick a glass", [], "mock", [])
    assert result.status == "clarification"
    assert result.steps == []
    assert "can't detect" in result.question.lower()


def test_a_failed_model_does_not_fall_back_to_word_matching(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("httpx.post", _boom)
    from app.planner import Planner

    cfg = fast_config()
    cfg.llm_enabled = True
    result = Planner(cfg).plan("Pick up the cup.", [_seen("cup-1", "clear Cup")], "mock", [])
    assert result.status == "clarification"
    assert result.planner == "local-llm"
    assert result.steps == []


def test_the_model_can_call_a_clear_cup_the_glass(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"id":"cup-1"}'}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())
    from app.planner import Planner
    from app.vision import Observation

    cfg = fast_config()
    cfg.llm_enabled = True
    cup = Observation(
        id="cup-1",
        label="Clear Cup",
        kind="Clear Cup",
        color="clear",
        name="Clear Cup",
        bbox=[0, 0, 10, 10],
        confidence=0.8,
        uncertain=False,
        attributes=[],
        table_x=None,
        table_y=None,
        zone=None,
        localizable=False,
        localization_note="",
        source="camera",
        held=False,
        frame_id="f",
    )
    bottle = Observation(
        id="bottle-1",
        label="Clear Water bottle",
        kind="Clear Water bottle",
        color="clear",
        name="Clear Water bottle",
        bbox=[0, 0, 10, 10],
        confidence=0.8,
        uncertain=False,
        attributes=[],
        table_x=None,
        table_y=None,
        zone=None,
        localizable=False,
        localization_note="",
        source="camera",
        held=False,
        frame_id="f",
    )
    assert Planner(cfg).choose_seen("Find the glass", [bottle, cup]) is cup


def test_a_bottle_of_water_answers_a_request_for_the_bottle():
    from app.planner import TargetSpec, _match
    from app.vision import Observation, _extract_json

    text = """
    { "observations": [
      { "label": "Bottle of water", "color": "green", "bbox": [0.4,0.6,0.2,0.2], "confidence": 0.9, "uncertain": 0.1 },
      { "label": "Bottle cap",
    """
    parsed = _extract_json(text)
    assert parsed["observations"][0]["label"] == "Bottle of water"
    item = Observation(
        id="vlm-bottle",
        label="Bottle of water",
        kind="Bottle of water",
        color="green",
        name="green Bottle of water",
        bbox=[280, 320, 120, 90],
        confidence=0.9,
        uncertain=False,
        attributes=[],
        table_x=None,
        table_y=None,
        zone=None,
        localizable=False,
        localization_note="",
        source="camera",
        held=False,
        frame_id="frame",
    )
    assert _match(TargetSpec(color=None, kind=None, phrase="bottle"), [item]) == [item]


def test_a_fresh_look_uses_the_detector_box_and_does_not_ask_again(monkeypatch):
    import io

    from PIL import Image

    from app.detect import Detection

    calls = {"n": 0}

    def fake_detect(image):
        calls["n"] += 1
        return [Detection("Bottle", [118, 150, 46, 120], 0.91)]

    monkeypatch.setattr("app.detect.detect_objects", fake_detect)
    from app.vision import Vision

    frame = io.BytesIO()
    Image.new("RGB", (640, 480), (210, 220, 215)).save(frame, format="JPEG")
    vision = Vision(fast_config())
    vision.mode = "hardware"
    monkeypatch.setattr(vision, "_read_hardware_frame", lambda: frame.getvalue())
    items = vision.capture("scan")
    assert items[0].label == "Bottle"
    assert items[0].bbox == [118, 150, 46, 120]
    assert items[0].bbox[2] < 80
    vision.capture("inspect")
    vision.capture("plan")
    assert calls["n"] == 1
    vision.invalidate("The arm moved, so the last camera observation is stale.")
    vision.capture("inspect")
    assert calls["n"] == 2
    vision.capture("confirm")
    assert calls["n"] == 3


def test_vlm_frame_is_shrunk_before_it_is_sent():
    import io

    from PIL import Image

    from app.vision import _vlm_jpeg

    image = Image.new("RGB", (640, 480), (20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    _data, width, height, _content = _vlm_jpeg(buffer.getvalue())
    assert max(width, height) <= 336
    assert width < 640 and height < 480


def test_camera_fallback_uses_a_mode_the_device_advertises():
    from app.cameras import parse_avfoundation_fallback

    text = """
Selected framerate (15.000000) is not supported by the device.
Supported modes:
  640x480@[15.000000 30.000000]fps
  1920x1080@[15.000000 30.000000]fps
Supported pixel formats:
  uyvy422
  yuyv422
"""
    assert parse_avfoundation_fallback(text) == {"size": "640x480", "rate": "30", "pixel": "uyvy422"}


def test_camera_list_names_the_device_the_operator_selects(monkeypatch):
    from app.cameras import parse_avfoundation_devices

    sample = """
[AVFoundation indev] AVFoundation video devices:
[AVFoundation indev] [0] OBS Virtual Camera
[AVFoundation indev] [3] MacBook Pro Camera
[AVFoundation indev] AVFoundation audio devices:
[AVFoundation indev] [0] MacBook Pro Microphone
"""
    devices = parse_avfoundation_devices(sample)
    assert [item["label"] for item in devices] == ["OBS Virtual Camera", "MacBook Pro Camera"]
    assert devices[1]["id"] == "3"

    opened: list[int] = []
    monkeypatch.setattr("app.vision.resolve_camera_index", lambda label: 3)
    monkeypatch.setattr(
        "app.vision.Vision._open_ffmpeg",
        lambda self, attempt=0: opened.append(self.camera_index),
    )
    monkeypatch.setattr(
        "app.main.label_for",
        lambda mode, index: "MacBook Pro Camera" if mode == "hardware" else "Mock tabletop camera",
    )
    with client_for() as client:
        switched = client.post("/api/settings", json={"camera_mode": "hardware", "camera_index": 3})
        assert switched.status_code == 200
        observation = switched.json()["observation"]
        assert observation["mode"] == "hardware"
        assert observation["camera_index"] == 3
        assert observation["camera_label"] == "MacBook Pro Camera"
        assert opened == [3]
        back = client.post("/api/settings", json={"camera_mode": "mock"})
        assert back.json()["observation"]["mode"] == "mock"
        assert back.json()["observation"]["camera_label"] == "Mock tabletop camera"


def _commands(client: TestClient) -> list[str]:
    return client.app.state.ctx.motion.command_log
