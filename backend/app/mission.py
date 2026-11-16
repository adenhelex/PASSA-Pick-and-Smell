"""Run a planned pick-and-smell sequence. Motion stays inside the controller."""

from __future__ import annotations

import threading
import time
import uuid

from app.config import ARM_JOINTS, Config
from app.enose import ENose
from app.geometry import aim_from_bbox, classify_table_point, grip_for_bbox
from app.motion import MotionController
from app.planner import Planner, _words_fit
from app.vision import Vision


class MissionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class Mission:
    def __init__(self, cfg: Config, motion: MotionController, vision: Vision, enose: ENose, planner: Planner):
        self.cfg = cfg
        self.motion = motion
        self.vision = vision
        self.enose = enose
        self.planner = planner
        self._lock = threading.RLock()
        self.picked = False
        self._bound_map: dict = {}
        self._cancel = threading.Event()
        self._clarify = threading.Event()
        self._clarify_value: dict | None = None
        self._thread: threading.Thread | None = None
        self._request = ""
        self._answers: list[dict] = []
        self._pinned = None
        self.settings_epoch = 0
        self.status = "idle"
        self.phase = "idle"
        self.plan: dict | None = None
        self.clarification: dict | None = None
        self.timeline: list[dict] = []
        self.record = {"request": "", "selected": None, "completed": []}
        self.last_stop: dict | None = None
        self.error = ""
        self.outcome = ""
        self._seq = 0
        self.on_change = lambda: None

    def export(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "phase": self.phase,
                "plan": self.plan,
                "clarification": self.clarification,
                "timeline": list(self.timeline),
                "record": {
                    "request": self.record["request"],
                    "selected": self.record["selected"],
                    "completed": list(self.record["completed"]),
                },
                "last_stop": self.last_stop,
                "error": self.error,
                "outcome": self.outcome,
                "running": self._thread is not None and self._thread.is_alive(),
            }

    def command(self, text: str) -> dict:
        if self._running():
            raise MissionError("A mission is running. Stop it before planning another.", 409)
        if text.strip().lower() in {"stop", "e-stop", "estop", "abort", "halt"}:
            return {"kind": "stop", "stop": self.stop()}
        self._request = text.strip()
        self._answers = []
        self._pinned = None
        self.vision.capture("plan")
        self._log("planning", "observation", _describe(self.vision.observations))
        return {"kind": "plan", "plan": self._build_plan()}

    def select_object(self, object_id: str) -> dict:
        if self._running():
            raise MissionError("A mission is running. Stop it before choosing another object.", 409)
        item = next((obs for obs in self.vision.observations if obs.id == object_id and not obs.held), None)
        if item is None:
            raise MissionError("That detection is not in the current look. Look again, then choose it.")
        self._request = item.name
        self._answers = []
        self._pinned = item
        self._log("planning", "observation", _describe(self.vision.observations))
        result = self.planner.plan_for_object(item, self.enose.mode)
        return {"kind": "plan", "plan": self._apply(result)}

    def clarify(self, text: str | None = None, object_id: str | None = None) -> dict:
        answer = {"text": text or "", "object_id": object_id}
        if self._running() and self.status == "awaiting_clarification":
            self._clarify_value = answer
            self._clarify.set()
            return {"kind": "plan", "plan": self.export()}
        if self.status != "awaiting_clarification":
            raise MissionError("The planner is not waiting for an answer.")
        self._answers.append(answer)
        return {"kind": "plan", "plan": self._build_plan()}

    def run(self) -> dict:
        if self._running():
            raise MissionError("A mission is already running.", 409)
        if self.clarification is not None or self.status == "awaiting_clarification":
            raise MissionError("Answer the question before running.")
        if not self.plan or self.plan.get("status") != "ready":
            raise MissionError("Plan a command before running. Stopping cancels the previous plan.")
        if self.plan.get("epoch") != self.settings_epoch:
            raise MissionError("Device settings changed. Plan again before running.")
        robot = self.motion.export()
        if robot["execution_mode"] == "armed" and (
            not robot["serial_connected"] or not robot["synced"] or robot["needs_status_sync"]
        ):
            raise MissionError("Armed motion needs a synchronized serial status. Connect and query status, or switch to dry-run.")
        if not robot["joints_known"]:
            raise MissionError("Commanded joint angles are unknown. Sync status or reset the dry-run pose.")
        self._cancel.clear()
        self.motion.release_cancel()
        self.motion.set_mission_active(True)
        with self._lock:
            self.status = "running"
            self.phase = "scanning"
            self.outcome = ""
            self.error = ""
        self._touch()
        plan = self.plan
        self._thread = threading.Thread(target=self._execute, args=(plan,), name="mission", daemon=True)
        self._thread.start()
        return {"started": True, "mission_id": plan["id"]}

    def stop(self) -> dict:
        """Software stop. UART stop is sent by the motion controller, not the planner."""
        self._cancel.set()
        self._clarify.set()
        result = self.motion.stop()
        payload = {
            "software_stop": result.software_stop,
            "uart_sent": result.uart_sent,
            "acknowledged": result.acknowledged,
            "detail": result.detail,
        }
        with self._lock:
            self.last_stop = payload
            self.plan = None
            self.clarification = None
            self.status = "stopped"
            self.phase = "stopped"
            self.outcome = "stopped"
            self._seq += 1
            self.timeline.append(
                {
                    "id": self._seq,
                    "ts": time.time(),
                    "phase": "stopped",
                    "level": "error",
                    "message": result.detail,
                }
            )
        self._touch()
        return payload

    def note_settings_changed(self) -> None:
        self.settings_epoch += 1
        self._touch()

    def _build_plan(self) -> dict:
        result = self.planner.plan(
            self._request,
            self.vision.observations,
            self.enose.mode,
            self._answers,
        )
        return self._apply(result)

    def _apply(self, result) -> dict:
        with self._lock:
            self.record["request"] = self._request
            self.record["completed"] = []
            self.error = ""
            if result.status == "clarification":
                self.status = "awaiting_clarification"
                self.phase = "idle"
                self.plan = None
                self.outcome = ""
                self.clarification = {"question": result.question, "options": result.options}
                self.record["selected"] = None
                self.vision.set_selected(None)
            elif result.status == "unavailable":
                self.status = "idle"
                self.phase = "idle"
                self.plan = None
                self.clarification = None
                self.outcome = "unavailable"
                self.error = result.summary
                self.vision.set_selected(None)
            else:
                selected = next((item for item in result.targets if item.get("object_id")), None)
                self.clarification = None
                self.outcome = ""
                self.plan = {
                    "id": uuid.uuid4().hex[:8],
                    "status": "ready",
                    "epoch": self.settings_epoch,
                    "planner": result.planner,
                    "planner_note": result.planner_note,
                    "intent": result.intent,
                    "summary": result.summary,
                    "final_state": result.final_state,
                    "enose_note": result.enose_note,
                    "steps": result.steps,
                    "targets": result.targets,
                }
                self.status = "ready"
                self.phase = "idle"
                self.record["selected"] = (
                    None
                    if selected is None
                    else {"id": selected["object_id"], "name": selected["name"]}
                )
                self.vision.set_selected(None if selected is None else selected.get("object_id"))
            self._seq += 1
            message = self.error or (self.clarification or {}).get("question") or (self.plan or {}).get("summary") or ""
            if message:
                self.timeline.append(
                    {
                        "id": self._seq,
                        "ts": time.time(),
                        "phase": "planning",
                        "level": "clarify" if self.clarification else "error" if self.error else "info",
                        "message": message,
                    }
                )
            snapshot = self.export()
        self._touch()
        return snapshot

    def _execute(self, plan: dict) -> None:
        skipped: set[int] = set()
        self.picked = False
        self._bound_map = {}
        try:
            for step in plan["steps"]:
                if self._cancel.is_set():
                    break
                index = step.get("object_index")
                if index in skipped and step["tool"] not in {"park", "scan", "report_scene"}:
                    continue
                outcome = self._step(step, plan)
                if outcome == "stop" or self._cancel.is_set():
                    break
                if outcome == "skip" and index is not None:
                    skipped.add(index)
                    continue
                if outcome == "abort":
                    with self._lock:
                        self.status = "blocked"
                        self.phase = "blocked"
                        self.outcome = "blocked"
                    break
                if outcome == "fail":
                    self._park_if_safe()
                    with self._lock:
                        self.status = "blocked"
                        self.phase = "blocked"
                        self.outcome = "blocked"
                    break
                with self._lock:
                    self.record["completed"].append(step["tool"])
            else:
                with self._lock:
                    if self.status == "running":
                        if skipped and not self.picked:
                            self.status = "blocked"
                            self.phase = "blocked"
                            self.outcome = "skipped"
                        else:
                            self.status = "complete"
                            self.phase = "complete"
                            self.outcome = "complete"
                        self._seq += 1
                        if self.outcome == "complete":
                            closing = plan.get("final_state") or "Mission finished."
                            level = "info"
                        else:
                            closing = "The pickup was skipped. The arm was parked without grasping."
                            level = "error"
                        self.timeline.append(
                            {
                                "id": self._seq,
                                "ts": time.time(),
                                "phase": self.phase,
                                "level": level,
                                "message": closing,
                            }
                        )
        finally:
            self.motion.release_cancel()
            self.motion.set_mission_active(False)
            self._touch()

    def _step(self, step: dict, plan: dict) -> str:
        tool = step["tool"]
        self._phase_for(tool)
        self._log(self.phase, "action", step["detail"])
        if tool == "scan":
            return self._scan(plan)
        if tool == "inspect":
            if step.get("purpose") == "confirm_pickup":
                return self._confirm_pickup(step)
            if plan.get("planner") == "selected":
                self._log("scanning", "observation", _describe(self.vision.observations))
                return "ok"
            self._sync_arm()
            items = self.vision.capture("inspect")
            self._log("scanning", "observation", _describe(items))
            return "ok"
        if tool == "report_scene":
            items = self.vision.observations
            self._log("scanning", "observation", _describe(items))
            return "ok"
        if tool == "locate":
            return self._locate(step)
        if tool == "approach":
            return self._approach(step)
        if tool == "grasp":
            return self._grasp(step)
        if tool == "lift":
            return self._lift_bound(step)
        if tool == "sample_odor":
            return self._sample(step)
        if tool == "place":
            return self._place(step)
        if tool == "park":
            return self._park()
        self._log(self.phase, "error", f"{tool} is not available on this arm.")
        return "abort"

    def _scan(self, plan: dict) -> str:
        if plan.get("planner") == "selected":
            name = ((plan.get("targets") or [{}])[0].get("name")) or "object you selected"
            self._log("scanning", "info", f"Keeping the {name} you selected. The arm follows it while it moves.")
            self._log("scanning", "observation", _describe(self.vision.observations))
            return "ok"
        poses = self._viewpoints(plan)
        # On a real camera, identify once from the last viewpoint. The wrist still
        # visits each selected pose, and that last look stays fresh for locate.
        look_at = max(0, len(poses) - 1)
        for index, pose in enumerate(poses):
            outcome = self._move(pose.angles(), self.cfg.gripper_open, slow=False)
            if outcome != "ok":
                return outcome
            self._sync_arm()
            if self.vision.mode != "hardware" or index == look_at:
                self.vision.capture("scan")
        self._log("scanning", "observation", _describe(self.vision.observations))
        return "ok"

    def _viewpoints(self, plan: dict):
        """Face the known target. A full sweep starts in the middle, not on the left."""
        poses = list(self.cfg.scan)
        if not poses:
            return poses
        wanted: list[float] = []
        for target in plan.get("targets") or []:
            zone = self.cfg.zones.get(target.get("zone") or "")
            if zone is not None:
                wanted.append(zone.approach.base)
        if not wanted:
            middle = len(poses) // 2
            return [poses[middle]] + [pose for index, pose in enumerate(poses) if index != middle]
        chosen = []
        used: set[int] = set()
        for base in wanted:
            nearest = min(range(len(poses)), key=lambda index: (abs(poses[index].base - base), index))
            if nearest in used:
                continue
            used.add(nearest)
            chosen.append(poses[nearest])
        return chosen

    def _locate(self, step: dict) -> str:
        self._sync_arm()
        if self.vision.export()["stale"]:
            self.vision.capture("locate")
        bound = self._resolve_live(step)
        if bound is None:
            self._log(
                "target_found",
                "error",
                f"I don't see {step.get('description') or 'that object'} after scanning. I did not command a pickup.",
            )
            return "skip"
        self._remember(bound)
        zone = None
        if bound["localizable"]:
            zone, _reason = classify_table_point(
                bound["table_x"],
                bound["table_y"],
                self.cfg,
                zone_id=bound.get("zone"),
            )
        if zone is None:
            bound["zone"] = None
            bound["aim"] = aim_from_bbox(bound.get("bbox") or [280, 200, 80, 80], self.cfg)
            self._bounds()[step["object_index"]] = bound
            self._phase("target_found")
            self._log(
                "target_found",
                "observation",
                f"Target found: {bound['name']}. The arm will follow it and adjust left, right, near, and far. "
                "Only the arm's joint limits apply.",
            )
            return "ok"
        bound["zone"] = zone.id
        bound["aim"] = None
        self._bounds()[step["object_index"]] = bound
        self._phase("target_found")
        self._log("target_found", "observation", f"Target found: {bound['name']} in {zone.id}.")
        return "ok"

    def _approach(self, step: dict) -> str:
        # Locate is folded in: bind the live target, then move to it.
        if self._require_bound(step) is None:
            outcome = self._locate(step)
            if outcome != "ok":
                return outcome
        bound = self._require_bound(step)
        if bound is None:
            return "skip"
        self._phase("approaching")
        self._refresh_aim(bound)
        if bound.get("aim"):
            self._log("approaching", "info", f"Following the {bound['name']} and adjusting near, far, left, and right.")
        return self._move(
            self._arm_pose(bound, "approach"),
            self.cfg.gripper_open,
            slow=False,
            follow=self._aim_follow(bound, "approach"),
        )

    def _grasp(self, step: dict) -> str:
        bound = self._require_bound(step)
        if bound is None:
            return "skip"
        self._phase("grasping")
        self._refresh_aim(bound)
        outcome = self._move(
            self._arm_pose(bound, "grasp"),
            self.cfg.gripper_open,
            slow=False,
            follow=self._aim_follow(bound, "grasp"),
        )
        if outcome != "ok":
            return outcome
        zone = self.cfg.zones.get(bound.get("zone") or "")
        return self._close_on(bound, zone)

    def _close_on(self, bound: dict, zone) -> str:
        if self.vision.export()["stale"]:
            self._log("grasping", "info", "The last observation is stale. Taking a new frame before closing the gripper.")
            self._sync_arm()
            self.vision.capture("pre-grasp")
        live = self._live_object(bound["object_id"])
        if zone is not None and live and live.zone and live.zone != zone.id and live.localizable:
            new_zone, reason = classify_table_point(live.table_x, live.table_y, self.cfg, zone_id=live.zone)
            if new_zone is None:
                self._log("grasping", "error", reason + " I stopped before closing.")
                return "fail"
            self._log("grasping", "observation", f"The {bound['name']} is now in {new_zone.id}. Replanning the grasp.")
            bound["zone"] = new_zone.id
            outcome = self._move(new_zone.grasp.angles(), self.cfg.gripper_open, slow=False)
            if outcome != "ok":
                return outcome
            zone = new_zone
        elif live is None and zone is not None:
            self._log("grasping", "error", f"I can no longer see the {bound['name']}. I did not close the gripper.")
            return "fail"
        angle = int(round(bound["close_angle"]))
        self._log(
            "grasping",
            "action",
            f"Closing the gripper to {angle}° for the {bound['name']}. "
            "This is a commanded angle; the firmware does not report grip force.",
        )
        current = self.motion.export()["angles"]
        arm = {name: current[name] for name in ARM_JOINTS}
        outcome = self._move(arm, angle, slow=False)
        if outcome != "ok":
            return outcome
        closed = self.motion.export()["angles"]
        if all(closed.get(name) is not None for name in ARM_JOINTS):
            bound["picked_pose"] = {name: float(closed[name]) for name in ARM_JOINTS}
        grasped = self.vision.attempt_grasp(bound["object_id"])
        if not grasped:
            self._log("grasping", "observation", f"The close command finished, but the {bound['name']} may still be on the table.")
        return "ok"

    def _lift_bound(self, step: dict) -> str:
        bound = self._require_bound(step)
        if bound is None:
            return "skip"
        return self._lift(bound)

    def _lift(self, bound: dict) -> str:
        current = self.motion.export()["angles"]
        if any(current[name] is None for name in ARM_JOINTS):
            self._log("grasping", "error", "Joint angles became unknown. I will not keep lifting.")
            return "abort"
        target = self._arm_pose(bound, "lift")
        start = {name: float(current[name]) for name in ARM_JOINTS}
        segments = max(1, self.cfg.lift_segments)
        for index in range(1, segments + 1):
            blend = index / segments
            arm = {
                name: start[name] + (target[name] - start[name]) * blend
                for name in ARM_JOINTS
            }
            outcome = self._move(arm, bound["close_angle"], slow=True)
            if outcome != "ok":
                return outcome
        return "ok"

    def _confirm_pickup(self, step: dict) -> str:
        bound = self._require_bound(step)
        if bound is None:
            return "skip"
        for attempt in (1, 2):
            self._sync_arm()
            self.vision.capture("confirm")
            seen = self._live_object(bound["object_id"])
            track = self.vision.tracks.get(bound["object_id"])
            if track and track.held and seen and seen.held:
                self._log("grasping", "observation", f"The {bound['name']} moved with the gripper.")
                self.picked = True
                return "ok"
            if track and track.held and seen is None:
                self._log(
                    "grasping",
                    "observation",
                    f"The {bound['name']} is not in the fresh frame. The pickup is uncertain because the view is obstructed.",
                )
                self.picked = True
                return "ok"
            if bound.get("aim"):
                self._log(
                    "grasping",
                    "observation",
                    f"The close on the {bound['name']} was commanded. The camera cannot prove it is in the gripper.",
                )
                self.picked = True
                return "ok"
            if attempt == 1:
                self._log(
                    "grasping",
                    "error",
                    f"The {bound['name']} is still on the table. The pickup did not happen. Trying once more.",
                )
                zone = self.cfg.zones.get(track.zone if track else bound["zone"])
                if zone is None:
                    return "fail"
                if self._move(zone.approach.angles(), self.cfg.gripper_open, slow=False) != "ok":
                    return "abort"
                if self._move(zone.grasp.angles(), self.cfg.gripper_open, slow=False) != "ok":
                    return "abort"
                if self._close_on(bound, zone) != "ok":
                    return "abort"
                if self._lift(bound) != "ok":
                    return "abort"
                continue
            self._log("grasping", "error", f"The {bound['name']} is still on the table after a second look. Stopping this pickup.")
            return "fail"
        return "fail"

    def _sample(self, step: dict) -> str:
        bound = self._require_bound(step)
        if bound is None:
            return "skip"
        self._phase("sampling")
        outcome = self._move(self.cfg.sample.angles(), bound["close_angle"], slow=False)
        if outcome != "ok":
            return outcome
        self._log("sampling", "action", f"Holding the {bound['name']} steady near the wrist inlet.")
        if not self._hold():
            return "stop"
        if self.enose.mode == "unavailable":
            failed = self.enose.baseline(self._cancel)
            self._log("sampling", "error", failed["detail"])
            return "ok"
        stages = (
            ("baseline", lambda: self.enose.baseline(self._cancel)),
            ("sample", lambda: self.enose.sample(bound["object_id"], bound["name"], self._cancel)),
            ("purge", lambda: self.enose.purge(self._cancel)),
        )
        for stage, read in stages:
            if self._cancel.is_set():
                return "stop"
            reading = read()
            if self._cancel.is_set():
                return "stop"
            level = "observation" if reading.get("ok") else "error"
            self._log("sampling", level, reading["detail"])
            if not reading.get("ok"):
                if stage != "purge" and self.enose.mode != "unavailable":
                    purge = self.enose.purge(self._cancel)
                    self._log("sampling", "error" if not purge.get("ok") else "observation", purge["detail"])
                return "fail" if stage == "purge" else "ok"
        return "ok"

    def _place(self, step: dict) -> str:
        bound = self._require_bound(step)
        if bound is None:
            return "skip"
        destination = step.get("destination") or "original"
        picked = bound.get("picked_pose")
        if destination == "original" and picked:
            self._phase("placing")
            outcome = self._move(picked, bound["close_angle"], slow=False)
            if outcome != "ok":
                return outcome
            outcome = self._move(picked, self.cfg.gripper_open, slow=False)
            if outcome != "ok":
                return outcome
            zone_id = bound.get("zone")
            track = self.vision.tracks.get(bound["object_id"])
            if track is not None and zone_id:
                self.vision.place(bound["object_id"], zone_id)
            elif track is not None:
                track.held = False
            self._log("placing", "observation", f"Put the {bound['name']} back where it was picked up.")
            return "ok"
        if bound.get("aim") and destination == "original":
            self._phase("placing")
            pose = self._arm_pose(bound, "grasp")
            outcome = self._move(pose, bound["close_angle"], slow=False)
            if outcome != "ok":
                return outcome
            outcome = self._move(pose, self.cfg.gripper_open, slow=False)
            if outcome != "ok":
                return outcome
            self._log("placing", "observation", f"Commanded the gripper open where the {bound['name']} was aimed.")
            return "ok"
        if destination == "original":
            spec = self.cfg.objects.get(bound["object_id"])
            zone_id = spec.zone if spec else None
        else:
            zone_id = destination
        zone = self.cfg.zones.get(zone_id or "")
        if zone is None or not zone.reachable:
            self._log("placing", "error", "The place location has no taught pose inside the workspace. I am still holding the object.")
            return "fail"
        occupant = self.vision.occupant(zone.id, ignore=bound["object_id"])
        if occupant is not None:
            self._log(
                "placing",
                "error",
                f"The {zone.id} zone already holds the {occupant.spec.name}. "
                f"I will not set the {bound['name']} down there.",
            )
            return "fail"
        self._phase("placing")
        outcome = self._move(zone.approach.angles(), bound["close_angle"], slow=False)
        if outcome != "ok":
            return outcome
        outcome = self._move(zone.grasp.angles(), bound["close_angle"], slow=False)
        if outcome != "ok":
            return outcome
        outcome = self._move(zone.grasp.angles(), self.cfg.gripper_open, slow=False)
        if outcome != "ok":
            return outcome
        self.vision.place(bound["object_id"], zone.id)
        self._log("placing", "observation", f"Commanded the gripper open on {zone.id}. Pickup release is not force-checked.")
        return "ok"

    def _park(self) -> str:
        holding = [track for track in self.vision.tracks.values() if track.held]
        gripper = holding[0].spec.close_angle if holding else self.cfg.gripper_open
        outcome = self._move(self.cfg.park.angles(), gripper, slow=False)
        if holding:
            self._log("complete", "info", f"Parked while still holding the {holding[0].spec.name}.")
        return outcome

    def _park_if_safe(self) -> None:
        robot = self.motion.export()
        if robot["needs_status_sync"] or not robot["joints_known"]:
            self._log("blocked", "error", "I did not park. The last motion outcome is uncertain until status is synced.")
            return
        self._park()

    def _move(self, arm: dict, gripper: float, slow: bool, follow=None) -> str:
        if self._cancel.is_set():
            return "stop"
        result = self.motion.move_pose(arm, gripper, slow=slow, follow=follow)
        if result.commands and result.detail != "Already at that pose.":
            self.vision.invalidate("The arm moved, so the last camera observation is stale.")
            self._sync_arm()
        if result.interrupted or self._cancel.is_set():
            self._log("stopped", "error", result.detail or "Stopped.")
            return "stop"
        if result.uncertain:
            self._log("blocked", "error", result.detail + " The plan was cancelled and the command was not replayed.")
            self._cancel.set()
            return "abort"
        if not result.ok:
            self._log("blocked", "error", result.detail)
            return "abort"
        return "ok"

    def _hold(self) -> bool:
        deadline = time.time() + self.cfg.sample_hold_s
        while time.time() < deadline:
            if self._cancel.is_set():
                return False
            time.sleep(0.02)
        return True

    def _resolve_live(self, step: dict) -> dict | None:
        object_id = step.get("object_id")
        if object_id:
            live = self._live_object(object_id)
            if live is not None:
                return self._bound_from_obs(live)
        description = (step.get("description") or "").strip()
        same_name = [
            item
            for item in self.vision.observations
            if not item.held and description and item.name.lower() == description.lower()
        ]
        if same_name:
            return self._bound_from_obs(same_name[0])
        pinned = self._pinned
        if pinned is not None and (
            (object_id and pinned.id == object_id) or (description and pinned.name.lower() == description.lower())
        ):
            return self._bound_from_obs(pinned)
        if object_id:
            return None
        if self.cfg.llm_enabled:
            chosen = self.planner.choose_seen(self._request or description, self.vision.observations)
            if chosen is not None:
                return self._bound_from_obs(chosen)
            return None
        hits = [
            item
            for item in self.vision.observations
            if not item.held and description and _words_fit(description, item)
        ]
        if len(hits) > 1 and len({item.name.lower() for item in hits}) == 1:
            return self._bound_from_obs(hits[0])
        if len(hits) > 1:
            answer = self._ask(
                f"Which {description} do you mean?",
                [{"id": item.id, "label": item.name} for item in hits],
            )
            if answer is None:
                return None
            chosen = next((item for item in hits if item.id == answer.get("object_id")), None)
            if chosen is None:
                text = (answer.get("text") or "").lower()
                chosen = next((item for item in hits if item.name in text or item.kind in text), None)
            if chosen is None:
                return None
            return self._bound_from_obs(chosen)
        if len(hits) == 1:
            return self._bound_from_obs(hits[0])
        return None

    def _bound_from_obs(self, obs) -> dict:
        spec = self.cfg.objects.get(obs.id)
        return {
            "object_id": obs.id,
            "name": obs.name,
            "zone": obs.zone,
            "table_x": obs.table_x if obs.table_x is not None else 0,
            "table_y": obs.table_y if obs.table_y is not None else 0,
            "localizable": bool(obs.localizable and obs.table_x is not None),
            "note": obs.localization_note,
            "bbox": list(obs.bbox) if obs.bbox else None,
            "aim": None,
            "close_angle": spec.close_angle if spec else grip_for_bbox(obs.bbox or [0, 0, 70, 70], self.cfg),
        }

    def _refresh_aim(self, bound: dict) -> None:
        if not bound.get("aim"):
            return
        live = self._live_object(bound["object_id"])
        if live is None or not live.bbox:
            return
        bound["bbox"] = list(live.bbox)
        bound["name"] = live.name
        bound["aim"] = aim_from_bbox(live.bbox, self.cfg)
        if self.cfg.objects.get(bound["object_id"]) is None:
            bound["close_angle"] = grip_for_bbox(live.bbox, self.cfg)

    def _aim_follow(self, bound: dict, kind: str):
        if not bound.get("aim"):
            return None

        def current() -> dict:
            live = self._live_object(bound["object_id"])
            if live is not None and live.bbox:
                bound["bbox"] = list(live.bbox)
                bound["aim"] = aim_from_bbox(live.bbox, self.cfg)
                if self.cfg.objects.get(bound["object_id"]) is None:
                    bound["close_angle"] = grip_for_bbox(live.bbox, self.cfg)
            return self._arm_pose(bound, kind)

        return current

    def _arm_pose(self, bound: dict, kind: str) -> dict:
        aim = bound.get("aim") or {}
        if kind in aim:
            return aim[kind]
        zone = self.cfg.zones[bound["zone"]]
        return getattr(zone, kind).angles()

    def _live_object(self, object_id: str):
        for item in self.vision.observations:
            if item.id == object_id:
                return item
        return None

    def _require_bound(self, step: dict) -> dict | None:
        return self._bounds().get(step.get("object_index"))

    def _bounds(self) -> dict:
        if not hasattr(self, "_bound_map"):
            self._bound_map = {}
        return self._bound_map

    def _ask(self, question: str, options: list[dict]) -> dict | None:
        with self._lock:
            self.status = "awaiting_clarification"
            self.clarification = {"question": question, "options": options}
        self._clarify.clear()
        self._touch()
        self._clarify.wait()
        if self._cancel.is_set():
            return None
        answer = self._clarify_value
        self._clarify_value = None
        with self._lock:
            self.clarification = None
            self.status = "running"
        self._touch()
        return answer

    def _remember(self, bound: dict) -> None:
        with self._lock:
            self.record["selected"] = {"id": bound["object_id"], "name": bound["name"]}
        self.vision.set_selected(bound["object_id"])

    def _sync_arm(self) -> None:
        angles = self.motion.export()["angles"]
        if all(angles.get(name) is not None for name in ARM_JOINTS):
            self.vision.arm_pose = {name: float(angles[name]) for name in ARM_JOINTS}

    def _phase_for(self, tool: str) -> None:
        mapping = {
            "scan": "scanning",
            "inspect": "scanning",
            "report_scene": "scanning",
            "locate": "target_found",
            "approach": "approaching",
            "grasp": "grasping",
            "lift": "grasping",
            "sample_odor": "sampling",
            "place": "placing",
            "park": "complete",
        }
        self._phase(mapping.get(tool, self.phase))

    def _phase(self, phase: str) -> None:
        with self._lock:
            if self.status != "stopped":
                self.phase = phase
        self._touch()

    def _log(self, phase: str, level: str, message: str) -> None:
        with self._lock:
            self._seq += 1
            self.timeline.append(
                {
                    "id": self._seq,
                    "ts": time.time(),
                    "phase": phase,
                    "level": level,
                    "message": message,
                }
            )
            if len(self.timeline) > 200:
                self.timeline = self.timeline[-200:]
        self._touch()

    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _touch(self) -> None:
        try:
            self.on_change()
        except Exception:
            pass


def _describe(items) -> str:
    if not items:
        return "The fresh frame shows no identifiable objects."
    bits = []
    for item in items:
        if item.uncertain:
            bits.append(f"{item.name} (uncertain)")
        elif item.held:
            bits.append(f"{item.name} with the gripper")
        elif item.zone:
            bits.append(f"{item.name} at {item.zone}")
        else:
            bits.append(item.name)
    return "Seeing " + ", ".join(bits) + "."
