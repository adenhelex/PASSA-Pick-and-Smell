"""One serial reader and one command dispatcher.

Stop writes `stop` immediately and does not wait behind the motion queue.
A command that times out or loses the port is never sent again.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from app.config import ARM_JOINTS, JOINTS, Config
from app.geometry import MotionRejected, compile_joint_script, jog_target, pose_problem


def parse_status(line: str) -> dict[str, float] | None:
    import re

    found: dict[str, float] = {}
    for name in JOINTS:
        match = re.search(rf"\b{name}\s*[=:]\s*(-?\d+(?:\.\d+)?)", line, re.I)
        if match:
            found[name] = float(match.group(1))
    if len(found) != len(JOINTS):
        return None
    return found


# Folded rest written into firmware/passa_braccio/passa_braccio.ino.
# Servo M6 is the cylinder. 73 parks the cup sideways.
FIRMWARE_HOME = {
    "base": 90,
    "shoulder": 45,
    "elbow": 180,
    "wrist_vertical": 180,
    "wrist_rotation": 90,
    "cylinder": 73,
}
FIRMWARE_LIMITS = {
    "base": (0, 180),
    "shoulder": (15, 165),
    "elbow": (0, 180),
    "wrist_vertical": (0, 180),
    "wrist_rotation": (0, 180),
    "cylinder": (10, 73),
}


def format_status(angles: dict[str, float], activity: str = "idle") -> str:
    parts = " ".join(f"{name}={int(round(angles[name]))}" for name in JOINTS)
    return f"STATUS {activity} {parts}"


@dataclass
class MoveResult:
    ok: bool
    interrupted: bool = False
    uncertain: bool = False
    detail: str = ""
    commands: list[str] = field(default_factory=list)


@dataclass
class StopResult:
    software_stop: bool
    uart_sent: bool
    acknowledged: bool
    detail: str


@dataclass
class _Job:
    kind: str
    payload: dict
    result: queue.Queue


class FirmwareSim:
    """Small stand-in for the Uno sketch. One command at a time, stop wins."""

    def __init__(self, move_s: float = 0.4, home: dict[str, int] | None = None, sample: int = 10):
        self.move_s = move_s
        self.home = {name: int(FIRMWARE_HOME[name]) for name in JOINTS}
        if home:
            self.home.update({name: int(round(home[name])) for name in JOINTS if name in home})
        self.sample = int(sample)
        self.angles = dict(self.home)
        self.commands: queue.Queue[str | None] = queue.Queue()
        self.out: queue.Queue[bytes] = queue.Queue()
        self.alive = True
        self.stall_next = False
        self._thread = threading.Thread(target=self._run, name="firmware-sim", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.alive = False
        self.commands.put(None)

    def submit(self, line: str) -> None:
        if not self.alive:
            raise ConnectionError("serial closed")
        self.commands.put(line)

    def readline(self) -> bytes:
        if not self.alive and self.out.empty():
            return b""
        try:
            return self.out.get(timeout=0.2)
        except queue.Empty:
            return b""

    def force_reset(self) -> None:
        self.angles = dict(self.home)
        self.out.put(b"READY Braccio UART\n")

    def _emit(self, text: str) -> None:
        self.out.put((text + "\n").encode())

    def _run(self) -> None:
        self._emit("READY Braccio UART")
        while self.alive:
            try:
                cmd = self.commands.get(timeout=0.05)
            except queue.Empty:
                continue
            if cmd is None or not self.alive:
                break
            self._handle(cmd)

    def _handle(self, cmd: str) -> None:
        if cmd == "stop":
            self._emit("OK stopped")
            return
        if cmd == "status":
            self._emit(format_status(self.angles))
            return
        if self.stall_next:
            self.stall_next = False
            self._emit("OK moving")
            while self.alive:
                try:
                    extra = self.commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                if extra == "stop":
                    self._emit("OK stopped")
                    return
                if extra is None:
                    return
            return
        if cmd == "home":
            self._emit("OK moving")
            if self._wait_or_stop(self.move_s):
                return
            self.angles = dict(self.home)
            self._emit("DONE")
            return
        if cmd.startswith("gripper"):
            self._emit("ERR the joint is cylinder; use cylinder rotate, park, or sample")
            return
        if cmd == "cylinder park":
            self._move_joint("cylinder", self.home["cylinder"])
            return
        if cmd == "cylinder sample":
            self._move_joint("cylinder", self.sample)
            return
        parts = cmd.split()
        if len(parts) == 3 and parts[1] == "rotate":
            joint = parts[0]
            try:
                angle = int(float(parts[2]))
            except ValueError:
                self._emit("ERR angle must be a whole number")
                return
            if joint not in JOINTS:
                self._emit("ERR unknown joint")
                return
            lo, hi = FIRMWARE_LIMITS[joint]
            if angle < lo or angle > hi:
                self._emit(f"ERR angle range {lo} to {hi}")
                return
            if joint == "cylinder" and angle not in (self.sample, int(self.home["cylinder"])):
                self._emit("ERR cylinder is 10 or 73 only")
                return
            self._move_joint(joint, angle)
            return
        self._emit("ERR unknown command")

    def _move_joint(self, joint: str, angle: int) -> None:
        if angle == int(self.angles[joint]):
            self._emit("DONE")
            return
        self._emit("OK moving")
        if self._wait_or_stop(self.move_s):
            return
        self.angles[joint] = angle
        self._emit("DONE")

    def _wait_or_stop(self, seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if not self.alive:
                return True
            try:
                extra = self.commands.get(timeout=0.02)
            except queue.Empty:
                continue
            if extra == "stop":
                self._emit("OK stopped")
                return True
            if extra is None:
                return True
            self._emit("ERR busy")
        return False


class SerialLink:
    def __init__(self, sim: FirmwareSim | None = None, raw=None):
        self.sim = sim
        self.raw = raw
        self._open = True

    @property
    def is_open(self) -> bool:
        if not self._open:
            return False
        if self.sim is not None:
            return self.sim.alive
        return bool(self.raw and self.raw.is_open)

    def write(self, data: bytes) -> None:
        if not self.is_open:
            raise ConnectionError("serial closed")
        if self.sim is not None:
            self.sim.submit(data.decode().strip())
            return
        self.raw.write(data)
        self.raw.flush()

    def readline(self) -> bytes:
        if self.sim is not None:
            return self.sim.readline()
        return self.raw.readline()

    def cancel_read(self) -> None:
        raw = self.raw
        if raw is None:
            return
        cancel = getattr(raw, "cancel_read", None)
        if cancel is not None:
            cancel()

    def close(self) -> None:
        self._open = False
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        elif self.raw is not None:
            try:
                self.raw.close()
            finally:
                self.raw = None


class MotionController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = cfg.mode if cfg.mode in {"dry_run", "armed"} else "dry_run"
        self.serial_mode = cfg.serial_mode
        self.serial_port = cfg.serial_port
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._lines: queue.Queue[str] = queue.Queue()
        self._cancel = threading.Event()
        self._stop_seen = threading.Event()
        self._inflight = threading.Event()
        self._mission_active = False
        self._alive = False
        self._reader_alive = False
        self._link: SerialLink | None = None
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="motion-dispatch", daemon=True)
        self._reader: threading.Thread | None = None
        self.connected = False
        self.synced = False
        self.joints_known = True
        self.joint_source = "simulated"
        self.needs_status_sync = False
        self.unexpected_reset = False
        self.moving = False
        self.command_log: list[str] = []
        self.last_error = ""
        self._angles: dict[str, float | None] = dict(cfg.park.angles())
        self._angles["cylinder"] = cfg.tool.park_servo_deg
        self.on_change = lambda: None

    def start(self) -> None:
        self._alive = True
        if not self._dispatcher.is_alive():
            self._dispatcher.start()

    def shutdown(self) -> None:
        self._alive = False
        self._cancel.set()
        self.disconnect(notify=False)
        self._jobs.put(None)
        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=2)

    def export(self) -> dict:
        with self._lock:
            known = self.joints_known
            angles = (
                {name: self._angles[name] for name in JOINTS}
                if known
                else {name: None for name in JOINTS}
            )
            return {
                "execution_mode": self.mode,
                "serial_connected": self.connected,
                "serial_mode": self.serial_mode,
                "port": self.serial_port,
                "synced": self.synced,
                "joints_known": known,
                "joint_source": self.joint_source if known else "unknown",
                "moving": self.moving,
                "activity": "moving" if self.moving else "idle",
                "angles": angles,
                "needs_status_sync": self.needs_status_sync,
                "unexpected_reset": self.unexpected_reset,
                "last_error": self.last_error,
                "pose_errors": list(self.cfg.pose_errors),
                "commands_sent": len(self.command_log),
                "uart": list(self.command_log[-40:]),
            }

    def set_mission_active(self, active: bool) -> None:
        with self._lock:
            self._mission_active = active

    def mission_is_active(self) -> bool:
        with self._lock:
            return self._mission_active

    def set_mode(self, mode: str) -> None:
        if mode not in {"dry_run", "armed"}:
            raise MotionRejected("Mode must be dry-run or armed.")
        if self.mission_is_active() or self.moving:
            raise MotionRejected("Wait until the arm is idle before changing mode.")
        with self._lock:
            self.mode = mode
        self._notify()

    def release_cancel(self) -> None:
        self._cancel.clear()

    def wait_still(self, seconds: float, on_tick) -> bool:
        """Hold the current pose. Stop cancels the wait. No joint command is sent."""
        deadline = time.time() + max(0.0, float(seconds))
        while True:
            if self._cancel.is_set():
                return False
            remaining = deadline - time.time()
            if remaining <= 0:
                return True
            try:
                on_tick(remaining)
            except Exception:
                pass
            if self._cancel.is_set():
                return False
            time.sleep(min(0.2, remaining))
        self._stop_seen.clear()

    def connect(self, port: str, confirm_reset: bool) -> dict:
        if not confirm_reset:
            raise MotionRejected(
                "Opening the port can reset the Arduino Uno and run the firmware home routine. "
                "Confirm to connect."
            )
        if self.mission_is_active():
            raise MotionRejected("Stop the mission before connecting.")
        self.disconnect(notify=False)
        self.release_cancel()
        self.serial_port = port or self.serial_port
        if self.serial_mode == "mock":
            link = SerialLink(sim=FirmwareSim(
                move_s=max(0.05, self.cfg.sim_joint_s),
                home={**self.cfg.park.angles(), "cylinder": self.cfg.tool.park_servo_deg},
                sample=int(round(self.cfg.tool.sample_servo_deg)),
            ))
        else:
            if not self.serial_port:
                raise MotionRejected("Choose the Arduino Uno serial port first.")
            link = self._acquire_hardware(self.serial_port)
        self._link = link
        self._reader_alive = True
        self._reader = threading.Thread(target=self._reader_loop, name="motion-reader", daemon=True)
        self._reader.start()
        if self.serial_mode == "mock":
            ready = self._wait_for(self.cfg.ready_token, self.cfg.ready_timeout_s)
            if ready is None:
                self.disconnect(notify=False)
                raise MotionRejected(
                    "Did not receive READY Braccio UART. The board may still be resetting, "
                    "or the sketch is not this firmware."
                )
        with self._lock:
            self.connected = True
            self.unexpected_reset = False
            self.joints_known = False
            self.synced = False
            self.joint_source = "unknown"
            self._angles = {name: None for name in JOINTS}
        self._notify()
        status = self.sync_status()
        if not status.ok:
            return {
                "connected": True,
                "synced": False,
                "detail": "The port is open but status did not synchronize. " + status.detail,
            }
        return {"connected": True, "synced": True, "detail": "READY Braccio UART. Status synchronized."}

    def disconnect(self, notify: bool = True) -> None:
        self._reader_alive = False
        with self._lock:
            link = self._link
            self._link = None
        if link is not None:
            try:
                link.cancel_read()
            except Exception:
                pass
            try:
                link.close()
            except Exception:
                pass
        reader = self._reader
        if reader is not None and reader is not threading.current_thread() and reader.is_alive():
            reader.join(timeout=1.0)
        with self._lock:
            was_connected = self.connected or self.joint_source == "firmware"
            self.connected = False
            self.synced = False
            if was_connected:
                self.joints_known = False
                self.joint_source = "unknown"
                self.needs_status_sync = True
                self._angles = {name: None for name in JOINTS}
            self.moving = False
        self._drain_lines()
        if notify:
            self._notify()

    def sync_status(self) -> MoveResult:
        if not self.connected or self._link is None:
            return MoveResult(ok=False, detail="Serial is not connected.")
        if self.moving:
            return MoveResult(ok=False, detail="Wait until the current motion finishes before reading status.")
        return self._submit(_Job("status", {}, queue.Queue()))

    def move_pose(self, arm: dict, gripper: float, slow: bool = False, follow=None) -> MoveResult:
        return self._submit(
            _Job("pose", {"arm": arm, "gripper": gripper, "slow": slow, "follow": follow}, queue.Queue())
        )

    def jog(self, joint: str, delta: float) -> MoveResult:
        return self._submit(_Job("jog", {"joint": joint, "delta": delta}, queue.Queue()))

    def cylinder_action(self, action: str) -> MoveResult:
        if action not in {"park", "sample"}:
            return MoveResult(ok=False, detail="Cylinder action must be park or sample.")
        return self._submit(_Job("cylinder", {"action": action}, queue.Queue()))

    def home(self) -> MoveResult:
        return self._submit(_Job("home", {}, queue.Queue()))

    def stop(self) -> StopResult:
        """Cancel the queue and, when the port is up and armed, send stop immediately."""
        self._cancel.set()
        self._fail_queued("Cancelled by stop.")
        result = StopResult(True, False, False, "")
        if self.mode != "armed":
            self._wait_inflight(self.cfg.stop_timeout_s)
            if not self.mission_is_active():
                self.release_cancel()
            result.detail = (
                "Software stop cancelled the queued motion. "
                "Dry-run mode did not send a UART stop."
            )
            self._notify()
            return result
        if not self.connected or self._link is None or not self._link.is_open:
            self._mark_uncertain("Serial connection lost while stopping.")
            if not self.mission_is_active():
                self.release_cancel()
            result.detail = "The serial connection is lost. The stop could not be confirmed."
            self._notify()
            return result
        self._stop_seen.clear()
        try:
            with self._write_lock:
                self._link.write(b"stop\n")
            self._record_uart("stop")
            result.uart_sent = True
        except Exception:
            self._mark_uncertain("Serial write failed during stop.")
            if not self.mission_is_active():
                self.release_cancel()
            result.detail = "The serial connection is lost. The stop could not be confirmed."
            self._notify()
            return result
        if not self._stop_seen.wait(self.cfg.stop_timeout_s):
            if not self.connected or self._link is None or not self._link.is_open:
                result.detail = "The serial connection is lost. The stop could not be confirmed."
            else:
                result.detail = (
                    "The stop was sent, but the Arduino did not acknowledge it. "
                    "The stop could not be confirmed."
                )
            self._mark_uncertain(result.detail)
            self._wait_inflight(0.3)
            if not self.mission_is_active():
                self.release_cancel()
            self._notify()
            return result
        result.acknowledged = True
        result.detail = "The Arduino acknowledged the stop. Queued actions were cancelled."
        self._mark_uncertain("Stop interrupted a hardware motion. Sync status before moving again.")
        self._wait_inflight(self.cfg.stop_timeout_s)
        if not self.mission_is_active():
            self.release_cancel()
        self._notify()
        return result

    def _submit(self, job: _Job) -> MoveResult:
        if not self._alive:
            return MoveResult(ok=False, detail="Motion controller is stopped.")
        self._jobs.put(job)
        try:
            return job.result.get(timeout=self.cfg.motion_timeout_s + 30)
        except queue.Empty:
            return MoveResult(ok=False, uncertain=True, detail="The motion dispatcher did not answer.")

    def _dispatch_loop(self) -> None:
        while self._alive:
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            if self._cancel.is_set() and job.kind != "status":
                job.result.put(MoveResult(ok=False, interrupted=True, detail="Cancelled by stop."))
                continue
            try:
                job.result.put(self._run_job(job))
            except Exception as exc:
                job.result.put(MoveResult(ok=False, detail=str(exc)))

    def _run_job(self, job: _Job) -> MoveResult:
        self._inflight.set()
        with self._lock:
            self.moving = True
        self._notify()
        try:
            if job.kind == "pose":
                return self._run_pose(
                    job.payload["arm"],
                    job.payload["gripper"],
                    job.payload["slow"],
                    job.payload.get("follow"),
                )
            if job.kind == "jog":
                return self._run_jog(job.payload["joint"], job.payload["delta"])
            if job.kind == "cylinder":
                return self._run_cylinder(job.payload["action"])
            if job.kind == "home":
                return self._run_home()
            if job.kind == "status":
                return self._run_status()
            return MoveResult(ok=False, detail="Unknown motion job.")
        finally:
            with self._lock:
                self.moving = False
            self._inflight.clear()
            self._notify()

    def _guard(self) -> str | None:
        if self.cfg.pose_errors:
            return "Taught poses fail validation: " + self.cfg.pose_errors[0]
        if self._cancel.is_set():
            return "Cancelled by stop."
        if self.mode == "dry_run":
            if not self.joints_known:
                return "Commanded joint angles are unknown. Reset the dry-run pose or sync status."
            return None
        if not self.connected:
            return "The serial port is not connected. Connect the Arduino, or switch back to dry-run."
        if not self.synced or not self.joints_known or self.needs_status_sync:
            return "Commanded angles are unknown. Query status before any motion."
        return None

    def _current(self) -> dict[str, float]:
        with self._lock:
            return {name: float(self._angles[name]) for name in JOINTS}  # type: ignore[arg-type]

    def _run_pose(self, arm: dict, gripper: float, slow: bool, follow=None) -> MoveResult:
        problem = self._guard()
        if problem:
            return MoveResult(ok=False, interrupted=problem == "Cancelled by stop.", detail=problem)
        if follow is not None and self.mode != "armed":
            return self._track_live(follow, gripper, slow)
        sent: list[str] = []
        # Armed motion still sends one joint at a time. The target is re-read
        # before each command so a moved object changes the next command.
        budget = 1 if follow is None else 12
        for _ in range(budget):
            target = arm
            if follow is not None:
                try:
                    target = follow() or arm
                except Exception:
                    target = arm
            try:
                script = compile_joint_script(self._current(), target, gripper, self.cfg)
            except MotionRejected as exc:
                if sent:
                    return MoveResult(ok=False, detail=exc.message, commands=sent)
                return MoveResult(ok=False, detail=exc.message)
            if follow is not None:
                current = self._current()
                script = [(joint, angle) for joint, angle in script if abs(current[joint] - angle) >= 3]
            if not script:
                detail = "Already at that pose." if not sent else "Pose reached."
                return MoveResult(ok=True, detail=detail, commands=sent)
            steps = script if follow is None else script[:1]
            for joint, angle in steps:
                if self._cancel.is_set():
                    return MoveResult(ok=False, interrupted=True, detail="Cancelled by stop.", commands=sent)
                outcome = self._command_joint(joint, angle, slow=slow)
                sent.append(f"{joint} rotate {angle}")
                if not outcome.ok:
                    outcome.commands = sent
                    return outcome
            if follow is None:
                return MoveResult(ok=True, detail="Pose reached.", commands=sent)
        return MoveResult(ok=True, detail="Pose reached.", commands=sent)

    def _track_live(self, follow, gripper: float, slow: bool) -> MoveResult:
        """Move every joint together toward the latest aim, at the pose rate.

        The picture is re-read every frame. A UART line is recorded only when
        the commanded angle for a joint changes, not on every frame.
        """
        sent: list[str] = []
        current = self._current()
        gripper_i = int(round(gripper))
        if abs(current["cylinder"] - gripper_i) >= 1:
            outcome = self._command_joint("cylinder", gripper_i, slow)
            sent.append(f"cylinder rotate {gripper_i}")
            if not outcome.ok:
                outcome.commands = sent
                return outcome
        duration = self.cfg.sim_joint_s * (2.2 if slow else 1.0)
        rate = self.cfg.pose_hz if self.cfg.pose_hz > 1 else 30
        dt = 1.0 / rate
        logged: dict[str, int] = {}
        stable = 0
        deadline = time.time() + 4.0
        last_target = None
        while time.time() < deadline:
            if self._cancel.is_set():
                return MoveResult(ok=False, interrupted=True, detail="Cancelled by stop.", commands=sent)
            try:
                target = follow() or last_target
            except Exception:
                target = last_target
            if not target:
                return MoveResult(ok=False, detail="The arm has no live target.", commands=sent)
            last_target = target
            problem = pose_problem(target, self.cfg, gripper_i)
            if problem:
                return MoveResult(ok=False, detail=problem, commands=sent)
            current = self._current()
            base_waiting = abs(float(target["base"]) - current["base"]) >= 3 and not (
                self.cfg.clearance_shoulder_min <= current["shoulder"] <= self.cfg.clearance_shoulder_max
            )
            steer = {joint: float(target[joint]) for joint in ARM_JOINTS}
            if base_waiting:
                steer["base"] = current["base"]
                steer["shoulder"] = float(self.cfg.transit_shoulder)
            farthest = max(abs(steer[joint] - current[joint]) for joint in ARM_JOINTS)
            if farthest <= 3:
                stable += 1
                if stable >= 3:
                    with self._lock:
                        for joint in ARM_JOINTS:
                            self._angles[joint] = float(target[joint])
                        self.joint_source = "simulated"
                        self.joints_known = True
                    self._notify()
                    return MoveResult(ok=True, detail="Pose reached.", commands=sent)
            else:
                stable = 0
            step_limit = farthest / duration * dt
            updated = dict(current)
            for joint in ARM_JOINTS:
                delta = steer[joint] - current[joint]
                if abs(delta) <= step_limit:
                    updated[joint] = steer[joint]
                else:
                    updated[joint] = current[joint] + step_limit * (1 if delta > 0 else -1)
                rounded = int(round(steer[joint]))
                if joint not in logged or abs(logged[joint] - rounded) >= 3:
                    if abs(rounded - current[joint]) >= 3:
                        logged[joint] = rounded
                        line = f"{joint} rotate {rounded}"
                        self._record_uart(line)
                        sent.append(line)
            with self._lock:
                for joint in ARM_JOINTS:
                    self._angles[joint] = updated[joint]
                self.joint_source = "simulated"
                self.joints_known = True
            self._notify()
            time.sleep(dt)
        return MoveResult(ok=True, detail="Pose reached.", commands=sent)

    def _run_jog(self, joint: str, delta: float) -> MoveResult:
        problem = self._guard()
        if problem:
            return MoveResult(ok=False, interrupted=problem == "Cancelled by stop.", detail=problem)
        try:
            angle = jog_target(self._current(), joint, delta, self.cfg)
        except MotionRejected as exc:
            return MoveResult(ok=False, detail=exc.message)
        if int(round(self._current()[joint])) == angle:
            return MoveResult(ok=True, detail="Already at that angle.")
        return self._command_joint(joint, angle, slow=False)

    def _run_cylinder(self, action: str) -> MoveResult:
        problem = self._guard()
        if problem:
            return MoveResult(ok=False, interrupted=problem == "Cancelled by stop.", detail=problem)
        if action == "park":
            angle = int(round(self.cfg.tool.park_servo_deg))
        elif action == "sample":
            angle = int(round(self.cfg.tool.sample_servo_deg))
        else:
            return MoveResult(ok=False, detail="Cylinder action must be park or sample.")
        return self._command_joint("cylinder", angle, slow=False)

    def _run_home(self) -> MoveResult:
        problem = self._guard()
        if problem:
            return MoveResult(ok=False, interrupted=problem == "Cancelled by stop.", detail=problem)
        if self.mode != "armed":
            return self._run_pose(self.cfg.park.angles(), self.cfg.tool.park_servo_deg, slow=False)
        outcome = self._exchange("home", self.cfg.motion_timeout_s)
        if outcome["kind"] != "done":
            return self._outcome_to_result(outcome, "home")
        return self._run_status()

    def _run_status(self) -> MoveResult:
        if not self.connected or self._link is None:
            return MoveResult(ok=False, detail="Serial is not connected.")
        outcome = self._exchange("status", min(2.5, self.cfg.motion_timeout_s))
        if outcome["kind"] != "status":
            self._mark_uncertain(outcome.get("detail") or "Status failed.")
            return MoveResult(ok=False, uncertain=True, detail=outcome.get("detail") or "Status failed.")
        with self._lock:
            self._angles = dict(outcome["angles"])
            self.joints_known = True
            self.synced = True
            self.needs_status_sync = False
            self.unexpected_reset = False
            self.joint_source = "firmware"
            self.last_error = ""
        self._notify()
        return MoveResult(ok=True, detail="Status synchronized.")

    def _command_joint(self, joint: str, angle: int, slow: bool) -> MoveResult:
        if joint == "cylinder" and angle not in {
            int(round(self.cfg.tool.sample_servo_deg)),
            int(round(self.cfg.tool.park_servo_deg)),
        }:
            return MoveResult(ok=False, detail="The cylinder only moves to 10° (sample) or 73° (park).")
        line = f"{joint} rotate {angle}"
        if self.mode != "armed":
            return self._simulate_joint(joint, angle, slow, line)
        outcome = self._exchange(line, self.cfg.motion_timeout_s * (2 if slow else 1))
        if outcome["kind"] == "done":
            with self._lock:
                self._angles[joint] = angle
                self.joint_source = "firmware"
            self._notify()
            return MoveResult(ok=True, detail="DONE", commands=[line])
        return self._outcome_to_result(outcome, line)

    def _record_uart(self, line: str) -> None:
        self.command_log.append(line)
        self._notify()

    def _simulate_joint(self, joint: str, angle: int, slow: bool, line: str) -> MoveResult:
        self._record_uart(line)
        start = float(self._current()[joint])
        if int(round(start)) == angle:
            return MoveResult(ok=True, detail="DONE", commands=[line])
        if joint == "cylinder":
            with self._lock:
                self._angles[joint] = angle
                self.joint_source = "simulated"
                self.joints_known = True
            self._notify()
            return MoveResult(ok=True, detail="DONE", commands=[line])
        duration = self.cfg.sim_joint_s * (2.2 if slow else 1.0)
        rate = self.cfg.pose_hz if self.cfg.pose_hz > 1 else 30
        steps = max(1, int(round(duration * rate)))
        for step in range(1, steps + 1):
            if self._cancel.is_set():
                return MoveResult(ok=False, interrupted=True, detail="Cancelled by stop.", commands=[line])
            time.sleep(duration / steps)
            if self._cancel.is_set():
                return MoveResult(ok=False, interrupted=True, detail="Cancelled by stop.", commands=[line])
            traveled = angle if step == steps else start + (angle - start) * (step / steps)
            with self._lock:
                self._angles[joint] = traveled
                self.joint_source = "simulated"
                self.joints_known = True
            self._notify()
        return MoveResult(ok=True, detail="DONE", commands=[line])

    def _exchange(self, line: str, timeout: float) -> dict:
        if self._cancel.is_set() and line != "status":
            return {"kind": "interrupted", "detail": "Cancelled by stop."}
        if self._link is None or not self._link.is_open:
            self._mark_uncertain("Serial disconnected before the command was sent.")
            return {"kind": "disconnected", "detail": "Serial disconnected before the command was sent."}
        try:
            with self._write_lock:
                self._link.write((line + "\n").encode())
        except Exception:
            self._mark_uncertain("The serial connection was lost.")
            return {"kind": "disconnected", "detail": "The serial connection was lost."}
        self._record_uart(line)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._cancel.is_set() and self._stop_seen.is_set() and line != "status":
                self._mark_uncertain("Stop interrupted the command before DONE.")
                return {"kind": "interrupted", "detail": "Stop interrupted the command before DONE."}
            if self._link is None or not self._link.is_open:
                self._mark_uncertain("The serial connection was lost while waiting.")
                return {"kind": "disconnected", "detail": "The serial connection was lost while waiting."}
            try:
                text = self._lines.get(timeout=min(0.05, max(0.01, deadline - time.time())))
            except queue.Empty:
                continue
            if text == "OK moving":
                continue
            if text == "DONE":
                if line == "status":
                    continue
                return {"kind": "done"}
            if text.upper().startswith("ERR"):
                return {"kind": "error", "detail": text}
            angles = parse_status(text)
            if angles and line == "status":
                return {"kind": "status", "angles": angles}
            if text == self.cfg.ready_token:
                # A reboot notice often sits in the buffer after Stop. A status
                # read should skip it and take the pose that follows.
                if line == "status":
                    continue
                return {
                    "kind": "reset",
                    "detail": "The Arduino reset unexpectedly. Status must be read again.",
                }
        self._mark_uncertain(f"Timed out waiting after '{line}'. The command was not sent again.")
        return {
            "kind": "timeout",
            "detail": (
                f"Timed out waiting after '{line}'. The outcome is uncertain, so it was not sent again."
            ),
        }

    def _outcome_to_result(self, outcome: dict, line: str) -> MoveResult:
        kind = outcome["kind"]
        detail = outcome.get("detail") or kind
        if kind == "interrupted":
            return MoveResult(ok=False, interrupted=True, uncertain=True, detail=detail, commands=[line])
        if kind in {"timeout", "disconnected", "reset"}:
            return MoveResult(ok=False, uncertain=True, detail=detail, commands=[line])
        return MoveResult(ok=False, detail=detail, commands=[line])

    def _acquire_hardware(self, port: str) -> SerialLink:
        """Open the Uno and wait through its USB reset until the sketch prints READY.

        Opening the port resets the board, and a second program holding the same
        device makes readline fail. Retry inside the ready window, and always
        close a port that did not answer so the next attempt can take it.
        """
        try:
            import serial
        except ImportError as exc:
            raise MotionRejected("pyserial is not installed.") from exc
        deadline = time.time() + self.cfg.ready_timeout_s
        last = "the board did not answer"
        while time.time() < deadline:
            try:
                raw = serial.Serial(port, self.cfg.baud, timeout=0.2, exclusive=True)
            except Exception as exc:
                last = str(exc).strip() or exc.__class__.__name__
                time.sleep(0.25)
                continue
            try:
                if self._read_token(raw, self.cfg.ready_token, deadline):
                    return SerialLink(raw=raw)
                last = "no READY Braccio UART"
            except Exception as exc:
                last = str(exc).strip() or exc.__class__.__name__
            try:
                raw.close()
            except Exception:
                pass
            time.sleep(0.25)
        message = self._connect_failure(port, last)
        self.last_error = message
        raise MotionRejected(message)

    def _read_token(self, raw, token: str, deadline: float) -> bool:
        while time.time() < deadline:
            line = raw.readline()
            if not getattr(raw, "is_open", True):
                return False
            if line.decode(errors="replace").strip() == token:
                return True
        return False

    def _connect_failure(self, port: str, last: str) -> str:
        lowered = last.lower()
        if any(word in lowered for word in ("busy", "multiple access", "resource busy")):
            return (
                f"{port} is already open in another program. "
                "Quit that program, then connect again. "
                f"({last})"
            )
        return (
            "Did not receive READY Braccio UART. The board may still be resetting, "
            f"or the sketch is not this firmware. Last port error: {last}"
        )

    def _reader_loop(self) -> None:
        while self._reader_alive and self._link is not None:
            link = self._link
            try:
                raw = link.readline()
            except Exception as exc:
                if self._reader_alive and link is self._link:
                    self._drop_link(f"Serial reader failed: {exc}")
                break
            if not self._reader_alive or link is not self._link:
                break
            if not link.is_open:
                if self._reader_alive:
                    self._drop_link("Serial port closed.")
                break
            if not raw:
                continue
            text = raw.decode(errors="replace").strip()
            if not text:
                continue
            if text == self.cfg.ready_token and self.connected:
                self._on_unexpected_reset()
                continue
            lowered = text.lower()
            if lowered in {"ok stopped", "stopped"} or lowered.startswith("ok stopped"):
                self._stop_seen.set()
                continue
            self._lines.put(text)
        self._reader_alive = False

    def _wait_for(self, token: str, timeout: float) -> str | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                text = self._lines.get(timeout=0.1)
            except queue.Empty:
                if self._link is not None and not self._link.is_open:
                    return None
                continue
            if text == token:
                return text
        return None

    def _drop_link(self, detail: str) -> None:
        self._reader_alive = False
        with self._lock:
            link = self._link
            self._link = None
        if link is not None:
            try:
                link.close()
            except Exception:
                pass
        self._on_disconnect(detail)

    def _on_disconnect(self, detail: str) -> None:
        with self._lock:
            self.connected = False
            self.synced = False
            self.joints_known = False
            self.joint_source = "unknown"
            self.needs_status_sync = True
            self.moving = False
            self._angles = {name: None for name in JOINTS}
            self.last_error = detail
        self._cancel.set()
        self._notify()

    def _on_unexpected_reset(self) -> None:
        with self._lock:
            self.unexpected_reset = True
            self.synced = False
            self.joints_known = False
            self.joint_source = "unknown"
            self.needs_status_sync = True
            self._angles = {name: None for name in JOINTS}
            self.last_error = (
                "Unexpected READY Braccio UART. The board reset. Sync status before moving."
            )
        self._cancel.set()
        self._lines.put(self.cfg.ready_token)
        self._notify()

    def _mark_uncertain(self, detail: str) -> None:
        with self._lock:
            self.joints_known = False
            self.synced = False
            self.needs_status_sync = True
            self.joint_source = "unknown"
            self._angles = {name: None for name in JOINTS}
            self.last_error = detail

    def _fail_queued(self, detail: str) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            if job is None:
                self._jobs.put(None)
                return
            job.result.put(MoveResult(ok=False, interrupted=True, detail=detail))

    def _drain_lines(self) -> None:
        while True:
            try:
                self._lines.get_nowait()
            except queue.Empty:
                return

    def _wait_inflight(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while self._inflight.is_set() and time.time() < deadline:
            time.sleep(0.01)

    def use_simulated_park(self) -> None:
        if self.mission_is_active() or self.moving:
            raise MotionRejected("Wait until the arm is idle.")
        if self.mode != "dry_run":
            raise MotionRejected("Simulated park is only used in dry-run.")
        with self._lock:
            self._angles = dict(self.cfg.park.angles())
            self._angles["cylinder"] = self.cfg.tool.park_servo_deg
            self.joints_known = True
            self.joint_source = "simulated"
            self.needs_status_sync = False
            self.synced = False
            self.unexpected_reset = False
            self.last_error = ""
        self.release_cancel()
        self._notify()

    def _notify(self) -> None:
        try:
            self.on_change()
        except Exception:
            pass
