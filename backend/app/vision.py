"""Table scene, mock camera, and VLM observations.

Mock objects carry authored table coordinates. Camera pixels are localized only
through the calibrated table-plane model.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO

import httpx
from PIL import Image, ImageDraw

from app.cameras import parse_avfoundation_fallback, resolve_camera_index
from app.config import Config, SceneObject
from app.geometry import pixel_to_table

WIDTH = 640
HEIGHT = 480


@dataclass
class Observation:
    id: str
    label: str
    kind: str
    color: str
    name: str
    bbox: list[int]
    confidence: float
    uncertain: bool
    attributes: list[str]
    table_x: float | None
    table_y: float | None
    zone: str | None
    localizable: bool
    localization_note: str
    source: str
    held: bool
    frame_id: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "color": self.color,
            "name": self.name,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "uncertain": self.uncertain,
            "attributes": self.attributes,
            "table_x": self.table_x,
            "table_y": self.table_y,
            "zone": self.zone,
            "localizable": self.localizable,
            "localization_note": self.localization_note,
            "source": self.source,
            "held": self.held,
            "frame_id": self.frame_id,
        }


@dataclass
class Track:
    spec: SceneObject
    zone: str | None
    held: bool = False
    visible: bool = True


class Vision:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = cfg.camera_mode
        self.camera_index = cfg.camera_index
        self.camera_label = "Mock tabletop camera" if cfg.camera_mode != "hardware" else f"Camera {cfg.camera_index}"
        self.camera_status = ""
        self.tracks = {
            spec.id: Track(spec=spec, zone=spec.zone, held=False, visible=True)
            for spec in cfg.objects.values()
        }
        self.allow_grasp = True
        self.occlude_held = False
        self.selected_id: str | None = None
        self.observations: list[Observation] = []
        self._det_seq = 0
        self._seen_at: dict[str, float] = {}
        self._manual: Observation | None = None
        self.seen = ""
        self._reply = ""
        self.frame_id = ""
        self.frame_ts = 0.0
        self.observation_ts = 0.0
        self.observation_stale = True
        self.stale_reason = "No observation yet."
        self.video_error = ""
        self.arm_pose = dict(cfg.park.angles())
        self._jpeg = b""
        self._hardware_jpeg = b""
        self._ffmpeg_log = ""
        self._ffmpeg_spec = {"size": "1280x720", "rate": "30", "pixel": "uyvy422"}
        self._ffmpeg_attempt = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self.on_change = lambda: None
        self._render(force_obs=True)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        if self.mode == "hardware":
            self._open_ffmpeg()

    def stop(self) -> None:
        self._stop.set()
        self._close_ffmpeg()

    def select_camera(self, mode: str, index: int | None, label: str) -> None:
        """Switch the preview. The live reader is restarted only when the device changes."""
        if mode not in {"mock", "hardware"}:
            raise ValueError(mode)
        if mode == "hardware" and label and not label.startswith("Camera "):
            resolved = resolve_camera_index(label)
            if resolved is None:
                self._close_ffmpeg()
                self.mode = mode
                self.cfg.camera_mode = mode
                self.camera_label = label
                self.camera_status = ""
                self.video_error = _iphone_black_reason(label) if "iPhone" in label else f"{label} is not connected. Refresh the camera list."
                self._message_frame(self.video_error)
                self.on_change()
                return
            index = resolved
        changed = mode != self.mode or (mode == "hardware" and index is not None and index != self.camera_index)
        self.cfg.camera_mode = mode
        if not changed:
            self.camera_label = label
            self.on_change()
            return
        self._close_ffmpeg()
        with self._lock:
            self.camera_label = label
            self.mode = mode
            self.video_error = ""
            self._hardware_jpeg = b""
            if mode == "hardware" and index is not None:
                self.camera_index = index
                self.cfg.camera_index = index
                self.camera_status = f"Opening {label}…"
                self.observations = []
                self._manual = None
                self.seen = ""
                self._reply = ""
                self.observation_ts = 0.0
                self.observation_stale = False
                self.stale_reason = ""
                self.selected_id = None
            else:
                self.camera_status = ""
        if mode == "hardware":
            self._message_frame(f"Opening {label}…")
            self._ffmpeg_spec = {"size": "1280x720", "rate": "30", "pixel": "uyvy422"}
            self._black_frames = 0
            self._open_ffmpeg(0)
        else:
            self._render(force_obs=True)
        self.on_change()

    def export(self) -> dict:
        with self._lock:
            age = time.time() - self.observation_ts if self.observation_ts else None
            video_age = time.time() - self.frame_ts if self.frame_ts else None
            return {
                "mode": self.mode,
                "camera_index": self.camera_index,
                "camera_label": self.camera_label,
                "camera_status": self.camera_status,
                "frame_id": self.frame_id,
                "frame_ts": self.frame_ts,
                "video_age_s": None if video_age is None else round(video_age, 2),
                "video_stale": video_age is None or video_age > self.cfg.stale_after_s,
                "observation_ts": self.observation_ts,
                "observation_age_s": None if age is None else round(age, 2),
                "stale": self.observation_stale,
                "stale_reason": self.stale_reason if self.observation_stale else "",
                "selected_id": self.selected_id,
                "seen": self.seen,
                "items": [item.as_dict() for item in self.observations],
                "video_error": self.video_error,
            }

    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def invalidate(self, reason: str) -> None:
        with self._lock:
            self.observation_stale = True
            self.stale_reason = reason
        self.on_change()

    def choose_point(self, u: float, v: float) -> Observation:
        """Make a pickup target at a click, including an object the detector did not name."""
        with self._lock:
            frame = self._hardware_jpeg or self._jpeg
        image = _open_frame(frame)
        bbox = _box_around(image, u, v) if image is not None else _hand_box(u, v, WIDTH, HEIGHT)
        if image is None:
            image = Image.new("RGB", (WIDTH, HEIGHT), (180, 180, 180))
        color = _color_name(image, bbox)
        name = f"{color} Object".strip()
        x, y, w, h = bbox
        fix = pixel_to_table(x + w / 2, y + h / 2, self.arm_pose, self.cfg)
        obs = Observation(
            id="chosen-object",
            label="Object",
            kind="Object",
            color=color,
            name=name,
            bbox=bbox,
            confidence=1.0,
            uncertain=False,
            attributes=["chosen"],
            table_x=fix.x,
            table_y=fix.y,
            zone=None,
            localizable=fix.ok,
            localization_note="" if fix.ok else fix.reason,
            source="camera",
            held=False,
            frame_id=self.frame_id,
        )
        with self._lock:
            self._manual = obs
            self.observations = self._keep_manual([item for item in self.observations if item.id != obs.id])
            self.seen = _format_seen(self.observations, "")
            self.observation_ts = time.time()
            self.observation_stale = False
            self.stale_reason = ""
        self.on_change()
        return obs

    def set_selected(self, object_id: str | None) -> None:
        with self._lock:
            self.selected_id = object_id
        if self.mode != "hardware":
            self._render(force_obs=False)
        self.on_change()

    def capture(self, reason: str = "inspect") -> list[Observation]:
        # A move marks the last look stale. Inspect, locate, and plan reuse that look
        # until then, so the VLM is not asked again while the wrist is still there.
        if self.mode == "hardware" and reason in {"inspect", "locate", "plan"} and self._look_is_fresh():
            return list(self.observations)
        if self.mode == "hardware":
            observations = self._capture_hardware()
        else:
            observations = self._mock_observations()
        with self._lock:
            if self.mode == "hardware":
                observations = self._assign_ids(observations)
            self.observations = self._keep_manual(observations)
            self.seen = _format_seen(observations, self._reply if self.mode == "hardware" else "")
            self.observation_ts = time.time()
            self.observation_stale = False
            self.stale_reason = ""
            self.frame_id = uuid.uuid4().hex[:8]
        if self.mode != "hardware":
            self._render(force_obs=False)
        self.on_change()
        return observations

    def _look_is_fresh(self) -> bool:
        with self._lock:
            if self.observation_stale or self.video_error or not self.observation_ts:
                return False
            return time.time() - self.observation_ts < 20

    def attempt_grasp(self, object_id: str) -> bool:
        track = self.tracks.get(object_id)
        if track is None or not self.allow_grasp or track.held:
            return False
        if not track.spec.localizable or not track.zone:
            return False
        zone = self.cfg.zones.get(track.zone)
        if zone is None or not zone.reachable:
            return False
        track.held = True
        track.zone = None
        return True

    def place(self, object_id: str, zone_id: str) -> None:
        track = self.tracks[object_id]
        track.held = False
        track.zone = zone_id
        track.visible = True

    def occupant(self, zone_id: str, ignore: str | None = None) -> Track | None:
        for track in self.tracks.values():
            if track.spec.id == ignore:
                continue
            if not track.held and track.zone == zone_id:
                return track
        return None

    def _mock_observations(self) -> list[Observation]:
        frame_id = "pending"
        items: list[Observation] = []
        for track in self.tracks.values():
            if track.held and self.occlude_held:
                continue
            if not track.visible:
                continue
            spec = track.spec
            if track.held:
                bbox = [300, 150, 70, 70]
                items.append(
                    self._obs(
                        spec,
                        bbox,
                        table_x=None,
                        table_y=None,
                        zone=None,
                        localizable=False,
                        note="The object is in the gripper. Table coordinates no longer apply.",
                        source="mock",
                        held=True,
                        attributes=["with_gripper"],
                        confidence=0.62 if self.occlude_held else 0.84,
                        uncertain=False,
                        frame_id=frame_id,
                    )
                )
                continue
            if not spec.localizable or not track.zone:
                image = spec.image or {"u": 80, "v": 60, "w": 48, "h": 40}
                bbox = [int(image["u"]), int(image["v"]), int(image.get("w", 48)), int(image.get("h", 40))]
                items.append(
                    self._obs(
                        spec,
                        bbox,
                        table_x=None,
                        table_y=None,
                        zone=None,
                        localizable=False,
                        note=(
                            "This detection has an image location only. "
                            "Without a tabletop coordinate from the calibrated camera model, "
                            "it cannot be turned into a servo command."
                        ),
                        source="mock",
                        held=False,
                        attributes=["image_only"],
                        confidence=0.7,
                        uncertain=True,
                        frame_id=frame_id,
                    )
                )
                continue
            zone = self.cfg.zones[track.zone]
            bbox = self._bbox_for_zone(zone.x, zone.y)
            items.append(
                self._obs(
                    spec,
                    bbox,
                    table_x=zone.x,
                    table_y=zone.y,
                    zone=zone.id,
                    localizable=True,
                    note="Mock scene coordinate, not a camera-to-table estimate.",
                    source="mock",
                    held=False,
                    attributes=[spec.color, spec.kind],
                    confidence=0.9,
                    uncertain=False,
                    frame_id=frame_id,
                )
            )
        return items

    def _capture_hardware(self) -> list[Observation]:
        frame = self._read_hardware_frame()
        if frame is None:
            return []
        found = self._detect_frame(frame)
        return [] if found is None else found

    def _detect_frame(self, jpeg: bytes) -> list[Observation]:
        from app.detect import detect_objects

        try:
            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        except Exception:
            self._reply = ""
            self.video_error = "The camera frame could not be read."
            return []
        if image.size != (WIDTH, HEIGHT):
            image = image.resize((WIDTH, HEIGHT))
        try:
            found = detect_objects(image)
        except Exception as exc:
            self._reply = ""
            self.video_error = f"Object detection failed: {exc}"
            return None
        self._reply = ""
        self.video_error = ""
        items: list[Observation] = []
        for hit in found:
            color = _color_name(image, hit.bbox)
            name = f"{color} {hit.label}".strip()
            if any(existing.name.lower() == name.lower() for existing in items):
                name = f"{name} {len(items) + 1}"
            x, y, w, h = hit.bbox
            fix = pixel_to_table(x + w / 2, y + h / 2, self.arm_pose, self.cfg)
            items.append(
                Observation(
                    id=f"det-{hit.label}-{x}-{y}",
                    label=hit.label,
                    kind=hit.label,
                    color=color,
                    name=name,
                    bbox=hit.bbox,
                    confidence=hit.confidence,
                    uncertain=hit.confidence < 0.4,
                    attributes=[hit.label],
                    table_x=fix.x,
                    table_y=fix.y,
                    zone=None,
                    localizable=fix.ok,
                    localization_note="" if fix.ok else fix.reason,
                    source="camera",
                    held=False,
                    frame_id="",
                )
            )
        return items

    def _ask_vlm(self, jpeg: bytes) -> list[Observation]:
        import base64

        image, sent_w, sent_h, frame = _vlm_jpeg(jpeg)
        prompt = (
            "Name each visible object once. "
            "Return JSON with an observations array. Each item has label, color, and place. "
            "place is one of: upper-left, upper-center, upper-right, middle-left, middle-center, "
            "middle-right, lower-left, lower-center, lower-right. "
            "place is where the center of the object appears in the picture, not where it sits in the room. "
            "A bottle near the middle height on the left side of the picture is middle-left, even if it rests on a table. "
            "A keyboard in the lower right of the picture is lower-right. "
            "Use a different place when objects are in different parts of the picture. "
            "Do not invent an example object. Leave observations empty only when the frame is blank."
        )
        encoded = base64.b64encode(image).decode()
        base = self.cfg.vlm_base_url.rstrip("/")
        try:
            if self.cfg.vlm_api == "ollama":
                response = httpx.post(
                    base + "/api/chat",
                    json={
                        "model": self.cfg.vlm_model,
                        "stream": False,
                        "think": False,
                        "format": "json",
                        "keep_alive": "30m",
                        "options": {"temperature": 0, "num_predict": 400, "num_ctx": 4096},
                        "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                    },
                    timeout=self.cfg.vlm_timeout_s,
                )
                response.raise_for_status()
                reply = response.json()["message"]["content"]
            else:
                response = httpx.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": self.cfg.vlm_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                                ],
                            }
                        ],
                        "temperature": 0,
                    },
                    timeout=self.cfg.vlm_timeout_s,
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            self._reply = ""
            self.video_error = f"VLM request failed: {exc}"
            return []
        self._reply = reply
        self.video_error = ""
        parsed = _extract_json(reply)
        items = []
        rows = parsed.get("observations", parsed.get("objects", []))
        if not isinstance(rows, list):
            rows = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            region = _parse_region(raw)
            bbox = _cell_bbox(region, frame, 3, 3) if region else None
            if bbox is None:
                bbox = _cell_bbox(_parse_cell(raw.get("cell")), frame) or _scale_bbox(
                    _bbox_pixels(raw.get("bbox") or raw.get("box") or [0, 0, 10, 10], sent_w, sent_h),
                    sent_w,
                    sent_h,
                )
            u = bbox[0] + bbox[2] / 2
            v = bbox[1] + bbox[3] / 2
            fix = pixel_to_table(u, v, self.arm_pose, self.cfg)
            color = str(raw.get("color") or "")
            label = str(raw.get("label") or "").strip()
            if not label:
                continue
            name = f"{color} {label}".strip()
            if any(existing.name.lower() == name.lower() for existing in items):
                continue
            items.append(
                Observation(
                    id=f"vlm-{label}-{bbox[0]}-{bbox[1]}",
                    label=label,
                    kind=label,
                    color=color,
                    name=name,
                    bbox=bbox,
                    confidence=float(raw.get("confidence") or 0.5),
                    uncertain=_as_bool(raw.get("uncertain", False)),
                    attributes=list(raw.get("attributes") or []),
                    table_x=fix.x,
                    table_y=fix.y,
                    zone=None,
                    localizable=fix.ok,
                    localization_note="" if fix.ok else fix.reason,
                    source="camera",
                    held=False,
                    frame_id="",
                )
            )
        return items

    def _obs(self, spec, bbox, **kwargs) -> Observation:
        return Observation(
            id=spec.id,
            label=spec.label,
            kind=spec.kind,
            color=spec.color,
            name=spec.name,
            bbox=bbox,
            confidence=kwargs["confidence"],
            uncertain=kwargs["uncertain"],
            attributes=kwargs["attributes"],
            table_x=kwargs["table_x"],
            table_y=kwargs["table_y"],
            zone=kwargs["zone"],
            localizable=kwargs["localizable"],
            localization_note=kwargs["note"],
            source=kwargs["source"],
            held=kwargs["held"],
            frame_id=kwargs["frame_id"],
        )

    def _bbox_for_zone(self, x: float, y: float) -> list[int]:
        u, v = _table_to_pixel(x, y, self.cfg)
        u = min(max(u, 36), WIDTH - 36)
        v = min(max(v, 48), HEIGHT - 36)
        return [int(u - 28), int(v - 28), 56, 56]

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.mode != "hardware":
                self._render(force_obs=False)
            self._stop.wait(0.15)

    def _open_ffmpeg(self, attempt: int = 0) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            with self._lock:
                self.video_error = "ffmpeg is not installed, so the selected camera cannot be opened."
                self.camera_status = ""
            self._message_frame(self.video_error)
            return
        spec = self._ffmpeg_spec
        self._ffmpeg_attempt = attempt
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-f",
            "avfoundation",
            "-framerate",
            spec["rate"],
            "-pixel_format",
            spec["pixel"],
            "-video_size",
            spec["size"],
            "-i",
            f"{self.camera_index}:none",
            "-an",
            "-vf",
            "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2",
            "-f",
            "mjpeg",
            "-q:v",
            "8",
            "-flush_packets",
            "1",
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            with self._lock:
                self.video_error = f"Could not open {self.camera_label}: {exc}"
                self.camera_status = ""
            self._message_frame(self.video_error)
            return
        self._ffmpeg_log = ""
        self._proc = proc
        threading.Thread(target=self._drain_ffmpeg, args=(proc,), name="ffmpeg-camera-err", daemon=True).start()
        threading.Thread(target=self._pump_ffmpeg, args=(proc,), name="ffmpeg-camera", daemon=True).start()
        threading.Thread(target=self._watch_open, args=(proc, self.camera_label), name="ffmpeg-camera-watch", daemon=True).start()
        threading.Thread(target=self._detect_live, args=(proc,), name="yolo", daemon=True).start()

    def _close_ffmpeg(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _pump_ffmpeg(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        buffer = b""
        announced = False
        previous_note = ""
        while not self._stop.is_set() and self._proc is proc:
            chunk = stream.read(8192)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer = buffer[-1:]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    buffer = buffer[start:]
                    break
                frame = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                note = ""
                if _frame_is_black(frame):
                    # Continuity Camera often sends black frames between good ones.
                    # Keep the last good picture so the dashboard does not go black
                    # while detection still uses the previous usable frame.
                    self._black_frames = getattr(self, "_black_frames", 0) + 1
                    with self._lock:
                        has_picture = bool(self._hardware_jpeg) and not _frame_is_black(self._hardware_jpeg)
                        if self._black_frames >= 12 and not has_picture:
                            note = _iphone_black_reason(self.camera_label)
                            self.video_error = note
                            self.camera_status = ""
                        elif has_picture:
                            self.video_error = ""
                            self.camera_status = "Holding the last clear frame while the camera blinks."
                    if note and note != previous_note:
                        previous_note = note
                        self.on_change()
                    continue
                self._black_frames = 0
                try:
                    preview = self._decorate(frame)
                except Exception:
                    preview = frame
                with self._lock:
                    self._hardware_jpeg = frame
                    self._jpeg = preview
                    self.frame_ts = time.time()
                    self.video_error = ""
                    self.camera_status = ""
                if not announced:
                    announced = True
                    self.on_change()
        if self._proc is proc:
            time.sleep(0.2)
            detail = self._ffmpeg_log.strip()
            fallback = parse_avfoundation_fallback(detail)
            if (
                fallback
                and fallback != self._ffmpeg_spec
                and self._ffmpeg_attempt == 0
                and self.mode == "hardware"
            ):
                self._proc = None
                self._ffmpeg_spec = fallback
                self._open_ffmpeg(1)
                return
            message = _camera_open_error(self.camera_label, detail)
            with self._lock:
                if not self._hardware_jpeg:
                    self.video_error = message
                    self.camera_status = ""
            if not self._hardware_jpeg:
                self._message_frame(message)
                self.on_change()

    def _watch_open(self, proc: subprocess.Popen, label: str) -> None:
        if self._stop.wait(4):
            return
        if self._proc is not proc or self._hardware_jpeg:
            return
        self.video_error = _iphone_black_reason(label)
        self.camera_status = ""
        self._message_frame(self.video_error)
        self.on_change()

    def _drain_ffmpeg(self, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        text = proc.stderr.read().decode(errors="replace").strip()
        if text:
            self._ffmpeg_log = text

    def _message_frame(self, text: str) -> None:
        image = Image.new("RGB", (WIDTH, HEIGHT), (36, 42, 40))
        draw = ImageDraw.Draw(image)
        y = 180
        words = text.split()
        line = ""
        for word in words:
            trial = f"{line} {word}".strip()
            if len(trial) > 48:
                draw.text((24, y), line, fill=(244, 236, 220))
                y += 22
                line = word
            else:
                line = trial
        if line:
            draw.text((24, y), line, fill=(244, 236, 220))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=70)
        with self._lock:
            self._jpeg = buf.getvalue()
            self.frame_ts = time.time()

    def _detect_live(self, proc: subprocess.Popen) -> None:
        """Keep boxes on the objects while the camera is open."""
        while not self._stop.is_set() and self._proc is proc and self.mode == "hardware":
            with self._lock:
                frame = self._hardware_jpeg
            if not frame or _frame_is_black(frame):
                if self._stop.wait(0.1):
                    return
                continue
            started = time.time()
            try:
                found = self._detect_frame(frame)
            except Exception as exc:
                self.video_error = f"Object detection failed: {exc}"
                self.on_change()
                if self._stop.wait(0.4):
                    return
                continue
            if self._proc is not proc:
                return
            if found is not None:
                self._apply_detections(found)
                try:
                    preview = self._decorate(frame)
                except Exception:
                    preview = frame
                with self._lock:
                    self._jpeg = preview
                    self.frame_ts = time.time()
                self.on_change()
            elapsed = time.time() - started
            if self._stop.wait(max(0.0, 0.15 - elapsed)):
                return

    def _apply_detections(self, items: list[Observation]) -> None:
        with self._lock:
            self.observations = self._keep_manual(self._assign_ids(items))
            self.seen = _format_seen(self.observations, "")
            self.observation_ts = time.time()
            self.observation_stale = False
            self.stale_reason = ""
            if not self.frame_id:
                self.frame_id = uuid.uuid4().hex[:8]

    def _assign_ids(self, items: list[Observation]) -> list[Observation]:
        """Keep the same id while an object moves, so a click still means that object."""
        previous = list(self.observations)
        used: set[str] = set()
        assigned: list[Observation] = []
        now = time.time()
        for item in items:
            match = self._match_track(item, previous, used)
            if match is None:
                self._det_seq += 1
                item.id = f"det-{item.label}-{self._det_seq}"
            else:
                item.id = match.id
                used.add(match.id)
            self._seen_at[item.id] = now
            assigned.append(item)
        selected = self.selected_id
        if selected and selected not in used:
            old = next((item for item in previous if item.id == selected), None)
            if old is not None and now - self._seen_at.get(selected, 0) < 0.6:
                assigned.append(old)
        return assigned

    def _match_track(self, item: Observation, previous: list[Observation], used: set[str]) -> Observation | None:
        best: Observation | None = None
        best_score = 0.15
        for old in previous:
            if old.id in used or old.label != item.label:
                continue
            score = _iou(old.bbox, item.bbox)
            if score > best_score:
                best = old
                best_score = score
        if best is not None:
            return best
        nearest: Observation | None = None
        nearest_d = 140.0
        cx, cy = _center(item.bbox)
        for old in previous:
            if old.id in used or old.label != item.label:
                continue
            ox, oy = _center(old.bbox)
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if dist < nearest_d:
                nearest = old
                nearest_d = dist
        return nearest

    def _keep_manual(self, items: list[Observation]) -> list[Observation]:
        manual = self._manual
        if manual is None or any(item.id == manual.id for item in items):
            return items
        return [*items, manual]

    def _read_hardware_frame(self) -> bytes | None:
        with self._lock:
            frame = self._hardware_jpeg
            label = self.camera_label
        if not frame:
            self.video_error = f"No frame from {label}."
            return None
        return frame

    def _render(self, force_obs: bool) -> None:
        if self.mode == "hardware":
            return
        image = Image.new("RGB", (WIDTH, HEIGHT), (18, 26, 22))
        draw = ImageDraw.Draw(image)
        draw.rectangle((70, 90, 580, 430), fill=(42, 56, 48))
        draw.text((80, 96), "validated table", fill=(150, 176, 160))
        with self._lock:
            observations = list(self.observations)
            selected = self.selected_id
            stale = self.observation_stale
            if force_obs and not observations:
                observations = self._mock_observations()
                self.observations = observations
                self.observation_ts = time.time()
                self.observation_stale = False
                self.stale_reason = ""
                stale = False
        palette = {
            "orange": (230, 140, 40),
            "blue": (70, 130, 210),
            "green": (70, 170, 90),
            "red": (200, 70, 60),
        }
        for item in observations:
            color = palette.get(item.color, (220, 220, 220))
            x, y, w, h = item.bbox
            selected_item = item.id == selected
            width = 4 if selected_item else 2
            draw.rectangle((x, y, x + w, y + h), outline=(255, 176, 32) if selected_item else color, width=width)
            draw.rectangle((x + 8, y + 8, x + w - 8, y + h - 8), fill=color)
            tag = item.name + ("  target" if selected_item else "")
            draw.text((x, max(0, y - 14)), tag, fill=(240, 244, 236))
        draw.text((16, 16), time.strftime("%H:%M:%S"), fill=(210, 220, 210))
        if stale:
            draw.rectangle((0, 0, WIDTH, 28), fill=(120, 70, 20))
            draw.text((12, 6), "OBSERVATION STALE", fill=(255, 236, 210))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=70)
        with self._lock:
            self._jpeg = buf.getvalue()
            self.frame_ts = time.time()
            self.frame_id = self.frame_id or uuid.uuid4().hex[:8]

    def _decorate(self, jpeg: bytes) -> bytes:
        image = Image.open(BytesIO(jpeg)).convert("RGB")
        draw = ImageDraw.Draw(image)
        with self._lock:
            observations = list(self.observations)
            selected = self.selected_id
        for item in observations:
            x, y, w, h = item.bbox
            color = (255, 176, 32) if item.id == selected else (120, 220, 140)
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=70)
        return buf.getvalue()


def _vlm_jpeg(jpeg: bytes) -> tuple[bytes, int, int, tuple[int, int, int, int]]:
    """Send a smaller frame. The model names a place; the box is that part of the photo."""
    full = (0, 0, WIDTH, HEIGHT)
    try:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception:
        return jpeg, WIDTH, HEIGHT, full
    if image.width < 8 or image.height < 8:
        return jpeg, WIDTH, HEIGHT, full
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT))
    content = _content_rect(image)
    image.thumbnail((336, 336), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=55)
    return buffer.getvalue(), image.width, image.height, content


def _content_rect(image: Image.Image) -> tuple[int, int, int, int]:
    """Ignore the black bars around a padded camera frame."""
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    step = 4

    def row_level(y: int) -> float:
        return sum(pixels[x, y] for x in range(0, width, step)) / max(1, len(range(0, width, step)))

    def col_level(x: int) -> float:
        return sum(pixels[x, y] for y in range(0, height, step)) / max(1, len(range(0, height, step)))

    rows = [y for y in range(height) if row_level(y) > 18]
    cols = [x for x in range(width) if col_level(x) > 18]
    if len(rows) < height // 5 or len(cols) < width // 5:
        return 0, 0, width, height
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def _draw_grid(image: Image.Image, content: tuple[int, int, int, int], cols: int = 4, rows: int = 4) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = content
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows
    for column in range(cols + 1):
        x = int(x0 + column * cell_w)
        draw.line((x, y0, x, y1), fill=(255, 220, 0), width=2)
    for row in range(rows + 1):
        y = int(y0 + row * cell_h)
        draw.line((x0, y, x1, y), fill=(255, 220, 0), width=2)
    for row in range(rows):
        for column in range(cols):
            draw.text((int(x0 + column * cell_w + 4), int(y0 + row * cell_h + 3)), f"{column + 1},{row + 1}", fill=(255, 220, 0))
    return canvas


def _open_frame(jpeg: bytes) -> Image.Image | None:
    if not jpeg:
        return None
    try:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception:
        return None
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT))
    return image


def _hand_box(u: float, v: float, width: int, height: int) -> list[int]:
    box_w, box_h = 72, 96
    left = min(max(0, int(round(u)) - box_w // 2), max(0, width - box_w))
    top = min(max(0, int(round(v)) - box_h // 2), max(0, height - box_h))
    return [left, top, min(box_w, width - left), min(box_h, height - top)]


def _box_around(image: Image.Image, u: float, v: float) -> list[int]:
    """Grow a box from the clicked pixel. A click on the table stays hand-sized."""
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    x = int(min(width - 1, max(0, round(u))))
    y = int(min(height - 1, max(0, round(v))))
    radius = 100
    x0, x1 = max(0, x - radius), min(width, x + radius)
    y0, y1 = max(0, y - radius), min(height, y + radius)
    crop = arr[y0:y1, x0:x1].astype(np.int16)
    seed = crop[y - y0, x - x0]
    mask = np.abs(crop - seed).sum(axis=2) <= 80
    seen = np.zeros(mask.shape, dtype=bool)
    stack = [(y - y0, x - x0)]
    seen[y - y0, x - x0] = True
    min_x = max_x = x - x0
    min_y = max_y = y - y0
    count = 0
    while stack:
        cy, cx = stack.pop()
        count += 1
        min_x, max_x = min(min_x, cx), max(max_x, cx)
        min_y, max_y = min(min_y, cy), max(max_y, cy)
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    window = mask.shape[0] * mask.shape[1]
    if count < 80 or count > window * 0.55:
        return _hand_box(u, v, width, height)
    return [x0 + min_x, y0 + min_y, max(8, max_x - min_x + 1), max(8, max_y - min_y + 1)]


def _center(bbox: list[int]) -> tuple[float, float]:
    return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2


def _iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _color_name(image: Image.Image, bbox: list[int]) -> str:
    x, y, w, h = bbox
    crop = image.crop((x, y, min(image.width, x + w), min(image.height, y + h))).resize((12, 12))
    pixels = crop.load()
    width, height = crop.size
    count = max(1, width * height)
    red = green = blue = 0.0
    for yy in range(height):
        for xx in range(width):
            pixel = pixels[xx, yy]
            red += pixel[0]
            green += pixel[1]
            blue += pixel[2]
    red /= count
    green /= count
    blue /= count
    bright = (red + green + blue) / 3
    spread = max(red, green, blue) - min(red, green, blue)
    if bright > 200 and spread < 35:
        return "clear"
    if bright > 170 and spread < 30:
        return "white"
    if bright < 55:
        return "black"
    if spread < 25:
        return "gray"
    if red > green + 25 and red > blue + 25:
        return "orange" if green > 90 else "red"
    if green > red + 15 and green > blue + 15:
        return "green"
    if blue > red + 15 and blue > green + 15:
        return "blue"
    if red > 140 and green > 120 and blue < 100:
        return "yellow"
    if red > 120 and blue > 120:
        return "purple"
    return "gray"


def _parse_region(raw) -> tuple[int, int] | None:
    """Turn a picture place into a 1-based column and row on a 3 by 3 grid."""
    if not isinstance(raw, dict):
        return None
    horizontal = str(raw.get("horizontal") or "")
    vertical = str(raw.get("vertical") or "")
    if horizontal and vertical:
        column = _place_axis(horizontal, ("left",), ("right",), ("center", "middle"))
        row = _place_axis(vertical, ("upper", "top"), ("lower", "bottom"), ("middle", "center"))
        if column and row:
            return column, row
    text = str(raw.get("place") or raw.get("position") or "").lower().replace("_", " ").replace("-", " ")
    if not text.strip():
        return None
    column = _place_axis(text, ("left",), ("right",), ("center", "middle"))
    row = _place_axis(text, ("upper", "top"), ("lower", "bottom"), ("middle", "center"))
    if column and row:
        return column, row
    return None


def _place_axis(text: str, low: tuple[str, ...], high: tuple[str, ...], mid: tuple[str, ...]) -> int | None:
    folded = text.lower()
    if any(word in folded for word in low):
        return 1
    if any(word in folded for word in high):
        return 3
    if any(word in folded for word in mid):
        return 2
    return None


def _parse_cell(raw) -> tuple[int, int] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return None
    numbers = re.findall(r"\d+", str(raw or ""))
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None


def _cell_bbox(cell: tuple[int, int] | None, content: tuple[int, int, int, int], cols: int = 4, rows: int = 4) -> list[int] | None:
    if cell is None:
        return None
    column, row = cell
    if not (1 <= column <= cols and 1 <= row <= rows):
        return None
    try:
        x0, y0, x1, y1 = content
    except (TypeError, ValueError):
        return None
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows
    pad = 6
    return [
        int(x0 + (column - 1) * cell_w + pad),
        int(y0 + (row - 1) * cell_h + pad),
        max(8, int(cell_w - pad * 2)),
        max(8, int(cell_h - pad * 2)),
    ]


def _scale_bbox(bbox: list[int], sent_w: int, sent_h: int) -> list[int]:
    if sent_w == WIDTH and sent_h == HEIGHT:
        return bbox
    sx = WIDTH / sent_w
    sy = HEIGHT / sent_h
    x = min(WIDTH - 1, int(round(bbox[0] * sx)))
    y = min(HEIGHT - 1, int(round(bbox[1] * sy)))
    return [x, y, max(8, int(round(bbox[2] * sx))), max(8, int(round(bbox[3] * sy)))]


def _bbox_pixels(raw, width: int, height: int) -> list[int]:
    values: list[float] = []
    for value in list(raw)[:4]:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    while len(values) < 4:
        values.append(10.0)
    largest = max(values)
    if 0 < largest <= 1.5:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    elif largest > max(width, height) and largest <= 1000:
        values = [
            values[0] / 1000 * width,
            values[1] / 1000 * height,
            values[2] / 1000 * width,
            values[3] / 1000 * height,
        ]
    x, y, third, fourth = values
    # Width and height are the size of the object. Treat the last two numbers as
    # the far corner only when using them as a size would run off the picture.
    overflows = x + third > width * 1.02 or y + fourth > height * 1.02
    if overflows and third > x and fourth > y:
        w, h = third - x, fourth - y
    else:
        w, h = third, fourth
    x = min(max(0.0, x), max(0.0, width - 1))
    y = min(max(0.0, y), max(0.0, height - 1))
    w = min(max(8.0, w), max(8.0, width - x))
    h = min(max(8.0, h), max(8.0, height - y))
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]


def _iphone_black_reason(label: str) -> str:
    if "Desk View" in label:
        return f"{label} stays black unless the phone is set down in Desk View. Choose iPhone Camera instead."
    if "iPhone" in label:
        return (
            f"{label} sent no picture. Unlock the iPhone and leave the screen on, then choose it again. "
            "A locked phone makes Continuity Camera black."
        )
    return f"{label} did not send a picture. Choose it again after it is connected."


def _frame_is_black(jpeg: bytes) -> bool:
    try:
        image = Image.open(BytesIO(jpeg)).convert("L")
    except Exception:
        return False
    image.thumbnail((32, 24))
    pixels = list(image.get_flattened_data())
    if not pixels:
        return False
    return sum(pixels) / len(pixels) < 8


def _camera_open_error(label: str, detail: str) -> str:
    lowered = detail.lower()
    if "not authorized" in lowered or "permission" in lowered:
        return f"macOS blocked {label}. Allow the terminal to use the camera, then choose it again."
    if detail:
        return f"Could not open {label}. {detail.splitlines()[-1]}"
    return f"No frame from {label}."


def _table_to_pixel(x: float, y: float, cfg: Config) -> tuple[float, float]:
    u = 80 + (x - cfg.x_min) / (cfg.x_max - cfg.x_min) * 480
    v = 130 + (cfg.y_max - y) / (cfg.y_max - cfg.y_min) * 280
    return u, v


def _format_seen(items, reply: str) -> str:
    if items:
        names = [f"{item.name}{' (uncertain)' if item.uncertain else ''}" for item in items]
        return "Saw " + ", ".join(names) + "."
    text = " ".join((reply or "").split())
    if text:
        return text[:700]
    return "The last look did not name anything."


def _as_bool(value) -> bool:
    if isinstance(value, bool) or value is None:
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) >= 0.5
    return str(value).strip().lower() in {"1", "true", "yes"}


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # A long reply can be cut off mid-object. Keep every complete object that was named.
    rows = []
    for chunk in re.findall(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and str(obj.get("label") or "").strip():
            rows.append(obj)
    if rows:
        return {"observations": rows}
    return {}
