"""Local HTTP API. The browser sends high-level actions; UART stays here."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import httpx

from app.cameras import camera_catalog, label_for
from app.config import Config
from app.ollama import ollama_base, scan_catalog
from app.context import Context
from app.geometry import MotionRejected
from app.mission import MissionError


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or Config.load()
    ctx = Context(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        ctx.start()
        yield
        ctx.shutdown()

    app = FastAPI(title="PASSA arm", lifespan=lifespan)
    app.state.ctx = ctx
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(MissionError)
    async def mission_error(_request, exc: MissionError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    @app.exception_handler(MotionRejected)
    async def motion_error(_request, exc: MotionRejected):
        return JSONResponse({"detail": exc.message}, status_code=400)

    @app.get("/")
    def root():
        return {"service": "passa-arm", "dashboard": "http://127.0.0.1:3000"}

    @app.get("/api/state")
    def state():
        return ctx.snapshot()

    @app.get("/api/llm/scan")
    def llm_scan(refresh: bool = False):
        return scan_catalog(refresh=refresh)

    @app.get("/api/llm/models")
    def llm_models(host: str = ""):
        base = ollama_base(host or ctx.cfg.llm_base_url)
        try:
            response = httpx.get(base + "/api/tags", timeout=4)
            response.raise_for_status()
            models = [item["name"] for item in response.json().get("models", []) if item.get("name")]
            error = ""
        except Exception as exc:
            models = []
            error = f"Could not reach Ollama at {base}. {exc.__class__.__name__}"
        return {"base_url": base, "models": models, "error": error}

    @app.get("/api/devices")
    def devices(refresh: bool = False):
        catalog = camera_catalog(refresh=refresh)
        return {
            "cameras": catalog["cameras"],
            "scanning": catalog["scanning"],
            "camera_error": catalog["error"],
            "ports": _serial_ports(),
            "enose": ctx.enose.export(),
            "settings": ctx.settings(),
            "robot": ctx.motion.export(),
        }

    @app.post("/api/settings")
    def update_settings(body: SettingsIn):
        _require_idle()
        data = body.model_dump(exclude_none=True)
        if "serial_mode" in data:
            if data["serial_mode"] not in {"mock", "hardware"}:
                raise HTTPException(400, "Serial mode must be mock or hardware.")
            ctx.motion.serial_mode = data["serial_mode"]
        if "serial_port" in data:
            ctx.motion.serial_port = data["serial_port"]
        if "camera_mode" in data or "camera_index" in data:
            mode = data.get("camera_mode", ctx.vision.mode)
            if mode not in {"mock", "hardware"}:
                raise HTTPException(400, "Camera mode must be mock or hardware.")
            index = int(data.get("camera_index", ctx.vision.camera_index))
            if mode == "hardware" and index < 0:
                raise HTTPException(400, "Choose a camera from the list.")
            label = (data.get("camera_label") or "").strip() or label_for(mode, index)
            ctx.vision.select_camera(mode, index, label)
        if "llm_enabled" in data:
            ctx.cfg.llm_enabled = bool(data["llm_enabled"])
        if "llm_base_url" in data:
            ctx.cfg.llm_base_url = ollama_base(data["llm_base_url"]) if ctx.cfg.llm_api == "ollama" else data["llm_base_url"].rstrip("/")
        if "llm_model" in data:
            ctx.cfg.llm_model = data["llm_model"].strip()
        if "vlm_enabled" in data:
            ctx.cfg.vlm_enabled = bool(data["vlm_enabled"])
        if "vlm_base_url" in data:
            ctx.cfg.vlm_base_url = data["vlm_base_url"].rstrip("/")
        if "vlm_model" in data:
            ctx.cfg.vlm_model = data["vlm_model"]
        if "enose_mode" in data:
            if data["enose_mode"] not in {"mock", "hardware", "unavailable"}:
                raise HTTPException(400, "E-nose mode must be mock, hardware, or unavailable.")
            ctx.enose.mode = data["enose_mode"]
            ctx.enose.connected = data["enose_mode"] == "hardware"
            ctx.enose.last_error = ""
        if "enose_base_url" in data:
            ctx.enose.base_url = data["enose_base_url"].rstrip("/")
            ctx.cfg.enose_base_url = ctx.enose.base_url
        ctx.mission.note_settings_changed()
        ctx.publish()
        return ctx.snapshot()

    @app.post("/api/execution-mode")
    def execution_mode(body: ModeIn):
        _require_idle()
        ctx.motion.set_mode(body.mode)
        ctx.mission.note_settings_changed()
        return ctx.snapshot()

    @app.post("/api/serial/connect")
    def connect(body: ConnectIn):
        _require_idle()
        result = ctx.motion.connect(body.port, body.confirm_reset)
        ctx.publish()
        return result

    @app.post("/api/serial/disconnect")
    def disconnect():
        _require_idle()
        ctx.motion.disconnect()
        return ctx.snapshot()

    @app.post("/api/serial/status")
    def sync_status():
        if ctx.motion.moving:
            raise HTTPException(409, "Wait until the arm stops before reading status.")
        result = ctx.motion.sync_status()
        if not result.ok:
            raise HTTPException(400, result.detail)
        return ctx.snapshot()

    @app.post("/api/simulation/park")
    def simulated_park():
        _require_idle()
        ctx.motion.use_simulated_park()
        return ctx.snapshot()

    @app.post("/api/look")
    def look():
        if ctx.mission.export()["running"]:
            raise HTTPException(409, "A mission is running.")
        try:
            ctx.vision.capture("look")
        except Exception as exc:
            ctx.vision.video_error = f"Look failed: {exc}"
        ctx.vision.set_selected(None)
        ctx.publish()
        return ctx.snapshot()

    @app.post("/api/select")
    def select(body: ClarifyIn):
        if body.u is not None and body.v is not None:
            chosen = ctx.vision.choose_point(body.u, body.v)
            ctx.mission.select_object(chosen.id)
        elif body.object_id:
            ctx.mission.select_object(body.object_id)
        else:
            raise HTTPException(400, "Choose an object in the picture.")
        return {"status": ctx.mission.status, "state": ctx.snapshot()}

    @app.post("/api/command")
    def command(body: TextIn):
        outcome = ctx.mission.command(body.text)
        if outcome["kind"] == "stop":
            return {"status": "stopped", "stop": outcome["stop"], "state": ctx.snapshot()}
        return {"status": ctx.mission.status, "state": ctx.snapshot()}

    @app.post("/api/clarify")
    def clarify(body: ClarifyIn):
        ctx.mission.clarify(body.text, body.object_id)
        return ctx.snapshot()

    @app.post("/api/run")
    def run():
        return ctx.mission.run()

    @app.post("/api/stop")
    def stop():
        # Direct to the motion controller. This route does not call the planner.
        return ctx.mission.stop()

    @app.post("/api/home")
    def home():
        _require_manual()
        return _finish(ctx.motion.home())

    @app.post("/api/gripper")
    def gripper(body: GripperIn):
        _require_manual()
        return _finish(ctx.motion.gripper_action(body.action))

    @app.post("/api/jog")
    def jog(body: JogIn):
        _require_manual()
        return _finish(ctx.motion.jog(body.joint, body.delta))

    @app.get("/api/events")
    async def events():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        loop = asyncio.get_running_loop()
        ctx.subscribe(loop, queue)

        async def gen():
            try:
                yield _sse(ctx.snapshot())
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15)
                        yield _sse(payload)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                ctx.unsubscribe(queue)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/camera/stream")
    def camera_stream():
        def gen():
            while True:
                frame = ctx.vision.jpeg()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.12)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    def _require_idle() -> None:
        mission = ctx.mission.export()
        if mission["running"] or mission["status"] in {"running", "awaiting_clarification"}:
            raise HTTPException(409, "Wait until the mission is idle.")

    def _require_manual() -> None:
        mission = ctx.mission.export()
        if mission["running"] or mission["status"] in {"running", "awaiting_clarification"}:
            raise HTTPException(409, "Manual jog is disabled during an active mission.")

    return app


def _finish(result):
    if not result.ok:
        code = 409 if result.interrupted else 400
        raise HTTPException(code, result.detail)
    return {"ok": True, "detail": result.detail}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _serial_ports() -> list[dict]:
    ports = [{"device": "mock://braccio", "description": "Mock Arduino Uno"}]
    try:
        from serial.tools import list_ports
    except ImportError:
        return ports
    for port in list_ports.comports():
        ports.append({"device": port.device, "description": port.description or port.device})
    return ports


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ClarifyIn(BaseModel):
    text: str | None = None
    object_id: str | None = None
    u: float | None = None
    v: float | None = None


class ModeIn(BaseModel):
    mode: str


class ConnectIn(BaseModel):
    port: str = ""
    confirm_reset: bool = False


class SettingsIn(BaseModel):
    serial_mode: str | None = None
    serial_port: str | None = None
    camera_mode: str | None = None
    camera_index: int | None = None
    camera_label: str | None = None
    llm_enabled: bool | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    vlm_enabled: bool | None = None
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    enose_mode: str | None = None
    enose_base_url: str | None = None


class GripperIn(BaseModel):
    action: str


class JogIn(BaseModel):
    joint: str
    delta: float


app = create_app()
