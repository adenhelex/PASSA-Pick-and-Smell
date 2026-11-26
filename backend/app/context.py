"""Wire the services and fan state out to server-sent events."""

from __future__ import annotations

import asyncio
import sys
import threading

from app.config import Config
from app.enose import ENose
from app.mission import Mission
from app.motion import MotionController
from app.planner import Planner
from app.store import SampleStore
from app.vision import Vision


class Context:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.motion = MotionController(cfg)
        self.vision = Vision(cfg)
        self.enose = ENose(cfg)
        self.planner = Planner(cfg)
        self.store = SampleStore(cfg.database_url)
        self.mission = Mission(cfg, self.motion, self.vision, self.enose, self.planner, self.store)
        self.motion.on_change = self.publish
        self.vision.on_change = self.publish
        self.mission.on_change = self.publish
        self._sub_lock = threading.Lock()
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self.latest: dict = {}

    def start(self) -> None:
        self.store.start()
        self.motion.start()
        self.vision.start()
        self.latest = self.snapshot()
        if "pytest" not in sys.modules:
            threading.Thread(target=self._warm_models, name="model-warm", daemon=True).start()

    def _warm_models(self) -> None:
        """Load the planner and vision models so the first decision does not wait on startup."""
        import httpx

        from app.ollama import ollama_base

        jobs: list[tuple[str, str]] = []
        if self.cfg.llm_enabled and self.cfg.llm_api == "ollama":
            jobs.append((ollama_base(self.cfg.llm_base_url), self.cfg.llm_model))
        if self.cfg.vlm_enabled and self.cfg.vlm_api == "ollama":
            jobs.append((self.cfg.vlm_base_url.rstrip("/"), self.cfg.vlm_model))
        seen: set[tuple[str, str]] = set()
        for base, model in jobs:
            if (base, model) in seen:
                continue
            seen.add((base, model))
            try:
                httpx.post(
                    base + "/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "think": False,
                        "keep_alive": "30m",
                        "options": {"num_predict": 1, "temperature": 0},
                        "messages": [{"role": "user", "content": "ok"}],
                    },
                    timeout=40,
                )
            except Exception:
                pass

    def shutdown(self) -> None:
        self.mission.stop()
        self.vision.stop()
        self.motion.shutdown()

    def snapshot(self) -> dict:
        return {
            "robot": self.motion.export(),
            "observation": self.vision.export(),
            "enose": self.enose.export(),
            "mission": self.mission.export(),
            "samples": self.store.export(),
            "settings": self.settings(),
        }

    def settings(self) -> dict:
        return {
            "execution_mode": self.motion.mode,
            "serial_mode": self.motion.serial_mode,
            "serial_port": self.motion.serial_port,
            "camera_mode": self.vision.mode,
            "camera_index": self.vision.camera_index,
            "llm_enabled": self.cfg.llm_enabled,
            "llm_base_url": self.cfg.llm_base_url,
            "llm_model": self.cfg.llm_model,
            "vlm_enabled": self.cfg.vlm_enabled,
            "vlm_base_url": self.cfg.vlm_base_url,
            "vlm_model": self.cfg.vlm_model,
            "enose_mode": self.enose.mode,
            "enose_base_url": self.enose.base_url,
            "baud": self.cfg.baud,
            "aim_offset_x": self.cfg.aim_offset_x,
            "aim_offset_y": self.cfg.aim_offset_y,
            "aim_offset_z": self.cfg.aim_offset_z,
        }

    def subscribe(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subs.append((loop, queue))

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subs = [(loop, item) for loop, item in self._subs if item is not queue]

    def publish(self) -> None:
        state = self.snapshot()
        with self._sub_lock:
            self.latest = state
            subs = list(self._subs)
        for loop, queue in subs:
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(self._offer, queue, state)
            except RuntimeError:
                continue

    @staticmethod
    def _offer(queue: asyncio.Queue, state: dict) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(state)
        except asyncio.QueueFull:
            pass
