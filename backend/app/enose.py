"""Mock and hardware MUI E-nose adapter.

Mock readings are always labeled as synthetic. An unavailable adapter does not
invent a smell.
"""

from __future__ import annotations

import time

import httpx

from app.config import Config

PATTERNS = {
    "orange_fruit": [1.22, 0.81, 1.15, 1.48, 0.64, 0.91, 1.33, 0.77],
    "orange_cup": [1.05, 0.97, 1.40, 0.92, 1.18, 0.73, 0.88, 1.26],
    "blue_cup": [0.88, 1.33, 1.02, 1.11, 0.79, 1.21, 0.95, 1.08],
    "green_block": [1.10, 1.10, 0.75, 1.25, 0.98, 0.84, 1.16, 0.91],
    "red_block": [0.96, 1.18, 1.07, 0.84, 1.29, 0.77, 1.02, 1.14],
}
_LABELS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")


def eight_channels(value) -> list[float] | None:
    """Pull R0–R7 out of a list or a dict. Missing sensors stay at 0."""
    nums: list[float] = []
    if isinstance(value, dict):
        labeled = []
        for name in _LABELS:
            if name in value or name.lower() in value:
                labeled.append(value.get(name, value.get(name.lower())))
        if len(labeled) == 8:
            nums = labeled
        elif isinstance(value.get("channels"), list):
            nums = value["channels"]
        else:
            return None
    elif isinstance(value, list):
        nums = value
    else:
        return None
    out: list[float] = []
    for item in nums:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(0.0)
    out = (out + [0.0] * 8)[:8]
    return [round(item, 4) for item in out]


class ENose:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = cfg.enose_mode
        self.base_url = cfg.enose_base_url
        self.timeout_s = cfg.enose_timeout_s
        self.last_error = ""
        self.connected = cfg.enose_mode == "hardware"

    def export(self) -> dict:
        if self.mode == "mock":
            detail = "Mock adapter. Every reading is synthetic."
            connected = False
        elif self.mode == "unavailable":
            detail = "Not connected."
            connected = False
        else:
            detail = self.last_error or "Hardware adapter configured."
            connected = self.connected and not self.last_error
        return {
            "mode": self.mode,
            "connected": connected,
            "base_url": self.base_url,
            "detail": detail,
        }

    def baseline(self, cancel, immediate: bool = False) -> dict:
        if self.mode == "unavailable":
            return self._unavailable()
        if self.mode == "mock":
            if not immediate:
                self._wait(cancel)
            return {
                "ok": True,
                "source": "mock",
                "stage": "baseline",
                "channels": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "detail": "Mock E-nose baseline (not a real measurement): R0–R7 held at 1.00.",
            }
        return self._hardware("baseline", {})

    def sample(self, object_id: str, object_name: str, cancel, immediate: bool = False) -> dict:
        if self.mode == "unavailable":
            return self._unavailable()
        if self.mode == "mock":
            if not immediate:
                self._wait(cancel)
            channels = eight_channels(PATTERNS.get(object_id, [1.1, 1.0, 0.9, 1.2, 1.0, 1.0, 1.0, 1.0]))
            text = ", ".join(f"R{index} {value:.2f}" for index, value in enumerate(channels or []))
            return {
                "ok": True,
                "source": "mock",
                "stage": "sample",
                "channels": channels,
                "detail": (
                    f"Mock E-nose result (not a real measurement): synthetic pattern for "
                    f"the {object_name}: {text}."
                ),
            }
        return self._hardware("sample", {"object": object_name})

    def purge(self, cancel, immediate: bool = False) -> dict:
        if self.mode == "unavailable":
            return self._unavailable()
        if self.mode == "mock":
            if not immediate:
                self._wait(cancel)
            return {
                "ok": True,
                "source": "mock",
                "stage": "purge",
                "detail": "Mock E-nose purge (not a real measurement): channels returned to baseline.",
            }
        return self._hardware("purge", {})

    def _unavailable(self) -> dict:
        return {
            "ok": False,
            "source": "unavailable",
            "detail": "Sampling could not be performed. The MUI E-nose is not connected.",
        }

    def _wait(self, cancel) -> None:
        deadline = time.time() + self.cfg.enose_step_s
        while time.time() < deadline:
            if cancel.is_set():
                return
            time.sleep(0.02)

    def _hardware(self, stage: str, payload: dict) -> dict:
        url = self.base_url.rstrip("/") + "/" + stage
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            self.connected = False
            self.last_error = str(exc)
            return {
                "ok": False,
                "source": "hardware",
                "detail": f"Sampling could not be performed. The MUI E-nose adapter did not respond ({stage}).",
            }
        self.connected = True
        self.last_error = ""
        detail = body.get("detail") or f"E-nose {stage} completed."
        reading = {"ok": True, "source": "hardware", "stage": stage, "detail": detail, "raw": body}
        channels = eight_channels(body)
        if channels is not None:
            reading["channels"] = channels
        return reading
