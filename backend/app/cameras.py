"""List the cameras ffmpeg can open through AVFoundation.

The index in this list is the same index passed to ffmpeg's avfoundation
input, so the name the operator picks is the device that is opened.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading

_LOCK = threading.Lock()
_cache: list[dict] | None = None
_error = ""
_scanning = False


def parse_avfoundation_devices(text: str) -> list[dict]:
    devices: list[dict] = []
    in_video = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not in_video:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if not match:
            continue
        index = int(match.group(1))
        name = match.group(2).strip()
        devices.append({"id": str(index), "index": index, "label": name})
    return devices


def camera_catalog(refresh: bool = False) -> dict:
    """Return the mock camera plus any discovered devices.

    The first call starts a scan and returns immediately. Pass refresh=True
    after plugging in a camera.
    """
    global _scanning
    with _LOCK:
        need_scan = refresh or _cache is None
        if need_scan and not _scanning:
            _scanning = True
            threading.Thread(target=_scan, name="camera-scan", daemon=True).start()
        devices = list(_cache or [])
        return {
            "cameras": [{"id": "mock", "label": "Mock tabletop camera"}, *devices],
            "scanning": _scanning or _cache is None,
            "error": _error,
        }


def parse_avfoundation_fallback(text: str) -> dict | None:
    """Pick a size, rate, and pixel format from an ffmpeg rejection."""
    sizes = re.findall(r"(\d+x\d+)@\[(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\]fps", text)
    pixels: list[str] = []
    in_pixels = False
    for line in text.splitlines():
        if "Supported pixel formats" in line:
            in_pixels = True
            continue
        if not in_pixels:
            continue
        name = line.strip().split()[-1] if line.strip() else ""
        if re.fullmatch(r"[0-9a-z]+", name):
            pixels.append(name)
        elif pixels:
            break
    if not sizes:
        return None
    chosen = None
    for want in ("1280x720", "640x480", "1920x1080"):
        for size, _low, high in sizes:
            if size == want:
                chosen = (size, high)
                break
        if chosen:
            break
    if chosen is None:
        chosen = (sizes[0][0], sizes[0][2])
    rate_value = float(chosen[1])
    rate = str(int(rate_value)) if rate_value.is_integer() else chosen[1]
    pixel = "uyvy422" if "uyvy422" in pixels or not pixels else pixels[0]
    return {"size": chosen[0], "rate": rate, "pixel": pixel}


def resolve_camera_index(label: str) -> int | None:
    """Find the current AVFoundation index for a camera name.

    Indexes move when a phone connects or another camera appears. The name is
    the stable identity.
    """
    global _cache, _error
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not label:
        return None
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    devices = parse_avfoundation_devices(f"{completed.stderr}\n{completed.stdout}")
    if devices:
        with _LOCK:
            _cache = devices
            _error = ""
    for device in devices:
        if device["label"] == label:
            return device["index"]
    return None


def label_for(mode: str, index: int) -> str:
    if mode != "hardware":
        return "Mock tabletop camera"
    with _LOCK:
        devices = list(_cache or [])
    for device in devices:
        if device["index"] == index:
            return device["label"]
    return f"Camera {index}"


def _scan() -> None:
    global _cache, _error, _scanning
    devices: list[dict] = []
    error = ""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        error = "ffmpeg is not installed, so attached cameras cannot be listed."
    else:
        try:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"Camera scan failed: {exc}"
        else:
            text = f"{completed.stderr}\n{completed.stdout}"
            devices = parse_avfoundation_devices(text)
            if not devices:
                error = "No cameras were reported by AVFoundation."
    with _LOCK:
        _cache = devices
        _error = error
        _scanning = False
