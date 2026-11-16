"""Capture dashboard screenshots for the deck with headless Chrome.

The app must already be running at 127.0.0.1:3000 and 127.0.0.1:8000.
Everything runs in dry-run. Nothing here arms the Uno.
"""

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333
API = "http://127.0.0.1:8000"
DASHBOARD = "http://127.0.0.1:3000"
OUT = Path(__file__).parent / "screenshots"
COMMAND = sys.argv[1] if len(sys.argv) > 1 else "pick the glass nearest the bottle"


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        API + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read() or b"{}")


class Page:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, timeout=60)
        self.next_id = 0

    def send(self, method, params=None):
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params or {}}))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("id") == self.next_id:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply.get("result", {})

    def eval(self, expression):
        result = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        return result.get("result", {}).get("value")

    def viewport(self, width, height):
        self.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 2, "mobile": False})

    def shot(self, name, clip=None):
        params = {"format": "png", "captureBeyondViewport": True}
        if clip:
            params["clip"] = {**clip, "scale": 1}
        data = self.send("Page.captureScreenshot", params)["data"]
        path = OUT / name
        path.write_bytes(base64.b64decode(data))
        print("saved", path.name)

    def panel(self, title):
        rect = self.eval(
            "(() => {"
            "const panels = [...document.querySelectorAll('section.panel')];"
            f"const hit = panels.find(p => (p.querySelector('header')?.textContent || '').trim().toLowerCase().startsWith({json.dumps(title.lower())}));"
            "if (!hit) return null;"
            "const r = hit.getBoundingClientRect();"
            "return {x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height};"
            "})()"
        )
        if not rect:
            raise RuntimeError(f"No panel named {title}")
        return rect


def wait(page, expression, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if page.eval(expression):
            return True
        time.sleep(0.5)
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={PORT}",
            "--user-data-dir=/tmp/passa-shot-profile",
            "--window-size=1600,1000",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
                page_target = next(t for t in targets if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.25)
        page = Page(page_target["webSocketDebuggerUrl"])
        page.send("Page.enable")
        page.send("Runtime.enable")
        page.viewport(1600, 1000)

        api("/api/simulation/park", {})
        page.send("Page.navigate", {"url": DASHBOARD})
        found = wait(page, "document.querySelectorAll('.frame button').length >= 2", 40)
        print("detections on screen:", found)
        time.sleep(2.5)

        page.shot("01-dashboard.png", {"x": 0, "y": 0, "width": 1600, "height": 1000})
        page.shot("02-digital-twin.png", page.panel("arm"))
        page.shot("03-camera-detections.png", page.panel("camera"))

        page.viewport(1600, 2200)
        time.sleep(1.0)
        page.shot("04-full-dashboard.png", {"x": 0, "y": 0, "width": 1600, "height": 2200})

        print("command:", COMMAND)
        planned = api("/api/command", {"text": COMMAND})
        mission = planned.get("mission") or planned.get("state", {}).get("mission") or {}
        print("status:", mission.get("status"))
        plan = mission.get("plan") or {}
        print("note:", plan.get("planner_note"))
        if mission.get("status") == "ready":
            api("/api/run", {})
            time.sleep(2.0)
            page.shot("05-command-running.png", page.panel("command"))
            page.shot("06-mission-running.png", page.panel("mission"))
            page.shot("07-twin-reaching.png", page.panel("arm"))
            deadline = time.time() + 90
            while time.time() < deadline:
                state = api("/api/state")
                if not state["mission"].get("running"):
                    break
                time.sleep(1.0)
            time.sleep(1.5)
        page.shot("08-command-result.png", page.panel("command"))
        page.shot("09-mission-timeline.png", page.panel("mission"))
        page.shot("10-robot-uart.png", page.panel("robot"))
        page.shot("11-full-after-run.png", {"x": 0, "y": 0, "width": 1600, "height": 2200})
    finally:
        chrome.terminate()


if __name__ == "__main__":
    main()
