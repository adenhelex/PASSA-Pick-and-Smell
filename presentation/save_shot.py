"""Decode a saved CDP Page.captureScreenshot response into a PNG file."""

import base64
import json
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
body = json.loads(source.read_text())
data = body.get("data") or body.get("result", {}).get("data")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(base64.b64decode(data))
print(target, target.stat().st_size)
