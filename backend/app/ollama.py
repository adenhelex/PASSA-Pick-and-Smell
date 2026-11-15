"""Address of an Ollama server, and a scan of this computer's local networks.

Only IPv4 blocks on local interfaces are probed, on port 11434. A block larger
than 1,024 addresses is narrowed to the /24 that contains this computer.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx

_LOCK = threading.Lock()
_hosts: list[dict] = []
_scanning = False
_error = ""
_checked = 0


def ollama_base(value: str) -> str:
    text = (value or "").strip().rstrip("/")
    if not text:
        return "http://127.0.0.1:11434"
    if "://" not in text:
        text = "http://" + text
    parsed = urlparse(text)
    host = parsed.hostname
    if not host:
        return "http://127.0.0.1:11434"
    port = parsed.port or 11434
    scheme = parsed.scheme or "http"
    return f"{scheme}://{host}:{port}"


def hosts_in_block(address: str, mask: str) -> list[str]:
    """Host addresses to probe for one interface. Loopback stays a single address."""
    try:
        interface = ipaddress.IPv4Interface(f"{address}/{mask}")
    except ValueError:
        return []
    network = interface.network
    if network.network_address.is_loopback:
        return ["127.0.0.1"]
    if network.prefixlen < 22:
        network = ipaddress.IPv4Network(f"{address}/24", strict=False)
    if network.num_addresses > 2048:
        return []
    return [str(host) for host in network.hosts()]


def local_ipv4_targets() -> list[str]:
    targets = {"127.0.0.1"}
    for address, mask in _interface_addresses():
        targets.update(hosts_in_block(address, mask))
    return sorted(targets, key=lambda item: (item != "127.0.0.1", ipaddress.IPv4Address(item)))


def probe_ollama(ip: str) -> dict | None:
    if not _port_open(ip, 11434, 0.2):
        return None
    url = f"http://{ip}:11434/api/tags"
    try:
        response = httpx.get(url, timeout=1.5)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or "models" not in payload:
        return None
    models = [item["name"] for item in payload.get("models", []) if isinstance(item, dict) and item.get("name")]
    return {"ip": ip, "base_url": f"http://{ip}:11434", "models": models}


def discover_ollama_hosts() -> list[dict]:
    found = []
    targets = local_ipv4_targets()
    workers = min(64, max(1, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(probe_ollama, targets):
            if result is not None:
                found.append(result)
    found.sort(key=lambda item: (item["ip"] != "127.0.0.1", ipaddress.IPv4Address(item["ip"])))
    return found


def scan_catalog(refresh: bool = False) -> dict:
    """Return hosts found so far. The first call, and refresh=true, start a scan."""
    global _scanning, _error
    with _LOCK:
        start = (refresh or (not _hosts and _checked == 0 and not _error)) and not _scanning
        if start:
            _scanning = True
            _error = ""
            threading.Thread(target=_scan, name="ollama-scan", daemon=True).start()
        return {"scanning": _scanning, "hosts": list(_hosts), "error": _error, "checked": _checked}


def reset_scan() -> None:
    global _hosts, _scanning, _error, _checked
    with _LOCK:
        _hosts = []
        _scanning = False
        _error = ""
        _checked = 0


def _scan() -> None:
    global _hosts, _scanning, _error, _checked
    try:
        found = discover_ollama_hosts()
        error = "" if found else "No Ollama server answered on this network."
        checked = len(local_ipv4_targets())
    except Exception as exc:
        found = []
        error = f"The network scan failed. {exc.__class__.__name__}"
        checked = 0
    with _LOCK:
        _hosts = found
        _error = error
        _checked = checked
        _scanning = False


def _port_open(ip: str, port: int, timeout: float) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
    except OSError:
        return False
    else:
        return True
    finally:
        sock.close()


def _interface_addresses() -> list[tuple[str, str]]:
    try:
        output = subprocess.check_output(["ifconfig"], text=True, timeout=3, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in output.splitlines():
        match = re.search(
            r"\binet\s+(\d+\.\d+\.\d+\.\d+)\s+netmask\s+(0x[0-9a-fA-F]+|\d+\.\d+\.\d+\.\d+)",
            line,
        )
        if not match:
            continue
        mask_token = match.group(2)
        if mask_token.startswith("0x"):
            mask = str(ipaddress.IPv4Address(int(mask_token, 16)))
        else:
            mask = mask_token
        found.append((match.group(1), mask))
    return found
