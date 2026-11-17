"""PostgreSQL log of what the arm picked and what the e-nose smelled.

`memory` is only for tests. The running app uses the URL in config or DATABASE_URL.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

_CREATE = """
CREATE TABLE IF NOT EXISTS smell_samples (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    object_name TEXT NOT NULL,
    object_id TEXT NOT NULL DEFAULT '',
    is_container BOOLEAN NOT NULL DEFAULT FALSE,
    contents TEXT NOT NULL DEFAULT '',
    r0 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r1 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r2 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r3 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r4 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r5 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r6 DOUBLE PRECISION NOT NULL DEFAULT 0,
    r7 DOUBLE PRECISION NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT ''
)
"""

_INSERT = """
INSERT INTO smell_samples (
    object_name, object_id, is_container, contents,
    r0, r1, r2, r3, r4, r5, r6, r7, source
) VALUES (
    %(object_name)s, %(object_id)s, %(is_container)s, %(contents)s,
    %(r0)s, %(r1)s, %(r2)s, %(r3)s, %(r4)s, %(r5)s, %(r6)s, %(r7)s, %(source)s
)
RETURNING id, created_at
"""

_LIST = """
SELECT id, created_at, object_name, object_id, is_container, contents,
       r0, r1, r2, r3, r4, r5, r6, r7, source
FROM smell_samples
ORDER BY id DESC
LIMIT %(limit)s
"""


def eight(values) -> list[float]:
    nums: list[float] = []
    for value in values or []:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            nums.append(0.0)
    nums = (nums + [0.0] * 8)[:8]
    return [round(item, 4) for item in nums]


class SampleStore:
    def __init__(self, url: str):
        self.url = (url or "").strip()
        self.ready = False
        self.error = ""
        self._memory: list[dict] = []
        self._next = 1
        self._lock = threading.Lock()

    @property
    def memory(self) -> bool:
        return self.url in {"", "memory", "memory://"}

    def start(self) -> None:
        if self.memory:
            self.ready = True
            self.error = ""
            return
        try:
            import psycopg

            with psycopg.connect(self.url, connect_timeout=3) as conn:
                conn.execute(_CREATE)
                conn.commit()
            self.ready = True
            self.error = ""
        except Exception as exc:
            self.ready = False
            self.error = str(exc)

    def save(
        self,
        *,
        object_name: str,
        object_id: str,
        is_container: bool,
        contents: str,
        channels,
        source: str,
    ) -> dict:
        r0, r1, r2, r3, r4, r5, r6, r7 = eight(channels)
        row = {
            "object_name": object_name,
            "object_id": object_id,
            "is_container": bool(is_container),
            "contents": contents or "",
            "r0": r0,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "r4": r4,
            "r5": r5,
            "r6": r6,
            "r7": r7,
            "source": source or "",
        }
        if self.memory:
            with self._lock:
                saved = {
                    "id": self._next,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **row,
                    "saved": True,
                }
                self._next += 1
                self._memory.append(saved)
            return saved
        if not self.ready:
            self.start()
        if not self.ready:
            return {**row, "id": 0, "created_at": "", "saved": False, "error": self.error}
        try:
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(self.url, connect_timeout=3, row_factory=dict_row) as conn:
                saved = conn.execute(_INSERT, row).fetchone()
                conn.commit()
            created = saved["created_at"]
            return {
                "id": int(saved["id"]),
                "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
                **row,
                "saved": True,
            }
        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            return {**row, "id": 0, "created_at": "", "saved": False, "error": self.error}

    def list(self, limit: int = 40) -> list[dict]:
        if self.memory:
            with self._lock:
                rows = list(reversed(self._memory))[:limit]
            return [_public(row) for row in rows]
        if not self.ready:
            return []
        try:
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(self.url, connect_timeout=3, row_factory=dict_row) as conn:
                rows = conn.execute(_LIST, {"limit": limit}).fetchall()
            return [_public(row) for row in rows]
        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            return []

    def export(self) -> dict:
        return {
            "ready": self.ready,
            "error": "" if self.ready else self.error,
            "items": self.list(),
        }


def _public(row: dict) -> dict:
    created = row.get("created_at")
    channels = [row.get(f"r{index}", 0) or 0 for index in range(8)]
    return {
        "id": int(row.get("id") or 0),
        "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created or ""),
        "object_name": row.get("object_name") or "",
        "object_id": row.get("object_id") or "",
        "is_container": bool(row.get("is_container")),
        "contents": row.get("contents") or "",
        "channels": eight(channels),
        "source": row.get("source") or "",
    }
