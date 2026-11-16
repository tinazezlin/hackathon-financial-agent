"""Sessions (MOCK of the hackathon's src/core/session.py - do not edit there).

One session = one run over the reporting dates. Its id names the output folder
outputs/<session id>/ and groups the Langfuse traces. session.url is the Langfuse
session link that is submitted together with the folder (deliverables.pdf).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.core.llm import ROOT  # noqa: F401  (also loads .env)

_CURRENT: "Session | None" = None


@dataclass
class Session:
    id: str
    dir: Path
    results: dict = field(default_factory=dict)   # date -> final graph state
    hashes: dict = field(default_factory=dict)    # date -> {"pdf": sha256, "weights": sha256}

    @property
    def url(self) -> str | None:
        """Langfuse session URL (None when tracing is off)."""
        if not langfuse_enabled():
            return None
        from langfuse import get_client

        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        try:
            pid = get_client()._get_project_id()
        except Exception:
            pid = None
        return f"{host.rstrip('/')}/project/{pid}/sessions/{self.id}" if pid else None


def langfuse_enabled() -> bool:
    if os.getenv("LANGFUSE_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def start_session(outputs_dir: str | Path | None = None) -> Session:
    global _CURRENT
    base = Path(outputs_dir) if outputs_dir else ROOT / "outputs"
    sid = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    d = base / sid
    d.mkdir(parents=True, exist_ok=True)
    _CURRENT = Session(id=sid, dir=d)
    print(f"Session {sid} -> {d}  (Langfuse tracing: {'on' if langfuse_enabled() else 'off'})")
    return _CURRENT


def current_session() -> Session:
    if _CURRENT is None:
        raise RuntimeError("Call start_session() first.")
    return _CURRENT
