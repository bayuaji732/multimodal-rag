"""
observability/tracer.py
────────────────────────
Lightweight structured tracing for the RAG pipeline.

Every completed query appends one JSON line to settings.trace_log_file.
The Tracer is a thread-safe singleton.

Usage
-----
    from observability.tracer import Tracer
    tracer = Tracer()
    tracer.log(trace)
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings

log = logging.getLogger(__name__)


@dataclass
class RagTrace:
    """
    Immutable record of a single RAG query execution.

    All timing fields are in milliseconds.
    ragas_scores is populated only for /query/evaluate requests.
    """

    query: str
    hop_type: str                    # "simple" | "multi_hop"
    sub_queries: list[str]
    rewritten_queries: list[str]
    hyde_used: bool
    chunks_retrieved: int
    nli_warnings: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    model: str

    # Auto-populated fields
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ragas_scores: Optional[dict] = None


class Tracer:
    """
    Thread-safe singleton that appends RagTrace records as JSON lines to a log file.

    The log file and its parent directories are created on first write if they
    do not already exist.
    """

    _instance: Optional["Tracer"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "Tracer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._file_lock = threading.Lock()
                cls._instance._log_path = Path(settings.trace_log_file)
            return cls._instance

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, trace: RagTrace) -> None:
        """
        Serialize *trace* to JSON and append it as a single line to the log file.
        Silently logs a warning on I/O failure — never raises.
        """
        try:
            payload = asdict(trace)
            line = json.dumps(payload, ensure_ascii=False)
            with self._file_lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            log.debug("Trace %s logged to %s", trace.trace_id, self._log_path)
        except Exception as exc:
            log.warning("Tracer.log failed for trace_id=%s: %s", trace.trace_id, exc)

    def new_trace_id(self) -> str:
        """Generate a fresh trace ID without creating a trace record."""
        return str(uuid.uuid4())