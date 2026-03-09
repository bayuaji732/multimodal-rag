"""
ingestion/pipeline.py
──────────────────────
Celery task: parse → chunk → embed → upsert.

Usage:
  from ingestion.pipeline import ingest_document
  result = ingest_document.delay(str(file_path), doc_id)
  result.get()  # blocks until done
"""
from __future__ import annotations

import logging
import platform
import traceback
import uuid
from pathlib import Path

from celery import Celery

from config import settings
from ingestion.embedder import embed_chunks
from ingestion.parser import chunk_text, parse_document
from ingestion.vector_store import delete_document, upsert_chunks

log = logging.getLogger(__name__)

# ─── Celery app ───────────────────────────────────────────────────────────────

celery_app = Celery(
    "multimodal_rag",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)

_IS_WINDOWS = platform.system() == "Windows"

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Windows: billiard prefork uses semaphores that crash on Windows.
    # Use "solo" (single-process, no IPC) or "threads" instead.
    worker_pool="solo" if _IS_WINDOWS else "prefork",
    task_routes={
        "ingestion.pipeline.ingest_document": {"queue": "ingestion"},
    },
)


# ─── Task ─────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="ingestion.pipeline.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
)
def ingest_document(
    self,
    file_path: str,
    doc_id: str | None = None,
    replace: bool = False,
    original_filename: str | None = None,
) -> dict:
    """
    Full ingestion pipeline for a single file.

    Parameters
    ----------
    file_path           : str        absolute path to the uploaded file
    doc_id              : str | None if None, a new UUID is generated
    replace             : bool       if True, delete existing chunks for doc_id first
    original_filename   : str | None the real filename shown in the UI
    Returns
    -------
    dict with {doc_id, doc_name, chunk_count, status}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_id = doc_id or str(uuid.uuid4())
    # Use original_filename if provided so the UI shows "report.pdf" not the UUID
    doc_name = original_filename or path.name
    log.info("[%s] Starting ingestion: %s", doc_id, doc_name)

    try:
        # 1. Optionally remove previous version
        if replace:
            log.info("[%s] Replacing existing chunks.", doc_id)
            delete_document(doc_id)

        # 2. Parse
        self.update_state(state="PROGRESS", meta={"step": "parsing", "doc_id": doc_id})
        raw_chunks = parse_document(path, doc_id=doc_id, doc_name=doc_name)
        log.info("[%s] Parsed %d raw chunks.", doc_id, len(raw_chunks))

        # 3. Chunk text (split large text chunks)
        chunks = chunk_text(
            raw_chunks,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        log.info("[%s] After chunking: %d chunks.", doc_id, len(chunks))

        # 4. Embed (dense + sparse)
        self.update_state(state="PROGRESS", meta={"step": "embedding", "doc_id": doc_id})
        chunks = embed_chunks(chunks)

        # 5. Upsert to Qdrant
        self.update_state(state="PROGRESS", meta={"step": "indexing", "doc_id": doc_id})
        n_upserted = upsert_chunks(chunks)

        log.info("[%s] Ingestion complete: %d chunks indexed.", doc_id, n_upserted)
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_count": n_upserted,
            "status": "success",
        }

    except Exception as exc:
        log.error("[%s] Ingestion failed: %s\n%s", doc_id, exc, traceback.format_exc())
        raise self.retry(exc=exc)


# ─── Synchronous convenience wrapper ──────────────────────────────────────────

def ingest_document_sync(file_path: str, doc_id: str | None = None) -> dict:
    """Run ingestion synchronously (no Celery required — useful for testing)."""
    return ingest_document.run(file_path, doc_id=doc_id)