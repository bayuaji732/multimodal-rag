"""
api/main.py
────────────
FastAPI application.

Accepted upload formats (v2):
  Table-native  → .csv, .xlsx, .xls
  Mixed-content → .pdf, .docx
  Image         → .png, .jpg, .jpeg, .webp
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from generation.generator import Citation, RAGGenerator
from ingestion.parser import SUPPORTED_FORMATS          # ← single source of truth
from ingestion.pipeline import celery_app, ingest_document
from ingestion.vector_store import delete_document, ensure_collection, get_client
from retrieval.retriever import MultiModalRetriever

app = FastAPI(
    title="Multi-Modal RAG Knowledge Engine",
    description="Ingest PDFs, images, tables (CSV/XLSX) — query with grounded, cited answers.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

retriever = MultiModalRetriever()
generator = RAGGenerator()


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    ensure_collection()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    task_id:  str
    doc_id:   str
    filename: str
    status:   str = "queued"


class JobStatus(BaseModel):
    task_id: str
    status:  str
    result:  dict | None = None


class QueryRequest(BaseModel):
    query:          str  = Field(..., min_length=3, max_length=1000)
    filter_doc_ids: list[str] | None = None
    apply_guard:    bool = True
    stream:         bool = False


class CitationOut(BaseModel):
    index:        int
    doc_name:     str
    page:         int
    chunk_type:   str
    text_snippet: str
    has_image:    bool
    # table extras (None for non-table citations)
    table_title:   str | None = None
    table_headers: list[str] | None = None
    table_rows:    int | None = None
    table_cols:    int | None = None


class QueryResponse(BaseModel):
    answer:            str
    citations:         list[CitationOut]
    warnings:          list[str]
    model:             str
    prompt_tokens:     int
    completion_tokens: int


class DocumentInfo(BaseModel):
    doc_id:      str
    doc_name:    str
    chunk_count: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, summary="Upload and index a document")
async def ingest(file: UploadFile = File(...)):
    """
    Accepts: PDF, CSV, XLSX, XLS, PNG, JPG, JPEG, WEBP, DOCX.
    Dispatches an async Celery task and returns immediately with a task_id.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported file type: '{suffix}'. "
            f"Allowed: {', '.join(SUPPORTED_FORMATS)}",
        )

    size_mb = 0
    doc_id  = str(uuid.uuid4())
    dest    = UPLOAD_DIR / f"{doc_id}{suffix}"

    with dest.open("wb") as f:
        chunk = await file.read(1024 * 1024)
        while chunk:
            size_mb += len(chunk) / 1_000_000
            if size_mb > settings.max_upload_mb:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
            f.write(chunk)
            chunk = await file.read(1024 * 1024)

    task = ingest_document.delay(str(dest), doc_id=doc_id, original_filename=file.filename)
    return IngestResponse(task_id=task.id, doc_id=doc_id, filename=file.filename)


@app.get("/jobs/{task_id}", response_model=JobStatus, summary="Check ingestion job status")
async def job_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return JobStatus(
        task_id=task_id,
        status=result.status,
        result=result.result if result.ready() else None,
    )


@app.post("/query", response_model=QueryResponse, summary="Query the knowledge base")
async def query(req: QueryRequest):
    chunks = retriever.retrieve(req.query, filter_doc_ids=req.filter_doc_ids)
    if not chunks:
        raise HTTPException(404, "No relevant context found in the knowledge base.")

    result = generator.generate(
        query=req.query,
        chunks=chunks,
        apply_guard=req.apply_guard,
    )

    citations_out = [
        CitationOut(
            index        = c.index,
            doc_name     = c.doc_name,
            page         = c.page,
            chunk_type   = c.chunk_type,
            text_snippet = c.text_snippet,
            has_image    = bool(c.image_b64),
        )
        for c in result.citations
    ]

    return QueryResponse(
        answer            = result.answer,
        citations         = citations_out,
        warnings          = result.warnings,
        model             = result.model,
        prompt_tokens     = result.prompt_tokens,
        completion_tokens = result.completion_tokens,
    )


@app.get("/query/stream", summary="Streaming SSE query")
async def query_stream(
    q:       str       = Query(..., min_length=3),
    doc_ids: str | None = Query(None, description="Comma-separated doc IDs"),
):
    filter_ids = doc_ids.split(",") if doc_ids else None
    chunks = retriever.retrieve(q, filter_doc_ids=filter_ids)

    async def _event_stream() -> AsyncIterator[str]:
        if not chunks:
            yield "data: No relevant context found.\n\ndata: [DONE]\n\n"
            return
        async for token in generator.stream(q, chunks):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/documents", response_model=list[DocumentInfo], summary="List indexed documents")
async def list_documents():
    client = get_client()
    seen: dict[str, DocumentInfo] = {}
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=["doc_id", "doc_name"],
            with_vectors=False,
        )
        for point in results:
            doc_id   = point.payload.get("doc_id", "")
            doc_name = point.payload.get("doc_name", "")
            if doc_id not in seen:
                seen[doc_id] = DocumentInfo(doc_id=doc_id, doc_name=doc_name, chunk_count=0)
            seen[doc_id].chunk_count += 1
        if next_offset is None:
            break
        offset = next_offset

    return list(seen.values())


@app.delete("/documents/{doc_id}", summary="Delete a document from the index")
async def remove_document(doc_id: str):
    try:
        delete_document(doc_id)
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/supported-formats", summary="List accepted upload formats")
async def supported_formats():
    """Returns the list of file extensions the API accepts."""
    return {"formats": SUPPORTED_FORMATS}


@app.get("/health")
async def health():
    return {"status": "ok", "collection": settings.qdrant_collection}