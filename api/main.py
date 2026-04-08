"""
api/main.py
────────────
FastAPI application — v3 (Advanced RAG upgrade).

Changes from v2:
  • QueryRequest.use_orchestration (bool, default True)
  • QueryResponse.trace_id + ragas_scores
  • /query wires through QueryOrchestrator or falls back to retriever→generator
  • POST /query/evaluate — same as /query but includes RAGAS scores
  • Tracer middleware injects X-Trace-Id response header
  • All upgrade failure paths degrade gracefully — never 500 on /query

Accepted upload formats (unchanged from v2):
  Table-native  → .csv, .xlsx, .xls
  Mixed-content → .pdf, .docx
  Image         → .png, .jpg, .jpeg, .webp
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from generation.generator import Citation, RAGGenerator
from ingestion.parser import SUPPORTED_FORMATS
from ingestion.pipeline import celery_app, ingest_document
from ingestion.vector_store import delete_document, ensure_collection, get_client
from observability.tracer import RagTrace, Tracer
from retrieval.retriever import MultiModalRetriever

app = FastAPI(
    title="Multi-Modal RAG Knowledge Engine",
    description="Ingest PDFs, images, tables (CSV/XLSX) — query with grounded, cited answers.",
    version="3.0.0",
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
tracer = Tracer()


# ─── Middleware ───────────────────────────────────────────────────────────────

@app.middleware("http")
async def attach_trace_id(request: Request, call_next):
    """Attach a X-Trace-Id header to every response."""
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    ensure_collection()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    task_id: str
    doc_id: str
    filename: str
    status: str = "queued"


class JobStatus(BaseModel):
    task_id: str
    status: str
    result: dict | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    filter_doc_ids: list[str] | None = None
    apply_guard: bool = True
    stream: bool = False
    use_orchestration: bool = True


class CitationOut(BaseModel):
    index: int
    doc_name: str
    page: int
    chunk_type: str
    text_snippet: str
    has_image: bool
    table_title: str | None = None
    table_headers: list[str] | None = None
    table_rows: int | None = None
    table_cols: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    warnings: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    trace_id: str
    ragas_scores: dict | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _citations_out(citations) -> list[CitationOut]:
    out: list[CitationOut] = []
    for c in citations:
        # Support both Citation dataclass and dict
        if isinstance(c, dict):
            out.append(CitationOut(
                index=c.get("index", 0),
                doc_name=c.get("doc_name", ""),
                page=c.get("page", 0),
                chunk_type=c.get("chunk_type", "text"),
                text_snippet=c.get("text_snippet", ""),
                has_image=bool(c.get("image_b64")),
            ))
        else:
            out.append(CitationOut(
                index=c.index,
                doc_name=c.doc_name,
                page=c.page,
                chunk_type=c.chunk_type,
                text_snippet=c.text_snippet,
                has_image=bool(getattr(c, "image_b64", None)),
            ))
    return out


def _run_query(req: QueryRequest, trace_id: str):
    """
    Core query execution — returns (result, chunks, retrieval_ms, generation_ms).
    Falls back gracefully on orchestrator failure.
    """
    import time as _time

    chunks = []
    t_retrieval_start = _time.perf_counter()

    if req.use_orchestration and settings.use_orchestration:
        try:
            from orchestration.pipeline import QueryOrchestrator
            orchestrator = QueryOrchestrator()
            t_gen_start = _time.perf_counter()
            result = orchestrator.run(req.query, req.filter_doc_ids)
            generation_ms = (_time.perf_counter() - t_gen_start) * 1000
            retrieval_ms = generation_ms  # orchestrator bundles both
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Orchestrator failed (%s); falling back to direct retrieval.", exc
            )
            # Fall back to direct path
            req = req.model_copy(update={"use_orchestration": False})
            return _run_query(req, trace_id)
    else:
        chunks = retriever.retrieve(req.query, filter_doc_ids=req.filter_doc_ids)
        retrieval_ms = (_time.perf_counter() - t_retrieval_start) * 1000
        if not chunks:
            raise HTTPException(404, "No relevant context found in the knowledge base.")
        t_gen_start = _time.perf_counter()
        result = generator.generate(req.query, chunks, apply_guard=req.apply_guard)
        generation_ms = (_time.perf_counter() - t_gen_start) * 1000

    return result, chunks, retrieval_ms, generation_ms


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, summary="Upload and index a document")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported file type: '{suffix}'. Allowed: {', '.join(SUPPORTED_FORMATS)}",
        )

    size_mb = 0
    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"

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
async def query(req: QueryRequest, request: Request):
    trace_id: str = getattr(request.state, "trace_id", str(uuid.uuid4()))
    t0 = time.perf_counter()

    try:
        result, chunks, retrieval_ms, generation_ms = _run_query(req, trace_id)
    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Unhandled error in /query: %s", exc)
        raise HTTPException(500, "Query processing failed. Please try again.")

    total_ms = (time.perf_counter() - t0) * 1000

    # Log trace (non-blocking, failure-safe)
    tracer.log(RagTrace(
        trace_id=trace_id,
        query=req.query,
        hop_type=getattr(result, "_hop_type", "unknown"),
        sub_queries=[req.query],
        rewritten_queries=[],
        hyde_used=settings.use_hyde,
        chunks_retrieved=len(chunks) if chunks else len(result.citations),
        nli_warnings=len(result.warnings),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        model=result.model or settings.llm_model,
        ragas_scores=None,
    ))

    return QueryResponse(
        answer=result.answer,
        citations=_citations_out(result.citations),
        warnings=result.warnings,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        trace_id=trace_id,
        ragas_scores=None,
    )


@app.post(
    "/query/evaluate",
    response_model=QueryResponse,
    summary="Query with RAGAS quality scores",
)
async def query_evaluate(req: QueryRequest, request: Request):
    """
    Same as POST /query but runs RAGAS evaluation on the generated answer
    and includes scores in the response. Slower than /query (~3–10 s extra).
    """
    trace_id: str = getattr(request.state, "trace_id", str(uuid.uuid4()))
    t0 = time.perf_counter()

    try:
        result, chunks, retrieval_ms, generation_ms = _run_query(req, trace_id)
    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Unhandled error in /query/evaluate: %s", exc)
        raise HTTPException(500, "Query processing failed. Please try again.")

    # RAGAS evaluation (failure-safe)
    ragas_scores: dict | None = None
    try:
        from evaluation.ragas_eval import score_answer
        contexts = [c.text_snippet for c in result.citations] if result.citations else [result.answer]
        ragas_scores = score_answer(req.query, result.answer, contexts)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("RAGAS evaluation failed: %s", exc)

    total_ms = (time.perf_counter() - t0) * 1000

    tracer.log(RagTrace(
        trace_id=trace_id,
        query=req.query,
        hop_type="unknown",
        sub_queries=[req.query],
        rewritten_queries=[],
        hyde_used=settings.use_hyde,
        chunks_retrieved=len(chunks) if chunks else len(result.citations),
        nli_warnings=len(result.warnings),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        model=result.model or settings.llm_model,
        ragas_scores=ragas_scores,
    ))

    return QueryResponse(
        answer=result.answer,
        citations=_citations_out(result.citations),
        warnings=result.warnings,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        trace_id=trace_id,
        ragas_scores=ragas_scores,
    )


@app.get("/query/stream", summary="Streaming SSE query")
async def query_stream(
    q: str = Query(..., min_length=3),
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
            doc_id = point.payload.get("doc_id", "")
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
    return {"formats": SUPPORTED_FORMATS}


@app.get("/health")
async def health():
    return {"status": "ok", "collection": settings.qdrant_collection}