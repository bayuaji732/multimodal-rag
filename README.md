# Multi-Modal RAG Knowledge Engine

Production RAG pipeline that ingests PDFs, images, and tables — retrieves across
modalities using vision-language embeddings and returns grounded, cited answers.

## Architecture

```
PDF / Image / Table
       ↓
Document Parser          (pdfplumber + unstructured.io)
       ↓ chunks
Multi-Modal Embedder     (CLIP ViT-L/14 + text-embedding-3-large)
       ↓ vectors
Vector Store             (Qdrant — hybrid BM25 + dense, RRF fusion)
       ↓ context
Re-ranker                (Cohere rerank-english-v3.0)
       ↓
Generator                (GPT-4o / Claude 3.5 + citation grounding)
       ↓
NLI Hallucination Guard  (DeBERTa-v3)
       ↓
Streaming Answer + Citations
```

## Project Structure

```
multimodal-rag/
├── config.py                 # All settings (pydantic-settings, .env)
├── ingestion/
│   ├── parser.py             # PDF/image/table → DocumentChunk[]
│   ├── embedder.py           # CLIP + OpenAI + BM25 sparse encoding
│   ├── vector_store.py       # Qdrant CRUD + hybrid search
│   └── pipeline.py           # Celery async ingestion task
├── retrieval/
│   └── retriever.py          # Query encode → hybrid search → rerank
├── generation/
│   └── generator.py          # RAG generation + NLI guard + streaming
├── api/
│   └── main.py               # FastAPI endpoints
├── utils/
│   └── models.py             # DocumentChunk, RetrievedChunk dataclasses
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Quickstart

### 1. Environment variables

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY (optional), COHERE_API_KEY (optional)
```

### 2. Start infrastructure

```bash
docker-compose up qdrant redis -d
```

### 3. Install dependencies

```bash
uv add -r requirements.txt
```

### 4. Start API + Worker

```bash
# Terminal 1 — API
uv run uvicorn api.main:app --reload

# Terminal 2 — Celery worker
uv run celery -A ingestion.pipeline.celery_app worker -Q ingestion -c 2 --loglevel=info
or
uv run celery -A ingestion.pipeline.celery_app worker -Q ingestion --pool=solo --loglevel=info
or
uv run celery -A ingestion.pipeline.celery_app worker -Q ingestion --pool=threads -c 4 --loglevel=info
```

### 5. Start the Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

Or use Docker Compose for everything:

```bash
docker-compose up --build
```

### 5. Usage

**Ingest a PDF:**

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@report.pdf"
# → {"task_id": "abc-123", "doc_id": "uuid", ...}
```

**Check job status:**

```bash
curl http://localhost:8000/jobs/abc-123
```

**Query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What accuracy did the model achieve on S&P500?"}'
```

**Streaming query:**

```bash
curl "http://localhost:8000/query/stream?q=What+accuracy+did+the+model+achieve"
```

**List documents:**

```bash
curl http://localhost:8000/documents
```

**Delete document:**

```bash
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

## Key Technical Highlights

| Feature                        | Implementation                                                |
| ------------------------------ | ------------------------------------------------------------- |
| Multi-modal retrieval          | CLIP (images) + OpenAI (text) + BM25 sparse, fused with RRF   |
| ColPali-style late interaction | Caption generation + CLIP patch-level image vectors           |
| Hallucination guard            | DeBERTa-v3 NLI — sentences below entailment threshold flagged |
| Re-ranking                     | Cohere rerank-v3 cross-encoder to boost precision@5           |
| Streaming                      | AsyncOpenAI SSE streaming with FastAPI `StreamingResponse`    |
| Async ingestion                | Celery + Redis job queue, monitored via Flower                |

## Demo Scenarios

1. **Charts in PDFs** — Upload a financial report PDF with embedded bar charts.
   Query: _"What was the revenue growth in Q3?"_ — answer cites the chart.

2. **Regression comparison** — A/B test dense-only vs. hybrid retrieval recall.

3. **Hallucination demo** — Ask a question not covered in the corpus; NLI guard
   flags the fabricated sentences with `[⚠ unverified]`.

4. **Streaming UI** — `/query/stream` endpoint feeds tokens to a frontend in real time.
