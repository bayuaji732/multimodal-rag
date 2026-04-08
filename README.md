# Multi-Modal RAG Knowledge Engine

Production RAG pipeline that ingests PDFs, images, **and structured tables** — retrieves across
modalities using vision-language embeddings and returns grounded, cited answers.

## What's New in v2 — Table Query Support

- **Native table ingestion**: CSV, XLSX/XLS, and DOCX tables are first-class citizens alongside PDFs and images.
- **LLM table summarization**: GPT-4o generates a rich natural-language summary of every table before embedding — enabling semantic queries like *"which quarter had the highest revenue?"* to match tabular data correctly.
- **Table-aware retrieval**: BM25 sparse encoding runs over both the NL summary and raw markdown for better keyword recall.
- **Table reasoning in generation**: The system prompt enforces strict table-reading rules (read every row, compute aggregates, quote exact cell values).
- **Paginated large tables**: CSV/XLSX sheets with many rows are split into `TABLE_PAGE_SIZE`-row chunks so embeddings stay within token limits.

## Architecture

```
PDF / Image / CSV / XLSX / XLS / DOCX
        ↓
Document Parser          (pdfplumber + unstructured.io + openpyxl + csv)
        ↓ chunks (text | image | table)
Table Summarizer         (GPT-4o — NL summary per table for semantic retrieval)
        ↓
Multi-Modal Embedder     (OpenAI text-embedding-3-large for text & tables,
                          CLIP ViT-L/14 for images + BM25 sparse)
        ↓ vectors
Vector Store             (Qdrant — hybrid BM25 + dense, RRF fusion)
        ↓ context
Re-ranker                (Cohere rerank-english-v3.0)
        ↓
Generator                (GPT-4o / Claude 3.5 + table reasoning + citation grounding)
        ↓
NLI Hallucination Guard  (DeBERTa-v3)
        ↓
Streaming Answer + Citations
```

## Supported File Formats

| Format          | Content extracted                                          |
| --------------- | ---------------------------------------------------------- |
| `.pdf`          | Text (non-table regions) + tables (bounding-box) + images |
| `.csv`          | Entire file as paginated table chunk(s)                   |
| `.xlsx` / `.xls`| Every worksheet → separate paginated table chunk(s)       |
| `.docx`         | Text + table elements                                     |
| `.png` / `.jpg` / `.jpeg` / `.webp` | Image chunk (CLIP-embedded)          |

## Project Structure

```
multimodal-rag/
├── config.py                 # All settings (pydantic-settings, .env)
├── ingestion/
│   ├── parser.py             # PDF/CSV/XLSX/DOCX/image → DocumentChunk[]
│   ├── embedder.py           # OpenAI text + CLIP image + BM25 sparse; table summarization
│   ├── vector_store.py       # Qdrant CRUD + hybrid search
│   └── pipeline.py           # Celery async ingestion task
├── retrieval/
│   └── retriever.py          # Query encode → hybrid search → rerank
├── generation/
│   └── generator.py          # RAG generation + table reasoning + NLI guard + streaming
├── api/
│   └── main.py               # FastAPI endpoints
├── utils/
│   └── models.py             # DocumentChunk (with table fields), RetrievedChunk dataclasses
├── pyproject.toml
├── .python-version
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
uv sync
# or
uv add -r requirements.txt
```

### 4. Start API + Worker

```bash
# Terminal 1 — API
uv run uvicorn api.main:app --reload

# Terminal 2 — Celery worker (choose one)
uv run celery -A ingestion.pipeline.celery_app worker -Q ingestion -c 2 --loglevel=info
# Windows / no fork support:
uv run celery -A ingestion.pipeline.celery_app worker -Q ingestion --pool=solo --loglevel=info
# Thread-based:
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

## API Usage

**Check supported formats:**

```bash
curl http://localhost:8000/supported-formats
```

**Ingest a PDF:**

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@report.pdf"
# → {"task_id": "abc-123", "doc_id": "uuid", "filename": "report.pdf", "status": "queued"}
```

**Ingest a CSV or spreadsheet:**

```bash
curl -X POST http://localhost:8000/ingest -F "file=@financials.csv"
curl -X POST http://localhost:8000/ingest -F "file=@data.xlsx"
```

**Check job status:**

```bash
curl http://localhost:8000/jobs/abc-123
```

**Query (with table reasoning):**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which quarter had the highest revenue?", "apply_guard": true}'
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

**Health check:**

```bash
curl http://localhost:8000/health
```

## Key Technical Highlights

| Feature                        | Implementation                                                        |
| ------------------------------ | --------------------------------------------------------------------- |
| Multi-modal retrieval          | CLIP (images) + OpenAI text-embedding-3-large (text/tables) + BM25   |
| Table semantic retrieval       | GPT-4o summarizes each table; embedding runs on the NL summary        |
| Table keyword recall           | BM25 sparse on summary + raw markdown combined                        |
| Table reasoning in generation  | System prompt enforces row-by-row reading, exact cell quoting, aggregates |
| Large table pagination         | CSV/XLSX split into 50-row chunks at parse time                       |
| ColPali-style late interaction | Caption generation + CLIP patch-level image vectors                   |
| Hallucination guard            | DeBERTa-v3 NLI — sentences below entailment threshold flagged `[⚠ unverified]` |
| Re-ranking                     | Cohere rerank-v3 cross-encoder to boost precision@5                   |
| Streaming                      | AsyncOpenAI SSE streaming with FastAPI `StreamingResponse`            |
| Async ingestion                | Celery + Redis job queue, monitored via Flower                        |

## Demo Scenarios

1. **Charts in PDFs** — Upload a financial report PDF with embedded bar charts.
   Query: _"What was the revenue growth in Q3?"_ — answer cites the chart.

2. **CSV / XLSX table queries** — Upload a spreadsheet with financial data.
   Query: _"Which product had the highest sales in 2024?"_ — model reads every row and computes the answer.

3. **Regression comparison** — A/B test dense-only vs. hybrid retrieval recall.

4. **Hallucination demo** — Ask a question not covered in the corpus; NLI guard
   flags the fabricated sentences with `[⚠ unverified]`.

5. **Streaming UI** — `/query/stream` endpoint feeds tokens to a frontend in real time.
