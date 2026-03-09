"""
ingestion/vector_store.py
──────────────────────────
Qdrant wrapper:
  • create / ensure collection with named vectors
  • upsert DocumentChunks (dense + sparse)
  • hybrid search (dense + BM25 sparse fusion via RRF)
  • delete by doc_id
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import settings
from utils.models import ChunkType, DocumentChunk, RetrievedChunk

log = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# CLIP output dim is 768; text-embedding-3-large is 3072.
# After fusing image+caption, the image vector is 768-d (CLIP space).
# We store a single "dense" vector; for cross-modal we project query to CLIP
# space separately (see retriever.py).
DENSE_DIM = settings.embedding_dim   # 3072 for text chunks


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        log.info("Collection '%s' already exists.", settings.qdrant_collection)
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=DENSE_DIM,
                distance=qm.Distance.COSINE,
                on_disk=True,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False)
            ),
        },
        optimizers_config=qm.OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
    )
    log.info("Created collection '%s'.", settings.qdrant_collection)


def upsert_chunks(chunks: list[DocumentChunk], client: QdrantClient | None = None) -> int:
    """Upsert embedded chunks into Qdrant. Returns count inserted."""
    client = client or get_client()
    ensure_collection(client)

    points: list[qm.PointStruct] = []
    for chunk in chunks:
        if chunk.dense_vector is None:
            log.warning("Chunk %s has no dense vector; skipping.", chunk.id)
            continue

        # Pad / trim dense vector to DENSE_DIM if needed (CLIP is 768)
        vec = chunk.dense_vector
        if len(vec) != DENSE_DIM:
            if len(vec) < DENSE_DIM:
                vec = vec + [0.0] * (DENSE_DIM - len(vec))
            else:
                vec = vec[:DENSE_DIM]

        sparse = None
        if chunk.sparse_indices and chunk.sparse_values:
            sparse = qm.SparseVector(
                indices=chunk.sparse_indices,
                values=chunk.sparse_values,
            )

        vectors: dict = {DENSE_VECTOR_NAME: vec}
        if sparse:
            vectors[SPARSE_VECTOR_NAME] = sparse

        points.append(qm.PointStruct(
            id=chunk.id,
            vector=vectors,
            payload=chunk.to_payload(),
        ))

    if not points:
        return 0

    # Upsert in batches of 128
    batch_size = 128
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch_size],
            wait=True,
        )
    log.info("Upserted %d points into '%s'.", len(points), settings.qdrant_collection)
    return len(points)


def delete_document(doc_id: str, client: QdrantClient | None = None) -> None:
    client = client or get_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(
                    key="doc_id",
                    match=qm.MatchValue(value=doc_id),
                )]
            )
        ),
    )
    log.info("Deleted all points for doc_id=%s", doc_id)


def hybrid_search(
    dense_vec: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    top_k: int = 10,
    filter_doc_ids: Optional[list[str]] = None,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    """
    Hybrid search: dense cosine + BM25 sparse, fused with Reciprocal Rank Fusion.
    """
    client = client or get_client()

    qfilter = None
    if filter_doc_ids:
        qfilter = qm.Filter(
            must=[qm.FieldCondition(
                key="doc_id",
                match=qm.MatchAny(any=filter_doc_ids),
            )]
        )

    # Pad dense vec if needed
    if len(dense_vec) != DENSE_DIM:
        dense_vec = (dense_vec + [0.0] * DENSE_DIM)[:DENSE_DIM]

    # ── dense search ─────────────────────────────────────────────────────────
    dense_hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=(DENSE_VECTOR_NAME, dense_vec),
        limit=top_k * 2,
        with_payload=True,
        query_filter=qfilter,
    )

    # ── sparse BM25 search ────────────────────────────────────────────────────
    sparse_hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=qm.NamedSparseVector(
            name=SPARSE_VECTOR_NAME,
            vector=qm.SparseVector(indices=sparse_indices, values=sparse_values),
        ),
        limit=top_k * 2,
        with_payload=True,
        query_filter=qfilter,
    )

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    id_to_payload: dict[str, dict] = {}
    K = 60  # RRF constant

    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (K + rank + 1)
        id_to_payload[pid] = hit.payload

    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.id)
        rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (K + rank + 1)
        id_to_payload.setdefault(pid, hit.payload)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    results: list[RetrievedChunk] = []
    for rank, pid in enumerate(sorted_ids):
        p = id_to_payload[pid]
        chunk = DocumentChunk(
            id=p.get("id", pid),
            doc_id=p.get("doc_id", ""),
            doc_name=p.get("doc_name", ""),
            chunk_type=ChunkType(p.get("chunk_type", "text")),
            text=p.get("text", ""),
            image_b64=p.get("image_b64"),
            page=p.get("page", 0),
        )
        results.append(RetrievedChunk(chunk=chunk, score=rrf_scores[pid], rank=rank))

    return results
