"""
retrieval/retriever.py
───────────────────────
1. Encode query  → dense (OpenAI) + CLIP + sparse (BM25)
2. Hybrid search → Qdrant RRF fusion
3. Re-rank       → Cohere cross-encoder
4. Return top-k  RetrievedChunk objects
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings
from ingestion.embedder import (
    embed_texts,
    embed_text_clip,
    get_sparse_encoder,
)
from ingestion.vector_store import hybrid_search
from utils.models import RetrievedChunk

log = logging.getLogger(__name__)

# ─── optional Cohere re-ranker ────────────────────────────────────────────────

def _rerank_cohere(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int,
) -> list[RetrievedChunk]:
    try:
        import cohere
        co = cohere.Client(settings.cohere_api_key)
        docs = [c.chunk.text for c in chunks]
        resp = co.rerank(
            model=settings.rerank_model,
            query=query,
            documents=docs,
            top_n=top_n,
        )
        reranked: list[RetrievedChunk] = []
        for r in resp.results:
            rc = chunks[r.index]
            rc.score = r.relevance_score
            rc.rank = len(reranked)
            reranked.append(rc)
        return reranked
    except Exception as exc:
        log.warning("Cohere rerank failed (%s); returning original order.", exc)
        return chunks[:top_n]


# ─── main retriever ───────────────────────────────────────────────────────────

class MultiModalRetriever:
    """
    retrieve(query, ...) → list[RetrievedChunk]

    For text queries:
      - OpenAI text embedding for dense text matching
      - CLIP text embedding  for cross-modal image matching
      - BM25 sparse          for keyword matching

    The dense query vector sent to Qdrant is a weighted blend of OpenAI + CLIP
    embeddings so a single search covers both text and image chunks.
    """

    def __init__(
        self,
        top_k_dense: int = settings.top_k_dense,
        top_k_sparse: int = settings.top_k_sparse,
        top_k_rerank: int = settings.top_k_rerank,
        use_reranker: bool = True,
    ):
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker and bool(settings.cohere_api_key)

    def retrieve(
        self,
        query: str,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Full retrieval pipeline for one query string."""

        # ── 1. Encode query ──────────────────────────────────────────────────
        # Dense: blend OpenAI text embed + CLIP text embed
        text_vec = embed_texts([query])[0]       # 3072-d
        clip_vec = embed_text_clip([query])[0]   # 768-d (will be padded)

        import numpy as np
        # Project CLIP to 3072 by zero-padding
        clip_padded = clip_vec + [0.0] * (len(text_vec) - len(clip_vec))
        # Weighted blend: 70% text, 30% CLIP (emphasise text relevance)
        blended = [0.7 * t + 0.3 * c for t, c in zip(text_vec, clip_padded)]
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = [v / norm for v in blended]

        # Sparse
        sparse_enc = get_sparse_encoder()
        sparse_idx, sparse_val = sparse_enc.encode(query)

        # ── 2. Hybrid Qdrant search ───────────────────────────────────────────
        candidates = hybrid_search(
            dense_vec=blended,
            sparse_indices=sparse_idx,
            sparse_values=sparse_val,
            top_k=max(self.top_k_dense, self.top_k_sparse),
            filter_doc_ids=filter_doc_ids,
        )
        log.debug("Hybrid search returned %d candidates.", len(candidates))

        if not candidates:
            return []

        # ── 3. Re-rank with Cohere ────────────────────────────────────────────
        if self.use_reranker and len(candidates) > self.top_k_rerank:
            candidates = _rerank_cohere(query, candidates, top_n=self.top_k_rerank)
        else:
            candidates = candidates[: self.top_k_rerank]

        return candidates
