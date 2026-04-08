"""
retrieval/retriever.py
───────────────────────
1. Encode query  → dense (OpenAI) + CLIP + sparse (BM25)
2. Optional HyDE → replace dense vector with hypothetical-doc embedding
3. Optional multi-query → run pipeline for each rewritten variant, merge results
4. Hybrid search → Qdrant RRF fusion
5. Re-rank       → Cohere cross-encoder
6. Return top-k  RetrievedChunk objects
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

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

    Optional enhancements (controlled by constructor flags):

    use_hyde (default False):
        Generate a hypothetical answer passage with an LLM and use its embedding
        as the dense query vector (HyDE — Hypothetical Document Embeddings).

    use_query_rewriting (default True):
        Generate n alternative phrasings of the query, run the full encode →
        hybrid-search pipeline for each, merge and deduplicate results by
        chunk.id (keeping highest score), then rerank the merged pool.

    With both flags False, behaviour is identical to the original code.
    """

    def __init__(
        self,
        top_k_dense: int = settings.top_k_dense,
        top_k_sparse: int = settings.top_k_sparse,
        top_k_rerank: int = settings.top_k_rerank,
        use_reranker: bool = True,
        use_query_rewriting: bool = settings.use_query_rewriting,
        use_hyde: bool = settings.use_hyde,
    ):
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker and bool(settings.cohere_api_key)
        self.use_query_rewriting = use_query_rewriting
        self.use_hyde = use_hyde

    # ── internal: encode one query string → run hybrid search ─────────────────

    def _encode_and_search(
        self,
        query_text: str,
        filter_doc_ids: Optional[list[str]] = None,
        dense_override: Optional[list[float]] = None,
    ) -> list[RetrievedChunk]:
        """
        Encode *query_text* and run a hybrid Qdrant search.

        If *dense_override* is provided it is used directly as the dense vector
        (used by HyDE to substitute the hypothetical-doc embedding).
        """
        if dense_override is not None:
            blended = dense_override
        else:
            text_vec = embed_texts([query_text])[0]        # 3072-d
            clip_vec = embed_text_clip([query_text])[0]    # 768-d

            # Project CLIP to 3072 by zero-padding
            clip_padded = clip_vec + [0.0] * (len(text_vec) - len(clip_vec))
            # Weighted blend: 70% text, 30% CLIP
            blended = [0.7 * t + 0.3 * c for t, c in zip(text_vec, clip_padded)]
            norm = np.linalg.norm(blended)
            if norm > 0:
                blended = (np.array(blended) / norm).tolist()

        sparse_enc = get_sparse_encoder()
        sparse_idx, sparse_val = sparse_enc.encode(query_text)

        candidates = hybrid_search(
            dense_vec=blended,
            sparse_indices=sparse_idx,
            sparse_values=sparse_val,
            top_k=max(self.top_k_dense, self.top_k_sparse),
            filter_doc_ids=filter_doc_ids,
        )
        return candidates

    # ── merge results from multiple sub-queries ────────────────────────────────

    @staticmethod
    def _merge_results(
        all_results: list[list[RetrievedChunk]],
    ) -> list[RetrievedChunk]:
        """Deduplicate by chunk.id, keep highest score."""
        best: dict[str, RetrievedChunk] = {}
        for results in all_results:
            for rc in results:
                cid = rc.chunk.id
                if cid not in best or rc.score > best[cid].score:
                    best[cid] = rc
        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)
        for rank, rc in enumerate(merged):
            rc.rank = rank
        return merged

    # ── public interface ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Full retrieval pipeline for one query string."""

        # ── 1. HyDE: build a hypothetical-doc dense vector ────────────────────
        hyde_vec: Optional[list[float]] = None
        if self.use_hyde:
            from retrieval.query_rewriter import generate_hyde_doc
            hyde_doc = generate_hyde_doc(query)
            log.debug("HyDE passage (%d chars): %s…", len(hyde_doc), hyde_doc[:80])
            raw_vec = embed_texts([hyde_doc])[0]
            norm = np.linalg.norm(raw_vec)
            hyde_vec = (np.array(raw_vec) / norm).tolist() if norm > 0 else raw_vec

        # ── 2. Build query list (original ± rewrites) ─────────────────────────
        if self.use_query_rewriting:
            from retrieval.query_rewriter import rewrite_query
            queries = rewrite_query(query, n=settings.n_rewrite_variants)
            log.debug("Query variants (%d): %s", len(queries), queries)
        else:
            queries = [query]

        # ── 3. Encode + search each variant ───────────────────────────────────
        all_results: list[list[RetrievedChunk]] = []
        for q in queries:
            # HyDE vector only for the original query; rewrites use their own encoding
            dense_ovr = hyde_vec if (self.use_hyde and q == query) else None
            candidates = self._encode_and_search(q, filter_doc_ids, dense_ovr)
            log.debug("Query '%s…' → %d candidates", q[:40], len(candidates))
            all_results.append(candidates)

        # ── 4. Merge + deduplicate ─────────────────────────────────────────────
        candidates = self._merge_results(all_results)
        log.debug("Merged pool: %d unique candidates.", len(candidates))

        if not candidates:
            return []

        # ── 5. Re-rank with Cohere ─────────────────────────────────────────────
        if self.use_reranker and len(candidates) > self.top_k_rerank:
            candidates = _rerank_cohere(query, candidates, top_n=self.top_k_rerank)
        else:
            candidates = candidates[: self.top_k_rerank]

        return candidates