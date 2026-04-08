"""
orchestration/pipeline.py
──────────────────────────
LangGraph-based multi-hop RAG orchestration.

Graph topology
──────────────
START → decompose_node → retrieve_node
          ↓ (multi_hop)          ↓ (simple)
  generate_partial_node → synthesize_node → END

All nodes are plain Python functions that accept a RAGState dict and return a
partial dict with only the keys they update. LangGraph merges these into the
running state automatically.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import TypedDict

from openai import OpenAI

from config import settings
from generation.generator import GenerationResult, RAGGenerator
from retrieval.retriever import MultiModalRetriever
from utils.models import RetrievedChunk

log = logging.getLogger(__name__)

# ─── Graph state ──────────────────────────────────────────────────────────────


class RAGState(TypedDict):
    query: str
    filter_doc_ids: list[str] | None
    sub_queries: list[str]
    hop_type: str                   # "simple" | "multi_hop"
    all_chunks: list               # list[RetrievedChunk]
    partial_answers: list[str]
    final_result: object | None    # GenerationResult | None


# ─── Helpers ──────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
Analyze this query. If answerable in one lookup, return:
{{"type": "simple", "sub_queries": ["{query}"]}}

If it requires comparing data, multi-step reasoning, or aggregating from multiple
sections, split into 2-4 focused atomic sub-questions. Return:
{{"type": "multi_hop", "sub_queries": ["q1", "q2", ...]}}

Return ONLY valid JSON.

Query: {query}
"""


def _llm_json(prompt: str) -> dict:
    """Call the configured LLM and parse JSON. Raises on failure."""
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = raw.strip("```json").strip("```").strip()
    return json.loads(raw)


def _dedup_chunks(all_results: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Merge multiple result lists, deduplicate by chunk.id keeping highest score."""
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


# ─── Graph nodes ──────────────────────────────────────────────────────────────

def decompose_node(state: RAGState) -> dict:
    """
    Classify the query as 'simple' or 'multi_hop' and optionally split it into
    atomic sub-questions. Falls back to simple / original query on any error.
    """
    query = state["query"]
    try:
        result = _llm_json(_DECOMPOSE_PROMPT.format(query=query))
        hop_type = result.get("type", "simple")
        sub_queries: list[str] = result.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]
        # Cap to configured max
        sub_queries = sub_queries[: settings.max_sub_queries]
        log.info("decompose_node: hop_type=%s, %d sub-queries", hop_type, len(sub_queries))
    except Exception as exc:
        log.warning("decompose_node failed (%s); falling back to simple.", exc)
        hop_type = "simple"
        sub_queries = [query]

    return {"sub_queries": sub_queries, "hop_type": hop_type}


def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve chunks for each sub-query in parallel, then deduplicate.
    Falls back to empty list on total failure.
    """
    sub_queries: list[str] = state["sub_queries"]
    filter_doc_ids = state["filter_doc_ids"]
    retriever = MultiModalRetriever()

    def _retrieve_one(q: str) -> list[RetrievedChunk]:
        try:
            return retriever.retrieve(q, filter_doc_ids=filter_doc_ids)
        except Exception as exc:
            log.warning("retrieve_node: retrieval failed for '%s…' (%s)", q[:40], exc)
            return []

    all_results: list[list[RetrievedChunk]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as pool:
        futures = {pool.submit(_retrieve_one, q): q for q in sub_queries}
        for fut in concurrent.futures.as_completed(futures):
            all_results.append(fut.result())

    merged = _dedup_chunks(all_results)
    log.info("retrieve_node: %d unique chunks after merge", len(merged))
    return {"all_chunks": merged}


def generate_partial_node(state: RAGState) -> dict:
    """
    Generate a partial answer for each sub-query (multi_hop path only).
    NLI guard is intentionally disabled here; it runs on the final synthesis.
    """
    sub_queries: list[str] = state["sub_queries"]
    all_chunks: list[RetrievedChunk] = state["all_chunks"]
    generator = RAGGenerator()
    partial_answers: list[str] = []

    for sub_q in sub_queries:
        try:
            result: GenerationResult = generator.generate(
                query=sub_q,
                chunks=all_chunks,
                apply_guard=False,
            )
            partial_answers.append(result.answer)
            log.debug("Partial answer for '%s…': %d chars", sub_q[:40], len(result.answer))
        except Exception as exc:
            log.warning("generate_partial_node: generation failed for '%s…' (%s)", sub_q[:40], exc)
            partial_answers.append("")

    return {"partial_answers": partial_answers}


def synthesize_node(state: RAGState) -> dict:
    """
    Produce the final answer.

    • simple path   → generate directly from all_chunks with guard enabled.
    • multi_hop path → build a synthesis prompt that includes partial answers,
                       then call generate() with all_chunks as context.
    """
    query: str = state["query"]
    all_chunks: list[RetrievedChunk] = state["all_chunks"]
    hop_type: str = state["hop_type"]
    partial_answers: list[str] = state.get("partial_answers", [])
    sub_queries: list[str] = state.get("sub_queries", [query])
    generator = RAGGenerator()

    if not all_chunks:
        log.warning("synthesize_node: no chunks available; returning empty answer.")
        return {"final_result": GenerationResult(
            answer="I could not find relevant information to answer your question.",
        )}

    try:
        if hop_type == "multi_hop" and partial_answers:
            # Build an enriched synthesis query
            numbered = "\n".join(
                f"{i+1}. Sub-question: {q}\n   Partial answer: {a}"
                for i, (q, a) in enumerate(zip(sub_queries, partial_answers))
            )
            synthesis_query = (
                f"Original question: {query}\n\n"
                f"The following sub-questions were answered:\n{numbered}\n\n"
                f"Using the partial answers and the provided context, give a comprehensive, "
                f"well-structured final answer to the original question."
            )
            result: GenerationResult = generator.generate(
                query=synthesis_query,
                chunks=all_chunks,
                apply_guard=True,
            )
        else:
            result = generator.generate(
                query=query,
                chunks=all_chunks,
                apply_guard=True,
            )

        log.info("synthesize_node: final answer generated (%d chars)", len(result.answer))
        return {"final_result": result}

    except Exception as exc:
        log.error("synthesize_node: generation failed (%s)", exc)
        return {"final_result": GenerationResult(
            answer="An error occurred while generating the answer. Please try again.",
        )}


# ─── Conditional routing ──────────────────────────────────────────────────────

def _route_after_retrieve(state: RAGState) -> str:
    """Route to partial generation for multi_hop, or straight to synthesis."""
    return "generate_partial_node" if state["hop_type"] == "multi_hop" else "synthesize_node"


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class QueryOrchestrator:
    """
    High-level entry point for multi-hop RAG queries.

    Uses a LangGraph StateGraph to coordinate decomposition, retrieval,
    optional partial generation, and final synthesis.

    Example
    -------
    >>> orch = QueryOrchestrator()
    >>> result = orch.run("Compare revenue in Q1 and Q2")
    >>> print(result.answer)
    """

    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self):
        """Compile the LangGraph StateGraph."""
        from langgraph.graph import END, START, StateGraph

        builder: StateGraph = StateGraph(RAGState)

        # Register nodes
        builder.add_node("decompose_node", decompose_node)
        builder.add_node("retrieve_node", retrieve_node)
        builder.add_node("generate_partial_node", generate_partial_node)
        builder.add_node("synthesize_node", synthesize_node)

        # Edges
        builder.add_edge(START, "decompose_node")
        builder.add_edge("decompose_node", "retrieve_node")
        builder.add_conditional_edges(
            "retrieve_node",
            _route_after_retrieve,
            {
                "generate_partial_node": "generate_partial_node",
                "synthesize_node": "synthesize_node",
            },
        )
        builder.add_edge("generate_partial_node", "synthesize_node")
        builder.add_edge("synthesize_node", END)

        return builder.compile()

    def run(
        self,
        query: str,
        filter_doc_ids: list[str] | None = None,
    ) -> GenerationResult:
        """
        Execute the RAG pipeline and return a GenerationResult.

        Parameters
        ----------
        query           : The user's natural-language question.
        filter_doc_ids  : Optional list of doc IDs to restrict retrieval scope.

        Returns
        -------
        GenerationResult with answer, citations, warnings, and token usage.
        """
        initial_state: RAGState = {
            "query": query,
            "filter_doc_ids": filter_doc_ids,
            "sub_queries": [],
            "hop_type": "simple",
            "all_chunks": [],
            "partial_answers": [],
            "final_result": None,
        }

        final_state: RAGState = self.graph.invoke(initial_state)
        result: GenerationResult | None = final_state.get("final_result")

        if result is None:
            log.error("QueryOrchestrator: graph returned no final_result.")
            return GenerationResult(
                answer="An unexpected error occurred. Please try again.",
            )

        return result