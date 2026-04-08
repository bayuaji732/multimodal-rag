"""
retrieval/query_rewriter.py
────────────────────────────
Query rewriting (multi-query) and HyDE (Hypothetical Document Embeddings).
All public functions are failure-safe: they log warnings and return sensible
defaults rather than propagating exceptions to callers.
"""
from __future__ import annotations

import json
import logging

from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a search query expert. Generate {n} alternative phrasings of the query below.
Preserve intent. Use different vocabulary. Each rewrite ≤ 20 words.
Return ONLY a valid JSON array of strings.

Query: {query}
"""

_HYDE_PROMPT = """\
Write a short passage (≤150 words) that would perfectly answer the question below.
Write as if extracted from an authoritative document. Be specific with numbers and names.

Question: {query}

Passage:
"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def rewrite_query(query: str, n: int | None = None) -> list[str]:
    """
    Generate n alternative phrasings of *query* using the configured LLM.

    Returns [original_query, rewrite_1, ..., rewrite_n].
    On any failure returns [query] — never raises.
    """
    n = n or settings.n_rewrite_variants
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.4,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": _REWRITE_PROMPT.format(n=n, query=query),
            }],
        )
        raw = resp.choices[0].message.content or "[]"
        # Strip possible markdown fences
        raw = raw.strip().strip("```json").strip("```").strip()
        rewrites: list[str] = json.loads(raw)
        if not isinstance(rewrites, list):
            raise ValueError("LLM did not return a JSON array")
        rewrites = [str(r) for r in rewrites if str(r).strip()]
        # Deduplicate while preserving order; original query always first
        seen: set[str] = {query}
        unique: list[str] = [query]
        for r in rewrites:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique[:n + 1]
    except Exception as exc:
        log.warning("rewrite_query failed (%s); falling back to original query.", exc)
        return [query]


def generate_hyde_doc(query: str) -> str:
    """
    Generate a hypothetical answer passage for HyDE retrieval.

    The generated passage is then embedded and used as the dense query vector,
    which often yields better semantic alignment with the stored document chunks.
    On any failure returns *query* — never raises.
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.7,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": _HYDE_PROMPT.format(query=query),
            }],
        )
        passage = (resp.choices[0].message.content or "").strip()
        if not passage:
            raise ValueError("Empty HyDE passage returned")
        return passage
    except Exception as exc:
        log.warning("generate_hyde_doc failed (%s); falling back to original query.", exc)
        return query