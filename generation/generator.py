"""
generation/generator.py
────────────────────────
RAG answer generator.

TABLE UPGRADES (v2):
  • Table chunks are rendered as BOTH the markdown table AND its NL summary in
    the context block — the model sees structured data AND searchable semantics.
  • System prompt explicitly instructs the model on table reasoning:
    - Read every row, not just headers
    - Compute aggregates if asked (sum, average, max, min)
    - Quote exact cell values when making numeric claims
  • _build_context_messages() separates TABLE blocks with a visual rule so the
    model doesn't conflate table rows with prose sentences.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import anthropic
import torch
from openai import AsyncOpenAI, OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings
from utils.models import ChunkType, DocumentChunk, RetrievedChunk

log = logging.getLogger(__name__)

# ─── NLI hallucination guard ──────────────────────────────────────────────────

_nli_tokenizer = None
_nli_model     = None


def _get_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        log.info("Loading NLI model: %s", settings.nli_model)
        _nli_tokenizer = AutoTokenizer.from_pretrained(settings.nli_model)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


def check_entailment(premise: str, hypothesis: str) -> float:
    tok, model = _get_nli()
    enc = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return probs[2].item()


def guard_answer(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    # For tables, use the summary as premise (markdown confuses NLI)
    context_parts = []
    for c in chunks[:5]:
        if c.chunk.chunk_type == ChunkType.TABLE and c.chunk.table_summary:
            context_parts.append(c.chunk.table_summary)
        else:
            context_parts.append(c.chunk.text)
    context = " ".join(context_parts)

    warnings: list[str] = []
    verified_parts: list[str] = []
    for sent in sentences:
        if len(sent.split()) < 5:
            verified_parts.append(sent)
            continue
        score = check_entailment(context, sent)
        if score < settings.nli_threshold:
            verified_parts.append(f"{sent} [⚠ unverified]")
            warnings.append(f"Low entailment ({score:.2f}): {sent[:80]}…")
        else:
            verified_parts.append(sent)
    return " ".join(verified_parts), warnings


# ─── Context builder ──────────────────────────────────────────────────────────

@dataclass
class Citation:
    index: int
    doc_name: str
    page: int
    chunk_type: str
    text_snippet: str
    image_b64: Optional[str] = None


@dataclass
class GenerationResult:
    answer:            str
    citations:         list[Citation] = field(default_factory=list)
    warnings:          list[str]      = field(default_factory=list)
    model:             str  = ""
    prompt_tokens:     int  = 0
    completion_tokens: int  = 0


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise, evidence-based assistant. Answer questions using ONLY the
provided context blocks. For every factual claim, cite the source with [n].

TABLE REASONING RULES — follow these strictly when context contains tables:
1. Read EVERY row in full; do not skip rows.
2. When asked for a maximum, minimum, sum, average, or count — compute it from
   the table data; do not guess.
3. Quote exact cell values (numbers, names, dates) when making numeric claims.
4. If a table has a title or caption, use it to understand the table's subject.
5. If multiple tables are relevant, compare them explicitly.
6. If the table does not contain enough information, say so clearly.

Formatting:
- Use bullet points for lists.
- Reproduce small relevant table excerpts (≤5 rows) in Markdown when helpful.
- Cite sources as [1], [2], … at the end of each sentence that uses that source.
"""


def _format_table_block(index: int, chunk: DocumentChunk) -> list[dict]:
    """
    Render a table context block with:
      • title (if any)
      • NL summary   → helps model understand semantics
      • raw markdown → gives model exact cell values to quote
    """
    parts: list[dict] = []

    header = f"\n[{index}] TABLE — Source: {chunk.doc_name}, page {chunk.page}"
    if chunk.table_title:
        header += f"\nTitle: {chunk.table_title}"
    if chunk.table_headers:
        header += f"\nColumns: {', '.join(chunk.table_headers)}"
    header += f"\nDimensions: {chunk.table_rows} data rows × {chunk.table_cols} columns"

    if chunk.table_summary:
        header += f"\n\nSummary: {chunk.table_summary}"

    header += "\n\nFull table (use exact values from here when answering):\n"
    header += chunk.text   # markdown table
    header += "\n" + "─" * 60 + "\n"

    parts.append({"type": "text", "text": header})
    return parts


def _build_context_messages(
    query: str,
    chunks: list[RetrievedChunk],
) -> tuple[list[dict], list[Citation]]:
    citations: list[Citation] = []
    context_parts: list[dict] = []

    context_parts.append({
        "type": "text",
        "text": "─── CONTEXT ───\n",
    })

    for i, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        cit = Citation(
            index=i,
            doc_name=chunk.doc_name,
            page=chunk.page,
            chunk_type=chunk.chunk_type.value,
            text_snippet=(chunk.table_summary or chunk.text)[:200],
            image_b64=chunk.image_b64,
        )
        citations.append(cit)

        if chunk.chunk_type == ChunkType.TABLE:
            context_parts.extend(_format_table_block(i, chunk))

        elif chunk.chunk_type == ChunkType.IMAGE:
            context_parts.append({
                "type": "text",
                "text": (
                    f"\n[{i}] IMAGE — Source: {chunk.doc_name}, page {chunk.page}\n"
                    f"Caption: {chunk.text}\n"
                ),
            })
            if chunk.image_b64:
                context_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{chunk.image_b64}",
                        "detail": "low",
                    },
                })

        else:  # TEXT
            context_parts.append({
                "type": "text",
                "text": (
                    f"\n[{i}] TEXT — Source: {chunk.doc_name}, page {chunk.page}\n"
                    f"{chunk.text}\n"
                ),
            })

    context_parts.append({
        "type": "text",
        "text": f"\n─── QUESTION ───\n{query}\n\nAnswer (cite sources with [n]):",
    })

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": context_parts},
    ]
    return messages, citations


# ─── Generator ────────────────────────────────────────────────────────────────

class RAGGenerator:

    def __init__(self, provider: str = settings.llm_provider):
        self.provider = provider

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        apply_guard: bool = True,
    ) -> GenerationResult:
        if not chunks:
            return GenerationResult(
                answer="I couldn't find relevant information in the knowledge base."
            )

        messages, citations = _build_context_messages(query, chunks)

        if self.provider == "openai":
            answer, usage = self._generate_openai(messages)
        else:
            answer, usage = self._generate_anthropic(messages)

        warnings: list[str] = []
        if apply_guard:
            answer, warnings = guard_answer(answer, chunks)
            if warnings:
                log.warning("NLI guard flagged %d sentences.", len(warnings))

        return GenerationResult(
            answer=answer,
            citations=citations,
            warnings=warnings,
            model=settings.llm_model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def _generate_openai(self, messages: list[dict]) -> tuple[str, dict]:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,   # 0 for factual table reading
        )
        text  = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        return text, usage

    def _generate_anthropic(self, messages: list[dict]) -> tuple[str, dict]:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user_content  = _convert_to_anthropic_content(messages[1]["content"])
        resp = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        text  = resp.content[0].text
        usage = {
            "prompt_tokens":     resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
        return text, usage

    async def stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AsyncIterator[str]:
        if not chunks:
            yield "I couldn't find relevant information to answer your question."
            return

        messages, _ = _build_context_messages(query, chunks)
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        async with client.chat.completions.stream(
            model=settings.llm_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ─── helpers ──────────────────────────────────────────────────────────────────

def _convert_to_anthropic_content(oai_content: list[dict]) -> list[dict]:
    out: list[dict] = []
    for block in oai_content:
        if block["type"] == "text":
            out.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image_url":
            url: str = block["image_url"]["url"]
            if url.startswith("data:image/png;base64,"):
                b64 = url.split(",", 1)[1]
                out.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
    return out