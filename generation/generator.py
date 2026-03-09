"""
generation/generator.py
────────────────────────
RAG answer generator with:
  • Multi-modal context assembly (text + image)
  • GPT-4o / Claude 3.5 generation
  • Citation extraction from model output
  • NLI hallucination guard — verifies each sentence against retrieved chunks
  • Streaming support
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
_nli_model = None


def _get_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        log.info("Loading NLI model: %s", settings.nli_model)
        _nli_tokenizer = AutoTokenizer.from_pretrained(settings.nli_model)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


def check_entailment(premise: str, hypothesis: str) -> float:
    """
    Returns entailment probability in [0, 1].
    Uses DeBERTa-v3 NLI model.
    """
    tok, model = _get_nli()
    enc = tok(premise, hypothesis, return_tensors="pt",
               truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # Label order: contradiction=0, neutral=1, entailment=2
    return probs[2].item()


def guard_answer(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """
    Split answer into sentences, verify each against top retrieved chunks.
    Returns (verified_answer, list_of_warnings).
    
    Sentences with entailment_score < threshold are flagged with [⚠ unverified].
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    context = " ".join(c.chunk.text for c in chunks[:5])  # top 5 chunks as premise
    warnings: list[str] = []
    verified_parts: list[str] = []

    for sent in sentences:
        if len(sent.split()) < 5:  # skip very short sentences
            verified_parts.append(sent)
            continue
        score = check_entailment(context, sent)
        if score < settings.nli_threshold:
            verified_parts.append(f"{sent} [⚠ unverified]")
            warnings.append(f"Low entailment ({score:.2f}): {sent[:80]}...")
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
    answer: str
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _build_context_messages(
    query: str,
    chunks: list[RetrievedChunk],
) -> tuple[list[dict], list[Citation]]:
    """
    Build OpenAI-style messages array with interleaved text and images.
    Returns (messages, citations).
    """
    citations: list[Citation] = []
    context_parts: list[dict] = []

    context_parts.append({
        "type": "text",
        "text": (
            "You are a precise, evidence-based assistant. Answer the user's question "
            "using ONLY the provided context. For every factual claim, cite the source "
            "using [1], [2], ... notation corresponding to the context blocks below.\n\n"
            "─── CONTEXT ───\n"
        ),
    })

    for i, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        cit = Citation(
            index=i,
            doc_name=chunk.doc_name,
            page=chunk.page,
            chunk_type=chunk.chunk_type.value,
            text_snippet=chunk.text[:200],
            image_b64=chunk.image_b64,
        )
        citations.append(cit)

        context_parts.append({
            "type": "text",
            "text": f"\n[{i}] Source: {chunk.doc_name}, page {chunk.page} ({chunk.chunk_type.value})\n{chunk.text}\n",
        })

        # Include image if present
        if chunk.chunk_type == ChunkType.IMAGE and chunk.image_b64:
            context_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chunk.image_b64}",
                    "detail": "low",
                },
            })

    context_parts.append({
        "type": "text",
        "text": f"\n─── QUESTION ───\n{query}\n\nAnswer (cite sources with [n]):",
    })

    messages = [{"role": "user", "content": context_parts}]
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
        """Synchronous RAG generation."""
        if not chunks:
            return GenerationResult(
                answer="I couldn't find relevant information in the knowledge base to answer your question.",
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
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        return text, usage

    def _generate_anthropic(self, messages: list[dict]) -> tuple[str, dict]:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        # Convert OpenAI-style image_url to Anthropic image blocks
        content = _convert_to_anthropic_content(messages[0]["content"])
        resp = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        text = resp.content[0].text
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
        return text, usage

    async def stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AsyncIterator[str]:
        """Async streaming generation — yields text tokens."""
        if not chunks:
            yield "I couldn't find relevant information to answer your question."
            return

        messages, _ = _build_context_messages(query, chunks)
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        async with client.chat.completions.stream(
            model=settings.llm_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ─── helpers ──────────────────────────────────────────────────────────────────

def _convert_to_anthropic_content(oai_content: list[dict]) -> list[dict]:
    """Convert OpenAI content blocks to Anthropic format."""
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
