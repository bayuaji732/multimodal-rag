"""
ingestion/embedder.py
─────────────────────
Multi-modal embedding layer.

TABLE UPGRADES (v2):
  • summarize_table()  — GPT-4o converts the markdown table to a rich NL summary
                         that describes WHAT the table shows, its headers, numeric
                         ranges, trends, and any notable cells.
  • Tables are embedded on the SUMMARY text (not raw markdown), which dramatically
    improves semantic retrieval: "highest revenue quarter" now matches the table.
  • The original markdown is kept in chunk.text for the generator to render.
  • table_summary stored in DocumentChunk and Qdrant payload for UI display.

Embedding strategy per chunk type:
  TEXT  / TABLE  → OpenAI text-embedding-3-large on (text / summary)
  IMAGE          → caption via GPT-4o Vision, then embed caption
  ALL            → BM25 sparse on the richest available text
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

from config import settings
from utils.models import ChunkType, DocumentChunk

log = logging.getLogger(__name__)

# ─── singletons ───────────────────────────────────────────────────────────────

_clip_model       = None
_clip_tokenizer   = None
_clip_transform   = None
_openai_client: Optional[object] = None


def _get_clip():
    global _clip_model, _clip_tokenizer, _clip_transform
    if _clip_model is None:
        import open_clip
        log.info("Loading CLIP via open_clip: ViT-L-14")
        _clip_model, _, _clip_transform = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _clip_model.eval()
    return _clip_model, _clip_tokenizer, _clip_transform


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ─── text dense embedding ──────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_openai()
    all_vecs: list[list[float]] = []
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(
            model=settings.embedding_model,
            input=texts[i : i + batch_size],
            encoding_format="float",
        )
        all_vecs.extend([d.embedding for d in resp.data])
    return all_vecs


# ─── image dense embedding ─────────────────────────────────────────────────────

def embed_images(b64_images: list[str]) -> list[list[float]]:
    import torch
    model, _, transform = _get_clip()
    imgs = [
        transform(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
        for b64 in b64_images
    ]
    batch = torch.stack(imgs)
    with torch.no_grad(), torch.autocast("cpu"):
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy().tolist()


def embed_text_clip(texts: list[str]) -> list[list[float]]:
    import torch
    model, tokenizer, _ = _get_clip()
    tokens = tokenizer(texts)
    with torch.no_grad(), torch.autocast("cpu"):
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy().tolist()


# ─── image captioning ──────────────────────────────────────────────────────────

def caption_image(b64: str, existing_caption: str = "") -> str:
    client = _get_openai()
    prompt = (
        "You are an expert document analyst. Describe this figure/chart/diagram "
        "in detail — include numbers, labels, trends, and any text visible. "
        "Be concise but complete (max 120 words)."
    )
    if existing_caption:
        prompt += f"\n\nExisting caption hint: {existing_caption}"
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Caption generation failed: %s", exc)
        return existing_caption or "Image"


# ─── TABLE SUMMARIZATION ──────────────────────────────────────────────────────

_TABLE_SUMMARY_PROMPT = """\
You are a data analyst. Below is a markdown table extracted from a document.

Your task:
1. Write a concise natural-language SUMMARY (2-5 sentences) that describes:
   - What the table is about (topic / subject)
   - Column headers and what each measures
   - Key numeric values, ranges, totals, or percentages visible
   - Any obvious trends, maximums, minimums, or notable rows
2. After the summary, list the column headers as a comma-separated line
   prefixed with "Headers:".

Do NOT reproduce the table verbatim. Focus on meaning and searchable facts.

Table:
{markdown}
"""


def summarize_table(markdown: str, title: str | None = None) -> str:
    """
    Use GPT-4o-mini to generate a rich NL summary of a markdown table.
    The summary is used as the embedding text so semantic search works
    ("what quarter had highest revenue" → finds the revenue table).
    """
    client = _get_openai()
    header_hint = f'Table title/caption: "{title}"\n\n' if title else ""
    prompt = _TABLE_SUMMARY_PROMPT.format(markdown=header_hint + markdown)

    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,         # gpt-4o-mini is fast + cheap
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Table summarization failed: %s", exc)
        # Fallback: first 500 chars of the raw markdown
        return markdown[:500]


# ─── sparse BM25 encoding ─────────────────────────────────────────────────────

class SparseEncoder:
    def __init__(self):
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        return self._tokenizer

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        from collections import Counter
        enc = self._get_tokenizer().encode(text)
        tf  = Counter(enc.ids)
        max_tf = max(tf.values())
        return list(tf.keys()), [v / max_tf for v in tf.values()]


_sparse_encoder: Optional[SparseEncoder] = None


def get_sparse_encoder() -> SparseEncoder:
    global _sparse_encoder
    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()
    return _sparse_encoder


# ─── main: embed a batch of chunks ────────────────────────────────────────────

def embed_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Sets dense_vector + sparse_indices/values on every chunk.
    For tables, first generates an LLM summary, embeds the summary,
    but preserves the original markdown in chunk.text for generation.
    """
    sparse_enc = get_sparse_encoder()

    text_idx  = [i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.TEXT]
    table_idx = [i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.TABLE]
    image_idx = [i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.IMAGE]

    # ── TEXT: embed as-is ─────────────────────────────────────────────────────
    if text_idx:
        texts = [chunks[i].text for i in text_idx]
        vecs  = embed_texts(texts)
        for idx, vec in zip(text_idx, vecs):
            chunks[idx].dense_vector = vec

    # ── TABLE: summarize → embed summary, keep markdown in .text ──────────────
    if table_idx:
        log.info("Summarizing %d table chunk(s) with LLM…", len(table_idx))
        summaries: list[str] = []
        for i in table_idx:
            chunk   = chunks[i]
            summary = summarize_table(chunk.text, title=chunk.table_title)
            chunk.table_summary = summary          # stored in payload
            summaries.append(summary)
            log.debug("Table summary [page %d]: %s", chunk.page, summary[:120])

        vecs = embed_texts(summaries)
        for idx, vec in zip(table_idx, vecs):
            norm = np.linalg.norm(vec)
            chunks[idx].dense_vector = (np.array(vec) / norm).tolist() if norm > 0 else vec

    # ── IMAGE: caption → embed caption ────────────────────────────────────────
    for i in image_idx:
        chunk   = chunks[i]
        caption = caption_image(chunk.image_b64, existing_caption=chunk.text)
        chunk.text = caption

        vec  = embed_texts([caption])[0]
        norm = np.linalg.norm(vec)
        chunk.dense_vector = (np.array(vec) / norm).tolist() if norm > 0 else vec

    # ── SPARSE: BM25 on richest text for each chunk ────────────────────────────
    # For tables: sparse on summary + markdown combined for better keyword recall
    for chunk in chunks:
        if chunk.chunk_type == ChunkType.TABLE and chunk.table_summary:
            sparse_text = chunk.table_summary + " " + chunk.text
        else:
            sparse_text = chunk.text
        idxs, vals = sparse_enc.encode(sparse_text)
        chunk.sparse_indices = idxs
        chunk.sparse_values  = vals

    return chunks