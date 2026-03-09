"""
ingestion/embedder.py
─────────────────────
Multi-modal embedding layer:

  TEXT / TABLE → OpenAI text-embedding-3-large  (dense)
                 BM25 tokeniser                  (sparse)
  IMAGE        → CLIP ViT-L/14                   (dense, projected to text space)
               + caption via GPT-4o Vision       → text dense embed too
               + BM25 on caption                 (sparse)

ColPali-style late interaction:
  For image chunks we produce *both* a CLIP vector and a caption-text vector,
  and store both under separate named vectors in Qdrant so the retriever can
  do patch-level matching when needed.
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

# ─── singletons (loaded once per worker) ──────────────────────────────────────
# Heavy ML imports are deferred inside _get_* functions so that Celery can
# import this module without triggering the full torch/torchvision/onnx chain
# at startup (which causes the ml_dtypes.float4_e2m1fn crash on Windows).

_clip_model = None
_clip_tokenizer = None
_clip_transform = None
_openai_client: Optional[object] = None


def _get_clip():
    """Lazy-load open_clip (avoids torchvision→onnx→ml_dtypes import at module load)."""
    global _clip_model, _clip_tokenizer, _clip_transform
    if _clip_model is None:
        import open_clip  # open_clip_torch — no torchvision dependency
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
    """Batch embed texts with OpenAI text-embedding-3-large."""
    client = _get_openai()
    all_vecs: list[list[float]] = []
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            encoding_format="float",
        )
        all_vecs.extend([d.embedding for d in resp.data])
    return all_vecs


# ─── image dense embedding (open_clip) ────────────────────────────────────────

def embed_images(b64_images: list[str]) -> list[list[float]]:
    """Encode base64 PNGs with CLIP (via open_clip). Returns 768-d vectors."""
    import io, base64, torch
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
    """Encode query text with CLIP text encoder for image retrieval."""
    import torch
    model, tokenizer, _ = _get_clip()
    tokens = tokenizer(texts)
    with torch.no_grad(), torch.autocast("cpu"):
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy().tolist()


# ─── image captioning (GPT-4o Vision) ─────────────────────────────────────────

def caption_image(b64: str, existing_caption: str = "") -> str:
    """Generate a rich text caption for an image chunk via GPT-4o Vision."""
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


# ─── sparse BM25 encoding ─────────────────────────────────────────────────────

class SparseEncoder:
    """BM25-style sparse encoding using BERT vocab token IDs."""

    def __init__(self):
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer  # lazy import
            self._tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        return self._tokenizer

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        enc = self._get_tokenizer().encode(text)
        ids = enc.ids
        from collections import Counter
        tf = Counter(ids)
        indices = list(tf.keys())
        max_tf = max(tf.values())
        values = [v / max_tf for v in tf.values()]
        return indices, values


_sparse_encoder: Optional[SparseEncoder] = None


def get_sparse_encoder() -> SparseEncoder:
    global _sparse_encoder
    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()
    return _sparse_encoder


# ─── main: embed a batch of chunks ────────────────────────────────────────────

def embed_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    In-place: sets dense_vector, sparse_indices, sparse_values on each chunk.
    Also enriches image chunk .text with GPT-4o captions.
    """
    sparse_enc = get_sparse_encoder()

    # Split by type for batching
    text_idx = [i for i, c in enumerate(chunks) if c.chunk_type != ChunkType.IMAGE]
    image_idx = [i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.IMAGE]

    # ── text/table dense embeddings ───────────────────────────────────────────
    if text_idx:
        texts = [chunks[i].text for i in text_idx]
        vecs = embed_texts(texts)
        for idx, vec in zip(text_idx, vecs):
            chunks[idx].dense_vector = vec

    # ── image chunks ──────────────────────────────────────────────────────────
    for i in image_idx:
        chunk = chunks[i]
        assert chunk.image_b64, "Image chunk missing base64 data"

        # 1. Generate rich caption
        caption = caption_image(chunk.image_b64, existing_caption=chunk.text)
        chunk.text = caption  # replace sparse caption with rich one

        # 2. Caption text vector (OpenAI, same space as text chunks)
        # CLIP is used query-side in the retriever for cross-modal matching.
        # Storing CLIP (768-d) + OpenAI (3072-d) together causes shape mismatch
        # in np.mean — so we use caption embedding as the canonical dense vector.
        caption_vec = embed_texts([caption])[0]
        norm = np.linalg.norm(caption_vec)
        chunk.dense_vector = (np.array(caption_vec) / norm).tolist() if norm > 0 else caption_vec

    # ── sparse for all chunks ─────────────────────────────────────────────────
    for chunk in chunks:
        idxs, vals = sparse_enc.encode(chunk.text)
        chunk.sparse_indices = idxs
        chunk.sparse_values = vals

    return chunks