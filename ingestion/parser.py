"""
ingestion/parser.py
───────────────────
Extracts text, images, and tables from uploaded files.

Supported inputs
  • PDF  → pdfplumber (text + tables) + unstructured (fallback / image extraction)
  • PNG/JPG/WEBP → direct image chunk
  • DOCX → unstructured

Output: list[DocumentChunk]
"""
from __future__ import annotations

import base64
import io
import logging
import re
import uuid
from pathlib import Path
from typing import Iterator

import pdfplumber
from PIL import Image
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Table, Image as UnstructuredImage, Text, Title, NarrativeText,
)

from utils.models import ChunkType, DocumentChunk

log = logging.getLogger(__name__)

# ─── helpers ──────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, max_px: int = 1024) -> str:
    """Resize (keep aspect) and encode to base64 PNG."""
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


# ─── per-format parsers ────────────────────────────────────────────────────────

class PDFParser:
    """
    Two-pass PDF parser:
      Pass 1 – pdfplumber  → text chunks + table chunks (markdown tables)
      Pass 2 – unstructured → image elements (figures / charts)
    """

    def parse(self, path: Path, doc_id: str, doc_name: str | None = None) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        doc_name = doc_name or path.name

        # ── Pass 1: pdfplumber for text & tables ─────────────────────────────
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Tables first (they get their own chunk type)
                for table in page.extract_tables():
                    if not table:
                        continue
                    md = _table_to_markdown(table)
                    if md:
                        chunks.append(DocumentChunk(
                            doc_id=doc_id,
                            doc_name=doc_name,
                            chunk_type=ChunkType.TABLE,
                            text=md,
                            page=page_num,
                        ))

                # Remaining text (remove table bounding boxes)
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text = _clean_text(text)
                if len(text) > 40:
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_type=ChunkType.TEXT,
                        text=text,
                        page=page_num,
                    ))

        # ── Pass 2: unstructured for images / figures ─────────────────────────
        try:
            elements = partition(filename=str(path), strategy="hi_res",
                                 extract_images_in_pdf=True)
            for el in elements:
                if isinstance(el, UnstructuredImage):
                    page_num = getattr(el.metadata, "page_number", 0) or 0
                    img_path = getattr(el.metadata, "image_path", None)
                    b64 = ""
                    if img_path and Path(img_path).exists():
                        img = Image.open(img_path).convert("RGB")
                        b64 = _pil_to_b64(img)
                    caption = _clean_text(str(el))
                    if b64:
                        chunks.append(DocumentChunk(
                            doc_id=doc_id,
                            doc_name=doc_name,
                            chunk_type=ChunkType.IMAGE,
                            text=caption or f"Figure on page {page_num}",
                            image_b64=b64,
                            page=page_num,
                        ))
        except Exception as exc:
            log.warning("unstructured image pass failed: %s", exc)

        log.info("PDFParser: %d chunks from %s", len(chunks), path.name)
        return chunks


class ImageParser:
    def parse(self, path: Path, doc_id: str, doc_name: str | None = None) -> list[DocumentChunk]:
        img = Image.open(path).convert("RGB")
        b64 = _pil_to_b64(img)
        return [DocumentChunk(
            doc_id=doc_id,
            doc_name=doc_name or path.name,
            chunk_type=ChunkType.IMAGE,
            text=f"Image: {path.stem}",
            image_b64=b64,
            page=1,
        )]


class DocxParser:
    def parse(self, path: Path, doc_id: str, doc_name: str | None = None) -> list[DocumentChunk]:
        elements = partition(filename=str(path))
        chunks: list[DocumentChunk] = []
        _doc_name = doc_name or path.name
        for el in elements:
            if isinstance(el, (Text, Title, NarrativeText)):
                t = _clean_text(str(el))
                if len(t) > 20:
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        doc_name=_doc_name,
                        chunk_type=ChunkType.TEXT,
                        text=t,
                    ))
            elif isinstance(el, Table):
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    doc_name=_doc_name,
                    chunk_type=ChunkType.TABLE,
                    text=_clean_text(el.text),
                ))
        return chunks


# ─── dispatcher ───────────────────────────────────────────────────────────────

SUFFIX_MAP = {
    ".pdf": PDFParser,
    ".png": ImageParser,
    ".jpg": ImageParser,
    ".jpeg": ImageParser,
    ".webp": ImageParser,
    ".docx": DocxParser,
}


def parse_document(path: Path, doc_id: str | None = None, doc_name: str | None = None) -> list[DocumentChunk]:
    """
    Top-level entry point.
    Returns list of DocumentChunk objects ready for embedding.
    """
    doc_id = doc_id or str(uuid.uuid4())
    suffix = path.suffix.lower()
    parser_cls = SUFFIX_MAP.get(suffix)
    if parser_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return parser_cls().parse(path, doc_id, doc_name=doc_name)


# ─── text chunker ─────────────────────────────────────────────────────────────

def chunk_text(
    chunks: list[DocumentChunk],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """
    Split TEXT chunks that exceed chunk_size (word-level).
    IMAGE and TABLE chunks are passed through unchanged.
    """
    result: list[DocumentChunk] = []
    for chunk in chunks:
        if chunk.chunk_type != ChunkType.TEXT or len(chunk.text.split()) <= chunk_size:
            result.append(chunk)
            continue
        words = chunk.text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            sub_text = " ".join(words[start:end])
            result.append(DocumentChunk(
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_name,
                chunk_type=chunk.chunk_type,
                text=sub_text,
                page=chunk.page,
            ))
            start += chunk_size - overlap
    return result


# ─── utils ────────────────────────────────────────────────────────────────────

def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert pdfplumber table (list of rows) to a markdown table string."""
    if not table:
        return ""
    rows = [[str(c or "") for c in row] for row in table]
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(filter(None, [header, sep, body]))