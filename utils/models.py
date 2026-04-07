from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class ChunkType(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class DocumentChunk:
    """Atomic unit stored in the vector store."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id:   str = ""
    doc_name: str = ""
    chunk_type: ChunkType = ChunkType.TEXT

    # content
    text:       str = ""                    # markdown for tables / caption for images / prose for text
    image_b64:  Optional[str] = None
    image_path: Optional[str] = None

    # ── TABLE-specific ─────────────────────────────────────────────────────────
    table_summary: Optional[str] = None    # LLM-generated NL summary used for retrieval embed
    table_title:   Optional[str] = None    # caption / heading found near the table
    table_rows:    int = 0                 # data row count (excl. header)
    table_cols:    int = 0                 # column count
    table_headers: list[str] = field(default_factory=list)

    # location metadata
    page: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None

    # embeddings (set after encoding)
    dense_vector:   Optional[list[float]] = None
    sparse_indices: Optional[list[int]]   = None
    sparse_values:  Optional[list[float]] = None

    def to_payload(self) -> dict:
        """Qdrant payload — everything except vectors."""
        return {
            "id":            self.id,
            "doc_id":        self.doc_id,
            "doc_name":      self.doc_name,
            "chunk_type":    self.chunk_type.value,
            "text":          self.text,
            "image_b64":     self.image_b64,
            "page":          self.page,
            "bbox":          list(self.bbox) if self.bbox else None,
            # table extras
            "table_summary": self.table_summary,
            "table_title":   self.table_title,
            "table_rows":    self.table_rows,
            "table_cols":    self.table_cols,
            "table_headers": self.table_headers,
        }


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rank:  int = 0