from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class DocumentChunk:
    """Atomic unit stored in the vector store."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    doc_name: str = ""
    chunk_type: ChunkType = ChunkType.TEXT

    # content
    text: str = ""                      # always populated; for images = alt/caption
    image_b64: Optional[str] = None     # base64 PNG for image chunks
    image_path: Optional[str] = None    # local path during ingestion

    # location metadata
    page: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None   # x0,y0,x1,y1

    # embeddings (set after encoding)
    dense_vector: Optional[list[float]] = None
    sparse_indices: Optional[list[int]] = None
    sparse_values: Optional[list[float]] = None

    def to_payload(self) -> dict:
        """Qdrant payload — everything except vectors."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "chunk_type": self.chunk_type.value,
            "text": self.text,
            "image_b64": self.image_b64,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rank: int = 0
