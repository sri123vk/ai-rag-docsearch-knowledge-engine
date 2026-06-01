from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    title: str
    text: str
    source_uri: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_uri: str
    page: int | None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    lexical_score: float
    vector_score: float
    final_score: float


@dataclass(frozen=True)
class EntityTriple:
    subject: str
    relation: str
    object: str
    chunk_id: str
    source_uri: str

