from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


NodeType = Literal["document", "chunk", "entity"]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    index: int
    entities: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    type: NodeType


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class RetrievedContext:
    chunk: Chunk
    score: float
    matched_entities: tuple[str, ...]
    graph_hops: tuple[str, ...]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    entities: list[str]
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
