from __future__ import annotations

import re
from collections.abc import Iterable

from app.models import Chunk, Document

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")
ENTITY_TERM = r"(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,})"
ENTITY_PATTERN = re.compile(rf"\b{ENTITY_TERM}(?:\s+{ENTITY_TERM})*\b")
STOP_ENTITIES = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "When",
    "Where",
    "Graph",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def extract_entities(text: str) -> tuple[str, ...]:
    candidates = {match.group(0).strip() for match in ENTITY_PATTERN.finditer(text)}
    cleaned = {
        candidate
        for candidate in candidates
        if candidate not in STOP_ENTITIES and len(candidate) > 2
    }
    return tuple(sorted(cleaned))


def chunk_text(text: str, max_words: int = 90, overlap: int = 18) -> list[str]:
    words = text.split()
    if not words:
        return []
    if max_words <= overlap:
        raise ValueError("max_words must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(documents: Iterable[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(chunk_text(document.text)):
            chunk_id = f"{document.id}:chunk:{index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=text,
                    index=index,
                    entities=extract_entities(text),
                )
            )
    return chunks
