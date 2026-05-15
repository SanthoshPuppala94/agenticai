from __future__ import annotations

from app.graph_store import InMemoryGraphStore
from app.models import Document
from app.retriever import HybridRetriever
from app.text_processing import chunk_text, extract_entities


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, max_words=10, overlap=2)

    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_extract_entities_finds_multi_word_entities() -> None:
    entities = extract_entities("Neo4j Vector Search works with Ollama and FastAPI.")

    assert "Neo4j Vector Search" in entities
    assert "Ollama" in entities
    assert "FastAPI" in entities


def test_graph_index_creates_document_chunk_entity_edges() -> None:
    graph = InMemoryGraphStore()
    graph.index_documents(
        [
            Document(
                id="demo",
                title="Demo",
                text="Graph RAG uses Neo4j for entity traversal and Ollama for answers.",
                source="memory",
            )
        ]
    )

    assert "demo" in graph.documents
    assert graph.chunks
    assert any(edge.relation == "HAS_CHUNK" for edge in graph.edges)
    assert any(edge.relation == "MENTIONS" for edge in graph.edges)


def test_hybrid_retriever_returns_relevant_context() -> None:
    graph = InMemoryGraphStore()
    graph.index_documents(
        [
            Document(
                id="demo",
                title="Demo",
                text=(
                    "Hybrid Retrieval combines lexical search with graph traversal. "
                    "Neo4j stores entity relationships for Graph RAG."
                ),
                source="memory",
            )
        ]
    )

    results = HybridRetriever(graph).retrieve("What does Hybrid Retrieval combine?", top_k=1)

    assert results
    assert "lexical search" in results[0].chunk.text
