from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.graph_store import InMemoryGraphStore, load_documents
from app.llm import AnswerGenerator
from app.models import AskRequest, AskResponse, Source
from app.retriever import HybridRetriever

app = FastAPI(title="Graph RAG Knowledge Assistant", version="1.0.0")

graph = InMemoryGraphStore()
documents = load_documents(settings.data_dir)
graph.index_documents(documents)
retriever = HybridRetriever(graph)
generator = AnswerGenerator(settings)


@app.get("/health")
def health() -> dict[str, int | str]:
    return {
        "status": "ok",
        "documents": len(graph.documents),
        "chunks": len(graph.chunks),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    contexts = retriever.retrieve(request.question, top_k=request.top_k)
    answer = generator.answer(request.question, contexts)
    sources = [
        Source(
            chunk_id=context.chunk.id,
            document_id=context.chunk.document_id,
            score=context.score,
            entities=list(context.graph_hops[:8]),
            text=context.chunk.text,
        )
        for context in contexts
    ]
    return AskResponse(answer=answer, sources=sources)
