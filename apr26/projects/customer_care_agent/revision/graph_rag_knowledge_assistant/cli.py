from __future__ import annotations

import argparse

from app.config import settings
from app.graph_store import InMemoryGraphStore, load_documents
from app.llm import AnswerGenerator
from app.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over a local Graph RAG index.")
    parser.add_argument("question", help="Question to ask the indexed documents")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    graph = InMemoryGraphStore()
    graph.index_documents(load_documents(settings.data_dir))
    contexts = HybridRetriever(graph).retrieve(args.question, top_k=args.top_k)
    answer = AnswerGenerator(settings).answer(args.question, contexts)

    print("\nAnswer\n------")
    print(answer)
    print("\nSources\n-------")
    for context in contexts:
        print(f"{context.chunk.id} | score={context.score} | entities={', '.join(context.graph_hops[:5])}")


if __name__ == "__main__":
    main()
