from __future__ import annotations

import math
from collections import Counter

from app.graph_store import InMemoryGraphStore
from app.models import RetrievedContext
from app.text_processing import extract_entities, tokenize


class HybridRetriever:
    def __init__(self, graph: InMemoryGraphStore) -> None:
        self.graph = graph

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedContext]:
        query_tokens = tokenize(question)
        query_entities = list(extract_entities(question))
        query_counter = Counter(query_tokens)

        entity_seed_chunks = self.graph.chunks_for_entities(query_entities)
        scored: list[RetrievedContext] = []

        for chunk in self.graph.chunks.values():
            lexical_score = self._lexical_score(query_counter, chunk.text)
            matched_entities = tuple(
                entity for entity in chunk.entities if entity.lower() in question.lower()
            )
            graph_boost = 0.25 if chunk.id in entity_seed_chunks else 0.0
            graph_hops = self._graph_hops(chunk.id)
            neighborhood_boost = min(len(graph_hops), 6) * 0.03
            score = lexical_score + graph_boost + neighborhood_boost
            if score > 0:
                scored.append(
                    RetrievedContext(
                        chunk=chunk,
                        score=round(score, 4),
                        matched_entities=matched_entities,
                        graph_hops=tuple(graph_hops),
                    )
                )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _lexical_score(query_counter: Counter[str], text: str) -> float:
        if not query_counter:
            return 0.0
        chunk_counter = Counter(tokenize(text))
        overlap = sum(min(count, chunk_counter[token]) for token, count in query_counter.items())
        norm = math.sqrt(sum(query_counter.values())) * math.sqrt(sum(chunk_counter.values()))
        return overlap / norm if norm else 0.0

    def _graph_hops(self, chunk_id: str) -> list[str]:
        labels: list[str] = []
        for node_id in sorted(self.graph.neighbors(chunk_id, depth=2)):
            node = self.graph.nodes.get(node_id)
            if node and node.type == "entity":
                labels.append(node.label)
        return labels
