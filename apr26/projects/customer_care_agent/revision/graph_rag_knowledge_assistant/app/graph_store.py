from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

from app.models import Chunk, Document, GraphEdge, GraphNode
from app.text_processing import build_chunks


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self.adjacency: dict[str, set[str]] = defaultdict(set)

    def index_documents(self, documents: list[Document]) -> None:
        for document in documents:
            self.documents[document.id] = document
            self._add_node(document.id, document.title, "document")

        for chunk in build_chunks(documents):
            self.chunks[chunk.id] = chunk
            self._add_node(chunk.id, f"{chunk.document_id} chunk {chunk.index}", "chunk")
            self._add_edge(chunk.document_id, chunk.id, "HAS_CHUNK")

            for entity in chunk.entities:
                entity_id = self._entity_id(entity)
                self._add_node(entity_id, entity, "entity")
                self._add_edge(chunk.id, entity_id, "MENTIONS")
                self.entity_to_chunks[entity].add(chunk.id)

        self._connect_related_entities()

    def neighbors(self, node_id: str, depth: int = 1) -> set[str]:
        visited = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        while queue:
            current, current_depth = queue.popleft()
            if current_depth == depth:
                continue
            for neighbor in self.adjacency.get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, current_depth + 1))
        visited.remove(node_id)
        return visited

    def chunks_for_entities(self, entities: list[str]) -> set[str]:
        chunk_ids: set[str] = set()
        for entity in entities:
            chunk_ids.update(self.entity_to_chunks.get(entity, set()))
        return chunk_ids

    def _add_node(self, node_id: str, label: str, node_type: str) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(id=node_id, label=label, type=node_type)  # type: ignore[arg-type]

    def _add_edge(self, source: str, target: str, relation: str) -> None:
        edge = GraphEdge(source=source, target=target, relation=relation)
        self.edges.append(edge)
        self.adjacency[source].add(target)
        self.adjacency[target].add(source)

    def _connect_related_entities(self) -> None:
        for chunk in self.chunks.values():
            entity_ids = [self._entity_id(entity) for entity in chunk.entities]
            for left_index, left in enumerate(entity_ids):
                for right in entity_ids[left_index + 1 :]:
                    self._add_edge(left, right, "RELATED_TO")

    @staticmethod
    def _entity_id(entity: str) -> str:
        return "entity:" + entity.lower().replace(" ", "-")


def load_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        documents.append(
            Document(
                id=path.stem.lower().replace(" ", "-"),
                title=path.stem.replace("_", " ").title(),
                text=path.read_text(encoding="utf-8"),
                source=str(path),
            )
        )
    return documents
