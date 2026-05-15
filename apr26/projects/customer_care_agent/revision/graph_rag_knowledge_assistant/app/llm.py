from __future__ import annotations

import requests

from app.config import Settings
from app.models import RetrievedContext


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer(self, question: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "I could not find enough evidence in the indexed documents to answer that."

        prompt = self._build_prompt(question, contexts)
        ollama_answer = self._try_ollama(prompt)
        if ollama_answer:
            return ollama_answer
        return self._extractive_answer(question, contexts)

    def _try_ollama(self, prompt: str) -> str | None:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None
        data = response.json()
        answer = str(data.get("response", "")).strip()
        return answer or None

    @staticmethod
    def _build_prompt(question: str, contexts: list[RetrievedContext]) -> str:
        evidence = "\n\n".join(
            f"Source {index + 1} ({context.chunk.id}): {context.chunk.text}"
            for index, context in enumerate(contexts)
        )
        return (
            "Answer the question using only the evidence below. "
            "If the evidence is incomplete, say what is missing.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        )

    @staticmethod
    def _extractive_answer(question: str, contexts: list[RetrievedContext]) -> str:
        best = contexts[0]
        entities = ", ".join(best.graph_hops[:5]) or "the retrieved graph context"
        return (
            f"Based on the indexed evidence, {best.chunk.text} "
            f"Related graph entities include: {entities}."
        )
