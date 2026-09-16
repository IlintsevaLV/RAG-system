"""Abstract ports for LLM, OCR, stores — implementations come in later stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from core.ir import Citation, RegionBlock


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, grammar: str | None = None) -> str:
        ...


class VisionClient(ABC):
    @abstractmethod
    def read_image(self, image_path: str, prompt: str) -> str:
        ...


class OCREngine(ABC):
    @abstractmethod
    def recognize(self, image_path: str) -> dict[str, Any]:
        """Return text + line boxes + confidences."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        ...


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        ...

    @abstractmethod
    def search(self, vector: list[float], *, limit: int = 10) -> list[dict]:
        ...


class GraphStore(ABC):
    @abstractmethod
    def upsert_nodes(self, nodes: list[dict]) -> None:
        ...

    @abstractmethod
    def upsert_edges(self, edges: list[dict]) -> None:
        ...

    @abstractmethod
    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        ...


class GroundedAnswer(ABC):
    """Contract for final answers: text + citations or refusal."""

    @abstractmethod
    def answer(self, question: str) -> tuple[str | None, list[Citation]]:
        ...


# Placeholder so layout pipeline can type against a parser later
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, doc_path: str) -> list[RegionBlock]:
        ...
