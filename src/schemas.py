from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MedicalDocument:
    doc_id: str
    title: str
    text: str
    source: str
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    head_type: str
    tail_type: str
    evidence: str
    confidence: float = 0.75


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    evidence: str = ""


@dataclass
class RetrievalResult:
    query: str
    linked_entities: list[dict[str, Any]]
    graph_triples: list[dict[str, Any]]
    graph_nodes: list[dict[str, Any]]
    graph_edges: list[dict[str, Any]]
    vector_hits: list[dict[str, Any]]
    vector_only_hits: list[dict[str, Any]]
    hybrid_context: str
