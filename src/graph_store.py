from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


class LocalGraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.nodes: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.alias_map: dict[str, str] = {}
        self.document_mentions: dict[str, list[str]] = {}
        self._outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def reset(self) -> None:
        self.nodes.clear()
        self.documents.clear()
        self.edges.clear()
        self.alias_map.clear()
        self.document_mentions.clear()
        self._outgoing.clear()
        self._incoming.clear()

    def add_document(self, doc: dict[str, Any]) -> None:
        self.documents[doc["doc_id"]] = doc
        self.document_mentions.setdefault(doc["doc_id"], [])

    def add_entity(self, name: str, entity_type: str, aliases: list[str] | None = None) -> str:
        aliases = aliases or []
        canonical = self.alias_map.get(name, name)
        node = self.nodes.setdefault(
            canonical,
            {
                "id": canonical,
                "name": canonical,
                "type": entity_type,
                "aliases": [],
            },
        )
        if node["type"] == "Entity" and entity_type != "Entity":
            node["type"] = entity_type
        for alias in aliases:
            if alias and alias != canonical and alias not in node["aliases"]:
                node["aliases"].append(alias)
                self.alias_map[alias] = canonical
        return canonical

    def add_alias(self, alias: str, canonical: str) -> None:
        canonical = self.add_entity(canonical, "Disease")
        self.alias_map[alias] = canonical
        alias_node = self.nodes.setdefault(
            alias,
            {
                "id": alias,
                "name": alias,
                "type": "Alias",
                "aliases": [],
            },
        )
        alias_node["canonical_name"] = canonical
        if alias not in self.nodes[canonical]["aliases"]:
            self.nodes[canonical]["aliases"].append(alias)
        self.add_edge(
            alias,
            "ALIAS_OF",
            canonical,
            evidence=f"{alias} 是 {canonical} 的别名",
            confidence=0.99,
            source_doc_id="alias_alignment",
        )

    def add_document_mention(self, doc_id: str, entity_name: str) -> None:
        canonical = self.alias_map.get(entity_name, entity_name)
        mentions = self.document_mentions.setdefault(doc_id, [])
        if canonical not in mentions:
            mentions.append(canonical)

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        evidence: str,
        confidence: float,
        source_doc_id: str,
    ) -> None:
        source = self.alias_map.get(source, source)
        target = self.alias_map.get(target, target)
        edge = {
            "source": source,
            "relation": relation,
            "target": target,
            "evidence": evidence,
            "confidence": float(confidence),
            "source_doc_id": source_doc_id,
        }
        edge_key = (edge["source"], edge["relation"], edge["target"], edge["source_doc_id"])
        if any(
            (item["source"], item["relation"], item["target"], item["source_doc_id"]) == edge_key
            for item in self.edges
        ):
            return
        self.edges.append(edge)
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": sorted(self.nodes.values(), key=lambda item: item["name"]),
            "edges": self.edges,
            "documents": list(self.documents.values()),
            "alias_map": self.alias_map,
            "document_mentions": self.document_mentions,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalGraphStore":
        store = cls(path)
        if not path.exists():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node in payload.get("nodes", []):
            store.nodes[node["name"]] = node
        for doc in payload.get("documents", []):
            store.documents[doc["doc_id"]] = doc
        store.alias_map.update(payload.get("alias_map", {}))
        store.document_mentions.update(payload.get("document_mentions", {}))
        for edge in payload.get("edges", []):
            store.edges.append(edge)
            store._outgoing[edge["source"]].append(edge)
            store._incoming[edge["target"]].append(edge)
        return store

    def entity_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "name": node["name"],
                "labels": ["Entity", node["type"]],
                "aliases": node.get("aliases", []),
            }
            for node in self.nodes.values()
        ]

    def multi_hop_neighbors(self, seeds: list[str], max_hops: int, limit: int = 30) -> dict[str, list[dict[str, Any]]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        triples: dict[tuple[str, str, str], dict[str, Any]] = {}
        queue: deque[tuple[str, int]] = deque()
        seen_depth: dict[str, int] = {}

        for seed in seeds:
            canonical = self.alias_map.get(seed, seed)
            if canonical in self.nodes:
                queue.append((canonical, 0))
                seen_depth[canonical] = 0

        while queue and len(edges) < limit:
            current, depth = queue.popleft()
            node = self.nodes.get(current)
            if node:
                nodes[current] = {
                    "id": node["name"],
                    "label": node["name"],
                    "type": node.get("type", "Entity"),
                }
            if depth >= max_hops:
                continue

            connected_edges = [*self._outgoing.get(current, []), *self._incoming.get(current, [])]
            for edge in connected_edges:
                source = edge["source"]
                target = edge["target"]
                neighbor = target if source == current else source
                edge_key = (source, edge["relation"], target)
                edges[edge_key] = {
                    "source": source,
                    "target": target,
                    "relation": edge["relation"],
                    "evidence": edge.get("evidence", ""),
                }
                triples[edge_key] = {
                    "head": source,
                    "relation": edge["relation"],
                    "tail": target,
                    "evidence": edge.get("evidence", ""),
                    "confidence": edge.get("confidence", 0.0),
                }
                if neighbor not in nodes and neighbor in self.nodes:
                    neighbor_node = self.nodes[neighbor]
                    nodes[neighbor] = {
                        "id": neighbor_node["name"],
                        "label": neighbor_node["name"],
                        "type": neighbor_node.get("type", "Entity"),
                    }
                next_depth = depth + 1
                if neighbor not in seen_depth or next_depth < seen_depth[neighbor]:
                    seen_depth[neighbor] = next_depth
                    queue.append((neighbor, next_depth))

        return {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "triples": list(triples.values()),
        }
