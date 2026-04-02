from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from langchain_chroma import Chroma

from src.config import Settings, get_settings
from src.data_loader import chunk_medical_documents, load_medical_documents
from src.graph_store import LocalGraphStore
from src.llm_client import build_embedding_model
from src.schemas import RetrievalResult


INTENT_KEYWORDS = {
    "HAS_SYMPTOM": ["症状", "表现"],
    "HAS_COMPLICATION": ["并发症", "后果"],
    "RECOMMENDED_DRUG": ["药", "用药", "治疗", "抗生素"],
    "NEEDS_EXAM": ["检查", "化验", "检验"],
    "BELONGS_TO_DEPARTMENT": ["科室", "挂号"],
    "HAS_RISK_FACTOR": ["高危", "危险因素", "易感"],
    "HAS_PATHOGEN": ["病原体", "致病菌", "细菌", "病毒"],
    "BELONGS_TO_DRUG_CLASS": ["抗生素", "药物类别"],
}


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embeddings = build_embedding_model(self.settings)
        self.vectorstore = Chroma(
            collection_name=self.settings.chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=str(self.settings.chroma_dir),
        )
        self.graph_store = LocalGraphStore.load(self.settings.graph_store_path)

    def close(self) -> None:
        return None

    def rebuild_vector_index(self, reset: bool = False) -> dict[str, Any]:
        docs = load_medical_documents(self.settings.data_dir)
        chunks = chunk_medical_documents(docs)
        if reset:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
            self.vectorstore = Chroma(
                collection_name=self.settings.chroma_collection,
                embedding_function=self.embeddings,
                persist_directory=str(self.settings.chroma_dir),
            )
        self.vectorstore.add_documents(chunks)
        return {"documents": len(docs), "chunks": len(chunks)}

    def refresh_graph_store(self) -> None:
        self.graph_store = LocalGraphStore.load(self.settings.graph_store_path)

    def retrieve(
        self,
        query: str,
        top_k_text: int | None = None,
        max_hops: int | None = None,
    ) -> RetrievalResult:
        self.refresh_graph_store()
        top_k = top_k_text or self.settings.top_k_text
        hops = max_hops or self.settings.max_graph_hops
        vector_hits = self._vector_search(query, top_k=top_k)
        linked_entities = self._link_entities(query)
        graph_payload = self._graph_search(linked_entities, hops, query)
        hybrid_context = self._build_hybrid_context(graph_payload["triples"], vector_hits)
        return RetrievalResult(
            query=query,
            linked_entities=linked_entities,
            graph_triples=graph_payload["triples"],
            graph_nodes=graph_payload["nodes"],
            graph_edges=graph_payload["edges"],
            vector_hits=vector_hits,
            vector_only_hits=vector_hits,
            hybrid_context=hybrid_context,
        )

    def get_intent_hints(self, query: str) -> list[str]:
        return sorted(self._detect_relation_hints(query))

    def _vector_search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        try:
            pairs = self.vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception:
            pairs = []
        results = []
        for doc, distance in pairs:
            score = 1.0 / (1.0 + max(float(distance), 0.0))
            results.append(
                {
                    "content": doc.page_content,
                    "score": round(float(score), 4),
                    "metadata": doc.metadata,
                }
            )
        return results

    def _link_entities(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.graph_store.entity_rows()
        candidates = []
        for row in rows:
            names = [row["name"], *row.get("aliases", [])]
            scores = [self._entity_match_score(query, candidate) for candidate in names if candidate]
            if not scores:
                continue
            score = max(scores)
            if score >= 0.32:
                candidates.append(
                    {
                        "name": row["name"],
                        "labels": row["labels"],
                        "score": round(score, 4),
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def _graph_search(
        self,
        linked_entities: list[dict[str, Any]],
        max_hops: int,
        query: str,
    ) -> dict[str, list[dict[str, Any]]]:
        relation_hints = self._detect_relation_hints(query)
        seed_names = [item["name"] for item in linked_entities[:3]]
        payload = self.graph_store.multi_hop_neighbors(seed_names, max_hops=max_hops, limit=40)
        triples = payload["triples"]
        for triple in triples:
            triple["highlight"] = triple["relation"] in relation_hints if relation_hints else False
        triples.sort(
            key=lambda item: (
                1 if item["highlight"] else 0,
                item["confidence"],
            ),
            reverse=True,
        )
        return {
            "nodes": payload["nodes"],
            "edges": payload["edges"],
            "triples": triples[:20],
        }

    @staticmethod
    def _entity_match_score(query: str, candidate: str) -> float:
        if candidate in query:
            return 1.0
        return SequenceMatcher(None, query, candidate).ratio()

    @staticmethod
    def _build_hybrid_context(
        graph_triples: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
    ) -> str:
        graph_lines = [f"{item['head']} -[{item['relation']}]-> {item['tail']}" for item in graph_triples[:12]]
        text_lines = [f"[{item['metadata'].get('title', 'unknown')}] {item['content']}" for item in vector_hits[:4]]
        return "\n".join(["图谱上下文：", *graph_lines, "", "文本上下文：", *text_lines]).strip()

    @staticmethod
    def _detect_relation_hints(query: str) -> set[str]:
        hits = set()
        for relation, keywords in INTENT_KEYWORDS.items():
            if any(keyword in query for keyword in keywords):
                hits.add(relation)
        return hits


if __name__ == "__main__":
    retriever = HybridRetriever()
    print(retriever.retrieve("肺炎的并发症脓毒症有哪些适用抗生素？"))
