from __future__ import annotations

import json
import re
from typing import Any

from tqdm import tqdm

from src.config import Settings, get_settings
from src.data_loader import load_medical_documents
from src.graph_store import LocalGraphStore
from src.llm_client import build_chat_model
from src.prompts import EXTRACTION_PROMPT
from src.schemas import MedicalDocument, Triple


RELATION_KEYWORDS = {
    "BELONGS_TO_DEPARTMENT": (r"所属科室为([^。]+)", "Department"),
    "HAS_SYMPTOM": (r"常见症状包括([^。]+)", "Symptom"),
    "HAS_COMPLICATION": (r"常见并发症包括([^。]+)", "Complication"),
    "RECOMMENDED_DRUG": (r"常用药物包括([^。]+)", "Drug"),
    "NEEDS_EXAM": (r"推荐检查包括([^。]+)", "Examination"),
    "HAS_RISK_FACTOR": (r"高危因素包括([^。]+)", "RiskFactor"),
    "HAS_PATHOGEN": (r"(?:病原体|致病因素或病原体)包括([^。]+)", "Pathogen"),
}


class MedicalKnowledgeGraphBuilder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.chat_model = build_chat_model(self.settings, temperature=0.0)
        self.alias_map: dict[str, str] = {}
        self.graph_store = LocalGraphStore(self.settings.graph_store_path)

    def close(self) -> None:
        return None

    def reset_graph(self) -> None:
        self.graph_store.reset()

    def build_graph(self, limit: int | None = None, reset: bool = False) -> dict[str, Any]:
        docs = load_medical_documents(self.settings.data_dir)
        if limit is not None:
            docs = docs[:limit]
        if reset:
            self.reset_graph()

        total_relations = 0
        for doc in tqdm(docs, desc="Building local graph"):
            entities, relations = self.extract_entities_and_relations(doc)
            total_relations += len(relations)
            self.write_document_and_knowledge(doc, entities, relations)

        self.graph_store.save()
        return {
            "documents": len(docs),
            "relations": total_relations,
            "graph_store": str(self.settings.graph_store_path),
        }

    def extract_entities_and_relations(
        self,
        doc: MedicalDocument,
    ) -> tuple[list[dict[str, Any]], list[Triple]]:
        heuristic_entities, heuristic_relations = self._heuristic_extract(doc)
        if not self.chat_model:
            return heuristic_entities, heuristic_relations

        try:
            raw = self.chat_model.invoke(EXTRACTION_PROMPT.format(text=doc.text)).content
            payload = self._parse_json_payload(raw)
            entities = payload.get("entities") or heuristic_entities
            relations = [
                Triple(
                    head=item["head"],
                    relation=item["relation"],
                    tail=item["tail"],
                    head_type=item["head_type"],
                    tail_type=item["tail_type"],
                    evidence=item.get("evidence", doc.text[:160]),
                    confidence=float(item.get("confidence", 0.8)),
                )
                for item in payload.get("relations", [])
                if item.get("head") and item.get("tail") and item.get("relation")
            ]
            return entities or heuristic_entities, relations or heuristic_relations
        except Exception:
            return heuristic_entities, heuristic_relations

    def write_document_and_knowledge(
        self,
        doc: MedicalDocument,
        entities: list[dict[str, Any]],
        relations: list[Triple],
    ) -> None:
        self.graph_store.add_document(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "source": doc.source,
                "category": doc.category,
            }
        )
        for entity in entities:
            canonical = self._upsert_entity(entity)
            self.graph_store.add_document_mention(doc.doc_id, canonical)

        for triple in relations:
            head = self._canonical_name(triple.head)
            tail = self._canonical_name(triple.tail)
            if head == tail:
                continue
            self._upsert_entity({"name": head, "type": triple.head_type, "aliases": []})
            self._upsert_entity({"name": tail, "type": triple.tail_type, "aliases": []})
            self.graph_store.add_edge(
                source=head,
                relation=self._sanitize_relation(triple.relation),
                target=tail,
                evidence=triple.evidence,
                confidence=triple.confidence,
                source_doc_id=doc.doc_id,
            )

    def _upsert_entity(self, entity: dict[str, Any]) -> str:
        name = self._canonical_name(entity["name"])
        entity_type = self._sanitize_label(entity.get("type", "Entity"))
        aliases = [alias for alias in entity.get("aliases", []) if alias and alias != name]
        canonical = self.graph_store.add_entity(name=name, entity_type=entity_type, aliases=aliases)
        for alias in aliases:
            self.alias_map[alias] = canonical
            self.graph_store.add_alias(alias, canonical)
        return canonical

    def _heuristic_extract(
        self,
        doc: MedicalDocument,
    ) -> tuple[list[dict[str, Any]], list[Triple]]:
        text = doc.text
        disease_name = doc.metadata.get("disease") or doc.title.split("-")[0]
        entities: dict[str, dict[str, Any]] = {
            disease_name: {"name": disease_name, "type": "Disease", "aliases": []}
        }
        relations: list[Triple] = []

        alias_match = re.search(r"别名包括([^。]+)", text)
        if alias_match:
            aliases = self._split_items(alias_match.group(1))
            entities[disease_name]["aliases"] = aliases
            for alias in aliases:
                self.alias_map[alias] = disease_name
                entities.setdefault(alias, {"name": alias, "type": "Alias", "aliases": []})
                relations.append(
                    Triple(
                        head=alias,
                        relation="ALIAS_OF",
                        tail=disease_name,
                        head_type="Alias",
                        tail_type="Disease",
                        evidence=alias_match.group(0),
                        confidence=0.95,
                    )
                )

        for relation, (pattern, tail_type) in RELATION_KEYWORDS.items():
            match = re.search(pattern, text)
            if not match:
                continue
            items = self._split_items(match.group(1))
            for item in items:
                if item in {"暂无明确记录", "暂无明确抗生素"}:
                    continue
                entities.setdefault(item, {"name": item, "type": tail_type, "aliases": []})
                relations.append(
                    Triple(
                        head=disease_name,
                        relation=relation,
                        tail=item,
                        head_type="Disease",
                        tail_type=tail_type,
                        evidence=match.group(0),
                        confidence=0.9,
                    )
                )

        anti_match = re.search(r"适用抗生素包括([^。]+)", text)
        if anti_match:
            for drug in self._split_items(anti_match.group(1)):
                if drug == "暂无明确抗生素":
                    continue
                entities.setdefault(drug, {"name": drug, "type": "Drug", "aliases": []})
                entities.setdefault("抗生素", {"name": "抗生素", "type": "DrugClass", "aliases": []})
                relations.append(
                    Triple(
                        head=drug,
                        relation="BELONGS_TO_DRUG_CLASS",
                        tail="抗生素",
                        head_type="Drug",
                        tail_type="DrugClass",
                        evidence=anti_match.group(0),
                        confidence=0.92,
                    )
                )

        return list(entities.values()), relations

    @staticmethod
    def _split_items(raw: str) -> list[str]:
        values = []
        for part in re.split(r"[、，,]", raw):
            item = part.strip().strip("。")
            if item:
                values.append(item)
        return values

    @staticmethod
    def _parse_json_payload(raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("No JSON payload found")
        return json.loads(match.group(0))

    def _canonical_name(self, name: str) -> str:
        return self.alias_map.get(name, self.graph_store.alias_map.get(name, name))

    @staticmethod
    def _sanitize_label(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", value) or "Entity"
        if cleaned[0].isdigit():
            cleaned = f"Label{cleaned}"
        return cleaned

    @staticmethod
    def _sanitize_relation(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value) or "RELATED_TO"
        if cleaned[0].isdigit():
            cleaned = f"REL_{cleaned}"
        return cleaned


if __name__ == "__main__":
    builder = MedicalKnowledgeGraphBuilder()
    result = builder.build_graph(reset=True)
    print(result)
