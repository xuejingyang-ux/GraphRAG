from __future__ import annotations

from typing import Any

from src.config import Settings, get_settings
from src.llm_client import build_chat_model
from src.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT
from src.retriever import HybridRetriever


class GraphRAGQAChain:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self.chat_model = build_chat_model(self.settings, temperature=0.1)

    def answer(self, question: str) -> dict[str, Any]:
        return self.run_pipeline(question)

    def run_pipeline(self, question: str, status_callback: Any | None = None) -> dict[str, Any]:
        intent_hints = self.retriever.get_intent_hints(question)
        self._emit_status(
            status_callback,
            "intent",
            "正在解析用户问题意图",
            detail=f"识别到 {len(intent_hints)} 类候选意图：{', '.join(intent_hints) if intent_hints else '未显式命中，转入宽检索'}",
        )

        retrieval = self.retriever.retrieve(question)
        linked_names = [item["name"] for item in retrieval.linked_entities[:4]]
        self._emit_status(
            status_callback,
            "retrieval",
            "正在执行混合检索",
            detail=(
                f"向量召回 {len(retrieval.vector_hits)} 段文本；"
                f"图谱命中实体 {linked_names if linked_names else '无'}；"
                f"最大检索跳数 {self.settings.max_graph_hops}"
            ),
        )

        graph_context = self._render_graph_context(retrieval.graph_triples)
        text_context = self._render_text_context(retrieval.vector_hits)
        if self._should_refuse(question, retrieval):
            answer = "无法根据当前知识库作答。当前问题没有足够的图谱关系或高置信文本证据支撑，知识库也不包含个人联系方式等越界信息。"
            trace = self._build_trace(question, retrieval, answer, intent_hints, refused=True)
            self._emit_status(
                status_callback,
                "complete",
                "已完成安全边界判断",
                detail="当前问题超出知识库覆盖范围或命中越界规则，系统返回拒答。",
                state="complete",
            )
            return {
                "answer": answer,
                "retrieval": retrieval,
                "trace": trace,
            }

        self._emit_status(
            status_callback,
            "generation",
            "正在生成最终回答",
            detail="已融合图谱三元组与文本片段，开始生成受约束答案。",
        )
        answer = self._generate_answer(question, graph_context, text_context)
        trace = self._build_trace(question, retrieval, answer, intent_hints, refused=False)
        self._emit_status(
            status_callback,
            "complete",
            "问答完成",
            detail=f"生成答案 1 条；图谱边 {len(retrieval.graph_edges)}；文本证据 {len(retrieval.vector_hits)}。",
            state="complete",
        )
        return {
            "answer": answer,
            "retrieval": retrieval,
            "trace": trace,
        }

    @staticmethod
    def _emit_status(
        status_callback: Any | None,
        step: str,
        label: str,
        detail: str = "",
        state: str = "running",
    ) -> None:
        if status_callback:
            status_callback(
                {
                    "step": step,
                    "label": label,
                    "detail": detail,
                    "state": state,
                }
            )

    def _build_trace(
        self,
        question: str,
        retrieval: Any,
        answer: str,
        intent_hints: list[str],
        refused: bool,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "refused": refused,
            "intent_hints": intent_hints,
            "linked_entities": retrieval.linked_entities,
            "graph_triples": retrieval.graph_triples,
            "graph_nodes": retrieval.graph_nodes,
            "graph_edges": retrieval.graph_edges,
            "vector_hits": retrieval.vector_hits,
            "hybrid_context": retrieval.hybrid_context,
        }

    def _generate_answer(self, question: str, graph_context: str, text_context: str) -> str:
        if self.chat_model:
            try:
                prompt = (
                    f"{QA_SYSTEM_PROMPT}\n\n"
                    + QA_USER_PROMPT.format(
                        question=question,
                        graph_context=graph_context or "无",
                        text_context=text_context or "无",
                    )
                )
                response = self.chat_model.invoke(prompt)
                return getattr(response, "content", response)
            except Exception:
                pass
        return self._rule_based_answer(question, graph_context, text_context)

    @staticmethod
    def _render_graph_context(graph_triples: list[dict[str, Any]]) -> str:
        if not graph_triples:
            return ""
        return "\n".join(
            f"{item['head']} -[{item['relation']}]-> {item['tail']}；证据：{item['evidence']}"
            for item in graph_triples[:12]
        )

    @staticmethod
    def _render_text_context(vector_hits: list[dict[str, Any]]) -> str:
        if not vector_hits:
            return ""
        return "\n".join(
            f"{item['metadata'].get('title', 'unknown')}（分数 {item['score']}）：{item['content']}"
            for item in vector_hits[:4]
        )

    @staticmethod
    def _should_refuse(question: str, retrieval: Any) -> bool:
        boundary_keywords = ["手机号", "电话", "住址", "身份证", "邮箱", "医生联系方式"]
        if any(keyword in question for keyword in boundary_keywords):
            return True
        if retrieval.graph_triples:
            return False
        if not retrieval.vector_hits:
            return True
        top_score = max(item["score"] for item in retrieval.vector_hits)
        return top_score < 0.35 and not retrieval.linked_entities

    @staticmethod
    def _rule_based_answer(question: str, graph_context: str, text_context: str) -> str:
        if "抗生素" in question and "BELONGS_TO_DRUG_CLASS" in graph_context:
            lines = []
            seen = set()
            for line in graph_context.splitlines():
                if "-[BELONGS_TO_DRUG_CLASS]-> 抗生素" in line:
                    drug = line.split(" -[", 1)[0].strip()
                    if drug not in seen:
                        seen.add(drug)
                        lines.append(drug)
            if lines:
                return (
                    "根据当前知识图谱，可以命中的适用抗生素包括："
                    + "、".join(lines)
                    + "。溯源摘要：答案主要来自图谱中的 BELONGS_TO_DRUG_CLASS 关系和相关治疗文本。"
                )

        if graph_context:
            first_lines = "；".join(graph_context.splitlines()[:4])
            return (
                "根据当前知识库，已检索到以下关键关系："
                + first_lines
                + "。如需更精确结论，请补充更具体的疾病、并发症或药物条件。"
            )

        if text_context:
            first_text = text_context.splitlines()[0]
            return f"当前仅命中文本证据：{first_text}。由于缺少图谱关系支撑，建议谨慎使用该回答。"

        return "无法根据当前知识库作答。"
