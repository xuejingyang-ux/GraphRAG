import atexit
import io
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from llm_client import (
    create_openai_client,
    extract_message_text,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
)


app = FastAPI(title="MedGraphRAG API", version="2.0.0")

_kg_manager = None
_kg_init_error = None
_kg_lock = threading.Lock()
_llm_client = None
_llm_model = None
_vector_retriever = None
_vector_init_error = None
_vector_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_DB_PATH = os.path.join(DATA_DIR, "qa_history.db")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
except Exception:
    pass

logging.getLogger("neo4j").setLevel(logging.ERROR)

os.makedirs(DATA_DIR, exist_ok=True)


def history_conn():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db():
    conn = history_conn()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            evidence_chains TEXT,
            records_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_qa_history_created_at ON qa_history(created_at DESC)")
    conn.commit()
    conn.close()


def save_history(question: str, answer: str, evidence_chains: Optional[List[str]], records_count: int) -> int:
    conn = history_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO qa_history (question, answer, evidence_chains, records_count)
        VALUES (?, ?, ?, ?)
        """,
        (
            question or "",
            answer or "",
            json.dumps(evidence_chains or [], ensure_ascii=False),
            int(records_count or 0),
        ),
    )
    conn.commit()
    history_id = int(c.lastrowid)
    conn.close()
    return history_id


def list_history(limit: int = 30) -> List[Dict[str, Any]]:
    conn = history_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, question, answer, evidence_chains, records_count, created_at
        FROM qa_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = c.fetchall()
    conn.close()

    result: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["evidence_chains"] = json.loads(item.get("evidence_chains") or "[]")
        except Exception:
            item["evidence_chains"] = []
        result.append(item)
    return result


def delete_history_item(history_id: int) -> bool:
    conn = history_conn()
    c = conn.cursor()
    c.execute("DELETE FROM qa_history WHERE id = ?", (int(history_id),))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def clear_history() -> int:
    conn = history_conn()
    c = conn.cursor()
    c.execute("DELETE FROM qa_history")
    affected = c.rowcount
    conn.commit()
    conn.close()
    return int(affected or 0)


def count_vector_chunks_in_db() -> int:
    db_path = os.path.join(DATA_DIR, "vector_chunks.db")
    if not os.path.exists(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT count(*) AS c FROM vector_chunks").fetchone()
        conn.close()
        return safe_int((row or {}).get("c", 0), 0)
    except Exception:
        return 0


init_history_db()


def get_kg_manager():
    global _kg_manager, _kg_init_error
    if _kg_manager is not None:
        return _kg_manager
    if _kg_init_error:
        raise RuntimeError(_kg_init_error)

    with _kg_lock:
        if _kg_manager is not None:
            return _kg_manager
        if _kg_init_error:
            raise RuntimeError(_kg_init_error)
        try:
            from knowledge_graph import KnowledgeGraphManager

            _kg_manager = KnowledgeGraphManager()
            return _kg_manager
        except Exception as exc:
            _kg_init_error = f"知识图谱初始化失败: {exc}"
            raise RuntimeError(_kg_init_error) from exc


def get_llm_client() -> Tuple[Any, str]:
    global _llm_client, _llm_model
    if _llm_client is None:
        _llm_model = get_llm_model(default="Pro/zai-org/GLM-5.1")
        _llm_client = create_openai_client(
            base_url=get_llm_base_url(),
            api_key=get_llm_api_key(),
            chat_model_alias=_llm_model,
        )
    return _llm_client, _llm_model


def get_vector_retriever():
    global _vector_retriever, _vector_init_error
    if _vector_retriever is not None:
        return _vector_retriever
    if _vector_init_error:
        raise RuntimeError(_vector_init_error)

    with _vector_lock:
        if _vector_retriever is not None:
            return _vector_retriever
        if _vector_init_error:
            raise RuntimeError(_vector_init_error)
        try:
            from vector_retriever import HybridVectorRetriever

            _vector_retriever = HybridVectorRetriever(base_dir=BASE_DIR)
            return _vector_retriever
        except Exception as exc:
            _vector_init_error = f"向量检索模块初始化失败: {exc}"
            raise RuntimeError(_vector_init_error) from exc


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def get_ask_llm_timeout_seconds(default: float = 20.0) -> float:
    raw = os.getenv("ASK_LLM_TIMEOUT", "").strip()
    if not raw:
        raw = os.getenv("LLM_ASK_TIMEOUT", "").strip()
    if not raw:
        raw = str(default)
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(1.0, min(value, 120.0))


def is_pure_numeric_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and re.fullmatch(r"\d+", text) is not None


def is_technical_id_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.fullmatch(r"\d+:[0-9a-fA-F-]{8,}:\d+", text):
        return True
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", text):
        return True
    return False


def is_noise_entity_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if text.lower() in {"unknown", "none", "null", "nan", "未命名", "未知实体"}:
        return True
    return is_pure_numeric_text(text) or is_technical_id_text(text)


def filter_numeric_only_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records or []:
        source = str(rec.get("source", "")).strip()
        target = str(rec.get("target", "")).strip()
        if not source or not target:
            continue
        if is_noise_entity_text(source) or is_noise_entity_text(target):
            continue
        cleaned.append(rec)
    return cleaned


def load_jsonl_records(path: str, max_records: int = 0) -> List[Dict[str, Any]]:
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    lines: List[str] = []
    last_error = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        raise RuntimeError(f"读取 JSONL 失败: {last_error}")

    records: List[Dict[str, Any]] = []
    for idx, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
        if max_records > 0 and len(records) >= max_records:
            break

    return records


def count_noise_nodes_in_neo4j() -> int:
    kg = get_kg_manager()
    if not getattr(kg, "driver", None):
        return 0
    with kg.driver.session() as session:
        record = session.run(
            """
            MATCH (n)
            WITH n, properties(n) AS p
            WHERE trim(toString(coalesce(p['name'], p['title'], p['名称'], ''))) =~ '^[0-9]+$'
               OR trim(toString(coalesce(p['name'], p['title'], p['名称'], ''))) =~ '^[0-9]+:[0-9a-fA-F-]{8,}:[0-9]+$'
            RETURN count(n) AS c
            """
        ).single()
    return safe_int((record or {}).get("c", 0), 0)


def count_corpus_records_in_neo4j() -> int:
    kg = get_kg_manager()
    if not getattr(kg, "driver", None):
        return 0
    with kg.driver.session() as session:
        record = session.run(
            """
            CALL {
                MATCH (d:Document) RETURN count(d) AS docs
            }
            CALL {
                MATCH (c:CorpusRecord) RETURN count(c) AS corpus_records
            }
            RETURN docs + corpus_records AS c
            """
        ).single()
    return safe_int((record or {}).get("c", 0), 0)


def normalize_graph(raw_graph: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    raw_graph = raw_graph or {}
    raw_nodes = raw_graph.get("nodes", []) or []
    raw_links = raw_graph.get("links", []) or []
    raw_focus_ids = raw_graph.get("focus_ids", []) or []

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    node_id_set = set()
    for node in raw_nodes:
        node_id = str(
            node.get("id")
            or node.get("tid")
            or node.get("elementId")
            or node.get("name")
        ).strip()
        node_label = str(node.get("name") or node.get("label") or node_id).strip()

        if is_noise_entity_text(node_label):
            continue
        if not node_id or node_id in node_id_set:
            continue
        node_id_set.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": node_label,
                "type": str(node.get("type") or "Entity"),
            }
        )

    edge_seen = set()
    for rel in raw_links:
        source = rel.get("source_id", rel.get("source"))
        target = rel.get("target_id", rel.get("target"))
        if source is None or target is None:
            continue

        source_id = str(source)
        target_id = str(target)
        if source_id not in node_id_set or target_id not in node_id_set:
            continue
        rel_type = str(rel.get("type") or "RELATED")
        edge_key = (source_id, target_id, rel_type)
        if edge_key in edge_seen:
            continue
        edge_seen.add(edge_key)

        edges.append(
            {
                "from": source_id,
                "to": target_id,
                "label": rel_type,
            }
        )

    focus_ids: List[str] = []
    if isinstance(raw_focus_ids, list):
        focus_ids = [str(item).strip() for item in raw_focus_ids if str(item).strip()]
    elif raw_focus_ids:
        focus_ids = [str(raw_focus_ids).strip()]

    return {"nodes": nodes, "edges": edges, "focus_ids": focus_ids}


def graph_from_kg_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    node_map: Dict[str, Dict[str, str]] = {}
    edges: List[Dict[str, str]] = []
    edge_seen = set()

    for rec in records:
        source = str(rec.get("source", "")).strip()
        target = str(rec.get("target", "")).strip()
        rel_type = str(rec.get("relationship", "RELATED")).strip() or "RELATED"
        source_type = str(rec.get("source_type", "Entity"))
        target_type = str(rec.get("target_type", "Entity"))

        if not source or not target:
            continue
        if is_noise_entity_text(source) or is_noise_entity_text(target):
            continue

        node_map[source] = {"id": source, "label": source, "type": source_type}
        node_map[target] = {"id": target, "label": target, "type": target_type}

        key = (source, target, rel_type)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        edges.append({"from": source, "to": target, "label": rel_type})

    return {"nodes": list(node_map.values()), "edges": edges}


def focus_ids_from_records(records: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    focus: List[str] = []
    for rec in records or []:
        for key in ("source", "target"):
            value = str(rec.get(key, "") or "").strip()
            if not value or value in focus:
                continue
            focus.append(value)
            if len(focus) >= limit:
                return focus
    return focus


def records_from_search_graph(graph_data: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
    nodes = {str(n.get("id")): n for n in graph_data.get("nodes", [])}
    records: List[Dict[str, Any]] = []

    for edge in graph_data.get("edges", [])[:limit]:
        sid = str(edge.get("from", ""))
        tid = str(edge.get("to", ""))
        src_node = nodes.get(sid, {})
        tgt_node = nodes.get(tid, {})
        if not sid or not tid:
            continue
        source_name = str(src_node.get("label", sid)).strip()
        target_name = str(tgt_node.get("label", tid)).strip()
        if is_noise_entity_text(source_name) or is_noise_entity_text(target_name):
            continue
        records.append(
            {
                "source": source_name,
                "source_type": src_node.get("type", "Entity"),
                "relationship": edge.get("label", "RELATED"),
                "target": target_name,
                "target_type": tgt_node.get("type", "Entity"),
            }
        )
    return records


def merge_graph_payload(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    node_map = {n["id"]: n for n in a.get("nodes", [])}
    for node in b.get("nodes", []):
        node_map[node["id"]] = node

    seen = set()
    edges = []
    for item in (a.get("edges", []) + b.get("edges", [])):
        key = (item.get("from"), item.get("to"), item.get("label"))
        if key in seen:
            continue
        seen.add(key)
        edges.append(item)

    return {"nodes": list(node_map.values()), "edges": edges}


def keywords_from_question(question: str) -> List[str]:
    chunks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", question or "")
    seen = set()
    out = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            out.append(chunk)
    return out[:6]


def build_text_from_jsonl_record(record: Dict[str, Any]) -> str:
    title = str(record.get("title", "") or "").strip()
    disease = str(record.get("disease", "") or "").strip()
    category = str(record.get("category", "") or "").strip()
    source = str(record.get("source", "") or "").strip()
    body = str(record.get("text", "") or "").strip()

    parts = []
    if title:
        parts.append(f"标题: {title}")
    if disease:
        parts.append(f"疾病: {disease}")
    if category:
        parts.append(f"类别: {category}")
    if source:
        parts.append(f"来源: {source}")
    if body:
        parts.append(f"正文: {body}")

    return "\n".join(parts).strip()


def build_evidence_chains(records: List[Dict[str, Any]], max_items: int = 10) -> List[str]:
    chains: List[str] = []
    seen = set()
    for item in records[: max_items * 3]:
        s = str(item.get("source", "未知实体")).strip()
        r = str(item.get("relationship", "关联")).strip()
        t = str(item.get("target", "未知实体")).strip()
        if not s or not t:
            continue
        if is_noise_entity_text(s) or is_noise_entity_text(t):
            continue
        key = (s, r, t)
        if key in seen:
            continue
        seen.add(key)
        chains.append(f"{s} --{r}--> {t}")
        if len(chains) >= max_items:
            break
    return chains


def format_vector_hits_for_prompt(vector_hits: Optional[List[Dict[str, Any]]], max_items: int = 5) -> str:
    items = vector_hits or []
    if not items:
        return "未命中文本向量证据。"

    lines: List[str] = []
    for i, hit in enumerate(items[:max_items], start=1):
        score = float(hit.get("score", 0.0) or 0.0)
        source = str(hit.get("source", "") or "").strip() or "unknown"
        text = str(hit.get("text", "") or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 220:
            text = text[:220] + "..."
        lines.append(f"{i}. [score={score:.4f}] 来源={source} | 片段={text}")
    return "\n".join(lines) if lines else "未命中文本向量证据。"


def trim_vector_hits(vector_hits: Optional[List[Dict[str, Any]]], max_items: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for hit in (vector_hits or [])[:max_items]:
        out.append(
            {
                "chunk_id": str(hit.get("chunk_id", "") or ""),
                "source": str(hit.get("source", "") or ""),
                "score": float(hit.get("score", 0.0) or 0.0),
                "text": str(hit.get("text", "") or "")[:300],
            }
        )
    return out


def format_kg_records_for_prompt(records: List[Dict[str, Any]], max_lines: int = 40) -> str:
    if not records:
        return "未检索到明确关系，系统将结合全图近邻关系进行回答。"

    lines: List[str] = []
    for item in records[:max_lines]:
        s = item.get("source", "未知实体")
        st = item.get("source_type", "Entity")
        r = item.get("relationship", "RELATED")
        t = item.get("target", "未知实体")
        tt = item.get("target_type", "Entity")
        if is_noise_entity_text(s) or is_noise_entity_text(t):
            continue
        lines.append(f"- [{st}] {s} --({r})-> [{tt}] {t}")
    if not lines:
        return "未检索到明确关系，系统将结合全图近邻关系进行回答。"
    return "\n".join(lines)


def sanitize_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "已基于图谱关系完成分析，请参考下方结论与证据链。"

    for pattern in [
        r"当前图谱证据不足[。；!！]?",
        r"图谱证据不足[。；!！]?",
        r"证据不足[。；!！]?",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "已基于图谱关系完成分析，请参考下方结论与证据链。"


def call_llm_with_kg(
    question: str,
    kg_records: List[Dict[str, Any]],
    vector_hits: Optional[List[Dict[str, Any]]] = None,
) -> str:
    kg_context = format_kg_records_for_prompt(kg_records)
    evidence_lines = build_evidence_chains(kg_records, max_items=12)
    evidence_context = "\n".join([f"- {line}" for line in evidence_lines]) if evidence_lines else "- 全图近邻关系链路已启用"
    vector_context = format_vector_hits_for_prompt(vector_hits, max_items=5)

    prompt = f"""你是一个医疗知识问答助手。请根据下列图谱与文本证据，用中文给出简洁、可靠的回答。

用户问题：
{question}

知识图谱关系：
{kg_context}

图谱证据链参考：
{evidence_context}

文本证据：
{vector_context}

要求：
1. 先给结论，再给简短解释。
2. 优先依据图谱关系，不要编造。
3. 回答尽量简洁，适合直接展示在网页中。
"""

    client, model_name = get_llm_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是严谨、专业的医学知识图谱问答助手，不编造图谱中不存在的关系。",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
        temperature=0.2,
    )
    answer = sanitize_answer_text(extract_message_text(response.choices[0].message))

    if "图谱证据链" not in answer:
        chains = build_evidence_chains(kg_records, max_items=8)
        if not chains:
            chains = ["全图近邻关系链路：系统已从 Neo4j 全图选取高相关关系作为本次推理参考。"]
        chain_md = "\n".join([f"{i + 1}. {line}" for i, line in enumerate(chains)])
        answer = f"{answer}\n\n### 图谱证据链（可追踪）\n{chain_md}"

    if "文本证据" not in answer:
        vector_md_lines = []
        for idx, hit in enumerate((vector_hits or [])[:4], start=1):
            tx = str(hit.get("text", "") or "").strip()
            tx = re.sub(r"\s+", " ", tx)
            if len(tx) > 160:
                tx = tx[:160] + "..."
            vector_md_lines.append(f"{idx}. {tx}")
        if not vector_md_lines:
            vector_md_lines.append("1. 未命中可用文本片段（本次回答主要依赖图谱关系）。")
        answer = f"{answer}\n\n### 文本证据（向量检索）\n" + "\n".join(vector_md_lines)

    return answer or "已完成图谱检索与问答生成，请重试。"


def build_fallback_answer(question: str, kg_records: List[Dict[str, Any]], vector_hits: Optional[List[Dict[str, Any]]] = None) -> str:
    evidence_chains = build_evidence_chains(kg_records, max_items=8)
    if not evidence_chains:
        evidence_chains = ["全图近邻关系链路：系统已从 Neo4j 全图选取高相关关系作为本次推理参考。"]
    vector_md_lines = []
    for idx, hit in enumerate((vector_hits or [])[:4], start=1):
        tx = str(hit.get("text", "") or "").strip()
        tx = re.sub(r"\s+", " ", tx)
        if len(tx) > 160:
            tx = tx[:160] + "..."
        vector_md_lines.append(f"{idx}. {tx}")
    if not vector_md_lines:
        vector_md_lines.append("1. 未命中可用文本片段（本次回答主要依赖图谱关系）。")

    lead = "根据当前检索结果，系统已完成图谱与文本的联合分析。"
    if question.strip():
        lead = f"针对“{question.strip()}”，系统已完成图谱与文本的联合分析。"
    return (
        f"### 结论\n{lead}\n\n"
        f"### 图谱证据链（可追踪）\n"
        + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(evidence_chains[:8])])
        + "\n\n### 文本证据（向量检索）\n"
        + "\n".join(vector_md_lines)
    )


def call_llm_with_timeout(
    question: str,
    kg_records: List[Dict[str, Any]],
    vector_hits: Optional[List[Dict[str, Any]]] = None,
    timeout_seconds: float = 20.0,
) -> str:
    result: Dict[str, Any] = {"answer": None, "error": None}

    def worker() -> None:
        try:
            result["answer"] = sanitize_answer_text(call_llm_with_kg(question, kg_records, vector_hits))
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(max(1.0, float(timeout_seconds)))
    if thread.is_alive():
        raise TimeoutError(f"LLM response timed out after {timeout_seconds:.1f}s")
    if result["error"] is not None:
        raise result["error"]
    return str(result["answer"] or "").strip()


def build_qa_result(question: str) -> Dict[str, Any]:
    kg = get_kg_manager()
    kg_records: List[Dict[str, Any]] = []
    graph_payload: Dict[str, Any] = {"nodes": [], "edges": []}
    vector_hits: List[Dict[str, Any]] = []
    linked_entities: List[Dict[str, Any]] = []

    # A) 向量检索（Embedding + Top-K 余弦相似度）
    try:
        vector_retriever = get_vector_retriever()
        vector_hits = vector_retriever.search(question, top_k=5)
    except Exception:
        traceback.print_exc()
        vector_hits = []

    # B) 先走“实体链接 + 多跳图谱检索”
    try:
        kg_records = kg.process_user_query(
            text=question,
            save_to_db=False,
            depth=2,
            similarity_threshold=0.68,
            top_k=6,
        )
        kg_records = filter_numeric_only_records(kg_records)
    except Exception:
        traceback.print_exc()
        kg_records = []

    try:
        linked_entities = kg.link_query_entities(text=question, max_entities=3)
    except Exception:
        linked_entities = []

    # C) 图谱语义检索无结果时，走关键词图搜索兜底
    if not kg_records:
        merged = {"nodes": [], "edges": []}
        for kw in keywords_from_question(question):
            try:
                g = normalize_graph(kg.search_nodes(kw))
                merged = merge_graph_payload(merged, g)
            except Exception:
                continue
        graph_payload = merged
        kg_records = records_from_search_graph(merged, limit=60)
        kg_records = filter_numeric_only_records(kg_records)

    # D) 关键词仍弱命中时，使用全图近邻关系兜底
    if not kg_records:
        try:
            full_graph = normalize_graph(kg.query_whole_graph(limit=180))
            if full_graph.get("nodes"):
                graph_payload = full_graph
            kg_records = records_from_search_graph(full_graph, limit=60)
            kg_records = filter_numeric_only_records(kg_records)
        except Exception:
            pass

    # E) 检索融合后交给大模型生成答案
    try:
        answer = call_llm_with_timeout(
            question,
            kg_records,
            vector_hits=vector_hits,
            timeout_seconds=get_ask_llm_timeout_seconds(default=20.0),
        )
    except Exception as exc:
        if not isinstance(exc, TimeoutError):
            traceback.print_exc()
        answer = build_fallback_answer(question, kg_records, vector_hits=vector_hits)
        answer = f"{answer}\n\n> LLM 调用失败：{exc}"
    evidence_chains = build_evidence_chains(kg_records, max_items=12)
    if not evidence_chains:
        evidence_chains = ["全图近邻关系链路：系统已从 Neo4j 全图选取高相关关系作为本次推理参考。"]

    graph_records = kg_records[:24] if kg_records else []
    if graph_records:
        graph_payload = graph_from_kg_records(graph_records)
        graph_focus = focus_ids_from_records(graph_records, limit=4)
        if linked_entities:
            for ent in linked_entities:
                name = str(ent.get("name") or "").strip()
                if name and name not in graph_focus:
                    graph_focus.insert(0, name)
        graph_payload["focus_ids"] = graph_focus[:4]

    if "图谱证据链" not in answer:
        chain_md = "\n".join([f"{i + 1}. {line}" for i, line in enumerate(evidence_chains[:8])])
        answer = f"{answer}\n\n### 图谱证据链（可追踪）\n{chain_md}"

    return {
        "success": True,
        "answer": answer,
        "graph": graph_payload,
        "records_count": len(kg_records),
        "evidence_chains": evidence_chains,
        "vector_hits": trim_vector_hits(vector_hits, max_items=5),
    }


def status_payload() -> Dict[str, Any]:
    payload = {
        "neo4j_connected": False,
        "entities": 0,
        "relationships": 0,
        "kg_error": None,
        "vector_chunks": 0,
        "vector_error": None,
    }
    try:
        kg = get_kg_manager()
        payload["neo4j_connected"] = bool(kg.check_connection())
        payload["neo4j_status"] = "已连接" if payload["neo4j_connected"] else f"未连接（{getattr(kg, '_connect_error', None) or '已启用本地回退'}）"
        stats = kg.get_kg_statistics()
        payload["entities"] = safe_int(stats.get("entities", 0), 0)
        payload["relationships"] = safe_int(stats.get("relationships", 0), 0)
    except Exception as exc:
        payload["kg_error"] = str(exc)
    try:
        if _vector_retriever is not None:
            vec_status = _vector_retriever.status()
            payload["vector_chunks"] = safe_int(vec_status.get("chunks", 0), 0)
            payload["vector_error"] = vec_status.get("last_error")
        else:
            payload["vector_chunks"] = count_vector_chunks_in_db()
    except Exception as exc:
        payload["vector_error"] = str(exc)
    return payload


def purge_numeric_nodes_in_neo4j() -> int:
    kg = get_kg_manager()
    if not getattr(kg, "driver", None):
        return 0
    with kg.driver.session() as session:
        record = session.run(
            """
            MATCH (n)
            WITH n, properties(n) AS p
            WHERE trim(toString(coalesce(p['name'], p['title'], p['名称'], ''))) =~ '^[0-9]+$'
               OR trim(toString(coalesce(p['name'], p['title'], p['名称'], ''))) =~ '^[0-9]+:[0-9a-fA-F-]{8,}:[0-9]+$'
            WITH collect(n) AS nodes
            FOREACH (x IN nodes | DETACH DELETE x)
            RETURN size(nodes) AS deleted_count
            """
        ).single()
    return safe_int((record or {}).get("deleted_count", 0), 0)


MAX_DOCUMENT_UPLOAD_BYTES = 25 * 1024 * 1024
DOCUMENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".csv", ".xls", ".xlsx", ".txt"}


def normalize_preview(text: str, limit: int = 1200) -> str:
    text = re.sub(r"\r\n?", "\n", str(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


async def read_document_upload(file: UploadFile) -> Dict[str, Any]:
    filename = os.path.basename(file.filename or "upload")
    data = await file.read()
    if not data:
        raise ValueError("上传文件为空。")
    if len(data) > MAX_DOCUMENT_UPLOAD_BYTES:
        raise ValueError("上传文件超过 25MB，请先压缩或拆分后再解析。")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in DOCUMENT_EXTENSIONS:
        raise ValueError("暂只支持 PDF、图片、CSV、Excel 和 TXT 文件。")

    return {
        "filename": filename,
        "ext": ext,
        "content_type": file.content_type or "application/octet-stream",
        "size": len(data),
        "data": data,
    }


def document_upload_error(exc: Exception) -> JSONResponse:
    status_code = 400 if isinstance(exc, ValueError) else 500
    return JSONResponse(status_code=status_code, content={"success": False, "error": str(exc)})


def read_pdf_pages(data: bytes, max_pages: int = 25) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("缺少 pypdf，无法解析 PDF。请先安装 pypdf。") from exc

    reader = PdfReader(io.BytesIO(data))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            pass

    pages: List[Dict[str, Any]] = []
    for index, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        mediabox = getattr(page, "mediabox", None)
        width = float(getattr(mediabox, "width", 0) or 0)
        height = float(getattr(mediabox, "height", 0) or 0)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pages.append(
            {
                "page": index,
                "width": round(width, 2),
                "height": round(height, 2),
                "line_count": len(lines),
                "char_count": len(text),
                "text": text,
                "preview": normalize_preview(text, 700),
            }
        )

    metadata = {}
    try:
        metadata = {str(k).lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    except Exception:
        metadata = {}

    return pages, {"page_count": len(reader.pages), "metadata": metadata}


def parse_uploaded_pdf(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload["ext"] != ".pdf":
        return {"success": False, "error": "PDF解析仅支持 .pdf 文件。", "filename": payload["filename"]}

    pages, info = read_pdf_pages(payload["data"])
    full_text = "\n\n".join(page.get("text", "") for page in pages).strip()
    return {
        "success": True,
        "task": "pdf",
        "filename": payload["filename"],
        "size": payload["size"],
        "page_count": info["page_count"],
        "parsed_pages": len(pages),
        "metadata": info["metadata"],
        "text_length": len(full_text),
        "text_preview": normalize_preview(full_text, 1800),
        "pages": [{k: v for k, v in page.items() if k != "text"} for page in pages[:12]],
    }


def detect_image_blocks(image: Any) -> Dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return {"block_count": 0, "blocks": [], "warning": "缺少 OpenCV，无法做图像块检测。"}

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    merged = cv2.dilate(threshold, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    height, width = gray.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < max(120, width * height * 0.00015) or w < 12 or h < 6:
            continue
        boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area": int(area)})
    boxes.sort(key=lambda item: (item["y"], item["x"]))
    return {"block_count": len(boxes), "blocks": boxes[:40]}


def ocr_image_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("缺少 Pillow，无法读取图片。") from exc

    image = Image.open(io.BytesIO(payload["data"]))
    image.load()
    blocks = detect_image_blocks(image)
    result = {
        "success": True,
        "task": "ocr",
        "filename": payload["filename"],
        "size": payload["size"],
        "image": {"width": image.width, "height": image.height, "mode": image.mode},
        "text": "",
        "text_preview": "",
        "ocr_engine": "not_available",
        "warnings": [],
        "layout_blocks": blocks,
    }

    try:
        import pytesseract

        try:
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        except Exception:
            text = pytesseract.image_to_string(image)
        result["text"] = text.strip()
        result["text_preview"] = normalize_preview(text, 1800)
        result["ocr_engine"] = "tesseract"
    except Exception as exc:
        result["warnings"].append(f"OCR 引擎不可用：{exc}。请安装 Tesseract 与 pytesseract 后重试。")
    return result


def recognize_uploaded_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload["ext"] in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return ocr_image_payload(payload)
    if payload["ext"] == ".pdf":
        pages, info = read_pdf_pages(payload["data"], max_pages=8)
        text = "\n\n".join(page.get("text", "") for page in pages).strip()
        return {
            "success": True,
            "task": "ocr",
            "filename": payload["filename"],
            "size": payload["size"],
            "page_count": info["page_count"],
            "text": text,
            "text_preview": normalize_preview(text, 1800),
            "ocr_engine": "pdf_text_layer",
            "warnings": ["当前环境未配置 PDF 页面渲染，已优先读取 PDF 文本层；扫描版 PDF 需安装 PyMuPDF 或 pdf2image + Tesseract。"],
        }
    return {"success": False, "error": "OCR识别仅支持 PDF 或图片文件。", "filename": payload["filename"]}


def dataframe_to_table_payload(df: Any, sheet_name: str = "table") -> Dict[str, Any]:
    limited = df.head(80)
    return {
        "sheet": sheet_name,
        "columns": [str(col) for col in list(limited.columns)],
        "row_count": int(len(df)),
        "preview_rows": limited.fillna("").astype(str).to_dict(orient="records"),
    }


def infer_text_tables(text: str) -> List[Dict[str, Any]]:
    rows: List[List[str]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if "\t" in raw:
            cells = [cell.strip() for cell in raw.split("\t")]
        elif "|" in raw:
            cells = [cell.strip() for cell in raw.strip("|").split("|")]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", raw)]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 2:
            rows.append(cells)
        if len(rows) >= 80:
            break

    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    columns = [f"列{i + 1}" for i in range(max_cols)]
    return [{"sheet": "text_table_1", "columns": columns, "row_count": len(rows), "preview_rows": [dict(zip(columns, row)) for row in normalized]}]


def structure_uploaded_tables(payload: Dict[str, Any]) -> Dict[str, Any]:
    ext = payload["ext"]
    tables: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if ext == ".csv":
        import pandas as pd

        text = decode_text_bytes(payload["data"])
        df = pd.read_csv(io.StringIO(text))
        tables.append(dataframe_to_table_payload(df, "csv"))
    elif ext in {".xls", ".xlsx"}:
        import pandas as pd

        sheets = pd.read_excel(io.BytesIO(payload["data"]), sheet_name=None)
        for name, df in sheets.items():
            tables.append(dataframe_to_table_payload(df, str(name)))
    elif ext == ".pdf":
        pages, _ = read_pdf_pages(payload["data"], max_pages=12)
        text = "\n".join(page.get("text", "") for page in pages)
        tables = infer_text_tables(text)
        if not tables:
            warnings.append("未从 PDF 文本层识别到稳定表格；复杂 PDF 表格建议安装 camelot/pdfplumber 后增强。")
    else:
        text = decode_text_bytes(payload["data"])
        tables = infer_text_tables(text)
        if not tables:
            warnings.append("当前文件未识别到可结构化的表格行。")

    return {
        "success": True,
        "task": "tables",
        "filename": payload["filename"],
        "size": payload["size"],
        "table_count": len(tables),
        "tables": tables[:8],
        "warnings": warnings,
    }


def analyze_uploaded_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    ext = payload["ext"]
    if ext == ".pdf":
        pages, info = read_pdf_pages(payload["data"], max_pages=20)
        layout_pages = []
        for page in pages:
            text = page.get("text", "")
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            layout_pages.append(
                {
                    "page": page["page"],
                    "width": page["width"],
                    "height": page["height"],
                    "line_count": len(lines),
                    "paragraph_count": len(paragraphs),
                    "estimated_columns": 2 if len(lines) > 25 and page["width"] > page["height"] * 0.7 else 1,
                    "preview": page["preview"],
                }
            )
        return {
            "success": True,
            "task": "layout",
            "filename": payload["filename"],
            "size": payload["size"],
            "page_count": info["page_count"],
            "pages": layout_pages,
            "warnings": ["当前版面分析基于 PDF 文本层和页面尺寸做轻量估计。"],
        }

    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("缺少 Pillow，无法读取图片。") from exc
        image = Image.open(io.BytesIO(payload["data"]))
        image.load()
        blocks = detect_image_blocks(image)
        return {
            "success": True,
            "task": "layout",
            "filename": payload["filename"],
            "size": payload["size"],
            "image": {"width": image.width, "height": image.height, "mode": image.mode},
            "layout_blocks": blocks,
            "warnings": [],
        }

    text = decode_text_bytes(payload["data"])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "success": True,
        "task": "layout",
        "filename": payload["filename"],
        "size": payload["size"],
        "line_count": len(lines),
        "paragraph_count": len([part for part in re.split(r"\n\s*\n", text) if part.strip()]),
        "preview": normalize_preview(text, 1200),
        "warnings": ["文本类文件无页面坐标，已输出逻辑段落结构。"],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=PAGE_TEMPLATE)


@app.get("/api/status")
def api_status():
    return status_payload()


@app.get("/api/graph/full")
def api_graph_full(limit: int = Query(default=300)):
    limit = safe_int(limit, 300)
    limit = max(50, min(limit, 1200))
    try:
        kg = get_kg_manager()
        graph = normalize_graph(kg.query_whole_graph(limit=limit))
        return {"success": True, "graph": graph}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc), "graph": {"nodes": [], "edges": []}},
        )


@app.get("/api/graph/search")
def api_graph_search(q: str = Query(default="")):
    q = (q or "").strip()
    if not q:
        return JSONResponse(status_code=400, content={"success": False, "error": "请输入搜索关键词。"})
    try:
        kg = get_kg_manager()
        graph = normalize_graph(kg.search_nodes(q))
        return {"success": True, "graph": graph}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc), "graph": {"nodes": [], "edges": []}},
        )


@app.post("/api/graph/purge_numeric_nodes")
def api_graph_purge_numeric_nodes():
    try:
        deleted_count = purge_numeric_nodes_in_neo4j()
        return {"success": True, "deleted_count": deleted_count}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"删除纯数字节点失败: {exc}"})


@app.get("/api/graph/verify_corpus")
def api_graph_verify_corpus():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(base_dir, "data", "medical_corpus.jsonl")
        total_jsonl = len(load_jsonl_records(corpus_path, max_records=0)) if os.path.exists(corpus_path) else 0
        corpus_nodes = count_corpus_records_in_neo4j()
        noise_nodes = count_noise_nodes_in_neo4j()
        stats = status_payload()
        return {
            "success": True,
            "jsonl_total": total_jsonl,
            "neo4j_corpus_records": corpus_nodes,
            "noise_nodes": noise_nodes,
            "entities": safe_int(stats.get("entities", 0), 0),
            "relationships": safe_int(stats.get("relationships", 0), 0),
            "verified": bool(total_jsonl > 0 and corpus_nodes == total_jsonl and noise_nodes == 0),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.post("/api/import/jsonl")
def api_import_jsonl(data: Optional[Dict[str, Any]] = Body(default=None)):
    data = data or {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "data", "medical_corpus.jsonl")

    raw_path = str(data.get("path", "") or default_path).strip()
    if not raw_path:
        raw_path = default_path
    path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    path = os.path.normpath(path)

    max_records = safe_int(data.get("max_records", 0), 0)
    if max_records < 0:
        max_records = 0
    max_records = min(max_records, 5000) if max_records > 0 else 0

    reset_graph = bool(data.get("reset", False))
    use_llm = bool(data.get("use_llm", True))

    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"success": False, "error": f"文件不存在: {path}"})

    from src.kg_builder import MedicalKnowledgeGraphBuilder

    builder = MedicalKnowledgeGraphBuilder(jsonl_path=path, use_llm=use_llm)
    try:
        summary = builder.build_graph(
            reset=reset_graph,
            limit=(max_records if max_records > 0 else None),
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"导入失败: {exc}"})
    finally:
        try:
            builder.close()
        except Exception:
            pass

    return {
        "success": True,
        "message": "JSONL 导入完成（读取文档 -> 抽取实体关系 -> 写入 Neo4j）。",
        "path": path,
        "total_records": safe_int(summary.get("total_records", 0), 0),
        "processed": safe_int(summary.get("processed", 0), 0),
        "failed": safe_int(summary.get("failed", 0), 0),
        "documents_in_graph": safe_int(summary.get("documents_in_graph", 0), 0),
        "entities_in_graph": safe_int(summary.get("entities_in_graph", 0), 0),
        "mentions_in_graph": safe_int(summary.get("mentions_in_graph", 0), 0),
        "relations_in_graph": safe_int(summary.get("relations_in_graph", 0), 0),
        "noise_nodes": safe_int(summary.get("noise_nodes", 0), 0),
        "verified": bool(summary.get("verified", False)),
        "errors": summary.get("sample_errors", [])[:10],
    }


@app.post("/api/ask")
def api_ask(data: Optional[Dict[str, Any]] = Body(default=None)):
    data = data or {}
    question = (data.get("question", "") or "").strip()
    save_flag = bool(data.get("save_history", True))
    if not question:
        return JSONResponse(status_code=400, content={"success": False, "error": "问题不能为空。"})
    try:
        result = build_qa_result(question)
        if save_flag:
            try:
                history_id = save_history(
                    question=question,
                    answer=result.get("answer", ""),
                    evidence_chains=result.get("evidence_chains", []),
                    records_count=safe_int(result.get("records_count", 0), 0),
                )
                result["history_id"] = history_id
            except Exception as history_exc:
                result["history_id"] = None
                result["history_error"] = str(history_exc)
        return result
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": f"问答失败: {exc}"})


@app.get("/api/history")
def api_history(limit: int = Query(default=50)):
    limit = safe_int(limit, 50)
    limit = max(1, min(limit, 500))
    try:
        items = list_history(limit=limit)
        return {"success": True, "items": items}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"读取历史失败: {exc}", "items": []})


@app.delete("/api/history/{history_id}")
def api_history_delete(history_id: int):
    try:
        deleted = delete_history_item(history_id)
        return {"success": True, "deleted": bool(deleted), "id": history_id}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"删除历史失败: {exc}"})


@app.delete("/api/history")
def api_history_clear():
    try:
        deleted_count = clear_history()
        return {"success": True, "deleted_count": deleted_count}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"清空历史失败: {exc}"})


@app.post("/api/document/pdf")
async def api_document_pdf(file: UploadFile = File(...)):
    try:
        return parse_uploaded_pdf(await read_document_upload(file))
    except Exception as exc:
        return document_upload_error(exc)


@app.post("/api/document/ocr")
async def api_document_ocr(file: UploadFile = File(...)):
    try:
        return recognize_uploaded_document(await read_document_upload(file))
    except Exception as exc:
        return document_upload_error(exc)


@app.post("/api/document/tables")
async def api_document_tables(file: UploadFile = File(...)):
    try:
        return structure_uploaded_tables(await read_document_upload(file))
    except Exception as exc:
        return document_upload_error(exc)


@app.post("/api/document/layout")
async def api_document_layout(file: UploadFile = File(...)):
    try:
        return analyze_uploaded_layout(await read_document_upload(file))
    except Exception as exc:
        return document_upload_error(exc)


# 旧入口统一收口到新问答页
@app.get("/login")
@app.get("/register")
@app.get("/health_profile")
@app.get("/diagnosis")
@app.get("/image_analysis")
@app.get("/profile/settings")
@app.get("/admin")
@app.get("/admin/{any_path:path}")
def legacy_redirect(any_path: Optional[str] = None):
    return RedirectResponse(url="/")


@atexit.register
def close_resources():
    global _kg_manager, _vector_retriever
    try:
        if _kg_manager is not None and hasattr(_kg_manager, "close"):
            _kg_manager.close()
    except Exception:
        pass
    try:
        if _vector_retriever is not None and hasattr(_vector_retriever, "close"):
            _vector_retriever.close()
    except Exception:
        pass


PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedGraphRAG 医疗知识图谱问答系统</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
        :root { --bg:#0a0a1a; --card:rgba(255,255,255,.05); --text:#e0f0ff; --dim:#8a9ba8; --primary:#00f5ff; --secondary:#ff00ff; --border:rgba(0,245,255,.14); }
        *{margin:0;padding:0;box-sizing:border-box}
        body{min-height:100vh;background:var(--bg);color:var(--text);font-family:'Noto Sans SC',sans-serif;overflow-x:hidden;position:relative}
        body::before{content:"";position:fixed;inset:0;background-image:linear-gradient(rgba(0,245,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,.03) 1px,transparent 1px);background-size:48px 48px;animation:moveGrid 24s linear infinite;pointer-events:none}
        body::after{content:"";position:fixed;width:760px;height:760px;left:50%;top:48%;transform:translate(-50%,-50%);background:radial-gradient(circle,rgba(0,245,255,.1),transparent 68%);pointer-events:none}
        #particles-js{position:fixed;inset:0;z-index:0;pointer-events:none}
        #particles-js canvas{opacity:.82;filter:saturate(1.22) brightness(1.08)}
        @keyframes moveGrid{from{transform:translate(0,0)}to{transform:translate(48px,48px)}}
        .wrap{width:min(1320px,94%);margin:24px auto 30px;position:relative;z-index:1}
        .hero{background:linear-gradient(135deg,rgba(0,245,255,.1),rgba(255,0,255,.07));border:1px solid var(--border);border-radius:20px;padding:22px 24px;margin-bottom:18px;backdrop-filter:blur(14px);box-shadow:0 12px 30px rgba(0,0,0,.35)}
        .hero h1{font-family:'Orbitron',sans-serif;font-size:clamp(22px,3vw,34px);letter-spacing:.6px;margin-bottom:8px;background:linear-gradient(135deg,var(--primary),#e0f0ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .hero p{color:var(--dim);line-height:1.75}
        .badge-row{margin-top:12px;display:flex;gap:10px;flex-wrap:wrap}
        .badge{border:1px solid var(--border);background:rgba(255,255,255,.03);color:var(--text);border-radius:999px;padding:6px 12px;font-size:12px}
        .grid{display:grid;grid-template-columns:1.05fr .95fr;gap:16px}
        .card{background:var(--card);border:1px solid var(--border);border-radius:18px;backdrop-filter:blur(14px);box-shadow:0 14px 34px rgba(0,0,0,.35);padding:18px;min-height:640px}
        .card h2{font-size:18px;color:var(--primary);margin-bottom:14px;display:flex;align-items:center;gap:8px}
        .qa-input{width:100%;min-height:130px;resize:vertical;border-radius:12px;border:1px solid var(--border);background:rgba(255,255,255,.03);color:var(--text);outline:none;font-size:15px;line-height:1.7;padding:12px 14px}
        .qa-input:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(0,245,255,.12)}
        .actions{margin-top:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .example-wrap{margin-top:12px;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:10px;background:rgba(255,255,255,.02)}
        .example-title{color:var(--dim);font-size:13px;margin-bottom:8px;display:flex;align-items:center;gap:6px}
        .example-list{display:flex;flex-wrap:wrap;gap:8px}
        .example-chip{border:1px solid var(--border);background:rgba(0,245,255,.07);color:var(--text);border-radius:999px;padding:6px 10px;cursor:pointer;font-size:12px;transition:.2s ease}
        .example-chip:hover{background:rgba(255,0,255,.14);border-color:rgba(255,0,255,.35)}
        .btn{border:none;border-radius:10px;padding:11px 16px;cursor:pointer;font-weight:700;letter-spacing:.2px;transition:.2s ease}
        .btn-primary{color:#04040e;background:linear-gradient(135deg,var(--primary),var(--secondary))}
        .btn-primary:hover{transform:translateY(-1px);box-shadow:0 9px 20px rgba(0,245,255,.28)}
        .btn-ghost{color:var(--text);background:rgba(255,255,255,.05);border:1px solid var(--border)}
        .btn-ghost:hover{background:rgba(0,245,255,.08)}
        .hint{color:var(--dim);font-size:13px}
        .answer{margin-top:14px;border-radius:12px;border:1px solid rgba(255,255,255,.08);min-height:320px;padding:14px;background:rgba(255,255,255,.02);overflow:auto}
        .answer h1,.answer h2,.answer h3{color:var(--primary);margin:10px 0}
        .answer p,.answer li{line-height:1.85}
        .answer ul,.answer ol{padding-left:20px}
        .answer code{background:rgba(0,245,255,.09);padding:2px 6px;border-radius:6px}
        .history-wrap{margin-top:12px;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:10px;background:rgba(255,255,255,.02)}
        .history-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
        .history-title{color:var(--dim);font-size:13px;display:flex;align-items:center;gap:6px}
        .history-list{display:flex;flex-direction:column;gap:8px;max-height:220px;overflow:auto}
        .history-empty{color:var(--dim);font-size:13px;padding:8px 2px}
        .history-item{border:1px solid var(--border);border-radius:10px;padding:9px 10px;background:rgba(0,245,255,.05);display:flex;align-items:flex-start;justify-content:space-between;gap:8px}
        .history-main{cursor:pointer;flex:1;min-width:0}
        .history-q{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .history-meta{font-size:12px;color:var(--dim);margin-top:4px}
        .history-del{border:1px solid rgba(255,0,255,.4);background:rgba(255,0,255,.1);color:#ffd7ff;border-radius:8px;padding:4px 8px;cursor:pointer}
        .history-del:hover{background:rgba(255,0,255,.2)}
        .doc-panel{margin-top:12px;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px;background:rgba(255,255,255,.02)}
        .doc-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
        .doc-title{color:var(--dim);font-size:13px;display:flex;align-items:center;gap:6px}
        .doc-file{width:100%;border:1px dashed var(--border);border-radius:12px;background:rgba(0,245,255,.04);color:var(--text);padding:10px;outline:none}
        .doc-file::file-selector-button{border:none;border-radius:8px;background:linear-gradient(135deg,var(--primary),var(--secondary));color:#04040e;font-weight:700;padding:8px 12px;margin-right:10px;cursor:pointer}
        .doc-actions{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:10px}
        .doc-actions .btn{padding:10px 8px;white-space:nowrap}
        .doc-result{margin-top:10px;min-height:74px;max-height:260px;overflow:auto;border:1px solid rgba(255,255,255,.08);border-radius:12px;background:rgba(0,0,0,.16);padding:10px;color:var(--dim);font-size:13px;line-height:1.75}
        .doc-result strong{color:var(--primary)}
        .doc-result pre{white-space:pre-wrap;color:var(--text);font-family:'Noto Sans SC',sans-serif}
        .graph-toolbar{display:flex;gap:8px;margin-bottom:10px}
        .graph-toolbar input{flex:1;border-radius:10px;border:1px solid var(--border);background:rgba(255,255,255,.03);color:var(--text);padding:10px 12px;outline:none}
        .graph-toolbar input:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(0,245,255,.12)}
        #graph{width:100%;height:560px;border:1px solid rgba(255,255,255,.08);border-radius:12px;background:rgba(0,0,0,.22)}
        .graph-foot{margin-top:10px;color:var(--dim);font-size:13px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
        @media (max-width:980px){.grid{grid-template-columns:1fr}.card{min-height:auto}.doc-actions{grid-template-columns:repeat(2,minmax(0,1fr))}#graph{height:460px}}
        @media (max-width:560px){.doc-actions{grid-template-columns:1fr}.doc-actions .btn{white-space:normal}}
    </style>
</head>
<body>
    <div id="particles-js" aria-hidden="true"></div>
    <div class="wrap">
        <section class="hero">
            <h1><i class="fa-solid fa-brain"></i> MedGraphRAG 医疗知识图谱问答系统</h1>
            <p>系统先进行图谱关系检索，再交给大模型回答，并在结果中给出可追踪证据链。</p>
            <div class="badge-row">
                <span class="badge" id="badgeConn">Neo4j: 检测中...</span>
                <span class="badge" id="badgeEntity">实体: -</span>
                <span class="badge" id="badgeRel">关系: -</span>
            </div>
        </section>

        <section class="grid">
            <article class="card">
                <h2><i class="fa-solid fa-comments"></i> 知识问答</h2>
                <textarea class="qa-input" id="question" placeholder="请输入问题，或点击下方与当前图谱强相关的示例问题"></textarea>
                <div class="actions">
                    <button class="btn btn-primary" id="askBtn"><i class="fa-solid fa-paper-plane"></i> 开始问答</button>
                    <span class="hint">回答会优先使用知识图谱中的关系作为证据，并附带证据链。</span>
                </div>
                <div class="example-wrap">
                    <div class="example-title"><i class="fa-solid fa-lightbulb"></i> 图谱强相关示例问题</div>
                    <div class="example-list" id="exampleQuestions"></div>
                </div>
                <div class="answer" id="answer">请输入问题后点击“开始问答”。</div>
                <div class="history-wrap">
                    <div class="history-head">
                        <div class="history-title"><i class="fa-solid fa-clock-rotate-left"></i> 问答历史（长期保存）</div>
                        <button class="btn btn-ghost" id="clearHistoryBtn" style="padding:6px 10px;font-size:12px;">清空</button>
                    </div>
                    <div class="history-list" id="historyList"></div>
                </div>
                <div class="doc-panel">
                    <div class="doc-head">
                        <div class="doc-title"><i class="fa-solid fa-file-waveform"></i> 文档解析</div>
                        <span class="hint">PDF / OCR / 表格 / 版面</span>
                    </div>
                    <input class="doc-file" id="docFile" type="file" accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.csv,.xls,.xlsx,.txt">
                    <div class="doc-actions">
                        <button class="btn btn-ghost doc-action" data-doc-task="pdf"><i class="fa-solid fa-file-pdf"></i> PDF解析</button>
                        <button class="btn btn-ghost doc-action" data-doc-task="ocr"><i class="fa-solid fa-eye"></i> OCR识别</button>
                        <button class="btn btn-ghost doc-action" data-doc-task="tables"><i class="fa-solid fa-table-cells"></i> 表格结构化</button>
                        <button class="btn btn-ghost doc-action" data-doc-task="layout"><i class="fa-solid fa-object-group"></i> 版面分析</button>
                    </div>
                    <div class="doc-result" id="docResult">等待解析结果。</div>
                </div>
            </article>

            <article class="card">
                <h2><i class="fa-solid fa-diagram-project"></i> Neo4j 知识图谱</h2>
                <div class="graph-toolbar">
                    <input id="graphKeyword" placeholder="输入图谱中的实体关键词，例如：疾病名、症状名、药物名">
                    <button class="btn btn-ghost" id="searchGraphBtn"><i class="fa-solid fa-magnifying-glass"></i> 搜索</button>
                    <button class="btn btn-ghost" id="resetGraphBtn"><i class="fa-solid fa-rotate"></i> 全图</button>
                </div>
                <div id="graph"></div>
                <div class="graph-foot">
                    <span id="graphInfo">图谱加载中...</span>
                    <span>提示：拖拽节点可探索关系。</span>
                </div>
            </article>
        </section>
    </div>

    <script>
        const answerEl = document.getElementById('answer');
        const askBtn = document.getElementById('askBtn');
        const questionEl = document.getElementById('question');
        const graphInfoEl = document.getElementById('graphInfo');
        const badgeConn = document.getElementById('badgeConn');
        const badgeEntity = document.getElementById('badgeEntity');
        const badgeRel = document.getElementById('badgeRel');
        const graphKeywordEl = document.getElementById('graphKeyword');
        const searchGraphBtn = document.getElementById('searchGraphBtn');
        const resetGraphBtn = document.getElementById('resetGraphBtn');
        const exampleQuestionsEl = document.getElementById('exampleQuestions');
        const historyListEl = document.getElementById('historyList');
        const clearHistoryBtn = document.getElementById('clearHistoryBtn');
        const docFileEl = document.getElementById('docFile');
        const docResultEl = document.getElementById('docResult');
        const docActionBtns = Array.from(document.querySelectorAll('.doc-action'));

        let network = null;
        let graphFitTimer = null;
        let historyItems = [];
        const docEndpointByTask = {
            pdf: '/api/document/pdf',
            ocr: '/api/document/ocr',
            tables: '/api/document/tables',
            layout: '/api/document/layout'
        };

        const typeColors = {
            Disease: '#ff6b6b',
            disease: '#ff6b6b',
            疾病: '#ff6b6b',
            Symptom: '#f39c12',
            symptom: '#f39c12',
            疾病症状: '#f39c12',
            症状: '#f39c12',
            Drug: '#2ecc71',
            drug: '#2ecc71',
            药品: '#2ecc71',
            Food: '#16a085',
            food: '#16a085',
            食物: '#16a085',
            Check: '#3498db',
            check: '#3498db',
            检查项目: '#3498db',
            Department: '#9b59b6',
            department: '#9b59b6',
            科目: '#9b59b6',
            Cure: '#e67e22',
            cure: '#e67e22',
            治疗方法: '#e67e22',
            Producer: '#e84393',
            producer: '#e84393',
            药品商: '#e84393',
            Document: '#95a5a6',
            document: '#95a5a6',
            CorpusRecord: '#7f8c8d',
            corpusrecord: '#7f8c8d',
            Entity: '#00f5ff',
            entity: '#00f5ff'
        };
        const fallbackTypePalette = ['#1abc9c', '#f1c40f', '#8e44ad', '#27ae60', '#d35400', '#2c3e50', '#c0392b', '#2980b9'];

        function getTypeColor(typeName) {
            const raw = String(typeName || 'Entity').trim();
            if (typeColors[raw]) return typeColors[raw];
            const lower = raw.toLowerCase();
            if (typeColors[lower]) return typeColors[lower];

            let hash = 0;
            for (let i = 0; i < raw.length; i += 1) hash = ((hash << 5) - hash) + raw.charCodeAt(i);
            const index = Math.abs(hash) % fallbackTypePalette.length;
            return fallbackTypePalette[index];
        }

        function safeText(text) {
            return String(text || '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        }

        function renderAnswer(markdownText) {
            if (window.marked) answerEl.innerHTML = marked.parse(markdownText || '');
            else answerEl.innerHTML = `<pre style="white-space:pre-wrap;">${safeText(markdownText || '')}</pre>`;
        }

        function formatBytes(bytes) {
            const size = Number(bytes || 0);
            if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(2)} MB`;
            if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
            return `${size} B`;
        }

        function summarizeDocumentResult(data) {
            const lines = [];
            const taskNames = { pdf: 'PDF解析', ocr: 'OCR识别', tables: '表格结构化', layout: '版面分析' };
            lines.push(`任务：${taskNames[data.task] || data.task || '文档解析'}`);
            lines.push(`文件：${data.filename || '-'}（${formatBytes(data.size)}）`);
            if (data.page_count !== undefined) lines.push(`页数：${data.page_count}`);
            if (data.parsed_pages !== undefined) lines.push(`已解析页：${data.parsed_pages}`);
            if (data.text_length !== undefined) lines.push(`文本长度：${data.text_length}`);
            if (data.table_count !== undefined) lines.push(`表格数量：${data.table_count}`);
            if (data.image) lines.push(`图像：${data.image.width} x ${data.image.height} / ${data.image.mode}`);
            if (data.ocr_engine) lines.push(`识别引擎：${data.ocr_engine}`);
            if (data.layout_blocks?.block_count !== undefined) lines.push(`版面块：${data.layout_blocks.block_count}`);
            if (Array.isArray(data.warnings) && data.warnings.length) lines.push(`提示：${data.warnings.join('；')}`);
            return lines.join('\n');
        }

        function renderDocumentResult(data) {
            const blocks = [`<strong>解析完成</strong><pre>${safeText(summarizeDocumentResult(data))}</pre>`];
            const preview = data.text_preview || data.preview || '';
            if (preview) blocks.push(`<strong>文本预览</strong><pre>${safeText(preview)}</pre>`);
            if (Array.isArray(data.pages) && data.pages.length) {
                const pageLines = data.pages.slice(0, 8).map((p) => `P${p.page}: ${p.width || '-'} x ${p.height || '-'}，行 ${p.line_count ?? '-'}，字 ${p.char_count ?? '-'}`);
                blocks.push(`<strong>页面结构</strong><pre>${safeText(pageLines.join('\n'))}</pre>`);
            }
            if (Array.isArray(data.tables) && data.tables.length) {
                const tableLines = data.tables.slice(0, 3).map((table) => {
                    const rows = (table.preview_rows || []).slice(0, 4);
                    return `${table.sheet || 'table'}：${table.row_count || 0} 行\n${JSON.stringify(rows, null, 2)}`;
                });
                blocks.push(`<strong>表格预览</strong><pre>${safeText(tableLines.join('\n\n'))}</pre>`);
            }
            if (data.layout_blocks?.blocks?.length) {
                blocks.push(`<strong>版面块预览</strong><pre>${safeText(JSON.stringify(data.layout_blocks.blocks.slice(0, 10), null, 2))}</pre>`);
            }
            docResultEl.innerHTML = blocks.join('');
        }

        async function runDocumentTask(task, btn) {
            const endpoint = docEndpointByTask[task];
            const file = docFileEl?.files?.[0];
            if (!endpoint || !file) {
                docResultEl.innerHTML = '<span style="color:#ffb703;">请先选择文件。</span>';
                return;
            }
            const formData = new FormData();
            formData.append('file', file);
            const originalHtml = btn.innerHTML;
            docActionBtns.forEach((item) => { item.disabled = true; });
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 解析中';
            docResultEl.textContent = '正在解析...';
            try {
                const resp = await fetch(endpoint, { method: 'POST', body: formData });
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '解析失败');
                renderDocumentResult(data);
            } catch (err) {
                docResultEl.innerHTML = `<span style="color:#ff6b6b;">解析失败：${safeText(err.message)}</span>`;
            } finally {
                docActionBtns.forEach((item) => { item.disabled = false; });
                btn.innerHTML = originalHtml;
            }
        }

        function buildGraphRelatedExamples(graph) {
            const examples = [];
            const nodes = graph?.nodes || [];
            const edges = graph?.edges || [];
            const nodeMap = new Map(nodes.map((n) => [String(n.id), { label: (n.label || n.id), type: (n.type || 'Entity') }]));
            const seen = new Set();

            function addExample(text) {
                const q = String(text || '').trim();
                if (!q || seen.has(q)) return;
                seen.add(q);
                examples.push(q);
            }

            function isValidLabel(label) {
                const text = String(label || '').trim();
                if (!text) return false;
                if (/^\d+$/.test(text)) return false;
                if (text.length > 40) return false;
                return true;
            }

            function relationToQuestion(source, target, relation) {
                const rel = String(relation || '').toLowerCase();
                if (!isValidLabel(source) || !isValidLabel(target)) return null;

                if (/(has[_-]?symptom|症状)/.test(rel)) return `${source}有哪些常见症状？`;
                if (/(common[_-]?drug|recommand[_-]?drug|药品|用药)/.test(rel)) return `${source}常用药有哪些？`;
                if (/(do[_-]?eat|recommand[_-]?eat|宜吃|饮食)/.test(rel)) return `${source}适合吃什么？`;
                if (/(no[_-]?eat|not[_-]?eat|忌吃)/.test(rel)) return `${source}不能吃什么？`;
                if (/(need[_-]?check|检查)/.test(rel)) return `${source}需要做哪些检查？`;
                if (/(belongs[_-]?to|科目|科室|department)/.test(rel)) return `${source}应该挂什么科？`;
                if (/(cure[_-]?way|治疗)/.test(rel)) return `${source}怎么治疗？`;
                if (/(acompany|并发)/.test(rel)) return `${source}会并发哪些疾病？`;
                if (/(produces|drug[_-]?detail|生产|厂商|药品商)/.test(rel)) return `${target}是哪个药品商生产的？`;
                return `${source}和${target}有什么关系？`;
            }

            for (const e of edges) {
                const fromNode = nodeMap.get(String(e.from)) || { label: String(e.from || ''), type: 'Entity' };
                const toNode = nodeMap.get(String(e.to)) || { label: String(e.to || ''), type: 'Entity' };
                const fromLabel = fromNode.label;
                const toLabel = toNode.label;
                const rel = String(e.label || '关联');
                if (!isValidLabel(fromLabel) || !isValidLabel(toLabel)) continue;
                addExample(relationToQuestion(fromLabel, toLabel, rel));
                if (examples.length >= 8) break;
            }

            if (examples.length === 0 && nodes.length > 0) {
                const diseaseNodes = nodes.filter((n) => String(n.type || '').toLowerCase() === 'disease').slice(0, 3);
                const focusNodes = (diseaseNodes.length ? diseaseNodes : nodes.slice(0, 3))
                    .map((n) => String(n.label || '').trim())
                    .filter(isValidLabel);
                for (const name of focusNodes) {
                    addExample(`${name}有哪些症状？`);
                    addExample(`${name}怎么治疗？`);
                    addExample(`${name}常用药有哪些？`);
                }
            }

            if (examples.length === 0) {
                addExample('感冒有哪些症状？');
                addExample('感冒怎么治疗？');
                addExample('高血压常用药有哪些？');
                addExample('糖尿病适合吃什么？');
                addExample('胃炎需要做哪些检查？');
            }

            return Array.from(new Set(examples)).slice(0, 5);
        }

        function renderExampleQuestions(graph) {
            const examples = buildGraphRelatedExamples(graph);
            exampleQuestionsEl.innerHTML = '';
            examples.forEach((q) => {
                const btn = document.createElement('button');
                btn.className = 'example-chip';
                btn.type = 'button';
                btn.textContent = q;
                btn.addEventListener('click', () => { questionEl.value = q; questionEl.focus(); });
                exampleQuestionsEl.appendChild(btn);
            });
        }

        function formatHistoryTime(rawTime) {
            const text = String(rawTime || '');
            if (!text) return '-';
            const normalized = text.includes('T') ? text : text.replace(' ', 'T');
            const dt = new Date(normalized);
            if (Number.isNaN(dt.getTime())) return text;
            return dt.toLocaleString('zh-CN', { hour12: false });
        }

        function renderHistoryList() {
            historyListEl.innerHTML = '';
            if (!historyItems.length) {
                const empty = document.createElement('div');
                empty.className = 'history-empty';
                empty.textContent = '暂无历史记录，开始提问后会自动保存。';
                historyListEl.appendChild(empty);
                return;
            }

            historyItems.forEach((item) => {
                const row = document.createElement('div');
                row.className = 'history-item';

                const main = document.createElement('div');
                main.className = 'history-main';
                main.title = '点击恢复这条问答';

                const q = document.createElement('div');
                q.className = 'history-q';
                q.textContent = item.question || '(无问题文本)';

                const meta = document.createElement('div');
                meta.className = 'history-meta';
                meta.textContent = `${formatHistoryTime(item.created_at)} ｜ 证据关系 ${item.records_count ?? 0}`;

                main.appendChild(q);
                main.appendChild(meta);
                main.addEventListener('click', () => {
                    questionEl.value = item.question || '';
                    renderAnswer(item.answer || '');
                    questionEl.focus();
                });

                const delBtn = document.createElement('button');
                delBtn.className = 'history-del';
                delBtn.type = 'button';
                delBtn.textContent = '删除';
                delBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    await deleteHistoryItem(item.id);
                });

                row.appendChild(main);
                row.appendChild(delBtn);
                historyListEl.appendChild(row);
            });
        }

        async function loadHistory() {
            try {
                const resp = await fetch('/api/history?limit=80');
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '加载历史失败');
                historyItems = Array.isArray(data.items) ? data.items : [];
                renderHistoryList();
            } catch (err) {
                historyItems = [];
                historyListEl.innerHTML = `<div class="history-empty">历史记录加载失败：${safeText(err.message)}</div>`;
            }
        }

        async function deleteHistoryItem(id) {
            try {
                const resp = await fetch(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '删除失败');
                await loadHistory();
            } catch (err) {
                renderAnswer(`删除历史失败：${err.message}`);
            }
        }

        async function clearAllHistory() {
            try {
                const resp = await fetch('/api/history', { method: 'DELETE' });
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '清空失败');
                await loadHistory();
            } catch (err) {
                renderAnswer(`清空历史失败：${err.message}`);
            }
        }

        function renderGraph(graph, opts = {}) {
            const shouldAutoFit = opts.autoFit !== false;
            const focusIds = new Set((graph?.focus_ids || []).map((id) => String(id)));
            const nodes = (graph?.nodes || []).map((n) => {
                const isFocus = focusIds.has(String(n.id));
                const color = getTypeColor(n.type);
                return {
                    id: n.id,
                    label: n.label || n.id,
                    title: `${n.label || n.id} (${n.type || 'Entity'})`,
                    color: {
                        background: isFocus ? '#ffb703' : color,
                        border: isFocus ? '#ffffff' : '#0b0d18',
                        highlight: { background: '#ff00ff', border: '#ffffff' }
                    },
                    font: { color: '#0a0a1a', size: 14, face: 'Noto Sans SC' },
                    shape: 'dot',
                    size: isFocus ? 28 : 18
                };
            });
            const edges = (graph?.edges || []).map((e) => ({
                from: e.from, to: e.to, label: e.label || 'RELATED', arrows: 'to',
                color: { color: 'rgba(0,245,255,.38)', highlight: '#ff00ff' },
                font: { color: '#9ab0c3', size: 11, strokeWidth: 0 }
            }));
            graphInfoEl.textContent = `节点 ${nodes.length} ｜ 边 ${edges.length}`;
            const container = document.getElementById('graph');
            if (!window.vis || !window.vis.Network) {
                container.innerHTML = '<div style="padding:16px;color:#ff6b6b;">图谱可视化组件加载失败，请检查网络后刷新页面。</div>';
                return;
            }
            const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
            const options = {
                autoResize: true,
                interaction: { hover: true, navigationButtons: true },
                physics: { enabled: true, stabilization: false, barnesHut: { gravitationalConstant: -5500, springLength: 160, springConstant: 0.045 } },
                layout: { improvedLayout: true },
                edges: { smooth: { type: 'dynamic' } }
            };
            if (!network) network = new vis.Network(container, data, options);
            else { network.setData(data); network.setOptions(options); }
            if (focusIds.size && network && typeof network.selectNodes === 'function') {
                const targets = Array.from(focusIds).filter((id) => nodes.some((n) => String(n.id) === String(id)));
                if (targets.length) {
                    network.selectNodes(targets, false);
                    network.focus(targets[0], { scale: 1.05, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
                }
            }
            if (shouldAutoFit && network && typeof network.fit === 'function') {
                window.clearTimeout(graphFitTimer);
                graphFitTimer = window.setTimeout(() => {
                    try {
                        network.fit({ animation: { duration: 300, easingFunction: 'easeInOutQuad' } });
                    } catch (err) {
                        console.warn('graph fit failed:', err);
                    }
                }, 180);
            }
        }

        async function fetchStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                badgeConn.textContent = `Neo4j: ${data.neo4j_status || (data.neo4j_connected ? '已连接' : '未连接')}`;
                badgeEntity.textContent = `实体: ${data.entities ?? '-'}`;
                badgeRel.textContent = `关系: ${data.relationships ?? '-'}`;
                if (data.kg_error) answerEl.innerHTML = `<p style="color:#ff6b6b;">知识图谱初始化异常：${safeText(data.kg_error)}</p>`;
            } catch (e) {
                badgeConn.textContent = 'Neo4j: 状态获取失败';
            }
        }

        async function loadFullGraph() {
            graphInfoEl.textContent = '加载全图中...';
            try {
                const resp = await fetch('/api/graph/full?limit=300');
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '加载失败');
                renderGraph(data.graph, { autoFit: true });
                renderExampleQuestions(data.graph);
            } catch (err) {
                graphInfoEl.textContent = `图谱加载失败：${err.message}`;
            }
        }

        async function searchGraph() {
            const q = graphKeywordEl.value.trim();
            if (!q) return loadFullGraph();
            graphInfoEl.textContent = '搜索图谱中...';
            try {
                const resp = await fetch(`/api/graph/search?q=${encodeURIComponent(q)}`);
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '搜索失败');
                renderGraph(data.graph, { autoFit: false });
                renderExampleQuestions(data.graph);
            } catch (err) {
                graphInfoEl.textContent = `搜索失败：${err.message}`;
            }
        }

        async function askQuestion() {
            const question = questionEl.value.trim();
            if (!question) return renderAnswer('请输入问题。');
            askBtn.disabled = true;
            askBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 问答中...';
            renderAnswer('正在检索知识图谱并生成回答，请稍候...');
            try {
                const resp = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await resp.json();
                if (!resp.ok || !data.success) throw new Error(data.error || '问答失败');
                renderAnswer(data.answer || '未生成回答');
                if (data.graph && Array.isArray(data.graph.nodes)) {
                    renderGraph(data.graph, { autoFit: false });
                    renderExampleQuestions(data.graph);
                }
                await loadHistory();
            } catch (err) {
                renderAnswer(`问答失败：${err.message}`);
            } finally {
                askBtn.disabled = false;
                askBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> 开始问答';
            }
        }

        function initParticles() {
            if (!window.particlesJS) return;
            try {
                window.particlesJS('particles-js', {
                    particles: {
                        number: { value: 96, density: { enable: true, value_area: 900 } },
                        color: { value: ['#00f5ff', '#7dd3fc', '#ff00ff'] },
                        shape: { type: 'circle' },
                        opacity: { value: 0.62, random: true },
                        size: { value: 3.1, random: true },
                        line_linked: {
                            enable: true,
                            distance: 165,
                            color: '#00f5ff',
                            opacity: 0.48,
                            width: 1.2
                        },
                        move: {
                            enable: true,
                            speed: 1.8,
                            direction: 'none',
                            random: false,
                            straight: false,
                            out_mode: 'out',
                            bounce: false
                        }
                    },
                    interactivity: {
                        detect_on: 'canvas',
                        events: {
                            onhover: { enable: true, mode: 'grab' },
                            onclick: { enable: true, mode: 'push' },
                            resize: true
                        },
                        modes: {
                            grab: { distance: 190, line_linked: { opacity: 0.68 } },
                            push: { particles_nb: 5 }
                        }
                    },
                    retina_detect: true
                });
            } catch (err) {
                console.warn('particles init failed:', err);
            }
        }

        askBtn.addEventListener('click', askQuestion);
        searchGraphBtn.addEventListener('click', searchGraph);
        resetGraphBtn.addEventListener('click', loadFullGraph);
        clearHistoryBtn.addEventListener('click', clearAllHistory);
        docActionBtns.forEach((btn) => {
            btn.addEventListener('click', () => runDocumentTask(btn.dataset.docTask, btn));
        });
        graphKeywordEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') searchGraph(); });
        questionEl.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) askQuestion(); });

        fetchStatus();
        loadFullGraph();
        loadHistory();
        renderExampleQuestions({ nodes: [], edges: [] });
        initParticles();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
