from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from src.schemas import MedicalDocument


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？；\n]+", text) if part.strip()]


def _read_json_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("items", [])


def load_medical_documents(data_dir: Path) -> list[MedicalDocument]:
    docs: list[MedicalDocument] = []
    for path in sorted(data_dir.glob("*.json*")):
        for row in _read_json_file(path):
            docs.append(
                MedicalDocument(
                    doc_id=str(row["doc_id"]),
                    title=row["title"],
                    text=clean_text(row["text"]),
                    source=row.get("source", path.name),
                    category=row.get("category", "general"),
                    metadata={k: v for k, v in row.items() if k not in {"doc_id", "title", "text", "source", "category"}},
                )
            )
    return docs


def _sentence_chunks(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;])|(?:\r?\n)+", text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _sliding_window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    sentences = _sentence_chunks(text)
    if not sentences:
        sentences = [text.strip()]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            step = max(chunk_size - chunk_overlap, 1)
            while start < len(sentence):
                chunk = sentence[start : start + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                start += step
            continue

        candidate = sentence if not current else f"{current}\n{sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        overlap = current[-chunk_overlap:].strip() if chunk_overlap and current else ""
        current = sentence if not overlap else f"{overlap}\n{sentence}"
        if len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = current[max(chunk_size - chunk_overlap, 1) :].strip()

    if current:
        chunks.append(current.strip())

    deduped: list[str] = []
    for chunk in chunks:
        if chunk and (not deduped or deduped[-1] != chunk):
            deduped.append(chunk)
    return deduped


def chunk_medical_documents(
    docs: list[MedicalDocument],
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        for index, chunk in enumerate(_sliding_window_chunks(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "source": doc.source,
                        "category": doc.category,
                        "chunk_index": index,
                    },
                )
            )
    return chunks
