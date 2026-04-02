from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.config import Settings


@dataclass
class ChatResponse:
    content: str


class OpenAIChatModel:
    def __init__(self, settings: Settings, temperature: float = 0.0) -> None:
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.resolved_base_url,
        )
        self.model = settings.resolved_model
        self.temperature = temperature

    def invoke(self, prompt: str) -> ChatResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        return ChatResponse(content=content)


def build_chat_model(settings: Settings, temperature: float = 0.0) -> Any | None:
    if not settings.llm_enabled:
        return None
    return OpenAIChatModel(settings=settings, temperature=temperature)


class LocalHashEmbeddings:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            index = self._stable_hash(f"{token}|index") % self.dimension
            sign = 1.0 if self._stable_hash(f"{token}|sign") % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower().strip()
        if not normalized:
            return []

        word_tokens = re.findall(r"[a-z0-9_]+", normalized)
        compact = re.sub(r"\s+", "", normalized)
        char_tokens = [char for char in compact if not char.isspace()]
        bigrams = [compact[index : index + 2] for index in range(max(len(compact) - 1, 0))]
        return [*word_tokens, *char_tokens, *bigrams]

    @staticmethod
    def _stable_hash(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")


def _torch_version_is_compatible() -> bool:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return False

    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        return False

    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) >= (2, 4)


def build_embedding_model(settings: Settings) -> Any:
    if not _torch_version_is_compatible():
        warnings.warn(
            "Falling back to LocalHashEmbeddings because PyTorch < 2.4 is installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return LocalHashEmbeddings()

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        embeddings.embed_query("医疗知识图谱")
        return embeddings
    except Exception as exc:
        warnings.warn(
            f"Falling back to LocalHashEmbeddings because HuggingFace embeddings are unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return LocalHashEmbeddings()
