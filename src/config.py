from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


PROVIDER_DEFAULTS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-ai/DeepSeek-V3",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-flash",
    },
    "tongyi": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data" / "medical_texts"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    graph_store_path: Path = Path(os.getenv("GRAPH_STORE_PATH", str(PROJECT_ROOT / "data" / "graph_store.json")))
    docs_dir: Path = PROJECT_ROOT / "docs"
    provider: str = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()
    api_key: str = os.getenv("LLM_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").strip()
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-zh-v1.5",
    ).strip()
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "medical_graph_rag").strip()
    top_k_text: int = int(os.getenv("TOP_K_TEXT", "4"))
    max_graph_hops: int = int(os.getenv("MAX_GRAPH_HOPS", "2"))

    @property
    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["zhipu"])["model"]

    @property
    def resolved_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        return PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["zhipu"])["base_url"]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
