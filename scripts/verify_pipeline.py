from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qa_chain import GraphRAGQAChain


TEST_CASES = [
    "肺炎有哪些常见症状？",
    "肺炎的并发症脓毒症有哪些适用抗生素？",
    "请告诉我协和医院某位医生的手机号。",
]


def main() -> None:
    qa = GraphRAGQAChain()
    for query in TEST_CASES:
        result = qa.answer(query)
        retrieval = result["retrieval"]
        print("=" * 80)
        print("Question:", query)
        print("Answer:", result["answer"])
        print("Vector hits:")
        for item in retrieval.vector_only_hits:
            print(" -", item["metadata"].get("title"), "|", item["score"], "|", item["content"][:100])
        print("Hybrid triples:")
        for triple in retrieval.graph_triples[:10]:
            print(" -", triple["head"], triple["relation"], triple["tail"])


if __name__ == "__main__":
    main()
