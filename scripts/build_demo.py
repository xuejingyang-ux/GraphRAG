from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg_builder import MedicalKnowledgeGraphBuilder
from src.retriever import HybridRetriever


def main() -> None:
    builder = MedicalKnowledgeGraphBuilder()
    retriever = HybridRetriever()
    try:
        graph_result = builder.build_graph(reset=True)
        vector_result = retriever.rebuild_vector_index(reset=True)
        print("Graph build:", graph_result)
        print("Vector build:", vector_result)
    finally:
        builder.close()
        retriever.close()


if __name__ == "__main__":
    main()
