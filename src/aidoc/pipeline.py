from __future__ import annotations

from pathlib import Path

from aidoc.chunking.semantic_chunker import SemanticChunker
from aidoc.embedding.local_embedding import LocalHashEmbeddingModel
from aidoc.io.local_loader import load_text_documents
from aidoc.kg.knowledge_graph import KnowledgeGraphExtractor
from aidoc.rag.answer_builder import CitationAnswerBuilder
from aidoc.retrieval.hybrid_retriever import HybridRetriever


class DocumentIntelligencePipeline:
    def __init__(self) -> None:
        self.chunker = SemanticChunker()
        self.embedding_model = LocalHashEmbeddingModel()
        self.retriever = HybridRetriever(self.embedding_model)
        self.graph_extractor = KnowledgeGraphExtractor()
        self.answer_builder = CitationAnswerBuilder()
        self.chunks = []
        self.triples = []

    def index_path(self, root: Path) -> dict[str, int]:
        documents = load_text_documents(root)
        self.chunks = [chunk for document in documents for chunk in self.chunker.chunk(document)]
        self.retriever.index(self.chunks)
        self.triples = self.graph_extractor.extract(self.chunks)
        return {
            "documents": len(documents),
            "chunks": len(self.chunks),
            "knowledge_graph_triples": len(self.triples),
        }

    def ask(self, question: str, limit: int = 5) -> dict[str, object]:
        evidence = self.retriever.search(question, limit=limit)
        return self.answer_builder.build(question, evidence)

