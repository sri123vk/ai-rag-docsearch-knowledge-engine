from __future__ import annotations

import math
from collections import Counter, defaultdict

from aidoc.embedding.local_embedding import LocalHashEmbeddingModel, cosine, tokenize
from aidoc.models import Chunk, RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        embedding_model: LocalHashEmbeddingModel,
        lexical_weight: float = 0.55,
        vector_weight: float = 0.45,
    ) -> None:
        self.embedding_model = embedding_model
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.chunks: list[Chunk] = []
        self.chunk_vectors: dict[str, list[float]] = {}
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.document_frequency: Counter[str] = Counter()
        self.average_length = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        lengths: list[int] = []
        self.term_frequencies.clear()
        self.document_frequency.clear()
        self.chunk_vectors.clear()

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            frequencies = Counter(tokens)
            self.term_frequencies[chunk.chunk_id] = frequencies
            for term in frequencies:
                self.document_frequency[term] += 1
            lengths.append(len(tokens))
            self.chunk_vectors[chunk.chunk_id] = self.embedding_model.embed(chunk.text)

        self.average_length = sum(lengths) / len(lengths) if lengths else 0.0

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        query_vector = self.embedding_model.embed(query)
        scored: list[RetrievedChunk] = []

        for chunk in self.chunks:
            lexical_score = self._bm25(query_tokens, chunk)
            vector_score = cosine(query_vector, self.chunk_vectors[chunk.chunk_id])
            final_score = self.lexical_weight * lexical_score + self.vector_weight * vector_score
            if final_score > 0:
                scored.append(
                    RetrievedChunk(
                        chunk=chunk,
                        lexical_score=lexical_score,
                        vector_score=vector_score,
                        final_score=final_score,
                    )
                )

        return sorted(scored, key=lambda hit: hit.final_score, reverse=True)[:limit]

    def _bm25(self, query_tokens: list[str], chunk: Chunk) -> float:
        frequencies = self.term_frequencies[chunk.chunk_id]
        chunk_length = sum(frequencies.values())
        if not query_tokens or chunk_length == 0 or self.average_length == 0:
            return 0.0

        k1 = 1.5
        b = 0.75
        total = 0.0
        corpus_size = len(self.chunks)
        for term in query_tokens:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            df = self.document_frequency.get(term, 0)
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * chunk_length / self.average_length)
            total += idf * (tf * (k1 + 1)) / denominator
        return total

