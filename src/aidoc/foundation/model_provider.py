from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from aidoc.embedding.local_embedding import LocalHashEmbeddingModel
from aidoc.models import DocumentLabel, Keyphrase, RiskLabel
from aidoc.nlp.patterns import DOMAIN_KEYWORDS, RISK_PATTERNS, tokenize_words


class FoundationModelProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def summarize(self, text: str, max_sentences: int = 2) -> str:
        raise NotImplementedError

    @abstractmethod
    def classify_document(self, text: str) -> DocumentLabel:
        raise NotImplementedError

    @abstractmethod
    def classify_risk(self, text: str) -> RiskLabel:
        raise NotImplementedError

    @abstractmethod
    def extract_keyphrases(self, text: str, limit: int = 8) -> list[Keyphrase]:
        raise NotImplementedError


class LocalFoundationModelProvider(FoundationModelProvider):
    """Deterministic local provider that mirrors foundation-model tasks.

    Production implementations can delegate these methods to Hugging Face,
    OpenAI, Bedrock, or a GPU-backed local model without changing the pipeline.
    """

    def __init__(self) -> None:
        self.embedding_model = LocalHashEmbeddingModel()

    def embed(self, text: str) -> list[float]:
        return self.embedding_model.embed(text)

    def summarize(self, text: str, max_sentences: int = 2) -> str:
        sentences = [sentence.strip() for sentence in text.replace("\n", " ").split(".") if sentence.strip()]
        selected = sentences[:max_sentences]
        return ". ".join(selected) + ("." if selected else "")

    def classify_document(self, text: str) -> DocumentLabel:
        lowered = text.lower()
        best_label = "general_compliance"
        best_score = 0
        for label, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > best_score:
                best_label = label
                best_score = score
        confidence = min(0.95, 0.45 + best_score * 0.1)
        return DocumentLabel(
            label=best_label,
            confidence=confidence,
            rationale=f"Matched {best_score} domain indicators for {best_label}.",
        )

    def classify_risk(self, text: str) -> RiskLabel:
        lowered = text.lower()
        matches = []
        for severity, patterns in RISK_PATTERNS.items():
            for pattern in patterns:
                if pattern in lowered:
                    matches.append((severity, pattern))

        if not matches:
            return RiskLabel(
                label="no_explicit_risk",
                severity="low",
                confidence=0.4,
                rationale="No configured risk indicators were detected.",
            )

        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        severity, pattern = max(matches, key=lambda item: severity_order[item[0]])
        return RiskLabel(
            label=f"{severity}_compliance_risk",
            severity=severity,
            confidence=min(0.95, 0.55 + 0.08 * len(matches)),
            rationale=f"Detected risk indicator: {pattern}.",
        )

    def extract_keyphrases(self, text: str, limit: int = 8) -> list[Keyphrase]:
        words = [word for word in tokenize_words(text) if len(word) > 4]
        counts = Counter(words)
        return [
            Keyphrase(text=phrase, score=count / max(1, len(words)))
            for phrase, count in counts.most_common(limit)
        ]


class ExternalFoundationModelProvider(FoundationModelProvider):
    """Interface placeholder for real 2026 model backends.

    Intended adapters:
    OpenAI embeddings and responses
    Amazon Bedrock Titan embeddings and Claude models
    Hugging Face sentence-transformers and cross-encoders
    GPU backed local LLMs served through vLLM or Ollama
    """

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Configure an external embedding backend.")

    def summarize(self, text: str, max_sentences: int = 2) -> str:
        raise NotImplementedError("Configure an external summarization backend.")

    def classify_document(self, text: str) -> DocumentLabel:
        raise NotImplementedError("Configure an external classification backend.")

    def classify_risk(self, text: str) -> RiskLabel:
        raise NotImplementedError("Configure an external risk classifier backend.")

    def extract_keyphrases(self, text: str, limit: int = 8) -> list[Keyphrase]:
        raise NotImplementedError("Configure an external keyphrase backend.")

