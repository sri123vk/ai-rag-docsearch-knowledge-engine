from __future__ import annotations

import hashlib
import math
import re


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


class LocalHashEmbeddingModel:
    """Deterministic local embedding model for demos and tests.

    This is not a substitute for a transformer embedding model. It gives the
    repository a reproducible vector retrieval path without external services.
    The production version can replace this class with OpenAI, Bedrock, or a
    local GPU backed sentence transformer.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return normalize(vector)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

