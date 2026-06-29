from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Iterable

import requests


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class HuggingFaceEmbeddingMatcher:
    """Optional semantic matcher. Disabled unless USE_EMBEDDINGS=true and HF_API_TOKEN is set."""

    def __init__(self) -> None:
        self.enabled = os.getenv("USE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}
        self.token = os.getenv("HF_API_TOKEN")
        self.model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.timeout = 12

    def available(self) -> bool:
        return bool(self.enabled and self.token)

    @lru_cache(maxsize=512)
    def embed(self, text: str) -> tuple[float, ...] | None:
        if not self.available():
            return None
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.post(url, headers=headers, json={"inputs": text}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return None
        if not data:
            return None
        if isinstance(data[0], list):
            vector = data[0]
        else:
            vector = data
        return tuple(float(value) for value in vector)

    def best_match(self, phrase: str, candidates: Iterable[str], threshold: float = 0.72) -> tuple[str, float] | None:
        phrase_vector = self.embed(phrase)
        if phrase_vector is None:
            return None
        best_name = None
        best_score = 0.0
        for candidate in candidates:
            candidate_vector = self.embed(candidate)
            if candidate_vector is None:
                continue
            score = cosine_similarity(list(phrase_vector), list(candidate_vector))
            if score > best_score:
                best_name = candidate
                best_score = score
        if best_name and best_score >= threshold:
            return best_name, best_score
        return None

