from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

from backend.transformer.normalizers.embeddings import HuggingFaceEmbeddingMatcher


CANONICAL_SKILLS = [
    "Python",
    "Flask",
    "FastAPI",
    "JavaScript",
    "TypeScript",
    "React",
    "Tailwind CSS",
    "Node.js",
    "REST APIs",
    "GraphQL",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Docker",
    "AWS",
    "Git",
    "Machine Learning",
    "Natural Language Processing",
    "Named Entity Recognition",
    "Embeddings",
    "RAG",
    "LangGraph",
    "Data Engineering",
    "ETL",
    "Pandas",
    "OpenRouter",
]

ALIASES = {
    "py": "Python",
    "python": "Python",
    "flask": "Flask",
    "fast api": "FastAPI",
    "fastapi": "FastAPI",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "react": "React",
    "reactjs": "React",
    "react js": "React",
    "react.js": "React",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "tailwind css": "Tailwind CSS",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "rest": "REST APIs",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "graphql": "GraphQL",
    "sql": "SQL",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "docker": "Docker",
    "aws": "AWS",
    "git": "Git",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "ner": "Named Entity Recognition",
    "named entity recognition": "Named Entity Recognition",
    "embedding": "Embeddings",
    "embeddings": "Embeddings",
    "rag": "RAG",
    "langgraph": "LangGraph",
    "etl": "ETL",
    "data engineering": "Data Engineering",
    "pandas": "Pandas",
    "openrouter": "OpenRouter",
}


def normalize_token(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9#+.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=512)
def canonicalize_skill(raw: str) -> tuple[str, float] | None:
    cleaned = normalize_token(raw)
    if not cleaned:
        return None
    if cleaned in ALIASES:
        return ALIASES[cleaned], 0.95

    best_alias = None
    best_score = 0
    for alias in ALIASES:
        score = fuzz.ratio(cleaned, alias)
        if score > best_score:
            best_alias = alias
            best_score = score
    if best_alias and best_score >= 88:
        return ALIASES[best_alias], round(best_score / 100 * 0.86, 3)

    semantic = HuggingFaceEmbeddingMatcher().best_match(raw, CANONICAL_SKILLS)
    if semantic:
        name, score = semantic
        return name, round(min(score, 0.78), 3)
    return None


def extract_skills_from_text(text: str) -> list[tuple[str, float, str]]:
    normalized = f" {normalize_token(text)} "
    matches: dict[str, tuple[float, str]] = {}
    for alias, canonical in ALIASES.items():
        alias_norm = normalize_token(alias)
        if not alias_norm:
            continue
        if f" {alias_norm} " in normalized:
            confidence = 0.82 if len(alias_norm) > 2 else 0.68
            previous = matches.get(canonical)
            if previous is None or confidence > previous[0]:
                matches[canonical] = (confidence, alias)
    return [(name, confidence, evidence) for name, (confidence, evidence) in matches.items()]

