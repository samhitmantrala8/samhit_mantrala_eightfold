from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

from backend.transformer.normalizers.embeddings import HuggingFaceEmbeddingMatcher


CANONICAL_SKILLS = [
    "C++",
    "C",
    "C#",
    "Go",
    "Java",
    "Python",
    "HTML",
    "CSS",
    "Shell",
    "Jupyter Notebook",
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
    "Kubernetes",
    "Helm",
    "AWS",
    "AWS ECR",
    "AWS OpenSearch",
    "Google Cloud",
    "Google Cloud Run",
    "Google Colab",
    "Git",
    "GitHub",
    "GitLab",
    "Kafka",
    "Redis",
    "gRPC",
    "Protocol Buffers",
    "MLOps",
    "Vector Databases",
    "Semantic Caching",
    "Image Generation",
    "Maxim AI",
    "Claude Skills",
    "Claude Code",
    "Cursor",
    "Machine Learning",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Named Entity Recognition",
    "PyTorch",
    "BERT",
    "Embeddings",
    "RAG",
    "Cohere Rerank",
    "LangGraph",
    "Bifrost",
    "ReAct Agents",
    "LLM-as-a-Judge",
    "Data Engineering",
    "ETL",
    "Pandas",
    "OpenRouter",
    "PageRank",
    "Community Detection",
    "Graph Neural Networks",
    "Speculative Decoding",
    "Quantization",
    "KV Caching",
    "FFmpeg",
    "Polynomial Regression",
    "Codeforces API",
    "Ping Monitoring",
    "Traffic Analytics",
    "MongoDB TTL Indexes",
    "Pagination",
    "Responsive Design",
    "Dark Mode",
    "Light Mode",
    "Anonymous Platforms",
    "Content Moderation",
    "Chart.js",
    "Netlify",
    "Render",
    "Express.js",
    "Supervised Fine-Tuning",
]

ALIASES = {
    "c++": "C++",
    "cpp": "C++",
    "c": "C",
    "c#": "C#",
    "csharp": "C#",
    "java": "Java",
    "golang": "Go",
    "go lang": "Go",
    "py": "Python",
    "python": "Python",
    "html": "HTML",
    "css": "CSS",
    "shell": "Shell",
    "bash": "Shell",
    "jupyter notebook": "Jupyter Notebook",
    "jupyter": "Jupyter Notebook",
    "notebook": "Jupyter Notebook",
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
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "helm": "Helm",
    "helm charts": "Helm",
    "aws": "AWS",
    "aws ecr": "AWS ECR",
    "ecr": "AWS ECR",
    "opensearch": "AWS OpenSearch",
    "aws opensearch": "AWS OpenSearch",
    "gcp": "Google Cloud",
    "google cloud": "Google Cloud",
    "google cloud run": "Google Cloud Run",
    "cloud run": "Google Cloud Run",
    "google colab": "Google Colab",
    "colab": "Google Colab",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "kafka": "Kafka",
    "redis": "Redis",
    "grpc": "gRPC",
    "protobuf": "Protocol Buffers",
    "protocol buffers": "Protocol Buffers",
    "mlops": "MLOps",
    "ml ops": "MLOps",
    "ml-ops": "MLOps",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    "semantic caching": "Semantic Caching",
    "imagegeneration": "Image Generation",
    "image generation": "Image Generation",
    "maxim ai": "Maxim AI",
    "claude skills": "Claude Skills",
    "claude code": "Claude Code",
    "cursor": "Cursor",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "pytorch": "PyTorch",
    "bert": "BERT",
    "ner": "Named Entity Recognition",
    "named entity recognition": "Named Entity Recognition",
    "embedding": "Embeddings",
    "embeddings": "Embeddings",
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "cohere rerank": "Cohere Rerank",
    "cohere rerank 3.5": "Cohere Rerank",
    "rerank": "Cohere Rerank",
    "langgraph": "LangGraph",
    "bifrost": "Bifrost",
    "react agent": "ReAct Agents",
    "react agents": "ReAct Agents",
    "llm as a judge": "LLM-as-a-Judge",
    "llm-as-a-judge": "LLM-as-a-Judge",
    "etl": "ETL",
    "data engineering": "Data Engineering",
    "pandas": "Pandas",
    "openrouter": "OpenRouter",
    "pagerank": "PageRank",
    "community detection": "Community Detection",
    "gnn": "Graph Neural Networks",
    "graph neural network": "Graph Neural Networks",
    "graph neural networks": "Graph Neural Networks",
    "speculative decoding": "Speculative Decoding",
    "quantization": "Quantization",
    "kv caching": "KV Caching",
    "ffmpeg": "FFmpeg",
    "polynomial regression": "Polynomial Regression",
    "codeforces api": "Codeforces API",
    "ping monitor": "Ping Monitoring",
    "ping monitoring": "Ping Monitoring",
    "website traffic": "Traffic Analytics",
    "traffic": "Traffic Analytics",
    "ttl": "MongoDB TTL Indexes",
    "ttl indexing": "MongoDB TTL Indexes",
    "time to live": "MongoDB TTL Indexes",
    "pagination": "Pagination",
    "responsive": "Responsive Design",
    "responsive design": "Responsive Design",
    "dark mode": "Dark Mode",
    "dark light mode": "Dark Mode",
    "light mode": "Light Mode",
    "anonymous": "Anonymous Platforms",
    "anonymously": "Anonymous Platforms",
    "hate": "Content Moderation",
    "abusive text": "Content Moderation",
    "content moderation": "Content Moderation",
    "chartjs": "Chart.js",
    "chart.js": "Chart.js",
    "netlify": "Netlify",
    "render": "Render",
    "expressjs": "Express.js",
    "express js": "Express.js",
    "express.js": "Express.js",
    "supervised fine tuning": "Supervised Fine-Tuning",
    "supervised fine-tuning": "Supervised Fine-Tuning",
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
    for canonical in CANONICAL_SKILLS:
        if normalize_token(canonical) == cleaned:
            return canonical, 0.95
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
        if re.search(rf"(?<![a-z0-9#+]){re.escape(alias_norm)}(?![a-z0-9#+])", normalized):
            confidence = 0.82 if len(alias_norm) > 2 else 0.68
            previous = matches.get(canonical)
            if previous is None or confidence > previous[0]:
                matches[canonical] = (confidence, alias)
    return [(name, confidence, evidence) for name, (confidence, evidence) in matches.items()]
