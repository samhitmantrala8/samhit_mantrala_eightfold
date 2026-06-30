from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.transformer.extractors.llm_extractor import configured_keys

OUTPUT_PATH = ROOT / "outputs" / "llm_merge_test_result.json"

COMBINED_SYSTEM_PROMPT = """You normalize candidate data.
Return exactly one valid JSON object with only these top-level keys:
canonical_sections, normalized_skills, merge_warnings.
Use only facts present in the input. Do not include task, schema, rules, markdown, or explanations.
Keep React, the frontend library, separate from ReAct Agents, the agentic AI pattern."""

SECTION_SYSTEM_PROMPT = """You normalize candidate resume section names.
Return exactly one valid JSON object with only this top-level key: canonical_sections.
Use only these canonical section keys: education, experience, skills, projects, achievements, links, online_coding_profile, other.
Merge duplicate or similar section headings. Use only facts present in the input."""

SKILL_SYSTEM_PROMPT = """You normalize candidate skills and aliases.
Return exactly one valid JSON object with only these top-level keys: normalized_skills, merge_warnings.
Each normalized skill must have: canonical_name, aliases_seen, source_sections, confidence, merge_reason.
Merge aliases like Golang/Go, Mongo DB/MongoDB, NodeJS/Node.js.
Keep React, the frontend library, separate from ReAct Agents, the agentic AI pattern.
Use only skills present in the input."""

DUMMY_CANDIDATE_PAGE = """
Samhit Test Candidate
Email: candidate@example.com | GitHub: github.com/example-user | LinkedIn: linkedin.com/in/example-user

Academics / Scholastic Record
- Indian Institute of Information Technology Jabalpur, Bachelor of Technology - CSE, CGPA 8.5/10, Nov 2022 - May 2026.
- Relevant courses: Data Structures, Artificial Intelligence, Database Management Systems, Object Oriented Programming.

Education Details
- IIIT Jabalpur, B.Tech Computer Science and Engineering. Same degree record as above, written by a recruiter in another CSV export.

Professional Background
- MindTickle, SDE Applied AI Intern, Pune, India, Jan 2026 - Present.
  Built async image generation RPC services using Golang, gRPC, Protocol Buffers, Kafka and Redis.
  Worked on semantic caching, LangGraph ReAct agents, RAG, AWS OpenSearch vector database, Cohere rerank and LLM-as-a-Judge monitoring.

Work Experience
- CREW, Machine Learning Intern, Remote Sydney team, Jun 2025 - Oct 2025.
  Implemented PageRank, community detection, graph neural networks, link prediction and text summarization optimizations.
  Used PyTorch, quantization, autocast, speculative decoding, KV caching, FFmpeg and Google Cloud Run.

Technical Strengths
- Languages: Python, py, Golang, Go, C++, JavaScript, JS, TypeScript, SQL.
- Frontend: ReactJS, React.js, React, TailwindCSS, ChartJS, responsive design, dark mode, light mode.
- Backend: Flask, FastAPI, NodeJS, ExpressJS, REST APIs, Mongo DB, MongoDB, MySQL.
- AI/ML: Machine Learning, artificial intelligence, NLP, NER, BERT, embeddings, TF-IDF, cosine similarity, RAG, retrieval augmented generation.
- Agentic AI: LangGraph, ReAct Agent, ReACT agents, LLM as a Judge, Claude Skills, OpenRouter.

Tools and Platforms
- Docker, Kubernetes, k8s, Helm Charts, AWS, AWS ECR, AWS OpenSearch, Google Cloud, Git, Github, GitLab.
- Kafka, Redis, gRPC, protobuf, Protocol Buffers, vector DB, semantic cache.

Projects / Applied Builds
- CodeForces Future Rating Predictor, June 2025.
  ReactJS, TailwindCSS, Flask, MongoDB, ChartJS, Codeforces API, polynomial regression, Netlify, Render.
- AnonGrievance, April 2024.
  ExpressJS, NodeJS, MongoDB TTL indexes, pagination, abusive text moderation, BERT fine tuning, dark and light themes.

Selected Project Work
- Same CodeForces project appears in another source as CF rating predictor with React, Flask, Mongo DB and Codeforces API.
- Same anonymous grievance platform appears as anonymous student platform with Node.js, Express.js and MongoDB.

Achievements and Recognition
- Amazon ML Challenge 2024: 391st out of 10000+ teams.
- Meta Hacker Cup 2025: Cleared Round 1.
- Codeforces Expert rating 1630, handle CinCout21.
- LeetCode rating 2043, handle clutchnuub21.

Online Coding Profile Metadata
- CF handle: CinCout21. LC handle: clutchnuub21. Kaggle profile: samhitmantrala.
""".strip()

SECTION_SOURCE_TEXT = """
Academics / Scholastic Record
- Indian Institute of Information Technology Jabalpur, Bachelor of Technology - CSE, CGPA 8.5/10, Nov 2022 - May 2026.
Education Details
- IIIT Jabalpur, B.Tech Computer Science and Engineering. Same degree record as above.
Professional Background
- MindTickle, SDE Applied AI Intern, Pune, India, Jan 2026 - Present.
Work Experience
- CREW, Machine Learning Intern, Remote Sydney team, Jun 2025 - Oct 2025.
Technical Strengths
- Languages, frontend, backend, AI/ML, agentic AI.
Tools and Platforms
- Docker, Kubernetes, cloud, Git, Kafka, Redis, gRPC.
Projects / Applied Builds
- CodeForces Future Rating Predictor. AnonGrievance.
Selected Project Work
- Same projects appear again under different names.
Achievements and Recognition
- Amazon ML Challenge, Meta Hacker Cup, Codeforces, LeetCode.
Online Coding Profile Metadata
- CF handle, LC handle, Kaggle profile.
""".strip()

SKILL_SOURCE_TEXT = """
Technical Strengths
- Languages: Python, py, Golang, Go, C++, JavaScript, JS, TypeScript, SQL.
- Frontend: ReactJS, React.js, React, TailwindCSS, ChartJS, responsive design, dark mode, light mode.
- Backend: Flask, FastAPI, NodeJS, ExpressJS, REST APIs, Mongo DB, MongoDB, MySQL.
- AI/ML: Machine Learning, artificial intelligence, NLP, NER, BERT, embeddings, TF-IDF, cosine similarity, RAG, retrieval augmented generation.
- Agentic AI: LangGraph, ReAct Agent, ReACT agents, LLM as a Judge, Claude Skills, OpenRouter.
Tools and Platforms
- Docker, Kubernetes, k8s, Helm Charts, AWS, AWS ECR, AWS OpenSearch, Google Cloud, Git, Github, GitLab.
- Kafka, Redis, gRPC, protobuf, Protocol Buffers, vector DB, semantic cache.
Projects / Applied Builds
- CodeForces Future Rating Predictor: ReactJS, TailwindCSS, Flask, MongoDB, ChartJS, Codeforces API, polynomial regression, Netlify, Render.
- AnonGrievance: ExpressJS, NodeJS, MongoDB TTL indexes, pagination, abusive text moderation, BERT fine tuning, dark and light themes.
""".strip()

EXPECTED_SECTIONS = {"education", "experience", "skills", "projects", "achievements", "links", "online_coding_profile"}
EXPECTED_SKILLS = {
    "Python",
    "Go",
    "React",
    "ReAct Agents",
    "RAG",
    "LangGraph",
    "MongoDB",
    "Node.js",
    "Express.js",
    "Docker",
    "Kubernetes",
    "AWS OpenSearch",
    "Codeforces API",
}


def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def output_path_for(model: str, mode: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model}_{mode}").strip("_")
    return ROOT / "outputs" / f"llm_merge_test_result_{safe}.json"


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def build_combined_prompt() -> dict[str, Any]:
    return {
        "required_output_shape": {
            "canonical_sections": {
                "education": ["short merged facts"],
                "experience": ["short merged facts"],
                "skills": ["short merged facts"],
                "projects": ["short merged facts"],
                "achievements": ["short merged facts"],
                "links": ["short merged facts"],
                "online_coding_profile": ["short merged facts"],
                "other": ["short merged facts"],
            },
            "normalized_skills": [
                {
                    "canonical_name": "Python",
                    "aliases_seen": ["Python", "py"],
                    "source_sections": ["Technical Strengths"],
                    "confidence": 0.95,
                    "merge_reason": "Exact alias/abbreviation match",
                }
            ],
            "merge_warnings": [],
        },
        "rules": " ".join([
            "Merge section names that clearly refer to the same canonical section.",
            "Append different skills under the skills section instead of overwriting them.",
            "Merge skill aliases, for example Golang and Go, Mongo DB and MongoDB, NodeJS and Node.js.",
            "Do not merge React, the frontend library, with ReAct Agent, the agentic AI pattern.",
            "Do not invent URLs, ratings, companies, or skills not present in the input.",
        ]),
        "candidate_text": DUMMY_CANDIDATE_PAGE,
    }


def build_section_prompt() -> dict[str, Any]:
    return {
        "output_shape": {
            "canonical_sections": {
                "education": ["short merged facts"],
                "experience": ["short merged facts"],
                "skills": ["short merged facts"],
                "projects": ["short merged facts"],
                "achievements": ["short merged facts"],
                "links": ["short merged facts"],
                "online_coding_profile": ["short merged facts"],
                "other": ["short merged facts"],
            }
        },
        "rules": [
            "Academics, Scholastic Record, and Education Details map to education.",
            "Professional Background and Work Experience map to experience.",
            "Technical Strengths and Tools and Platforms map to skills.",
            "Projects / Applied Builds and Selected Project Work map to projects.",
            "Achievements and Recognition maps to achievements.",
            "Online Coding Profile Metadata maps to online_coding_profile.",
        ],
        "candidate_text": SECTION_SOURCE_TEXT,
    }


def build_skill_prompt() -> dict[str, Any]:
    return {
        "output_shape": {
            "normalized_skills": [
                {
                    "canonical_name": "Go",
                    "aliases_seen": ["Golang", "Go"],
                    "source_sections": ["Technical Strengths"],
                    "confidence": 0.95,
                    "merge_reason": "same programming language alias",
                }
            ],
            "merge_warnings": [],
        },
        "must_include_if_present": sorted(EXPECTED_SKILLS),
        "do_not_merge_pairs": [["React", "ReAct Agents"]],
        "candidate_text": SKILL_SOURCE_TEXT,
    }


def call_openrouter(
    key: str,
    model: str,
    system_prompt: str,
    user_prompt: dict[str, Any],
    max_tokens: int = 2200,
) -> tuple[dict[str, Any] | None, str | None, float]:
    start = time.perf_counter()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5177",
            "X-Title": "Candidate Transformer Merge Test",
        },
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
        },
        timeout=45,
    )
    elapsed = time.perf_counter() - start
    if response.status_code in {402, 403, 429}:
        return None, f"provider returned status {response.status_code}", elapsed
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return extract_json(content), None, elapsed


def merge_decomposed_payloads(section_payload: dict[str, Any], skill_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_sections": section_payload.get("canonical_sections") or {},
        "normalized_skills": skill_payload.get("normalized_skills") or [],
        "merge_warnings": skill_payload.get("merge_warnings") or [],
    }


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    sections = payload.get("canonical_sections") or {}
    section_names = set(sections)
    skills = payload.get("normalized_skills") or []
    skill_names = {
        item.get("canonical_name")
        for item in skills
        if isinstance(item, dict) and isinstance(item.get("canonical_name"), str)
    }

    react = next((item for item in skills if item.get("canonical_name") == "React"), {})
    react_agents = next((item for item in skills if item.get("canonical_name") == "ReAct Agents"), {})
    react_aliases = set(react.get("aliases_seen") or [])
    agent_aliases = set(react_agents.get("aliases_seen") or [])

    checks = {
        "has_expected_sections": sorted(EXPECTED_SECTIONS - section_names),
        "has_expected_skills": sorted(EXPECTED_SKILLS - skill_names),
        "react_and_react_agent_separated": bool(react and react_agents and not react_aliases.intersection(agent_aliases)),
        "go_alias_merged": any(
            item.get("canonical_name") == "Go" and {"Golang", "Go"}.issubset(set(item.get("aliases_seen") or []))
            for item in skills
        ),
        "mongodb_alias_merged": any(
            item.get("canonical_name") == "MongoDB" and {"Mongo DB", "MongoDB"}.issubset(set(item.get("aliases_seen") or []))
            for item in skills
        ),
    }
    score = 0
    score += 1 if not checks["has_expected_sections"] else 0
    score += 1 if len(checks["has_expected_skills"]) <= 2 else 0
    score += 1 if checks["react_and_react_agent_separated"] else 0
    score += 1 if checks["go_alias_merged"] else 0
    score += 1 if checks["mongodb_alias_merged"] else 0
    return {"score": score, "max_score": 5, "checks": checks}


def run_combined(key: str, model: str) -> tuple[dict[str, Any] | None, str | None, float]:
    return call_openrouter(key, model, COMBINED_SYSTEM_PROMPT, build_combined_prompt(), max_tokens=3500)


def run_decomposed(key: str, model: str) -> tuple[dict[str, Any] | None, str | None, float]:
    sections, error, section_seconds = call_openrouter(key, model, SECTION_SYSTEM_PROMPT, build_section_prompt(), max_tokens=1400)
    if error or sections is None:
        return None, f"section merge failed: {error}", section_seconds
    skills, error, skill_seconds = call_openrouter(key, model, SKILL_SYSTEM_PROMPT, build_skill_prompt(), max_tokens=2600)
    if error or skills is None:
        return None, f"skill merge failed: {error}", section_seconds + skill_seconds
    return merge_decomposed_payloads(sections, skills), None, section_seconds + skill_seconds


def main() -> int:
    load_dotenv(ROOT / ".env")
    keys = configured_keys()
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    mode = os.getenv("LLM_MERGE_TEST_MODE", "decomposed").strip().lower()
    print(f"Configured OpenRouter keys: {len(keys)}")
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    if not keys:
        print("No OpenRouter keys configured.")
        return 2

    attempts = []
    for index, key in enumerate(keys, start=1):
        print(f"Trying key {index}: {mask_key(key)}")
        try:
            if mode == "combined":
                payload, error, elapsed = run_combined(key, model)
            else:
                payload, error, elapsed = run_decomposed(key, model)
        except Exception as exc:
            attempts.append({"key_index": index, "key": mask_key(key), "error": str(exc)})
            print(f"  failed: {exc}")
            continue
        if error:
            attempts.append({"key_index": index, "key": mask_key(key), "error": error, "seconds": round(elapsed, 2)})
            print(f"  skipped: {error} ({elapsed:.2f}s)")
            continue
        assert payload is not None
        evaluation = evaluate(payload)
        result = {
            "model": model,
            "mode": mode,
            "used_key_index": index,
            "used_key": mask_key(key),
            "seconds": round(elapsed, 2),
            "evaluation": evaluation,
            "payload": payload,
            "failed_attempts": attempts,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_output_path = output_path_for(model, mode)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        model_output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  success: score {evaluation['score']}/{evaluation['max_score']} ({elapsed:.2f}s)")
        print(f"Wrote result to {model_output_path}")
        return 0 if evaluation["score"] >= 4 else 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": model, "mode": mode, "failed_attempts": attempts}
    model_output_path = output_path_for(model, mode)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"All keys failed. Wrote attempts to {model_output_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
