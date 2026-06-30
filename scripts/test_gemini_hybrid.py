from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.transformer.pipeline import transform_paths
from backend.transformer.section_classifier import CANONICAL_SECTION_LABELS, classify_line


OUTPUT_PATH = ROOT / "outputs" / "gemini_hybrid_test_result.json"

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
]

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["section", "content", "ambiguous"]},
                    "canonical_section": {
                        "type": "string",
                        "enum": [
                            "education",
                            "experience",
                            "skills",
                            "projects",
                            "achievements",
                            "links",
                            "competitive_programming",
                            "certifications",
                            "extracurriculars",
                            "none",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["id", "kind", "canonical_section", "reason"],
            },
        }
    },
    "required": ["items"],
}

SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "normalized_skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string"},
                    "aliases_seen": {"type": "array", "items": {"type": "string"}},
                    "source": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["canonical_name", "aliases_seen", "source", "confidence"],
            },
        },
        "merge_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["normalized_skills", "merge_warnings"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
        "incorrect": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["passed", "missing", "incorrect", "strengths", "improvements"],
}

SECTION_SYSTEM_PROMPT = """You are a deterministic section-vs-content classifier for candidate profiles.
You receive short resume lines with nearby following lines.
Return only JSON matching the provided schema.

Decision rules:
- Classify the line itself, not the content it belongs to.
- kind=section only when the line is a heading/label introducing a block.
- kind=content when the line is a skill, URL, contact field, degree detail, bullet, or inline field.
- canonical_section is the section heading represented by the line if kind=section, otherwise "none".
- Skills such as React, ReAct Agent, Machine Learning, RAG, Python, and Project Management are content, not section headings.
- Recognize multilingual section headings, including Spanish, French, German, Hindi transliterations, and mixed English headings.
- Use only these canonical sections: education, experience, skills, projects, achievements, links, competitive_programming, certifications, extracurriculars.
- Be conservative. If unsure whether a line itself is a heading, use ambiguous with canonical_section="none"."""

SKILL_SYSTEM_PROMPT = """You are a deterministic skill alias normalizer for candidate profiles.
Return only JSON matching the provided schema.

Rules:
- Merge aliases that clearly mean the same technology or concept.
- Keep different concepts separate, especially React frontend vs ReAct Agents.
- Do not invent skills.
- Normalize common variants: Golang -> Go, ReactJS/React.js -> React, NodeJS -> Node.js, Mongo DB -> MongoDB.
- Preserve multilingual mentions only when they clearly map to a skill in the input."""

REVIEW_SYSTEM_PROMPT = """You are a strict evaluator for a candidate data transformer demo.
Return only JSON matching the schema.
Use the expected facts as the source of truth.
Mark passed=true only if all required facts are present and no listed fact is materially wrong.
Do not penalize extra low-risk fields if required facts are present."""

HYBRID_SECTION_CASES = [
    {"line": "Formacion Academica", "next_lines": ["IIIT Jabalpur, B.Tech Computer Science; CGPA 8.5/10"], "expected_kind": "section", "expected_section": "education"},
    {"line": "Experience Professionnelle", "next_lines": ["MindTickle - SDE Applied AI Intern, Jan 2026 - Present"], "expected_kind": "section", "expected_section": "experience"},
    {"line": "Technische Fahigkeiten", "next_lines": ["Python, ReactJS, Flask, MongoDB"], "expected_kind": "section", "expected_section": "skills"},
    {"line": "Proyectos Aplicados", "next_lines": ["CodeForces Future Rating Predictor"], "expected_kind": "section", "expected_section": "projects"},
    {"line": "Reconnaissance et prix", "next_lines": ["Amazon ML Challenge 2024: 391st place"], "expected_kind": "section", "expected_section": "achievements"},
    {"line": "React", "next_lines": ["ReactJS, React.js, React, TailwindCSS"], "expected_kind": "content", "expected_section": None},
    {"line": "ReAct Agent", "next_lines": ["LangGraph ReAct Agent and RAG"], "expected_kind": "content", "expected_section": None},
    {"line": "GitHub: github.com/example-user", "next_lines": [], "expected_kind": "content", "expected_section": None},
    {"line": "Languages: Python, Golang, JS", "next_lines": [], "expected_kind": "content", "expected_section": None},
    {"line": "Project Management", "next_lines": ["Coordinated releases"], "expected_kind": "content", "expected_section": None},
]

SKILL_INPUT = [
    {"source": "Technical Strengths", "value": "Python, py, Golang, Go, ReactJS, React.js, React, ReAct Agent, ReACT agents"},
    {"source": "Tools and Platforms", "value": "NodeJS, Node.js, Mongo DB, MongoDB, Kubernetes, k8s"},
    {"source": "AI/ML", "value": "RAG, retrieval augmented generation, embeddings, NER, Machine Learning"},
]

MULTILINGUAL_DEMO_TEXT = """
Samhit Demo Candidate Email: demo@example.com GitHub: github.com/example-user
Formacion Academica
IIIT Jabalpur, Bachelor of Technology - Computer Science and Engineering; CGPA: 8.5/10 November 2022 - May 2026
Experience Professionnelle
MindTickle (SDE Applied AI Intern) Pune, India
January 2026 - Present
Built services using Golang, Kafka, Redis, gRPC and LangGraph ReAct Agent.
Technische Fahigkeiten
Python, ReactJS, Flask, Mongo DB, Docker, Kubernetes, RAG.
Proyectos Aplicados
CodeForces Future Rating Predictor June 2025
Tech Stack: ReactJS, TailwindCSS, Flask, MongoDB.
Reconnaissance et prix
Amazon ML Challenge 2024: Secured 391st place.
""".strip()

DEMO_EXPECTED = {
    "full_name": "Samhit Demo Candidate",
    "email": "demo@example.com",
    "education": ["IIIT Jabalpur", "Computer Science", "8.5/10"],
    "experience": ["MindTickle", "SDE Applied AI Intern"],
    "project": "CodeForces Future Rating Predictor",
    "skills": ["Python", "React", "Flask", "MongoDB", "Docker", "Kubernetes", "RAG", "LangGraph", "ReAct Agents"],
    "achievement": "Amazon ML Challenge 2024",
}


class GeminiRoundRobin:
    def __init__(self, keys: list[str], model: str) -> None:
        self.keys = keys
        self.model = model
        self.index = 0
        self.events: list[dict[str, Any]] = []

    def next_key(self) -> tuple[int, str]:
        if not self.keys:
            raise RuntimeError("No Gemini keys configured")
        position = self.index + 1
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return position, key

    def generate_json(self, system_prompt: str, payload: dict[str, Any], schema: dict[str, Any], label: str) -> dict[str, Any]:
        last_error = None
        for _attempt in range(len(self.keys)):
            key_position, key = self.next_key()
            start = time.perf_counter()
            try:
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
                        "generationConfig": {
                            "temperature": 0,
                            "topP": 1,
                            "topK": 1,
                            "maxOutputTokens": 4096,
                            "responseMimeType": "application/json",
                            "responseSchema": schema,
                        },
                    },
                    timeout=60,
                )
                elapsed = round(time.perf_counter() - start, 2)
                self.events.append({"label": label, "key_position": key_position, "status": response.status_code, "seconds": elapsed})
                if response.status_code in {429, 403, 503}:
                    last_error = f"Gemini status {response.status_code}"
                    continue
                response.raise_for_status()
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except Exception as exc:
                elapsed = round(time.perf_counter() - start, 2)
                self.events.append({"label": label, "key_position": key_position, "error": str(exc), "seconds": elapsed})
                last_error = str(exc)
        raise RuntimeError(last_error or "Gemini request failed")


def configured_gemini_keys() -> list[str]:
    values = []
    values.extend(os.getenv(f"gem{index}") or os.getenv(f"GEM{index}") or "" for index in range(1, 6))
    values.extend(os.getenv(f"GEMINI_KEY_{index}") or "" for index in range(1, 6))
    values.append(os.getenv("GEMINI_KEYS") or "")
    keys: list[str] = []
    seen = set()
    for raw in values:
        for key in re.split(r"[\s,]+", raw.strip()):
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def model_candidates() -> list[str]:
    requested = os.getenv("GEMINI_MODEL")
    if requested:
        return [requested]
    return GEMINI_MODELS


def line_accuracy(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    failures = []
    passed = 0
    for row in rows:
        expected_section = row["expected_section"] or "none"
        actual_section = row.get("actual_section") or "none"
        ok = row["expected_kind"] == row.get("actual_kind") and expected_section == actual_section
        if ok:
            passed += 1
        else:
            failures.append(row)
    return passed, failures


def deterministic_section_rows() -> list[dict[str, Any]]:
    rows = []
    for index, case in enumerate(HYBRID_SECTION_CASES, start=1):
        result = classify_line(case["line"], case["next_lines"])
        rows.append(
            {
                "id": index,
                "line": case["line"],
                "expected_kind": case["expected_kind"],
                "expected_section": case["expected_section"],
                "actual_kind": result.kind,
                "actual_section": result.canonical_section,
                "section_score": result.section_score,
                "content_score": result.content_score,
            }
        )
    return rows


def gemini_section_rows(client: GeminiRoundRobin) -> list[dict[str, Any]]:
    payload = {
        "items": [
            {"id": index, "line": case["line"], "next_lines": case["next_lines"]}
            for index, case in enumerate(HYBRID_SECTION_CASES, start=1)
        ]
    }
    response = client.generate_json(SECTION_SYSTEM_PROMPT, payload, SECTION_SCHEMA, "section-classification")
    items = {item["id"]: item for item in response.get("items", [])}
    rows = []
    for index, case in enumerate(HYBRID_SECTION_CASES, start=1):
        item = items.get(index, {})
        actual_section = item.get("canonical_section")
        rows.append(
            {
                "id": index,
                "line": case["line"],
                "expected_kind": case["expected_kind"],
                "expected_section": case["expected_section"],
                "actual_kind": item.get("kind"),
                "actual_section": None if actual_section == "none" else actual_section,
                "reason": item.get("reason"),
            }
        )
    return rows


def gemini_skill_merge(client: GeminiRoundRobin) -> dict[str, Any]:
    return client.generate_json(SKILL_SYSTEM_PROMPT, {"skill_sources": SKILL_INPUT}, SKILL_SCHEMA, "skill-merge")


def gemini_canonicalize_section_headings(client: GeminiRoundRobin, text: str) -> tuple[str, list[dict[str, Any]]]:
    lines = text.splitlines()
    non_empty = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    payload = {
        "items": [
            {
                "id": item_index,
                "line": line,
                "next_lines": [candidate for _line_index, candidate in non_empty[item_index : item_index + 3]],
            }
            for item_index, (_line_index, line) in enumerate(non_empty, start=1)
        ]
    }
    response = client.generate_json(SECTION_SYSTEM_PROMPT, payload, SECTION_SCHEMA, "demo-section-canonicalization")
    items = {item["id"]: item for item in response.get("items", [])}
    rewritten = list(lines)
    rows = []
    for item_index, (line_index, original) in enumerate(non_empty, start=1):
        item = items.get(item_index, {})
        canonical = item.get("canonical_section")
        if item.get("kind") == "section" and canonical and canonical != "none":
            label = CANONICAL_SECTION_LABELS.get(canonical, canonical.replace("_", " ").title())
            rewritten[line_index] = label
        rows.append(
            {
                "id": item_index,
                "original": original,
                "rewritten": rewritten[line_index].strip(),
                "kind": item.get("kind"),
                "canonical_section": None if canonical == "none" else canonical,
                "reason": item.get("reason"),
            }
        )
    return "\n".join(rewritten), rows


def run_demo_transform(text: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "multilingual_demo.txt"
        path.write_text(text, encoding="utf-8")
        return transform_paths([path], default_region="IN", use_llm=False)


def review_demo(client: GeminiRoundRobin, transform_result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "expected": DEMO_EXPECTED,
        "canonical_profile": transform_result["default_profile"],
    }
    return client.generate_json(REVIEW_SYSTEM_PROMPT, payload, REVIEW_SCHEMA, "demo-review")


def required_demo_checks(profile: dict[str, Any]) -> dict[str, Any]:
    skills = {skill["name"] for skill in profile.get("skills", [])}
    return {
        "full_name": profile.get("full_name") == "Samhit Demo Candidate",
        "email": "demo@example.com" in profile.get("emails", []),
        "education": bool(profile.get("education")) and profile["education"][0].get("cgpa") == "8.5/10",
        "experience": any(item.get("company") == "MindTickle" for item in profile.get("experience", [])),
        "project": any(item.get("title") == "CodeForces Future Rating Predictor" for item in profile.get("projects", [])),
        "achievement": any(item.get("title") == "Amazon ML Challenge 2024" for item in profile.get("achievements", [])),
        "skills": sorted(set(DEMO_EXPECTED["skills"]) - skills),
    }


def run_with_model(model: str, keys: list[str]) -> dict[str, Any]:
    client = GeminiRoundRobin(keys, model)
    deterministic_rows = deterministic_section_rows()
    deterministic_passed, deterministic_failures = line_accuracy(deterministic_rows)
    gemini_rows = gemini_section_rows(client)
    gemini_passed, gemini_failures = line_accuracy(gemini_rows)
    skill_payload = gemini_skill_merge(client)
    deterministic_transform = run_demo_transform(MULTILINGUAL_DEMO_TEXT)
    canonicalized_text, canonicalization_rows = gemini_canonicalize_section_headings(client, MULTILINGUAL_DEMO_TEXT)
    hybrid_transform = run_demo_transform(canonicalized_text)
    review_payload = review_demo(client, hybrid_transform)
    deterministic_profile = deterministic_transform["default_profile"]
    hybrid_profile = hybrid_transform["default_profile"]
    return {
        "model": model,
        "key_count": len(keys),
        "events": client.events,
        "deterministic_sections": {
            "passed": deterministic_passed,
            "total": len(deterministic_rows),
            "failures": deterministic_failures,
            "rows": deterministic_rows,
        },
        "gemini_sections": {
            "passed": gemini_passed,
            "total": len(gemini_rows),
            "failures": gemini_failures,
            "rows": gemini_rows,
        },
        "gemini_skill_merge": skill_payload,
        "demo_transform_deterministic_only": {
            "checks": required_demo_checks(deterministic_profile),
            "profile": deterministic_profile,
            "validation_errors": deterministic_transform.get("validation_errors"),
            "extraction_errors": deterministic_transform.get("extraction_errors"),
        },
        "demo_section_canonicalization": {
            "rewritten_text": canonicalized_text,
            "rows": canonicalization_rows,
        },
        "demo_transform_hybrid": {
            "checks": required_demo_checks(hybrid_profile),
            "profile": hybrid_profile,
            "validation_errors": hybrid_transform.get("validation_errors"),
            "extraction_errors": hybrid_transform.get("extraction_errors"),
        },
        "gemini_demo_review": review_payload,
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    keys = configured_gemini_keys()
    if not keys:
        print("No Gemini keys configured. Set gem1..gem5 or GEMINI_KEY_1..GEMINI_KEY_5.")
        return 2

    errors = []
    for model in model_candidates():
        print(f"Testing Gemini model: {model} with {len(keys)} keys")
        try:
            result = run_with_model(model, keys)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                "sections:",
                f"deterministic {result['deterministic_sections']['passed']}/{result['deterministic_sections']['total']}",
                f"gemini {result['gemini_sections']['passed']}/{result['gemini_sections']['total']}",
            )
            print("deterministic demo checks:", json.dumps(result["demo_transform_deterministic_only"]["checks"], ensure_ascii=False))
            print("hybrid demo checks:", json.dumps(result["demo_transform_hybrid"]["checks"], ensure_ascii=False))
            print("gemini review passed:", result["gemini_demo_review"].get("passed"))
            print(f"Wrote {OUTPUT_PATH}")
            return 0
        except Exception as exc:
            print(f"  failed: {exc}")
            errors.append({"model": model, "error": str(exc)})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    print(f"All Gemini model attempts failed. Wrote {OUTPUT_PATH}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
