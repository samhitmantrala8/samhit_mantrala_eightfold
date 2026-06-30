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
from backend.transformer.section_classifier import classify_line


OUTPUT_PATH = ROOT / "outputs" / "section_classifier_test_result.json"

CASES = [
    {
        "line": "Academics / Scholastic Record",
        "next_lines": ["Indian Institute of Information Technology Jabalpur", "Bachelor of Technology - CSE; CGPA 8.5/10"],
        "expected_kind": "section",
        "expected_section": "education",
    },
    {
        "line": "Education Details",
        "next_lines": ["IIIT Jabalpur, B.Tech Computer Science and Engineering"],
        "expected_kind": "section",
        "expected_section": "education",
    },
    {
        "line": "Professional Background",
        "next_lines": ["MindTickle, SDE Applied AI Intern, Pune, India, Jan 2026 - Present."],
        "expected_kind": "section",
        "expected_section": "experience",
    },
    {
        "line": "Technical Strengths",
        "next_lines": ["Languages: Python, py, Golang, Go, C++, JavaScript, JS"],
        "expected_kind": "section",
        "expected_section": "skills",
    },
    {
        "line": "Tools and Platforms",
        "next_lines": ["Docker, Kubernetes, k8s, Helm Charts, AWS, GitHub"],
        "expected_kind": "section",
        "expected_section": "skills",
    },
    {
        "line": "Projects / Applied Builds",
        "next_lines": ["CodeForces Future Rating Predictor, June 2025."],
        "expected_kind": "section",
        "expected_section": "projects",
    },
    {
        "line": "Selected Project Work",
        "next_lines": ["Same CodeForces project appears in another source."],
        "expected_kind": "section",
        "expected_section": "projects",
    },
    {
        "line": "Achievements and Recognition",
        "next_lines": ["Amazon ML Challenge 2024: 391st out of 10000+ teams."],
        "expected_kind": "section",
        "expected_section": "achievements",
    },
    {
        "line": "Online Coding Profile Metadata",
        "next_lines": ["CF handle: CinCout21. LC handle: clutchnuub21."],
        "expected_kind": "section",
        "expected_section": "online_coding_profile",
    },
    {
        "line": "React",
        "next_lines": ["ReactJS, React.js, React, TailwindCSS, ChartJS."],
        "expected_kind": "content",
        "expected_section": None,
    },
    {
        "line": "ReAct Agent",
        "next_lines": ["Agentic AI: LangGraph, ReAct Agent, RAG."],
        "expected_kind": "content",
        "expected_section": None,
    },
    {
        "line": "Machine Learning",
        "next_lines": ["Implemented graph neural networks and link prediction."],
        "expected_kind": "content",
        "expected_section": None,
    },
    {
        "line": "Languages: Python, py, Golang, Go, C++",
        "next_lines": [],
        "expected_kind": "content",
        "expected_section": None,
    },
    {
        "line": "GitHub: github.com/example-user",
        "next_lines": [],
        "expected_kind": "content",
        "expected_section": None,
    },
    {
        "line": "Project Management",
        "next_lines": ["Coordinated release planning and stakeholder communication."],
        "expected_kind": "content",
        "expected_section": None,
    },
]

LLM_SYSTEM_PROMPT = """Classify resume lines.
Return exactly one JSON object with key "items".
For each input item, return: id, kind, canonical_section, reason.
kind must be one of: section, content, ambiguous.
canonical_section must be one of: education, experience, skills, projects, achievements, links, online_coding_profile, certifications, extracurriculars, null.
Do not classify skill names like React, Machine Learning, RAG, or ReAct Agent as section headings."""


def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def output_path_for(model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")
    return ROOT / "outputs" / f"section_classifier_test_result_{safe}.json"


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def deterministic_results() -> list[dict[str, Any]]:
    rows = []
    for index, case in enumerate(CASES, start=1):
        result = classify_line(case["line"], case["next_lines"])
        passed = result.kind == case["expected_kind"] and result.canonical_section == case["expected_section"]
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
                "passed": passed,
                "reasons": result.reasons,
            }
        )
    return rows


def call_llm(model: str, keys: list[str]) -> dict[str, Any]:
    payload = {
        "items": [
            {
                "id": index,
                "line": case["line"],
                "next_lines": case["next_lines"],
            }
            for index, case in enumerate(CASES, start=1)
        ]
    }
    attempts = []
    for index, key in enumerate(keys, start=1):
        start = time.perf_counter()
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5177",
                    "X-Title": "Candidate Transformer Section Test",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 1400,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                },
                timeout=45,
            )
            elapsed = round(time.perf_counter() - start, 2)
            if response.status_code in {402, 403, 429}:
                attempts.append({"key_index": index, "key": mask_key(key), "error": f"status {response.status_code}", "seconds": elapsed})
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = extract_json(content)
            return {"model": model, "used_key_index": index, "used_key": mask_key(key), "seconds": elapsed, "payload": parsed, "failed_attempts": attempts}
        except Exception as exc:
            attempts.append({"key_index": index, "key": mask_key(key), "error": str(exc), "seconds": round(time.perf_counter() - start, 2)})
    return {"model": model, "failed_attempts": attempts}


def evaluate_llm(llm_result: dict[str, Any]) -> dict[str, Any]:
    items = (llm_result.get("payload") or {}).get("items") or []
    by_id = {item.get("id"): item for item in items if isinstance(item, dict)}
    rows = []
    for index, case in enumerate(CASES, start=1):
        item = by_id.get(index, {})
        kind = item.get("kind")
        section = item.get("canonical_section")
        if section == "null":
            section = None
        rows.append(
            {
                "id": index,
                "line": case["line"],
                "expected_kind": case["expected_kind"],
                "expected_section": case["expected_section"],
                "actual_kind": kind,
                "actual_section": section,
                "passed": kind == case["expected_kind"] and section == case["expected_section"],
                "reason": item.get("reason"),
            }
        )
    return {"passed": sum(1 for row in rows if row["passed"]), "total": len(rows), "rows": rows}


def main() -> int:
    load_dotenv(ROOT / ".env")
    deterministic = deterministic_results()
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    keys = configured_keys()
    result: dict[str, Any] = {
        "deterministic": {
            "passed": sum(1 for row in deterministic if row["passed"]),
            "total": len(deterministic),
            "rows": deterministic,
        }
    }
    print(f"Deterministic: {result['deterministic']['passed']}/{result['deterministic']['total']}")

    if keys and os.getenv("RUN_LLM_SECTION_TEST", "false").lower() in {"1", "true", "yes"}:
        llm_result = call_llm(model, keys)
        llm_result["evaluation"] = evaluate_llm(llm_result) if llm_result.get("payload") else None
        result["llm"] = llm_result
        evaluation = llm_result.get("evaluation")
        if evaluation:
            print(f"LLM {model}: {evaluation['passed']}/{evaluation['total']} in {llm_result.get('seconds')}s")
        else:
            print(f"LLM {model}: failed")
    elif not keys:
        result["llm"] = {"skipped": "no OpenRouter keys configured"}
        print("LLM: skipped, no OpenRouter keys configured")
    else:
        result["llm"] = {"skipped": "RUN_LLM_SECTION_TEST is not enabled"}
        print("LLM: skipped, RUN_LLM_SECTION_TEST is not enabled")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "llm" in result and not result["llm"].get("skipped"):
        output_path_for(model).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0 if result["deterministic"]["passed"] == result["deterministic"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
