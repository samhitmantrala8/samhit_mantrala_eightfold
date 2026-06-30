from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
from typing import Any

import requests

from backend.transformer.gemini_hybrid import configured_gemini_keys, next_gemini_key
from backend.transformer.normalizers.skills import canonicalize_skill, normalize_token


logger = logging.getLogger(__name__)

TASK_TYPE = "candidate_profile_agent"
DEFAULT_SCORE_THRESHOLD = 8.0
DEFAULT_MAX_LOOPS = 3
MIN_SCORE_IMPROVEMENT = 0.3

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "mode": {"type": "string", "enum": ["deterministic", "react"]},
                    "target_fields": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "integer"},
                },
                "required": ["name", "purpose", "mode", "target_fields", "priority"],
            },
        }
    },
    "required": ["tasks"],
}

REACT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale_summary": {"type": "string"},
        "confidence": {"type": "number"},
        "proposed_output": {
            "type": "object",
            "properties": {
                "profile_summary": {"type": "string"},
                "skills_add": {"type": "array", "items": {"type": "string"}},
                "skills_remove": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": ["profile_summary", "skills_add", "skills_remove", "notes"],
        },
    },
    "required": ["rationale_summary", "confidence", "proposed_output"],
}

REACT_EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "passed": {"type": "boolean"},
        "use_output": {"type": "boolean"},
        "verdict": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "problem": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["field", "severity", "problem", "evidence"],
            },
        },
        "improvement_hints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "action": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["field", "action", "reason"],
            },
        },
    },
    "required": ["score", "passed", "use_output", "verdict", "issues", "improvement_hints"],
}

PLANNER_SYSTEM_PROMPT = """You decompose candidate-profile transformation QA into small executable tasks.
Return only JSON matching the schema.

Rules:
- Split the main work into deterministic tasks and ReACT tasks.
- deterministic tasks are checks that local code can answer: schema, coverage counts, duplicate skills, validation errors, provenance.
- react tasks are ambiguity/refinement tasks that need language judgment: summary quality, missing evidence-backed skills, semantic section review, hallucination review.
- Keep 4-6 tasks total.
- Do not modify the candidate profile in this planner.
- Do not reveal or request API keys."""

REACT_TASK_PROMPT_TEMPLATE = """You are one bounded ReACT-style agent inside a candidate data transformer.
Task name: {task_name}
Task purpose: {task_purpose}
Target fields: {target_fields}

Good examples from memory are provided below. They are examples that previously scored >= 8/10.
Use them as style and decision guidance, but do not copy unsupported facts.

GOOD_EXAMPLES_JSON:
{good_examples_json}

Operational rules:
- Use a ReACT discipline internally: inspect the task input, choose a safe action, produce a candidate output, then wait for evaluator feedback in the next loop.
- Return only JSON matching the provided schema.
- Do not expose hidden chain-of-thought. Put only a concise rationale_summary with observable reasons and evidence references.
- You may propose only profile_summary and evidence-backed skills_add/skills_remove.
- Do not invent companies, dates, degrees, projects, achievements, links, emails, phone numbers, or locations.
- Any skill you add must appear in the source excerpt or already be supported by canonical profile evidence.
- If unsure, leave arrays empty and explain uncertainty in notes.
- The deterministic extractor remains the source of truth."""

REACT_EVALUATOR_PROMPT_TEMPLATE = """You are the evaluator tool for one bounded ReACT agent task.
Task name: {task_name}
Task purpose: {task_purpose}
Target fields: {target_fields}

Score the candidate output from 1 to 10.
Set passed=true and use_output=true only when score >= 8 and the output is supported by the source/canonical profile.
Set use_output=false for unsupported, hallucinated, overly broad, duplicate, or schema-unsafe output.

Criteria:
- factuality: every proposed change is supported by source evidence.
- coverage: the task objective is addressed.
- schema safety: output fits allowed fields and types.
- deduplication: output does not add duplicate skills or repeated claims.
- confidence: uncertainty is represented conservatively.

Return only JSON matching the schema. Do not reveal or request API keys."""


def gemini_model() -> str:
    return os.getenv("GEMINI_AGENT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "task"


def call_gemini_json(
    task_name: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    response_schema: dict[str, Any],
    max_output_tokens: int = 4096,
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    keys = configured_gemini_keys()
    if not keys:
        logger.info("gemini_json skipped task=%s reason=no_keys", task_name)
        return None, [f"{task_name}: Gemini keys not configured; used deterministic fallback"], []

    model = gemini_model()
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    request_body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0,
            "topP": 1,
            "topK": 1,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }

    for _attempt in range(len(keys)):
        key_position, key = next_gemini_key(keys)
        start = time.perf_counter()
        logger.info("gemini_json call start task=%s model=%s key_index=%s", task_name, model, key_position)
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=request_body,
                timeout=60,
            )
            elapsed = round(time.perf_counter() - start, 2)
            event = {
                "task": task_name,
                "model": model,
                "key_index": key_position,
                "status": response.status_code,
                "seconds": elapsed,
            }
            events.append(event)
            logger.info("gemini_json response task=%s key_index=%s status=%s seconds=%s", task_name, key_position, response.status_code, elapsed)
            if response.status_code in {403, 429, 503}:
                errors.append(f"{task_name}: Gemini returned status {response.status_code}")
                logger.warning("gemini_json retryable_status task=%s key_index=%s status=%s", task_name, key_position, response.status_code)
                continue
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = parse_json_text(content)
            logger.info("gemini_json parsed task=%s keys=%s", task_name, list(parsed.keys()))
            return parsed, errors, events
        except Exception as exc:
            elapsed = round(time.perf_counter() - start, 2)
            events.append({"task": task_name, "model": model, "key_index": key_position, "error": str(exc), "seconds": elapsed})
            errors.append(f"{task_name}: Gemini call failed: {exc}")
            logger.exception("gemini_json failed task=%s key_index=%s seconds=%s", task_name, key_position, elapsed)

    logger.warning("gemini_json exhausted_keys task=%s attempts=%s errors=%s", task_name, len(keys), len(errors))
    return None, errors, events


def clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 1.0
    return round(max(1.0, min(10.0, number)), 2)


def safe_list(value: Any, limit: int = 20) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": profile.get("full_name"),
        "emails": safe_list(profile.get("emails"), 3),
        "phones": safe_list(profile.get("phones"), 3),
        "links": profile.get("links") or {},
        "headline": profile.get("headline"),
        "education": safe_list(profile.get("education"), 6),
        "experience": safe_list(profile.get("experience"), 8),
        "projects": safe_list(profile.get("projects"), 8),
        "achievements": safe_list(profile.get("achievements"), 8),
        "skills": safe_list(profile.get("skills"), 80),
        "resume_sections": profile.get("resume_sections") or {},
        "semantic_mappings": safe_list(profile.get("semantic_mappings"), 40),
        "profile_summary": profile.get("profile_summary"),
        "overall_confidence": profile.get("overall_confidence"),
        "extraction_errors": safe_list(profile.get("extraction_errors"), 20),
    }


def source_excerpt(source_texts: list[str], char_limit: int = 18000) -> str:
    joined = "\n\n--- SOURCE BREAK ---\n\n".join(text.strip() for text in source_texts if text.strip())
    if len(joined) <= char_limit:
        return joined
    head = joined[: char_limit // 2]
    tail = joined[-char_limit // 2 :]
    return f"{head}\n\n--- SOURCE TRUNCATED ---\n\n{tail}"


def trace_input_excerpt(profile: dict[str, Any], source_texts: list[str]) -> dict[str, Any]:
    digest = hashlib.sha256("\n".join(source_texts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return {
        "source_digest": digest,
        "source_count": len(source_texts),
        "source_excerpt": source_excerpt(source_texts, 1200),
        "profile_counts": {
            "education": len(profile.get("education") or []),
            "experience": len(profile.get("experience") or []),
            "projects": len(profile.get("projects") or []),
            "achievements": len(profile.get("achievements") or []),
            "skills": len(profile.get("skills") or []),
        },
    }


def output_preview(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": profile.get("full_name"),
        "headline": profile.get("headline"),
        "education_count": len(profile.get("education") or []),
        "experience_count": len(profile.get("experience") or []),
        "project_count": len(profile.get("projects") or []),
        "achievement_count": len(profile.get("achievements") or []),
        "top_skills": [skill.get("name") for skill in (profile.get("skills") or [])[:16]],
        "summary": profile.get("profile_summary"),
    }


def compact_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for item in examples[:6]:
        compacted.append(
            {
                "task_type": item.get("task_type"),
                "score": item.get("score"),
                "input": item.get("example_input") or item.get("input_excerpt"),
                "output": item.get("example_output") or item.get("output_preview"),
                "evaluation": item.get("evaluator_json"),
            }
        )
    return compacted


def deterministic_plan() -> list[dict[str, Any]]:
    return [
        {
            "name": "schema_validation",
            "purpose": "Check canonical schema and validator errors with local code.",
            "mode": "deterministic",
            "target_fields": ["validation_errors", "canonical_profile"],
            "priority": 1,
        },
        {
            "name": "coverage_check",
            "purpose": "Check whether high-signal profile sections were extracted.",
            "mode": "deterministic",
            "target_fields": ["education", "experience", "projects", "achievements", "skills", "links"],
            "priority": 2,
        },
        {
            "name": "dedupe_confidence_check",
            "purpose": "Check duplicate skills and confidence/provenance coverage.",
            "mode": "deterministic",
            "target_fields": ["skills", "provenance", "overall_confidence"],
            "priority": 3,
        },
        {
            "name": "summary_and_skill_refinement",
            "purpose": "Use language judgment to improve summary and identify evidence-backed missing skills.",
            "mode": "react",
            "target_fields": ["profile_summary", "skills"],
            "priority": 4,
        },
        {
            "name": "semantic_mapping_review",
            "purpose": "Review ambiguous headings and LLM semantic mapping quality.",
            "mode": "react",
            "target_fields": ["semantic_mappings", "resume_sections"],
            "priority": 5,
        },
    ]


def normalize_task(raw: dict[str, Any], fallback_priority: int) -> dict[str, Any]:
    name = str(raw.get("name") or f"task_{fallback_priority}").strip()
    purpose = str(raw.get("purpose") or "Check candidate profile quality.").strip()
    mode = raw.get("mode") if raw.get("mode") in {"deterministic", "react"} else "react"
    target_fields = [str(item) for item in safe_list(raw.get("target_fields"), 8) if str(item).strip()]
    if not target_fields:
        target_fields = ["canonical_profile"]
    try:
        priority = int(raw.get("priority", fallback_priority))
    except (TypeError, ValueError):
        priority = fallback_priority
    return {"name": name, "purpose": purpose, "mode": mode, "target_fields": target_fields, "priority": priority}


def ensure_base_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {slug(task["name"]) for task in tasks}
    base = deterministic_plan()
    for task in base[:3]:
        if slug(task["name"]) not in names:
            tasks.append(task)
    tasks.sort(key=lambda item: item.get("priority", 99))
    return tasks[:6]


def plan_tasks(profile: dict[str, Any], source_texts: list[str], memory_examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], str]:
    logger.info("agent plan_tasks start source_texts=%s memory_examples=%s", len(source_texts), len(memory_examples))
    planner_prompt = PLANNER_SYSTEM_PROMPT
    payload = {
        "canonical_profile": compact_profile(profile),
        "source_excerpt": source_excerpt(source_texts, 6000),
        "good_memory_examples": compact_examples(memory_examples),
    }
    result, errors, events = call_gemini_json("agent_task_decomposition", planner_prompt, payload, PLANNER_SCHEMA, 2048)
    raw_tasks = result.get("tasks") if isinstance(result, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        logger.info("agent plan_tasks fallback deterministic errors=%s", len(errors))
        return deterministic_plan(), errors, events, planner_prompt
    tasks = [normalize_task(task, index) for index, task in enumerate(raw_tasks[:6], start=1) if isinstance(task, dict)]
    tasks = ensure_base_tasks(tasks)
    logger.info("agent plan_tasks done tasks=%s errors=%s", len(tasks), len(errors))
    return tasks, errors, events, planner_prompt


def deterministic_task_output(task: dict[str, Any], profile: dict[str, Any], validation_errors: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    name = slug(task["name"])
    profile_counts = {
        "education": len(profile.get("education") or []),
        "experience": len(profile.get("experience") or []),
        "projects": len(profile.get("projects") or []),
        "achievements": len(profile.get("achievements") or []),
        "skills": len(profile.get("skills") or []),
        "links": sum(1 for value in (profile.get("links") or {}).values() if value),
    }
    if name == "schema_validation":
        score = 10.0 if not validation_errors else max(1.0, 10.0 - 1.5 * len(validation_errors))
        issues = [
            {"field": "schema", "severity": "high", "problem": error, "evidence": "Local schema validator"}
            for error in validation_errors[:8]
        ]
        output = {"checks": {"validation_errors": validation_errors}, "summary": "Schema validation completed locally."}
    elif name == "coverage_check":
        required = ["education", "experience", "projects", "skills"]
        present = sum(1 for field in required if profile_counts[field] > 0)
        score = round(10.0 * present / len(required), 2)
        issues = [
            {"field": field, "severity": "medium", "problem": f"No {field} items extracted.", "evidence": "Local coverage count"}
            for field in required
            if profile_counts[field] == 0
        ]
        output = {"checks": profile_counts, "summary": "Coverage check completed locally."}
    elif name == "dedupe_confidence_check":
        skill_names = [skill.get("name") for skill in profile.get("skills", []) if skill.get("name")]
        duplicate_count = len(skill_names) - len(set(skill_names))
        provenance_count = len(profile.get("provenance") or [])
        confidence = float(profile.get("overall_confidence") or 0)
        score = max(1.0, 10.0 - duplicate_count - (0 if provenance_count else 2) - (0 if confidence >= 0.5 else 1))
        issues = []
        if duplicate_count:
            issues.append({"field": "skills", "severity": "medium", "problem": f"{duplicate_count} duplicate skill labels found.", "evidence": "Local duplicate count"})
        if not provenance_count:
            issues.append({"field": "provenance", "severity": "medium", "problem": "No provenance entries present.", "evidence": "Local provenance count"})
        output = {"checks": {"duplicate_skills": duplicate_count, "provenance_count": provenance_count, "overall_confidence": confidence}, "summary": "Dedupe and confidence check completed locally."}
    else:
        score = 8.0
        issues = []
        output = {"checks": profile_counts, "summary": "Generic deterministic check completed locally."}

    evaluation = {
        "score": clamp_score(score),
        "passed": score >= DEFAULT_SCORE_THRESHOLD,
        "use_output": True,
        "verdict": output["summary"],
        "issues": issues,
        "improvement_hints": [],
    }
    return output, evaluation


def build_react_system_prompt(task: dict[str, Any], memory_examples: list[dict[str, Any]]) -> str:
    return REACT_TASK_PROMPT_TEMPLATE.format(
        task_name=task["name"],
        task_purpose=task["purpose"],
        target_fields=", ".join(task["target_fields"]),
        good_examples_json=json.dumps(compact_examples(memory_examples), ensure_ascii=False, indent=2),
    )


def build_react_evaluator_prompt(task: dict[str, Any]) -> str:
    return REACT_EVALUATOR_PROMPT_TEMPLATE.format(
        task_name=task["name"],
        task_purpose=task["purpose"],
        target_fields=", ".join(task["target_fields"]),
    )


def source_supports_skill(raw_skill: str, source_blob: str) -> bool:
    raw_norm = normalize_token(raw_skill)
    if not raw_norm or len(raw_norm) < 2:
        return False
    source_norm = f" {normalize_token(source_blob)} "
    if re.search(rf"(?<![a-z0-9#+]){re.escape(raw_norm)}(?![a-z0-9#+])", source_norm):
        return True
    canonical = canonicalize_skill(raw_skill)
    if not canonical:
        return False
    canonical_norm = normalize_token(canonical[0])
    return bool(re.search(rf"(?<![a-z0-9#+]){re.escape(canonical_norm)}(?![a-z0-9#+])", source_norm))


def apply_refinement(profile: dict[str, Any], proposed_output: dict[str, Any], source_texts: list[str], confidence: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info(
        "agent apply_refinement start confidence=%s proposed_skills_add=%s proposed_skills_remove=%s",
        confidence,
        len(safe_list(proposed_output.get("skills_add"), 24)),
        len(safe_list(proposed_output.get("skills_remove"), 24)),
    )
    updated = copy.deepcopy(profile)
    applied: list[dict[str, Any]] = []
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    source_blob = "\n".join(source_texts)

    summary = str(proposed_output.get("profile_summary") or "").strip()
    if confidence >= 0.65 and 40 <= len(summary) <= 1200:
        updated["profile_summary"] = re.sub(r"\s+", " ", summary)
        updated.setdefault("provenance", []).append(
            {
                "field": "profile_summary",
                "source": "gemini-react-agent:evaluator-approved",
                "method": "react-agent-summary-refinement",
                "confidence": round(min(0.78, confidence), 3),
            }
        )
        applied.append({"field": "profile_summary", "action": "replace", "confidence": round(confidence, 3)})

    existing = {normalize_token(skill.get("name", "")): skill for skill in updated.get("skills", [])}
    for raw_skill in safe_list(proposed_output.get("skills_add"), 24):
        canonical = canonicalize_skill(str(raw_skill))
        if not canonical:
            continue
        skill_name, canonical_confidence = canonical
        skill_key = normalize_token(skill_name)
        if skill_key in existing:
            continue
        if not source_supports_skill(str(raw_skill), source_blob) and not source_supports_skill(skill_name, source_blob):
            continue
        new_skill = {
            "name": skill_name,
            "confidence": round(min(0.72, max(0.55, confidence * canonical_confidence)), 3),
            "sources": ["gemini-react-agent:evaluator-approved"],
        }
        updated.setdefault("skills", []).append(new_skill)
        existing[skill_key] = new_skill
        updated.setdefault("provenance", []).append(
            {
                "field": "skills",
                "source": "gemini-react-agent:evaluator-approved",
                "method": "react-agent-skill-add",
                "confidence": new_skill["confidence"],
                "evidence": str(raw_skill)[:120],
            }
        )
        applied.append({"field": "skills", "action": "add", "value": skill_name, "confidence": new_skill["confidence"]})

    updated["skills"] = sorted(updated.get("skills", []), key=lambda item: (-float(item.get("confidence") or 0), item.get("name") or ""))
    logger.info("agent apply_refinement done applied_changes=%s", len(applied))
    return updated, applied


def run_deterministic_task(task: dict[str, Any], profile: dict[str, Any], validation_errors: list[str], threshold: float) -> dict[str, Any]:
    output, evaluation = deterministic_task_output(task, profile, validation_errors)
    score = clamp_score(evaluation.get("score"))
    passed = score >= threshold
    return {
        "task_name": task["name"],
        "purpose": task["purpose"],
        "mode": "deterministic",
        "target_fields": task["target_fields"],
        "system_prompt": "Deterministic local task. No LLM system prompt was used.",
        "evaluator_prompt": "Deterministic local evaluator. No LLM evaluator prompt was used.",
        "loops": 1,
        "final_score": score,
        "passed": passed,
        "accepted": passed,
        "stopping_reason": "deterministic task completed",
        "final_output": output,
        "iterations": [
            {
                "loop": 1,
                "action": "local_check",
                "observation": output,
                "rationale_summary": "Local deterministic rules evaluated this task without an LLM call.",
                "evaluation": evaluation,
                "score": score,
                "passed": passed,
                "request_events": [],
            }
        ],
        "request_events": [],
    }


def react_user_payload(
    task: dict[str, Any],
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str],
    previous_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "task": task,
        "canonical_profile": compact_profile(profile),
        "source_excerpt": source_excerpt(source_texts),
        "validation_errors": validation_errors,
    }
    if previous_evaluation:
        payload["previous_evaluation"] = previous_evaluation
    return payload


def run_react_task(
    task: dict[str, Any],
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str],
    memory_examples: list[dict[str, Any]],
    max_loops: int,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    system_prompt = build_react_system_prompt(task, memory_examples)
    evaluator_prompt = build_react_evaluator_prompt(task)
    errors: list[str] = []
    request_events: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    best_score = 0.0
    previous_score = 0.0
    stagnant_loops = 0
    accepted_output: dict[str, Any] | None = None
    accepted_evaluation: dict[str, Any] | None = None
    stopping_reason = "max loops reached"
    previous_evaluation: dict[str, Any] | None = None

    for loop_index in range(1, max_loops + 1):
        task_slug = slug(task["name"])
        logger.info("react task loop start task=%s loop=%s", task["name"], loop_index)
        candidate, candidate_errors, candidate_events = call_gemini_json(
            f"react_{task_slug}_generate_loop_{loop_index}",
            system_prompt,
            react_user_payload(task, profile, source_texts, validation_errors, previous_evaluation),
            REACT_OUTPUT_SCHEMA,
            4096,
        )
        errors.extend(candidate_errors)
        request_events.extend(candidate_events)

        iteration: dict[str, Any] = {
            "loop": loop_index,
            "action": "generate_candidate",
            "request_events": list(candidate_events),
            "candidate_output": candidate,
            "rationale_summary": candidate.get("rationale_summary") if isinstance(candidate, dict) else "No candidate output was produced.",
        }
        if not isinstance(candidate, dict):
            iteration["observation"] = "Gemini did not return a valid candidate JSON object."
            iteration["score"] = 1.0
            iteration["passed"] = False
            iterations.append(iteration)
            stopping_reason = "candidate generation failed"
            break

        proposed_output = candidate.get("proposed_output") if isinstance(candidate.get("proposed_output"), dict) else {}
        evaluator_payload = {
            "task": task,
            "canonical_profile": compact_profile(profile),
            "source_excerpt": source_excerpt(source_texts),
            "candidate_output": proposed_output,
            "candidate_rationale_summary": candidate.get("rationale_summary"),
            "candidate_confidence": candidate.get("confidence"),
            "validation_errors": validation_errors,
        }
        evaluation, eval_errors, eval_events = call_gemini_json(
            f"react_{task_slug}_evaluate_loop_{loop_index}",
            evaluator_prompt,
            evaluator_payload,
            REACT_EVALUATOR_SCHEMA,
            4096,
        )
        errors.extend(eval_errors)
        request_events.extend(eval_events)
        iteration["request_events"].extend(eval_events)

        if not isinstance(evaluation, dict):
            evaluation = {
                "score": 1.0,
                "passed": False,
                "use_output": False,
                "verdict": "Evaluator did not return a valid JSON object.",
                "issues": [],
                "improvement_hints": [],
            }
        evaluation["score"] = clamp_score(evaluation.get("score"))
        evaluation["passed"] = bool(evaluation.get("passed")) and evaluation["score"] >= threshold
        evaluation["use_output"] = bool(evaluation.get("use_output")) and evaluation["passed"]
        previous_evaluation = evaluation
        score = evaluation["score"]
        best_score = max(best_score, score)
        iteration["evaluation"] = evaluation
        iteration["score"] = score
        iteration["passed"] = evaluation["passed"]
        iteration["observation"] = evaluation.get("verdict")
        iterations.append(iteration)
        logger.info("react task evaluated task=%s loop=%s score=%s passed=%s", task["name"], loop_index, score, evaluation["passed"])

        if evaluation["passed"] and evaluation["use_output"]:
            accepted_output = candidate
            accepted_evaluation = evaluation
            stopping_reason = "score threshold reached"
            break
        if loop_index >= max_loops:
            break
        if score - previous_score < MIN_SCORE_IMPROVEMENT and loop_index > 1:
            stagnant_loops += 1
        else:
            stagnant_loops = 0
        previous_score = score
        if stagnant_loops >= 2:
            stopping_reason = "score stopped improving"
            break

    final_score = clamp_score(accepted_evaluation.get("score") if accepted_evaluation else best_score or 1.0)
    task_trace = {
        "task_name": task["name"],
        "purpose": task["purpose"],
        "mode": "react",
        "target_fields": task["target_fields"],
        "system_prompt": system_prompt,
        "evaluator_prompt": evaluator_prompt,
        "loops": len(iterations),
        "final_score": final_score,
        "passed": bool(accepted_output and final_score >= threshold),
        "accepted": bool(accepted_output and final_score >= threshold),
        "stopping_reason": stopping_reason,
        "final_output": accepted_output.get("proposed_output") if isinstance(accepted_output, dict) else None,
        "discarded": not bool(accepted_output and final_score >= threshold),
        "iterations": iterations,
        "request_events": request_events,
    }
    good_examples = []
    if accepted_output and accepted_evaluation and final_score >= threshold:
        good_examples.append(
            {
                "task_type": f"{TASK_TYPE}:{slug(task['name'])}",
                "score": final_score,
                "input": {
                    "task": task,
                    "canonical_profile": compact_profile(profile),
                    "source_excerpt": source_excerpt(source_texts, 1600),
                },
                "output": accepted_output.get("proposed_output"),
                "evaluation": accepted_evaluation,
            }
        )
    return task_trace, accepted_output, good_examples, errors


def env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def run_agentic_llmops(
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str] | None = None,
    memory_examples: list[dict[str, Any]] | None = None,
    max_loops: int | None = None,
    score_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    validation_errors = validation_errors or []
    memory_examples = memory_examples or []
    max_loops = min(max_loops or env_int("AGENT_MAX_LOOPS", DEFAULT_MAX_LOOPS), 5)
    score_threshold = score_threshold or env_float("AGENT_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)
    working = copy.deepcopy(profile)
    errors: list[str] = []
    request_events: list[dict[str, Any]] = []
    good_examples_to_store: list[dict[str, Any]] = []
    llm_available = bool(configured_gemini_keys())
    logger.info(
        "agent run start llm_available=%s max_loops=%s score_threshold=%s good_memory_examples=%s validation_errors=%s",
        llm_available,
        max_loops,
        score_threshold,
        len(memory_examples),
        len(validation_errors),
    )

    if llm_available:
        tasks, task_errors, task_events, planner_prompt = plan_tasks(working, source_texts, memory_examples)
        errors.extend(task_errors)
        request_events.extend(task_events)
    else:
        tasks = deterministic_plan()
        planner_prompt = "Deterministic fallback planner. Gemini keys were not configured."

    trace: dict[str, Any] = {
        "enabled": True,
        "task_type": TASK_TYPE,
        "mode": "per-task-react-agents" if llm_available else "deterministic-fallback-evaluator",
        "model": gemini_model(),
        "score_threshold": score_threshold,
        "max_loops": max_loops,
        "memory_examples_used": len(memory_examples),
        "planner_prompt": planner_prompt,
        "tasks": tasks,
        "task_traces": [],
        "iterations": [],
        "request_events": request_events,
        "input_excerpt": trace_input_excerpt(profile, source_texts),
    }

    for task in tasks:
        normalized = normalize_task(task, len(trace["task_traces"]) + 1)
        if normalized["mode"] == "deterministic" or not llm_available:
            task_trace = run_deterministic_task(normalized, working, validation_errors, score_threshold)
            trace["task_traces"].append(task_trace)
            continue

        task_trace, accepted_output, good_examples, task_errors = run_react_task(
            normalized,
            working,
            source_texts,
            validation_errors,
            memory_examples,
            max_loops,
            score_threshold,
        )
        errors.extend(task_errors)
        trace["task_traces"].append(task_trace)
        request_events.extend(task_trace.get("request_events", []))
        good_examples_to_store.extend(good_examples)

        if accepted_output and task_trace.get("accepted"):
            proposed_output = accepted_output.get("proposed_output") if isinstance(accepted_output.get("proposed_output"), dict) else {}
            confidence = float(accepted_output.get("confidence") or 0.0)
            working, applied = apply_refinement(working, proposed_output, source_texts, confidence)
            task_trace["applied_changes"] = applied
        else:
            task_trace["applied_changes"] = []

    task_scores = [float(task.get("final_score") or 0.0) for task in trace["task_traces"]]
    final_score = round(sum(task_scores) / len(task_scores), 2) if task_scores else 0.0
    passed_tasks = sum(1 for task in trace["task_traces"] if task.get("passed"))
    stopping_reason = "completed per-task execution"
    final_evaluation = {
        "score": final_score,
        "passed": bool(task_scores and all(score >= score_threshold for score in task_scores)),
        "verdict": f"{passed_tasks}/{len(task_scores)} decomposed tasks passed.",
        "issues": [
            {
                "field": task.get("task_name"),
                "severity": "medium",
                "problem": task.get("stopping_reason"),
                "evidence": f"score={task.get('final_score')}",
            }
            for task in trace["task_traces"]
            if not task.get("passed")
        ],
    }

    working["llmops"] = {
        key: value
        for key, value in trace.items()
        if key not in {"input_excerpt"}
    }
    working["llmops"]["final_score"] = final_score
    working["llmops"]["stopping_reason"] = stopping_reason
    working["llmops"]["final_evaluation"] = final_evaluation

    trace["final_score"] = final_score
    trace["stopping_reason"] = stopping_reason
    trace["final_evaluation"] = final_evaluation
    trace["output_preview"] = output_preview(working)
    trace["request_events"] = request_events
    trace["good_examples"] = good_examples_to_store
    logger.info("agent run done final_score=%s tasks=%s good_examples=%s errors=%s", final_score, len(trace["task_traces"]), len(good_examples_to_store), len(errors))
    return working, trace, errors
