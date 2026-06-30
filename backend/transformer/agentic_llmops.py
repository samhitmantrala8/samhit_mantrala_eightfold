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
DEFAULT_SCORE_THRESHOLD = 8.5
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
                    "priority": {"type": "integer"},
                },
                "required": ["name", "purpose", "priority"],
            },
        }
    },
    "required": ["tasks"],
}

EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "verdict": {"type": "string"},
        "field_scores": {
            "type": "object",
            "properties": {
                "coverage": {"type": "number"},
                "factuality": {"type": "number"},
                "schema": {"type": "number"},
                "deduplication": {"type": "number"},
                "confidence": {"type": "number"},
            },
            "required": ["coverage", "factuality", "schema", "deduplication", "confidence"],
        },
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
    "required": ["score", "verdict", "field_scores", "issues", "improvement_hints"],
}

REFINER_SCHEMA = {
    "type": "object",
    "properties": {
        "profile_summary": {"type": "string"},
        "skills_add": {"type": "array", "items": {"type": "string"}},
        "skills_remove": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["profile_summary", "skills_add", "skills_remove", "confidence", "notes"],
}

PLANNER_SYSTEM_PROMPT = """You decompose candidate profile quality checks into small deterministic tasks.
Return only JSON matching the schema.

Rules:
- Keep the plan short and practical.
- Prefer tasks that can be evaluated from the canonical profile and source excerpts.
- Do not ask for browsing, training, or unbounded agent behavior.
- Include only task names and purposes; do not modify the profile."""

EVALUATOR_SYSTEM_PROMPT = """You are an LLM-as-a-judge evaluator for a candidate data transformation pipeline.
Return only JSON matching the schema.

Score from 1 to 10 using these criteria:
- Factuality: every extracted field must be supported by source evidence or provenance.
- Coverage: education, experience, projects, achievements, links, skills, and contact data should be present when source evidence exists.
- Schema: canonical JSON fields should be valid and internally consistent.
- Deduplication: duplicate skills/sections should be merged without losing distinct skills.
- Confidence: uncertain LLM/semantic mappings should have lower confidence and clear reasons.

Guardrails:
- Penalize hallucinated skills, companies, dates, links, and claims.
- Penalize missing high-signal sections when evidence appears in the source excerpt.
- Be strict but not impossible. A clean evidence-backed profile can score 9+.
- Do not reveal or request API keys."""

REFINER_SYSTEM_PROMPT = """You are a bounded profile refiner.
Return only JSON matching the schema.

Allowed output:
- A better profile_summary using only supported facts.
- skills_add for skills explicitly present in the source excerpts.
- skills_remove only for exact duplicates or clearly unsupported skill labels.
- notes explaining the intended refinement.

Do not invent companies, dates, degrees, projects, achievements, links, or contact details.
Do not rewrite the whole profile. The deterministic extractor remains the source of truth."""


def gemini_model() -> str:
    return os.getenv("GEMINI_AGENT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


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
            events.append(
                {
                    "task": task_name,
                    "model": model,
                    "key_index": key_position,
                    "status": response.status_code,
                    "seconds": elapsed,
                }
            )
            logger.info(
                "gemini_json call response task=%s model=%s key_index=%s status=%s seconds=%s",
                task_name,
                model,
                key_position,
                response.status_code,
                elapsed,
            )
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
            logger.exception("gemini_json call failed task=%s key_index=%s seconds=%s", task_name, key_position, elapsed)

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
        "semantic_mappings": safe_list(profile.get("semantic_mappings"), 30),
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
                "quality": item.get("quality"),
                "score": item.get("score"),
                "input_excerpt": item.get("input_excerpt"),
                "output_preview": item.get("output_preview"),
                "evaluator_json": item.get("evaluator_json"),
            }
        )
    return compacted


def deterministic_plan() -> list[dict[str, Any]]:
    return [
        {"name": "schema_and_required_fields", "purpose": "Check canonical shape and required field coverage.", "priority": 1},
        {"name": "evidence_and_hallucination", "purpose": "Compare extracted fields against source excerpts and provenance.", "priority": 2},
        {"name": "skill_section_deduplication", "purpose": "Check duplicated or mismapped sections and skills.", "priority": 3},
        {"name": "summary_quality", "purpose": "Ensure the summary covers evidence-backed education, experience, projects, achievements, and skills.", "priority": 4},
    ]


def deterministic_evaluation(profile: dict[str, Any], validation_errors: list[str]) -> dict[str, Any]:
    coverage_parts = [
        bool(profile.get("full_name")),
        bool(profile.get("emails") or profile.get("phones")),
        bool(profile.get("education")),
        bool(profile.get("experience")),
        bool(profile.get("projects")),
        bool(profile.get("skills")),
    ]
    coverage = 10 * sum(coverage_parts) / len(coverage_parts)
    schema = 10.0 if not validation_errors else max(1.0, 10.0 - 1.5 * len(validation_errors))
    skill_names = [skill.get("name") for skill in profile.get("skills", []) if skill.get("name")]
    duplicate_count = len(skill_names) - len(set(skill_names))
    dedupe = max(1.0, 10.0 - duplicate_count)
    confidence = 10.0 * float(profile.get("overall_confidence") or 0.5)
    factuality = 8.0 if profile.get("provenance") else 6.0
    score = round((coverage * 0.3) + (factuality * 0.25) + (schema * 0.2) + (dedupe * 0.15) + (confidence * 0.1), 2)
    issues = []
    if not profile.get("education"):
        issues.append({"field": "education", "severity": "medium", "problem": "No education facts extracted.", "evidence": "Deterministic coverage check"})
    if not profile.get("experience"):
        issues.append({"field": "experience", "severity": "medium", "problem": "No experience facts extracted.", "evidence": "Deterministic coverage check"})
    for error in validation_errors[:5]:
        issues.append({"field": "schema", "severity": "high", "problem": error, "evidence": "Local schema validator"})
    return {
        "score": clamp_score(score),
        "verdict": "deterministic fallback evaluation",
        "field_scores": {
            "coverage": round(coverage, 2),
            "factuality": factuality,
            "schema": schema,
            "deduplication": dedupe,
            "confidence": round(confidence, 2),
        },
        "issues": issues,
        "improvement_hints": [],
    }


def build_agent_payload(
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str],
    memory_examples: list[dict[str, Any]],
    tasks: list[dict[str, Any]] | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "task_type": TASK_TYPE,
        "canonical_profile": compact_profile(profile),
        "source_excerpt": source_excerpt(source_texts),
        "validation_errors": validation_errors,
        "memory_examples": compact_examples(memory_examples),
    }
    if tasks is not None:
        payload["decomposed_tasks"] = tasks
    if evaluation is not None:
        payload["previous_evaluation"] = evaluation
    return payload


def plan_tasks(profile: dict[str, Any], source_texts: list[str], memory_examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    logger.info("agent plan_tasks start source_texts=%s memory_examples=%s", len(source_texts), len(memory_examples))
    payload = {
        "canonical_profile": compact_profile(profile),
        "source_excerpt": source_excerpt(source_texts, 6000),
        "memory_examples": compact_examples(memory_examples),
    }
    result, errors, events = call_gemini_json("agent_task_decomposition", PLANNER_SYSTEM_PROMPT, payload, PLANNER_SCHEMA, 2048)
    tasks = result.get("tasks") if isinstance(result, dict) else None
    if not isinstance(tasks, list) or not tasks:
        logger.info("agent plan_tasks fallback deterministic errors=%s", len(errors))
        return deterministic_plan(), errors, events
    logger.info("agent plan_tasks done tasks=%s errors=%s", len(tasks[:6]), len(errors))
    return tasks[:6], errors, events


def evaluate_profile(
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str],
    memory_examples: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    logger.info("agent evaluate_profile start tasks=%s source_texts=%s", len(tasks), len(source_texts))
    payload = build_agent_payload(profile, source_texts, validation_errors, memory_examples, tasks=tasks)
    result, errors, events = call_gemini_json("agent_profile_evaluation", EVALUATOR_SYSTEM_PROMPT, payload, EVALUATOR_SCHEMA, 4096)
    if not isinstance(result, dict):
        fallback = deterministic_evaluation(profile, validation_errors)
        logger.info("agent evaluate_profile fallback score=%s errors=%s", fallback.get("score"), len(errors))
        return fallback, errors, events
    result["score"] = clamp_score(result.get("score"))
    logger.info("agent evaluate_profile done score=%s issues=%s errors=%s", result.get("score"), len(result.get("issues", [])), len(errors))
    return result, errors, events


def refine_profile(
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str],
    memory_examples: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    logger.info("agent refine_profile start previous_score=%s", evaluation.get("score"))
    payload = build_agent_payload(profile, source_texts, validation_errors, memory_examples, tasks=tasks, evaluation=evaluation)
    result, errors, events = call_gemini_json("agent_profile_refinement", REFINER_SYSTEM_PROMPT, payload, REFINER_SCHEMA, 4096)
    logger.info("agent refine_profile done has_result=%s errors=%s", isinstance(result, dict), len(errors))
    return result, errors, events


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


def apply_refinement(profile: dict[str, Any], refinement: dict[str, Any], source_texts: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info(
        "agent apply_refinement start confidence=%s proposed_skills_add=%s proposed_skills_remove=%s",
        refinement.get("confidence"),
        len(safe_list(refinement.get("skills_add"), 24)),
        len(safe_list(refinement.get("skills_remove"), 24)),
    )
    updated = copy.deepcopy(profile)
    applied: list[dict[str, Any]] = []
    confidence = max(0.0, min(1.0, float(refinement.get("confidence") or 0.0)))
    source_blob = "\n".join(source_texts)

    summary = str(refinement.get("profile_summary") or "").strip()
    if confidence >= 0.65 and 40 <= len(summary) <= 1200:
        updated["profile_summary"] = re.sub(r"\s+", " ", summary)
        updated.setdefault("provenance", []).append(
            {
                "field": "profile_summary",
                "source": "gemini-agent:evaluator-loop",
                "method": "agentic-summary-refinement",
                "confidence": round(min(0.78, confidence), 3),
            }
        )
        applied.append({"field": "profile_summary", "action": "replace", "confidence": round(confidence, 3)})

    existing = {normalize_token(skill.get("name", "")): skill for skill in updated.get("skills", [])}
    for raw_skill in safe_list(refinement.get("skills_add"), 24):
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
            "sources": ["gemini-agent:evidence-backed-refinement"],
        }
        updated.setdefault("skills", []).append(new_skill)
        existing[skill_key] = new_skill
        updated.setdefault("provenance", []).append(
            {
                "field": "skills",
                "source": "gemini-agent:evidence-backed-refinement",
                "method": "agentic-skill-add",
                "confidence": new_skill["confidence"],
                "evidence": str(raw_skill)[:120],
            }
        )
        applied.append({"field": "skills", "action": "add", "value": skill_name, "confidence": new_skill["confidence"]})

    updated["skills"] = sorted(updated.get("skills", []), key=lambda item: (-float(item.get("confidence") or 0), item.get("name") or ""))
    logger.info("agent apply_refinement done applied_changes=%s", len(applied))
    return updated, applied


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
    max_loops = max_loops or env_int("AGENT_MAX_LOOPS", DEFAULT_MAX_LOOPS)
    max_loops = min(max_loops, 5)
    score_threshold = score_threshold or env_float("AGENT_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)
    working = copy.deepcopy(profile)
    errors: list[str] = []
    request_events: list[dict[str, Any]] = []
    llm_available = bool(configured_gemini_keys())
    logger.info(
        "agent run start llm_available=%s max_loops=%s score_threshold=%s memory_examples=%s validation_errors=%s",
        llm_available,
        max_loops,
        score_threshold,
        len(memory_examples),
        len(validation_errors),
    )

    if llm_available:
        tasks, task_errors, task_events = plan_tasks(working, source_texts, memory_examples)
        errors.extend(task_errors)
        request_events.extend(task_events)
    else:
        tasks = deterministic_plan()
    logger.info("agent tasks ready count=%s mode=%s", len(tasks), "llm" if llm_available else "deterministic")

    trace: dict[str, Any] = {
        "enabled": True,
        "task_type": TASK_TYPE,
        "mode": "bounded-gemini-evaluator-loop" if llm_available else "deterministic-fallback-evaluator",
        "model": gemini_model(),
        "score_threshold": score_threshold,
        "max_loops": max_loops,
        "memory_examples_used": len(memory_examples),
        "tasks": tasks,
        "iterations": [],
        "request_events": request_events,
        "input_excerpt": trace_input_excerpt(profile, source_texts),
    }

    best_profile = copy.deepcopy(working)
    best_score = 0.0
    previous_score = 0.0
    stagnant_loops = 0
    stopping_reason = "max loops reached"
    final_evaluation: dict[str, Any] = {}

    for loop_index in range(1, max_loops + 1):
        logger.info("agent loop start loop=%s", loop_index)
        if llm_available:
            evaluation, eval_errors, eval_events = evaluate_profile(working, source_texts, validation_errors, memory_examples, tasks)
            errors.extend(eval_errors)
            request_events.extend(eval_events)
        else:
            evaluation = deterministic_evaluation(working, validation_errors)
            eval_events = []

        score = clamp_score(evaluation.get("score"))
        logger.info("agent loop evaluation loop=%s score=%s best_score=%s", loop_index, score, best_score)
        final_evaluation = evaluation
        iteration: dict[str, Any] = {
            "loop": loop_index,
            "score": score,
            "evaluation": evaluation,
            "request_events": eval_events,
            "applied_changes": [],
        }

        if score > best_score:
            best_score = score
            best_profile = copy.deepcopy(working)

        if score >= score_threshold:
            stopping_reason = "score threshold reached"
            logger.info("agent stop reason=%s loop=%s score=%s", stopping_reason, loop_index, score)
            trace["iterations"].append(iteration)
            break

        if loop_index >= max_loops:
            logger.info("agent stop reason=max loops reached loop=%s score=%s", loop_index, score)
            trace["iterations"].append(iteration)
            break

        if score - previous_score < MIN_SCORE_IMPROVEMENT and loop_index > 1:
            stagnant_loops += 1
        else:
            stagnant_loops = 0
        previous_score = score

        if stagnant_loops >= 2:
            stopping_reason = "score stopped improving"
            logger.info("agent stop reason=%s loop=%s stagnant_loops=%s", stopping_reason, loop_index, stagnant_loops)
            trace["iterations"].append(iteration)
            break

        if not llm_available:
            stopping_reason = "Gemini keys not configured"
            logger.info("agent stop reason=%s", stopping_reason)
            trace["iterations"].append(iteration)
            break

        refinement, refine_errors, refine_events = refine_profile(working, source_texts, validation_errors, memory_examples, tasks, evaluation)
        errors.extend(refine_errors)
        request_events.extend(refine_events)
        iteration["request_events"].extend(refine_events)
        if not isinstance(refinement, dict):
            stopping_reason = "refiner unavailable"
            logger.info("agent stop reason=%s loop=%s", stopping_reason, loop_index)
            trace["iterations"].append(iteration)
            break

        refined_profile, applied = apply_refinement(working, refinement, source_texts)
        iteration["refinement"] = {
            "confidence": refinement.get("confidence"),
            "notes": refinement.get("notes"),
            "skills_add": safe_list(refinement.get("skills_add"), 20),
            "skills_remove": safe_list(refinement.get("skills_remove"), 20),
        }
        iteration["applied_changes"] = applied
        trace["iterations"].append(iteration)
        if not applied:
            stopping_reason = "no safe supported changes from refiner"
            logger.info("agent stop reason=%s loop=%s", stopping_reason, loop_index)
            break
        working = refined_profile

    best_profile["llmops"] = {
        key: value
        for key, value in trace.items()
        if key not in {"input_excerpt"}
    }
    best_profile["llmops"]["final_score"] = best_score
    best_profile["llmops"]["stopping_reason"] = stopping_reason
    best_profile["llmops"]["final_evaluation"] = final_evaluation

    trace["final_score"] = best_score
    trace["stopping_reason"] = stopping_reason
    trace["final_evaluation"] = final_evaluation
    trace["output_preview"] = output_preview(best_profile)
    trace["request_events"] = request_events
    logger.info("agent run done final_score=%s stopping_reason=%s iterations=%s errors=%s", best_score, stopping_reason, len(trace["iterations"]), len(errors))
    return best_profile, trace, errors
