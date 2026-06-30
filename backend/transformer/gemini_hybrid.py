from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

import requests

from backend.transformer.section_classifier import CANONICAL_SECTION_LABELS, classify_line


logger = logging.getLogger(__name__)

GEMINI_SECTION_SCHEMA = {
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
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "kind", "canonical_section", "confidence", "reason"],
            },
        }
    },
    "required": ["items"],
}

GEMINI_SECTION_SYSTEM_PROMPT = """You are a deterministic semantic classifier for candidate/resume text.
Return only JSON matching the schema.

Task:
- Classify whether each input line is itself a section heading, content inside a section, or ambiguous.
- Map section headings to canonical English section names.
- Include a confidence from 0 to 1 for your classification.

Rules:
- Classify the line itself, not the block it belongs to.
- kind=section only when the line is a heading/label introducing a block.
- kind=content when the line is a skill, URL, contact field, degree detail, bullet, or inline field.
- canonical_section is the section represented by the line if kind=section; otherwise use "none".
- Skills such as React, ReAct Agent, Machine Learning, RAG, Python, and Project Management are content, not section headings.
- Recognize multilingual or transliterated headings, including Spanish, French, German, Hindi/Indian-English variants, and mixed English headings.
- Use only these canonical sections: education, experience, skills, projects, achievements, links, competitive_programming, certifications, extracurriculars.
- Be conservative. If unsure whether a line itself is a heading, use ambiguous with canonical_section="none".
- Do not invent information that is not present in the line or its nearby context."""

_ROUND_ROBIN_LOCK = threading.Lock()
_ROUND_ROBIN_INDEX = 0


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
    logger.debug("configured_gemini_keys count=%s", len(keys))
    return keys


def next_gemini_key(keys: list[str]) -> tuple[int, str]:
    global _ROUND_ROBIN_INDEX
    with _ROUND_ROBIN_LOCK:
        position = _ROUND_ROBIN_INDEX + 1
        key = keys[_ROUND_ROBIN_INDEX]
        _ROUND_ROBIN_INDEX = (_ROUND_ROBIN_INDEX + 1) % len(keys)
        return position, key


def candidate_line_items(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    lines = text.splitlines()
    non_empty = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    items: list[dict[str, Any]] = []
    for item_index, (line_index, line) in enumerate(non_empty, start=1):
        next_lines = [candidate for _candidate_index, candidate in non_empty[item_index : item_index + 3]]
        deterministic = classify_line(line, next_lines)
        token_count = len(re.findall(r"[A-Za-z0-9+#.]+", line))
        looks_heading_like = token_count <= 8 and not re.search(r"[.!?]$", line.strip())
        if deterministic.kind == "ambiguous" or looks_heading_like:
            items.append(
                {
                    "id": len(items) + 1,
                    "line_index": line_index,
                    "line": line,
                    "next_lines": next_lines,
                    "deterministic_kind": deterministic.kind,
                    "deterministic_section": deterministic.canonical_section,
                    "deterministic_section_score": deterministic.section_score,
                    "deterministic_content_score": deterministic.content_score,
                }
            )
    return lines, items


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return round(max(0.0, min(1.0, number)), 3)


def call_gemini_section_classifier(items: list[dict[str, Any]], source: str) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    keys = configured_gemini_keys()
    if not keys:
        logger.info("gemini section classifier skipped source=%s reason=no_keys", source)
        return None, [], []
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    errors: list[str] = []
    request_events: list[dict[str, Any]] = []
    payload = {
        "source": source,
        "items": [
            {
                "id": item["id"],
                "line": item["line"],
                "next_lines": item["next_lines"],
                "deterministic_hint": {
                    "kind": item["deterministic_kind"],
                    "canonical_section": item["deterministic_section"] or "none",
                    "section_score": item["deterministic_section_score"],
                    "content_score": item["deterministic_content_score"],
                },
            }
            for item in items
        ],
    }

    for _attempt in range(len(keys)):
        key_position, key = next_gemini_key(keys)
        start = time.perf_counter()
        logger.info("gemini section classifier call start source=%s model=%s key_index=%s items=%s", source, model, key_position, len(items))
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": GEMINI_SECTION_SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "topP": 1,
                        "topK": 1,
                        "maxOutputTokens": 4096,
                        "responseMimeType": "application/json",
                        "responseSchema": GEMINI_SECTION_SCHEMA,
                    },
                },
                timeout=60,
            )
            elapsed = round(time.perf_counter() - start, 2)
            request_events.append({"source": source, "model": model, "key_index": key_position, "status": response.status_code, "seconds": elapsed})
            logger.info(
                "gemini section classifier response source=%s key_index=%s status=%s seconds=%s",
                source,
                key_position,
                response.status_code,
                elapsed,
            )
            if response.status_code in {403, 429, 503}:
                errors.append(f"gemini:{source}: model returned status {response.status_code}")
                logger.warning("gemini section classifier retryable_status source=%s key_index=%s status=%s", source, key_position, response.status_code)
                continue
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
            logger.info("gemini section classifier parsed source=%s items=%s", source, len(parsed.get("items", [])))
            return parsed, errors, request_events
        except Exception as exc:
            elapsed = round(time.perf_counter() - start, 2)
            request_events.append({"source": source, "model": model, "key_index": key_position, "error": str(exc), "seconds": elapsed})
            errors.append(f"gemini:{source}: semantic mapping failed: {exc}")
            logger.exception("gemini section classifier failed source=%s key_index=%s seconds=%s", source, key_position, elapsed)
    logger.warning("gemini section classifier exhausted_keys source=%s attempts=%s errors=%s", source, len(keys), len(errors))
    return None, errors, request_events


def canonicalize_section_headings(text: str, source: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    lines, items = candidate_line_items(text)
    if not items:
        logger.info("canonicalize_section_headings skipped source=%s reason=no_candidate_lines", source)
        return text, [], []
    logger.info("canonicalize_section_headings start source=%s candidate_lines=%s total_lines=%s", source, len(items), len(lines))

    payload, errors, request_events = call_gemini_section_classifier(items, source)
    if not payload:
        logger.info("canonicalize_section_headings fallback source=%s errors=%s", source, len(errors))
        return text, [], errors

    by_id = {item.get("id"): item for item in payload.get("items", []) if isinstance(item, dict)}
    source_items = {item["id"]: item for item in items}
    mappings: list[dict[str, Any]] = []
    rewritten = list(lines)

    for item_id, source_item in source_items.items():
        mapped = by_id.get(item_id)
        if not mapped:
            continue
        kind = mapped.get("kind") if mapped.get("kind") in {"section", "content", "ambiguous"} else "ambiguous"
        canonical = mapped.get("canonical_section")
        canonical_section = canonical if canonical and canonical != "none" else None
        confidence = clamp_confidence(mapped.get("confidence"))
        reason = str(mapped.get("reason") or "").strip()
        mapped_to = CANONICAL_SECTION_LABELS.get(canonical_section, canonical_section) if canonical_section else None

        if kind == "section" and canonical_section and confidence >= 0.62:
            rewritten[source_item["line_index"]] = str(mapped_to)

        mappings.append(
            {
                "source": source,
                "method": "gemini-section-canonicalization",
                "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                "original": source_item["line"],
                "kind": kind,
                "mapped_to": mapped_to,
                "canonical_section": canonical_section,
                "confidence": confidence,
                "reason": reason,
                "applied": bool(kind == "section" and canonical_section and confidence >= 0.62),
            }
        )

    for event in request_events:
        mappings.append(
            {
                "source": source,
                "method": "gemini-request",
                "model": event.get("model"),
                "original": "Gemini semantic mapping request",
                "kind": "request",
                "mapped_to": None,
                "canonical_section": None,
                "confidence": 1.0 if event.get("status") == 200 else 0.0,
                "reason": f"key_index={event.get('key_index')}, status={event.get('status', event.get('error'))}, seconds={event.get('seconds')}",
                "applied": False,
            }
        )

    applied_count = sum(1 for mapping in mappings if mapping.get("applied"))
    logger.info("canonicalize_section_headings done source=%s mappings=%s applied=%s errors=%s", source, len(mappings), applied_count, len(errors))
    return "\n".join(rewritten), mappings, errors
