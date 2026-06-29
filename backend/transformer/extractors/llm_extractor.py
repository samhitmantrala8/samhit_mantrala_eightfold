from __future__ import annotations

import json
import os
from typing import Any

import requests

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.skills import canonicalize_skill


SYSTEM_PROMPT = """You extract candidate facts for a deterministic hiring-data transformer.
Return only JSON. Extract only facts explicitly present in the text. Do not infer missing facts.
Each extracted value must include a short evidence string copied from the text."""


def configured_keys() -> list[str]:
    raw = os.getenv("OPENROUTER_KEYS") or os.getenv("OPENROUTER_API_KEY") or ""
    return [key.strip() for key in raw.replace("\n", ",").split(",") if key.strip()]


def extract_text_with_llm(text: str, source: str) -> ExtractionBundle:
    if os.getenv("USE_LLM_EXTRACTOR", "false").lower() not in {"1", "true", "yes"}:
        return ExtractionBundle([], [])

    keys = configured_keys()
    if not keys:
        return ExtractionBundle([], ["llm: enabled but no OpenRouter key configured"])

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    user_prompt = {
        "task": "Extract candidate facts from this text.",
        "schema": {
            "full_name": {"value": "string|null", "evidence": "string|null"},
            "headline": {"value": "string|null", "evidence": "string|null"},
            "years_experience": {"value": "number|null", "evidence": "string|null"},
            "skills": [{"name": "string", "evidence": "string"}],
            "experience": [{"company": "string|null", "title": "string|null", "summary": "string|null", "evidence": "string|null"}],
            "education": [{"institution": "string|null", "degree": "string|null", "field": "string|null", "end_year": "number|null", "evidence": "string|null"}],
        },
        "text": text[:12000],
    }

    last_error = None
    for key in keys:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "Eightfold Candidate Transformer",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_prompt)},
                    ],
                },
                timeout=25,
            )
            if response.status_code in {429, 402, 403}:
                last_error = f"llm: key rejected or rate limited with status {response.status_code}"
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return facts_from_llm_payload(json.loads(content), source)
        except Exception as exc:
            last_error = f"llm: extraction failed: {exc}"
            continue
    return ExtractionBundle([], [last_error or "llm: extraction failed"])


def facts_from_llm_payload(payload: dict[str, Any], source: str) -> ExtractionBundle:
    facts: list[ExtractedFact] = []
    llm_source = f"{source}:llm"

    def value_of(key: str) -> tuple[Any, str | None]:
        node = payload.get(key)
        if isinstance(node, dict):
            return node.get("value"), node.get("evidence")
        return None, None

    for key, field, confidence in [
        ("full_name", "full_name", 0.58),
        ("headline", "headline", 0.55),
        ("years_experience", "years_experience", 0.52),
    ]:
        value, evidence = value_of(key)
        if value not in (None, ""):
            facts.append(ExtractedFact(field, value, llm_source, "llm-json-extraction", confidence, evidence))

    for item in payload.get("skills") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        canonical = canonicalize_skill(str(item["name"]))
        if canonical:
            facts.append(
                ExtractedFact(
                    "skills",
                    {"name": canonical[0]},
                    llm_source,
                    "llm-json-extraction:skill",
                    min(canonical[1], 0.7),
                    item.get("evidence"),
                )
            )

    for item in payload.get("experience") or []:
        if isinstance(item, dict):
            facts.append(ExtractedFact("experience", item, llm_source, "llm-json-extraction:experience", 0.54, item.get("evidence")))
    for item in payload.get("education") or []:
        if isinstance(item, dict):
            facts.append(ExtractedFact("education", item, llm_source, "llm-json-extraction:education", 0.54, item.get("evidence")))
    return ExtractionBundle(facts, [])

