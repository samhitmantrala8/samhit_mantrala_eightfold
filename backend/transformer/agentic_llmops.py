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
from backend.transformer.normalizers.contact import normalize_email, normalize_phone, normalize_url
from backend.transformer.normalizers.skills import canonicalize_skill, normalize_token


logger = logging.getLogger(__name__)

TASK_TYPE = "candidate_profile_agent"
CANONICAL_MAPPING_AGENT = "canonical_mapping_agent"
ACCEPT_CONFIDENCE = 0.90
DISCARD_CONFIDENCE = 0.70
EVALUATOR_PASS_SCORE = 8.0
DEFAULT_MAX_LOOPS = 3
MIN_SCORE_IMPROVEMENT = 0.3

CANONICAL_FIELD_SPECS = [
    {"field": "full_name", "agent": "full_name_agent", "kind": "scalar"},
    {"field": "emails", "agent": "emails_agent", "kind": "list"},
    {"field": "phones", "agent": "phones_agent", "kind": "list"},
    {"field": "location", "agent": "location_agent", "kind": "object"},
    {"field": "links", "agent": "links_agent", "kind": "object"},
    {"field": "headline", "agent": "headline_agent", "kind": "scalar"},
    {"field": "years_experience", "agent": "years_experience_agent", "kind": "scalar"},
    {"field": "education", "agent": "education_agent", "kind": "list"},
    {"field": "experience", "agent": "experience_agent", "kind": "list"},
    {"field": "projects", "agent": "projects_agent", "kind": "list"},
    {"field": "skills", "agent": "skills_agent", "kind": "list"},
    {"field": "achievements", "agent": "achievements_agent", "kind": "list"},
    {"field": "certifications", "agent": "certifications_agent", "kind": "list"},
    {"field": "publications", "agent": "publications_agent", "kind": "list"},
    {"field": "online_coding_profile", "agent": "online_coding_profile_agent", "kind": "object"},
    {"field": "github_repositories", "agent": "github_repositories_agent", "kind": "list"},
    {"field": "languages", "agent": "languages_agent", "kind": "list"},
    {"field": "extracurriculars", "agent": "extracurriculars_agent", "kind": "list"},
    {"field": "profile_summary", "agent": "profile_summary_agent", "kind": "scalar"},
    {"field": "other_sections", "agent": "other_sections_agent", "kind": "list"},
    {"field": "others", "agent": "others_agent", "kind": "list"},
]

FIELD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale_summary": {"type": "string"},
        "value_json": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["rationale_summary", "value_json", "confidence", "evidence"],
}

FIELD_EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "passed": {"type": "boolean"},
        "use_output": {"type": "boolean"},
        "verdict": {"type": "string"},
        "rubric_scores": {
            "type": "object",
            "properties": {
                "correctness": {"type": "number"},
                "format": {"type": "number"},
                "evidence": {"type": "number"},
                "specificity": {"type": "number"},
            },
            "required": ["correctness", "format", "evidence", "specificity"],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "evidence": {"type": "string"},
                },
                "required": ["problem", "severity", "evidence"],
            },
        },
        "improvement_hint": {"type": "string"},
    },
    "required": ["score", "passed", "use_output", "verdict", "rubric_scores", "issues", "improvement_hint"],
}

CANONICAL_MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale_summary": {"type": "string"},
        "patches_json": {"type": "string"},
        "remaining_others_json": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["rationale_summary", "patches_json", "remaining_others_json", "confidence", "evidence"],
}

CANONICAL_MAPPING_PROMPT = """You are canonical_mapping_agent, a bounded ReACT-style mapper for candidate profile data.

Goal:
- Convert odd ATS keys, unknown sections, and values currently in others into known canonical fields when clearly supported.
- Leave uncertain content in others.

Known canonical fields:
{canonical_fields_json}

Return only JSON matching the schema.

Rules:
- Do not expose hidden chain-of-thought. Use rationale_summary with short observable reasoning only.
- patches_json must be a JSON array of objects: {{"field": canonical_field_name, "value": canonical_value}}.
- remaining_others_json must be a JSON array of objects that should stay in others.
- Only use canonical field names listed above.
- Do not invent missing values, dates, links, companies, schools, or skills.
- Keep values in others unless the mapping is direct and evidence-backed.
- The evaluator must score >= 8 before patches are applied."""

CANONICAL_MAPPING_EVALUATOR_PROMPT = """You are a strict evaluator for canonical_mapping_agent.

Evaluate whether proposed patches correctly move values from unknown ATS fields or others into known canonical fields.

Rubrics, each 1-10:
1. correctness: every patch maps to the right canonical field.
2. format: every patch uses valid field names and canonical value shape.
3. evidence: every patch is supported by the mapper input.
4. specificity: patches are precise and do not move unrelated content.

Set passed=true and use_output=true only when score >= 8 and all proposed mappings are supported.
If any patch is hallucinated, ambiguous, or overbroad, set use_output=false.
Return only JSON matching the schema."""

FIELD_AGENT_PROMPT_TEMPLATE = """You are {agent_name}, a bounded ReACT-style extraction agent for one canonical candidate field.

Canonical field: {field}
Expected JSON value kind: {kind}

Field-specific instructions:
{field_instructions}

Good examples for this same field agent are provided below. Use them as guidance for format and strictness only.
GOOD_EXAMPLES_JSON:
{examples_json}

Task:
- Extract or repair only the canonical field named above.
- Use the field-specific input, canonical profile snapshot, and previous evaluator feedback.
- Return only JSON matching the schema.
- Put the final field value in value_json as a JSON-encoded value.
- Do not output markdown.

Rules:
- Do not expose hidden chain-of-thought. Use rationale_summary with short observable reasoning only.
- Do not invent names, dates, companies, links, projects, emails, phones, skills, or achievements.
- If the source does not support a value, return value_json as null and confidence below 0.7.
- Prefer exact text from the source for names, institutions, companies, titles, project names, and links.
- For emails, return a JSON array of normalized email strings.
- For phones, return a JSON array of E.164 phone strings.
- For links, return an object with linkedin, github, portfolio, other.
- For list fields, return a JSON array.
- For object fields, return a JSON object.
- For scalar fields, return a JSON string, number, or null."""

FIELD_EVALUATOR_PROMPT_TEMPLATE = """You are a strict evaluator for one canonical candidate field.

Canonical field: {field}
Agent name: {agent_name}
Expected JSON value kind: {kind}

Field-specific validation rules:
{field_instructions}

Evaluate the proposed output using only the provided field-specific input and canonical profile snapshot.

Rubrics, each 1-10:
1. correctness: the value matches the candidate's true field value in the input.
2. format: the value follows the required canonical type and formatting.
3. evidence: the value is directly supported by the input or existing provenance.
4. specificity: the value is neither vague nor padded with unrelated information.

Set passed=true and use_output=true only when score >= 8 and the output is supported.
Set use_output=false for hallucinated, unrelated, overbroad, duplicate, or schema-unsafe output.
Return only JSON matching the schema."""

FIELD_SPECIFIC_INSTRUCTIONS = {
    "full_name": "Return exactly the candidate/person full name as a string. Ignore organization names, college names, job titles, usernames, and labels like 'resume'. Preserve casing when present.",
    "emails": "Return a JSON array of email strings only. Do not infer emails from names or domains. Reject placeholder or invalid email-like text.",
    "phones": "Return a JSON array of E.164 phone strings only. Use the provided default phone region only when the source lacks a country code. Do not invent a country code when the number is ambiguous.",
    "location": "Return an object with city, region, country. Use candidate location only, not company, college, or project deployment locations unless explicitly marked as candidate address/location.",
    "links": "Return an object with linkedin, github, portfolio, other. Only include valid URLs from source text. Put Codeforces, LeetCode, Kaggle, X/Twitter, blogs, and miscellaneous profiles in other unless one of the dedicated keys applies.",
    "headline": "Return a concise current role/headline string only when supported by source. Prefer current role + current company. Do not create marketing copy.",
    "years_experience": "Return a number representing professional experience years only when date ranges or explicit years are present. Do not count education duration or project duration.",
    "education": "Return an array of objects with institution, degree, field, end_year, cgpa. Map keys such as college, school, university, institute, academics, degree, branch, major, CGPA/GPA into this field.",
    "experience": "Return an array of objects with company, title, role, location, duration, start, end, summary. Map employer/company/organization and designation/position/job title. Do not include projects as experience unless explicitly job/internship history.",
    "projects": "Return an array of project objects with title, date, tech_stack, links, bullets. Keep only source-supported projects and do not duplicate work experience bullets as projects.",
    "skills": "Return an array of skill names or skill objects. Include only technologies, frameworks, languages, tools, concepts, and platforms explicitly present in source. Do not add generic soft skills unless listed as skills.",
    "achievements": "Return an array of achievements with title, summary, links. Include awards, ranks, contest results, recognitions, and notable quantified accomplishments only when explicitly present.",
    "certifications": "Return an array of certifications/licenses only. Do not treat courses, college subjects, or ordinary skills as certifications.",
    "publications": "Return an array of publications/research papers/articles only when title/venue/link or clear publication wording is present.",
    "online_coding_profile": "Return an object for online coding profiles such as Codeforces, LeetCode, Kaggle, HackerRank, GeeksforGeeks, CodeChef. Include handle, profile_url, rating, rank, max_rating, badges when present.",
    "github_repositories": "Return an array of GitHub repository objects only when repository data or GitHub URLs are present. Do not create repositories from generic projects without GitHub evidence.",
    "languages": "Return spoken/human languages only, not programming languages. Programming languages belong in skills.",
    "extracurriculars": "Return activities, leadership, volunteering, clubs, sports, and community work only when explicitly listed.",
    "profile_summary": "Return one concise paragraph based only on canonical profile facts. Mention missing major sections only by omission, never as 'no information available'.",
    "other_sections": "Preserve unmapped resume sections as title/content objects. Do not discard content that might matter but cannot be confidently mapped.",
    "others": "Preserve unmapped ATS or JSON fields as title/content/source objects. This is a fallback bucket, not a place to invent normalized fields.",
}


def field_instructions(field: str) -> str:
    return FIELD_SPECIFIC_INSTRUCTIONS.get(field, "Extract only this canonical field using direct evidence from the provided input.")


def gemini_model() -> str:
    return os.getenv("GEMINI_AGENT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "field"


def safe_list(value: Any, limit: int = 20) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 1.0
    return round(max(1.0, min(10.0, number)), 2)


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return round(max(0.0, min(1.0, number)), 3)


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
            event = {"task": task_name, "model": model, "key_index": key_position, "status": response.status_code, "seconds": elapsed}
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


def field_value(profile: dict[str, Any], field: str) -> Any:
    return profile.get(field)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return not value or not any(bool(item) for item in value.values())
    if isinstance(value, (list, str)) and not value:
        return True
    return False


def empty_value(field: str, kind: str) -> Any:
    if field == "location":
        return {"city": None, "region": None, "country": None}
    if field == "links":
        return {"linkedin": None, "github": None, "portfolio": None, "other": []}
    if kind == "list":
        return []
    if kind == "object":
        return {}
    return None


def clear_field(profile: dict[str, Any], field: str, kind: str) -> None:
    profile[field] = empty_value(field, kind)


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": profile.get("full_name"),
        "emails": safe_list(profile.get("emails"), 3),
        "phones": safe_list(profile.get("phones"), 3),
        "location": profile.get("location") or {},
        "links": profile.get("links") or {},
        "headline": profile.get("headline"),
        "years_experience": profile.get("years_experience"),
        "education": safe_list(profile.get("education"), 6),
        "experience": safe_list(profile.get("experience"), 8),
        "projects": safe_list(profile.get("projects"), 8),
        "achievements": safe_list(profile.get("achievements"), 8),
        "certifications": safe_list(profile.get("certifications"), 8),
        "publications": safe_list(profile.get("publications"), 8),
        "online_coding_profile": profile.get("online_coding_profile") or {},
        "github_repositories": safe_list(profile.get("github_repositories"), 8),
        "languages": safe_list(profile.get("languages"), 12),
        "extracurriculars": safe_list(profile.get("extracurriculars"), 8),
        "skills": safe_list(profile.get("skills"), 80),
        "profile_summary": profile.get("profile_summary"),
        "other_sections": safe_list(profile.get("other_sections"), 8),
        "others": safe_list(profile.get("others"), 12),
    }


def source_excerpt(source_texts: list[str], char_limit: int = 18000) -> str:
    joined = "\n\n--- SOURCE BREAK ---\n\n".join(text.strip() for text in source_texts if text.strip())
    if len(joined) <= char_limit:
        return joined
    head = joined[: char_limit // 2]
    tail = joined[-char_limit // 2 :]
    return f"{head}\n\n--- SOURCE TRUNCATED ---\n\n{tail}"


def lines_matching(source_texts: list[str], patterns: list[str], limit: int = 30) -> str:
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    matches = []
    for text in source_texts:
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned and any(regex.search(cleaned) for regex in regexes):
                matches.append(cleaned)
            if len(matches) >= limit:
                return "\n".join(matches)
    return "\n".join(matches)


def field_relevant_input(field: str, profile: dict[str, Any], source_texts: list[str]) -> str:
    sections = profile.get("resume_sections") or {}
    section_map = {
        "education": ["Education", "Academics"],
        "experience": ["Experience", "Professional Background"],
        "projects": ["Projects"],
        "skills": ["Skills", "Skills Summary"],
        "achievements": ["Achievements"],
        "certifications": ["Certifications"],
        "publications": ["Publications"],
        "extracurriculars": ["Extracurriculars"],
        "other_sections": list(sections.keys()),
        "others": list(sections.keys()),
    }
    if field in {"emails", "phones", "links", "full_name", "location", "headline"}:
        text = lines_matching(source_texts, [r"email|phone|mobile|linkedin|github|portfolio|location|address|headline|name|http|www\.|\.com"], 40)
        return text or source_excerpt(source_texts, 1800)
    if field in section_map:
        chunks = [sections[name] for name in section_map[field] if name in sections and sections[name]]
        if chunks:
            return "\n".join(chunks)[:4000]
    if field == "github_repositories":
        return json.dumps(profile.get("github_repositories") or profile.get("projects") or [], ensure_ascii=False)[:4000]
    if field == "online_coding_profile":
        return lines_matching(source_texts, [r"codeforces|leetcode|kaggle|hacker cup|contest|rating"], 40) or source_excerpt(source_texts, 1800)
    if field == "profile_summary":
        return json.dumps(compact_profile(profile), ensure_ascii=False)[:5000]
    if field == "others":
        return json.dumps({"others": profile.get("others") or [], "other_sections": profile.get("other_sections") or []}, ensure_ascii=False)[:4000]
    return source_excerpt(source_texts, 2500)


def trace_input_excerpt(profile: dict[str, Any], source_texts: list[str]) -> dict[str, Any]:
    digest = hashlib.sha256("\n".join(source_texts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return {
        "source_digest": digest,
        "source_count": len(source_texts),
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
        "education_count": len(profile.get("education") or []),
        "experience_count": len(profile.get("experience") or []),
        "project_count": len(profile.get("projects") or []),
        "skill_count": len(profile.get("skills") or []),
        "summary": profile.get("profile_summary"),
    }


def provenance_for_field(profile: dict[str, Any], field: str) -> list[dict[str, Any]]:
    provenance = profile.get("provenance") or []
    if field == "location":
        return [item for item in provenance if str(item.get("field", "")).startswith("location.")]
    if field == "links":
        return [item for item in provenance if str(item.get("field", "")).startswith("links.")]
    return [item for item in provenance if item.get("field") == field]


def field_confidence(profile: dict[str, Any], field: str) -> float:
    if field == "skills" and profile.get("skills"):
        return round(max(float(skill.get("confidence") or 0.0) for skill in profile["skills"]), 3)
    facts = provenance_for_field(profile, field)
    if facts:
        return round(max(float(item.get("confidence") or 0.0) for item in facts), 3)
    value = field_value(profile, field)
    return 0.0 if is_missing(value) else 0.5


def local_evaluate_field(field: str, value: Any, confidence: float, profile: dict[str, Any]) -> dict[str, Any]:
    score = 1.0
    issues: list[dict[str, str]] = []
    if is_missing(value):
        issues.append({"problem": "Field is missing.", "severity": "medium", "evidence": "empty canonical value"})
        score = 1.0
    elif field == "emails":
        invalid = [item for item in value if not normalize_email(str(item))]
        score = 9.0 if not invalid else 5.0
    elif field == "phones":
        invalid = [item for item in value if not re.fullmatch(r"\+[1-9]\d{7,14}", str(item))]
        score = 9.0 if not invalid else 5.0
    elif field == "links":
        flat_links = [item for item in [value.get("github"), value.get("linkedin"), value.get("portfolio"), *(value.get("other") or [])] if item]
        invalid = [item for item in flat_links if not normalize_url(str(item))]
        score = 8.5 if flat_links and not invalid else 6.0
    elif field == "full_name":
        score = 9.0 if isinstance(value, str) and len(value.split()) >= 2 and not any(char.isdigit() for char in value) else 5.0
    elif isinstance(value, list):
        score = 8.5 if value else 1.0
    elif isinstance(value, dict):
        score = 8.0 if any(bool(item) for item in value.values()) else 1.0
    elif value:
        score = 8.0
    score = max(score, min(9.5, confidence * 10)) if score >= 8.0 else score
    return {
        "score": clamp_score(score),
        "passed": score >= EVALUATOR_PASS_SCORE,
        "use_output": score >= EVALUATOR_PASS_SCORE,
        "verdict": "Local deterministic field evaluator completed.",
        "rubric_scores": {
            "correctness": clamp_score(score),
            "format": clamp_score(score),
            "evidence": clamp_score(max(1.0, confidence * 10)),
            "specificity": clamp_score(score),
        },
        "issues": issues,
        "improvement_hint": "Escalate to field agent if score is below threshold.",
    }


def compact_examples(examples: list[dict[str, Any]], agent_name: str) -> list[dict[str, Any]]:
    compacted = []
    for item in examples:
        task_type = str(item.get("task_type") or "")
        if not task_type.endswith(f":{agent_name}"):
            continue
        compacted.append(
            {
                "score": item.get("score"),
                "input": item.get("example_input") or item.get("input_excerpt"),
                "output": item.get("example_output") or item.get("output_preview"),
                "evaluation": item.get("evaluator_json"),
            }
        )
        if len(compacted) >= 5:
            break
    return compacted


def build_field_system_prompt(spec: dict[str, str], examples: list[dict[str, Any]]) -> str:
    return FIELD_AGENT_PROMPT_TEMPLATE.format(
        agent_name=spec["agent"],
        field=spec["field"],
        kind=spec["kind"],
        field_instructions=field_instructions(spec["field"]),
        examples_json=json.dumps(compact_examples(examples, spec["agent"]), ensure_ascii=False, indent=2),
    )


def build_field_evaluator_prompt(spec: dict[str, str]) -> str:
    return FIELD_EVALUATOR_PROMPT_TEMPLATE.format(
        agent_name=spec["agent"],
        field=spec["field"],
        kind=spec["kind"],
        field_instructions=field_instructions(spec["field"]),
    )


def parse_value_json(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def normalize_field_value(field: str, value: Any, default_region: str) -> Any:
    if field == "emails":
        items = value if isinstance(value, list) else [value]
        return [email for item in items if (email := normalize_email(str(item)))]
    if field == "phones":
        items = value if isinstance(value, list) else [value]
        return [phone for item in items if (phone := normalize_phone(str(item), default_region))]
    if field == "links":
        value = value if isinstance(value, dict) else {}
        return {
            "linkedin": normalize_url(value.get("linkedin")) if value.get("linkedin") else None,
            "github": normalize_url(value.get("github")) if value.get("github") else None,
            "portfolio": normalize_url(value.get("portfolio")) if value.get("portfolio") else None,
            "other": [url for item in value.get("other", []) if (url := normalize_url(str(item)))],
        }
    if field == "skills":
        items = value if isinstance(value, list) else []
        skills = []
        seen = set()
        for item in items:
            raw = item.get("name") if isinstance(item, dict) else str(item)
            canonical = canonicalize_skill(raw)
            if canonical and canonical[0] not in seen:
                skills.append({"name": canonical[0], "confidence": round(min(0.82, canonical[1]), 3), "sources": ["field-agent"]})
                seen.add(canonical[0])
        return skills
    return value


def spec_for_field(field: str) -> dict[str, str] | None:
    for spec in CANONICAL_FIELD_SPECS:
        if spec["field"] == field:
            return spec
    return None


def canonical_mapping_input(profile: dict[str, Any], source_texts: list[str]) -> dict[str, Any]:
    return {
        "others": safe_list(profile.get("others"), 20),
        "other_sections": safe_list(profile.get("other_sections"), 20),
        "semantic_mappings": safe_list(profile.get("semantic_mappings"), 20),
        "resume_section_labels": list((profile.get("resume_sections") or {}).keys()),
        "compact_profile": compact_profile(profile),
        "source_excerpt": source_excerpt(source_texts, 5000),
    }


def needs_canonical_mapping(profile: dict[str, Any]) -> bool:
    return bool(profile.get("others") or profile.get("other_sections") or profile.get("semantic_mappings"))


def value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def merge_list(existing: Any, incoming: Any) -> list[Any]:
    merged = list(existing) if isinstance(existing, list) else []
    seen = {value_key(item) for item in merged}
    incoming_items = incoming if isinstance(incoming, list) else [incoming]
    for item in incoming_items:
        if is_missing(item):
            continue
        key = value_key(item)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def merge_object(existing: Any, incoming: Any) -> dict[str, Any]:
    base = dict(existing) if isinstance(existing, dict) else {}
    if not isinstance(incoming, dict):
        return base
    for key, value in incoming.items():
        if is_missing(value):
            continue
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            base[key] = merge_object(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            base[key] = merge_list(current, value)
        elif is_missing(current):
            base[key] = value
    return base


def add_mapping_provenance(profile: dict[str, Any], field: str, confidence: float) -> None:
    profile.setdefault("provenance", []).append(
        {
            "field": field,
            "source": "canonical_mapping_agent",
            "method": "gemini-canonical-mapping",
            "confidence": round(max(0.0, min(0.99, confidence)), 3),
        }
    )


def apply_canonical_patch(profile: dict[str, Any], field: str, value: Any, default_region: str, confidence: float) -> bool:
    spec = spec_for_field(field)
    if not spec or field in {"others", "other_sections"}:
        return False
    normalized = normalize_field_value(field, value, default_region)
    if is_missing(normalized):
        return False
    if spec["kind"] == "list":
        profile[field] = merge_list(profile.get(field), normalized)
        add_mapping_provenance(profile, field, confidence)
        return True
    if spec["kind"] == "object":
        profile[field] = merge_object(profile.get(field), normalized)
        add_mapping_provenance(profile, field, confidence)
        return True
    if is_missing(profile.get(field)):
        profile[field] = normalized
        add_mapping_provenance(profile, field, confidence)
        return True
    return False


def apply_canonical_mapping(
    profile: dict[str, Any],
    patches: list[dict[str, Any]],
    remaining_others: Any,
    default_region: str,
    confidence: float,
) -> list[str]:
    applied: list[str] = []
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        field = str(patch.get("field") or "")
        if apply_canonical_patch(profile, field, patch.get("value"), default_region, confidence):
            applied.append(field)
    if isinstance(remaining_others, list):
        profile["others"] = remaining_others
    return applied


def value_matches_kind(value: Any, kind: str) -> bool:
    if value is None:
        return False
    if kind == "list":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return isinstance(value, (str, int, float))


def run_canonical_mapping_agent(
    profile: dict[str, Any],
    source_texts: list[str],
    memory_examples: list[dict[str, Any]],
    default_region: str,
    max_loops: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    system_prompt = CANONICAL_MAPPING_PROMPT.format(
        canonical_fields_json=json.dumps(
            [{"field": spec["field"], "kind": spec["kind"]} for spec in CANONICAL_FIELD_SPECS if spec["field"] not in {"others", "other_sections"}],
            ensure_ascii=False,
            indent=2,
        )
    )
    field_input = canonical_mapping_input(profile, source_texts)
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    best_score = 0.0
    accepted_payload: dict[str, Any] | None = None
    accepted_evaluation: dict[str, Any] | None = None
    stopping_reason = "max loops reached"
    working = copy.deepcopy(profile)
    examples = compact_examples(memory_examples, CANONICAL_MAPPING_AGENT)

    for loop_index in range(1, max_loops + 1):
        payload = {
            "mapper_input": field_input,
            "good_examples": examples,
            "previous_attempts": iterations[-2:],
        }
        result, result_errors, result_events = call_gemini_json(
            f"{CANONICAL_MAPPING_AGENT}_generate_loop_{loop_index}",
            system_prompt,
            payload,
            CANONICAL_MAPPING_SCHEMA,
            4096,
        )
        errors.extend(result_errors)
        events.extend(result_events)
        iteration = {
            "loop": loop_index,
            "action": "canonical_mapping_generate",
            "request_events": list(result_events),
            "candidate_output": result,
            "rationale_summary": result.get("rationale_summary") if isinstance(result, dict) else "No mapper JSON returned.",
        }
        if not isinstance(result, dict):
            iteration.update({"score": 1.0, "passed": False, "observation": "Mapper did not return valid JSON."})
            iterations.append(iteration)
            stopping_reason = "mapping generation failed"
            break

        patches = parse_value_json(result.get("patches_json"))
        remaining_others = parse_value_json(result.get("remaining_others_json"))
        if not isinstance(patches, list):
            patches = []
        if not isinstance(remaining_others, list):
            remaining_others = profile.get("others") or []
        evaluator_payload = {
            "mapper_input": field_input,
            "candidate_patches": patches,
            "remaining_others": remaining_others,
            "candidate_confidence": result.get("confidence"),
            "candidate_rationale_summary": result.get("rationale_summary"),
        }
        evaluation, eval_errors, eval_events = call_gemini_json(
            f"{CANONICAL_MAPPING_AGENT}_evaluate_loop_{loop_index}",
            CANONICAL_MAPPING_EVALUATOR_PROMPT,
            evaluator_payload,
            FIELD_EVALUATOR_SCHEMA,
            2048,
        )
        errors.extend(eval_errors)
        events.extend(eval_events)
        iteration["request_events"].extend(eval_events)
        if not isinstance(evaluation, dict):
            evaluation = {
                "score": 1.0,
                "passed": False,
                "use_output": False,
                "verdict": "Canonical mapping evaluator did not return valid JSON; patches were not applied.",
                "rubric_scores": {"correctness": 1.0, "format": 1.0, "evidence": 1.0, "specificity": 1.0},
                "issues": [{"problem": "Missing evaluator result.", "severity": "high", "evidence": "no evaluator JSON"}],
                "improvement_hint": "Keep content in others unless an evaluator passes the mapping.",
            }
        evaluation["score"] = clamp_score(evaluation.get("score"))
        evaluation["passed"] = bool(evaluation.get("passed")) and evaluation["score"] >= EVALUATOR_PASS_SCORE
        evaluation["use_output"] = bool(evaluation.get("use_output")) and evaluation["passed"]
        best_score = max(best_score, evaluation["score"])
        iteration.update(
            {
                "candidate_value": {"patches": patches, "remaining_others": remaining_others},
                "evaluation": evaluation,
                "score": evaluation["score"],
                "passed": evaluation["passed"],
                "observation": evaluation.get("verdict"),
            }
        )
        iterations.append(iteration)
        logger.info("canonical mapping evaluated loop=%s score=%s passed=%s", loop_index, evaluation["score"], evaluation["passed"])
        if evaluation["passed"] and evaluation["use_output"]:
            accepted_payload = {"patches": patches, "remaining_others": remaining_others}
            accepted_evaluation = evaluation
            stopping_reason = "score threshold reached"
            break

    applied_fields: list[str] = []
    final_score = clamp_score(accepted_evaluation.get("score") if accepted_evaluation else best_score or 1.0)
    if accepted_payload:
        applied_fields = apply_canonical_mapping(
            working,
            accepted_payload["patches"],
            accepted_payload["remaining_others"],
            default_region,
            max(0.8, min(0.96, final_score / 10)),
        )
    accepted = bool(accepted_payload and applied_fields and final_score >= EVALUATOR_PASS_SCORE)
    if not accepted:
        final_score = clamp_score(min(final_score, EVALUATOR_PASS_SCORE - 0.01))
    trace = {
        "task_name": CANONICAL_MAPPING_AGENT,
        "field": "others",
        "purpose": "Map unknown ATS keys and other sections into canonical fields when confidence is high.",
        "mode": "react",
        "target_fields": ["others"],
        "system_prompt": system_prompt,
        "evaluator_prompt": CANONICAL_MAPPING_EVALUATOR_PROMPT,
        "field_specific_input": json.dumps(field_input, ensure_ascii=False)[:2500],
        "loops": len(iterations),
        "final_score": final_score,
        "passed": accepted,
        "accepted": accepted,
        "discarded": not accepted,
        "stopping_reason": stopping_reason if accepted else "mapping score below threshold or no supported patches",
        "final_output": {"applied_fields": applied_fields, "others": working.get("others", [])} if accepted else None,
        "iterations": iterations,
        "request_events": events,
    }
    good_examples = []
    if accepted:
        good_examples.append(
            {
                "task_type": f"{TASK_TYPE}:{CANONICAL_MAPPING_AGENT}",
                "score": final_score,
                "input": {
                    "system_prompt": system_prompt,
                    "mapper_input": field_input,
                },
                "output": trace["final_output"],
                "evaluation": accepted_evaluation or {},
            }
        )
    return working if accepted else profile, trace, good_examples, errors


def run_llm_evaluator(
    spec: dict[str, str],
    candidate_value: Any,
    field_input: str,
    profile: dict[str, Any],
    default_region: str,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], str]:
    evaluator_prompt = build_field_evaluator_prompt(spec)
    payload = {
        "field": spec["field"],
        "candidate_value": candidate_value,
        "field_specific_input": field_input,
        "canonical_profile": compact_profile(profile),
        "default_phone_region": default_region,
    }
    result, errors, events = call_gemini_json(f"{spec['agent']}_evaluate_deterministic", evaluator_prompt, payload, FIELD_EVALUATOR_SCHEMA, 2048)
    if not isinstance(result, dict):
        result = local_evaluate_field(spec["field"], candidate_value, 0.75, profile)
    result["score"] = clamp_score(result.get("score"))
    result["passed"] = bool(result.get("passed")) and result["score"] >= EVALUATOR_PASS_SCORE
    result["use_output"] = bool(result.get("use_output")) and result["passed"]
    return result, errors, events, evaluator_prompt


def run_field_agent(
    spec: dict[str, str],
    profile: dict[str, Any],
    source_texts: list[str],
    memory_examples: list[dict[str, Any]],
    default_region: str,
    previous_evaluation: dict[str, Any] | None,
    max_loops: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[str]]:
    field = spec["field"]
    agent_name = spec["agent"]
    system_prompt = build_field_system_prompt(spec, memory_examples)
    evaluator_prompt = build_field_evaluator_prompt(spec)
    field_input = field_relevant_input(field, profile, source_texts)
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    best_score = 0.0
    previous_score = 0.0
    stagnant = 0
    accepted_value = None
    accepted_evaluation: dict[str, Any] | None = None
    stopping_reason = "max loops reached"

    for loop_index in range(1, max_loops + 1):
        payload = {
            "field": field,
            "field_specific_input": field_input,
            "canonical_profile": compact_profile(profile),
            "current_value": field_value(profile, field),
            "previous_evaluation": previous_evaluation,
            "default_phone_region": default_region,
        }
        result, result_errors, result_events = call_gemini_json(f"{agent_name}_generate_loop_{loop_index}", system_prompt, payload, FIELD_OUTPUT_SCHEMA, 4096)
        errors.extend(result_errors)
        events.extend(result_events)
        iteration = {
            "loop": loop_index,
            "action": "field_agent_generate",
            "request_events": list(result_events),
            "candidate_output": result,
            "rationale_summary": result.get("rationale_summary") if isinstance(result, dict) else "No candidate JSON returned.",
        }
        if not isinstance(result, dict):
            iteration["score"] = 1.0
            iteration["passed"] = False
            iteration["observation"] = "Field agent did not return valid JSON."
            iterations.append(iteration)
            stopping_reason = "candidate generation failed"
            break

        raw_value = parse_value_json(result.get("value_json"))
        candidate_value = normalize_field_value(field, raw_value, default_region)
        evaluator_payload = {
            "field": field,
            "field_specific_input": field_input,
            "canonical_profile": compact_profile(profile),
            "candidate_value": candidate_value,
            "candidate_rationale_summary": result.get("rationale_summary"),
            "candidate_confidence": result.get("confidence"),
        }
        evaluation, eval_errors, eval_events = call_gemini_json(f"{agent_name}_evaluate_loop_{loop_index}", evaluator_prompt, evaluator_payload, FIELD_EVALUATOR_SCHEMA, 2048)
        errors.extend(eval_errors)
        events.extend(eval_events)
        iteration["request_events"].extend(eval_events)
        if not isinstance(evaluation, dict):
            evaluation = local_evaluate_field(field, candidate_value, clamp_confidence(result.get("confidence")), profile)
        evaluation["score"] = clamp_score(evaluation.get("score"))
        evaluation["passed"] = bool(evaluation.get("passed")) and evaluation["score"] >= EVALUATOR_PASS_SCORE
        evaluation["use_output"] = bool(evaluation.get("use_output")) and evaluation["passed"]
        score = evaluation["score"]
        best_score = max(best_score, score)
        iteration["candidate_value"] = candidate_value
        iteration["evaluation"] = evaluation
        iteration["score"] = score
        iteration["passed"] = evaluation["passed"]
        iteration["observation"] = evaluation.get("verdict")
        iterations.append(iteration)
        logger.info("field agent evaluated agent=%s loop=%s score=%s passed=%s", agent_name, loop_index, score, evaluation["passed"])

        if (
            evaluation["passed"]
            and evaluation["use_output"]
            and value_matches_kind(candidate_value, spec["kind"])
            and not is_missing(candidate_value)
        ):
            accepted_value = candidate_value
            accepted_evaluation = evaluation
            stopping_reason = "score threshold reached"
            break
        if loop_index >= max_loops:
            break
        if score - previous_score < MIN_SCORE_IMPROVEMENT and loop_index > 1:
            stagnant += 1
        else:
            stagnant = 0
        previous_score = score
        previous_evaluation = evaluation
        if stagnant >= 2:
            stopping_reason = "score stopped improving"
            break

    final_score = clamp_score(accepted_evaluation.get("score") if accepted_evaluation else best_score or 1.0)
    accepted = bool(accepted_value is not None and not is_missing(accepted_value) and final_score >= EVALUATOR_PASS_SCORE)
    if not accepted:
        final_score = clamp_score(min(final_score, EVALUATOR_PASS_SCORE - 0.01))
        if stopping_reason == "score threshold reached":
            stopping_reason = "candidate value was empty or unsupported after evaluator review"
    trace = {
        "task_name": agent_name,
        "field": field,
        "purpose": f"Extract and validate canonical field: {field}",
        "mode": "react",
        "target_fields": [field],
        "system_prompt": system_prompt,
        "evaluator_prompt": evaluator_prompt,
        "field_specific_input": field_input[:2500],
        "loops": len(iterations),
        "final_score": final_score,
        "passed": accepted,
        "accepted": accepted,
        "discarded": not accepted,
        "stopping_reason": stopping_reason,
        "final_output": accepted_value,
        "iterations": iterations,
        "request_events": events,
    }
    good_examples = []
    if trace["accepted"]:
        good_examples.append(
            {
                "task_type": f"{TASK_TYPE}:{agent_name}",
                "score": final_score,
                "input": {
                    "system_prompt": system_prompt,
                    "field": field,
                    "field_specific_input": field_input[:2500],
                    "current_value": field_value(profile, field),
                },
                "output": accepted_value,
                "evaluation": accepted_evaluation or {},
            }
        )
    return accepted_value, trace, good_examples, errors


def set_field(profile: dict[str, Any], field: str, value: Any) -> None:
    profile[field] = value


def env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def run_agentic_llmops(
    profile: dict[str, Any],
    source_texts: list[str],
    validation_errors: list[str] | None = None,
    memory_examples: list[dict[str, Any]] | None = None,
    max_loops: int | None = None,
    score_threshold: float | None = None,
    default_region: str = "US",
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    validation_errors = validation_errors or []
    memory_examples = memory_examples or []
    max_loops = min(max_loops or env_int("AGENT_MAX_LOOPS", DEFAULT_MAX_LOOPS), 5)
    threshold = score_threshold or EVALUATOR_PASS_SCORE
    working = copy.deepcopy(profile)
    errors: list[str] = []
    request_events: list[dict[str, Any]] = []
    good_examples_to_store: list[dict[str, Any]] = []
    llm_available = bool(configured_gemini_keys())
    logger.info(
        "field gateway start fields=%s llm_available=%s max_loops=%s threshold=%s memory_examples=%s",
        len(CANONICAL_FIELD_SPECS),
        llm_available,
        max_loops,
        threshold,
        len(memory_examples),
    )

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "task_type": TASK_TYPE,
        "mode": "field-level-react-agents" if llm_available else "field-level-deterministic-gateway",
        "model": gemini_model(),
        "confidence_thresholds": {"accept": ACCEPT_CONFIDENCE, "discard": DISCARD_CONFIDENCE, "evaluator_pass": threshold},
        "memory_examples_used": len(memory_examples),
        "task_traces": [],
        "request_events": request_events,
        "input_excerpt": trace_input_excerpt(profile, source_texts),
    }

    if llm_available and needs_canonical_mapping(working):
        logger.info("canonical mapping agent start")
        working, mapping_trace, mapping_examples, mapping_errors = run_canonical_mapping_agent(
            working,
            source_texts,
            memory_examples,
            default_region,
            max_loops,
        )
        errors.extend(mapping_errors)
        request_events.extend(mapping_trace.get("request_events", []))
        diagnostics["task_traces"].append(mapping_trace)
        good_examples_to_store.extend(mapping_examples)
        logger.info(
            "canonical mapping agent done accepted=%s score=%s applied=%s errors=%s",
            mapping_trace.get("accepted"),
            mapping_trace.get("final_score"),
            (mapping_trace.get("final_output") or {}).get("applied_fields"),
            len(mapping_errors),
        )

    for spec in CANONICAL_FIELD_SPECS:
        field = spec["field"]
        agent = spec["agent"]
        value = field_value(working, field)
        confidence = field_confidence(working, field)
        field_input = field_relevant_input(field, working, source_texts)
        trace = {
            "task_name": f"{field}_deterministic_gateway",
            "agent_name": agent,
            "field": field,
            "purpose": f"Validate canonical field: {field}",
            "mode": "deterministic",
            "target_fields": [field],
            "deterministic_confidence": confidence,
            "system_prompt": "Deterministic field gateway. No LLM generation prompt was used.",
            "evaluator_prompt": "Local deterministic evaluator unless Gemini is available for medium-confidence review.",
            "field_specific_input": field_input[:2500],
            "loops": 1,
            "iterations": [],
            "request_events": [],
        }
        logger.info("field gateway field=%s confidence=%s missing=%s", field, confidence, is_missing(value))

        if field in {"others", "other_sections"}:
            evaluation = {
                "score": 10.0,
                "passed": True,
                "use_output": True,
                "verdict": "Fallback container preserved; canonical mapper may move values but failed mappings remain here.",
                "issues": [],
                "improvement_hint": "",
            }
            trace.update(
                {
                    "status": "preserved_fallback_container",
                    "stopping_reason": "fallback container is preserved",
                    "final_score": 10.0,
                    "passed": True,
                    "accepted": True,
                    "final_output": value,
                }
            )
            needs_agent = False
        elif is_missing(value):
            trace.update({"status": "missing", "stopping_reason": "missing deterministic value", "final_score": 1.0, "passed": False, "accepted": False})
            needs_agent = True
            evaluation = {"score": 1.0, "passed": False, "use_output": False, "verdict": "Field missing from deterministic output.", "issues": [], "improvement_hint": "Run field agent if available."}
        elif confidence < DISCARD_CONFIDENCE:
            clear_field(working, field, spec["kind"])
            trace.update({"status": "discarded_low_confidence", "stopping_reason": "deterministic confidence below discard threshold", "final_score": 1.0, "passed": False, "accepted": False})
            needs_agent = True
            evaluation = {"score": 1.0, "passed": False, "use_output": False, "verdict": "Discarded low-confidence deterministic value.", "issues": [], "improvement_hint": "Run field agent if available."}
        elif confidence >= ACCEPT_CONFIDENCE:
            trace.update({"status": "accepted_high_confidence", "stopping_reason": "deterministic confidence met accept threshold", "final_score": 10.0, "passed": True, "accepted": True, "final_output": value})
            needs_agent = False
            evaluation = {"score": 10.0, "passed": True, "use_output": True, "verdict": "Accepted high-confidence deterministic value.", "issues": [], "improvement_hint": ""}
        else:
            if llm_available:
                evaluation, eval_errors, eval_events, evaluator_prompt = run_llm_evaluator(spec, value, field_input, working, default_region)
                errors.extend(eval_errors)
                trace["request_events"].extend(eval_events)
                request_events.extend(eval_events)
                trace["evaluator_prompt"] = evaluator_prompt
                trace["mode"] = "deterministic_with_llm_evaluation"
            else:
                evaluation = local_evaluate_field(field, value, confidence, working)
            accepted = bool(evaluation.get("use_output")) and float(evaluation.get("score", 0)) >= threshold
            if not accepted:
                clear_field(working, field, spec["kind"])
            trace.update(
                {
                    "status": "accepted_after_evaluation" if accepted else "needs_field_agent",
                    "stopping_reason": "deterministic evaluator passed" if accepted else "deterministic evaluator failed",
                    "final_score": clamp_score(evaluation.get("score")),
                    "passed": accepted,
                    "accepted": accepted,
                    "final_output": value if accepted else None,
                }
            )
            needs_agent = not accepted

        trace["iterations"].append(
            {
                "loop": 1,
                "action": "deterministic_field_gateway",
                "observation": trace.get("stopping_reason"),
                "rationale_summary": "Deterministic field confidence was checked against accept/discard thresholds, then evaluated when needed.",
                "evaluation": evaluation,
                "score": trace.get("final_score"),
                "passed": trace.get("passed"),
                "request_events": trace.get("request_events", []),
            }
        )

        if needs_agent and llm_available:
            accepted_value, agent_trace, good_examples, agent_errors = run_field_agent(
                spec,
                working,
                source_texts,
                memory_examples,
                default_region,
                evaluation,
                max_loops,
            )
            errors.extend(agent_errors)
            request_events.extend(agent_trace.get("request_events", []))
            diagnostics["task_traces"].append(trace)
            diagnostics["task_traces"].append(agent_trace)
            good_examples_to_store.extend(good_examples)
            if agent_trace.get("accepted"):
                set_field(working, field, accepted_value)
            else:
                clear_field(working, field, spec["kind"])
            continue

        diagnostics["task_traces"].append(trace)

    scores = [float(item.get("final_score") or 0.0) for item in diagnostics["task_traces"] if item.get("mode") != "react" or item.get("accepted")]
    final_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    passed = sum(1 for item in diagnostics["task_traces"] if item.get("passed"))
    diagnostics["final_score"] = final_score
    diagnostics["stopping_reason"] = "completed field-level gateway"
    diagnostics["final_evaluation"] = {
        "score": final_score,
        "passed": True,
        "verdict": f"{passed}/{len(diagnostics['task_traces'])} field checks or agents passed.",
    }
    diagnostics["output_preview"] = output_preview(working)
    diagnostics["request_events"] = request_events
    diagnostics["good_examples"] = good_examples_to_store
    logger.info("field gateway done final_score=%s traces=%s good_examples=%s errors=%s", final_score, len(diagnostics["task_traces"]), len(good_examples_to_store), len(errors))
    return working, diagnostics, errors
