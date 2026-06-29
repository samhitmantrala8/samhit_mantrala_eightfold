from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from backend.transformer.extractors.llm_extractor import configured_keys


SUMMARY_SYSTEM_PROMPT = """You write concise candidate profile summaries for a data transformation system.
Use only facts present in the provided canonical profile and source snippets. Do not invent missing details.
Return exactly one paragraph, no bullets, no markdown."""
BULLET_MARKERS = ("\u00c2\u2022", "\u0100\u2022", "\u0095", "\u2022")


def normalize_bullet_markers(text: str | None) -> str:
    if not text:
        return ""
    for marker in BULLET_MARKERS:
        text = text.replace(marker, " \u2022 ")
    return re.sub(r"\s+", " ", text).strip()


def clean_inline_text(text: str | None) -> str:
    text = normalize_bullet_markers(text)
    text = text.replace("\u2022", " ")
    return re.sub(r"\s+", " ", text).strip(" -:")


def extract_resume_sections(texts: list[str]) -> dict[str, str]:
    section_names = [
        "Education",
        "Experience",
        "Projects",
        "Achievements",
        "Skills Summary",
        "Skills",
        "Extra Curriculars",
        "Extracurriculars",
        "Certifications",
    ]
    sections: dict[str, str] = {}
    joined = "\n".join(texts)
    pattern = re.compile(
        rf"(?im)^({'|'.join(re.escape(name) for name in section_names)})\s*$"
    )
    matches = list(pattern.finditer(joined))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        body = normalize_bullet_markers(joined[start:end].strip())
        if body:
            body = re.sub(r"\s+", " ", body)
            sections[match.group(1)] = body[:1800]
    return sections


def summary_context(profile: dict[str, Any], source_texts: list[str]) -> dict[str, Any]:
    return {
        "full_name": profile.get("full_name"),
        "headline": profile.get("headline"),
        "emails": profile.get("emails"),
        "phones": profile.get("phones"),
        "links": profile.get("links"),
        "education": profile.get("education"),
        "experience": profile.get("experience"),
        "skills": [skill.get("name") for skill in profile.get("skills", [])],
        "sections": extract_resume_sections(source_texts),
    }


def deterministic_summary(profile: dict[str, Any], source_texts: list[str]) -> str:
    name = clean_inline_text(profile.get("full_name")) or "The candidate"
    education = profile.get("education") or []
    experience = profile.get("experience") or []
    skills = [skill.get("name") for skill in profile.get("skills", [])[:14]]
    sections = extract_resume_sections(source_texts)

    education_text = ""
    if education:
        first = education[0]
        education_text = f" educated at {clean_inline_text(first.get('institution'))}"
        if first.get("degree"):
            education_text += f" with {clean_inline_text(first.get('degree'))}"
        if first.get("field"):
            education_text += f" in {clean_inline_text(first.get('field'))}"
        if first.get("cgpa"):
            education_text += f" and CGPA {clean_inline_text(first.get('cgpa'))}"

    experience_text = ""
    if experience:
        roles = []
        for item in experience[:3]:
            role = clean_inline_text(item.get("role") or item.get("title"))
            company = clean_inline_text(item.get("company"))
            if role and company:
                roles.append(f"{role} at {company}")
        if roles:
            experience_text = " Experience includes " + ", ".join(roles) + "."

    project_text = " Projects and achievements are present in the source resume." if any(
        key in sections for key in ("Projects", "Achievements")
    ) else ""
    skill_text = f" Key skills include {', '.join(skills)}." if skills else ""
    summary = f"{name} is a candidate{education_text}.{experience_text}{project_text}{skill_text}"
    return re.sub(r"\s+", " ", summary).strip()


def generate_profile_summary(profile: dict[str, Any], source_texts: list[str]) -> tuple[str, dict[str, Any], list[str]]:
    keys = configured_keys()
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    context = summary_context(profile, source_texts)
    prompt = {
        "instruction": (
            "Write one paragraph of 90-130 words summarizing the candidate. Cover education, experience, "
            "projects, achievements, skills, and extra/other sections when present. Mention CGPA, role, "
            "company, location, and duration when available. Do not include unsupported claims."
        ),
        "candidate_context": context,
    }

    errors: list[str] = []
    if keys:
        for key in keys:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:5177",
                        "X-Title": "Eightfold Candidate Transformer",
                    },
                    json={
                        "model": model,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(prompt)},
                        ],
                    },
                    timeout=25,
                )
                if response.status_code in {429, 402, 403}:
                    errors.append(f"profile_summary: OpenRouter key rejected or rate limited with status {response.status_code}")
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                content = re.sub(r"\s+", " ", content)
                if content:
                    return content, {"source": "openrouter", "method": "llm-profile-summary", "confidence": 0.72}, errors
            except Exception as exc:
                errors.append(f"profile_summary: LLM summary failed: {exc}")

    fallback = deterministic_summary(profile, source_texts)
    if not keys:
        errors.append("profile_summary: OpenRouter keys not configured; used deterministic fallback")
    return fallback, {"source": "local-fallback", "method": "deterministic-profile-summary", "confidence": 0.42}, errors
