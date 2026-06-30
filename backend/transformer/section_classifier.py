from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from backend.transformer.normalizers.skills import canonicalize_skill, extract_skills_from_text


SECTION_ALIASES = {
    "education": [
        "education",
        "academics",
        "academic background",
        "academic details",
        "scholastic record",
        "education details",
        "qualifications",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "professional background",
        "employment history",
        "work history",
        "internships",
    ],
    "skills": [
        "skills",
        "skills summary",
        "technical skills",
        "technical strengths",
        "tools and platforms",
        "toolkit",
        "technologies",
        "tech stack",
    ],
    "projects": [
        "projects",
        "selected projects",
        "project work",
        "selected project work",
        "applied builds",
        "portfolio projects",
    ],
    "achievements": [
        "achievements",
        "awards",
        "honors",
        "recognition",
        "achievements and recognition",
    ],
    "links": [
        "links",
        "profiles",
        "online profiles",
        "social links",
    ],
    "online_coding_profile": [
        "Online Coding Profile",
        "Online Coding Profile metadata",
        "coding profiles",
        "programming profiles",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses",
    ],
    "extracurriculars": [
        "extra curriculars",
        "extracurriculars",
        "activities",
        "leadership",
    ],
}

CANONICAL_SECTION_LABELS = {
    "education": "Education",
    "experience": "Experience",
    "skills": "Skills",
    "projects": "Projects",
    "achievements": "Achievements",
    "links": "Links",
    "online_coding_profile": "Online Coding Profile",
    "certifications": "Certifications",
    "extracurriculars": "Extracurriculars",
}

ACTION_VERB_RE = re.compile(
    r"^(built|created|developed|implemented|used|deployed|worked|led|managed|reduced|optimized|optimised|designed)\b",
    re.IGNORECASE,
)
CONTACT_RE = re.compile(r"\b(email|mobile|phone|github|linkedin|kaggle|leetcode|codeforces)\s*:", re.IGNORECASE)
DATE_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}|\b(?:19|20)\d{2}\b", re.IGNORECASE)
DEGREE_RE = re.compile(r"\b(bachelor|master|b\.?tech|m\.?tech|degree|cgpa|university|institute)\b", re.IGNORECASE)
NON_SECTION_SKILL_PHRASE_RE = re.compile(r"\b(project management|product management|people management)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SectionClassification:
    text: str
    kind: str
    canonical_section: str | None
    section_score: float
    content_score: float
    reasons: list[str]


def normalize_line(line: str) -> str:
    line = line.replace("\u2022", " ").replace("â€¢", " ")
    line = re.sub(r"^[\s\-*?]+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def words(line: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9+#.]+", line)


def best_section_match(line: str) -> tuple[str | None, float, str | None]:
    cleaned = normalize_line(line).strip(":")
    best_section = None
    best_alias = None
    best_score = 0.0
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            score = fuzz.token_set_ratio(cleaned.lower(), alias.lower()) / 100
            if score > best_score:
                best_section = section
                best_alias = alias
                best_score = score
    return best_section, best_score, best_alias


def is_inline_field(line: str) -> bool:
    if ":" not in line:
        return False
    prefix, suffix = line.split(":", 1)
    if not suffix.strip():
        return False
    if len(words(prefix)) <= 4:
        return True
    return False


def following_context_score(section: str | None, next_lines: list[str]) -> tuple[float, list[str]]:
    if not section or not next_lines:
        return 0.0, []
    joined = " ".join(normalize_line(line) for line in next_lines[:3])
    reasons: list[str] = []
    score = 0.0
    if section == "education" and DEGREE_RE.search(joined):
        score += 0.2
        reasons.append("following lines contain education terms")
    if section == "experience" and (DATE_RE.search(joined) or ACTION_VERB_RE.search(joined)):
        score += 0.18
        reasons.append("following lines look like work history")
    if section == "skills" and (extract_skills_from_text(joined) or joined.count(",") >= 2):
        score += 0.18
        reasons.append("following lines look like skill lists")
    if section == "projects" and re.search(r"\b(project|built|developed|app|platform|predictor)\b", joined, re.IGNORECASE):
        score += 0.18
        reasons.append("following lines look like project content")
    if section == "achievements" and re.search(r"\b(challenge|cup|rank|secured|award|top)\b", joined, re.IGNORECASE):
        score += 0.18
        reasons.append("following lines look like achievements")
    if section == "online_coding_profile" and re.search(r"\b(codeforces|leetcode|kaggle|handle|rating)\b", joined, re.IGNORECASE):
        score += 0.18
        reasons.append("following lines contain coding profile terms")
    if section == "links" and CONTACT_RE.search(joined):
        score += 0.18
        reasons.append("following lines contain profile/contact links")
    return min(score, 0.25), reasons


def classify_line(line: str, next_lines: list[str] | None = None) -> SectionClassification:
    text = normalize_line(line)
    next_lines = next_lines or []
    token_count = len(words(text))
    reasons: list[str] = []
    if not text:
        return SectionClassification(text, "content", None, 0.0, 0.0, ["empty line"])

    section, section_similarity, alias = best_section_match(text)
    section_score = 0.0
    content_score = 0.0

    if token_count <= 6:
        section_score += 0.22
        reasons.append("short line")
    elif token_count >= 10:
        content_score += 0.22
        reasons.append("long line")

    if text.endswith(":") and token_count <= 6:
        section_score += 0.1
        reasons.append("heading-style trailing colon")

    if not re.search(r"[.!?]$", text):
        section_score += 0.08
        reasons.append("no sentence punctuation")
    else:
        content_score += 0.12
        reasons.append("sentence punctuation")

    if section_similarity >= 0.92:
        section_score += 0.36
        reasons.append(f"strong section alias match: {alias}")
    elif section_similarity >= 0.78:
        section_score += 0.24
        reasons.append(f"possible section alias match: {alias}")

    context_score, context_reasons = following_context_score(section, next_lines)
    section_score += context_score
    reasons.extend(context_reasons)

    skill_match = canonicalize_skill(text)
    skill_hits = extract_skills_from_text(text)
    if skill_match:
        content_score += 0.48
        section_score -= 0.24
        reasons.append(f"known skill phrase: {skill_match[0]}")
    elif skill_hits:
        content_score += 0.22
        reasons.append("contains known skill aliases")

    if NON_SECTION_SKILL_PHRASE_RE.search(text):
        content_score += 0.36
        section_score -= 0.2
        reasons.append("skill-like management phrase")

    if "," in text or ";" in text:
        content_score += 0.2
        reasons.append("list-like content")

    if is_inline_field(text):
        content_score += 0.28
        section_score -= 0.14
        reasons.append("inline field with value after colon")

    if CONTACT_RE.search(text):
        content_score += 0.26
        reasons.append("contact/profile field")

    if ACTION_VERB_RE.search(text):
        content_score += 0.24
        section_score -= 0.12
        reasons.append("starts with action verb")

    if DATE_RE.search(text) or DEGREE_RE.search(text):
        content_score += 0.16
        reasons.append("contains date or degree detail")

    section_score = max(0.0, min(1.0, section_score))
    content_score = max(0.0, min(1.0, content_score))

    if (skill_match or NON_SECTION_SKILL_PHRASE_RE.search(text)) and content_score >= 0.36 and section_score < 0.5:
        kind = "content"
    elif section_score >= 0.62 and section_score - content_score >= 0.14:
        kind = "section"
    elif content_score >= 0.52 and content_score >= section_score:
        kind = "content"
    elif section_score >= 0.56 and content_score < 0.5:
        kind = "section"
    else:
        kind = "ambiguous"

    return SectionClassification(
        text=text,
        kind=kind,
        canonical_section=section if kind == "section" and section_similarity >= 0.68 else None,
        section_score=round(section_score, 3),
        content_score=round(content_score, 3),
        reasons=reasons,
    )
