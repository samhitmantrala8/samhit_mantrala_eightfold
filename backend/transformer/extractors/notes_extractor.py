from __future__ import annotations

import re
from pathlib import Path

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.contact import EMAIL_RE, PHONE_RE, URL_RE, classify_link, normalize_url
from backend.transformer.normalizers.skills import extract_skills_from_text


NAME_RE = re.compile(r"(?im)^(?:candidate|name)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,80})$")
HEADLINE_RE = re.compile(r"(?im)^headline\s*[:\-]\s*(.{4,160})$")
YEARS_RE = re.compile(r"\b(\d{1,2})(?:\+)?\s+years?\b", re.IGNORECASE)
ROLE_RE = re.compile(
    r"\b(?:worked|working|experience)\s+(?:as\s+)?(?P<title>[A-Za-z0-9 /+.-]{3,60})\s+at\s+(?P<company>[A-Z][A-Za-z0-9 &.-]{2,80})",
    re.IGNORECASE,
)


def extract_notes(path: Path, use_llm: bool = False) -> ExtractionBundle:
    facts: list[ExtractedFact] = []
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="ignore")
    except Exception as exc:
        return ExtractionBundle([], [f"{path.name}: failed to read text: {exc}"])

    source = f"notes:{path.name}"

    for match in NAME_RE.finditer(text):
        facts.append(ExtractedFact("full_name", match.group(1).strip(), source, "notes-regex:name", 0.68, match.group(0)))
    for match in HEADLINE_RE.finditer(text):
        facts.append(ExtractedFact("headline", match.group(1).strip(), source, "notes-regex:headline", 0.65, match.group(0)))
    for match in EMAIL_RE.finditer(text):
        facts.append(ExtractedFact("emails", match.group(0), source, "notes-regex:email", 0.82, match.group(0)))
    for match in PHONE_RE.finditer(text):
        facts.append(ExtractedFact("phones", match.group(0), source, "notes-regex:phone", 0.72, match.group(0)))
    for match in URL_RE.finditer(text):
        url = normalize_url(match.group(0))
        if url:
            facts.append(ExtractedFact(classify_link(url), url, source, "notes-regex:url", 0.78, match.group(0)))
    for match in YEARS_RE.finditer(text):
        facts.append(ExtractedFact("years_experience", int(match.group(1)), source, "notes-regex:years", 0.62, match.group(0)))
    for match in ROLE_RE.finditer(text):
        facts.append(
            ExtractedFact(
                "experience",
                {
                    "company": match.group("company").strip(),
                    "title": match.group("title").strip(),
                    "start": None,
                    "end": None,
                    "summary": match.group(0),
                },
                source,
                "notes-regex:experience",
                0.58,
                match.group(0),
            )
        )

    for skill, confidence, evidence in extract_skills_from_text(text):
        facts.append(ExtractedFact("skills", {"name": skill}, source, "notes-skill-alias-fuzzy", confidence, evidence))

    if use_llm:
        from backend.transformer.extractors.llm_extractor import extract_text_with_llm

        llm_bundle = extract_text_with_llm(text, source)
        facts.extend(llm_bundle.facts)
        errors.extend(llm_bundle.errors)

    return ExtractionBundle(facts, errors)

