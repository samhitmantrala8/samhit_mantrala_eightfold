from __future__ import annotations

import csv
import re
from pathlib import Path

from rapidfuzz import fuzz

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.contact import classify_link, normalize_url
from backend.transformer.normalizers.skills import extract_skills_from_text


FIELD_ALIASES = {
    "name": ["name", "full name", "full_name", "candidate name", "candidate full name"],
    "email": ["email", "email address", "email_id", "primary email", "primary_email", "mail"],
    "phone": ["phone", "phone number", "mobile", "mobile number", "contact number"],
    "company": ["current company", "company", "employer", "current employer", "organization"],
    "title": ["title", "current title", "job title", "role", "position", "designation"],
    "github": ["github", "github url", "github profile", "github link"],
    "linkedin": ["linkedin", "linkedin url", "linkedin profile", "linkedin link"],
    "portfolio": ["portfolio", "website", "personal website", "portfolio url"],
    "skills": ["skills", "skill set", "technical skills", "tech stack", "technologies"],
    "codeforces": ["codeforces", "cf handle", "codeforces handle", "codeforces profile"],
}


def normalize_header(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def best_column(row: dict[str, str], aliases: list[str]) -> tuple[str | None, float]:
    best_key = None
    best_score = 0.0
    normalized_aliases = [normalize_header(alias) for alias in aliases]
    for key in row:
        header = normalize_header(key)
        for alias in normalized_aliases:
            score = max(fuzz.ratio(header, alias), fuzz.token_set_ratio(header, alias)) / 100
            if score > best_score:
                best_key = key
                best_score = score
    return best_key, best_score


def first_present(row: dict[str, str], field: str) -> tuple[str | None, str | None, float]:
    key, score = best_column(row, FIELD_ALIASES[field])
    if key and score >= 0.82:
        value = row.get(key)
        if value and value.strip():
            return value.strip(), key, score
    return None, None, 0.0


def extract_csv(path: Path) -> ExtractionBundle:
    facts: list[ExtractedFact] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return ExtractionBundle([], [f"{path.name}: CSV has no header row"])
            for index, row in enumerate(reader, start=1):
                source = f"csv:{path.name}#row{index}"
                name, name_key, name_score = first_present(row, "name")
                email, email_key, email_score = first_present(row, "email")
                phone, phone_key, phone_score = first_present(row, "phone")
                company, company_key, company_score = first_present(row, "company")
                title, title_key, title_score = first_present(row, "title")
                github, github_key, github_score = first_present(row, "github")
                linkedin, linkedin_key, linkedin_score = first_present(row, "linkedin")
                portfolio, portfolio_key, portfolio_score = first_present(row, "portfolio")
                skills, skills_key, _skills_score = first_present(row, "skills")
                codeforces, _codeforces_key, _codeforces_score = first_present(row, "codeforces")

                if name:
                    facts.append(ExtractedFact("full_name", name, source, f"csv-column:{name_key}", min(0.92, 0.72 + name_score * 0.2)))
                if email:
                    facts.append(ExtractedFact("emails", email, source, f"csv-column:{email_key}", min(0.95, 0.74 + email_score * 0.21)))
                if phone:
                    facts.append(ExtractedFact("phones", phone, source, f"csv-column:{phone_key}", min(0.92, 0.72 + phone_score * 0.2)))
                if company or title:
                    facts.append(
                        ExtractedFact(
                            "experience",
                            {"company": company, "title": title, "start": None, "end": None, "summary": None},
                            source,
                            "csv-current-role",
                            min(0.82, 0.62 + max(company_score, title_score) * 0.18),
                        )
                    )
                    if title:
                        facts.append(ExtractedFact("headline", title, source, f"csv-column:{title_key}", 0.72))
                for raw_url, key, score in [(github, github_key, github_score), (linkedin, linkedin_key, linkedin_score), (portfolio, portfolio_key, portfolio_score)]:
                    url = normalize_url(raw_url)
                    if url:
                        facts.append(ExtractedFact(classify_link(url), url, source, f"csv-column:{key}", min(0.9, 0.68 + score * 0.2)))
                if codeforces:
                    value = codeforces.strip()
                    url = normalize_url(value) if "codeforces.com" in value.lower() else f"https://codeforces.com/profile/{value}"
                    facts.append(ExtractedFact("links.other", url, source, "csv-column:codeforces", 0.74))
                if skills:
                    for skill, confidence, evidence in extract_skills_from_text(skills):
                        facts.append(ExtractedFact("skills", {"name": skill}, source, f"csv-column:{skills_key}", min(0.86, confidence), evidence))
    except Exception as exc:
        errors.append(f"{path.name}: failed to parse CSV: {exc}")
    return ExtractionBundle(facts, errors)
