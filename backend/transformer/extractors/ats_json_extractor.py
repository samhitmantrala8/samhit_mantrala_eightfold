from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.contact import EMAIL_RE, PHONE_RE, normalize_url
from backend.transformer.normalizers.dates import normalize_month
from backend.transformer.normalizers.skills import canonicalize_skill


NAME_KEYS = {"name", "fullname", "full_name", "candidate_name", "candidatefullname", "personname", "profilename"}
EMAIL_KEYS = {"email", "emailaddress", "email_id", "mail", "primaryemail"}
PHONE_KEYS = {"phone", "phonenumber", "mobile", "mobilenumber", "contactnumber", "telephone"}
HEADLINE_KEYS = {"headline", "title", "currenttitle", "currentrole", "summarytitle", "professionalheadline"}
EXPERIENCE_KEYS = {"experience", "experiences", "workhistory", "workexperience", "work_experience", "employment", "jobs", "professionalexperience"}
EDUCATION_KEYS = {"education", "educations", "academics", "academic", "academic_history", "qualifications", "schooling"}
PROJECT_KEYS = {"projects", "project", "portfolio_projects", "projectexperience", "project_experience"}
SKILL_KEYS = {"skills", "technicalskills", "technical_skills", "techstack", "tech_stack", "technologies", "tools", "frameworks", "databases", "programminglanguages"}
ACHIEVEMENT_KEYS = {"achievements", "awards", "honors", "accomplishments", "recognition"}
LINK_KEYS = {"github", "linkedin", "portfolio", "website", "url", "profileurl", "profile_url", "links"}
LOCATION_KEYS = {"location", "address", "city", "region", "state", "country"}


def norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def norm_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def compact(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def leaf_items(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        items: list[tuple[tuple[str, ...], Any]] = []
        for key, child in value.items():
            items.extend(leaf_items(child, (*path, str(key))))
        return items
    if isinstance(value, list):
        items = []
        for index, child in enumerate(value):
            items.extend(leaf_items(child, (*path, str(index))))
        return items
    return [(path, value)]


def top_level_key_for_path(path: tuple[str, ...]) -> str | None:
    return path[0] if path else None


def key_matches(path: tuple[str, ...], aliases: set[str]) -> bool:
    normalized = [norm_key(part) for part in path if not part.isdigit()]
    joined = norm_key(" ".join(path))
    normalized_aliases = {norm_key(alias) for alias in aliases}
    return any(part in normalized_aliases for part in normalized) or any(alias in joined for alias in normalized_aliases)


def terminal_key_matches(path: tuple[str, ...], aliases: set[str]) -> bool:
    for part in reversed(path):
        if not part.isdigit():
            return norm_key(part) in {norm_key(alias) for alias in aliases}
    return False


def collect_strings(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(collect_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(collect_strings(item))
        return strings
    return [str(value)]


def split_values(value: Any) -> list[str]:
    raw_items = collect_strings(value)
    parts: list[str] = []
    for raw in raw_items:
        chunks = re.split(r"[,;|\n\r]+", raw)
        for chunk in chunks:
            cleaned = re.sub(r"\s+", " ", chunk).strip(" -")
            if cleaned:
                parts.append(cleaned)
    return parts


def first_value(item: dict[str, Any], aliases: list[str]) -> Any:
    lookup = {norm_key(key): value for key, value in item.items()}
    for alias in aliases:
        value = lookup.get(norm_key(alias))
        if value not in (None, "", [], {}):
            return value
    return None


def first_key_value(item: dict[str, Any], aliases: list[str]) -> tuple[str | None, Any]:
    alias_keys = {norm_key(alias) for alias in aliases}
    for key, value in item.items():
        if norm_key(str(key)) in alias_keys and value not in (None, "", [], {}):
            return str(key), value
    return None, None


def consume_if_present(consumed: set[str], *keys: str | None) -> None:
    for key in keys:
        if key:
            consumed.add(key)


def as_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    return value if isinstance(value, list) else [value]


def top_level_section(data: dict[str, Any], aliases: set[str]) -> tuple[str | None, Any]:
    normalized_aliases = {norm_key(alias) for alias in aliases}
    for key, value in data.items():
        normalized_key = norm_key(key)
        if normalized_key in normalized_aliases or any(alias in normalized_key for alias in normalized_aliases):
            return key, value
    return None, None


def add_skill_facts(facts: list[ExtractedFact], value: Any, source: str, method: str, confidence: float, evidence: str | None = None) -> None:
    seen: set[str] = set()
    for raw in split_values(value):
        canonical = canonicalize_skill(raw)
        if canonical and canonical[0] not in seen:
            facts.append(ExtractedFact("skills", {"name": canonical[0]}, source, method, min(confidence, canonical[1]), evidence or raw))
            seen.add(canonical[0])


def extract_location_facts(facts: list[ExtractedFact], data: dict[str, Any], source: str, consumed: set[str]) -> None:
    key, location = top_level_section(data, LOCATION_KEYS)
    if key:
        consumed.add(key)
    if isinstance(location, dict):
        city = first_value(location, ["city", "town"])
        region = first_value(location, ["region", "state", "province"])
        country = first_value(location, ["country", "nation"])
        if city:
            facts.append(ExtractedFact("location.city", str(city), source, "ats-json:location.city", 0.82, compact(location)))
        if region:
            facts.append(ExtractedFact("location.region", str(region), source, "ats-json:location.region", 0.82, compact(location)))
        if country:
            facts.append(ExtractedFact("location.country", str(country), source, "ats-json:location.country", 0.82, compact(location)))
    else:
        for path, value in leaf_items(data):
            if value in (None, ""):
                continue
            if key_matches(path, {"city"}):
                facts.append(ExtractedFact("location.city", str(value), source, "ats-json:location.city", 0.74, norm_path(path)))
            elif key_matches(path, {"state", "region", "province"}):
                facts.append(ExtractedFact("location.region", str(value), source, "ats-json:location.region", 0.74, norm_path(path)))
            elif key_matches(path, {"country", "nation"}):
                facts.append(ExtractedFact("location.country", str(value), source, "ats-json:location.country", 0.74, norm_path(path)))


def extract_experience_facts(facts: list[ExtractedFact], data: dict[str, Any], source: str, consumed: set[str]) -> None:
    key, section = top_level_section(data, EXPERIENCE_KEYS)
    if key:
        consumed.add(key)
    for index, item in enumerate(as_list(section), start=1):
        if isinstance(item, str):
            cleaned = re.sub(r"\s+", " ", item).strip()
            if cleaned:
                facts.append(
                    ExtractedFact(
                        "experience",
                        {"company": cleaned, "title": None, "role": None, "location": None, "duration": None, "start": None, "end": None, "summary": None},
                        f"{source}#experience{index}",
                        "ats-json:experience-string",
                        0.72,
                        cleaned,
                    )
                )
            continue
        if not isinstance(item, dict):
            continue
        value = {
            "company": first_value(item, ["company", "companyName", "employer", "organization", "org", "company_name"]),
            "title": first_value(item, ["title", "jobTitle", "position", "role", "designation"]),
            "role": first_value(item, ["role", "title", "jobTitle", "position", "designation"]),
            "location": first_value(item, ["location", "city", "place"]),
            "duration": first_value(item, ["duration", "period", "dateRange", "date_range"]),
            "start": normalize_month(str(first_value(item, ["start", "startDate", "from", "begin"]) or "")),
            "end": normalize_month(str(first_value(item, ["end", "endDate", "to", "until"]) or "")),
            "summary": first_value(item, ["summary", "description", "responsibilities", "details", "bullets"]),
        }
        if value["company"] or value["title"] or value["role"]:
            facts.append(ExtractedFact("experience", value, f"{source}#experience{index}", "ats-json:experience", 0.84, compact(item)))

    company_key, company = first_key_value(data, ["company", "companyName", "employer", "organization", "org", "workplace"])
    title_key, title = first_key_value(data, ["title", "jobTitle", "position", "role", "designation", "currentRole"])
    duration_key, duration = first_key_value(data, ["duration", "period", "dateRange", "date_range", "workDuration"])
    if company or title:
        value = {
            "company": company,
            "title": title,
            "role": title,
            "location": first_value(data, ["workLocation", "companyLocation", "officeLocation"]),
            "duration": duration,
            "start": normalize_month(str(first_value(data, ["start", "startDate", "from", "begin"]) or "")),
            "end": normalize_month(str(first_value(data, ["end", "endDate", "to", "until"]) or "")),
            "summary": first_value(data, ["workSummary", "responsibilities", "jobDescription"]),
        }
        facts.append(ExtractedFact("experience", value, f"{source}#experience_flat", "ats-json:experience-flat-alias", 0.78, compact(value)))
        consume_if_present(consumed, company_key, title_key, duration_key)


def extract_education_facts(facts: list[ExtractedFact], data: dict[str, Any], source: str, consumed: set[str]) -> None:
    key, section = top_level_section(data, EDUCATION_KEYS)
    if key:
        consumed.add(key)
    for index, item in enumerate(as_list(section), start=1):
        if isinstance(item, str):
            cleaned = re.sub(r"\s+", " ", item).strip()
            if cleaned:
                facts.append(
                    ExtractedFact(
                        "education",
                        {"institution": cleaned, "degree": None, "field": None, "end_year": None, "cgpa": None},
                        f"{source}#education{index}",
                        "ats-json:education-string",
                        0.72,
                        cleaned,
                    )
                )
            continue
        if not isinstance(item, dict):
            continue
        value = {
            "institution": first_value(item, ["institution", "school", "university", "college", "institute", "instituteName", "schoolName"]),
            "degree": first_value(item, ["degree", "qualification", "program"]),
            "field": first_value(item, ["field", "major", "branch", "specialization", "discipline"]),
            "end_year": first_value(item, ["end_year", "endYear", "graduationYear", "year", "passingYear"]),
            "cgpa": first_value(item, ["cgpa", "gpa", "grade", "score"]),
        }
        if value["institution"] or value["degree"]:
            facts.append(ExtractedFact("education", value, f"{source}#education{index}", "ats-json:education", 0.82, compact(item)))

    institution_key, institution = first_key_value(data, ["institution", "school", "university", "college", "institute", "collegeName", "schoolName", "instituteName"])
    degree_key, degree = first_key_value(data, ["degree", "qualification", "program", "course"])
    field_key, field = first_key_value(data, ["field", "major", "branch", "specialization", "discipline", "stream"])
    year_key, end_year = first_key_value(data, ["end_year", "endYear", "graduationYear", "year", "passingYear", "passoutYear"])
    cgpa_key, cgpa = first_key_value(data, ["cgpa", "gpa", "grade", "score"])
    if institution or degree or field or cgpa:
        value = {
            "institution": institution,
            "degree": degree,
            "field": field,
            "end_year": end_year,
            "cgpa": cgpa,
        }
        facts.append(ExtractedFact("education", value, f"{source}#education_flat", "ats-json:education-flat-alias", 0.8, compact(value)))
        consume_if_present(consumed, institution_key, degree_key, field_key, year_key, cgpa_key)


def extract_project_facts(facts: list[ExtractedFact], data: dict[str, Any], source: str, consumed: set[str]) -> None:
    key, section = top_level_section(data, PROJECT_KEYS)
    if key:
        consumed.add(key)
    for index, item in enumerate(as_list(section), start=1):
        if isinstance(item, str):
            facts.append(ExtractedFact("projects", {"title": item, "date": None, "tech_stack": [], "links": [], "bullets": []}, f"{source}#project{index}", "ats-json:project", 0.72, item))
            continue
        if not isinstance(item, dict):
            continue
        tech_stack = split_values(first_value(item, ["tech_stack", "techStack", "technologies", "tools", "skills"]) or [])
        links = []
        for raw in split_values(first_value(item, ["links", "link", "url", "repo", "github", "demo"]) or []):
            if url := normalize_url(raw):
                links.append(url)
        bullets = split_values(first_value(item, ["bullets", "description", "summary", "details"]) or [])
        value = {
            "title": first_value(item, ["title", "name", "projectName", "project_name"]),
            "date": first_value(item, ["date", "duration", "period"]),
            "tech_stack": tech_stack,
            "links": links,
            "bullets": bullets[:8],
        }
        if value["title"]:
            facts.append(ExtractedFact("projects", value, f"{source}#project{index}", "ats-json:project", 0.8, compact(item)))
            add_skill_facts(facts, tech_stack, f"{source}#project{index}", "ats-json:project-tech", 0.72, compact(item))


def extract_achievement_facts(facts: list[ExtractedFact], data: dict[str, Any], source: str, consumed: set[str]) -> None:
    key, section = top_level_section(data, ACHIEVEMENT_KEYS)
    if key:
        consumed.add(key)
    for index, item in enumerate(as_list(section), start=1):
        if isinstance(item, dict):
            title = first_value(item, ["title", "name", "award", "achievement"]) or compact(item, 120)
            summary = first_value(item, ["summary", "description", "details"]) or title
            links = [url for raw in split_values(first_value(item, ["links", "url"]) or []) if (url := normalize_url(raw))]
        else:
            title = compact(item, 120)
            summary = compact(item, 300)
            links = []
        if title:
            facts.append(ExtractedFact("achievements", {"title": title, "summary": summary, "links": links}, f"{source}#achievement{index}", "ats-json:achievement", 0.76, summary))


def extract_ats_json(path: Path) -> ExtractionBundle:
    facts: list[ExtractedFact] = []
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ExtractionBundle([], [f"{path.name}: expected a JSON object"])
    except Exception as exc:
        return ExtractionBundle([], [f"{path.name}: failed to parse JSON: {exc}"])

    source = f"ats:{path.name}"
    consumed_top_level: set[str] = set()

    for path_parts, value in leaf_items(data):
        if value in (None, ""):
            continue
        top = top_level_key_for_path(path_parts)
        if terminal_key_matches(path_parts, NAME_KEYS) and isinstance(value, str) and "@" not in value and not any(part.isdigit() for part in path_parts[-1:]):
            facts.append(ExtractedFact("full_name", value, source, "ats-json:name-alias", 0.9, norm_path(path_parts)))
            if top:
                consumed_top_level.add(top)
        if key_matches(path_parts, EMAIL_KEYS):
            for match in EMAIL_RE.finditer(str(value)):
                facts.append(ExtractedFact("emails", match.group(0), source, "ats-json:email-alias", 0.94, norm_path(path_parts)))
                if top:
                    consumed_top_level.add(top)
        elif isinstance(value, str) and EMAIL_RE.search(value):
            for match in EMAIL_RE.finditer(value):
                facts.append(ExtractedFact("emails", match.group(0), source, "ats-json:email-value", 0.78, norm_path(path_parts)))
        if key_matches(path_parts, PHONE_KEYS):
            for match in PHONE_RE.finditer(str(value)):
                facts.append(ExtractedFact("phones", match.group(0), source, "ats-json:phone-alias", 0.9, norm_path(path_parts)))
                if top:
                    consumed_top_level.add(top)
        if key_matches(path_parts, HEADLINE_KEYS) and isinstance(value, str) and len(value) <= 180:
            facts.append(ExtractedFact("headline", value, source, "ats-json:headline-alias", 0.76, norm_path(path_parts)))
            if top:
                consumed_top_level.add(top)
        if key_matches(path_parts, LINK_KEYS):
            for raw in split_values(value):
                if url := normalize_url(raw):
                    field = "links.github" if "github.com" in url.lower() else "links.linkedin" if "linkedin.com" in url.lower() else "links.portfolio" if key_matches(path_parts, {"portfolio", "website"}) else "links.other"
                    facts.append(ExtractedFact(field, url, source, "ats-json:link-alias", 0.82, norm_path(path_parts)))
                    if top:
                        consumed_top_level.add(top)

    extract_location_facts(facts, data, source, consumed_top_level)
    extract_experience_facts(facts, data, source, consumed_top_level)
    extract_education_facts(facts, data, source, consumed_top_level)
    extract_project_facts(facts, data, source, consumed_top_level)
    extract_achievement_facts(facts, data, source, consumed_top_level)

    skill_key, skill_section = top_level_section(data, SKILL_KEYS)
    if skill_key:
        consumed_top_level.add(skill_key)
        add_skill_facts(facts, skill_section, source, "ats-json:skills-section", 0.82, skill_key)
    for path_parts, value in leaf_items(data):
        if value in (None, ""):
            continue
        if key_matches(path_parts, SKILL_KEYS):
            add_skill_facts(facts, value, source, "ats-json:skills-alias", 0.76, norm_path(path_parts))
            if top := top_level_key_for_path(path_parts):
                consumed_top_level.add(top)

    for key, value in data.items():
        if key in consumed_top_level or value in (None, "", [], {}):
            continue
        facts.append(ExtractedFact("others", {"title": str(key), "content": value, "source": source}, source, "ats-json:unmapped", 0.62, compact(value)))

    return ExtractionBundle(facts, errors)
