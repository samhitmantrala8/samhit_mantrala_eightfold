from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from backend.transformer.facts import ExtractedFact


SOURCE_WEIGHTS = {
    "ats": 0.06,
    "csv": 0.05,
    "github": 0.01,
    "docx": -0.02,
    "pdf": -0.02,
    "notes": -0.02,
}


def source_weight(source: str) -> float:
    prefix = source.split(":", 1)[0]
    return SOURCE_WEIGHTS.get(prefix, 0.0)


def score(fact: ExtractedFact) -> float:
    return max(0.0, min(0.99, fact.confidence + source_weight(fact.source)))


def clean_dict(value: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: value.get(key) for key in keys}


def best_fact(facts: list[ExtractedFact]) -> ExtractedFact | None:
    if not facts:
        return None
    return sorted(facts, key=lambda item: (score(item), len(str(item.value))), reverse=True)[0]


def provenance_entry(fact: ExtractedFact) -> dict[str, Any]:
    entry = {
        "field": fact.field,
        "source": fact.source,
        "method": fact.method,
        "confidence": round(score(fact), 3),
    }
    if fact.evidence:
        entry["evidence"] = fact.evidence[:180]
    return entry


def stable_candidate_id(facts: list[ExtractedFact]) -> str:
    anchors: list[str] = []
    for fact in facts:
        if fact.field in {"emails", "phones", "full_name", "links.github", "links.linkedin"} and fact.value:
            anchors.append(f"{fact.field}:{str(fact.value).strip().lower()}")
    if not anchors:
        anchors = [f"{fact.source}:{fact.field}:{json.dumps(fact.value, sort_keys=True, default=str)}" for fact in facts[:20]]
    raw = "|".join(sorted(anchors)) or "empty-candidate"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"cand_{digest}"


def merge_facts(facts: list[ExtractedFact], extraction_errors: list[str] | None = None) -> dict[str, Any]:
    by_field: dict[str, list[ExtractedFact]] = defaultdict(list)
    for fact in facts:
        by_field[fact.field].append(fact)

    profile: dict[str, Any] = {
        "full_name": None,
        "emails": [],
        "phones": [],
        "location": {"city": None, "region": None, "country": None},
        "links": {"linkedin": None, "github": None, "portfolio": None, "other": []},
        "headline": None,
        "years_experience": None,
        "skills": [],
        "experience": [],
        "education": [],
        "projects": [],
        "achievements": [],
        "certifications": [],
        "publications": [],
        "online_coding_profile": {},
        "github_repositories": [],
        "languages": [],
        "extracurriculars": [],
        "other_sections": [],
        "others": [],
        "profile_summary": None,
        "resume_sections": {},
        "provenance": [],
        "overall_confidence": 0.0,
        "extraction_errors": extraction_errors or [],
    }

    for field in ("full_name", "headline", "years_experience"):
        winner = best_fact(by_field[field])
        if winner:
            profile[field] = winner.value

    for part in ("city", "region", "country"):
        winner = best_fact(by_field[f"location.{part}"])
        if winner:
            profile["location"][part] = winner.value

    for link_field, target_key in [
        ("links.github", "github"),
        ("links.linkedin", "linkedin"),
        ("links.portfolio", "portfolio"),
    ]:
        winner = best_fact(by_field[link_field])
        if winner:
            profile["links"][target_key] = winner.value

    other_links = []
    for fact in sorted(by_field["links.other"], key=score, reverse=True):
        if fact.value not in other_links:
            other_links.append(fact.value)
    profile["links"]["other"] = other_links

    for field in ("emails", "phones"):
        seen = set()
        ordered = []
        for fact in sorted(by_field[field], key=score, reverse=True):
            if fact.value and fact.value not in seen:
                ordered.append(fact.value)
                seen.add(fact.value)
        profile[field] = ordered

    skill_groups: dict[str, list[ExtractedFact]] = defaultdict(list)
    for fact in by_field["skills"]:
        name = fact.value.get("name") if isinstance(fact.value, dict) else str(fact.value)
        if name:
            skill_groups[name].append(fact)
    for name, group in sorted(skill_groups.items()):
        sources = sorted({fact.source for fact in group})
        boosted = min(0.99, max(score(fact) for fact in group) + 0.04 * (len(sources) - 1))
        profile["skills"].append({"name": name, "confidence": round(boosted, 3), "sources": sources})
    profile["skills"].sort(key=lambda item: (-item["confidence"], item["name"]))

    experience_groups: dict[tuple[str, str], list[ExtractedFact]] = defaultdict(list)
    for fact in by_field["experience"]:
        if not isinstance(fact.value, dict):
            continue
        company = (fact.value.get("company") or "").strip()
        title = (fact.value.get("title") or "").strip()
        if not company and not title:
            continue
        experience_groups[(company.lower(), title.lower())].append(fact)
    for group in experience_groups.values():
        winner = best_fact(group)
        if winner and isinstance(winner.value, dict):
            profile["experience"].append(clean_dict(winner.value, ["company", "title", "role", "location", "duration", "start", "end", "summary"]))

    education_groups: dict[tuple[str, str], list[ExtractedFact]] = defaultdict(list)
    for fact in by_field["education"]:
        if not isinstance(fact.value, dict):
            continue
        institution = (fact.value.get("institution") or "").strip()
        degree = (fact.value.get("degree") or "").strip()
        if institution or degree:
            education_groups[(institution.lower(), degree.lower())].append(fact)
    for group in education_groups.values():
        winner = best_fact(group)
        if winner and isinstance(winner.value, dict):
            profile["education"].append(clean_dict(winner.value, ["institution", "degree", "field", "end_year", "cgpa"]))

    project_groups: dict[str, list[ExtractedFact]] = defaultdict(list)
    for fact in by_field["projects"]:
        if not isinstance(fact.value, dict):
            continue
        title = (fact.value.get("title") or "").strip()
        if title:
            project_groups[title.lower()].append(fact)
    for group in project_groups.values():
        winner = best_fact(group)
        if winner and isinstance(winner.value, dict):
            project = clean_dict(winner.value, ["title", "date", "tech_stack", "links", "bullets"])
            profile["projects"].append(project)
            if winner.source.startswith("github:") or "#repo:" in winner.source:
                repo = {
                    "name": project.get("title"),
                    "date": project.get("date"),
                    "tech_stack": project.get("tech_stack") or [],
                    "links": project.get("links") or [],
                    "bullets": project.get("bullets") or [],
                    "source": winner.source,
                }
                profile["github_repositories"].append(repo)

    achievement_groups: dict[str, list[ExtractedFact]] = defaultdict(list)
    for fact in by_field["achievements"]:
        if not isinstance(fact.value, dict):
            continue
        title = (fact.value.get("title") or fact.value.get("summary") or "").strip()
        if title:
            achievement_groups[title.lower()].append(fact)
    for group in achievement_groups.values():
        winner = best_fact(group)
        if winner and isinstance(winner.value, dict):
            profile["achievements"].append(clean_dict(winner.value, ["title", "summary", "links"]))

    other_groups: dict[str, list[ExtractedFact]] = defaultdict(list)
    for fact in by_field["others"]:
        if not isinstance(fact.value, dict):
            continue
        title = (fact.value.get("title") or fact.evidence or fact.source or "Other").strip()
        other_groups[title.lower()].append(fact)
    for group in other_groups.values():
        winner = best_fact(group)
        if winner and isinstance(winner.value, dict):
            profile["others"].append(clean_dict(winner.value, ["title", "content", "source"]))

    competitive_winner = best_fact(by_field["online_coding_profile"])
    if competitive_winner and isinstance(competitive_winner.value, dict):
        profile["online_coding_profile"] = competitive_winner.value

    accepted_fields = {
        "full_name",
        "headline",
        "years_experience",
        "emails",
        "phones",
        "skills",
        "experience",
        "education",
        "projects",
        "achievements",
        "online_coding_profile",
        "others",
    }
    accepted_fields.update({f"location.{key}" for key in ("city", "region", "country")})
    accepted_fields.update({"links.github", "links.linkedin", "links.portfolio", "links.other"})
    profile["provenance"] = [provenance_entry(fact) for fact in facts if fact.field in accepted_fields]

    confidence_components = []
    for field in ("full_name", "emails", "phones", "headline"):
        field_facts = by_field[field]
        if field_facts:
            confidence_components.append(score(best_fact(field_facts)))
    if profile["skills"]:
        confidence_components.append(sum(skill["confidence"] for skill in profile["skills"]) / len(profile["skills"]))
    profile["overall_confidence"] = round(sum(confidence_components) / len(confidence_components), 3) if confidence_components else 0.0
    profile["candidate_id"] = stable_candidate_id(facts)
    return profile
