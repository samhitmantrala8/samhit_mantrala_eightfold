from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.dates import normalize_month


def dig(data: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        node: Any = data
        found = True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                found = False
                break
        if found and node not in (None, ""):
            return node
    return None


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
    name = dig(data, ["candidateName", "fullName", "name", "profile.name", "person.fullName"])
    email = dig(data, ["emailAddress", "email", "contact.email", "profile.email"])
    phone = dig(data, ["phoneNumber", "phone", "contact.phone", "profile.phone"])
    headline = dig(data, ["headline", "summary.title", "currentTitle"])
    location = dig(data, ["location", "profile.location"])

    if name:
        facts.append(ExtractedFact("full_name", str(name), source, "ats-field:name", 0.92))
    if email:
        facts.append(ExtractedFact("emails", str(email), source, "ats-field:email", 0.95))
    if phone:
        facts.append(ExtractedFact("phones", str(phone), source, "ats-field:phone", 0.92))
    if headline:
        facts.append(ExtractedFact("headline", str(headline), source, "ats-field:headline", 0.8))
    if isinstance(location, dict):
        for key in ("city", "region", "country"):
            if location.get(key):
                facts.append(ExtractedFact(f"location.{key}", str(location[key]), source, f"ats-field:location.{key}", 0.82))

    experiences = data.get("experience") or data.get("workHistory") or []
    if isinstance(experiences, dict):
        experiences = [experiences]
    if isinstance(experiences, list):
        for index, item in enumerate(experiences, start=1):
            if not isinstance(item, dict):
                continue
            facts.append(
                ExtractedFact(
                    "experience",
                    {
                        "company": item.get("company") or item.get("companyName") or item.get("employer"),
                        "title": item.get("title") or item.get("jobTitle") or item.get("position"),
                        "start": normalize_month(str(item.get("start") or item.get("startDate") or "")),
                        "end": normalize_month(str(item.get("end") or item.get("endDate") or "")),
                        "summary": item.get("summary"),
                    },
                    f"{source}#experience{index}",
                    "ats-field:experience",
                    0.84,
                )
            )

    education = data.get("education") or []
    if isinstance(education, dict):
        education = [education]
    if isinstance(education, list):
        for index, item in enumerate(education, start=1):
            if not isinstance(item, dict):
                continue
            facts.append(
                ExtractedFact(
                    "education",
                    {
                        "institution": item.get("institution") or item.get("school"),
                        "degree": item.get("degree"),
                        "field": item.get("field") or item.get("major"),
                        "end_year": item.get("end_year") or item.get("graduationYear"),
                    },
                    f"{source}#education{index}",
                    "ats-field:education",
                    0.82,
                )
            )
    return ExtractionBundle(facts, errors)

