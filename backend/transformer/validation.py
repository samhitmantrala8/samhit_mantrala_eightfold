from __future__ import annotations

import re
from typing import Any


PHONE_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def validate_default_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(profile.get("candidate_id"), str) or not profile["candidate_id"]:
        errors.append("candidate_id must be a non-empty string")
    if profile.get("full_name") is not None and not isinstance(profile["full_name"], str):
        errors.append("full_name must be string or null")
    if not isinstance(profile.get("emails"), list):
        errors.append("emails must be a list")
    if not isinstance(profile.get("phones"), list):
        errors.append("phones must be a list")
    else:
        for phone in profile["phones"]:
            if not PHONE_E164_RE.fullmatch(phone):
                errors.append(f"phone is not E.164: {phone}")
    for item in profile.get("experience", []):
        for key in ("start", "end"):
            if item.get(key) and not MONTH_RE.fullmatch(item[key]):
                errors.append(f"experience.{key} must be YYYY-MM: {item[key]}")
    if not isinstance(profile.get("skills"), list):
        errors.append("skills must be a list")
    else:
        for skill in profile["skills"]:
            if "name" not in skill or "confidence" not in skill or "sources" not in skill:
                errors.append("each skill needs name, confidence, and sources")
    if not isinstance(profile.get("provenance"), list):
        errors.append("provenance must be a list")
    if not isinstance(profile.get("projects", []), list):
        errors.append("projects must be a list")
    if not isinstance(profile.get("achievements", []), list):
        errors.append("achievements must be a list")
    if not isinstance(profile.get("overall_confidence"), (int, float)):
        errors.append("overall_confidence must be numeric")
    if profile.get("profile_summary") is not None and not isinstance(profile["profile_summary"], str):
        errors.append("profile_summary must be string or null")
    if not isinstance(profile.get("resume_sections", {}), dict):
        errors.append("resume_sections must be an object")
    if not isinstance(profile.get("semantic_mappings", []), list):
        errors.append("semantic_mappings must be a list")
    return errors
