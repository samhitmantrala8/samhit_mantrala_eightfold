from __future__ import annotations

import re
from typing import Any

from backend.transformer.normalizers.contact import normalize_phone
from backend.transformer.normalizers.skills import canonicalize_skill


TOKEN_RE = re.compile(r"([A-Za-z_][\w]*)(?:\[(\d*|\*)\])?")


class ProjectionError(ValueError):
    pass


def read_path(data: Any, path: str) -> Any:
    parts = path.split(".")

    def walk(node: Any, index: int) -> Any:
        if index >= len(parts):
            return node
        match = TOKEN_RE.fullmatch(parts[index].replace("[]", "[*]"))
        if not match:
            raise ProjectionError(f"Unsupported path syntax: {path}")
        key, item_index = match.group(1), match.group(2)
        if not isinstance(node, dict) or key not in node:
            return None
        value = node[key]
        if item_index is None:
            return walk(value, index + 1)
        if not isinstance(value, list):
            return None
        if item_index in {"", "*"}:
            return [walk(item, index + 1) for item in value]
        position = int(item_index)
        if position >= len(value):
            return None
        return walk(value[position], index + 1)

    return walk(data, 0)


def set_path(output: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = output
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def missing(value: Any) -> bool:
    return value is None or value == [] or value == ""


def normalize_projected(value: Any, normalization: str | None, default_region: str) -> Any:
    if normalization is None:
        return value
    norm = normalization.lower()
    if norm == "e164":
        if isinstance(value, list):
            return [phone for item in value if (phone := normalize_phone(str(item), default_region))]
        return normalize_phone(str(value), default_region)
    if norm == "canonical":
        if isinstance(value, list):
            canonical = []
            for item in value:
                result = canonicalize_skill(str(item))
                if result:
                    canonical.append(result[0])
            return sorted(set(canonical))
        result = canonicalize_skill(str(value))
        return result[0] if result else value
    return value


def validate_type(value: Any, expected: str) -> bool:
    if value is None:
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string[]":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "object[]":
        return isinstance(value, list) and all(isinstance(item, dict) for item in value)
    return True


def project_profile(profile: dict[str, Any], config: dict[str, Any] | None, default_region: str = "US") -> tuple[dict[str, Any], list[str]]:
    if not config:
        return profile, []

    on_missing = config.get("on_missing", "null")
    output: dict[str, Any] = {}
    errors: list[str] = []

    for field in config.get("fields", []):
        target_path = field["path"]
        source_path = field.get("from", target_path)
        if target_path == "extraction_errors" or source_path == "extraction_errors":
            continue
        expected_type = field.get("type")
        value = read_path(profile, source_path)
        value = normalize_projected(value, field.get("normalize"), default_region)

        field_on_missing = field.get("on_missing", on_missing)
        if missing(value):
            if field.get("required") or field_on_missing == "error":
                errors.append(f"Missing required field: {target_path} from {source_path}")
                if field_on_missing == "error":
                    continue
            if field_on_missing == "omit":
                continue
            value = None

        if expected_type and not validate_type(value, expected_type):
            errors.append(f"Field {target_path} expected {expected_type}, got {type(value).__name__}")
            continue
        set_path(output, target_path, value)

    if config.get("include_confidence"):
        output["overall_confidence"] = profile.get("overall_confidence")
    if config.get("include_provenance"):
        output["provenance"] = profile.get("provenance", [])
    if "candidate_id" in output:
        candidate_id = output.pop("candidate_id")
        output["candidate_id"] = candidate_id
    return output, errors
