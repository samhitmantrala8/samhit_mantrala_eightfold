from __future__ import annotations

from pathlib import Path
from typing import Iterable

from backend.transformer.extractors.ats_json_extractor import extract_ats_json
from backend.transformer.extractors.csv_extractor import extract_csv
from backend.transformer.extractors.github_extractor import extract_github
from backend.transformer.extractors.notes_extractor import extract_notes
from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.merge import merge_facts
from backend.transformer.normalizers.contact import classify_link, normalize_email, normalize_phone, normalize_url
from backend.transformer.normalizers.skills import canonicalize_skill
from backend.transformer.projection import project_profile
from backend.transformer.summary import extract_resume_sections, generate_profile_summary
from backend.transformer.validation import validate_default_profile


def extract_from_path(path: Path, use_llm: bool = False) -> ExtractionBundle:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return extract_csv(path)
    if suffix == ".json":
        return extract_ats_json(path)
    if suffix in {".txt", ".md"}:
        return extract_notes(path, use_llm=use_llm)
    return ExtractionBundle([], [f"{path.name}: unsupported source type"])


def normalize_fact(fact: ExtractedFact, default_region: str) -> ExtractedFact | None:
    field = fact.field
    value = fact.value
    if field == "emails":
        normalized = normalize_email(str(value))
        return ExtractedFact(field, normalized, fact.source, fact.method, fact.confidence, fact.evidence) if normalized else None
    if field == "phones":
        normalized = normalize_phone(str(value), default_region)
        return ExtractedFact(field, normalized, fact.source, fact.method, fact.confidence, fact.evidence) if normalized else None
    if field.startswith("links."):
        normalized = normalize_url(str(value))
        if not normalized:
            return None
        return ExtractedFact(classify_link(normalized), normalized, fact.source, fact.method, fact.confidence, fact.evidence)
    if field == "skills":
        raw = value.get("name") if isinstance(value, dict) else str(value)
        canonical = canonicalize_skill(str(raw))
        if canonical:
            return ExtractedFact(field, {"name": canonical[0]}, fact.source, fact.method, min(fact.confidence, canonical[1]), fact.evidence)
        return None
    if field == "years_experience":
        try:
            years = float(value)
        except (TypeError, ValueError):
            return None
        return ExtractedFact(field, years, fact.source, fact.method, fact.confidence, fact.evidence)
    return fact


def normalize_facts(facts: Iterable[ExtractedFact], default_region: str) -> list[ExtractedFact]:
    return [
        normalized
        for fact in facts
        if (normalized := normalize_fact(fact, default_region)) is not None
    ]


def transform_paths(
    paths: Iterable[Path],
    config: dict | None = None,
    github_url: str | None = None,
    linkedin_url: str | None = None,
    default_region: str = "US",
    use_llm: bool = False,
) -> dict:
    raw_facts: list[ExtractedFact] = []
    extraction_errors: list[str] = []
    source_texts: list[str] = []

    for path in paths:
        if not path.exists():
            extraction_errors.append(f"{path}: missing input")
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            try:
                source_texts.append(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                source_texts.append(path.read_text(encoding="latin-1", errors="ignore"))
            except Exception:
                pass
        bundle = extract_from_path(path, use_llm=use_llm)
        raw_facts.extend(bundle.facts)
        extraction_errors.extend(bundle.errors)

    if github_url:
        bundle = extract_github(github_url)
        raw_facts.extend(bundle.facts)
        extraction_errors.extend(bundle.errors)

    if linkedin_url:
        raw_facts.append(ExtractedFact("links.linkedin", linkedin_url, "input:linkedin", "user-supplied-url", 0.9))

    normalized_facts = normalize_facts(raw_facts, default_region)

    fetched_github_urls = {fact.value for fact in normalized_facts if fact.field == "links.github" and fact.source.startswith("github:")}
    discovered_github_urls = {
        fact.value
        for fact in normalized_facts
        if fact.field == "links.github" and fact.value not in fetched_github_urls
    }
    for discovered_url in sorted(discovered_github_urls):
        bundle = extract_github(discovered_url)
        normalized_facts.extend(normalize_facts(bundle.facts, default_region))
        extraction_errors.extend(bundle.errors)

    default_profile = merge_facts(normalized_facts, extraction_errors)
    default_profile["resume_sections"] = extract_resume_sections(source_texts)
    summary, summary_meta, summary_errors = generate_profile_summary(default_profile, source_texts)
    default_profile["profile_summary"] = summary
    default_profile["provenance"].append(
        {
            "field": "profile_summary",
            "source": summary_meta["source"],
            "method": summary_meta["method"],
            "confidence": summary_meta["confidence"],
        }
    )
    extraction_errors.extend(summary_errors)
    default_profile["extraction_errors"] = extraction_errors
    validation_errors = validate_default_profile(default_profile)
    custom_output, projection_errors = project_profile(default_profile, config, default_region)

    return {
        "default_profile": default_profile,
        "custom_output": custom_output if config else None,
        "extraction_errors": extraction_errors,
        "validation_errors": validation_errors + projection_errors,
    }
