from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from backend.history import recent_llmops_examples, record_llmops_trace
from backend.transformer.agentic_llmops import run_agentic_llmops
from backend.transformer.extractors.ats_json_extractor import extract_ats_json
from backend.transformer.extractors.codeforces_extractor import extract_codeforces, extract_codeforces_handles
from backend.transformer.extractors.csv_extractor import extract_csv
from backend.transformer.extractors.docx_extractor import extract_docx, extract_docx_text
from backend.transformer.extractors.github_extractor import extract_github
from backend.transformer.extractors.notes_extractor import extract_notes, extract_notes_from_text
from backend.transformer.extractors.pdf_extractor import extract_pdf, extract_pdf_text
from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.gemini_hybrid import canonicalize_section_headings
from backend.transformer.merge import merge_facts
from backend.transformer.normalizers.contact import classify_link, normalize_email, normalize_phone, normalize_url
from backend.transformer.normalizers.skills import canonicalize_skill
from backend.transformer.projection import project_profile
from backend.transformer.summary import extract_resume_sections, generate_profile_summary
from backend.transformer.validation import validate_default_profile


logger = logging.getLogger(__name__)
KNOWN_RESUME_SECTION_LABELS = {
    "Education",
    "Experience",
    "Professional Background",
    "Projects",
    "Skills",
    "Skills Summary",
    "Achievements",
    "Online Coding Profile",
    "Certifications",
    "Publications",
    "Extracurriculars",
    "Links",
}


TEXT_RESUME_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def move_candidate_id_to_bottom(profile: dict) -> None:
    candidate_id = profile.pop("candidate_id", None)
    if candidate_id is not None:
        profile["candidate_id"] = candidate_id


def extract_from_path(path: Path, use_llm: bool = False) -> ExtractionBundle:
    suffix = path.suffix.lower()
    logger.info("extract_from_path start file=%s suffix=%s use_llm=%s", path.name, suffix, use_llm)
    if suffix == ".csv":
        bundle = extract_csv(path)
    elif suffix == ".json":
        bundle = extract_ats_json(path)
    elif suffix in {".txt", ".md"}:
        bundle = extract_notes(path, use_llm=use_llm)
    elif suffix == ".pdf":
        bundle = extract_pdf(path, use_llm=use_llm)
    elif suffix == ".docx":
        bundle = extract_docx(path, use_llm=use_llm)
    else:
        bundle = ExtractionBundle([], [f"{path.name}: unsupported source type"])
    logger.info("extract_from_path done file=%s facts=%s errors=%s", path.name, len(bundle.facts), len(bundle.errors))
    return bundle


def text_from_path(path: Path) -> str | None:
    logger.info("text_from_path start file=%s suffix=%s", path.name, path.suffix.lower())
    if path.suffix.lower() in {".txt", ".md", ".json"}:
        try:
            text = path.read_text(encoding="utf-8")
            logger.info("text_from_path done file=%s chars=%s encoding=utf-8", path.name, len(text))
            return text
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1", errors="ignore")
            logger.info("text_from_path done file=%s chars=%s encoding=latin-1", path.name, len(text))
            return text
        except Exception as exc:
            logger.exception("text_from_path failed file=%s error=%s", path.name, exc)
            return None
    if path.suffix.lower() == ".pdf":
        try:
            text = extract_pdf_text(path)
            logger.info("text_from_path done file=%s chars=%s pdf=true", path.name, len(text))
            return text
        except Exception as exc:
            logger.exception("text_from_path failed file=%s error=%s", path.name, exc)
            return None
    if path.suffix.lower() == ".docx":
        try:
            text = extract_docx_text(path)
            logger.info("text_from_path done file=%s chars=%s docx=true", path.name, len(text))
            return text
        except Exception as exc:
            logger.exception("text_from_path failed file=%s error=%s", path.name, exc)
            return None
    logger.info("text_from_path skipped file=%s reason=non_text_source", path.name)
    return None


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
    fact_list = list(facts)
    logger.info("normalize_facts start facts=%s default_region=%s", len(fact_list), default_region)
    normalized_facts = [
        normalized
        for fact in fact_list
        if (normalized := normalize_fact(fact, default_region)) is not None
    ]
    logger.info("normalize_facts done input=%s output=%s dropped=%s", len(fact_list), len(normalized_facts), len(fact_list) - len(normalized_facts))
    return normalized_facts


def codeforces_handle_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "codeforces.com" not in parsed.netloc.lower():
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() == "profile":
        return parts[1]
    return None


def transform_paths(
    paths: Iterable[Path],
    config: dict | None = None,
    github_url: str | None = None,
    linkedin_url: str | None = None,
    default_region: str = "US",
    use_llm: bool = False,
    use_gemini_hybrid: bool = False,
    use_agentic_llmops: bool = False,
) -> dict:
    started = time.perf_counter()
    paths = list(paths)
    logger.info(
        "transform_paths start files=%s github_url=%s linkedin_url=%s config=%s default_region=%s use_llm=%s use_gemini_hybrid=%s use_agentic_llmops=%s",
        [path.name for path in paths],
        bool(github_url),
        bool(linkedin_url),
        bool(config),
        default_region,
        use_llm,
        use_gemini_hybrid,
        use_agentic_llmops,
    )
    raw_facts: list[ExtractedFact] = []
    extraction_errors: list[str] = []
    source_texts: list[str] = []
    semantic_mappings: list[dict] = []
    agent_diagnostics: dict | None = None

    for path in paths:
        path_started = time.perf_counter()
        if not path.exists():
            logger.warning("transform_paths missing input file=%s", path)
            extraction_errors.append(f"{path}: missing input")
            continue
        logger.info("source processing start file=%s suffix=%s", path.name, path.suffix.lower())
        source_text = text_from_path(path)
        if source_text:
            source = f"{path.suffix.lower().lstrip('.') or 'text'}:{path.name}"
            if use_gemini_hybrid and path.suffix.lower() in TEXT_RESUME_SUFFIXES:
                logger.info("gemini semantic mapping start source=%s chars=%s", source, len(source_text))
                source_text, mappings, gemini_errors = canonicalize_section_headings(source_text, source)
                semantic_mappings.extend(mappings)
                extraction_errors.extend(gemini_errors)
                logger.info("gemini semantic mapping done source=%s mappings=%s errors=%s", source, len(mappings), len(gemini_errors))
            source_texts.append(source_text)
            if path.suffix.lower() in TEXT_RESUME_SUFFIXES:
                logger.info("notes extraction start source=%s use_llm=%s", source, use_llm)
                bundle = extract_notes_from_text(source_text, source, use_llm=use_llm)
                raw_facts.extend(bundle.facts)
                extraction_errors.extend(bundle.errors)
                logger.info(
                    "notes extraction done source=%s facts=%s errors=%s seconds=%s",
                    source,
                    len(bundle.facts),
                    len(bundle.errors),
                    round(time.perf_counter() - path_started, 2),
                )
                continue
        bundle = extract_from_path(path, use_llm=use_llm)
        raw_facts.extend(bundle.facts)
        extraction_errors.extend(bundle.errors)
        logger.info(
            "source processing done file=%s facts_added=%s errors_added=%s seconds=%s",
            path.name,
            len(bundle.facts),
            len(bundle.errors),
            round(time.perf_counter() - path_started, 2),
        )

    if github_url:
        logger.info("github enrichment start source=input_url")
        bundle = extract_github(github_url)
        raw_facts.extend(bundle.facts)
        extraction_errors.extend(bundle.errors)
        logger.info("github enrichment done source=input_url facts=%s errors=%s", len(bundle.facts), len(bundle.errors))

    if linkedin_url:
        logger.info("linkedin url stored source=input_url")
        raw_facts.append(ExtractedFact("links.linkedin", linkedin_url, "input:linkedin", "user-supplied-url", 0.9))

    normalized_facts = normalize_facts(raw_facts, default_region)
    logger.info("facts after initial normalization raw=%s normalized=%s", len(raw_facts), len(normalized_facts))

    fetched_github_urls = {fact.value for fact in normalized_facts if fact.field == "links.github" and fact.source.startswith("github:")}
    discovered_github_urls = {
        fact.value
        for fact in normalized_facts
        if fact.field == "links.github" and fact.value not in fetched_github_urls
    }
    logger.info("github discovered urls count=%s", len(discovered_github_urls))
    for discovered_url in sorted(discovered_github_urls):
        logger.info("github enrichment start source=discovered_url url=%s", discovered_url)
        bundle = extract_github(discovered_url)
        enriched_facts = normalize_facts(bundle.facts, default_region)
        normalized_facts.extend(enriched_facts)
        extraction_errors.extend(bundle.errors)
        logger.info("github enrichment done source=discovered_url facts=%s normalized=%s errors=%s", len(bundle.facts), len(enriched_facts), len(bundle.errors))

    codeforces_handles = []
    for text in source_texts:
        for handle in extract_codeforces_handles(text):
            if handle not in codeforces_handles:
                codeforces_handles.append(handle)
    for fact in normalized_facts:
        if fact.field == "links.other":
            handle = codeforces_handle_from_url(str(fact.value))
            if handle and handle not in codeforces_handles:
                codeforces_handles.append(handle)
    logger.info("codeforces handles discovered count=%s handles=%s", len(codeforces_handles), codeforces_handles)
    for handle in codeforces_handles:
        logger.info("codeforces enrichment start handle=%s", handle)
        bundle = extract_codeforces(handle)
        enriched_facts = normalize_facts(bundle.facts, default_region)
        normalized_facts.extend(enriched_facts)
        extraction_errors.extend(bundle.errors)
        logger.info("codeforces enrichment done handle=%s facts=%s normalized=%s errors=%s", handle, len(bundle.facts), len(enriched_facts), len(bundle.errors))

    logger.info("merge_facts start normalized_facts=%s extraction_errors=%s", len(normalized_facts), len(extraction_errors))
    default_profile = merge_facts(normalized_facts, extraction_errors)
    logger.info(
        "merge_facts done full_name=%s skills=%s education=%s experience=%s projects=%s achievements=%s provenance=%s",
        default_profile.get("full_name"),
        len(default_profile.get("skills", [])),
        len(default_profile.get("education", [])),
        len(default_profile.get("experience", [])),
        len(default_profile.get("projects", [])),
        len(default_profile.get("achievements", [])),
        len(default_profile.get("provenance", [])),
    )
    logger.info("resume section extraction start source_texts=%s", len(source_texts))
    default_profile["resume_sections"] = extract_resume_sections(source_texts)
    logger.info("resume section extraction done sections=%s", list(default_profile["resume_sections"].keys()))
    default_profile["other_sections"] = [
        {"title": name, "content": body}
        for name, body in default_profile["resume_sections"].items()
        if name not in KNOWN_RESUME_SECTION_LABELS
    ]
    if default_profile["other_sections"]:
        existing_other_titles = {str(item.get("title", "")).lower() for item in default_profile.get("others", []) if isinstance(item, dict)}
        for item in default_profile["other_sections"]:
            title = str(item.get("title", ""))
            if title.lower() not in existing_other_titles:
                default_profile.setdefault("others", []).append({**item, "source": "resume_sections"})
    logger.info("other sections populated count=%s", len(default_profile["other_sections"]))
    default_profile["semantic_mappings"] = semantic_mappings
    logger.info("profile summary generation start source_texts=%s", len(source_texts))
    summary, summary_meta, summary_errors = generate_profile_summary(default_profile, source_texts)
    logger.info("profile summary generation done method=%s source=%s errors=%s", summary_meta.get("method"), summary_meta.get("source"), len(summary_errors))
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

    if use_agentic_llmops:
        logger.info("agentic llmops start")
        pre_agent_validation_errors = validate_default_profile(default_profile)
        logger.info("agentic llmops pre-validation errors=%s", len(pre_agent_validation_errors))
        try:
            examples = recent_llmops_examples(limit=100)
            logger.info("agentic llmops memory loaded examples=%s", len(examples))
        except Exception as exc:
            examples = []
            extraction_errors.append(f"llmops_agent: failed to load memory examples: {exc}")
            logger.exception("agentic llmops memory load failed")
        default_profile, llmops_trace, llmops_errors = run_agentic_llmops(
            default_profile,
            source_texts,
            validation_errors=pre_agent_validation_errors,
            memory_examples=examples,
            default_region=default_region,
        )
        extraction_errors.extend(llmops_errors)
        agent_diagnostics = llmops_trace
        logger.info(
            "agentic llmops done final_score=%s stopping_reason=%s errors=%s",
            llmops_trace.get("final_score"),
            llmops_trace.get("stopping_reason"),
            len(llmops_errors),
        )
        try:
            trace_id = record_llmops_trace(llmops_trace)
            agent_diagnostics["trace_id"] = trace_id
            logger.info("agentic llmops trace stored trace_id=%s", trace_id)
        except Exception as exc:
            extraction_errors.append(f"llmops_agent: failed to store trace: {exc}")
            logger.exception("agentic llmops trace store failed")
        default_profile.pop("llmops", None)

    default_profile.pop("extraction_errors", None)
    move_candidate_id_to_bottom(default_profile)
    logger.info("validation start")
    validation_errors = validate_default_profile(default_profile)
    logger.info("validation done validation_errors=%s", len(validation_errors))
    logger.info("custom projection start enabled=%s", bool(config))
    custom_output, projection_errors = project_profile(default_profile, config, default_region)
    logger.info("custom projection done projection_errors=%s", len(projection_errors))
    logger.info("transform_paths done seconds=%s", round(time.perf_counter() - started, 2))

    return {
        "default_profile": default_profile,
        "agent_diagnostics": agent_diagnostics,
        "custom_output": custom_output if config else None,
        "extraction_errors": extraction_errors,
        "validation_errors": validation_errors + projection_errors,
    }
