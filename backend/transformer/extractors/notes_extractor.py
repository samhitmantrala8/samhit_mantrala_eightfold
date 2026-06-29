from __future__ import annotations

import re
from pathlib import Path

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.contact import EMAIL_RE, PHONE_RE, URL_RE, classify_link, normalize_url
from backend.transformer.normalizers.dates import normalize_month
from backend.transformer.normalizers.skills import extract_skills_from_text


NAME_RE = re.compile(r"(?im)^(?:candidate|name)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,80})$")
HEADLINE_RE = re.compile(r"(?im)^headline\s*[:\-]\s*(.{4,160})$")
YEARS_RE = re.compile(r"\b(\d{1,2})(?:\+)?\s+years?\b", re.IGNORECASE)
BARE_URL_RE = re.compile(
    r"(?<![A-Za-z0-9./:-])(?:github\.com|linkedin\.com|www\.linkedin\.com|www\.kaggle\.com|kaggle\.com)/[^\s),]+",
    re.IGNORECASE,
)
DATE_RANGE_RE = re.compile(
    r"(?P<start>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4})\s*[-–]\s*(?P<end>Present|Current|Now|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4})",
    re.IGNORECASE,
)
ROLE_RE = re.compile(
    r"\b(?:worked|working|experience)\s+(?:as\s+)?(?P<title>[A-Za-z0-9 /+.-]{3,60})\s+at\s+(?P<company>[A-Z][A-Za-z0-9 &.-]{2,80})",
    re.IGNORECASE,
)
SECTION_HEADERS = {"education", "experience", "projects", "achievements", "skills summary", "skills"}
ACTION_PREFIX_RE = re.compile(
    r"^(?:developed|built|used|implemented|reduced|deployed|containerized|visualized|tracked|handled|leveraged|created|stored|fine-tuned|optimised|optimized)\b",
    re.IGNORECASE,
)
BULLET_REPLACEMENTS = (
    ("\u00c2\u2022", "\u2022"),
    ("\u0100\u2022", "\u2022"),
    ("\u0095", "\u2022"),
)
BULLET_PREFIX_RE = re.compile(r"^[\s\-*?]*(?:\u2022+\s*)+")


def clean_line(line: str) -> str:
    for bad, good in BULLET_REPLACEMENTS:
        line = line.replace(bad, good)
    line = line.replace("Â•", "•").replace("\u2022", "•").replace("\t", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def strip_leading_marker(value: str) -> str:
    value = clean_line(value)
    value = BULLET_PREFIX_RE.sub("", value)
    return value.strip(" -:")


def non_empty_lines(text: str) -> list[str]:
    return [clean_line(line) for line in text.splitlines() if clean_line(line)]


def header_name(text: str) -> str | None:
    for line in non_empty_lines(text)[:4]:
        if line.lower() in SECTION_HEADERS:
            continue
        candidate = re.split(r"\b(?:Email|LinkedIn|Mobile|Phone|GitHub)\b\s*:", line, maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = candidate.strip(" -|")
        if re.fullmatch(r"[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3}", candidate):
            return candidate
    return None


def section_lines(text: str, start_header: str, end_headers: set[str]) -> list[str]:
    lines = non_empty_lines(text)
    start_index = None
    for index, line in enumerate(lines):
        if line.lower() == start_header.lower():
            start_index = index + 1
            break
    if start_index is None:
        return []
    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if lines[index].lower() in end_headers:
            end_index = index
            break
    return lines[start_index:end_index]


def split_columns(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s{2,}|\t+", line) if part.strip()]


def strip_trailing_location(line: str) -> str:
    country_match = re.search(r",\s*(?:India|United States|USA|Australia|Canada|UK|United Kingdom)(?:\s*\(Remote\))?$", line)
    if not country_match:
        return line.strip()
    before_country = line[: country_match.start()].strip()
    tokens = before_country.split()
    if len(tokens) >= 2 and tokens[-1].lower() == tokens[-2].lower():
        return " ".join(tokens[:-1]).strip()
    return before_country


def parse_education(text: str, source: str) -> list[ExtractedFact]:
    lines = section_lines(text, "Education", {"experience", "projects", "achievements", "skills summary", "skills"})
    if not lines:
        return []

    useful = [line for line in lines if not line.lower().startswith("courses:")]
    if not useful:
        return []

    institution_line = useful[0].lstrip("•-? ").strip()
    institution_line = strip_leading_marker(institution_line)
    columns = split_columns(institution_line)
    institution = strip_trailing_location(columns[0] if columns else institution_line)

    degree_line = next(
        (
            line
            for line in useful[1:]
            if re.search(r"\b(Bachelor|Master|B\.?Tech|M\.?Tech|Degree|Computer Science)\b", line, re.IGNORECASE)
        ),
        "",
    )
    degree_text = re.sub(r";.*$", "", degree_line).strip()
    degree = degree_text
    field = None
    if " - " in degree_text:
        degree, field = [part.strip() for part in degree_text.split(" - ", 1)]
    elif "-" in degree_text:
        degree, field = [part.strip() for part in degree_text.split("-", 1)]

    year_matches = re.findall(r"\b(?:19|20)\d{2}\b", " ".join(lines))
    end_year = int(year_matches[-1]) if year_matches else None
    cgpa_match = re.search(r"\bCGPA\s*:\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?)", " ".join(lines), re.IGNORECASE)
    cgpa = re.sub(r"\s+", "", cgpa_match.group(1)) if cgpa_match else None

    if not institution and not degree:
        return []
    return [
        ExtractedFact(
            "education",
            {"institution": institution or None, "degree": degree or None, "field": field or None, "end_year": end_year, "cgpa": cgpa},
            source,
            "notes-resume-section:education",
            0.76,
            " | ".join(useful[:3]),
        )
    ]


def matching_paren_close(value: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(value)):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def parse_role_header(line: str) -> tuple[str, str, str | None] | None:
    header = line.lstrip("•-*?Â ").strip()
    header = strip_leading_marker(header)
    open_index = header.find("(")
    if open_index <= 0:
        return None
    close_index = matching_paren_close(header, open_index)
    if close_index < 0:
        return None
    company = header[:open_index].strip(" -:")
    company = strip_leading_marker(company)
    title = header[open_index + 1 : close_index].strip()
    location = header[close_index + 1 :].strip(" -")
    if not company or not title:
        return None
    return company, title, location or None


def is_role_header_line(line: str) -> bool:
    parsed = parse_role_header(line)
    if not parsed:
        return False
    company, _title, location = parsed
    if ACTION_PREFIX_RE.search(company):
        return False
    if len(company.split()) > 8:
        return False
    if location and "," in location:
        return True
    return bool(re.search(r"\b(?:intern|engineer|developer|manager|analyst|scientist|consultant)\b", line, re.IGNORECASE))


def summary_from_block(block: list[str]) -> str | None:
    summary_lines = []
    for line in block[1:]:
        if DATE_RANGE_RE.search(line) or line.lower().startswith("(team:"):
            continue
        cleaned = line.lstrip("•-*? ").strip()
        cleaned = strip_leading_marker(cleaned)
        if cleaned:
            summary_lines.append(cleaned)
        if len(summary_lines) >= 2:
            break
    summary = " ".join(summary_lines).strip()
    return summary[:360] if summary else None


def parse_experience(text: str, source: str) -> list[ExtractedFact]:
    lines = section_lines(text, "Experience", {"projects", "achievements", "skills summary", "skills", "education"})
    facts: list[ExtractedFact] = []
    if not lines:
        return facts

    blocks: list[list[str]] = []
    for line in lines:
        if is_role_header_line(line):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)

    for index, block in enumerate(blocks):
        parsed = parse_role_header(block[0])
        if not parsed:
            continue
        company, title, location = parsed
        block_text = " ".join(block)
        date_match = DATE_RANGE_RE.search(block_text)
        start = normalize_month(date_match.group("start")) if date_match else None
        end = None
        duration = date_match.group(0).replace("–", "-") if date_match else None
        current_role = False
        if date_match:
            raw_end = date_match.group("end")
            current_role = raw_end.lower() in {"present", "current", "now"}
            end = None if current_role else normalize_month(raw_end)

        facts.append(
            ExtractedFact(
                "experience",
                {
                    "company": company,
                    "title": title,
                    "role": title,
                    "location": location,
                    "duration": duration,
                    "start": start,
                    "end": end,
                    "summary": summary_from_block(block),
                },
                source,
                "notes-resume-section:experience",
                0.76,
                " | ".join(block[:3]),
            )
        )
        if index == 0 or current_role:
            facts.append(
                ExtractedFact(
                    "headline",
                    f"{title} at {company}",
                    source,
                    "notes-resume-section:current-role",
                    0.66 if current_role else 0.6,
                    block[0],
                )
            )
    return facts


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

    parsed_name = header_name(text)
    if parsed_name:
        facts.append(ExtractedFact("full_name", parsed_name, source, "notes-resume-header:name", 0.86, non_empty_lines(text)[0]))
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
    for match in BARE_URL_RE.finditer(text):
        url = normalize_url(match.group(0))
        if url:
            facts.append(ExtractedFact(classify_link(url), url, source, "notes-regex:bare-url", 0.76, match.group(0)))
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

    facts.extend(parse_education(text, source))
    facts.extend(parse_experience(text, source))

    if use_llm:
        from backend.transformer.extractors.llm_extractor import extract_text_with_llm

        llm_bundle = extract_text_with_llm(text, source)
        facts.extend(llm_bundle.facts)
        errors.extend(llm_bundle.errors)

    return ExtractionBundle(facts, errors)
