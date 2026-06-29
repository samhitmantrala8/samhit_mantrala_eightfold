from __future__ import annotations

import csv
from pathlib import Path

from backend.transformer.facts import ExtractedFact, ExtractionBundle


def first_present(row: dict[str, str], names: list[str]) -> str | None:
    lowered = {key.lower().strip(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value and value.strip():
            return value.strip()
    return None


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
                name = first_present(row, ["name", "full_name", "candidate_name"])
                email = first_present(row, ["email", "emails", "primary_email"])
                phone = first_present(row, ["phone", "phones", "mobile"])
                company = first_present(row, ["current_company", "company", "employer"])
                title = first_present(row, ["title", "current_title", "job_title", "role"])

                if name:
                    facts.append(ExtractedFact("full_name", name, source, "csv-column:name", 0.9))
                if email:
                    facts.append(ExtractedFact("emails", email, source, "csv-column:email", 0.94))
                if phone:
                    facts.append(ExtractedFact("phones", phone, source, "csv-column:phone", 0.9))
                if company or title:
                    facts.append(
                        ExtractedFact(
                            "experience",
                            {"company": company, "title": title, "start": None, "end": None, "summary": None},
                            source,
                            "csv-current-role",
                            0.78,
                        )
                    )
                    if title:
                        facts.append(ExtractedFact("headline", title, source, "csv-column:title", 0.72))
    except Exception as exc:
        errors.append(f"{path.name}: failed to parse CSV: {exc}")
    return ExtractionBundle(facts, errors)

