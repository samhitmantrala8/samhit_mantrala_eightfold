from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from backend.transformer.extractors.notes_extractor import extract_notes_from_text
from backend.transformer.facts import ExtractionBundle


def extract_pdf_text(path: Path, max_pages: int = 25) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def extract_pdf(path: Path, use_llm: bool = False) -> ExtractionBundle:
    try:
        text = extract_pdf_text(path)
    except Exception as exc:
        return ExtractionBundle([], [f"{path.name}: failed to parse PDF: {exc}"])
    if not text.strip():
        return ExtractionBundle([], [f"{path.name}: no extractable PDF text found"])
    return extract_notes_from_text(text, f"pdf:{path.name}", use_llm=use_llm)
