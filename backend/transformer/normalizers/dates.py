from __future__ import annotations

import re
from datetime import datetime

from dateutil import parser


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def normalize_month(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if text.lower() in {"present", "current", "now"}:
        return None
    if re.fullmatch(r"(19|20)\d{2}", text):
        return f"{text}-01"
    try:
        dt = parser.parse(text, default=datetime(1900, 1, 1), fuzzy=True)
    except (ValueError, OverflowError):
        match = YEAR_RE.search(text)
        return f"{match.group(0)}-01" if match else None
    if dt.year < 1900:
        return None
    return f"{dt.year:04d}-{dt.month:02d}"

