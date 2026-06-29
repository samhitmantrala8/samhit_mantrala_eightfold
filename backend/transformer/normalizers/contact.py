from __future__ import annotations

import re
from urllib.parse import urlparse

import phonenumbers


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,5}\d{2,4}")
URL_RE = re.compile(r"https?://[^\s),]+", re.IGNORECASE)


def normalize_email(value: str | None) -> str | None:
    if not value:
        return None
    match = EMAIL_RE.search(value.strip())
    return match.group(0).lower() if match else None


def normalize_phone(value: str | None, default_region: str = "US") -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    try:
        parsed = phonenumbers.parse(cleaned, None if cleaned.startswith("+") else default_region)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_possible_number(parsed) or not phonenumbers.is_valid_number(parsed):
        return None
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().rstrip(".,")
    if not candidate.startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    if not parsed.netloc:
        return None
    return parsed.geturl()


def classify_link(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "github.com" in host:
        return "links.github"
    if "linkedin.com" in host:
        return "links.linkedin"
    return "links.other"

