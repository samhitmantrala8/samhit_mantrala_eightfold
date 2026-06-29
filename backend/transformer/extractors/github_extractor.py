from __future__ import annotations

import os
from urllib.parse import urlparse

import requests

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.skills import canonicalize_skill


def username_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url if url.startswith(("http://", "https://")) else f"https://{url}")
    if "github.com" not in parsed.netloc.lower():
        return None
    parts = [part for part in parsed.path.split("/") if part]
    return parts[0] if parts else None


def extract_github(url: str | None) -> ExtractionBundle:
    username = username_from_url(url)
    if not username:
        return ExtractionBundle([], [])

    facts: list[ExtractedFact] = []
    errors: list[str] = []
    source = f"github:{username}"
    facts.append(ExtractedFact("links.github", f"https://github.com/{username}", source, "github-url", 0.9))

    try:
        response = requests.get(f"https://api.github.com/users/{username}", timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return ExtractionBundle(facts, [f"github:{username}: API lookup failed: {exc}"])

    if data.get("name"):
        facts.append(ExtractedFact("full_name", data["name"], source, "github-api:name", 0.72))
    if data.get("bio"):
        facts.append(ExtractedFact("headline", data["bio"], source, "github-api:bio", 0.62))
    if data.get("location"):
        facts.append(ExtractedFact("location.city", data["location"], source, "github-api:location", 0.45))

    if os.getenv("FETCH_GITHUB_REPOS", "false").lower() in {"1", "true", "yes"}:
        try:
            repos = requests.get(f"https://api.github.com/users/{username}/repos?per_page=20", timeout=8)
            repos.raise_for_status()
            for repo in repos.json():
                language = repo.get("language")
                if language:
                    canonical = canonicalize_skill(language)
                    if canonical:
                        facts.append(ExtractedFact("skills", {"name": canonical[0]}, source, "github-api:repo-language", 0.62))
        except requests.RequestException as exc:
            errors.append(f"github:{username}: repo language lookup failed: {exc}")

    return ExtractionBundle(facts, errors)

