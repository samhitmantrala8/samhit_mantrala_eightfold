from __future__ import annotations

import re

import requests

from backend.transformer.facts import ExtractedFact, ExtractionBundle


CODEFORCES_HANDLE_PATTERNS = [
    re.compile(r"\b(?:codeforces|cf)\s+(?:handle|profile)\s*[:\-]?\s*(?P<handle>[A-Za-z0-9_.-]{3,24})\b", re.IGNORECASE),
    re.compile(r"\b(?:codeforces|cf)\s*[:\-]\s*(?P<handle>[A-Za-z0-9_.-]{3,24})\b", re.IGNORECASE),
    re.compile(r"\bcodeforces\b[^\n]{0,120}\blink\s*[:\-]?\s*(?P<handle>[A-Za-z0-9_.-]{3,24})\b", re.IGNORECASE),
]


def extract_codeforces_handles(text: str) -> list[str]:
    handles: list[str] = []
    for pattern in CODEFORCES_HANDLE_PATTERNS:
        for match in pattern.finditer(text):
            handle = match.group("handle").strip(" .,:;()")
            if handle.lower() in {"rating", "expert", "profile", "handle", "competitive", "platform"}:
                continue
            if handle not in handles:
                handles.append(handle)
    return handles[:5]


def extract_codeforces(handle: str) -> ExtractionBundle:
    source = f"codeforces:{handle}"
    facts: list[ExtractedFact] = [
        ExtractedFact("links.other", f"https://codeforces.com/profile/{handle}", source, "codeforces-handle", 0.72)
    ]
    try:
        response = requests.get("https://codeforces.com/api/user.info", params={"handles": handle}, timeout=8)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return ExtractionBundle(facts, [f"codeforces:{handle}: API lookup failed: {exc}"])

    if payload.get("status") != "OK" or not payload.get("result"):
        return ExtractionBundle(facts, [f"codeforces:{handle}: API returned no user"])

    user = payload["result"][0]
    rating = user.get("rating")
    rank = user.get("rank")
    max_rating = user.get("maxRating")
    max_rank = user.get("maxRank")
    parts = [f"Codeforces handle {handle}"]
    if rating:
        parts.append(f"rating {rating}")
    if rank:
        parts.append(f"rank {rank}")
    if max_rating:
        parts.append(f"max rating {max_rating}")
    if max_rank:
        parts.append(f"max rank {max_rank}")
    facts.append(
        ExtractedFact(
            "online_coding_profile",
            {
                "codeforces": {
                    "handle": handle,
                    "profile_url": f"https://codeforces.com/profile/{handle}",
                    "rating": rating,
                    "rank": rank,
                    "max_rating": max_rating,
                    "max_rank": max_rank,
                }
            },
            source,
            "codeforces-api:user.info",
            0.74,
        )
    )
    facts.append(
        ExtractedFact(
            "achievements",
            {
                "title": "Codeforces Profile",
                "summary": ", ".join(parts),
                "links": [f"https://codeforces.com/profile/{handle}"],
            },
            source,
            "codeforces-api:user.info",
            0.66,
        )
    )
    return ExtractionBundle(facts, [])
