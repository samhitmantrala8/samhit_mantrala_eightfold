from __future__ import annotations

import os
import re
from html import unescape
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

import requests

from backend.transformer.facts import ExtractedFact, ExtractionBundle
from backend.transformer.normalizers.contact import normalize_url
from backend.transformer.normalizers.skills import canonicalize_skill

TRUE_VALUES = {"1", "true", "yes", "on"}
TAG_RE = re.compile(r"<[^>]+>")


def username_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url if url.startswith(("http://", "https://")) else f"https://{url}")
    if "github.com" not in parsed.netloc.lower():
        return None
    parts = [part for part in parsed.path.split("/") if part]
    return parts[0] if parts else None


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "candidate-transformer",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def get_json(url: str) -> Any:
    response = requests.get(url, headers=github_headers(), timeout=8)
    response.raise_for_status()
    return response.json()


def get_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "candidate-transformer"}, timeout=12)
    response.raise_for_status()
    return response.text


def is_rate_limit_error(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {403, 429}


def strip_html(value: str | None) -> str | None:
    if not value:
        return None
    return clean_text(unescape(TAG_RE.sub(" ", value)))


def skill_name(raw: str | None) -> str | None:
    if not raw:
        return None
    canonical = canonicalize_skill(raw)
    return canonical[0] if canonical else clean_text(raw)


def repo_date(repo: dict[str, Any]) -> str | None:
    for key in ("pushed_at", "updated_at", "created_at"):
        value = repo.get(key)
        if isinstance(value, str) and value:
            return value[:10]
    return None


def repo_links(repo: dict[str, Any]) -> list[str]:
    links: list[str] = []
    for raw in (repo.get("html_url"), repo.get("homepage")):
        url = normalize_url(raw) if isinstance(raw, str) else None
        if url and url not in links:
            links.append(url)
    return links


def repo_bullets(repo: dict[str, Any], tech_stack: list[str]) -> list[str]:
    bullets: list[str] = []
    description = clean_text(repo.get("description"))
    if description:
        bullets.append(description[:360])
    if tech_stack:
        bullets.append(f"Public GitHub repository using {', '.join(tech_stack[:5])}.")
    stars = repo.get("stargazers_count")
    forks = repo.get("forks_count")
    activity = []
    if isinstance(stars, int) and stars > 0:
        activity.append(f"{stars} stars")
    if isinstance(forks, int) and forks > 0:
        activity.append(f"{forks} forks")
    if activity:
        bullets.append("GitHub activity: " + ", ".join(activity) + ".")
    return bullets[:4]


def fetch_repositories(username: str, limit: int) -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    page = 1
    per_page = min(100, max(1, limit))
    encoded_username = quote(username, safe="")
    while len(repos) < limit:
        url = (
            f"https://api.github.com/users/{encoded_username}/repos"
            f"?per_page={per_page}&page={page}&sort=updated&type=owner"
        )
        page_items = get_json(url)
        if not isinstance(page_items, list) or not page_items:
            break
        repos.extend(item for item in page_items if isinstance(item, dict))
        if len(page_items) < per_page:
            break
        page += 1
    return repos[:limit]


def fetch_repo_languages(username: str, repo_name: str) -> dict[str, int]:
    encoded_username = quote(username, safe="")
    encoded_repo = quote(repo_name, safe="")
    data = get_json(f"https://api.github.com/repos/{encoded_username}/{encoded_repo}/languages")
    return data if isinstance(data, dict) else {}


def parse_web_profile(html: str, facts: list[ExtractedFact], source: str) -> None:
    name_match = re.search(
        r'<span[^>]+class="[^"]*\bp-name\b[^"]*"[^>]*>(?P<value>.*?)</span>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    bio_match = re.search(
        r'<div[^>]+class="[^"]*\bp-note\b[^"]*"[^>]*data-bio-text="(?P<value>[^"]*)"',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    location_match = re.search(
        r'<span[^>]+class="[^"]*\bp-label\b[^"]*"[^>]*>(?P<value>.*?)</span>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    name = strip_html(name_match.group("value")) if name_match else None
    bio = strip_html(bio_match.group("value")) if bio_match else None
    location = strip_html(location_match.group("value")) if location_match else None
    if name:
        facts.append(ExtractedFact("full_name", name, source, "github-web:name", 0.64))
    if bio:
        facts.append(ExtractedFact("headline", bio, source, "github-web:bio", 0.54))
    if location:
        facts.append(ExtractedFact("location.city", location, source, "github-web:location", 0.38))


def parse_web_repositories(username: str, html: str, limit: int) -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    item_re = re.compile(
        r'<li\b(?=[^>]*itemprop="owns")[\s\S]*?</li>',
        re.IGNORECASE,
    )
    name_re = re.compile(
        rf'<a\s+href="/{re.escape(username)}/(?P<slug>[^"]+)"[^>]+itemprop="name codeRepository"[^>]*>(?P<name>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    desc_re = re.compile(
        r'<p[^>]+itemprop="description"[^>]*>(?P<value>.*?)</p>',
        re.IGNORECASE | re.DOTALL,
    )
    language_re = re.compile(
        r'<span[^>]+itemprop="programmingLanguage"[^>]*>(?P<value>.*?)</span>',
        re.IGNORECASE | re.DOTALL,
    )
    date_re = re.compile(r'<relative-time[^>]+datetime="(?P<value>[^"]+)"', re.IGNORECASE)
    for item_match in item_re.finditer(html):
        if len(repos) >= limit:
            break
        block = item_match.group(0)
        name_match = name_re.search(block)
        if not name_match:
            continue
        name = strip_html(name_match.group("name")) or clean_text(name_match.group("slug"))
        if not name:
            continue
        description = strip_html(desc_re.search(block).group("value")) if desc_re.search(block) else None
        language = strip_html(language_re.search(block).group("value")) if language_re.search(block) else None
        updated = date_re.search(block).group("value")[:10] if date_re.search(block) else None
        repos.append(
            {
                "name": name,
                "description": description,
                "language": language,
                "html_url": f"https://github.com/{username}/{name}",
                "homepage": None,
                "pushed_at": updated,
                "stargazers_count": None,
                "forks_count": None,
            }
        )
    return repos


def fetch_web_repositories(username: str, limit: int, facts: list[ExtractedFact], source: str) -> list[dict[str, Any]]:
    html = get_text(f"https://github.com/{quote(username, safe='')}?tab=repositories")
    parse_web_profile(html, facts, source)
    return parse_web_repositories(username, html, limit)


def add_repo_facts(
    facts: list[ExtractedFact],
    errors: list[str],
    username: str,
    repos: list[dict[str, Any]],
    language_limit: int,
    method_prefix: str,
) -> None:
    source = f"github:{username}"
    language_rate_limited = False
    for index, repo in enumerate(repos):
        repo_name = clean_text(repo.get("name"))
        if not repo_name:
            continue
        repo_source = f"{source}#repo:{repo_name}"
        languages: dict[str, int] = {}
        if method_prefix == "github-api" and index < language_limit and not language_rate_limited:
            try:
                languages = fetch_repo_languages(username, repo_name)
            except requests.RequestException as exc:
                if is_rate_limit_error(exc):
                    errors.append(f"github:{username}: repo language lookup stopped because GitHub rate limit was reached")
                    language_rate_limited = True
                else:
                    errors.append(f"github:{username}/{repo_name}: language lookup failed: {exc}")

        raw_languages = [repo.get("language"), *languages.keys()]
        tech_stack: list[str] = []
        for raw_language in raw_languages:
            name = skill_name(raw_language if isinstance(raw_language, str) else None)
            if name and name not in tech_stack:
                tech_stack.append(name)
            canonical = canonicalize_skill(raw_language) if isinstance(raw_language, str) else None
            if canonical:
                facts.append(
                    ExtractedFact(
                        "skills",
                        {"name": canonical[0]},
                        repo_source,
                        f"{method_prefix}:repo-language",
                        0.62 if method_prefix == "github-api" and index < language_limit else 0.54,
                    )
                )

        facts.append(
            ExtractedFact(
                "projects",
                {
                    "title": repo_name,
                    "date": repo_date(repo),
                    "tech_stack": tech_stack,
                    "links": repo_links(repo),
                    "bullets": repo_bullets(repo, tech_stack),
                },
                repo_source,
                f"{method_prefix}:repo",
                0.6 if method_prefix == "github-api" else 0.52,
                clean_text(repo.get("description")) or repo.get("html_url"),
            )
        )


def extract_github(url: str | None) -> ExtractionBundle:
    username = username_from_url(url)
    if not username:
        return ExtractionBundle([], [])

    facts: list[ExtractedFact] = []
    errors: list[str] = []
    source = f"github:{username}"
    facts.append(ExtractedFact("links.github", f"https://github.com/{username}", source, "github-url", 0.9))

    try:
        data = get_json(f"https://api.github.com/users/{quote(username, safe='')}")
    except requests.RequestException as exc:
        if env_flag("FETCH_GITHUB_WEB_FALLBACK", True):
            try:
                repo_limit = env_int("GITHUB_REPO_LIMIT", 50, 1, 100)
                repos = fetch_web_repositories(username, repo_limit, facts, source)
                add_repo_facts(facts, errors, username, repos, 0, "github-web")
                errors.append(f"github:{username}: GitHub API unavailable; used public web fallback")
            except requests.RequestException as web_exc:
                errors.append(f"github:{username}: API lookup failed: {exc}")
                errors.append(f"github:{username}: public web fallback failed: {web_exc}")
        else:
            errors.append(f"github:{username}: API lookup failed: {exc}")
        return ExtractionBundle(facts, errors)

    if isinstance(data, dict):
        name = clean_text(data.get("name"))
        bio = clean_text(data.get("bio"))
        location = clean_text(data.get("location"))
        email = clean_text(data.get("email"))
        blog = normalize_url(data.get("blog")) if isinstance(data.get("blog"), str) else None
        twitter_username = clean_text(data.get("twitter_username"))
        if name:
            facts.append(ExtractedFact("full_name", name, source, "github-api:name", 0.72))
        if bio:
            facts.append(ExtractedFact("headline", bio, source, "github-api:bio", 0.62))
        if location:
            facts.append(ExtractedFact("location.city", location, source, "github-api:location", 0.45))
        if email:
            facts.append(ExtractedFact("emails", email, source, "github-api:public-email", 0.58))
        if blog:
            facts.append(ExtractedFact("links.portfolio", blog, source, "github-api:blog", 0.58))
        if twitter_username:
            facts.append(ExtractedFact("links.other", f"https://x.com/{twitter_username}", source, "github-api:twitter", 0.42))

    if env_flag("FETCH_GITHUB_REPOS", True):
        repo_limit = env_int("GITHUB_REPO_LIMIT", 50, 1, 100)
        language_limit = env_int("GITHUB_REPO_LANGUAGE_LIMIT", 20, 0, repo_limit)
        try:
            repos = fetch_repositories(username, repo_limit)
        except requests.RequestException as exc:
            if env_flag("FETCH_GITHUB_WEB_FALLBACK", True):
                try:
                    repos = fetch_web_repositories(username, repo_limit, facts, source)
                    add_repo_facts(facts, errors, username, repos, 0, "github-web")
                    errors.append(f"github:{username}: GitHub repo API unavailable; used public web fallback")
                except requests.RequestException as web_exc:
                    errors.append(f"github:{username}: repo lookup failed: {exc}")
                    errors.append(f"github:{username}: public web fallback failed: {web_exc}")
            else:
                errors.append(f"github:{username}: repo lookup failed: {exc}")
            return ExtractionBundle(facts, errors)

        add_repo_facts(facts, errors, username, repos, language_limit, "github-api")

    return ExtractionBundle(facts, errors)
