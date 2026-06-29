from __future__ import annotations

from urllib.parse import urlparse

import requests

from backend.transformer.extractors.github_extractor import extract_github


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self.payload


class RateLimitResponse(FakeResponse):
    def __init__(self):
        super().__init__({}, 403)

    def raise_for_status(self):
        error = requests.HTTPError("403 Client Error: rate limit exceeded")
        error.response = type("Response", (), {"status_code": 403})()
        raise error


class TextResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


def test_github_extraction_fetches_profile_repos_languages_by_default(monkeypatch):
    monkeypatch.delenv("FETCH_GITHUB_REPOS", raising=False)
    monkeypatch.setenv("GITHUB_REPO_LIMIT", "5")
    monkeypatch.setenv("GITHUB_REPO_LANGUAGE_LIMIT", "2")

    def fake_get(url, **_kwargs):
        parsed = urlparse(url)
        if parsed.path == "/users/example-user":
            return FakeResponse(
                {
                    "name": "Example User ",
                    "bio": "Builds useful developer tools.",
                    "location": "Pune, India",
                    "email": "example@example.com",
                    "blog": "example.dev",
                    "twitter_username": "example_user",
                }
            )
        if parsed.path == "/users/example-user/repos":
            return FakeResponse(
                [
                    {
                        "name": "project-one",
                        "description": "A Flask and React dashboard.",
                        "language": "Python",
                        "html_url": "https://github.com/example-user/project-one",
                        "homepage": "https://project-one.example.dev",
                        "pushed_at": "2026-02-10T12:00:00Z",
                        "stargazers_count": 7,
                        "forks_count": 1,
                    },
                    {
                        "name": "frontend-lab",
                        "description": "Frontend experiments.",
                        "language": "JavaScript",
                        "html_url": "https://github.com/example-user/frontend-lab",
                        "homepage": "",
                        "pushed_at": "2025-12-01T09:30:00Z",
                        "stargazers_count": 0,
                        "forks_count": 0,
                    },
                ]
            )
        if parsed.path == "/repos/example-user/project-one/languages":
            return FakeResponse({"Python": 12000, "HTML": 2200, "CSS": 900})
        if parsed.path == "/repos/example-user/frontend-lab/languages":
            return FakeResponse({"JavaScript": 9000, "CSS": 1800})
        raise AssertionError(f"unexpected GitHub URL: {url}")

    monkeypatch.setattr("backend.transformer.extractors.github_extractor.requests.get", fake_get)

    bundle = extract_github("https://github.com/example-user")
    facts = bundle.facts
    skill_names = {
        fact.value["name"]
        for fact in facts
        if fact.field == "skills" and isinstance(fact.value, dict)
    }
    projects = [fact.value for fact in facts if fact.field == "projects"]

    assert bundle.errors == []
    assert any(fact.field == "full_name" and fact.value == "Example User" for fact in facts)
    assert any(fact.field == "headline" and fact.value == "Builds useful developer tools." for fact in facts)
    assert any(fact.field == "emails" and fact.value == "example@example.com" for fact in facts)
    assert any(fact.field == "links.portfolio" and fact.value == "https://example.dev" for fact in facts)
    assert {"Python", "HTML", "CSS", "JavaScript"} <= skill_names
    assert [project["title"] for project in projects] == ["project-one", "frontend-lab"]
    assert {"Python", "HTML", "CSS"} <= set(projects[0]["tech_stack"])
    assert "https://project-one.example.dev" in projects[0]["links"]
    assert projects[0]["date"] == "2026-02-10"


def test_github_language_rate_limit_keeps_repo_projects(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_LIMIT", "3")
    monkeypatch.setenv("GITHUB_REPO_LANGUAGE_LIMIT", "3")

    def fake_get(url, **_kwargs):
        parsed = urlparse(url)
        if parsed.path == "/users/example-user":
            return FakeResponse({"name": "Example User"})
        if parsed.path == "/users/example-user/repos":
            return FakeResponse(
                [
                    {
                        "name": "api-service",
                        "description": "Backend API service.",
                        "language": "Python",
                        "html_url": "https://github.com/example-user/api-service",
                        "pushed_at": "2026-01-05T00:00:00Z",
                    },
                    {
                        "name": "web-client",
                        "description": "Frontend client.",
                        "language": "JavaScript",
                        "html_url": "https://github.com/example-user/web-client",
                        "pushed_at": "2026-01-04T00:00:00Z",
                    },
                ]
            )
        if parsed.path == "/repos/example-user/api-service/languages":
            return RateLimitResponse()
        raise AssertionError(f"unexpected GitHub URL after rate limit: {url}")

    monkeypatch.setattr("backend.transformer.extractors.github_extractor.requests.get", fake_get)

    bundle = extract_github("https://github.com/example-user")
    projects = [fact.value for fact in bundle.facts if fact.field == "projects"]
    skill_names = {
        fact.value["name"]
        for fact in bundle.facts
        if fact.field == "skills" and isinstance(fact.value, dict)
    }

    assert len(projects) == 2
    assert {"Python", "JavaScript"} <= skill_names
    assert bundle.errors == ["github:example-user: repo language lookup stopped because GitHub rate limit was reached"]


def test_github_web_fallback_populates_repos_when_api_is_rate_limited(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_LIMIT", "5")
    html = """
    <span class="p-name vcard-fullname" itemprop="name"> Example User </span>
    <ol>
      <li class="public source" itemprop="owns" itemscope itemtype="http://schema.org/Code">
        <a href="/example-user/web-project" itemprop="name codeRepository"> web-project </a>
        <p itemprop="description">A public web project.</p>
        <span itemprop="programmingLanguage">TypeScript</span>
        <relative-time datetime="2026-03-15T10:00:00Z">Mar 15, 2026</relative-time>
      </li>
    </ol>
    """

    def fake_get(url, **_kwargs):
        parsed = urlparse(url)
        if parsed.netloc == "api.github.com" and parsed.path == "/users/example-user":
            return RateLimitResponse()
        if parsed.netloc == "github.com" and parsed.path == "/example-user":
            return TextResponse(html)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("backend.transformer.extractors.github_extractor.requests.get", fake_get)

    bundle = extract_github("https://github.com/example-user")
    projects = [fact.value for fact in bundle.facts if fact.field == "projects"]
    skill_names = {
        fact.value["name"]
        for fact in bundle.facts
        if fact.field == "skills" and isinstance(fact.value, dict)
    }

    assert any(fact.field == "full_name" and fact.value == "Example User" for fact in bundle.facts)
    assert projects[0]["title"] == "web-project"
    assert projects[0]["date"] == "2026-03-15"
    assert projects[0]["links"] == ["https://github.com/example-user/web-project"]
    assert "TypeScript" in skill_names
    assert bundle.errors == ["github:example-user: GitHub API unavailable; used public web fallback"]
