from __future__ import annotations

from backend.transformer.projection import project_profile


def test_projection_renames_fields_and_controls_missing_values():
    profile = {
        "full_name": "Test Candidate",
        "headline": None,
        "links": {"github": "https://github.com/example", "linkedin": None},
    }
    config = {
        "on_missing": "omit",
        "fields": [
            {"path": "candidate_name", "from": "full_name", "type": "string"},
            {"path": "github_url", "from": "links.github", "type": "string"},
            {"path": "linkedin_url", "from": "links.linkedin", "type": "string", "on_missing": "null"},
            {"path": "headline", "from": "headline", "type": "string"},
        ],
    }

    output, errors = project_profile(profile, config)

    assert errors == []
    assert output == {
        "candidate_name": "Test Candidate",
        "github_url": "https://github.com/example",
        "linkedin_url": None,
    }
