from __future__ import annotations

import json
from pathlib import Path

from backend.transformer.pipeline import transform_paths


ROOT = Path(__file__).resolve().parents[1]


def test_default_profile_normalizes_and_merges_sources():
    result = transform_paths(
        [
            ROOT / "samples" / "recruiter_export.csv",
            ROOT / "samples" / "ats_profile.json",
            ROOT / "samples" / "recruiter_notes.txt",
        ],
        default_region="US",
    )
    profile = result["default_profile"]

    assert result["validation_errors"] == []
    assert profile["full_name"] == "Samhit Mantrala"
    assert profile["emails"][0] == "samhit.mantrala@example.com"
    assert profile["phones"][0] == "+14155550198"
    assert profile["links"]["github"] == "https://github.com/samhitmantrala8"
    assert any(skill["name"] == "React" for skill in profile["skills"])
    assert any(skill["name"] == "Python" for skill in profile["skills"])
    assert profile["provenance"]


def test_custom_projection_supports_renames_and_arrays():
    config = json.loads((ROOT / "configs" / "custom_output.json").read_text(encoding="utf-8"))
    result = transform_paths(
        [ROOT / "samples" / "recruiter_export.csv", ROOT / "samples" / "recruiter_notes.txt"],
        config=config,
        default_region="US",
    )
    projected = result["custom_output"]

    assert result["validation_errors"] == []
    assert projected["primary_email"] == "samhit.mantrala@example.com"
    assert projected["phone"] == "+14155550198"
    assert "React" in projected["skills"]
    assert "overall_confidence" in projected
    assert projected["provenance"]


def test_bad_or_sparse_source_degrades_gracefully():
    result = transform_paths([ROOT / "samples" / "bad_source.txt"], default_region="US")
    profile = result["default_profile"]

    assert result["validation_errors"] == []
    assert profile["candidate_id"].startswith("cand_")
    assert profile["full_name"] is None
    assert profile["emails"] == []

