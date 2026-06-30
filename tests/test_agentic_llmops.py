from __future__ import annotations

from backend.transformer.agentic_llmops import run_agentic_llmops


def base_profile() -> dict:
    return {
        "full_name": "Test Candidate",
        "emails": ["test@example.com"],
        "phones": [],
        "location": {"city": None, "region": None, "country": None},
        "links": {"linkedin": None, "github": None, "portfolio": None, "other": []},
        "headline": None,
        "years_experience": None,
        "skills": [{"name": "Python", "confidence": 0.82, "sources": ["notes:test.txt"]}],
        "experience": [],
        "education": [],
        "projects": [],
        "achievements": [],
        "certifications": [],
        "publications": [],
        "online_coding_profile": {},
        "github_repositories": [],
        "languages": [],
        "extracurriculars": [],
        "other_sections": [],
        "others": [],
        "profile_summary": "Test Candidate has Python experience.",
        "resume_sections": {"Skills": "Python, FastAPI"},
        "semantic_mappings": [],
        "provenance": [],
        "overall_confidence": 0.72,
        "extraction_errors": [],
    }


def evaluation(score: float, verdict: str) -> dict:
    return {
        "score": score,
        "passed": score >= 8,
        "use_output": score >= 8,
        "verdict": verdict,
        "rubric_scores": {
            "correctness": score,
            "format": score,
            "evidence": score,
            "specificity": score,
        },
        "issues": [],
        "improvement_hint": "",
    }


def test_agentic_loop_refines_supported_skill_without_network(monkeypatch):
    monkeypatch.setattr("backend.transformer.agentic_llmops.configured_gemini_keys", lambda: ["key_1"])
    monkeypatch.setattr(
        "backend.transformer.agentic_llmops.CANONICAL_FIELD_SPECS",
        [{"field": "skills", "agent": "skills_agent", "kind": "list"}],
    )

    def fake_call(task_name, _system_prompt, _user_payload, _response_schema, _max_output_tokens=4096):
        event = {"task": task_name, "model": "test-model", "key_index": 1, "status": 200, "seconds": 0.01}
        if task_name == "skills_agent_evaluate_deterministic":
            return evaluation(6.0, "Deterministic skill list is incomplete."), [], [event]
        if task_name == "skills_agent_generate_loop_1":
            return {
                "rationale_summary": "The source explicitly lists FastAPI in the skills section.",
                "value_json": '["Python", "FastAPI"]',
                "confidence": 0.82,
                "evidence": "Skills section says Python, FastAPI.",
            }, [], [event]
        if task_name == "skills_agent_evaluate_loop_1":
            return evaluation(9.1, "Profile is now strong."), [], [event]
        raise AssertionError(task_name)

    monkeypatch.setattr("backend.transformer.agentic_llmops.call_gemini_json", fake_call)

    profile, trace, errors = run_agentic_llmops(
        base_profile(),
        ["Skills\nPython, FastAPI"],
        validation_errors=[],
        memory_examples=[],
        max_loops=3,
        score_threshold=8.5,
    )

    skill_names = {skill["name"] for skill in profile["skills"]}
    assert errors == []
    assert "FastAPI" in skill_names
    react_trace = next(item for item in trace["task_traces"] if item["mode"] == "react")
    assert react_trace["accepted"] is True
    assert react_trace["loops"] == 1
    assert react_trace["final_score"] == 9.1
    assert trace["good_examples"][0]["score"] == 9.1
    assert "llmops" not in profile
    assert trace["mode"] == "field-level-react-agents"


def test_agentic_loop_has_deterministic_fallback_without_keys(monkeypatch):
    monkeypatch.setattr("backend.transformer.agentic_llmops.configured_gemini_keys", lambda: [])
    monkeypatch.setattr(
        "backend.transformer.agentic_llmops.CANONICAL_FIELD_SPECS",
        [{"field": "skills", "agent": "skills_agent", "kind": "list"}],
    )

    profile, trace, errors = run_agentic_llmops(
        base_profile(),
        ["Skills\nPython"],
        validation_errors=[],
        memory_examples=[],
        max_loops=3,
    )

    assert errors == []
    assert "llmops" not in profile
    assert trace["mode"] == "field-level-deterministic-gateway"
    assert trace["request_events"] == []
    assert trace["task_traces"]
    assert all(item["mode"] == "deterministic" for item in trace["task_traces"])
    assert trace["task_traces"][0]["iterations"][0]["evaluation"]["verdict"]


def test_deterministic_field_is_discarded_when_agent_evaluator_fails(monkeypatch):
    monkeypatch.setattr("backend.transformer.agentic_llmops.configured_gemini_keys", lambda: ["key_1"])
    monkeypatch.setattr(
        "backend.transformer.agentic_llmops.CANONICAL_FIELD_SPECS",
        [{"field": "skills", "agent": "skills_agent", "kind": "list"}],
    )

    def fake_call(task_name, _system_prompt, _user_payload, _response_schema, _max_output_tokens=4096):
        event = {"task": task_name, "model": "test-model", "key_index": 1, "status": 200, "seconds": 0.01}
        if task_name == "skills_agent_evaluate_deterministic":
            return evaluation(7.5, "Deterministic output is below the pass threshold."), [], [event]
        if task_name == "skills_agent_generate_loop_1":
            return {
                "rationale_summary": "No additional supported skill value can be recovered.",
                "value_json": "[]",
                "confidence": 0.2,
                "evidence": "",
            }, [], [event]
        if task_name == "skills_agent_evaluate_loop_1":
            return evaluation(4.0, "Generated value is not good enough."), [], [event]
        raise AssertionError(task_name)

    monkeypatch.setattr("backend.transformer.agentic_llmops.call_gemini_json", fake_call)

    profile, trace, errors = run_agentic_llmops(
        base_profile(),
        ["Skills\nPython"],
        validation_errors=[],
        memory_examples=[],
        max_loops=1,
    )

    assert errors == []
    assert profile["skills"] == []
    deterministic_trace = next(item for item in trace["task_traces"] if item["mode"] == "deterministic_with_llm_evaluation")
    react_trace = next(item for item in trace["task_traces"] if item["mode"] == "react")
    assert deterministic_trace["accepted"] is False
    assert deterministic_trace["final_score"] == 7.5
    assert react_trace["accepted"] is False
    assert react_trace["final_score"] == 4.0


def test_canonical_mapping_agent_moves_supported_other_value(monkeypatch):
    monkeypatch.setattr("backend.transformer.agentic_llmops.configured_gemini_keys", lambda: ["key_1"])
    monkeypatch.setattr(
        "backend.transformer.agentic_llmops.CANONICAL_FIELD_SPECS",
        [{"field": "skills", "agent": "skills_agent", "kind": "list"}],
    )
    profile_input = base_profile()
    profile_input["skills"] = []
    profile_input["others"] = [{"title": "toolkit", "content": ["FastAPI"]}]

    def fake_call(task_name, _system_prompt, _user_payload, _response_schema, _max_output_tokens=4096):
        event = {"task": task_name, "model": "test-model", "key_index": 1, "status": 200, "seconds": 0.01}
        if task_name == "canonical_mapping_agent_generate_loop_1":
            return {
                "rationale_summary": "The unknown toolkit field contains a supported framework skill.",
                "patches_json": '[{"field": "skills", "value": ["FastAPI"]}]',
                "remaining_others_json": "[]",
                "confidence": 0.9,
                "evidence": "toolkit: FastAPI",
            }, [], [event]
        if task_name == "canonical_mapping_agent_evaluate_loop_1":
            return evaluation(9.0, "The mapping is supported."), [], [event]
        if task_name == "skills_agent_evaluate_deterministic":
            return evaluation(9.0, "Mapped skill is acceptable."), [], [event]
        raise AssertionError(task_name)

    monkeypatch.setattr("backend.transformer.agentic_llmops.call_gemini_json", fake_call)

    profile, trace, errors = run_agentic_llmops(
        profile_input,
        ["toolkit: FastAPI"],
        validation_errors=[],
        memory_examples=[],
        max_loops=1,
    )

    assert errors == []
    assert {skill["name"] for skill in profile["skills"]} == {"FastAPI"}
    assert profile["others"] == []
    mapper_trace = next(item for item in trace["task_traces"] if item["task_name"] == "canonical_mapping_agent")
    assert mapper_trace["accepted"] is True
    assert mapper_trace["final_score"] == 9.0
