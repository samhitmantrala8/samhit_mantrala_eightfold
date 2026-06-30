from __future__ import annotations

from backend.transformer.agentic_llmops import run_agentic_llmops


def base_profile() -> dict:
    return {
        "candidate_id": "cand_test",
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
        "issues": [],
        "improvement_hints": [],
    }


def test_agentic_loop_refines_supported_skill_without_network(monkeypatch):
    monkeypatch.setattr("backend.transformer.agentic_llmops.configured_gemini_keys", lambda: ["key_1"])

    def fake_call(task_name, _system_prompt, _user_payload, _response_schema, _max_output_tokens=4096):
        event = {"task": task_name, "model": "test-model", "key_index": 1, "status": 200, "seconds": 0.01}
        if task_name == "agent_task_decomposition":
            return {
                "tasks": [
                    {
                        "name": "summary_and_skill_refinement",
                        "purpose": "Improve the summary and recover evidence-backed missing skills.",
                        "mode": "react",
                        "target_fields": ["profile_summary", "skills"],
                        "priority": 1,
                    }
                ]
            }, [], [event]
        if task_name == "react_summary_and_skill_refinement_generate_loop_1":
            return {
                "rationale_summary": "The source explicitly lists FastAPI in the skills section.",
                "confidence": 0.82,
                "proposed_output": {
                    "profile_summary": "Test Candidate has Python and FastAPI experience supported by the source.",
                    "skills_add": ["FastAPI"],
                    "skills_remove": [],
                    "notes": "Add only the supported FastAPI skill.",
                },
            }, [], [event]
        if task_name == "react_summary_and_skill_refinement_evaluate_loop_1":
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
    assert profile["llmops"]["mode"] == "per-task-react-agents"


def test_agentic_loop_has_deterministic_fallback_without_keys(monkeypatch):
    monkeypatch.setattr("backend.transformer.agentic_llmops.configured_gemini_keys", lambda: [])

    profile, trace, errors = run_agentic_llmops(
        base_profile(),
        ["Skills\nPython"],
        validation_errors=[],
        memory_examples=[],
        max_loops=3,
    )

    assert errors == []
    assert profile["llmops"]["mode"] == "deterministic-fallback-evaluator"
    assert trace["request_events"] == []
    assert trace["task_traces"]
    assert all(item["mode"] == "deterministic" for item in trace["task_traces"])
    assert trace["task_traces"][0]["iterations"][0]["evaluation"]["verdict"]
