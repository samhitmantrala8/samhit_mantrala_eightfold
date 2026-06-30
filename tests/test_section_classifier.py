from __future__ import annotations

from backend.transformer.section_classifier import classify_line


def test_section_headings_are_detected_with_context():
    result = classify_line(
        "Academics / Scholastic Record",
        ["Indian Institute of Information Technology Jabalpur", "Bachelor of Technology - CSE; CGPA 8.5/10"],
    )

    assert result.kind == "section"
    assert result.canonical_section == "education"


def test_skill_words_are_not_promoted_to_sections():
    for line in ["React", "Machine Learning", "RAG"]:
        result = classify_line(line, ["Built projects using this skill."])
        assert result.kind == "content"
        assert result.canonical_section is None


def test_inline_skill_field_is_content_not_heading():
    result = classify_line("Languages: Python, py, Golang, Go, C++")

    assert result.kind == "content"
    assert result.canonical_section is None


def test_ambiguous_project_management_does_not_become_projects_section():
    result = classify_line("Project Management", ["Coordinated releases and timelines."])

    assert result.kind in {"content", "ambiguous"}
    assert result.canonical_section != "projects"


def test_similar_section_names_map_to_canonical_sections():
    cases = [
        ("Professional Background", "experience"),
        ("Technical Strengths", "skills"),
        ("Selected Project Work", "projects"),
        ("Achievements and Recognition", "achievements"),
        ("Online Coding Profile Metadata", "online_coding_profile"),
    ]

    for line, section in cases:
        result = classify_line(line, ["- example content"])
        assert result.kind == "section"
        assert result.canonical_section == section


def test_contact_lines_are_content():
    result = classify_line("GitHub: github.com/example-user")

    assert result.kind == "content"
    assert result.canonical_section is None
