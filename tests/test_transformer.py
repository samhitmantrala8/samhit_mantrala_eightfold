from __future__ import annotations

import json
from pathlib import Path

from backend.transformer.facts import ExtractedFact, ExtractionBundle
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


def test_resume_shaped_text_extracts_sections_without_llm(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        """
Aarav Mehta Email: aarav@example.com LinkedIn : Aarav Mehta Mobile: +91-9876543210
GitHub : github.com/aaravmehta
Education
Â• Indian Institute of Information Technology Jabalpur Jabalpur, India
Bachelor of Technology - Computer Science and Engineering; CGPA: 8.5/10 November 2022 - May 2026
Courses: Data Structures and Algorithms, Artificial Intelligence, Database Management Systems
Experience
Â• MindTickle (SDE (Applied AI) Intern) Pune, Maharashtra, India
(Team: Centre of Excellence for Machine Learning) January 2026 - Present
? Developed asynchronous RPC services using gRPC, Kafka, Redis, protobuf, Golang, LangGraph ReAct Agent, RAG, AWS OpenSearch, Cohere-Rerank-3.5, Docker, Kubernetes and Helm Charts.
Â• CREW (Machine Learning Intern) Sydney, Australia (Remote)
(Team: Machine Learning) June 2025 - October 2025
? Deployed a Flask app on Google Cloud Run using FFMPeg and Google Cloud APIs.
Projects
• CodeForces Future Rating Predictor : Github Link: June 2025
? Tech Stack: ReactJS, TailwindCSS, Flask, MongoDB.
Skills Summary
• Programming Languages and Databases: C++, Golang, Python, MongoDB, MySQL
• Frameworks: ReactJS, Flask, FastAPI, LangGraph, PyTorch
""".strip(),
        encoding="utf-8",
    )

    result = transform_paths([resume], default_region="IN")
    profile = result["default_profile"]
    skill_names = {skill["name"] for skill in profile["skills"]}

    assert result["validation_errors"] == []
    assert profile["full_name"] == "Aarav Mehta"
    assert profile["phones"] == ["+919876543210"]
    assert profile["links"]["github"] == "https://github.com/aaravmehta"
    assert profile["education"][0]["institution"] == "Indian Institute of Information Technology Jabalpur"
    assert profile["education"][0]["field"] == "Computer Science and Engineering"
    assert profile["education"][0]["end_year"] == 2026
    assert profile["education"][0]["cgpa"] == "8.5/10"
    assert profile["experience"][0]["company"] == "MindTickle"
    assert profile["experience"][0]["title"] == "SDE (Applied AI) Intern"
    assert profile["experience"][0]["role"] == "SDE (Applied AI) Intern"
    assert profile["experience"][0]["location"] == "Pune, Maharashtra, India"
    assert profile["experience"][0]["duration"] == "January 2026 - Present"
    assert profile["experience"][0]["start"] == "2026-01"
    assert profile["experience"][0]["end"] is None
    assert profile["headline"] == "SDE (Applied AI) Intern at MindTickle"
    assert profile["projects"][0]["title"] == "CodeForces Future Rating Predictor"
    assert profile["projects"][0]["date"] == "June 2025"
    assert {"ReactJS", "TailwindCSS", "Flask", "MongoDB"} <= set(profile["projects"][0]["tech_stack"])
    assert {"Go", "Kafka", "Redis", "gRPC", "Kubernetes", "LangGraph", "ReAct Agents", "C++"} <= skill_names


def test_windows_bullet_marker_does_not_leak_into_summary_or_fields(tmp_path):
    resume = tmp_path / "windows_bullet_resume.txt"
    resume.write_text(
        """
Test Candidate Email: test@example.com Mobile: +91-9876543210
Education
\x95 Test Institute Test City, India
Bachelor of Technology - Computer Science; CGPA: 9.1/10 May 2026
Experience
\x95 ExampleCo (Software Engineering Intern) Pune, Maharashtra, India
January 2026 - Present
\x95 Built APIs with Flask and Python.
""".strip(),
        encoding="utf-8",
    )

    profile = transform_paths([resume], default_region="IN")["default_profile"]

    assert profile["education"][0]["institution"] == "Test Institute Test City"
    assert profile["experience"][0]["company"] == "ExampleCo"
    assert "\x95" not in profile["profile_summary"]
    assert "•" not in profile["profile_summary"]


def test_projects_section_is_structured_and_skills_are_evidence_based(tmp_path):
    resume = tmp_path / "projects_resume.txt"
    resume.write_text(
        """
Test Candidate Email: test@example.com
Projects
• CodeForces Future Rating Predictor : Github Link: June 2025
Built a web app that predicts Codeforces users' future rating changes using Polynomial Regression, based on contest history fetched from the Codeforces API.
Visualized the predicted trend with ChartJS and used a ping monitor to keep the backend active.
Tech Stack: ReactJS, TailwindCSS, Flask, MongoDB. Frontend deployed on Netlify, backend on Render.
Tracked and stored website traffic with MongoDB; handled 10000+ requests with positive community feedback - Community Comments, Deployed link: https://cfratingpredictor.netlify.app.
• AnonGrievance - (Full Stack Development, NLP) : Github Link: April 2024
Implemented a bilingual hate/abusive text classification model to disallow vulgar text and fine-tuned using Supervised Fine-Tuning.
Developed Frontend using ReactJS and Tailwind CSS and backend using ExpressJS, NodeJS and MongoDB.
Used MongoDB's TTL indexing to keep the database clean.
Implemented Pagination and Dark/Light mode themes.
""".strip(),
        encoding="utf-8",
    )

    profile = transform_paths([resume], default_region="IN")["default_profile"]
    project_titles = [project["title"] for project in profile["projects"]]
    skill_names = {skill["name"] for skill in profile["skills"]}

    assert project_titles == ["CodeForces Future Rating Predictor", "AnonGrievance - (Full Stack Development, NLP)"]
    assert profile["projects"][0]["date"] == "June 2025"
    assert "https://cfratingpredictor.netlify.app" in profile["projects"][0]["links"]
    assert {"ReactJS", "TailwindCSS", "Flask", "MongoDB"} <= set(profile["projects"][0]["tech_stack"])
    assert {
        "Polynomial Regression",
        "Codeforces API",
        "Ping Monitoring",
        "Traffic Analytics",
        "Netlify",
        "Render",
        "Supervised Fine-Tuning",
        "Express.js",
        "Node.js",
        "MongoDB TTL Indexes",
        "Pagination",
        "Dark Mode",
        "Light Mode",
    } <= skill_names


def test_github_url_discovered_in_text_triggers_github_enrichment(tmp_path, monkeypatch):
    resume = tmp_path / "github_resume.txt"
    resume.write_text(
        """
Test Candidate Email: test@example.com
GitHub: github.com/example-user
""".strip(),
        encoding="utf-8",
    )

    def fake_extract_github(url):
        assert url == "https://github.com/example-user"
        return ExtractionBundle(
            [
                ExtractedFact("links.github", "https://github.com/example-user", "github:example-user", "github-url", 0.9),
                ExtractedFact("full_name", "Example User", "github:example-user", "github-api:name", 0.72),
                ExtractedFact("headline", "Open source developer", "github:example-user", "github-api:bio", 0.62),
            ],
            [],
        )

    monkeypatch.setattr("backend.transformer.pipeline.extract_github", fake_extract_github)
    profile = transform_paths([resume], default_region="US")["default_profile"]

    assert profile["links"]["github"] == "https://github.com/example-user"
    assert profile["full_name"] == "Test Candidate"
    assert profile["headline"] == "Open source developer"
    assert any(item["source"] == "github:example-user" for item in profile["provenance"])


def test_alternate_section_headings_parse_deterministically(tmp_path):
    resume = tmp_path / "alternate_sections.txt"
    resume.write_text(
        """
Test Candidate Email: test@example.com
Academics / Scholastic Record
IIIT Jabalpur Jabalpur, India
Bachelor of Technology - Computer Science and Engineering; CGPA: 8.5/10 November 2022 - May 2026
Professional Background
MindTickle (SDE Applied AI Intern) Pune, India
January 2026 - Present
Built services using Go, Kafka, Redis and LangGraph.
Projects / Applied Builds
CodeForces Future Rating Predictor June 2025
Tech Stack: ReactJS, TailwindCSS, Flask, MongoDB.
Achievements and Recognition
Amazon ML Challenge 2024: Secured 391st place.
Technical Strengths
Python, ReactJS, MongoDB, Docker, Kubernetes.
""".strip(),
        encoding="utf-8",
    )

    profile = transform_paths([resume], default_region="IN")["default_profile"]
    skill_names = {skill["name"] for skill in profile["skills"]}

    assert profile["education"][0]["institution"] == "IIIT Jabalpur"
    assert profile["education"][0]["cgpa"] == "8.5/10"
    assert profile["experience"][0]["company"] == "MindTickle"
    assert profile["projects"][0]["title"] == "CodeForces Future Rating Predictor"
    assert profile["achievements"][0]["title"] == "Amazon ML Challenge 2024"
    assert {"Python", "React", "MongoDB", "Docker", "Kubernetes"} <= skill_names


def test_fuzzy_csv_headers_extract_expected_fields(tmp_path):
    csv_file = tmp_path / "fuzzy.csv"
    csv_file.write_text(
        "Candidate Full Name,email_id,Mobile Number,Current Employer,Current Role,GitHub Profile,Technical Skills,CF Handle\n"
        "Test Candidate,test@example.com,+91-9876543210,ExampleCo,ML Intern,github.com/example,Python; ReactJS; Mongo DB,CinCout21\n",
        encoding="utf-8",
    )

    profile = transform_paths([csv_file], default_region="IN")["default_profile"]
    skill_names = {skill["name"] for skill in profile["skills"]}

    assert profile["full_name"] == "Test Candidate"
    assert profile["emails"] == ["test@example.com"]
    assert profile["phones"] == ["+919876543210"]
    assert profile["experience"][0]["company"] == "ExampleCo"
    assert profile["links"]["github"] == "https://github.com/example"
    assert "https://codeforces.com/profile/CinCout21" in profile["links"]["other"]
    assert {"Python", "React", "MongoDB"} <= skill_names
