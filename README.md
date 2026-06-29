# Multi-Source Candidate Data Transformer

React + Tailwind frontend, Flask backend, and a deterministic Python transformer for the Eightfold engineering intern assignment.

The core pipeline is:

```text
load sources -> extract facts -> normalize -> merge/confidence -> canonical profile -> project custom output -> validate
```

The implementation supports:

- Recruiter CSV export
- ATS JSON blob
- Recruiter notes TXT
- GitHub profile, public repository, project, and language enrichment through the public GitHub API, either from the URL field or from a GitHub URL found in uploaded text
- LinkedIn profile URL normalization/storage when a real `linkedin.com/...` URL is supplied
- Structured resume projects with title, date, tech stack, links, and evidence bullets
- Optional OpenRouter LLM extraction for messy text
- Mandatory profile summary generation through OpenRouter when a key is configured, with local fallback only if a key is unavailable
- Optional Hugging Face embedding matching for semantic skill canonicalization

LLM and embedding helpers are deliberately optional. The transformer still runs end-to-end without network keys.

## Backend Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## CLI Demo

```bash
python -m backend.cli --inputs samples/recruiter_export.csv samples/ats_profile.json samples/recruiter_notes.txt --config configs/custom_output.json --out outputs/demo_result.json
```

The command prints a bundle containing:

- `default_profile`
- `custom_output`
- `extraction_errors`
- `validation_errors`

## Flask API

```bash
python -m flask --app backend.app run --host 127.0.0.1 --port 5055
```

Health check:

```bash
curl http://127.0.0.1:5055/api/health
```

Main endpoint:

```text
POST /api/transform
multipart form fields:
- files: one or more CSV, JSON, TXT, or MD files
- config: optional runtime projection JSON
- github_url: optional GitHub profile URL
- linkedin_url: optional LinkedIn profile URL
- default_region: phone parsing region, default US
- use_llm: true or false
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5177
```

This repo defaults to Flask `5055` and Vite `5177` to avoid collisions with the common Flask `5000` port.

The Vite dev server proxies `/api` to Flask on port `5055`.

GitHub repo enrichment is enabled by default and is bounded by `GITHUB_REPO_LIMIT` and `GITHUB_REPO_LANGUAGE_LIMIT`. LinkedIn handling is intentionally conservative: the transformer stores and normalizes real LinkedIn URLs, but it does not scrape LinkedIn or guess a LinkedIn URL from a person's name. If LinkedIn-like profile facts are included in an uploaded source file, they can still be extracted from that source.

## Tests

```bash
pytest
```

The tests cover default schema output, custom projection, phone normalization, skill canonicalization, provenance, and graceful handling of sparse input.

## Optional AI Configuration

Do not commit a real key. Add it to `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
USE_LLM_EXTRACTOR=true
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

The LLM extractor asks for strict JSON with evidence spans and uses temperature `0`. LLM output is validated, normalized, and merged by the deterministic engine before it can affect the final profile.

Optional semantic skill matching:

```env
HF_API_TOKEN=your_huggingface_token
USE_EMBEDDINGS=true
HF_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

Embeddings are only used for controlled skill canonicalization when exact alias and fuzzy matching do not resolve a mention.

## Assumptions and Descoped Items

- The current engine merges one candidate bundle at a time. For thousands of candidates, the same fact model can be grouped by email, phone, or candidate source ID before merging.
- PDF/DOCX resume parsing and scanned-document OCR are descoped from the first implementation because TXT notes already satisfy the unstructured-source requirement.
- Full ReACT agents, vector RAG, rerankers, and VLMs are intentionally not core dependencies. They add nondeterminism and operational risk for a task that is mainly explainable ETL.
