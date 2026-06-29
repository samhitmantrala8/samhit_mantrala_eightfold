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
- Optional GitHub profile URL
- Optional OpenRouter LLM extraction for messy text
- Mandatory profile summary generation through OpenRouter when keys are configured, with local fallback only if keys are unavailable
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

## Tests

```bash
pytest
```

The tests cover default schema output, custom projection, phone normalization, skill canonicalization, provenance, and graceful handling of sparse input.

## Optional AI Configuration

Do not commit real keys. Add them to `.env`:

```env
OPENROUTER_KEYS=key_one,key_two,key_three
USE_LLM_EXTRACTOR=true
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

The LLM extractor asks for strict JSON with evidence spans and uses temperature `0`. If one OpenRouter key is rate limited or rejected, the extractor tries the next configured key. LLM output is validated, normalized, and merged by the deterministic engine before it can affect the final profile.

Optional semantic skill matching:

```env
HF_API_TOKEN=your_huggingface_token
USE_EMBEDDINGS=true
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are only used for controlled skill canonicalization when exact alias and fuzzy matching do not resolve a mention.

## Assumptions and Descoped Items

- The current engine merges one candidate bundle at a time. For thousands of candidates, the same fact model can be grouped by email, phone, or candidate source ID before merging.
- PDF/DOCX resume parsing and scanned-document OCR are descoped from the first implementation because TXT notes already satisfy the unstructured-source requirement.
- Full ReACT agents, vector RAG, rerankers, and VLMs are intentionally not core dependencies. They add nondeterminism and operational risk for a task that is mainly explainable ETL.
