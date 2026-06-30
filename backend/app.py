from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.history import recent_llmops_examples, recent_llmops_traces, recent_transforms, record_transform, transform_by_id
from backend.transformer.gemini_hybrid import configured_gemini_keys
from backend.transformer.pipeline import transform_paths


load_dotenv()
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_UPLOAD_FILES = 5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES = {".csv", ".json", ".txt", ".md", ".pdf", ".docx"}


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/history")
    def history():
        return jsonify({"items": recent_transforms()})

    @app.get("/api/history/<int:transform_id>")
    def history_item(transform_id: int):
        result = transform_by_id(transform_id)
        if result is None:
            return jsonify({"error": "Transform not found"}), 404
        return jsonify(result)

    @app.get("/api/llmops/examples")
    def llmops_examples():
        return jsonify({"items": recent_llmops_examples()})

    @app.get("/api/llmops/traces")
    def llmops_traces():
        return jsonify({"items": recent_llmops_traces()})

    @app.post("/api/transform")
    def transform():
        request_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        uploaded_files = [file for file in request.files.getlist("files") if file.filename]
        logger.info("[%s] /api/transform received files=%s", request_id, len(uploaded_files))
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            logger.warning("[%s] rejected upload: too many files count=%s", request_id, len(uploaded_files))
            return jsonify({"error": f"Upload at most {MAX_UPLOAD_FILES} files per transform."}), 400
        github_url = request.form.get("github_url") or None
        linkedin_url = request.form.get("linkedin_url") or None
        config_text = request.form.get("config") or ""
        default_region = request.form.get("default_region") or os.getenv("DEFAULT_PHONE_REGION", "US")
        use_llm = (request.form.get("use_llm") or os.getenv("USE_LLM_EXTRACTOR", "")).lower() in {"1", "true", "yes"}
        has_gemini_keys = bool(configured_gemini_keys())
        use_gemini_hybrid = has_gemini_keys and (request.form.get("use_gemini_hybrid") or os.getenv("USE_GEMINI_HYBRID", "true")).lower() in {"1", "true", "yes", "auto"}
        use_agentic_llmops = (request.form.get("use_agentic_llmops") or os.getenv("USE_AGENTIC_LLMOPS", "true")).lower() in {"1", "true", "yes", "auto"}
        logger.info(
            "[%s] flags default_region=%s use_llm=%s use_gemini_hybrid=%s use_agentic_llmops=%s github_url=%s linkedin_url=%s config=%s",
            request_id,
            default_region,
            use_llm,
            use_gemini_hybrid,
            use_agentic_llmops,
            bool(github_url),
            bool(linkedin_url),
            bool(config_text.strip()),
        )

        config = None
        if config_text.strip():
            try:
                config = json.loads(config_text)
                logger.info("[%s] custom config parsed fields=%s", request_id, len(config.get("fields", [])) if isinstance(config, dict) else "unknown")
            except json.JSONDecodeError as exc:
                logger.exception("[%s] invalid config JSON", request_id)
                return jsonify({"error": f"Invalid config JSON: {exc}"}), 400

        with tempfile.TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for file in uploaded_files:
                safe_name = Path(file.filename).name
                suffix = Path(safe_name).suffix.lower()
                if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
                    logger.warning("[%s] rejected file name=%s suffix=%s unsupported", request_id, safe_name, suffix)
                    return jsonify({"error": f"{safe_name}: unsupported file type. Use CSV, JSON, TXT, MD, PDF, or DOCX."}), 400
                destination = Path(tmp) / safe_name
                file.save(destination)
                size = destination.stat().st_size
                logger.info("[%s] saved upload name=%s suffix=%s size_bytes=%s", request_id, safe_name, suffix, size)
                if size > MAX_UPLOAD_BYTES:
                    logger.warning("[%s] rejected file name=%s size_bytes=%s exceeds limit", request_id, safe_name, size)
                    return jsonify({"error": f"{safe_name}: file exceeds 10 MB limit."}), 400
                paths.append(destination)

            logger.info("[%s] starting transform paths=%s", request_id, [path.name for path in paths])
            result = transform_paths(
                paths,
                config=config,
                github_url=github_url,
                linkedin_url=linkedin_url,
                default_region=default_region,
                use_llm=use_llm,
                use_gemini_hybrid=use_gemini_hybrid,
                use_agentic_llmops=use_agentic_llmops,
            )

        result["history_id"] = record_transform(result, len(uploaded_files))
        status = 200 if not result.get("validation_errors") else 422
        elapsed = round(time.perf_counter() - started, 2)
        logger.info(
            "[%s] /api/transform completed status=%s history_id=%s validation_errors=%s extraction_errors=%s seconds=%s",
            request_id,
            status,
            result["history_id"],
            len(result.get("validation_errors", [])),
            len(result.get("extraction_errors", [])),
            elapsed,
        )
        return jsonify(result), status

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5055")), debug=True)
