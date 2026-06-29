from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.transformer.pipeline import transform_paths


load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/transform")
    def transform():
        uploaded_files = request.files.getlist("files")
        github_url = request.form.get("github_url") or None
        config_text = request.form.get("config") or ""
        default_region = request.form.get("default_region") or os.getenv("DEFAULT_PHONE_REGION", "US")
        use_llm = (request.form.get("use_llm") or "").lower() in {"1", "true", "yes"}

        config = None
        if config_text.strip():
            try:
                config = json.loads(config_text)
            except json.JSONDecodeError as exc:
                return jsonify({"error": f"Invalid config JSON: {exc}"}), 400

        with tempfile.TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for file in uploaded_files:
                if not file.filename:
                    continue
                safe_name = Path(file.filename).name
                destination = Path(tmp) / safe_name
                file.save(destination)
                paths.append(destination)

            result = transform_paths(
                paths,
                config=config,
                github_url=github_url,
                default_region=default_region,
                use_llm=use_llm,
            )

        status = 200 if not result.get("validation_errors") else 422
        return jsonify(result), status

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

