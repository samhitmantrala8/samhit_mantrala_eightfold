from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from backend.transformer.pipeline import transform_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transform messy candidate inputs into a canonical profile.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files such as CSV, JSON, TXT, PDF, or DOCX.")
    parser.add_argument("--config", help="Optional custom output config JSON.")
    parser.add_argument("--github-url", help="Optional GitHub profile URL.")
    parser.add_argument("--linkedin-url", help="Optional LinkedIn profile URL.")
    parser.add_argument("--default-region", default="US", help="Default region for local phone numbers.")
    parser.add_argument("--use-llm", action="store_true", help="Enable optional legacy LLM text extraction.")
    parser.add_argument("--out", help="Write the full result bundle to this JSON file.")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    result = transform_paths(
        [Path(path) for path in args.inputs],
        config=config,
        github_url=args.github_url,
        linkedin_url=args.linkedin_url,
        default_region=args.default_region,
        use_llm=args.use_llm,
    )

    payload = json.dumps(result, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if result.get("validation_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
