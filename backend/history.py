from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / ".runtime" / "transform_history.sqlite"
logger = logging.getLogger(__name__)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS transformations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            candidate_id TEXT,
            full_name TEXT,
            source_count INTEGER NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS llmops_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            task_type TEXT NOT NULL,
            score REAL,
            quality TEXT NOT NULL,
            input_excerpt TEXT NOT NULL,
            output_preview TEXT NOT NULL,
            evaluator_json TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS llmops_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            task_type TEXT NOT NULL,
            quality TEXT NOT NULL,
            score REAL NOT NULL,
            input_excerpt TEXT NOT NULL,
            output_preview TEXT NOT NULL,
            evaluator_json TEXT NOT NULL
        )
        """
    )
    return connection


def record_transform(result: dict[str, Any], source_count: int) -> int:
    profile = result.get("default_profile") or {}
    payload = json.dumps(result, ensure_ascii=False)
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO transformations (created_at, candidate_id, full_name, source_count, result_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                profile.get("candidate_id"),
                profile.get("full_name"),
                source_count,
                payload,
            ),
        )
        transform_id = int(cursor.lastrowid)
        logger.info(
            "history record_transform id=%s candidate_id=%s full_name=%s source_count=%s",
            transform_id,
            profile.get("candidate_id"),
            profile.get("full_name"),
            source_count,
        )
        return transform_id


def recent_transforms(limit: int = 20) -> list[dict[str, Any]]:
    logger.info("history recent_transforms limit=%s", limit)
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, candidate_id, full_name, source_count
            FROM transformations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def transform_by_id(transform_id: int) -> dict[str, Any] | None:
    logger.info("history transform_by_id id=%s", transform_id)
    with connect() as connection:
        row = connection.execute(
            "SELECT result_json FROM transformations WHERE id = ?",
            (transform_id,),
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def record_llmops_trace(trace: dict[str, Any]) -> int:
    score = float(trace.get("final_score") or 0.0)
    quality = "good" if score >= 8.0 else "bad"
    task_type = str(trace.get("task_type") or "candidate_profile_agent")
    input_excerpt = json.dumps(trace.get("input_excerpt") or {}, ensure_ascii=False)
    output_preview = json.dumps(trace.get("output_preview") or {}, ensure_ascii=False)
    evaluator_json = json.dumps(trace.get("final_evaluation") or {}, ensure_ascii=False)
    trace_json = json.dumps(trace, ensure_ascii=False)

    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO llmops_traces (
                created_at, task_type, score, quality, input_excerpt, output_preview, evaluator_json, trace_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                task_type,
                score,
                quality,
                input_excerpt,
                output_preview,
                evaluator_json,
                trace_json,
            ),
        )
        if score >= 8.0 or score < 8.0:
            connection.execute(
                """
                INSERT INTO llmops_examples (
                    created_at, task_type, quality, score, input_excerpt, output_preview, evaluator_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    task_type,
                    quality,
                    score,
                    input_excerpt,
                    output_preview,
                    evaluator_json,
                ),
            )
        trace_id = int(cursor.lastrowid)
        logger.info("history record_llmops_trace id=%s task_type=%s quality=%s score=%s", trace_id, task_type, quality, score)
        return trace_id


def recent_llmops_examples(task_type: str = "candidate_profile_agent", limit: int = 6) -> list[dict[str, Any]]:
    logger.info("history recent_llmops_examples task_type=%s limit=%s", task_type, limit)
    per_quality = max(1, limit // 2)
    examples: list[dict[str, Any]] = []
    with connect() as connection:
        for quality in ("good", "bad"):
            rows = connection.execute(
                """
                SELECT id, created_at, task_type, quality, score, input_excerpt, output_preview, evaluator_json
                FROM llmops_examples
                WHERE task_type = ? AND quality = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (task_type, quality, per_quality),
            ).fetchall()
            for row in rows:
                item = dict(row)
                for key in ("input_excerpt", "output_preview", "evaluator_json"):
                    try:
                        item[key] = json.loads(item[key])
                    except (TypeError, json.JSONDecodeError):
                        item[key] = {}
                examples.append(item)
    return examples[:limit]


def recent_llmops_traces(limit: int = 20) -> list[dict[str, Any]]:
    logger.info("history recent_llmops_traces limit=%s", limit)
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, task_type, score, quality
            FROM llmops_traces
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
