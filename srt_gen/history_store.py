from __future__ import annotations

import sqlite3
from typing import Any

from .config import DB_PATH, MAX_HISTORY_ITEMS


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_type: str,
    default_sql: str,
) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in existing_columns:
        return
    conn.execute(
        f"ALTER TABLE {table_name} "
        f"ADD COLUMN {column_name} {column_type} NOT NULL DEFAULT {default_sql}"
    )


def init_db() -> None:
    with _get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcription_history (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                file_stem TEXT NOT NULL,
                transcript TEXT NOT NULL,
                txt TEXT NOT NULL,
                srt TEXT NOT NULL,
                md TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="model_name",
            column_type="TEXT",
            default_sql="''",
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="source_duration_seconds",
            column_type="REAL",
            default_sql="0",
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="is_video_input",
            column_type="INTEGER",
            default_sql="0",
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="total_time_seconds",
            column_type="REAL",
            default_sql="0",
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="stage1_time_seconds",
            column_type="REAL",
            default_sql="0",
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="stage2_time_seconds",
            column_type="REAL",
            default_sql="0",
        )
        _ensure_column(
            conn,
            table_name="transcription_history",
            column_name="stage3_time_seconds",
            column_type="REAL",
            default_sql="0",
        )


def load_history_from_db() -> list[dict[str, Any]]:
    with _get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, display_name, file_stem, model_name, source_duration_seconds, "
            "is_video_input, total_time_seconds, stage1_time_seconds, "
            "stage2_time_seconds, stage3_time_seconds, transcript, txt, srt, md, "
            "created_at "
            "FROM transcription_history "
            "ORDER BY created_at DESC LIMIT ?",
            (MAX_HISTORY_ITEMS,),
        ).fetchall()
    return [dict(row) for row in rows]


def persist_record_to_db(record: dict[str, Any]) -> None:
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO transcription_history
                (
                    id,
                    display_name,
                    file_stem,
                    model_name,
                    source_duration_seconds,
                    is_video_input,
                    total_time_seconds,
                    stage1_time_seconds,
                    stage2_time_seconds,
                    stage3_time_seconds,
                    transcript,
                    txt,
                    srt,
                    md,
                    created_at
                )
            VALUES (
                :id,
                :display_name,
                :file_stem,
                :model_name,
                :source_duration_seconds,
                :is_video_input,
                :total_time_seconds,
                :stage1_time_seconds,
                :stage2_time_seconds,
                :stage3_time_seconds,
                :transcript,
                :txt,
                :srt,
                :md,
                :created_at
            )
            """,
            record,
        )
        conn.execute(
            """
            DELETE FROM transcription_history
            WHERE id NOT IN (
                SELECT id FROM transcription_history
                ORDER BY created_at DESC LIMIT ?
            )
            """,
            (MAX_HISTORY_ITEMS,),
        )
