from __future__ import annotations

import sqlite3
from typing import Any

from .config import DB_PATH, MAX_HISTORY_ITEMS


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


def load_history_from_db() -> list[dict[str, Any]]:
    with _get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, display_name, file_stem, transcript, txt, srt, md, created_at "
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
                (id, display_name, file_stem, transcript, txt, srt, md, created_at)
            VALUES (:id, :display_name, :file_stem, :transcript, :txt, :srt, :md, :created_at)
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
