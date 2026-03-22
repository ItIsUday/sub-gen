from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from .history_store import load_history_from_db, persist_record_to_db


def ensure_temp_workspace() -> Path:
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="srt_gen_")
    return Path(st.session_state.temp_dir)


def ensure_ui_state() -> None:
    if "transcription_history" not in st.session_state:
        st.session_state.transcription_history = load_history_from_db()
    if "active_transcription_id" not in st.session_state:
        history: list[dict[str, Any]] = st.session_state.transcription_history
        st.session_state.active_transcription_id = history[0]["id"] if history else None


def save_transcription_record(record: dict[str, Any]) -> None:
    persist_record_to_db(record)
    st.session_state.transcription_history = load_history_from_db()
    st.session_state.active_transcription_id = record["id"]


def get_record_by_id(record_id: str | None) -> dict[str, Any] | None:
    if not record_id:
        return None
    history: list[dict[str, Any]] = st.session_state.transcription_history
    return next((item for item in history if item["id"] == record_id), None)
