from __future__ import annotations

import html
import io
from typing import Any

import streamlit as st

from .transcript import format_history_timestamp


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            --content-padding: 0.75rem;
            --control-spacing: 0.35rem;
        }

        .stApp [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
            max-width: 1100px;
            margin-left: auto;
            margin-right: auto;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        .stApp div[data-testid="stPopover"],
        .stApp div[data-testid="stButton"],
        .stApp div[data-testid="stDownloadButton"],
        .stApp div[data-testid="stSelectbox"] {
            padding-top: var(--control-spacing);
            padding-bottom: var(--control-spacing);
        }

        @media (max-width: 768px) {
            .stApp [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_media_metadata(metadata: dict[str, Any]) -> None:
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", metadata["duration"])
        with col2:
            st.metric("File Size", metadata["file_size"])
        with col3:
            st.metric("Bitrate", metadata["bitrate"])
        with col4:
            if metadata["has_video"]:
                st.metric("Resolution", metadata["resolution"])
            else:
                st.metric("Type", "Audio-only")

        if metadata["has_video"]:
            st.caption(f"Video codec: {metadata['video_codec']}")


def render_transcription_stats(
    total_time: float,
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    segments: list[dict[str, Any]],
) -> None:
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{total_time:.1f}s")
        with col2:
            st.metric("Segments", len(segments))
        with col3:
            avg_segment_duration = (
                (float(segments[-1]["end"]) / len(segments)) if segments else 0
            )
            st.metric("Avg Segment", f"{avg_segment_duration:.1f}s")

    st.caption(
        f"Breakdown: Extraction ({stage1_time:.1f}s) -> "
        f"Model ({stage2_time:.1f}s) -> Transcription ({stage3_time:.1f}s)"
    )


def render_transcription_output(record: dict[str, Any]) -> None:
    st.subheader("Transcript (timestamped)")
    transcript_text = html.escape(str(record.get("transcript", "")))
    st.markdown(
        (
            "<div style='white-space: pre-wrap; max-height: 420px; overflow-y: auto; "
            "padding: var(--content-padding); border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem;'>"
            f"{transcript_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.popover("Download"):
        st.download_button(
            "Download .txt",
            data=io.BytesIO(record["txt"].encode("utf-8")),
            file_name=f"{record['file_stem']}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.download_button(
            "Download .srt",
            data=io.BytesIO(record["srt"].encode("utf-8")),
            file_name=f"{record['file_stem']}.srt",
            mime="application/x-subrip",
            use_container_width=True,
        )
        st.download_button(
            "Download .md",
            data=io.BytesIO(record["md"].encode("utf-8")),
            file_name=f"{record['file_stem']}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_history_header(chosen_record: dict[str, Any] | None) -> None:
    ts_text = format_history_timestamp(
        str(chosen_record.get("created_at", "")) if chosen_record else ""
    )
    st.markdown(
        (
            "<div style='text-align:right; color:#6b7280; padding-top:0.35rem; white-space:nowrap;'>"
            f"Created {ts_text}"
            "</div>"
        )
        if ts_text
        else " ",
        unsafe_allow_html=True,
    )
