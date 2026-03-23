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

        .transcription-estimate {
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 0.9rem;
            background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.92));
            padding: 1rem;
            margin-bottom: 0.8rem;
        }

        .transcription-estimate__header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin-bottom: 0.8rem;
        }

        .transcription-estimate__title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f172a;
        }

        .transcription-estimate__total {
            font-size: 0.9rem;
            color: #334155;
        }

        .transcription-estimate__grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.65rem;
        }

        .transcription-estimate__step {
            border-radius: 0.8rem;
            padding: 0.8rem 0.9rem;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            transition: all 120ms ease;
        }

        .transcription-estimate__step.is-current {
            background: #0f172a;
            border-color: #0f172a;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.14);
        }

        .transcription-estimate__step.is-complete {
            background: rgba(226, 232, 240, 0.72);
            border-color: rgba(100, 116, 139, 0.18);
        }

        .transcription-estimate__eyebrow {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.3rem;
        }

        .transcription-estimate__step.is-current .transcription-estimate__eyebrow {
            color: rgba(226, 232, 240, 0.9);
        }

        .transcription-estimate__label {
            font-size: 1rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }

        .transcription-estimate__step.is-current .transcription-estimate__label {
            color: #f8fafc;
        }

        .transcription-estimate__duration {
            font-size: 0.9rem;
            color: #475569;
        }

        .transcription-estimate__step.is-current .transcription-estimate__duration {
            color: rgba(226, 232, 240, 0.92);
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


def format_eta(seconds: float) -> str:
    rounded_seconds = max(1, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"~{hours}h {minutes}m"
    if minutes:
        return f"~{minutes}m {secs}s"
    return f"~{secs}s"


def render_transcription_estimate(
    stage_estimates: list[dict[str, Any]],
    current_stage_key: str,
    target: Any | None = None,
) -> None:
    total_seconds = sum(float(stage["expected_seconds"]) for stage in stage_estimates)
    cards: list[str] = []
    active_found = False

    for stage in stage_estimates:
        stage_key = str(stage["key"])
        state_class = ""
        step_status = "Queued"
        if stage_key == current_stage_key:
            state_class = "is-current"
            step_status = "Current"
            active_found = True
        elif not active_found:
            state_class = "is-complete"
            step_status = "Done"

        cards.append(
            "<div class='transcription-estimate__step {state_class}'>"
            "<div class='transcription-estimate__eyebrow'>{step_status}</div>"
            "<div class='transcription-estimate__label'>{label}</div>"
            "<div class='transcription-estimate__duration'>{duration}</div>"
            "</div>".format(
                state_class=state_class,
                step_status=html.escape(step_status),
                label=html.escape(str(stage["label"])),
                duration=html.escape(format_eta(float(stage["expected_seconds"]))),
            )
        )

    render_target = target if target is not None else st
    render_target.markdown(
        (
            "<div class='transcription-estimate'>"
            "<div class='transcription-estimate__header'>"
            "<div class='transcription-estimate__title'>Expected step timings</div>"
            f"<div class='transcription-estimate__total'>Estimated total {html.escape(format_eta(total_seconds))}</div>"
            "</div>"
            "<div class='transcription-estimate__grid'>"
            f"{''.join(cards)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
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
