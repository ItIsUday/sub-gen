from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger("srt_gen")


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = format_timestamp(float(seg["start"]))
        end = format_timestamp(float(seg["end"]))
        text = str(seg.get("text", "")).strip()
        lines.extend([str(idx), f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip() + "\n"


def to_txt(segments: list[dict[str, Any]]) -> str:
    return (
        "\n".join(str(seg.get("text", "")).strip() for seg in segments).strip() + "\n"
    )


def to_markdown(segments: list[dict[str, Any]]) -> str:
    output = ["# Transcript", ""]
    for seg in segments:
        start = format_timestamp(float(seg["start"]))
        end = format_timestamp(float(seg["end"]))
        text = str(seg.get("text", "")).strip()
        output.append(f"- **{start} -> {end}** {text}")
    return "\n".join(output).strip() + "\n"


def build_transcription_record(
    *,
    file_name: str,
    model_label: str,
    model_name: str,
    source_duration_seconds: float,
    is_video_input: bool,
    total_time_seconds: float,
    stage1_time_seconds: float,
    stage2_time_seconds: float,
    stage3_time_seconds: float,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    transcript_lines = [
        f"[{format_timestamp(float(seg['start']))} - {format_timestamp(float(seg['end']))}] "
        f"{str(seg.get('text', '')).strip()}"
        for seg in segments
    ]
    transcript_with_timestamps = "\n".join(transcript_lines)
    record_id = uuid.uuid4().hex
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": record_id,
        "display_name": f"{file_name} | {model_label}",
        "file_stem": Path(file_name).stem,
        "model_name": model_name,
        "source_duration_seconds": source_duration_seconds,
        "is_video_input": int(is_video_input),
        "total_time_seconds": total_time_seconds,
        "stage1_time_seconds": stage1_time_seconds,
        "stage2_time_seconds": stage2_time_seconds,
        "stage3_time_seconds": stage3_time_seconds,
        "transcript": transcript_with_timestamps,
        "txt": to_txt(segments),
        "srt": to_srt(segments),
        "md": to_markdown(segments),
        "created_at": created_at,
    }


def format_history_option(record: dict[str, Any]) -> str:
    display_name = str(record.get("display_name", "")).strip()
    file_name, model_name = display_name, ""
    if " | " in display_name:
        file_name, model_name = display_name.rsplit(" | ", 1)
    return f"{file_name} ({model_name})" if model_name else file_name


def format_history_timestamp(created_at: str | None) -> str:
    if not created_at:
        return ""
    try:
        dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d, %H:%M")
    except ValueError:
        logger.warning("Unexpected timestamp format: %s", created_at)
        return created_at
