from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import ffmpeg

from .config import AudioTrack


logger = logging.getLogger("srt_gen")


def pick_local_media_file(allowed_extensions: tuple[str, ...]) -> Path | None:
    """Open a native file picker and return a selected media path.

    Uses `osascript` on macOS to avoid Tkinter main-thread crashes in Streamlit.
    """
    if platform.system() != "Darwin":
        logger.warning("Native file picker currently supported on macOS only")
        return None

    extension_filters = ", ".join(f'"{ext}"' for ext in allowed_extensions)
    script = (
        'set pickedFile to choose file with prompt "Select media file" '
        f"of type {{{extension_filters}}}\n"
        "POSIX path of pickedFile"
    )

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        logger.exception("Failed to launch macOS file picker via osascript")
        return None

    if result.returncode != 0:
        # User-cancelled picker is expected and should not be treated as an error.
        logger.info("No local file selected from picker")
        return None

    selected = result.stdout.strip()
    return Path(selected) if selected else None


def check_system_dependencies() -> list[str]:
    missing = []
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            missing.append(binary)
    return missing


def get_media_metadata(media_path: Path) -> dict[str, Any]:
    logger.info("Extracting metadata from: %s", media_path)
    probe = ffmpeg.probe(str(media_path))

    format_info = probe.get("format", {})
    duration_seconds = float(format_info.get("duration", 0))
    file_size_bytes = int(format_info.get("size", 0))
    bitrate_bps = int(format_info.get("bit_rate", 0))

    def format_bytes(bytes_val: int) -> str:
        size = float(bytes_val)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def format_duration(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )

    metadata = {
        "duration": format_duration(duration_seconds),
        "duration_seconds": duration_seconds,
        "file_size": format_bytes(file_size_bytes),
        "bitrate": f"{bitrate_bps / 1_000_000:.1f} Mbps"
        if bitrate_bps > 0
        else "Unknown",
        "has_video": video_stream is not None,
        "video_codec": video_stream.get("codec_name", "unknown")
        if video_stream
        else None,
        "resolution": f"{video_stream.get('width')}x{video_stream.get('height')}"
        if video_stream
        else None,
    }

    logger.info("Extracted metadata: %s", metadata)
    return metadata


def probe_audio_tracks(media_path: Path) -> list[AudioTrack]:
    logger.info("Probing media tracks: %s", media_path)
    probe = ffmpeg.probe(str(media_path))
    tracks: list[AudioTrack] = []
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags", {})
        tracks.append(
            AudioTrack(
                stream_index=int(stream.get("index", 0)),
                codec=str(stream.get("codec_name", "unknown")),
                channels=stream.get("channels"),
                language=str(tags.get("language", "und")),
                title=str(tags.get("title", "")).strip(),
            )
        )
    logger.info("Detected %d audio track(s)", len(tracks))
    return tracks


def extract_audio_to_wav(
    source_path: Path,
    output_path: Path,
    track_index: int | None,
) -> None:
    logger.info(
        "Extracting audio from '%s' -> '%s' (track=%s)",
        source_path,
        output_path,
        track_index,
    )
    input_stream = ffmpeg.input(str(source_path))
    output_kwargs: dict[str, Any] = {
        "ac": 1,
        "ar": 16000,
        "format": "wav",
    }
    if track_index is not None:
        output_kwargs["map"] = f"0:{track_index}"
    (
        ffmpeg.output(input_stream, str(output_path), **output_kwargs)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
