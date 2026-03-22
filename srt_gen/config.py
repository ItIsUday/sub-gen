from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelOption:
    label: str
    model_name: str
    speed_accuracy_hint: str


@dataclass(frozen=True)
class AudioTrack:
    stream_index: int
    codec: str
    channels: int | None
    language: str
    title: str

    @property
    def display_name(self) -> str:
        channel_text = f"{self.channels}ch" if self.channels else "unknown channels"
        title = f" | {self.title}" if self.title else ""
        return (
            f"Track {self.stream_index} | {self.codec} | {channel_text} | "
            f"{self.language}{title}"
        )


SUPPORTED_EXTENSIONS = ("mp4", "mkv", "mp3", "m4a", "wav")
VIDEO_EXTENSIONS = {".mp4", ".mkv"}
MODEL_OPTIONS = [
    ModelOption(
        label="Tiny",
        model_name="tiny",
        speed_accuracy_hint="Smallest and fastest download. Lowest accuracy (~64MB).",
    ),
    ModelOption(
        label="Base",
        model_name="base",
        speed_accuracy_hint="Very fast, good for clear speech (~140MB).",
    ),
    ModelOption(
        label="Small",
        model_name="small",
        speed_accuracy_hint="Good balance of speed and accuracy (~466MB).",
    ),
    ModelOption(
        label="Medium",
        model_name="medium",
        speed_accuracy_hint="Strong quality on varied speech (~1.5GB).",
    ),
    ModelOption(
        label="Large",
        model_name="large",
        speed_accuracy_hint="Best accuracy with largest model, slowest (~2.9GB).",
    ),
]

MAX_HISTORY_ITEMS = 5
DB_PATH = Path.home() / ".srt_gen_history.db"
