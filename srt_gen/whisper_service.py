from __future__ import annotations

import logging
import time
from typing import Any

import whisper


logger = logging.getLogger("srt_gen")


def load_model_with_timing(model_name: str) -> tuple[Any, float]:
    start_time = time.time()
    model = whisper.load_model(model_name)
    elapsed = time.time() - start_time
    logger.info("Loaded model '%s' in %.2fs", model_name, elapsed)
    return model, elapsed


def transcribe_with_timing(
    model: Any,
    audio_path: str,
    beam_size: int,
    temperature: float,
) -> tuple[dict[str, Any], float]:
    start_time = time.time()
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language="en",
        beam_size=beam_size,
        temperature=temperature,
    )
    elapsed = time.time() - start_time
    logger.info("Transcribed audio in %.2fs", elapsed)
    return result, elapsed
