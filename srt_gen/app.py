from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import ffmpeg
import streamlit as st

from .config import (
    MODEL_OPTIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from .history_store import init_db
from .media import (
    check_system_dependencies,
    extract_audio_to_wav,
    get_media_metadata,
    pick_local_media_file,
    probe_audio_tracks,
)
from .state import (
    ensure_temp_workspace,
    ensure_ui_state,
    get_record_by_id,
    save_transcription_record,
)
from .transcript import build_transcription_record, format_history_option
from .ui import (
    apply_app_styles,
    render_history_header,
    render_media_metadata,
    render_transcription_estimate,
    render_transcription_output,
    render_transcription_stats,
)
from .whisper_service import load_model_with_timing, transcribe_with_timing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("srt_gen")


def ensure_input_state() -> None:
    if "selected_local_file_path" not in st.session_state:
        st.session_state.selected_local_file_path = None


def build_stage_estimates(
    *,
    duration_seconds: float,
    is_video_input: bool,
    model_name: str,
    model_label: str,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    extraction_seconds = max(4.0, duration_seconds * 0.12) if is_video_input else 1.0
    stage_estimates: list[dict[str, Any]] = []
    model_history = [
        item
        for item in history
        if str(item.get("model_name", "")) == model_name
        and float(item.get("source_duration_seconds", 0.0)) > 0
        and float(item.get("stage3_time_seconds", 0.0)) > 0
    ]

    if model_history:
        transcription_factors = [
            float(item["stage3_time_seconds"]) / float(item["source_duration_seconds"])
            for item in model_history
        ]
        learned_realtime_factor = sum(transcription_factors) / len(
            transcription_factors
        )
    else:
        learned_realtime_factor = next(
            option.transcription_realtime_factor
            for option in MODEL_OPTIONS
            if option.model_name == model_name
        )

    if is_video_input:
        video_history = [
            item
            for item in model_history
            if bool(item.get("is_video_input"))
            and float(item.get("stage1_time_seconds", 0.0)) > 0
        ]
        if video_history:
            extraction_factors = [
                float(item["stage1_time_seconds"])
                / float(item["source_duration_seconds"])
                for item in video_history
            ]
            extraction_seconds = max(
                1.0,
                duration_seconds * (sum(extraction_factors) / len(extraction_factors)),
            )
        stage_estimates.append(
            {
                "key": "extract",
                "label": "Extract audio",
                "expected_seconds": extraction_seconds,
            }
        )

    model_config = next(
        option for option in MODEL_OPTIONS if option.model_name == model_name
    )
    load_history = [
        float(item["stage2_time_seconds"])
        for item in model_history
        if float(item.get("stage2_time_seconds", 0.0)) > 0
    ]
    stage_estimates.append(
        {
            "key": "load_model",
            "label": f"Load {model_label} model",
            "expected_seconds": (
                sum(load_history) / len(load_history)
                if load_history
                else model_config.estimated_load_seconds
            ),
        }
    )
    stage_estimates.append(
        {
            "key": "transcribe",
            "label": "Transcribe speech",
            "expected_seconds": max(3.0, duration_seconds * learned_realtime_factor),
        }
    )
    return stage_estimates


def get_estimated_total_seconds(stage_estimates: list[dict[str, Any]]) -> float:
    return sum(float(stage["expected_seconds"]) for stage in stage_estimates)


def get_stage_progress(
    stage_estimates: list[dict[str, Any]], current_stage_key: str
) -> int:
    total_seconds = get_estimated_total_seconds(stage_estimates)
    if total_seconds <= 0:
        return 0

    completed_seconds = 0.0
    for stage in stage_estimates:
        if str(stage["key"]) == current_stage_key:
            break
        completed_seconds += float(stage["expected_seconds"])

    return min(95, max(5, int((completed_seconds / total_seconds) * 100)))


def run_app() -> None:
    logger.info("App started")
    init_db()

    st.set_page_config(
        page_title="Local Subtitle Generator", page_icon=":movie_camera:", layout="wide"
    )
    apply_app_styles()

    st.title("Local Subtitle Generator")
    st.caption("Generate English subtitles locally using OpenAI Whisper and ffmpeg.")

    missing_binaries = check_system_dependencies()
    if missing_binaries:
        missing = ", ".join(missing_binaries)
        st.error(
            f"Missing required system tools: {missing}. Install with Homebrew, then restart the app."
        )
        st.stop()

    workspace = ensure_temp_workspace()
    ensure_ui_state()
    ensure_input_state()

    selected_model = st.selectbox(
        "Whisper model",
        MODEL_OPTIONS,
        index=2,
        format_func=lambda model: model.label,
        help="Larger models are slower but generally produce more accurate subtitles.",
    )
    st.info(selected_model.speed_accuracy_hint)

    with st.expander("Advanced inference settings", expanded=False):
        beam_size = st.slider(
            "Beam size",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher values can improve accuracy but increase latency.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Lower values are more deterministic. Higher values increase diversity.",
        )

    selected_track_index: int | None = None
    source_path: Path | None = None
    source_display_name: str | None = None
    metadata: dict[str, Any] | None = None

    st.caption("Source: Local file picker")
    picker_col, clear_col = st.columns([3, 1], gap="small")
    with picker_col:
        open_picker = st.button("Choose Local File", use_container_width=True)
    with clear_col:
        clear_picker = st.button("Clear", use_container_width=True)

    if clear_picker:
        st.session_state.selected_local_file_path = None

    if open_picker:
        selected_path = pick_local_media_file(SUPPORTED_EXTENSIONS)
        if selected_path is not None:
            st.session_state.selected_local_file_path = str(selected_path)

    selected_local_file = st.session_state.selected_local_file_path
    if selected_local_file:
        candidate_path = Path(selected_local_file).expanduser()
        if not candidate_path.exists():
            st.error("Selected local file no longer exists.")
            st.stop()
        if not candidate_path.is_file():
            st.error("Selected path is not a file.")
            st.stop()
        if candidate_path.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(f".{ext}" for ext in SUPPORTED_EXTENSIONS)
            st.error(f"Unsupported file extension. Allowed: {allowed}")
            st.stop()

        source_path = candidate_path
        source_display_name = candidate_path.name
        st.caption(f"Using local file: {candidate_path}")

    if source_path is not None:
        file_ext = source_path.suffix.lower()
        logger.info("Input file extension resolved as '%s'", file_ext)

        try:
            metadata = get_media_metadata(source_path)
            render_media_metadata(metadata)
        except Exception as err:
            logger.warning("Could not extract metadata: %s", err)

        if file_ext in VIDEO_EXTENSIONS:
            try:
                tracks = probe_audio_tracks(source_path)
            except ffmpeg.Error as err:
                logger.exception("Failed while probing audio tracks")
                details = (
                    err.stderr.decode("utf-8", errors="replace")
                    if err.stderr
                    else str(err)
                )
                st.error(
                    f"Could not inspect media tracks. The file may be corrupted.\n\n{details}"
                )
                st.stop()

            if not tracks:
                st.error("No audio tracks were found in this video.")
                st.stop()

            if len(tracks) > 1:
                chosen_track = st.selectbox(
                    "Select audio track",
                    tracks,
                    format_func=lambda track: track.display_name,
                )
                selected_track_index = chosen_track.stream_index
            else:
                selected_track_index = tracks[0].stream_index
                st.caption(f"Using audio track {selected_track_index}.")

    if st.button("Generate Subtitles", type="primary", disabled=source_path is None):
        if source_path is None or source_display_name is None:
            st.warning("Choose a local file first.")
            st.stop()

        file_ext = source_path.suffix.lower()
        is_video_input = file_ext in VIDEO_EXTENSIONS
        extracted_audio_path = workspace / f"audio_{uuid.uuid4().hex}.wav"
        duration_seconds = (
            float(metadata.get("duration_seconds", 0.0)) if metadata else 0.0
        )
        stage_estimates = build_stage_estimates(
            duration_seconds=duration_seconds,
            is_video_input=is_video_input,
            model_name=selected_model.model_name,
            model_label=selected_model.label,
            history=st.session_state.transcription_history,
        )
        progress_container = st.container()
        estimate_placeholder = progress_container.empty()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()

        start_time = time.time()

        try:
            if is_video_input:
                render_transcription_estimate(
                    stage_estimates, "extract", target=estimate_placeholder
                )
                status_text.info("Stage 1/3: Extracting audio with ffmpeg...")
                progress_bar.progress(get_stage_progress(stage_estimates, "extract"))
                stage1_start = time.time()
                extract_audio_to_wav(
                    source_path, extracted_audio_path, selected_track_index
                )
                stage1_time = time.time() - stage1_start
                transcription_input = extracted_audio_path

                model_status = (
                    f"Stage 2/3: Loading Whisper {selected_model.label} model..."
                )
                transcribe_status = "Stage 3/3: Transcribing audio..."
                load_stage_key = "load_model"
            else:
                stage1_time = 0.0
                transcription_input = source_path
                model_status = (
                    f"Stage 1/2: Loading Whisper {selected_model.label} model..."
                )
                transcribe_status = "Stage 2/2: Transcribing audio..."
                load_stage_key = "load_model"

            render_transcription_estimate(
                stage_estimates, load_stage_key, target=estimate_placeholder
            )
            status_text.info(model_status)
            progress_bar.progress(get_stage_progress(stage_estimates, load_stage_key))
            model, stage2_time = load_model_with_timing(selected_model.model_name)

            render_transcription_estimate(
                stage_estimates, "transcribe", target=estimate_placeholder
            )
            progress_bar.progress(get_stage_progress(stage_estimates, "transcribe"))
            status_text.info(transcribe_status)
            result, stage3_time = transcribe_with_timing(
                model,
                str(transcription_input),
                beam_size,
                temperature,
            )
            progress_bar.progress(100)

            total_time = time.time() - start_time
            estimate_placeholder.empty()
            progress_container.empty()
            st.success("Transcription completed successfully!")

            segments = result.get("segments", [])
            logger.info("Transcription produced %d segment(s)", len(segments))

            if segments:
                render_transcription_stats(
                    total_time,
                    stage1_time,
                    stage2_time,
                    stage3_time,
                    segments,
                )

        except ffmpeg.Error as err:
            logger.exception("ffmpeg extraction failed")
            estimate_placeholder.empty()
            progress_container.empty()
            details = (
                err.stderr.decode("utf-8", errors="replace") if err.stderr else str(err)
            )
            st.error(
                "Audio extraction failed. Ensure the file is valid and ffmpeg supports it.\n\n"
                f"{details}"
            )
            st.stop()
        except (RuntimeError, MemoryError) as err:
            logger.exception("Runtime or memory failure during transcription")
            estimate_placeholder.empty()
            progress_container.empty()
            st.error(
                "Transcription failed, likely due to memory pressure. "
                "Try a smaller model (Tiny/Base) or shorter input clips.\n\n"
                f"{err}"
            )
            st.stop()
        except Exception as err:
            logger.exception("Unexpected transcription exception")
            estimate_placeholder.empty()
            progress_container.empty()
            st.error(
                f"Transcription failed: {err}\n\n"
                "Check the terminal for detailed error logs."
            )
            st.stop()

        segments = result.get("segments", [])
        if not segments:
            st.warning("No speech segments were detected.")
            st.stop()

        record = build_transcription_record(
            file_name=source_display_name,
            model_label=selected_model.label,
            model_name=selected_model.model_name,
            source_duration_seconds=duration_seconds,
            is_video_input=is_video_input,
            total_time_seconds=total_time,
            stage1_time_seconds=stage1_time,
            stage2_time_seconds=stage2_time,
            stage3_time_seconds=stage3_time,
            segments=segments,
        )
        save_transcription_record(record)

    st.divider()
    st.subheader("Transcription History")
    history: list[dict[str, Any]] = st.session_state.transcription_history
    if not history:
        st.info(
            "No transcriptions yet. Choose a local file and click **Generate Subtitles** to get started."
        )
        return

    st.caption("Transcription history (last 5)")
    selector_col, ts_col = st.columns([7, 2], gap="small")
    with selector_col:
        selected_history_id = st.selectbox(
            "Transcription history (last 5)",
            options=[item["id"] for item in history],
            index=0,
            accept_new_options=False,
            format_func=lambda item_id: next(
                format_history_option(item) for item in history if item["id"] == item_id
            ),
            help="Switch between your last 5 generated transcriptions.",
            label_visibility="collapsed",
        )

    chosen_record = get_record_by_id(selected_history_id)
    with ts_col:
        render_history_header(chosen_record)

    if chosen_record is not None:
        render_transcription_output(chosen_record)
