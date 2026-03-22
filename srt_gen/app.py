from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import ffmpeg
import streamlit as st

from .config import MODEL_OPTIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS
from .history_store import init_db
from .media import (
    check_system_dependencies,
    extract_audio_to_wav,
    get_media_metadata,
    probe_audio_tracks,
    save_uploaded_media,
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
    render_transcription_output,
    render_transcription_stats,
)
from .whisper_service import load_model_with_timing, transcribe_with_timing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("srt_gen")


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

    uploaded_file = st.file_uploader(
        "Upload media (.mp4, .mkv, .mp3, .m4a, .wav)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=False,
    )

    selected_track_index: int | None = None
    source_path: Path | None = None

    if uploaded_file:
        source_path = save_uploaded_media(uploaded_file, workspace)
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

    if st.button("Generate Subtitles", type="primary", disabled=uploaded_file is None):
        if not uploaded_file or source_path is None:
            st.warning("Upload a media file first.")
            st.stop()

        extracted_audio_path = workspace / f"audio_{uuid.uuid4().hex}.wav"
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()

        start_time = time.time()

        try:
            status_text.info("Stage 1/3: Extracting audio with ffmpeg...")
            progress_bar.progress(15)
            stage1_start = time.time()
            extract_audio_to_wav(
                source_path, extracted_audio_path, selected_track_index
            )
            stage1_time = time.time() - stage1_start
            progress_bar.progress(33)

            status_text.info(
                f"Stage 2/3: Loading Whisper {selected_model.label} model..."
            )
            progress_bar.progress(50)
            model, stage2_time = load_model_with_timing(selected_model.model_name)
            progress_bar.progress(66)

            status_text.info("Stage 3/3: Transcribing audio...")
            result, stage3_time = transcribe_with_timing(
                model,
                str(extracted_audio_path),
                beam_size,
                temperature,
            )
            progress_bar.progress(100)

            total_time = time.time() - start_time
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
            progress_container.empty()
            st.error(
                "Transcription failed, likely due to memory pressure. "
                "Try a smaller model (Tiny/Base) or shorter input clips.\n\n"
                f"{err}"
            )
            st.stop()
        except Exception as err:
            logger.exception("Unexpected transcription exception")
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
            file_name=uploaded_file.name,
            model_label=selected_model.label,
            segments=segments,
        )
        save_transcription_record(record)

    st.divider()
    st.subheader("Transcription History")
    history: list[dict[str, Any]] = st.session_state.transcription_history
    if not history:
        st.info(
            "No transcriptions yet. Upload a file and click **Generate Subtitles** to get started."
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
