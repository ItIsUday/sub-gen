# AGENTS.md

## Purpose
This repository contains a local subtitle generation app built with Streamlit, ffmpeg, and OpenAI Whisper.

## Agent Roles
- `App Agent`: Owns `srt_gen/app.py` orchestration and Streamlit flow.
- `UI Agent`: Owns `srt_gen/ui.py` and all rendering/layout helpers.
- `Media Agent`: Owns `srt_gen/media.py` for ffmpeg/probe/extraction logic.
- `ML Agent`: Owns `srt_gen/whisper_service.py` for model loading/transcription.
- `History Agent`: Owns `srt_gen/history_store.py` and `srt_gen/state.py` persistence/session behavior.
- `Transcript Agent`: Owns `srt_gen/transcript.py` output formatting and record creation.

## Guardrails
- Keep `main.py` as a thin entrypoint only.
- Preserve local-first design: no server-side transcription.
- Prefer native local-file selection over browser uploads.
- Keep transcription history capped at 5 items unless requirements change.
- Prefer small, testable functions and avoid large monolithic handlers.
- Run `uv run ruff check .` and `uv run ruff format .` before committing.

## PR Checklist
- App still launches with `uv run streamlit run main.py`.
- Local-file select/download flow persists across reruns.
- History selector correctly loads latest 5 records.
- Large local video selection still transcribes without browser transport errors.
