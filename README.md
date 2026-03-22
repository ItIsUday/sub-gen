# srt-gen

Local subtitle generator for English vocabulary practice. The app runs fully on-device with Streamlit, ffmpeg, and OpenAI Whisper.

## Features

- Local file picker for `.mp4`, `.mkv`, `.mp3`, `.m4a`, or `.wav`
- Automatic audio extraction from video via ffmpeg
- Direct transcription for audio files (no extraction step)
- Multi-track audio selection for video files
- English-only transcription with selectable Whisper model size (Tiny through Large)
- Advanced decoding controls: beam size, temperature
- Persistent in-app transcription history (latest 5)
- Download transcript as `.txt`, `.srt`, or `.md`

## Setup

1. Install system tools:

```bash
brew install uv ffmpeg
```

2. Sync dependencies:

```bash
uv sync
```

## Run

```bash
uv run streamlit run main.py
```

Open the local URL shown by Streamlit in your browser.

## Project Structure

```text
main.py                  # Thin entrypoint
srt_gen/
	app.py                 # Streamlit app orchestration
	ui.py                  # UI rendering helpers
	media.py               # ffmpeg metadata/probe/extract logic
	whisper_service.py     # OpenAI Whisper model + transcription
	history_store.py       # SQLite persistence for transcription history
	state.py               # Streamlit session state helpers
	transcript.py          # txt/srt/md formatters and record builders
	config.py              # Shared dataclasses and constants
```

Processing behavior:

- Video input (`.mp4`, `.mkv`): ffmpeg extracts mono 16kHz audio, then Whisper transcribes.
- Audio input (`.mp3`, `.m4a`, `.wav`): Whisper transcribes directly (extraction skipped).

## Model Sizes

- **Tiny** (~64MB): Fastest, lowest accuracy
- **Base** (~140MB): Fast, suitable for clear speech
- **Small** (~466MB): Good balance of speed and accuracy
- **Medium** (~1.5GB): Strong quality on varied speech
- **Large** (~2.9GB): Best accuracy, slowest inference

Models are downloaded from OpenAI's CDN on first use, then cached locally for offline use.

## Development

Run lint/format before commits:

```bash
uv run ruff check .
uv run ruff format .
```

Additional development docs:

- `AGENTS.md`: ownership boundaries and agent workflow guidance
- `SKILLS.md`: coding best practices for video + Streamlit + ML integration
- `CONTRIBUTORS.md`: contributor credits
