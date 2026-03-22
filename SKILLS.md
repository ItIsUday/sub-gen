# SKILLS.md

## Video Coding Best Practices
- Extract concerns by boundary: UI, media I/O, ML inference, persistence, formatting.
- Keep ffmpeg operations isolated from Streamlit widgets.
- Cache heavyweight model artifacts where safe; avoid repeated model downloads.
- Track stage timings for user feedback and easier performance diagnostics.
- Guard long-running tasks with clear progress states and actionable errors.

## Python/Project Practices
- Use explicit typing on public function boundaries.
- Keep module responsibilities narrow and cohesive.
- Prefer immutable configuration constants in a dedicated config module.
- Avoid duplicating transformation logic across UI and business code.
- Keep README operational: setup, run, limits, architecture, troubleshooting.

## Streamlit Practices
- Persist user-visible outputs in `st.session_state` to survive reruns.
- Keep rendering helpers pure and side-effect light.
- Use compact containers/columns for dense metadata display.
- Ensure controls are disabled only when necessary and provide guidance messages.

## Git Practices
- Make small, clean commits by concern (`feat`, `refactor`, `docs`, `chore`).
- Include lint/format pass before each push.
- Keep generated/cache directories excluded from version control.
