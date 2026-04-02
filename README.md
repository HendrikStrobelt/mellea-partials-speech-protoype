# Mellea WebRTC Voice Assistant

Real-time voice conversation system: browser mic → WebRTC → STT → LLM → TTS → browser speaker.

```
Browser Mic → WebRTC → VAD (silero) → STT (Whisper/Granite)
    → Mellea stream_with_chunking (LM Studio) → sentence chunks
    → Kokoro TTS → audio frames → WebRTC → Browser Speaker
```

The [Mellea-partials](https://github.com/HendrikStrobelt/Mellea-partials) library streams validated LLM output sentence-by-sentence, which maps naturally to per-sentence TTS synthesis for low-latency audio responses.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [LM Studio](https://lmstudio.ai/) with a model loaded and the local server running
- macOS/Linux (espeak-ng optional, for extended language support in Kokoro)

## Installation

### From PyPI (when published)

```bash
uv pip install mellea-partial-webrtc
```

### From source (uv)

```bash
git clone https://github.com/HendrikStrobelt/mellea-partial-webrtc
cd mellea-partial-webrtc
uv sync
```

> **Note:** Three dependencies are sourced outside PyPI and require uv to resolve:
> `mellea` (git), `mellea-partial` (git), `en-core-web-sm` (direct wheel URL).
> They are declared in `[tool.uv.sources]` and installed automatically by `uv sync`.

## Quick Start

```bash
# 1. Install dependencies (uv)
uv sync

# 2. Start LM Studio and load a model, then enable the local server (default: http://localhost:1234)

# 3. Start the server
mellea-webrtc
# or: uv run mellea-webrtc

# 4. Open http://localhost:8080 in your browser, click Start, and speak
```

## File Structure

```
mellea-partial-webrtc/
├── pyproject.toml              # project metadata and dependencies
├── src/
│   └── mellea_webrtc/
│       ├── __init__.py
│       ├── server.py           # aiohttp app, WebRTC signaling, entry point
│       ├── pipeline.py         # Orchestrates VAD → STT → LLM → TTS per utterance
│       ├── vad.py              # Silero-VAD wrapper, detects utterance boundaries
│       ├── stt.py              # STT protocol + Whisper & Granite backends
│       ├── llm.py              # Mellea-partials integration (stream_with_chunking)
│       ├── tts.py              # Kokoro TTS wrapper
│       ├── audio_utils.py      # Resample/format conversion helpers
│       ├── tracks.py           # Custom MediaStreamTrack for TTS output
│       └── static/
│           └── index.html      # Browser client
└── tests/
    └── test_stt.py             # Standalone WebRTC test client
```

## Configuration

All options are set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_BACKEND` | `whisper` | `whisper` or `granite` |
| `WHISPER_MODEL` | `base` | `base`, `small`, or `medium` |
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible endpoint |
| `LM_STUDIO_MODEL` | `local-model` | Model name as shown in LM Studio |
| `TTS_VOICE` | `bf_emma` | Kokoro voice (British English `bf_*` voices) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8080` | Server port |

Example with custom settings:

```bash
LM_STUDIO_MODEL="llama-3.2-3b" WHISPER_MODEL="small" mellea-webrtc
```

## STT Backends

### Whisper (default)

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), runs on CPU, no GPU required.

```bash
STT_BACKEND=whisper mellea-webrtc
```

### Granite Speech (optional)

Uses IBM Granite Speech via `transformers`. Requires a CUDA GPU and additional dependencies:

```bash
uv sync --extra granite
STT_BACKEND=granite mellea-webrtc
```

Set `GRANITE_MODEL` to override the default model (e.g. `ibm-granite/granite-speech-3.3-8b`).

## Dependencies

Key packages:

- **[aiortc](https://github.com/aiortc/aiortc)** — WebRTC server-side implementation
- **[aiohttp](https://docs.aiohttp.org/)** — async HTTP server and signaling
- **[silero-vad](https://github.com/snakers4/silero-vad)** — voice activity detection
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — STT (default backend)
- **[mellea](https://github.com/generative-computing/mellea)** — LLM orchestration framework
- **[kokoro](https://github.com/hexgrad/kokoro)** — TTS synthesis (British English `bf_emma`)
- **torch / torchaudio** — tensor ops and audio resampling

## Audio Pipeline Details

| Stage | Sample Rate | Format |
|-------|-------------|--------|
| WebRTC in/out | 48 kHz | s16 mono, 960 samples/frame (20ms) |
| VAD / STT | 16 kHz | float32 mono |
| TTS output | 24 kHz | float32 mono |

One utterance is processed at a time. When speech is detected, further utterances are queued until the current response finishes playing.

## Notes

- The browser client enables echo cancellation and noise suppression automatically.
- Kokoro requires the `en_core_web_sm` spacy model (installed automatically via `uv sync`).
- LM Studio must have the OpenAI-compatible local server enabled (default port 1234).
