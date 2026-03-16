"""Orchestrates VAD → STT → LLM → TTS per utterance."""

import asyncio
import json
import logging
from typing import Any

import av
import numpy as np
import torch
import torchaudio.functional as AF

from audio_utils import pcm24k_to_webrtc_frames
from llm import generate_response
from stt import create_stt_backend, STTBackend
from tracks import TTSOutputTrack
from tts import TextToSpeech
from vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

_RESAMPLE_BUF_SAMPLES = 9600  # 200ms at 48kHz before batch-resampling
_to_mono = av.AudioResampler(format='s16', layout='mono', rate=48000)


class AudioPipeline:
    """Wires together VAD, STT, LLM, TTS and feeds audio to the output track."""

    def __init__(self, output_track: TTSOutputTrack, log_channel: Any = None) -> None:
        self._output = output_track
        self._log_channel = log_channel
        self._stt: STTBackend = create_stt_backend()
        self._tts = TextToSpeech()
        self._vad = VoiceActivityDetector(on_utterance=None)  # sync push API
        self._busy = False  # process one utterance at a time
        self._resample_buf: list[np.ndarray] = []  # accumulated int16 mono at 48kHz

    def _emit(self, event: str, **data) -> None:
        if not self._log_channel:
            return
        if self._log_channel.readyState != "open":
            logger.debug(
                "Log channel not open (state=%s), dropping: %s",
                self._log_channel.readyState,
                event,
            )
            return
        self._log_channel.send(json.dumps({"event": event, **data}))

    def feed_audio(self, frame: av.AudioFrame) -> None:
        """Called with each incoming WebRTC audio frame from the browser mic."""
        # Extract int16 mono samples at 48kHz into the buffer
        out_frames = _to_mono.resample(frame)
        arr = out_frames[0].to_ndarray()  # (1, 960) — true mono
        mono = arr[0]
        self._resample_buf.append(mono)

        total = sum(len(b) for b in self._resample_buf)
        if total < _RESAMPLE_BUF_SAMPLES:
            return  # keep accumulating

        # Batch-resample 200ms of audio at once to avoid sinc edge artifacts
        combined = np.concatenate(self._resample_buf)
        self._resample_buf = []
        float_mono = combined.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(float_mono).unsqueeze(0)  # (1, samples)
        resampled = AF.resample(tensor, 48000, 16000)
        pcm = resampled.squeeze(0)  # (samples,) at 16kHz float32

        utterances = self._vad.push(pcm)
        for utterance in utterances:
            duration = round(utterance.shape[-1] / 16000, 2)
            self._emit("vad_utterance", duration=duration)
            if self._busy:
                logger.warning("Pipeline busy, dropping utterance (%.2fs)", duration)
            else:
                asyncio.ensure_future(self._process_utterance(utterance))

    async def _process_utterance(self, audio: torch.Tensor) -> None:
        self._busy = True
        try:
            # STT
            text = await self._stt.transcribe(audio)
            if not text:
                logger.debug("Empty transcription, skipping")
                self._emit("stt_empty")
                return
            self._emit("stt_result", text=text)

            # LLM → sentence queue
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            self._emit("llm_start", prompt=text)
            asyncio.ensure_future(self._safe_generate(text, sentence_queue))

            # TTS each sentence as it arrives
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                self._emit("llm_sentence", sentence=sentence)
                await self._synthesize_and_enqueue(sentence)

            self._emit("llm_done")

        except Exception as exc:
            logger.exception("Pipeline error")
            self._emit("pipeline_error", error=str(exc))
        finally:
            self._busy = False

    async def _safe_generate(self, text: str, sentence_queue: asyncio.Queue) -> None:
        """Wrap generate_response to guarantee the sentinel is always put."""
        try:
            await generate_response(text, sentence_queue)
        except Exception:
            logger.exception("LLM generation failed")
            self._emit("pipeline_error", error="LLM generation failed")
        finally:
            await sentence_queue.put(None)  # guarantee consumer loop exits

    async def _synthesize_and_enqueue(self, sentence: str) -> None:
        self._emit("tts_start", sentence=sentence)
        chunks = await self._tts.synthesize(sentence)
        for audio_chunk in chunks:
            frames = pcm24k_to_webrtc_frames(audio_chunk)
            for frame in frames:
                self._output.enqueue(frame)
        self._emit("tts_done")
