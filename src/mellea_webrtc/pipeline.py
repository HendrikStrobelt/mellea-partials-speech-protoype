"""Orchestrates VAD → STT → LLM → TTS per utterance."""

import asyncio
import json
import logging
import os
from typing import Any

import av
import numpy as np
import torch
import torchaudio.functional as AF

from mellea_webrtc.audio_utils import pcm24k_to_webrtc_frames
from mellea_webrtc.llm import generate_response
from mellea_webrtc.stt import create_stt_backend, STTBackend
from mellea_webrtc.tracks import TTSOutputTrack
from mellea_webrtc.tts import TextToSpeech
from mellea_webrtc.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

_RESAMPLE_BUF_SAMPLES = 9600  # 200ms at 48kHz before batch-resampling
_BARGEIN_ENERGY_THRESHOLD = float(os.environ.get("BARGEIN_ENERGY_THRESHOLD", "0.005"))
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
        self._generation_task: asyncio.Task | None = None
        self._llm_task: asyncio.Task | None = None
        self._llm_result = None  # StreamChunkingResult | None
        self._generation_epoch: int = 0

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

        was_in_speech = self._vad._in_speech
        utterances = self._vad.push(pcm)

        # Fast barge-in: interrupt on speech onset, not complete utterance
        if self._busy and self._vad._in_speech and not was_in_speech:
            batch_energy = float(pcm.pow(2).mean().sqrt())
            if batch_energy >= _BARGEIN_ENERGY_THRESHOLD:
                logger.info("Speech onset barge-in (energy=%.4f)", batch_energy)
                asyncio.ensure_future(self._interrupt(reset_vad=False))
                return  # skip utterance processing this cycle

        for utterance in utterances:
            duration = round(utterance.shape[-1] / 16000, 2)
            self._emit("vad_utterance", duration=duration)
            if self._busy:
                if self._is_likely_echo(utterance):
                    logger.debug("Barge-in suppressed: likely echo (%.2fs)", duration)
                else:
                    asyncio.ensure_future(self._handle_bargein(utterance))
            else:
                self._output.flush()  # clear any lingering TTS audio
                self._generation_task = asyncio.ensure_future(self._process_utterance(utterance))

    def _is_likely_echo(self, utterance: torch.Tensor) -> bool:
        """Return True if utterance energy is too low to be real speech (echo residual)."""
        energy = float(utterance.pow(2).mean().sqrt())
        return energy < _BARGEIN_ENERGY_THRESHOLD

    async def _interrupt(self, reset_vad: bool = True) -> None:
        """Silence output, cancel in-flight LLM/TTS, reset state."""
        flushed = self._output.flush()
        logger.debug("Barge-in: flushed %d queued frames", flushed)

        self._generation_epoch += 1

        if self._llm_result is not None and hasattr(self._llm_result, "_task"):
            self._llm_result._task.cancel()

        if self._llm_task is not None and not self._llm_task.done():
            self._llm_task.cancel()
            try:
                await self._llm_task
            except (asyncio.CancelledError, Exception):
                pass
        self._llm_task = None

        if self._generation_task is not None and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except (asyncio.CancelledError, Exception):
                pass

        self._llm_result = None
        self._generation_task = None
        self._busy = False
        if reset_vad:
            self._vad.reset()
        self._emit("barge_in")
        logger.info("Barge-in: pipeline interrupted")

    async def _handle_bargein(self, utterance: torch.Tensor) -> None:
        await self._interrupt()
        self._generation_task = asyncio.ensure_future(self._process_utterance(utterance))

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
            epoch = self._generation_epoch
            self._llm_task = asyncio.ensure_future(self._safe_generate(text, sentence_queue))

            # TTS each sentence as it arrives
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                if epoch != self._generation_epoch:
                    break
                self._emit("llm_sentence", sentence=sentence)
                await self._synthesize_and_enqueue(sentence, epoch)

            self._emit("llm_done")

        except asyncio.CancelledError:
            logger.debug("_process_utterance cancelled (barge-in)")
            return
        except Exception as exc:
            logger.exception("Pipeline error")
            self._emit("pipeline_error", error=str(exc))
        finally:
            self._busy = False

    async def _safe_generate(self, text: str, sentence_queue: asyncio.Queue) -> None:
        """Wrap generate_response to guarantee the sentinel is always put."""
        try:
            self._llm_result = await generate_response(text, sentence_queue)
        except asyncio.CancelledError:
            logger.debug("LLM generation cancelled (barge-in)")
            raise
        except Exception:
            logger.exception("LLM generation failed")
            self._emit("pipeline_error", error="LLM generation failed")
        finally:
            self._llm_result = None
            await sentence_queue.put(None)  # guarantee consumer loop exits

    async def _synthesize_and_enqueue(self, sentence: str, epoch: int) -> None:
        self._emit("tts_start", sentence=sentence)
        chunks = await self._tts.synthesize(sentence)
        if epoch != self._generation_epoch:
            logger.debug("Stale TTS output discarded (epoch %d != %d)", epoch, self._generation_epoch)
            return
        if chunks:
            combined = np.concatenate(chunks)
            frames = pcm24k_to_webrtc_frames(combined)
            for frame in frames:
                self._output.enqueue(frame)
        self._emit("tts_done")
