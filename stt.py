"""Speech-to-text backends."""

import asyncio
import logging
import os
from typing import Protocol

import torch

logger = logging.getLogger(__name__)

STT_BACKEND = os.environ.get("STT_BACKEND", "whisper")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")


class STTBackend(Protocol):
    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        ...


class WhisperSTT:
    """STT backend using faster-whisper (CPU-friendly)."""

    def __init__(self, model_size: str = WHISPER_MODEL) -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model: %s", model_size)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        loop = asyncio.get_event_loop()
        audio_np = audio_16k.numpy()
        logger.debug("Audio stats: shape=%s min=%.4f max=%.4f", audio_np.shape, audio_np.min(), audio_np.max())

        def _run():
            segments, info = self._model.transcribe(
                audio_np,
                beam_size=3,
                language=WHISPER_LANGUAGE,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.debug("Whisper detected language: %s (%.2f)", info.language, info.language_probability)
            return text

        text = await loop.run_in_executor(None, _run)
        logger.info("STT: %r", text)
        return text


class GraniteSpeechSTT:
    """STT backend using IBM Granite Speech (requires CUDA)."""

    def __init__(self) -> None:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        model_name = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-speech-3.3-8b")
        logger.info("Loading Granite Speech model: %s", model_name)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        loop = asyncio.get_event_loop()
        audio_np = audio_16k.numpy()

        def _run():
            inputs = self._processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(**inputs)
            text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return text

        text = await loop.run_in_executor(None, _run)
        logger.info("STT: %r", text)
        return text


def create_stt_backend() -> STTBackend:
    backend = STT_BACKEND.lower()
    if backend == "granite":
        return GraniteSpeechSTT()
    return WhisperSTT()
