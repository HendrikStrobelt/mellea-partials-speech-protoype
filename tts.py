"""Kokoro TTS wrapper."""

import asyncio
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

TTS_VOICE = os.environ.get("TTS_VOICE", "bf_emma")
TTS_SAMPLE_RATE = 24000


class TextToSpeech:
    """Kokoro TTS: synthesizes text to 24kHz float32 numpy audio."""

    def __init__(self) -> None:
        from kokoro import KPipeline

        logger.info("Loading Kokoro TTS pipeline (lang='b', voice=%s)", TTS_VOICE)
        self._pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")
        self._voice = TTS_VOICE

    async def synthesize(self, text: str) -> list[np.ndarray]:
        """Synthesize text into a list of 24kHz float32 audio arrays."""
        loop = asyncio.get_event_loop()

        def _run():
            chunks = []
            generator = self._pipeline(text, voice=self._voice, speed=1.0)
            for _graphemes, _phonemes, audio in generator:
                if audio is not None and len(audio) > 0:
                    # KPipeline returns torch.Tensor; convert to numpy
                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    chunks.append(audio)
            return chunks

        chunks = await loop.run_in_executor(None, _run)
        logger.debug("TTS synthesized %d chunk(s) for %r", len(chunks), text[:40])
        return chunks
