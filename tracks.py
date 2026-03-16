"""Custom MediaStreamTrack for TTS audio output."""

import asyncio
import fractions

import av
import numpy as np
from aiortc import MediaStreamTrack

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20ms at 48kHz
TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class TTSOutputTrack(MediaStreamTrack):
    """Audio output track that plays TTS-generated audio frames."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.AudioFrame | None] = asyncio.Queue()
        self._pts: int = 0

    def enqueue(self, frame: av.AudioFrame) -> None:
        """Add an audio frame to the playback queue."""
        self._queue.put_nowait(frame)

    async def recv(self) -> av.AudioFrame:
        await asyncio.sleep(FRAME_SAMPLES / SAMPLE_RATE)  # ~20ms pacing

        try:
            frame = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            frame = _silence_frame()

        frame.pts = self._pts
        frame.time_base = TIME_BASE
        self._pts += FRAME_SAMPLES
        return frame


def _silence_frame() -> av.AudioFrame:
    import numpy as np

    data = np.zeros((1, FRAME_SAMPLES), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(data, format="s16", layout="mono")
    frame.sample_rate = SAMPLE_RATE
    return frame
