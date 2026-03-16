"""Audio format conversion helpers."""

import fractions
import numpy as np
import torch
import torchaudio.functional as F
import av


WEBRTC_SAMPLE_RATE = 48000
VAD_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
FRAME_SAMPLES = 960  # 20ms at 48kHz

_to_mono = av.AudioResampler(format='s16', layout='mono', rate=48000)


def webrtc_frame_to_pcm16k(frame: av.AudioFrame) -> torch.Tensor:
    """Convert aiortc AudioFrame (48kHz s16) to 16kHz float32 mono tensor."""
    # frame.to_ndarray() shape: (channels, samples)
    out_frames = _to_mono.resample(frame)
    arr = out_frames[0].to_ndarray()
    mono = arr[0].astype(np.float32)
    # Normalize to [-1.0, 1.0]
    mono /= 32768.0
    tensor = torch.from_numpy(mono).unsqueeze(0)  # (1, samples)
    # Resample 48kHz -> 16kHz
    resampled = F.resample(tensor, WEBRTC_SAMPLE_RATE, VAD_SAMPLE_RATE)
    return resampled.squeeze(0)  # (samples,)


def pcm24k_to_webrtc_frames(audio: np.ndarray) -> list[av.AudioFrame]:
    """Convert 24kHz float32 numpy audio to list of 48kHz s16 mono AudioFrames."""
    # audio: (samples,) at 24kHz float32
    tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    # Resample 24kHz -> 48kHz
    resampled = F.resample(tensor, TTS_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
    # Convert to int16
    samples_48k = (resampled.squeeze(0).numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)

    frames = []
    offset = 0
    while offset < len(samples_48k):
        chunk = samples_48k[offset : offset + FRAME_SAMPLES]
        # Pad last frame to FRAME_SAMPLES if needed
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        # from_ndarray expects shape (channels, samples) for s16
        frame = av.AudioFrame.from_ndarray(
            chunk.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frames.append(frame)
        offset += FRAME_SAMPLES
    return frames
