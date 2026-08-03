"""Minimal 16-bit PCM WAV read/write helpers (stdlib `wave` only).

Moved out of `selftest.py` so both the CLI selftest tool and the
`VoiceEngine` recorder (save/load takes) and the control server (WAV
download endpoint) can share the same read/write code without importing
the CLI module.
"""
from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np


def read_wav(path: str) -> Tuple[int, np.ndarray]:
    """Read a mono (or first-channel-of-multi) 16-bit PCM WAV as float32 [-1, 1]."""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sampwidth={sampwidth}")

    data = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]

    float_data = (data.astype(np.float32)) / 32768.0
    return sample_rate, float_data


def write_wav(path: str, sample_rate: int, mono: np.ndarray) -> None:
    """Write a mono float32 [-1, 1] array as a 16-bit PCM WAV file."""
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def wav_bytes(sample_rate: int, mono: np.ndarray) -> bytes:
    """Encode a mono float32 [-1, 1] array as in-memory 16-bit PCM WAV bytes
    (used by the control server's download endpoint, no temp file needed)."""
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
