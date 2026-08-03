from __future__ import annotations

import numpy as np

from voicepranks.dsp.effect import MinionEffect

SAMPLE_RATE = 48000
BLOCK = 256


def _gen_sine(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _process_in_blocks(effect: MinionEffect, signal: np.ndarray, block: int = BLOCK) -> np.ndarray:
    out_chunks = []
    n = signal.shape[0]
    for start in range(0, n, block):
        chunk = signal[start:start + block]
        out_chunks.append(effect.process(chunk))
    return np.concatenate(out_chunks)


def _dominant_freq(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    windowed = signal * np.hanning(len(signal))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    mag = np.abs(spectrum)
    peak_idx = np.argmax(mag)
    return float(freqs[peak_idx])


def test_full_intensity_shifts_pitch_up():
    effect = MinionEffect(SAMPLE_RATE)
    effect.set_intensity(1.0)

    signal = _gen_sine(220.0, 1.0)
    out = _process_in_blocks(effect, signal)

    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))

    # WSOLA is a streaming algorithm and may emit a different number of
    # samples than went in -- pick the middle of whatever came out.
    n = out.shape[0]
    start = int(0.25 * n)
    end = int(0.75 * n)
    middle = out[start:end]

    dominant = _dominant_freq(middle)
    assert dominant > 220.0 * 1.2, f"expected clearly shifted pitch, got {dominant}"


def test_zero_intensity_is_near_passthrough_pitch():
    effect = MinionEffect(SAMPLE_RATE)
    effect.set_intensity(0.0)

    signal = _gen_sine(220.0, 1.0)
    out = _process_in_blocks(effect, signal)

    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))

    n = out.shape[0]
    start = int(0.25 * n)
    end = int(0.75 * n)
    middle = out[start:end]

    dominant = _dominant_freq(middle)
    assert abs(dominant - 220.0) / 220.0 < 0.05, f"dominant freq {dominant} not within 5% of 220"
