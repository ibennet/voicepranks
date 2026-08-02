from __future__ import annotations

import numpy as np

from minion_voice.dsp.biquad import PeakingEQ

SAMPLE_RATE = 48000
BLOCK = 256


def _gen_sine(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _process_in_blocks(eq: PeakingEQ, signal: np.ndarray, block: int = BLOCK) -> np.ndarray:
    out_chunks = []
    n = signal.shape[0]
    for start in range(0, n, block):
        chunk = signal[start:start + block]
        out_chunks.append(eq.process(chunk))
    return np.concatenate(out_chunks)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def test_peaking_eq_boosts_center_frequency():
    eq = PeakingEQ(SAMPLE_RATE, center_hz=2000.0, q=1.0, gain_db=9.0)

    signal = _gen_sine(2000.0, 0.5)
    out = _process_in_blocks(eq, signal)

    assert out.shape[0] == signal.shape[0]
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))

    # Skip filter settling transient.
    settle = int(0.05 * SAMPLE_RATE)
    in_rms = _rms(signal[settle:])
    out_rms = _rms(out[settle:])

    assert out_rms > in_rms * 1.5, f"expected boosted RMS, got in={in_rms} out={out_rms}"


def test_peaking_eq_leaves_far_frequency_unchanged():
    eq = PeakingEQ(SAMPLE_RATE, center_hz=2000.0, q=1.0, gain_db=9.0)

    signal = _gen_sine(100.0, 0.5)
    out = _process_in_blocks(eq, signal)

    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))

    settle = int(0.05 * SAMPLE_RATE)
    in_rms = _rms(signal[settle:])
    out_rms = _rms(out[settle:])

    assert abs(out_rms - in_rms) / in_rms < 0.10, f"expected near-unchanged RMS, got in={in_rms} out={out_rms}"
