from __future__ import annotations

import numpy as np

from minion_voice.dsp.minionese import Minionese

SAMPLE_RATE = 48000
BLOCK = 256


def _gen_sawtooth(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    phase = (t * freq) % 1.0
    saw = 2.0 * phase - 1.0
    return (0.5 * saw).astype(np.float32)


def _process_in_blocks(effect: Minionese, signal: np.ndarray, block: int = BLOCK) -> np.ndarray:
    out_chunks = []
    n = signal.shape[0]
    for start in range(0, n, block):
        chunk = signal[start:start + block]
        out_chunks.append(effect.process(chunk))
    return np.concatenate(out_chunks)


def test_minionese_runs_and_produces_finite_voiced_output():
    effect = Minionese(SAMPLE_RATE, seed=0)
    effect.set_intensity(1.0)

    signal = _gen_sawtooth(150.0, 2.0)
    out = _process_in_blocks(effect, signal)

    assert out.shape[0] > 0
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 0.0


def test_minionese_reset_reproducible_with_same_seed():
    signal = _gen_sawtooth(150.0, 1.0)

    effect1 = Minionese(SAMPLE_RATE, seed=42)
    out1 = _process_in_blocks(effect1, signal)

    effect2 = Minionese(SAMPLE_RATE, seed=42)
    out2 = _process_in_blocks(effect2, signal)

    assert out1.shape[0] == out2.shape[0]
    assert np.allclose(out1, out2)

    # reset() should put state back to the same starting point too.
    effect1.reset()
    out3 = _process_in_blocks(effect1, signal)
    assert out3.shape[0] == out1.shape[0]
    assert np.allclose(out3, out1)


def test_minionese_silence_in_stays_silent():
    effect = Minionese(SAMPLE_RATE, seed=0)
    effect.set_intensity(1.0)

    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out = _process_in_blocks(effect, silence)

    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    if out.shape[0] > 0:
        rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        assert rms < 1e-3
