"""Tests for the time-domain shuffle gibberish engine (`MinioneseShuffle`)
and the `MinionEffect.use_shuffle` toggle that selects between it and the
STFT formant engine. Each setter is checked to (a) stay NaN/Inf-free in
block processing and (b) actually change output vs. the default.
"""
from __future__ import annotations

import numpy as np

from minion_voice.dsp.effect import MinionEffect
from minion_voice.dsp.minionese_shuffle import MinioneseShuffle

SAMPLE_RATE = 48000
BLOCK = 256


def _gen_sawtooth(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    phase = (t * freq) % 1.0
    return (0.5 * (2.0 * phase - 1.0)).astype(np.float32)


def _process_in_blocks(effect, signal: np.ndarray, block: int = BLOCK) -> np.ndarray:
    out_chunks = []
    for start in range(0, signal.shape[0], block):
        out_chunks.append(effect.process(signal[start:start + block]))
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)


def _assert_finite(out: np.ndarray) -> None:
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


def _assert_differs(a: np.ndarray, b: np.ndarray) -> None:
    n = min(a.shape[0], b.shape[0])
    assert n > 0
    assert not np.allclose(a[:n], b[:n]), "expected the setter to change output"


def _shuffle_out(seed: int = 0, configure=None) -> np.ndarray:
    m = MinioneseShuffle(SAMPLE_RATE, seed=seed)
    if configure is not None:
        configure(m)
    m.set_intensity(1.0)
    out = _process_in_blocks(m, _gen_sawtooth(150.0, 1.5))
    _assert_finite(out)
    return out


def test_shuffle_process_is_finite_and_nonempty():
    out = _shuffle_out()
    assert out.shape[0] > 0


def _dominant_freq(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    windowed = signal.astype(np.float64) * np.hanning(len(signal))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(np.abs(spectrum)))])


def _dominant_at_intensity(t: float) -> float:
    m = MinioneseShuffle(SAMPLE_RATE, seed=0)
    m.set_intensity(t)
    out = _process_in_blocks(m, _gen_sawtooth(150.0, 2.0))
    _assert_finite(out)
    n = out.shape[0]
    return _dominant_freq(out[n // 4 : 3 * n // 4])


def test_intensity_scales_pitch_to_transparent_at_zero():
    lo = _dominant_at_intensity(0.0)
    hi = _dominant_at_intensity(1.0)
    # At intensity 0 the pitch shift is 0 semitones -> fundamental unchanged
    # (~150 Hz); at full intensity it's clearly pitched up.
    assert abs(lo - 150.0) < 20.0, f"intensity 0 should preserve pitch, got {lo}"
    assert hi > lo * 1.2, f"full intensity should pitch up, got {hi} vs {lo}"


def test_intensity_zero_output_is_low_artifact_passthrough():
    m = MinioneseShuffle(SAMPLE_RATE, seed=0)
    m.set_intensity(0.0)
    sig = _gen_sawtooth(150.0, 2.0)
    out = _process_in_blocks(m, sig)
    _assert_finite(out)
    # Transparent passthrough (no crossfade ripple): the output level tracks
    # the input level closely rather than being amplitude-modulated.
    n = min(out.shape[0], sig.shape[0])
    assert n > 0
    in_rms = float(np.sqrt(np.mean(sig[:n].astype(np.float64) ** 2)))
    out_rms = float(np.sqrt(np.mean(out[:n].astype(np.float64) ** 2)))
    assert abs(out_rms - in_rms) / in_rms < 0.1


def test_shuffle_set_semitones_changes_output():
    _assert_differs(_shuffle_out(), _shuffle_out(configure=lambda m: m.set_semitones(0.0)))


def test_shuffle_set_wobble_ms_changes_output():
    _assert_differs(_shuffle_out(), _shuffle_out(configure=lambda m: m.set_wobble_ms(10.0)))


def test_shuffle_set_chunk_ms_rebuild_stays_finite_and_changes_output():
    _assert_differs(_shuffle_out(), _shuffle_out(configure=lambda m: m.set_chunk_ms(300.0)))


def test_shuffle_set_shuffle_k_rebuild_stays_finite_and_changes_output():
    _assert_differs(_shuffle_out(), _shuffle_out(configure=lambda m: m.set_shuffle_k(6)))


def test_shuffle_set_fade_ms_changes_output():
    _assert_differs(_shuffle_out(), _shuffle_out(configure=lambda m: m.set_fade_ms(5.0)))


def test_shuffle_set_reverse_prob_changes_output_and_stays_finite():
    # Full reversal probability must change the output vs. no reversal, and
    # stay click-free/finite (crossfades still stitch reversed chunks).
    _assert_differs(_shuffle_out(), _shuffle_out(configure=lambda m: m.set_reverse_prob(1.0)))


def test_reverse_prob_scales_with_intensity():
    # The gibberish character (incl. chunk reversal) is governed by intensity:
    # effective reversal probability = base * intensity.
    m = MinioneseShuffle(SAMPLE_RATE, seed=0)
    m.set_reverse_prob(1.0)
    m.set_intensity(0.5)
    assert abs(m._scramble.reverse_prob - 0.5) < 1e-9
    m.set_intensity(0.0)
    assert m._scramble.reverse_prob == 0.0
    m.set_intensity(1.0)
    assert abs(m._scramble.reverse_prob - 1.0) < 1e-9


def test_shuffle_reverse_prob_zero_is_noop():
    # reverse_prob=0 must be identical to the default (no reversal path taken).
    a = _shuffle_out()
    b = _shuffle_out(configure=lambda m: m.set_reverse_prob(0.0))
    n = min(a.shape[0], b.shape[0])
    assert np.allclose(a[:n], b[:n])


def test_shuffle_set_max_slew_bounds_sample_to_sample_delta():
    m = MinioneseShuffle(SAMPLE_RATE, seed=0)
    m.set_max_slew(0.01)
    m.set_intensity(1.0)
    out = _process_in_blocks(m, _gen_sawtooth(150.0, 1.5))
    _assert_finite(out)
    if out.shape[0] > 1:
        max_delta = float(np.max(np.abs(np.diff(out.astype(np.float64)))))
        assert max_delta <= 0.01 + 1e-6, f"slew should bound deltas to ~0.01, got {max_delta}"


def test_effect_use_shuffle_toggle_routes_to_different_engine():
    signal = _gen_sawtooth(150.0, 1.5)

    formant = MinionEffect(SAMPLE_RATE)
    formant.set_gibberish(True)
    formant.set_use_shuffle(False)
    formant.set_intensity(1.0)
    out_formant = _process_in_blocks(formant, signal)
    _assert_finite(out_formant)

    shuffle = MinionEffect(SAMPLE_RATE)
    shuffle.set_gibberish(True)
    shuffle.set_use_shuffle(True)
    shuffle.set_intensity(1.0)
    out_shuffle = _process_in_blocks(shuffle, signal)
    _assert_finite(out_shuffle)

    # The two engines are fundamentally different algorithms -- their output
    # must not coincide.
    _assert_differs(out_formant, out_shuffle)
