"""Tests for the live-adjustable setters added to the DSP stages so every
knob in `params.PARAM_SPECS` can actually be tuned at runtime. Each test
checks the setter (a) doesn't blow up NaN/Inf in block processing and
(b) actually changes behavior relative to the default.
"""
from __future__ import annotations

import time

import numpy as np

from minion_voice.dsp.effect import MinionEffect
from minion_voice.dsp.minionese import Minionese
from minion_voice.dsp.pitch import PitchShifter
from minion_voice.dsp.ramp import IntensityRamp

SAMPLE_RATE = 48000
BLOCK = 256


def _gen_sine(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _gen_sawtooth(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    phase = (t * freq) % 1.0
    saw = 2.0 * phase - 1.0
    return (0.5 * saw).astype(np.float32)


def _process_in_blocks(effect, signal: np.ndarray, block: int = BLOCK) -> np.ndarray:
    out_chunks = []
    n = signal.shape[0]
    for start in range(0, n, block):
        chunk = signal[start:start + block]
        out_chunks.append(effect.process(chunk))
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)


def _dominant_freq(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    windowed = signal * np.hanning(len(signal))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    mag = np.abs(spectrum)
    peak_idx = np.argmax(mag)
    return float(freqs[peak_idx])


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def _assert_finite(out: np.ndarray) -> None:
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


# -- MinionEffect ----------------------------------------------------------


def test_effect_set_max_semitones_changes_shift_amount():
    signal = _gen_sine(220.0, 1.0)

    default_effect = MinionEffect(SAMPLE_RATE)
    default_effect.set_intensity(1.0)
    default_out = _process_in_blocks(default_effect, signal)
    _assert_finite(default_out)

    lowered = MinionEffect(SAMPLE_RATE)
    lowered.set_max_semitones(2.0)
    lowered.set_intensity(1.0)
    lowered_out = _process_in_blocks(lowered, signal)
    _assert_finite(lowered_out)

    def _mid_dominant(out):
        n = out.shape[0]
        return _dominant_freq(out[int(0.25 * n):int(0.75 * n)])

    assert _mid_dominant(lowered_out) < _mid_dominant(default_out)


def test_effect_set_eq_enabled_routes_eq_into_process_chain():
    signal = _gen_sine(2000.0, 1.0)
    settle = int(0.05 * SAMPLE_RATE)

    bypassed = MinionEffect(SAMPLE_RATE)
    bypassed.set_max_semitones(0.0)
    bypassed.set_eq_center_hz(2000.0)
    bypassed.set_eq_gain_db(12.0)
    bypassed.set_intensity(1.0)  # eq_enabled still False -- must stay bypassed
    bypassed_out = _process_in_blocks(bypassed, signal)
    _assert_finite(bypassed_out)

    boosted = MinionEffect(SAMPLE_RATE)
    boosted.set_max_semitones(0.0)
    boosted.set_eq_enabled(True)
    boosted.set_eq_center_hz(2000.0)
    boosted.set_eq_gain_db(12.0)
    boosted.set_intensity(1.0)
    boosted_out = _process_in_blocks(boosted, signal)
    _assert_finite(boosted_out)

    bypassed_rms = _rms(bypassed_out[settle:])
    boosted_rms = _rms(boosted_out[settle:])
    assert boosted_rms > bypassed_rms * 1.3, f"expected EQ boost once enabled, got {bypassed_rms} vs {boosted_rms}"


def test_effect_eq_center_and_q_setters_update_underlying_eq():
    effect = MinionEffect(SAMPLE_RATE)
    effect.set_eq_center_hz(3000.0)
    effect.set_eq_q(2.0)
    assert effect.eq_center_hz == 3000.0
    assert effect.eq.center_hz == 3000.0
    assert effect.eq_q == 2.0
    assert effect.eq.q == 2.0


# -- Minionese ---------------------------------------------------------


def _minionese_out(seed: int = 0, configure=None) -> np.ndarray:
    m = Minionese(SAMPLE_RATE, seed=seed)
    if configure is not None:
        configure(m)
    m.set_intensity(1.0)
    signal = _gen_sawtooth(150.0, 1.5)
    out = _process_in_blocks(m, signal)
    _assert_finite(out)
    return out


def _assert_differs(a: np.ndarray, b: np.ndarray) -> None:
    n = min(a.shape[0], b.shape[0])
    assert n > 0
    assert not np.allclose(a[:n], b[:n]), "expected the setter to change output"


def test_minionese_set_semitones_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_semitones(0.0))
    _assert_differs(default_out, changed_out)


def test_minionese_set_wobble_ms_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_wobble_ms(8.0))
    _assert_differs(default_out, changed_out)


def test_minionese_set_chunk_ms_rebuild_stays_finite_and_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_chunk_ms(300.0))
    _assert_differs(default_out, changed_out)


def test_minionese_set_shuffle_k_rebuild_stays_finite_and_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_shuffle_k(2))
    _assert_differs(default_out, changed_out)


def test_minionese_set_fade_ms_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_fade_ms(2.0))
    _assert_differs(default_out, changed_out)


def test_minionese_set_max_slew_bounds_sample_to_sample_delta():
    tight = Minionese(SAMPLE_RATE, seed=0)
    tight.set_max_slew(0.01)
    tight.set_intensity(1.0)
    signal = _gen_sawtooth(150.0, 1.5)
    out = _process_in_blocks(tight, signal)
    _assert_finite(out)

    if out.shape[0] > 1:
        max_delta = float(np.max(np.abs(np.diff(out.astype(np.float64)))))
        assert max_delta <= 0.01 + 1e-6, f"slew limiter should bound deltas to ~0.01, got {max_delta}"


def test_minionese_set_floor_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_floor(0.0))
    _assert_differs(default_out, changed_out)


def test_minionese_set_resid_f_changes_output():
    default_out = _minionese_out()
    changed_out = _minionese_out(configure=lambda m: m.set_resid_f(0.0))
    _assert_differs(default_out, changed_out)


def test_minionese_set_vad_thresh_gates_output():
    # A constant-amplitude tone sits at ~1x the adaptive VAD peak, so only a
    # threshold above 1x gates it -- pushing past the UI range proves the
    # knob is wired into the gate (its real use is silencing quieter-than-
    # peak passages like pauses/background noise).
    default_out = _minionese_out()
    gated_out = _minionese_out(configure=lambda m: m.set_vad_thresh(2.0))
    _assert_differs(default_out, gated_out)
    # With everything gated, the synthesized output collapses to near-silence.
    assert float(np.max(np.abs(gated_out))) < float(np.max(np.abs(default_out)))


# -- PitchShifter --------------------------------------------------------


def test_pitchshifter_set_frame_and_tol_rebuild_window_and_still_shifts():
    shifter = PitchShifter(SAMPLE_RATE)
    shifter.set_frame(512)
    shifter.set_tol(64)
    shifter.set_semitones(12.0)

    assert shifter.L == 512
    assert shifter.Hs == 256
    assert shifter.win.shape[0] == 512
    assert shifter.tol == 64

    signal = _gen_sine(220.0, 1.0)
    out = _process_in_blocks(shifter, signal)
    _assert_finite(out)

    n = out.shape[0]
    middle = out[int(0.25 * n):int(0.75 * n)]
    dominant = _dominant_freq(middle)
    assert abs(dominant - 440.0) / 440.0 < 0.08, f"expected ~440Hz after rebuild+shift, got {dominant}"


# -- IntensityRamp -------------------------------------------------------


def test_ramp_set_duration_changes_ramp_speed():
    ramp = IntensityRamp(duration_s=5.0)
    ramp.set_duration(0.01)
    ramp.start()
    time.sleep(0.05)
    assert ramp.current() == 1.0
