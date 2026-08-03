"""Tests for the live-adjustable setters added to the DSP stages so every
knob in `params.PARAM_SPECS` can actually be tuned at runtime. Each test
checks the setter (a) doesn't blow up NaN/Inf in block processing and
(b) actually changes behavior relative to the default.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from minion_voice.dsp.distortion import Distortion
from minion_voice.dsp.effect import MinionEffect
from minion_voice.dsp.growl import Growl
from minion_voice.dsp.laugh import GoofyLaugh
from minion_voice.dsp.minionese import Minionese
from minion_voice.dsp.pitch import PitchShifter
from minion_voice.dsp.ramp import IntensityRamp
from minion_voice.dsp.reverb import Reverb
from minion_voice.params import build_effect_registry
from minion_voice.presets import get_preset

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


def test_effect_nasality_changes_output():
    signal = _gen_sawtooth(150.0, 1.0)

    flat = MinionEffect(SAMPLE_RATE)
    flat.set_max_semitones(0.0)
    flat.set_intensity(1.0)
    flat_out = _process_in_blocks(flat, signal)
    _assert_finite(flat_out)

    nasal = MinionEffect(SAMPLE_RATE)
    nasal.set_max_semitones(0.0)
    nasal.set_nasality(0.9)
    nasal.set_intensity(1.0)
    nasal_out = _process_in_blocks(nasal, signal)
    _assert_finite(nasal_out)

    _assert_differs(flat_out, nasal_out)


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


# -- Distortion ----------------------------------------------------------


def test_distortion_drive_changes_output_and_stays_finite():
    signal = _gen_sine(220.0, 0.5)

    clean = Distortion(SAMPLE_RATE)  # drive defaults to 1.0
    clean_out = _process_in_blocks(clean, signal)
    _assert_finite(clean_out)

    gritty = Distortion(SAMPLE_RATE)
    gritty.set_drive(12.0)
    gritty_out = _process_in_blocks(gritty, signal)
    _assert_finite(gritty_out)

    _assert_differs(clean_out, gritty_out)
    # tanh soft-clip: bounded output, no runaway.
    assert float(np.max(np.abs(gritty_out))) <= 1.0 + 1e-6


def test_distortion_silence_stays_silent():
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    out = _process_in_blocks(Distortion(SAMPLE_RATE, drive=8.0), silence)
    _assert_finite(out)
    assert float(np.max(np.abs(out))) < 1e-6


# -- Growl (ring / AM modulation) ----------------------------------------


def test_growl_depth_zero_is_passthrough():
    signal = _gen_sawtooth(150.0, 0.5)
    out = _process_in_blocks(Growl(SAMPLE_RATE), signal)  # depth defaults to 0
    _assert_finite(out)
    n = min(out.shape[0], signal.shape[0])
    assert np.allclose(out[:n], signal[:n], atol=1e-6)


def test_growl_depth_changes_output():
    signal = _gen_sawtooth(150.0, 0.5)
    clean = _process_in_blocks(Growl(SAMPLE_RATE), signal)

    rough_mod = Growl(SAMPLE_RATE)
    rough_mod.set_rate_hz(70.0)
    rough_mod.set_depth(0.6)
    rough = _process_in_blocks(rough_mod, signal)
    _assert_finite(rough)

    _assert_differs(clean, rough)


def test_growl_phase_continuous_across_blocks():
    # Fed block-by-block vs in one shot must match: the carrier phase carries
    # across process() calls, so output is click-free at block boundaries.
    signal = _gen_sawtooth(150.0, 0.3)

    blocked_mod = Growl(SAMPLE_RATE)
    blocked_mod.set_rate_hz(70.0)
    blocked_mod.set_depth(1.0)
    blocked = _process_in_blocks(blocked_mod, signal, block=200)

    oneshot_mod = Growl(SAMPLE_RATE)
    oneshot_mod.set_rate_hz(70.0)
    oneshot_mod.set_depth(1.0)
    oneshot = oneshot_mod.process(signal)

    assert np.allclose(blocked, oneshot, atol=1e-5)


def test_growl_silence_stays_silent():
    g = Growl(SAMPLE_RATE)
    g.set_depth(1.0)
    out = _process_in_blocks(g, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    _assert_finite(out)
    assert float(np.max(np.abs(out))) < 1e-6


# -- GoofyLaugh (interval laugh generator) -------------------------------


def _laugh_over(signal, seed=0, interval=0.3, gain=0.8, scale=1.0):
    g = GoofyLaugh(SAMPLE_RATE, seed=seed)
    g.set_interval_s(interval)
    g.set_gain(gain)
    g.set_intensity_scale(scale)
    out = _process_in_blocks(g, signal)
    _assert_finite(out)
    return out


def test_laugh_triggers_and_modifies_voice():
    # Loud speech + a short interval => a laugh fires and changes the output.
    voice = _gen_sine(200.0, 2.0) * 0.5
    out = _laugh_over(voice)
    n = min(voice.shape[0], out.shape[0])
    assert not np.allclose(voice[:n], out[:n]), "expected a laugh to be mixed in"


def test_laugh_zero_interval_is_passthrough():
    voice = _gen_sine(200.0, 1.0) * 0.5
    out = _laugh_over(voice, interval=0.0)  # 0 disables laughing
    n = min(voice.shape[0], out.shape[0])
    assert np.allclose(voice[:n], out[:n], atol=1e-6)


def test_laugh_intensity_scale_zero_is_passthrough():
    voice = _gen_sine(200.0, 2.0) * 0.5
    out = _laugh_over(voice, scale=0.0)
    n = min(voice.shape[0], out.shape[0])
    assert np.allclose(voice[:n], out[:n], atol=1e-6)


def test_laugh_does_not_fire_into_silence():
    # Input below the speech gate => it never laughs into dead air.
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    out = _laugh_over(silence)
    assert float(np.max(np.abs(out))) < 1e-6


def test_laugh_interval_controls_spacing():
    # A shorter interval fires more laughs over the same signal than a longer one.
    voice = _gen_sine(200.0, 6.0) * 0.5

    def laugh_count(interval):
        g = GoofyLaugh(SAMPLE_RATE)
        g.set_interval_s(interval)
        g.set_gain(0.8)
        onsets, prev = 0, False
        for i in range(0, voice.shape[0], BLOCK):
            idle = g._pos >= g._laugh.shape[0]
            g.process(voice[i:i + BLOCK])
            if idle and g._pos < g._laugh.shape[0]:
                onsets += 1
        return onsets

    assert laugh_count(1.0) > laugh_count(3.0)


def test_laugh_volume_matches_voice_level():
    # Volume-matching: a louder voice yields a louder laugh (deterministic
    # timing, so the laugh contribution is comparable).
    def laugh_deviation(voice_amp):
        voice = _gen_sine(200.0, 3.0) * voice_amp
        out = _laugh_over(voice, gain=1.0)
        n = min(voice.shape[0], out.shape[0])
        return float(np.max(np.abs(out[:n] - voice[:n])))

    quiet = laugh_deviation(0.1)
    loud = laugh_deviation(0.4)
    assert loud > quiet * 1.5, f"laugh should scale with voice level ({quiet} vs {loud})"


def test_laugh_overlays_gibberish_mode():
    # The laugh is a top-level overlay: it must also fire when the gibberish
    # engine is active, not only on the plain path.
    effect = MinionEffect(SAMPLE_RATE)
    effect.set_gibberish(True)
    effect.set_laugh_enabled(True)
    effect.laugh.set_interval_s(0.3)
    effect.laugh.set_gain(0.8)
    effect.set_intensity(1.0)

    voice = _gen_sawtooth(150.0, 2.5) * 0.5
    lg = effect.laugh
    fired = False
    for i in range(0, voice.shape[0], BLOCK):
        idle = lg._pos >= lg._laugh.shape[0]
        out = effect.process(voice[i:i + BLOCK])
        _assert_finite(out)
        if idle and lg._pos < lg._laugh.shape[0]:
            fired = True
    assert fired, "laugh should fire in gibberish mode too"


def test_laugh_preserves_block_length():
    g = GoofyLaugh(SAMPLE_RATE, seed=0)
    g.set_interval_s(0.3)
    g.set_gain(0.8)
    block = _gen_sine(200.0, 0.1) * 0.5  # single block, > any internal chunk
    out = g.process(block)
    _assert_finite(out)
    assert out.shape[0] == block.shape[0]


# -- Reverb --------------------------------------------------------------


def _reverb_out(configure=None) -> np.ndarray:
    rv = Reverb(SAMPLE_RATE)
    if configure is not None:
        configure(rv)
    signal = _gen_sawtooth(150.0, 1.0)
    out = _process_in_blocks(rv, signal)
    _assert_finite(out)
    return out


def test_reverb_mix_zero_is_dry_passthrough():
    signal = _gen_sawtooth(150.0, 0.5)
    rv = Reverb(SAMPLE_RATE, mix=0.0)
    out = _process_in_blocks(rv, signal)
    _assert_finite(out)
    n = min(out.shape[0], signal.shape[0])
    assert np.allclose(out[:n], signal[:n], atol=1e-6)


def test_reverb_set_mix_changes_output():
    dry = _reverb_out(configure=lambda rv: rv.set_mix(0.0))
    wet = _reverb_out(configure=lambda rv: rv.set_mix(0.6))
    _assert_differs(dry, wet)


def test_reverb_set_room_size_changes_tail():
    small = _reverb_out(configure=lambda rv: (rv.set_mix(0.6), rv.set_room_size(0.2)))
    big = _reverb_out(configure=lambda rv: (rv.set_mix(0.6), rv.set_room_size(0.95)))
    _assert_differs(small, big)


def test_reverb_set_damp_changes_output():
    bright = _reverb_out(configure=lambda rv: (rv.set_mix(0.6), rv.set_damp(0.0)))
    dark = _reverb_out(configure=lambda rv: (rv.set_mix(0.6), rv.set_damp(1.0)))
    _assert_differs(bright, dark)


def test_reverb_silence_stays_silent():
    rv = Reverb(SAMPLE_RATE, mix=0.6, room_size=0.9)
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    out = _process_in_blocks(rv, silence)
    _assert_finite(out)
    assert float(np.max(np.abs(out))) < 1e-6


def test_reverb_preserves_block_length():
    rv = Reverb(SAMPLE_RATE, mix=0.5)
    # A single block much longer than the shortest delay line exercises the
    # internal sub-chunking; output length must still equal input length.
    block = _gen_sawtooth(150.0, 0.1)  # ~4800 samples, >> any delay line
    out = rv.process(block)
    _assert_finite(out)
    assert out.shape[0] == block.shape[0]


# -- scary preset (deeper + harsher, end to end) -------------------------


def _hf_fraction(signal: np.ndarray, cutoff_hz: float = 1500.0) -> float:
    """Fraction of spectral magnitude above `cutoff_hz` -- a proxy for how
    harsh/bright (harmonically rich) the signal is."""
    windowed = signal * np.hanning(len(signal))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)
    total = float(np.sum(mag)) + 1e-12
    return float(np.sum(mag[freqs >= cutoff_hz])) / total


def _mid(o: np.ndarray) -> np.ndarray:
    """The steady-state middle half of a (variable-length) output block."""
    n = o.shape[0]
    return o[int(0.25 * n):int(0.75 * n)]


def _build_scary_effect() -> MinionEffect:
    effect = MinionEffect(SAMPLE_RATE)
    reg = build_effect_registry(effect)
    for key, val in get_preset("scary").items():
        if key == "gibberish":
            effect.set_gibberish(bool(val))
        else:
            reg[key].set(val)
    effect.set_intensity(1.0)
    return effect


def test_effect_negative_semitones_pitches_down():
    # The "deeper" ingredient in isolation: a negative shift lowers pitch
    # (no post-effects, so it's deterministic).
    effect = MinionEffect(SAMPLE_RATE)
    effect.set_max_semitones(-7.0)
    effect.set_intensity(1.0)

    freq = 220.0
    out = _process_in_blocks(effect, _gen_sine(freq, 1.0))
    _assert_finite(out)
    n = out.shape[0]
    dominant = _dominant_freq(out[int(0.25 * n):int(0.75 * n)])
    assert dominant < freq, f"expected deeper (dominant < {freq}), got {dominant}"


def test_scary_preset_is_harsher_than_plain_pitch_down():
    signal = _gen_sine(220.0, 1.5)

    # Baseline: same downward pitch shift, but none of the grit stages.
    plain = MinionEffect(SAMPLE_RATE)
    plain.set_max_semitones(-7.0)
    plain.set_intensity(1.0)
    plain_out = _process_in_blocks(plain, signal)
    _assert_finite(plain_out)

    scary_out = _process_in_blocks(_build_scary_effect(), signal)
    _assert_finite(scary_out)
    assert _rms(scary_out) > 0.0

    # The growl + hard-clip distortion should inject substantially more
    # high-frequency harmonic energy than a clean pitch-down.
    assert _hf_fraction(_mid(scary_out)) > _hf_fraction(_mid(plain_out)) * 1.5


def test_scary_intensity_scales_every_stage():
    effect = _build_scary_effect()  # ends at intensity 1.0

    # At full intensity every stage is at its target amount.
    effect.set_intensity(1.0)
    assert effect.growl.depth > 0.0
    assert effect.distortion.drive > 1.0
    assert effect.reverb.mix > 0.0
    assert effect.pitch.ratio < 1.0  # pitched down

    # At zero intensity every stage falls back to neutral (unprocessed).
    effect.set_intensity(0.0)
    assert effect.growl.depth == 0.0            # no modulation
    assert effect.distortion.drive == pytest.approx(1.0)  # clean
    assert effect.reverb.mix == 0.0             # fully dry
    assert effect.pitch.ratio == pytest.approx(1.0)  # no shift

    # Targets survive, so intensity brings the full effect back.
    effect.set_intensity(1.0)
    assert effect.growl.depth > 0.0


def test_scary_low_intensity_is_tamer_than_high():
    signal = _gen_sine(220.0, 1.5)

    hi = _build_scary_effect()
    hi.set_intensity(1.0)
    hi_out = _process_in_blocks(hi, signal)
    _assert_finite(hi_out)

    lo = _build_scary_effect()
    lo.set_intensity(0.15)
    lo_out = _process_in_blocks(lo, signal)
    _assert_finite(lo_out)

    # Lower intensity => less grit/harmonic energy (the whole effect eases off,
    # not just the pitch).
    assert _hf_fraction(_mid(lo_out)) < _hf_fraction(_mid(hi_out))


# -- IntensityRamp -------------------------------------------------------


def test_ramp_set_duration_changes_ramp_speed():
    ramp = IntensityRamp(duration_s=5.0)
    ramp.set_duration(0.01)
    ramp.start()
    time.sleep(0.05)
    assert ramp.current() == 1.0
