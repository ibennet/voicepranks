"""The ramp-in duration is entered as hours/minutes/seconds and can be
cleared to zero (no ramp). Covers the UI's seconds<->H/M/S split and that the
engine accepts the widened range (up to hours). No Tk root is created -- only
the static converter and the engine are exercised.
"""
from __future__ import annotations

from voicepranks.audio.engine import VoiceEngine
from voicepranks.params import PARAM_SPECS_BY_NAME
from voicepranks.ui.app import VoicePranksApp


def test_split_seconds_breaks_into_h_m_s():
    assert VoicePranksApp._split_seconds(0) == (0, 0, 0.0)
    assert VoicePranksApp._split_seconds(90) == (0, 1, 30.0)
    assert VoicePranksApp._split_seconds(3661.5) == (1, 1, 1.5)


def test_split_seconds_clamps_negative():
    assert VoicePranksApp._split_seconds(-10) == (0, 0, 0.0)


def test_ramp_param_range_allows_hours():
    spec = PARAM_SPECS_BY_NAME["ramp.duration_s"]
    assert spec.max >= 3600.0  # at least an hour is representable


def test_engine_accepts_long_ramp_and_clear():
    engine = VoiceEngine()
    # 1h 1m 1.5s
    engine.set_param("ramp.duration_s", 3661.5)
    assert engine.ramp.duration_s == 3661.5
    # Clear -> no ramp: once started, intensity is full immediately.
    engine.set_param("ramp.duration_s", 0.0)
    assert engine.ramp.duration_s == 0.0
    engine.ramp.start()
    assert engine.ramp.current() == 1.0


def test_restart_ramp_sets_duration_and_restarts_from_zero():
    engine = VoiceEngine()
    engine.set_manual_intensity(0.7)  # manual override in effect
    engine.restart_ramp(4.0)
    # Duration applied, manual override cleared, timer just started (~0).
    assert engine.ramp.duration_s == 4.0
    assert engine._manual_intensity is None
    assert engine.ramp.current() < 0.05


def test_restart_ramp_without_duration_keeps_current():
    engine = VoiceEngine()
    engine.ramp.set_duration(3.0)
    engine.restart_ramp()
    assert engine.ramp.duration_s == 3.0
    assert engine.ramp.current() < 0.05


def test_reenable_without_ramp_reapplies_intensity():
    # With no ramp, disabling sets the effect to intensity 0; re-enabling must
    # restore full intensity even though the audio callback caches the last
    # applied value (regression guard for the intensity-cache optimization).
    import numpy as np

    engine = VoiceEngine(sample_rate=48000, blocksize=256)
    engine.ramp.set_duration(0.0)  # no ramp -> full intensity immediately
    block = (0.1 * np.sin(2 * np.pi * 200 * np.arange(256) / 48000)).astype(np.float32).reshape(-1, 1)

    engine.set_enabled(True)
    engine._input_callback(block, 256, None, None)
    assert engine.effect.intensity == 1.0

    engine.set_enabled(False)
    assert engine.effect.intensity == 0.0

    engine.set_enabled(True)
    engine._input_callback(block, 256, None, None)
    assert engine.effect.intensity == 1.0  # restored, not stuck at 0
