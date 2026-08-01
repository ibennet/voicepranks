"""Recorder tests, driven entirely through `VoiceEngine` methods and a
synthetic dry take fed straight into `_input_callback` -- no audio
hardware or real `sd.InputStream`/`OutputStream` is touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from minion_voice.audio.engine import VoiceEngine

SAMPLE_RATE = 48000
BLOCK = 256


def _gen_sine(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _feed_dry_take(engine: VoiceEngine, signal: np.ndarray, block: int = BLOCK) -> None:
    """Simulate the mic feeding `signal` through the input callback in
    fixed-size blocks (mirrors how `sd.InputStream` would call it)."""
    n = signal.shape[0]
    for start in range(0, n, block):
        chunk = signal[start:start + block]
        if chunk.shape[0] < block:
            chunk = np.pad(chunk, (0, block - chunk.shape[0]))
        engine._input_callback(chunk.reshape(-1, 1), block, None, None)


def test_record_start_stop_captures_dry_take():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    signal = _gen_sine(220.0, 1.0)

    engine.record_start()
    assert engine.get_status()["recording"] is True
    _feed_dry_take(engine, signal)
    raw = engine.record_stop()

    assert engine.get_status()["recording"] is False
    assert raw.shape[0] >= signal.shape[0]
    assert not np.any(np.isnan(raw))
    assert not np.any(np.isinf(raw))
    # The captured take should be the *dry* mic signal, not run through
    # the effect -- close to the padded input regardless of `enabled`.
    assert np.allclose(raw[: signal.shape[0]], signal, atol=1e-6)


def test_recording_captures_dry_signal_even_when_effect_enabled():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.set_param("enabled", True)
    engine.set_param("intensity", 1.0)

    signal = _gen_sine(220.0, 0.5)
    engine.record_start()
    _feed_dry_take(engine, signal)
    raw = engine.record_stop()

    assert not np.any(np.isnan(raw))
    # Dry take should still closely match the original signal, since
    # recording happens before the pitch-shift effect is applied.
    assert np.allclose(raw[: signal.shape[0]], signal, atol=1e-6)


def test_render_current_returns_pitch_shifted_buffer_with_no_nans():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    signal = _gen_sine(220.0, 1.0)

    engine.record_start()
    _feed_dry_take(engine, signal)
    engine.record_stop()

    engine.set_param("effect.max_semitones", 12.0)
    engine.set_param("intensity", 1.0)

    rendered = engine.render_current()
    assert rendered.shape[0] > 0
    assert not np.any(np.isnan(rendered))
    assert not np.any(np.isinf(rendered))

    windowed = rendered * np.hanning(len(rendered))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(rendered), d=1.0 / SAMPLE_RATE)
    dominant = float(freqs[np.argmax(np.abs(spectrum))])
    assert dominant > 220.0 * 1.3, f"expected a clearly pitched-up render, got {dominant} Hz"


def test_render_current_without_a_take_raises():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    with pytest.raises(RuntimeError):
        engine.render_current()


def test_save_and_play_use_the_requested_take(tmp_path, monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    signal = _gen_sine(220.0, 0.5)

    engine.record_start()
    _feed_dry_take(engine, signal)
    engine.record_stop()
    engine.render_current()

    out_path = tmp_path / "take.wav"
    engine.save(str(out_path), which="rendered")
    assert out_path.exists() and out_path.stat().st_size > 0

    played = {}

    def fake_play(buf, sample_rate, device=None):
        played["buf"] = buf
        played["sample_rate"] = sample_rate

    monkeypatch.setattr("minion_voice.audio.engine.sd.play", fake_play)
    engine.play("raw")
    assert played["sample_rate"] == SAMPLE_RATE
    assert not np.any(np.isnan(played["buf"]))
