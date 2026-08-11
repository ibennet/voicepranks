"""Laugh-to-both-outputs + playback-volume tests.

Cover the four seams added by "every laugh plays to both the outgoing cable and
your headphones, at a controllable playback level":
  - `playback_gain` round-trips through the registry and scales `sd.play`,
  - the Play Laugh button mixes an overlay onto the OUTGOING ring when running
    (and only auditions to the headphones when stopped),
  - the automatic laugh flags a headphone echo that the poll plays when the
    monitor is off (and skips when it's on, to avoid doubling).

Driven without audio hardware -- the input callback is invoked directly, rings
are inspected, and `sd.play` is monkeypatched to capture buffers.
"""
from __future__ import annotations

import numpy as np

from voicepranks.audio.engine import VoiceEngine

SAMPLE_RATE = 48000
BLOCK = 256


def _capture_sd_play(monkeypatch):
    """Monkeypatch `engine.sd.play` to record the buffers it's handed."""
    calls = []
    monkeypatch.setattr(
        "voicepranks.audio.engine.sd.play",
        lambda buf, *a, **k: calls.append(np.asarray(buf, dtype=np.float32).copy()),
    )
    return calls


def _drive_input(engine: VoiceEngine, signal: np.ndarray, block: int = BLOCK) -> None:
    for start in range(0, signal.shape[0], block):
        chunk = signal[start:start + block]
        if chunk.shape[0] < block:
            chunk = np.pad(chunk, (0, block - chunk.shape[0]))
        engine._input_callback(chunk.reshape(-1, 1), block, None, None)


def _loud_signal(duration_s: float = 0.3) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (0.6 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)


# -- playback_gain -------------------------------------------------------

def test_playback_gain_round_trips_and_clamps():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    assert engine.snapshot()["playback_gain"] == 1.0
    engine.set_param("playback_gain", 1.5)
    assert engine.playback_gain == 1.5
    assert engine.snapshot()["playback_gain"] == 1.5
    engine.set_param("playback_gain", -2.0)  # no negatives
    assert engine.playback_gain == 0.0


def test_play_scales_take_by_playback_gain(monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine._live_take = np.full(100, 0.4, dtype=np.float32)
    engine.set_playback_gain(2.0)
    calls = _capture_sd_play(monkeypatch)

    engine.play("live")

    assert len(calls) == 1
    assert np.allclose(calls[0], 0.8)  # 0.4 * 2.0


def test_play_laugh_scales_headphones_by_playback_gain(monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.set_playback_gain(0.5)
    calls = _capture_sd_play(monkeypatch)

    engine.play_laugh()

    clip = engine.effect.laugh_clip()
    assert len(calls) == 1
    assert np.allclose(calls[0], np.asarray(clip, dtype=np.float32) * 0.5)


# -- Play Laugh -> outgoing overlay --------------------------------------

def test_play_laugh_when_stopped_is_headphones_only(monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    calls = _capture_sd_play(monkeypatch)

    engine.play_laugh()  # not running

    assert len(calls) == 1  # played to headphones
    assert engine._laugh_overlay is None  # but nothing queued to the cable


def test_play_laugh_when_running_mixes_overlay_into_output_ring(monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.running = True  # simulate live output stream, no hardware
    _capture_sd_play(monkeypatch)

    engine.play_laugh()
    assert engine._laugh_overlay is not None

    clip = np.asarray(engine.effect.laugh_clip(), dtype=np.float32)
    # Silence in -> the ring carries the injected laugh only. Drive well past
    # the clip length so the whole overlay flushes through.
    _drive_input(engine, np.zeros(clip.shape[0] + 4 * BLOCK, dtype=np.float32))

    out = engine.ring.read(engine.ring.count)
    assert float(np.max(np.abs(out))) > 0.1  # laugh reached the outgoing ring
    assert engine._laugh_overlay is None  # overlay consumed


def test_stop_clears_pending_overlay(monkeypatch):
    # A clip queued just before stop() must not survive to burst onto the cable
    # on the next start().
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.running = True
    _capture_sd_play(monkeypatch)
    engine.play_laugh()
    assert engine._laugh_overlay is not None

    engine.stop()  # no streams open; just exercises the teardown path

    assert engine._laugh_overlay is None
    assert engine._laugh_overlay_pos == 0


def test_output_overlay_does_not_leak_into_monitor_feed(monkeypatch):
    # The overlay is for the cable only; the monitor feed (processed) must stay
    # a single source so the headphone side isn't doubled.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.running = True
    engine.monitor_enabled = True
    engine.monitor_stream = object()  # truthy sentinel; never touched here
    _capture_sd_play(monkeypatch)

    engine.play_laugh()
    clip = np.asarray(engine.effect.laugh_clip(), dtype=np.float32)
    _drive_input(engine, np.zeros(clip.shape[0] + 4 * BLOCK, dtype=np.float32))

    mon = engine.monitor_ring.read(engine.monitor_ring.count)
    assert float(np.max(np.abs(mon))) < 1e-6  # silent input stayed silent in monitor


# -- automatic laugh -> headphone echo -----------------------------------

def _fire_auto_laugh(engine: VoiceEngine) -> None:
    engine.enabled = True
    engine.set_manual_intensity(1.0)  # keep the effect (and laugh scale) at full
    engine.effect.set_laugh_enabled(True)
    engine.effect.set_laugh_interval_s(0.01)  # fire almost immediately
    _drive_input(engine, _loud_signal(0.3))


def test_automatic_laugh_ticks_fire_count():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    _fire_auto_laugh(engine)
    assert engine.effect.laugh.fire_count > 0
    # The poll hasn't run yet, so the engine's last-seen count still lags.
    assert engine._last_laugh_fire_count == 0


def test_poll_echoes_fired_laugh_when_monitor_off(monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.monitor_enabled = False
    _fire_auto_laugh(engine)
    calls = _capture_sd_play(monkeypatch)

    engine.poll_pending_laugh_echo()

    assert len(calls) == 1
    assert engine._last_laugh_fire_count == engine.effect.laugh.fire_count
    # A second poll with no new fire is a no-op (the echo doesn't repeat).
    engine.poll_pending_laugh_echo()
    assert len(calls) == 1


def test_poll_skips_echo_when_monitor_on(monkeypatch):
    # Monitor already carries the laugh to the listening device; echoing would
    # double it. The counter still syncs so a later monitor-off fire is fresh.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.monitor_enabled = True
    _fire_auto_laugh(engine)
    calls = _capture_sd_play(monkeypatch)

    engine.poll_pending_laugh_echo()

    assert len(calls) == 0
    assert engine._last_laugh_fire_count == engine.effect.laugh.fire_count


def test_poll_is_a_noop_when_no_new_laugh(monkeypatch):
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    calls = _capture_sd_play(monkeypatch)
    engine.poll_pending_laugh_echo()
    assert len(calls) == 0
