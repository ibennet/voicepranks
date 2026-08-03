"""Live-monitor tests: the engine feeds a second (monitor) ring the same
processed audio so it can be played to an audible device while the main
output feeds the virtual cable. Driven without audio hardware -- the input
callback is invoked directly and the monitor ring is inspected, and the
`monitor` param is exercised through the registry.
"""
from __future__ import annotations

import numpy as np

from voicepranks.audio.engine import VoiceEngine

SAMPLE_RATE = 48000
BLOCK = 256


class _FakeStream:
    """Stand-in so `monitor_enabled and monitor_stream is not None` is true
    without opening real audio hardware."""

    def stop(self):
        pass

    def close(self):
        pass


def _drive_input(engine: VoiceEngine, signal: np.ndarray, block: int = BLOCK) -> None:
    for start in range(0, signal.shape[0], block):
        chunk = signal[start:start + block].reshape(-1, 1)
        engine._input_callback(chunk, chunk.shape[0], None, None)


def test_monitor_param_round_trips_through_registry():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    assert engine.snapshot()["monitor"] is False
    # Not running -> applies immediately (no audio thread to defer to).
    engine.set_param("monitor", True)
    assert engine.monitor_enabled is True
    assert engine.snapshot()["monitor"] is True
    engine.set_param("monitor", False)
    assert engine.monitor_enabled is False


def test_monitor_ring_receives_processed_audio_when_enabled():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.enabled = True
    engine.effect.set_intensity(1.0)
    # Simulate a running engine with an open monitor sink, without hardware.
    engine.running = True
    engine.monitor_enabled = True
    engine.monitor_stream = _FakeStream()

    signal = (0.5 * np.sin(2 * np.pi * 180 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
    _drive_input(engine, signal)

    # The monitor ring got fed processed audio (same as the main ring).
    assert engine.monitor_ring.count > 0
    out = engine.monitor_ring.read(engine.monitor_ring.count)
    assert not np.any(np.isnan(out))
    assert float(np.max(np.abs(out))) > 0.0


def test_monitor_ring_untouched_when_disabled():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.enabled = True
    engine.effect.set_intensity(1.0)
    engine.running = True
    engine.monitor_enabled = False  # monitor off

    signal = (0.5 * np.sin(2 * np.pi * 180 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
    _drive_input(engine, signal)

    # Main output still fed, but the monitor ring stays empty.
    assert engine.ring.count > 0
    assert engine.monitor_ring.count == 0
