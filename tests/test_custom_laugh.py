"""Custom "goofy laugh" recording tests.

Cover the three seams of the record-your-own-laugh feature:
  - `GoofyLaugh` clip selection / persistence (`save_custom`, `reset_to_stock`,
    and custom-over-stock priority on construction),
  - the `VoiceEngine` laugh recorder driven through `_input_callback` (no real
    audio hardware, mirroring `test_recorder.py`),
  - the silent-recording guard.

Every test points the custom-clip path at a `tmp_path` file so the real
`~/.voicepranks/custom_laugh.wav` is never touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from voicepranks.audio.engine import VoiceEngine
from voicepranks.dsp.laugh import GoofyLaugh

SAMPLE_RATE = 48000
BLOCK = 256


def _loud_signal(duration_s: float = 0.5) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (0.6 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)


def _feed_dry_take(engine: VoiceEngine, signal: np.ndarray, block: int = BLOCK) -> None:
    """Feed `signal` through the input callback in fixed-size blocks, the way
    `sd.InputStream` would (copied from test_recorder's helper)."""
    n = signal.shape[0]
    for start in range(0, n, block):
        chunk = signal[start:start + block]
        if chunk.shape[0] < block:
            chunk = np.pad(chunk, (0, block - chunk.shape[0]))
        engine._input_callback(chunk.reshape(-1, 1), block, None, None)


# -- GoofyLaugh clip selection / persistence -----------------------------

def test_save_custom_installs_and_persists(tmp_path):
    g = GoofyLaugh(SAMPLE_RATE)
    g._custom_path = tmp_path / "custom_laugh.wav"
    assert g.using_custom is False

    clip = _loud_signal()
    g.save_custom(clip)

    assert g.using_custom is True
    assert g._custom_path.exists(), "custom clip should be written to disk"
    # The live sample is now the recording, and it's what the next laugh uses.
    np.testing.assert_allclose(g._sample, clip, atol=1e-4)
    assert g._next_laugh() is g._sample


def test_custom_clip_is_preferred_on_construction(tmp_path):
    custom_path = tmp_path / "custom_laugh.wav"

    # Record + persist with one instance...
    g1 = GoofyLaugh(SAMPLE_RATE)
    g1._custom_path = custom_path
    g1.save_custom(_loud_signal())

    # ...a fresh instance pointed at the same path loads the custom clip, not
    # the stock one (mirrors an app restart).
    g2 = GoofyLaugh(SAMPLE_RATE)
    g2._custom_path = custom_path
    g2._sample = g2._load_active_sample()
    assert g2.using_custom is True
    assert g2._sample is not None and g2._sample.size > 0


def test_reset_to_stock_reverts_and_deletes(tmp_path):
    g = GoofyLaugh(SAMPLE_RATE)
    g._custom_path = tmp_path / "custom_laugh.wav"
    g.save_custom(_loud_signal())
    assert g._custom_path.exists()

    g.reset_to_stock()

    assert g.using_custom is False
    assert not g._custom_path.exists(), "revert should remove the saved clip"
    # Stock clip is bundled in this repo, so a sample is still available.
    assert g._sample is not None and g._sample.size > 0


def test_save_custom_rejects_silent(tmp_path):
    g = GoofyLaugh(SAMPLE_RATE)
    g._custom_path = tmp_path / "custom_laugh.wav"
    with pytest.raises(ValueError):
        g.save_custom(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert g.using_custom is False
    assert not g._custom_path.exists()


# -- VoiceEngine laugh recorder ------------------------------------------

def _engine(tmp_path) -> VoiceEngine:
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.effect.laugh._custom_path = tmp_path / "custom_laugh.wav"
    return engine


def test_engine_records_and_installs_custom_laugh(tmp_path):
    engine = _engine(tmp_path)
    assert engine.get_status()["custom_laugh"] is False

    engine.record_laugh_start()
    assert engine.get_status()["laugh_recording"] is True
    _feed_dry_take(engine, _loud_signal())
    take = engine.record_laugh_stop()

    assert take.size > 0
    status = engine.get_status()
    assert status["laugh_recording"] is False
    assert status["custom_laugh"] is True
    assert engine.effect.laugh._custom_path.exists()


def test_engine_reset_laugh_to_stock(tmp_path):
    engine = _engine(tmp_path)
    engine.record_laugh_start()
    _feed_dry_take(engine, _loud_signal())
    engine.record_laugh_stop()
    assert engine.get_status()["custom_laugh"] is True

    engine.reset_laugh_to_stock()
    assert engine.get_status()["custom_laugh"] is False


def test_laugh_recording_does_not_disturb_take_recorder(tmp_path):
    """Laugh capture and take capture share a lock but separate buffers -- a
    laugh recording must not consume or corrupt an in-progress take."""
    engine = _engine(tmp_path)
    engine.record_start()
    engine.record_laugh_start()
    _feed_dry_take(engine, _loud_signal())
    engine.record_laugh_stop()
    live = engine.record_stop()

    assert live.size > 0, "take recorder should still have captured audio"
    assert engine.get_status()["custom_laugh"] is True


def test_engine_record_laugh_stop_silent_raises(tmp_path):
    engine = _engine(tmp_path)
    engine.record_laugh_start()
    _feed_dry_take(engine, np.zeros(SAMPLE_RATE, dtype=np.float32))
    with pytest.raises(ValueError):
        engine.record_laugh_stop()
    assert engine.get_status()["custom_laugh"] is False
