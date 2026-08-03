"""Settings persistence + device resolution, and the engine playback device.
No audio hardware is touched.
"""
from __future__ import annotations

import json

from voicepranks import settings as settings_mod
from voicepranks.audio.engine import VoiceEngine

INPUTS = [(1, "MacBook Mic"), (3, "USB Mic")]
OUTPUTS = [(2, "Headphones"), (5, "BlackHole 2ch")]


def test_device_entry_and_resolve_by_name_prefers_name():
    entry = settings_mod.device_entry(99, "USB Mic")  # stale index, good name
    # Name match wins even though the stored index (99) is not in the list.
    assert settings_mod.resolve_device(entry, INPUTS) == 3


def test_resolve_falls_back_to_index_then_none():
    # No name match, but the stored index is valid.
    assert settings_mod.resolve_device({"index": 5, "name": "Gone"}, OUTPUTS) == 5
    # Neither name nor index valid -> None (default/auto).
    assert settings_mod.resolve_device({"index": 77, "name": "Gone"}, OUTPUTS) is None
    assert settings_mod.resolve_device(None, OUTPUTS) is None


def test_resolve_all_maps_playback_against_outputs():
    s = {
        "input_device": settings_mod.device_entry(1, "MacBook Mic"),
        "output_device": settings_mod.device_entry(5, "BlackHole 2ch"),
        "playback_device": settings_mod.device_entry(2, "Headphones"),
    }
    resolved = settings_mod.resolve_all(s, INPUTS, OUTPUTS)
    assert resolved == {"input_device": 1, "output_device": 5, "playback_device": 2}


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", path)

    s = settings_mod.update_devices(
        {}, input_device=settings_mod.device_entry(1, "MacBook Mic")
    )
    settings_mod.save(s)
    assert path.exists()
    assert json.loads(path.read_text())["input_device"]["name"] == "MacBook Mic"
    assert settings_mod.load() == s


def test_load_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "nope.json")
    assert settings_mod.load() == {}


def test_update_devices_ignores_unknown_keys():
    out = settings_mod.update_devices({}, bogus=settings_mod.device_entry(1, "x"))
    assert "bogus" not in out


# -- engine playback device --------------------------------------------------


def test_playback_device_param_round_trips():
    engine = VoiceEngine()
    assert engine.snapshot()["io.playback_device"] == -1  # default/None
    engine.set_param("io.playback_device", 7)
    assert engine.playback_device == 7
    assert engine.snapshot()["io.playback_device"] == 7
    engine.set_param("io.playback_device", -1)
    assert engine.playback_device is None


def test_play_targets_playback_device(tmp_path, monkeypatch):
    import numpy as np

    engine = VoiceEngine()
    engine.set_playback_device(9)
    # Give it a live take to play.
    engine._live_take = (0.2 * np.sin(np.arange(4800) / 48000.0)).astype(np.float32)

    captured = {}

    def fake_play(buf, sr, device=None):
        captured["device"] = device

    monkeypatch.setattr("voicepranks.audio.engine.sd.play", fake_play)
    engine.play("live")
    # Play must route to the listening device, not output_device.
    assert captured["device"] == 9


def test_negative_device_index_means_default_none():
    # The UI/API send -1 for "(default)"; set_param must resolve that to None
    # (not a literal device index -1) for input and output devices.
    engine = VoiceEngine()
    engine.set_param("io.input_device", 3)
    engine.set_param("io.output_device", 5)
    assert engine.input_device == 3 and engine.output_device == 5
    engine.set_param("io.input_device", -1)
    engine.set_param("io.output_device", -1)
    assert engine.input_device is None
    assert engine.output_device is None
