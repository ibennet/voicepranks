"""Preset tests: the two built-in presets apply their sound-character params
through the engine, and the same works over the control server. No audio
hardware is touched (the engine is constructed but never started).
"""
from __future__ import annotations

import http.client
import json

import pytest

from voicepranks import presets
from voicepranks.audio.engine import VoiceEngine
from voicepranks.control_server import ControlServer
from voicepranks.params import PARAM_SPECS_BY_NAME

SAMPLE_RATE = 48000
BLOCK = 256


def test_both_builtin_presets_exist():
    assert presets.preset_names() == ["animalese", "minion", "scary", "goofy"]


def test_apply_animalese_sets_scramble_values():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.apply_preset("animalese")
    snap = engine.snapshot()
    assert snap["minionese.use_shuffle"] is True
    assert snap["shuffle.shuffle_k"] == 6
    assert snap["shuffle.reverse_prob"] == 0.5
    assert snap["shuffle.chunk_ms"] == 80.0


def test_apply_minion_sets_connected_values():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.apply_preset("minion")
    snap = engine.snapshot()
    assert snap["shuffle.shuffle_k"] == 1
    assert snap["shuffle.reverse_prob"] == 0.0
    assert snap["shuffle.wobble_ms"] == 4.0


def test_apply_scary_sets_deep_growl_values():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.apply_preset("scary")
    snap = engine.snapshot()
    # Non-gibberish -- transforms real speech via the plain path.
    assert snap["gibberish"] is False
    # Deeper: a *negative* pitch shift (down, not the usual chipmunk up).
    assert snap["effect.max_semitones"] < 0.0
    # Grit + space are switched on.
    assert snap["distortion.enabled"] is True
    assert snap["distortion.drive"] > 1.0
    assert snap["reverb.enabled"] is True
    assert snap["reverb.mix"] > 0.0


def test_apply_goofy_sets_laugh_and_warble_values():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.apply_preset("goofy")
    snap = engine.snapshot()
    assert snap["gibberish"] is False
    assert snap["effect.max_semitones"] < 0.0  # deep, stable down-shift
    assert snap["effect.nasality"] > 0.0       # nasal honk
    assert snap["growl.enabled"] is True       # the warble
    assert snap["distortion.enabled"] is True  # gravel/grit
    assert snap["reverb.enabled"] is False
    # Goofy is the one preset that turns the random laugh on (vol 1, every 15s).
    assert snap["laugh.enabled"] is True
    assert snap["laugh.gain"] == 1.0
    assert snap["laugh.interval_s"] == 15.0


def test_switching_plain_presets_does_not_leak_enable_flags():
    # Regression: presets only set the params they list, so a plain-path preset
    # must explicitly set every enable flag or the previous preset's stages leak
    # through. scary enables distortion/reverb; goofy must turn reverb back off.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)

    engine.apply_preset("scary")
    assert engine.snapshot()["reverb.enabled"] is True

    engine.apply_preset("goofy")
    assert engine.snapshot()["reverb.enabled"] is False   # did NOT leak from scary

    engine.apply_preset("scary")
    assert engine.snapshot()["reverb.enabled"] is True


def test_goofy_enables_laugh_other_presets_disable_it():
    # Selecting goofy turns the random laugh on; any other preset turns it off.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)

    engine.apply_preset("goofy")
    assert engine.snapshot()["laugh.enabled"] is True

    for name in ("scary", "minion", "animalese"):
        engine.apply_preset(name)
        assert engine.snapshot()["laugh.enabled"] is False, f"{name} should disable the laugh"

    # ...and switching back to goofy re-enables it.
    engine.apply_preset("goofy")
    assert engine.snapshot()["laugh.enabled"] is True


def test_none_neutral_removes_all_effects():
    # The "None" reset (NEUTRAL) leaves a plain, unprocessed voice.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.apply_preset("scary")  # turn a bunch of stuff on first
    for name, value in presets.NEUTRAL.items():
        engine.set_param(name, value)
    snap = engine.snapshot()
    assert snap["gibberish"] is False
    assert snap["effect.eq_enabled"] is False
    assert snap["effect.nasality"] == 0.0
    assert snap["effect.max_semitones"] == 0.0
    assert snap["growl.enabled"] is False
    assert snap["distortion.enabled"] is False
    assert snap["reverb.enabled"] is False
    assert snap["laugh.enabled"] is False


def test_gibberish_preset_clears_plain_path_effects():
    # Regression: apply "scary" (plain-path distortion/reverb on), then a
    # gibberish preset, then untoggle gibberish -- the plain voice must be
    # clean, NOT the leftover scary effects.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.apply_preset("scary")
    engine.apply_preset("minion")
    engine.set_param("gibberish", False)  # user unchecks the toggle
    snap = engine.snapshot()
    assert snap["distortion.enabled"] is False
    assert snap["reverb.enabled"] is False
    assert snap["effect.nasality"] == 0.0


def test_snapshot_reflects_pending_params_while_running():
    # Regression: while running, set_param stashes to _pending (applied on the
    # audio thread). snapshot() must reflect the queued value so a save / API
    # read right after doesn't persist the stale pre-change value.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.running = True  # simulate a running engine (no audio thread to drain)
    try:
        engine.set_param("distortion.drive", 7.0)  # -> stashed in _pending
        assert engine.snapshot()["distortion.drive"] == 7.0
    finally:
        engine.running = False


def test_unknown_preset_raises():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    with pytest.raises(KeyError):
        engine.apply_preset("nope")


def test_presets_only_reference_valid_nonsession_params():
    for name, values in presets.PRESETS.items():
        for key in values:
            assert key in PARAM_SPECS_BY_NAME, f"{name} sets unknown param {key}"
            # Presets are voice character only -- never device/session state,
            # and never `intensity` (a live master control the user owns).
            assert not key.startswith("io."), f"{name} must not set {key}"
            assert key not in ("enabled", "monitor", "intensity"), f"{name} must not set {key}"


def test_applying_a_preset_preserves_the_user_intensity():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.set_param("intensity", 0.4)
    engine.apply_preset("animalese")
    # Preset changes character but leaves the master intensity where the user
    # put it -- the intensity control scales the applied preset.
    assert engine.snapshot()["intensity"] == 0.4


def test_apply_preset_does_not_change_enabled_or_devices():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    engine.set_param("enabled", False)
    before_enabled = engine.snapshot()["enabled"]
    before_out = engine.output_device
    engine.apply_preset("animalese")
    after = engine.snapshot()
    assert after["enabled"] == before_enabled
    assert engine.output_device == before_out


# -- control server round-trip ------------------------------------------------


@pytest.fixture()
def server():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    srv = ControlServer(engine)
    base_url = srv.start(host="127.0.0.1", port=0)
    yield srv, engine, base_url
    srv.stop()


def _get(base_url: str, path: str):
    conn = http.client.HTTPConnection(base_url.split("://", 1)[1])
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def _post(base_url: str, path: str, payload: dict):
    conn = http.client.HTTPConnection(base_url.split("://", 1)[1])
    body = json.dumps(payload)
    conn.request("POST", path, body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def test_api_lists_and_applies_presets(server):
    _srv, _engine, base_url = server

    status, data = _get(base_url, "/api/presets")
    assert status == 200
    assert data["names"] == ["animalese", "minion", "scary", "goofy"]

    status, data = _post(base_url, "/api/presets/apply", {"name": "minion"})
    assert status == 200 and data["ok"] is True

    status, state = _get(base_url, "/api/state")
    assert state["values"]["shuffle.shuffle_k"] == 1
    assert state["values"]["shuffle.reverse_prob"] == 0.0


def test_api_apply_unknown_preset_is_400(server):
    _srv, _engine, base_url = server
    status, data = _post(base_url, "/api/presets/apply", {"name": "bogus"})
    assert status == 400
    assert data["ok"] is False
