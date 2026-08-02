"""Preset tests: the two built-in presets apply their sound-character params
through the engine, and the same works over the control server. No audio
hardware is touched (the engine is constructed but never started).
"""
from __future__ import annotations

import http.client
import json

import pytest

from minion_voice import presets
from minion_voice.audio.engine import VoiceEngine
from minion_voice.control_server import ControlServer
from minion_voice.params import PARAM_SPECS_BY_NAME

SAMPLE_RATE = 48000
BLOCK = 256


def test_both_builtin_presets_exist():
    assert presets.preset_names() == ["animalese", "minion"]


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


def test_unknown_preset_raises():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=BLOCK)
    with pytest.raises(KeyError):
        engine.apply_preset("nope")


def test_presets_only_reference_valid_nonsession_params():
    for name, values in presets.PRESETS.items():
        for key in values:
            assert key in PARAM_SPECS_BY_NAME, f"{name} sets unknown param {key}"
            # Presets are voice character only -- never device/session state.
            assert not key.startswith("io."), f"{name} must not set {key}"
            assert key not in ("enabled", "monitor"), f"{name} must not set {key}"


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
    assert data["names"] == ["animalese", "minion"]

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
