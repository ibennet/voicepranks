"""ControlServer HTTP round-trip tests. No audio hardware is touched --
the engine is never started (no `sd.InputStream`/`OutputStream`), only
its param registry and status/snapshot machinery, driven purely over
`http.client` against a real `ThreadingHTTPServer` on an ephemeral port.
"""
from __future__ import annotations

import http.client
import json

import numpy as np
import pytest

from voicepranks.audio.engine import VoiceEngine
from voicepranks.control_server import TOKEN_HEADER, ControlServer, resolve_save_path


TEST_TOKEN = "test-token-abc123"


@pytest.fixture
def server(tmp_path):
    engine = VoiceEngine()
    srv = ControlServer(engine, token=TEST_TOKEN, save_root=tmp_path / "recordings")
    base_url = srv.start(host="127.0.0.1", port=0)
    yield srv, engine, base_url
    srv.stop()


def _get(base_url: str, path: str, headers: dict = None):
    host_port = base_url.split("://", 1)[1]
    conn = http.client.HTTPConnection(host_port)
    sent = {TOKEN_HEADER: TEST_TOKEN} if headers is None else headers
    conn.request("GET", path, headers=sent)
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    return resp.status, body


def _post(base_url: str, path: str, payload: dict, headers: dict = None):
    host_port = base_url.split("://", 1)[1]
    conn = http.client.HTTPConnection(host_port)
    body = json.dumps(payload).encode("utf-8")
    sent = {"Content-Type": "application/json"}
    sent.update({TOKEN_HEADER: TEST_TOKEN} if headers is None else headers)
    conn.request("POST", path, body=body, headers=sent)
    resp = conn.getresponse()
    out = json.loads(resp.read())
    conn.close()
    return resp.status, out


def test_get_state_returns_specs_values_status(server):
    _srv, _engine, base_url = server
    status, data = _get(base_url, "/api/state")
    assert status == 200
    assert "specs" in data and "values" in data and "status" in data
    assert len(data["specs"]) == len(data["values"])
    names = {spec["name"] for spec in data["specs"]}
    assert "minionese.semitones" in names
    assert "effect.max_semitones" in names


def test_post_params_then_get_shows_new_value(server):
    _srv, _engine, base_url = server

    status, out = _post(base_url, "/api/params", {"minionese.semitones": 6.5})
    assert status == 200
    assert out["ok"] is True
    assert out["values"]["minionese.semitones"] == pytest.approx(6.5)

    _status, data = _get(base_url, "/api/state")
    assert data["values"]["minionese.semitones"] == pytest.approx(6.5)


def test_post_params_bool_and_multi(server):
    _srv, engine, base_url = server

    status, out = _post(base_url, "/api/params", {"gibberish": True, "effect.eq_enabled": True})
    assert status == 200
    assert out["ok"] is True
    assert engine.effect.gibberish is True
    assert engine.effect.eq_enabled is True


def test_post_params_unknown_name_reports_error_but_keeps_server_alive(server):
    _srv, _engine, base_url = server

    status, out = _post(base_url, "/api/params", {"nonexistent.param": 1.0})
    assert status == 400
    assert out["ok"] is False
    assert "nonexistent.param" in out["errors"]

    # Server should still be responsive after an error.
    status2, _data = _get(base_url, "/api/state")
    assert status2 == 200


def test_record_render_play_flow_over_http(server, monkeypatch):
    _srv, engine, base_url = server

    played = {}

    def fake_play(buf, sample_rate, device=None):
        played["buf"] = buf
        played["sample_rate"] = sample_rate

    monkeypatch.setattr("voicepranks.audio.engine.sd.play", fake_play)

    status, _out = _post(base_url, "/api/record/start", {})
    assert status == 200

    sine = (0.3 * np.sin(2 * np.pi * 220 * np.arange(4096) / engine.sample_rate)).astype(np.float32)
    for i in range(0, len(sine), 256):
        engine._input_callback(sine[i:i + 256].reshape(-1, 1), 256, None, None)

    status, out = _post(base_url, "/api/record/stop", {})
    assert status == 200
    assert out["status"]["has_raw_take"] is True

    status, out = _post(base_url, "/api/render", {})
    assert status == 200
    assert out["status"]["has_rendered_take"] is True

    status, out = _post(base_url, "/api/play", {"which": "rendered"})
    assert status == 200
    assert out["ok"] is True
    assert "buf" in played
    assert not np.any(np.isnan(played["buf"]))


def test_devices_get_lists_input_and_output(server):
    _srv, _engine, base_url = server
    status, data = _get(base_url, "/api/devices")
    assert status == 200
    assert "input" in data and "output" in data


# -- security regressions -------------------------------------------------
# The control API can open the microphone and read recordings back, so these
# pin the auth gates rather than any user-visible feature.


def test_api_rejects_request_without_token(server):
    _srv, _engine, base_url = server
    status, body = _get(base_url, "/api/state", headers={})
    assert status == 403
    assert body["ok"] is False


def test_api_rejects_wrong_token(server):
    _srv, _engine, base_url = server
    status, _ = _get(base_url, "/api/state", headers={TOKEN_HEADER: "wrong"})
    assert status == 403


def test_token_accepted_as_query_param(server):
    _srv, _engine, base_url = server
    status, body = _get(base_url, f"/api/state?token={TEST_TOKEN}", headers={})
    assert status == 200
    assert "values" in body


def test_post_without_token_cannot_start_engine(server):
    _srv, _engine, base_url = server
    status, _ = _post(base_url, "/api/engine/toggle", {}, headers={})
    assert status == 403


def test_cross_origin_request_rejected_even_with_token(server):
    """A page on evil.com can't forge Origin, so this is the drive-by case."""
    _srv, _engine, base_url = server
    status, _ = _get(
        base_url,
        "/api/state",
        headers={TOKEN_HEADER: TEST_TOKEN, "Origin": "https://evil.example"},
    )
    assert status == 403


def test_same_origin_request_accepted(server):
    """The browser UI sends its own loopback Origin -- the gate must let it through."""
    _srv, _engine, base_url = server
    status, body = _get(
        base_url,
        "/api/state",
        headers={TOKEN_HEADER: TEST_TOKEN, "Origin": base_url},
    )
    assert status == 200
    assert "values" in body


def test_rebound_host_header_rejected(server):
    """DNS rebinding resolves attacker.example to 127.0.0.1 but keeps its Host."""
    _srv, _engine, base_url = server
    status, _ = _get(
        base_url,
        "/api/state",
        headers={TOKEN_HEADER: TEST_TOKEN, "Host": "attacker.example"},
    )
    assert status == 403


def test_no_wildcard_cors_header(server):
    _srv, _engine, base_url = server
    host_port = base_url.split("://", 1)[1]
    conn = http.client.HTTPConnection(host_port)
    conn.request("GET", "/api/state", headers={TOKEN_HEADER: TEST_TOKEN})
    resp = conn.getresponse()
    resp.read()
    assert resp.getheader("Access-Control-Allow-Origin") is None
    conn.close()


def test_index_page_served_without_token_and_carries_one(server):
    srv, _engine, base_url = server
    host_port = base_url.split("://", 1)[1]
    conn = http.client.HTTPConnection(host_port)
    conn.request("GET", "/")
    resp = conn.getresponse()
    page = resp.read().decode("utf-8")
    conn.close()
    assert resp.status == 200
    assert srv.token in page
    assert "__CONTROL_TOKEN__" not in page


@pytest.mark.parametrize(
    "bad",
    [
        "../../../../etc/passwd.wav",
        "/etc/passwd.wav",
        "sub/../../escape.wav",
        "notawav.txt",
    ],
)
def test_resolve_save_path_rejects_escapes(tmp_path, bad):
    with pytest.raises(ValueError):
        resolve_save_path(bad, tmp_path)


def test_resolve_save_path_allows_nested_relative_wav(tmp_path):
    out = resolve_save_path("takes/take1.wav", tmp_path)
    assert out.parent.exists()
    assert str(out).startswith(str(tmp_path.resolve()))


def test_save_endpoint_rejects_traversal(server):
    _srv, _engine, base_url = server
    status, body = _post(base_url, "/api/save", {"path": "../../pwned.wav", "which": "live"})
    assert status == 400
    assert "ok" in body and body["ok"] is False
