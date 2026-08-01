"""Headless HTTP JSON control API for a live `VoiceEngine`.

Pure stdlib (`http.server`), no new dependencies. Wraps
`http.server.ThreadingHTTPServer` and runs it in a daemon thread so it can
sit alongside the Tkinter UI (both drive the same engine instance) or run
standalone (`minion_voice/server.py`).

Both the human (Tkinter) and the API mutate the engine exclusively
through `VoiceEngine.set_param`/`snapshot`, so a slider drag and a `curl`
are interchangeable, and the Tkinter sliders can pick up API-driven
changes on their next status poll.

Endpoints (all JSON in/out unless noted):

    GET  /api/state                -> {specs, values, status}
    POST /api/params                {"name": value, ...} -> {ok, values}
    POST /api/engine/start          {input_device?, output_device?}
    POST /api/engine/stop
    POST /api/engine/toggle
    GET  /api/devices               -> {input: [...], output: [...]}
    POST /api/devices               {input_device?, output_device?}
    POST /api/record/start
    POST /api/record/stop           -> {ok, take_seconds}
    POST /api/render                -> {ok, take_seconds}
    POST /api/play                  {"which": "raw"|"rendered"}
    POST /api/save                  {"path": ..., "which": "raw"|"rendered"}
    GET  /api/recording.wav?which=raw|rendered  -> audio/wav bytes
    GET  /                          -> minimal read-only status page
"""
from __future__ import annotations

import dataclasses
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from . import params as params_mod
from .audio.devices import list_input_devices, list_output_devices

_WEBUI_INDEX = Path(__file__).parent / "webui" / "index.html"


def _spec_to_dict(spec: params_mod.ParamSpec) -> dict:
    return dataclasses.asdict(spec)


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    # Quiet by default; flip on for debugging.
    def log_message(self, fmt, *args) -> None:  # noqa: D401
        pass

    # -- helpers -----------------------------------------------------

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    @property
    def engine(self):
        return self.server.engine

    # -- routing -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        query = parse_qs(urlsplit(self.path).query)
        try:
            if path == "/" or path == "/index.html":
                self._handle_index()
            elif path == "/api/state":
                self._handle_state()
            elif path == "/api/devices":
                self._handle_devices_get()
            elif path == "/api/recording.wav":
                which = (query.get("which") or ["rendered"])[0]
                self._handle_recording_wav(which)
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"no such route: {path}")
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/api/params":
                self._handle_params_post()
            elif path == "/api/engine/start":
                self._handle_engine_start()
            elif path == "/api/engine/stop":
                self._handle_engine_stop()
            elif path == "/api/engine/toggle":
                self._handle_engine_toggle()
            elif path == "/api/devices":
                self._handle_devices_post()
            elif path == "/api/record/start":
                self.engine.record_start()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/record/stop":
                self.engine.record_stop()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/render":
                self.engine.render_current()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/play":
                body = self._read_json_body()
                self.engine.play(body.get("which", "rendered"))
                self._send_json({"ok": True})
            elif path == "/api/save":
                body = self._read_json_body()
                path_arg = body.get("path")
                if not path_arg:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "missing 'path'")
                    return
                self.engine.save(path_arg, body.get("which", "rendered"))
                self._send_json({"ok": True, "path": path_arg})
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"no such route: {path}")
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # -- route handlers --------------------------------------------------

    def _handle_index(self) -> None:
        if _WEBUI_INDEX.exists():
            self._send_bytes(_WEBUI_INDEX.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send_bytes(b"<html><body>minion_voice control server</body></html>", "text/html")

    def _handle_state(self) -> None:
        self._send_json(
            {
                "specs": [_spec_to_dict(s) for s in params_mod.PARAM_SPECS],
                "values": self.engine.snapshot(),
                "status": self.engine.get_status(),
            }
        )

    def _handle_params_post(self) -> None:
        body = self._read_json_body()
        if not isinstance(body, dict) or not body:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "expected a JSON object of {name: value}")
            return
        errors = {}
        for name, value in body.items():
            try:
                self.engine.set_param(name, value)
            except Exception as exc:
                errors[name] = str(exc)
        if errors:
            self._send_json({"ok": False, "errors": errors, "values": self.engine.snapshot()}, status=HTTPStatus.BAD_REQUEST)
        else:
            self._send_json({"ok": True, "values": self.engine.snapshot()})

    def _handle_engine_start(self) -> None:
        body = self._read_json_body()
        input_device = body.get("input_device")
        output_device = body.get("output_device")
        self.engine.start(input_device=input_device, output_device=output_device)
        self._send_json({"ok": True, "status": self.engine.get_status()})

    def _handle_engine_stop(self) -> None:
        self.engine.stop()
        self._send_json({"ok": True, "status": self.engine.get_status()})

    def _handle_engine_toggle(self) -> None:
        if not self.engine.running:
            self.engine.start()
        self.engine.set_enabled(not self.engine.enabled)
        self._send_json({"ok": True, "status": self.engine.get_status()})

    def _handle_devices_get(self) -> None:
        self._send_json(
            {
                "input": [{"index": idx, "name": name} for idx, name in list_input_devices()],
                "output": [{"index": idx, "name": name} for idx, name in list_output_devices()],
                "current": {
                    "input_device": self.engine.input_device,
                    "output_device": self.engine.output_device,
                },
            }
        )

    def _handle_devices_post(self) -> None:
        body = self._read_json_body()
        if "input_device" in body:
            self.engine.set_param("io.input_device", body["input_device"])
        if "output_device" in body:
            self.engine.set_param("io.output_device", body["output_device"])
        self._send_json({"ok": True, "values": self.engine.snapshot()})

    def _handle_recording_wav(self, which: str) -> None:
        try:
            body = self.engine.take_bytes(which)
        except Exception as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        self._send_bytes(body, "audio/wav")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler_cls, engine) -> None:
        super().__init__(addr, handler_cls)
        self.engine = engine


class ControlServer:
    """Runs the HTTP control API for `engine` in a background daemon thread."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self._httpd: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> Optional[str]:
        if self._httpd is None:
            return None
        host, port = self._httpd.server_address[:2]
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    def start(self, host: str = "127.0.0.1", port: int = 8765) -> str:
        """Start the server; returns the base URL it's listening on.
        Pass `port=0` to bind an ephemeral free port (useful for tests)."""
        if self._httpd is not None:
            return self.base_url
        self._httpd = _Server((host, port), _Handler, self.engine)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="minion-control-server")
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
