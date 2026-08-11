"""Headless HTTP JSON control API for a live `VoiceEngine`.

Pure stdlib (`http.server`), no new dependencies. Wraps
`http.server.ThreadingHTTPServer` and runs it in a daemon thread so it can
sit alongside the Tkinter UI (both drive the same engine instance) or run
standalone (`voicepranks/server.py`).

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
    POST /api/laugh/record/start    -> {ok, status}   (record a custom goofy laugh)
    POST /api/laugh/record/stop     -> {ok, status}   (install it; 400 if silent)
    POST /api/laugh/reset           -> {ok, status}   (revert to the stock laugh)
    GET  /api/presets               -> {names, presets}
    POST /api/presets/apply         {"name": "minion"} -> {ok, values}
    POST /api/render                -> {ok, take_seconds}
    POST /api/play                  {"which": "live"|"raw"|"rendered"}
    POST /api/save                  {"path": ..., "which": "live"|"raw"|"rendered"}
    GET  /api/recording.wav?which=live|raw|rendered  -> audio/wav bytes
    GET  /                          -> minimal read-only status page
"""
from __future__ import annotations

import dataclasses
import hmac
import json
import os
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from . import params as params_mod
from . import presets as presets_mod
from . import settings as settings_mod
from .audio.devices import list_input_devices, list_output_devices
from .resources import resource_path

# Resolved via `resource_path` so the status page is found inside a PyInstaller
# build too (a `__file__`-relative path misses it in the packaged macOS .app).
_WEBUI_INDEX = resource_path("webui", "index.html")

# The API can start the mic and read back recordings, so it is authenticated
# even though it only listens on loopback: any web page you visit can reach
# 127.0.0.1, and a browser will happily attach no credentials at all.
TOKEN_HEADER = "X-VoicePranks-Token"
_TOKEN_PLACEHOLDER = "__CONTROL_TOKEN__"

# Hostnames that mean "this machine". A DNS-rebinding attacker resolves their
# own domain to 127.0.0.1, so the Host header (not the socket address) is what
# distinguishes a real local client from a rebound one.
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}

# `POST /api/save` writes a WAV wherever it is told, so confine it to one
# directory rather than letting an API caller choose an arbitrary path.
SAVE_ROOT = settings_mod.SETTINGS_DIR / "recordings"


def _is_loopback_name(hostname: Optional[str]) -> bool:
    if not hostname:
        return False
    return hostname.strip("[]").lower() in _LOOPBACK_NAMES


def resolve_save_path(raw: str, root: Optional[Path] = None) -> Path:
    """Map a caller-supplied `/api/save` path onto a file inside `root`.

    Rejects absolute paths, drive letters and `..` traversal, then re-checks
    the *resolved* path is still under `root` so a symlink inside the
    directory can't be used to escape it. Raises ValueError if it can't.
    """
    root = (root or SAVE_ROOT).expanduser()
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise ValueError("path must be relative to the recordings directory")
    if any(part == ".." for part in candidate.parts):
        raise ValueError("path must not contain '..'")
    if candidate.suffix.lower() != ".wav":
        raise ValueError("path must end in .wav")

    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    full = (resolved_root / candidate).resolve()
    if full != resolved_root and resolved_root not in full.parents:
        raise ValueError("path escapes the recordings directory")
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


# PARAM_SPECS are frozen and never change, so serialize them once at import
# rather than deep-copying every spec on each /api/state request.
_SPECS_JSON = [dataclasses.asdict(spec) for spec in params_mod.PARAM_SPECS]


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
        # Deliberately no Access-Control-Allow-Origin: a wildcard here would
        # let any site you visit read /api/state and the recorded audio.
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

    # -- authentication --------------------------------------------------

    def _authorized(self, query: dict) -> bool:
        """True if this request may touch /api/*.

        Three independent gates, all cheap:
        1. `Host` must name loopback -- defeats DNS rebinding, where an
           attacker's domain resolves to 127.0.0.1 but still sends its own
           name in Host.
        2. `Origin`, when present, must be loopback -- a browser sets it on
           cross-site requests and scripts cannot forge it, so this blocks
           drive-by POSTs from a page you happen to have open.
        3. A token must match. The status page gets it injected server-side;
           `curl` users read it from the startup banner or the token file.
        """
        if not _is_loopback_name(urlsplit(f"//{self.headers.get('Host', '')}").hostname):
            return False

        origin = self.headers.get("Origin")
        if origin is not None and not _is_loopback_name(urlsplit(origin).hostname):
            return False

        supplied = self.headers.get(TOKEN_HEADER) or (query.get("token") or [""])[0]
        return hmac.compare_digest(supplied, self.server.token)

    # -- routing -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        query = parse_qs(urlsplit(self.path).query)
        # "/" is the token-bearing status page itself, so it is the one route
        # that can't require a token. It exposes no engine state, and the
        # token it carries is unreadable cross-origin now that the wildcard
        # CORS header is gone.
        if path.startswith("/api/") and not self._authorized(query):
            self._send_error_json(HTTPStatus.FORBIDDEN, "missing or invalid control token")
            return
        try:
            if path == "/" or path == "/index.html":
                self._handle_index()
            elif path == "/api/state":
                self._handle_state()
            elif path == "/api/devices":
                self._handle_devices_get()
            elif path == "/api/presets":
                self._send_json({"names": presets_mod.preset_names(), "presets": presets_mod.PRESETS})
            elif path == "/api/recording.wav":
                which = (query.get("which") or ["live"])[0]
                self._handle_recording_wav(which)
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"no such route: {path}")
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not self._authorized(parse_qs(urlsplit(self.path).query)):
            self._send_error_json(HTTPStatus.FORBIDDEN, "missing or invalid control token")
            return
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
            elif path == "/api/presets/apply":
                body = self._read_json_body()
                name = body.get("name")
                if not name:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "missing 'name'")
                    return
                try:
                    values = self.engine.apply_preset(name)
                except KeyError as exc:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send_json({"ok": True, "values": values})
            elif path == "/api/record/start":
                self.engine.record_start()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/record/stop":
                self.engine.record_stop()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/laugh/record/start":
                self.engine.record_laugh_start()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/laugh/record/stop":
                try:
                    self.engine.record_laugh_stop()
                except ValueError as exc:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/laugh/reset":
                self.engine.reset_laugh_to_stock()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/render":
                self.engine.render_current()
                self._send_json({"ok": True, "status": self.engine.get_status()})
            elif path == "/api/play":
                body = self._read_json_body()
                self.engine.play(body.get("which", "live"))
                self._send_json({"ok": True})
            elif path == "/api/save":
                body = self._read_json_body()
                path_arg = body.get("path")
                if not path_arg:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "missing 'path'")
                    return
                try:
                    target = resolve_save_path(str(path_arg), self.server.save_root)
                except ValueError as exc:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self.engine.save(str(target), body.get("which", "live"))
                self._send_json({"ok": True, "path": str(target)})
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"no such route: {path}")
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # -- route handlers --------------------------------------------------

    def _handle_index(self) -> None:
        if _WEBUI_INDEX.exists():
            # Hand the page its own token so opening the URL still just works.
            # Safe to embed: without wildcard CORS a cross-origin script can
            # issue this request but cannot read the response back.
            page = _WEBUI_INDEX.read_text(encoding="utf-8").replace(
                _TOKEN_PLACEHOLDER, self.server.token
            )
            self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send_bytes(b"<html><body>voicepranks control server</body></html>", "text/html")

    def _handle_state(self) -> None:
        self._send_json(
            {
                "specs": _SPECS_JSON,
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

    def __init__(self, addr, handler_cls, engine, token, save_root) -> None:
        super().__init__(addr, handler_cls)
        self.engine = engine
        self.token = token
        self.save_root = save_root


class ControlServer:
    """Runs the HTTP control API for `engine` in a background daemon thread.

    Every `/api/*` call must present `token` (via the `X-VoicePranks-Token`
    header or a `?token=` query param). One is generated per process unless
    `VOICEPRANKS_CONTROL_TOKEN` overrides it, and `start()` drops a copy in
    `~/.voicepranks/control-token` so CLI callers can pick it up.
    """

    def __init__(self, engine, token: Optional[str] = None, save_root: Optional[Path] = None) -> None:
        self.engine = engine
        self.token = token or os.environ.get("VOICEPRANKS_CONTROL_TOKEN") or secrets.token_urlsafe(32)
        self.save_root = save_root or SAVE_ROOT
        self._httpd: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def token_path(self) -> Path:
        return settings_mod.SETTINGS_DIR / "control-token"

    def _write_token_file(self) -> None:
        """Persist the token 0600 so `curl` users can read it back.

        Created with the mode already applied rather than write-then-chmod,
        which would leave the token briefly readable at the umask default.
        """
        try:
            settings_mod.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.token)
        except OSError:
            pass  # non-fatal: the banner still prints the token

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
        self._write_token_file()
        self._httpd = _Server((host, port), _Handler, self.engine, self.token, self.save_root)
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
