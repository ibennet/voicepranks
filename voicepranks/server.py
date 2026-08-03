"""Headless entrypoint: run the voice engine + HTTP control API with no UI.

    python -m voicepranks.server

Starts the same `VoiceEngine` + `ControlServer` that the Tkinter app wires
up, but with no window -- useful for running on a machine without a
display, or for driving the whole thing purely over the JSON API
(`curl`/Claude). The engine itself is *not* started (no mic capture) until
`POST /api/engine/start` or `/api/engine/toggle` is called, so this is
safe to launch even before picking devices.
"""
from __future__ import annotations

import os
import sys
import time

from .audio.engine import VoiceEngine
from .control_server import ControlServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    host = os.environ.get("MINION_SERVER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("MINION_SERVER_PORT", str(DEFAULT_PORT)))
    if len(argv) >= 1:
        host = argv[0]
    if len(argv) >= 2:
        port = int(argv[1])

    engine = VoiceEngine()
    server = ControlServer(engine)
    base_url = server.start(host=host, port=port)

    print(f"voicepranks control server listening at {base_url}")
    print("Endpoints:")
    print(f"  GET  {base_url}/api/state")
    print(f"  POST {base_url}/api/params            {{'name': value, ...}}")
    print(f"  POST {base_url}/api/engine/start       {{input_device?, output_device?}}")
    print(f"  POST {base_url}/api/engine/stop")
    print(f"  POST {base_url}/api/engine/toggle")
    print(f"  GET  {base_url}/api/devices")
    print(f"  POST {base_url}/api/devices            {{input_device?, output_device?}}")
    print(f"  POST {base_url}/api/record/start")
    print(f"  POST {base_url}/api/record/stop")
    print(f"  POST {base_url}/api/render")
    print(f"  POST {base_url}/api/play               {{which: 'raw'|'rendered'}}")
    print(f"  POST {base_url}/api/save               {{path, which}}")
    print(f"  GET  {base_url}/api/recording.wav?which=raw|rendered")
    print(f"  GET  {base_url}/                        (status page)")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if engine.running:
            engine.stop()


if __name__ == "__main__":
    main()
