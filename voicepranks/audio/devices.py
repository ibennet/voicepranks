"""Audio device discovery helpers (sounddevice-backed)."""
from __future__ import annotations

from typing import List, Tuple

import sounddevice as sd

_VIRTUAL_DEVICE_NAME_HINTS = ("cable", "blackhole")

_INSTALL_INSTRUCTIONS = (
    "No virtual audio output device was found.\n"
    "This app needs a virtual audio cable so other apps (Discord/Zoom/OBS) "
    "can pick up the processed microphone signal.\n\n"
    "Windows: install VB-CABLE from https://vb-audio.com/Cable/\n"
    "macOS: install BlackHole from https://existential.audio/blackhole/\n\n"
    "After installing, restart this app so it can detect the new device."
)


def list_input_devices() -> List[Tuple[int, str]]:
    """Return [(index, name)] for all devices with input channels."""
    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            result.append((idx, dev.get("name", f"Device {idx}")))
    return result


def list_output_devices() -> List[Tuple[int, str]]:
    """Return [(index, name)] for all devices with output channels."""
    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0:
            result.append((idx, dev.get("name", f"Device {idx}")))
    return result


def default_input_device() -> int:
    """Return the index of the system default input device."""
    default = sd.default.device
    if isinstance(default, (list, tuple)):
        idx = default[0]
    else:
        idx = default
    if idx is None or idx < 0:
        raise RuntimeError("No default input device is available on this system.")
    return int(idx)


def find_virtual_output() -> int:
    """Find an output-capable virtual audio device (VB-CABLE / BlackHole).

    Raises RuntimeError with install instructions if none is found.
    """
    for idx, name in list_output_devices():
        lowered = name.lower()
        if any(hint in lowered for hint in _VIRTUAL_DEVICE_NAME_HINTS):
            return idx
    raise RuntimeError(_INSTALL_INSTRUCTIONS)
