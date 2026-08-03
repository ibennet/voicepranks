"""Persistent user settings, editable outside the app.

Holds the three audio devices the app routes between:

- ``input_device``    -- the physical mic to record/process from
- ``output_device``   -- the virtual cable other apps hear as a mic
                         (VB-CABLE / BlackHole); where processed audio goes
- ``playback_device`` -- your speakers/headphones, used only for *listening*
                         (the Play button and the live monitor)

Settings live in a small JSON file at ``~/.voicepranks/settings.json`` so
they survive restarts and can be edited by hand outside the application.
Each device is stored as ``{"index": int, "name": str}``; on load we prefer
to re-resolve by *name* against the currently-available devices (indices can
shuffle between reboots / device plug events) and fall back to the stored
index, then to ``None`` (system default / auto).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SETTINGS_DIR = Path.home() / ".voicepranks"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

_DEVICE_KEYS = ("input_device", "output_device", "playback_device")


def load() -> Dict:
    """Load the settings dict, or an empty dict if none / unreadable."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(settings: Dict) -> None:
    """Write the settings dict to disk (creating the directory)."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
    tmp.replace(SETTINGS_PATH)


def device_entry(index: Optional[int], name: Optional[str]) -> Optional[Dict]:
    """Build a stored device entry, or None when no device is selected."""
    if index is None:
        return None
    return {"index": int(index), "name": name}


def resolve_device(
    entry: Optional[Dict],
    available: List[Tuple[int, str]],
) -> Optional[int]:
    """Resolve a stored device entry to a current device index.

    Prefers matching by name (robust to index reshuffles); falls back to the
    stored index if it's still a valid device; else None (default / auto).
    `available` is a list of (index, name) as returned by the device
    listing helpers.
    """
    if not entry:
        return None
    valid = {idx for idx, _ in available}
    name = entry.get("name")
    if name:
        for idx, dev_name in available:
            if dev_name == name:
                return idx
    idx = entry.get("index")
    if isinstance(idx, int) and idx in valid:
        return idx
    return None


def resolve_all(
    settings: Dict,
    input_devices: List[Tuple[int, str]],
    output_devices: List[Tuple[int, str]],
) -> Dict[str, Optional[int]]:
    """Resolve all three stored devices to current indices.

    Playback is an output device, so it resolves against `output_devices`.
    """
    return {
        "input_device": resolve_device(settings.get("input_device"), input_devices),
        "output_device": resolve_device(settings.get("output_device"), output_devices),
        "playback_device": resolve_device(settings.get("playback_device"), output_devices),
    }


def update_devices(settings: Dict, **entries: Optional[Dict]) -> Dict:
    """Return a copy of `settings` with the given device entries replaced.

    Pass e.g. ``input_device=device_entry(3, "Mic")`` or ``None`` to clear.
    Only recognized device keys are applied.
    """
    out = dict(settings)
    for key, entry in entries.items():
        if key in _DEVICE_KEYS:
            out[key] = entry
    return out
