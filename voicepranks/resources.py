"""Locate bundled package data files -- both from source and inside a
PyInstaller build.

PyInstaller unpacks bundled `datas` under `sys._MEIPASS` at runtime. On macOS a
one-folder `.app` splits the payload (code under `Contents/Frameworks`, data
under `Contents/Resources`), so a `__file__`-relative path to a data file misses
it entirely; `sys._MEIPASS` -- which PyInstaller points at the data root, with
the symlinks needed to reach it -- is the reliable base. When not frozen, fall
back to the source-tree layout relative to this package.

Every bundled asset must resolve through `resource_path` so it's found in both
run modes; a `__file__`-relative path silently works from source but breaks in
the packaged app (notably the macOS .app), which is exactly the trap this avoids.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Package root (the directory holding this file), used when running from source.
_PACKAGE_ROOT = Path(__file__).resolve().parent


def resource_path(*parts: str) -> Path:
    """Absolute path to a bundled data file from its package-relative parts,
    e.g. ``resource_path("assets", "goofy_laugh.wav")``. Works when running from
    source and inside a PyInstaller build (including a macOS ``.app``)."""
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        return Path(base) / "voicepranks" / Path(*parts)
    return _PACKAGE_ROOT / Path(*parts)
