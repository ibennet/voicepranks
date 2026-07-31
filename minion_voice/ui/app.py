"""Minimal tkinter GUI for the Minion Voice engine."""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Tuple

from ..audio.devices import default_input_device, list_input_devices, list_output_devices
from ..audio.engine import VoiceEngine

STATUS_REFRESH_MS = 250


class MinionVoiceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Minion Voice — voicepranks")

        self.engine = VoiceEngine()
        self.error_message: Optional[str] = None

        self._slider_dragging = False

        self._build_widgets()
        self._populate_devices()
        self._refresh_status()

    # -- UI construction -------------------------------------------------

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        self.toggle_var = tk.StringVar(value="Turn On")
        self.toggle_button = ttk.Button(frame, text="Turn On", command=self._on_toggle)
        self.toggle_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Intensity").grid(row=1, column=0, sticky="w")
        self.intensity_var = tk.DoubleVar(value=100.0)
        self.intensity_slider = ttk.Scale(
            frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.intensity_var,
            command=self._on_slider_move,
        )
        self.intensity_slider.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Input device").grid(row=2, column=0, sticky="w")
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(frame, textvariable=self.input_var, state="readonly")
        self.input_combo.grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Output device").grid(row=3, column=0, sticky="w")
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(frame, textvariable=self.output_var, state="readonly")
        self.output_combo.grid(row=3, column=1, sticky="ew", pady=4)

        self.status_label = ttk.Label(frame, text="", justify="left")
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        frame.columnconfigure(1, weight=1)

    def _populate_devices(self) -> None:
        self._input_devices: List[Tuple[int, str]] = []
        self._output_devices: List[Tuple[int, str]] = []

        try:
            self._input_devices = list_input_devices()
            self.input_combo["values"] = [f"{idx}: {name}" for idx, name in self._input_devices]
            try:
                default_idx = default_input_device()
            except Exception:
                default_idx = None
            for i, (idx, _name) in enumerate(self._input_devices):
                if idx == default_idx:
                    self.input_combo.current(i)
                    break
            else:
                if self._input_devices:
                    self.input_combo.current(0)
        except Exception as exc:
            self.error_message = f"Could not list input devices: {exc}"

        try:
            self._output_devices = list_output_devices()
            self.output_combo["values"] = [f"{idx}: {name}" for idx, name in self._output_devices]

            virtual_idx = None
            try:
                from ..audio.devices import find_virtual_output
                virtual_idx = find_virtual_output()
            except RuntimeError as exc:
                self.error_message = str(exc)

            for i, (idx, _name) in enumerate(self._output_devices):
                if idx == virtual_idx:
                    self.output_combo.current(i)
                    break
            else:
                if self._output_devices:
                    self.output_combo.current(0)
        except Exception as exc:
            self.error_message = f"Could not list output devices: {exc}"

    def _selected_device_index(self, combo: ttk.Combobox) -> Optional[int]:
        value = combo.get()
        if not value:
            return None
        try:
            return int(value.split(":", 1)[0])
        except (ValueError, IndexError):
            return None

    # -- event handlers ----------------------------------------------------

    def _on_toggle(self) -> None:
        if not self.engine.running:
            input_idx = self._selected_device_index(self.input_combo)
            output_idx = self._selected_device_index(self.output_combo)
            try:
                self.engine.start(input_device=input_idx, output_device=output_idx)
            except RuntimeError as exc:
                self.error_message = str(exc)
                self._refresh_status()
                return
            except Exception as exc:
                self.error_message = f"Failed to start audio: {exc}"
                self._refresh_status()
                return

        currently_enabled = self.engine.enabled
        self.engine.set_enabled(not currently_enabled)

        if not self._slider_dragging and self.engine.enabled:
            # Let the auto-ramp drive intensity unless the user has already
            # dragged the slider to a manual value.
            pass

        self.toggle_button.config(text="Turn Off" if self.engine.enabled else "Turn On")

    def _on_slider_move(self, _value: str) -> None:
        self._slider_dragging = True
        t = self.intensity_var.get() / 100.0
        self.engine.set_manual_intensity(t)

    # -- status polling ------------------------------------------------

    def _refresh_status(self) -> None:
        status = self.engine.get_status()
        error_suffix = f"\nError: {self.error_message}" if self.error_message else ""
        text = (
            f"Intensity: {status['intensity'] * 100:.0f}%   "
            f"Underruns: {status['underruns']}   "
            f"Overruns: {status['overruns']}   "
            f"Running: {status['running']}"
            f"{error_suffix}"
        )
        self.status_label.config(text=text)
        self.root.after(STATUS_REFRESH_MS, self._refresh_status)


def _warn_if_broken_tk() -> None:
    """Apple's system-Python Tk (8.5) renders blank/broken windows on modern
    macOS. Warn loudly with the fix instead of leaving the user staring at an
    empty window."""
    if tk.TkVersion < 8.6:
        sys.stderr.write(
            "\n[minion_voice] WARNING: this Python is using Tk "
            f"{tk.TkVersion}, which renders blank windows on modern macOS.\n"
            "  Fix: install a Python with a newer Tk, e.g.\n"
            "    brew install python-tk\n"
            "  then recreate your venv with that Python "
            "(/opt/homebrew/bin/python3).\n\n"
        )


def run() -> None:
    _warn_if_broken_tk()
    root = tk.Tk()
    MinionVoiceApp(root)
    root.mainloop()
