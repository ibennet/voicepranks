"""Minimal tkinter GUI for the Minion Voice engine.

Every tunable in `params.PARAM_SPECS` gets a generated slider/checkbox in
a scrollable grid here, so the desktop UI never hardcodes the param list
-- add a knob to the registry and it shows up automatically. The same
registry backs the HTTP control API (`control_server.py`), which this app
also starts by default (set `MINION_NO_SERVER=1` to disable), so Claude
or a `curl` can tweak params live while this window is open; sliders pick
up out-of-band API changes on the next status poll.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

from ..audio.devices import default_input_device, list_input_devices, list_output_devices
from ..audio.engine import VoiceEngine
from ..control_server import ControlServer
from ..params import PARAM_SPECS, ParamSpec

STATUS_REFRESH_MS = 250

GROUP_LABELS = {
    "global": "Global",
    "effect": "Plain mode (pitch + EQ)",
    "minionese": "Minionese — formant engine",
    "shuffle": "Minionese — shuffle engine",
    "pitch": "Pitch engine (advanced)",
    "io": "I/O (restarts audio)",
}
GROUP_ORDER = ["global", "effect", "minionese", "shuffle", "pitch", "io"]

# These are already covered by the dedicated device combos below, so they
# are left out of the generated grid to avoid two controls for one param.
_SKIP_PARAM_NAMES = {"enabled", "io.input_device", "io.output_device", "monitor"}


class MinionVoiceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Minion Voice — voicepranks")

        self.engine = VoiceEngine()
        self.error_message: Optional[str] = None

        self._slider_dragging = False  # legacy flag, kept for compat
        self._dragging: Dict[str, bool] = {}
        self._param_widgets: Dict[str, dict] = {}
        self._suppress_commands = False

        self.control_server: Optional[ControlServer] = None
        if os.environ.get("MINION_NO_SERVER") != "1":
            self.control_server = ControlServer(self.engine)
            url = self.control_server.start()
            sys.stderr.write(f"[minion_voice] control API listening at {url}\n")

        self._build_widgets()
        self._populate_devices()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_status()

    # -- UI construction -------------------------------------------------

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)
        frame.columnconfigure(1, weight=1)

        self.toggle_button = ttk.Button(frame, text="Turn On", command=self._on_toggle)
        self.toggle_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Input device").grid(row=1, column=0, sticky="w")
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(frame, textvariable=self.input_var, state="readonly")
        self.input_combo.grid(row=1, column=1, sticky="ew", pady=4)
        self.input_combo.bind("<<ComboboxSelected>>", self._on_input_device_change)

        ttk.Label(frame, text="Output device").grid(row=2, column=0, sticky="w")
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(frame, textvariable=self.output_var, state="readonly")
        self.output_combo.grid(row=2, column=1, sticky="ew", pady=4)
        self.output_combo.bind("<<ComboboxSelected>>", self._on_output_device_change)

        self._build_recorder_row(frame, row=3)

        # Scrollable area holding one generated control per PARAM_SPECS
        # entry, grouped by `spec.group`.
        scroll_container = ttk.Frame(frame)
        scroll_container.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(4, 8))
        scroll_container.rowconfigure(0, weight=1)
        scroll_container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_container, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._param_grid = ttk.Frame(canvas)
        self._param_grid.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._param_grid, anchor="nw")
        self._param_grid.columnconfigure(1, weight=1)

        self._build_param_controls(self._param_grid)

        self.status_label = ttk.Label(frame, text="", justify="left")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_recorder_row(self, frame: ttk.Frame, row: int) -> None:
        rec_frame = ttk.Frame(frame)
        rec_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(rec_frame, text="Record", command=self._on_record).pack(side="left", padx=(0, 4))
        ttk.Button(rec_frame, text="Stop", command=self._on_record_stop).pack(side="left", padx=4)
        # Primary flow: Play replays the live processed take verbatim (the
        # effect as it was applied in real time -- no re-render).
        ttk.Button(rec_frame, text="Play", command=self._on_play).pack(side="left", padx=4)
        # Secondary A/B flow: re-render the dry take with current params.
        ttk.Button(rec_frame, text="Re-render & Play", command=self._on_rerender_play).pack(side="left", padx=4)

        # Live monitor: also hear the processed effect on your speakers while
        # recording. Off by default -- Record works without it. Bound to the
        # same `monitor` param as the API, hence skipped from the generated
        # grid (see `_SKIP_PARAM_NAMES`).
        self.monitor_var = tk.BooleanVar(value=self.engine.monitor_enabled)
        ttk.Checkbutton(
            rec_frame,
            text="Live monitor",
            variable=self.monitor_var,
            command=self._on_monitor_toggle,
        ).pack(side="left", padx=(12, 0))

    def _build_param_controls(self, parent: ttk.Frame) -> None:
        groups: Dict[str, List[ParamSpec]] = {}
        for spec in PARAM_SPECS:
            if spec.name in _SKIP_PARAM_NAMES:
                continue
            groups.setdefault(spec.group, []).append(spec)

        row = 0
        for group in GROUP_ORDER:
            specs = groups.get(group)
            if not specs:
                continue
            header = ttk.Label(parent, text=GROUP_LABELS.get(group, group))
            header.configure(font=("TkDefaultFont", 10, "bold"))
            header.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 2))
            row += 1
            for spec in specs:
                self._build_one_control(parent, spec, row)
                row += 1

    def _build_one_control(self, parent: ttk.Frame, spec: ParamSpec, row: int) -> None:
        ttk.Label(parent, text=spec.label).grid(row=row, column=0, sticky="w", padx=(4, 4))

        if spec.kind == "bool":
            var = tk.BooleanVar(value=bool(spec.default))
            cb = ttk.Checkbutton(
                parent, variable=var, command=lambda s=spec, v=var: self._on_param_checkbox(s, v)
            )
            cb.grid(row=row, column=1, columnspan=2, sticky="w")
            self._param_widgets[spec.name] = {"kind": "bool", "var": var, "spec": spec}
            return

        var = tk.DoubleVar(value=float(spec.default))
        scale = ttk.Scale(
            parent,
            from_=spec.min,
            to=spec.max,
            orient="horizontal",
            variable=var,
            command=lambda val, s=spec: self._on_param_slider(s, val),
        )
        scale.grid(row=row, column=1, sticky="ew", padx=(4, 4))
        readout = ttk.Label(parent, text=self._format_value(spec, spec.default), width=8)
        readout.grid(row=row, column=2, sticky="w")
        scale.bind("<ButtonPress-1>", lambda _e, n=spec.name: self._dragging.__setitem__(n, True))
        scale.bind("<ButtonRelease-1>", lambda _e, n=spec.name: self._dragging.__setitem__(n, False))
        self._param_widgets[spec.name] = {"kind": spec.kind, "var": var, "readout": readout, "spec": spec}

    @staticmethod
    def _format_value(spec: ParamSpec, value) -> str:
        if spec.kind == "int":
            return str(int(round(float(value))))
        return f"{float(value):.3g}"

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

        self.engine.set_enabled(not self.engine.enabled)
        self.toggle_button.config(text="Turn Off" if self.engine.enabled else "Turn On")

    def _on_input_device_change(self, _evt) -> None:
        idx = self._selected_device_index(self.input_combo)
        if idx is None:
            return
        try:
            self.engine.set_param("io.input_device", idx)
        except Exception as exc:
            self.error_message = f"Failed to switch input device: {exc}"

    def _on_output_device_change(self, _evt) -> None:
        idx = self._selected_device_index(self.output_combo)
        if idx is None:
            return
        try:
            self.engine.set_param("io.output_device", idx)
        except Exception as exc:
            self.error_message = f"Failed to switch output device: {exc}"

    def _on_param_slider(self, spec: ParamSpec, value_str: str) -> None:
        if self._suppress_commands:
            return
        self._slider_dragging = True
        try:
            value = float(value_str)
        except (TypeError, ValueError):
            return
        if spec.kind == "int":
            value = int(round(value))
        widget = self._param_widgets.get(spec.name)
        if widget is not None:
            widget["readout"].config(text=self._format_value(spec, value))
        try:
            self.engine.set_param(spec.name, value)
        except Exception as exc:
            self.error_message = f"Failed to set {spec.name}: {exc}"

    def _on_param_checkbox(self, spec: ParamSpec, var: tk.BooleanVar) -> None:
        if self._suppress_commands:
            return
        try:
            self.engine.set_param(spec.name, bool(var.get()))
        except Exception as exc:
            self.error_message = f"Failed to set {spec.name}: {exc}"

    # -- recorder ----------------------------------------------------------

    def _on_record(self) -> None:
        # Run the effect live during capture so the user hears the real-time
        # result (via the live monitor) while recording, not just the offline
        # re-render. Recording still captures the DRY signal, so re-render
        # with tweaked params keeps working.
        if not self.engine.running:
            input_idx = self._selected_device_index(self.input_combo)
            output_idx = self._selected_device_index(self.output_combo)
            try:
                self.engine.start(input_device=input_idx, output_device=output_idx)
            except Exception as exc:
                self.error_message = f"Could not start audio for recording: {exc}"
                return
        if not self.engine.enabled:
            self.engine.set_enabled(True)
            self.toggle_button.config(text="Turn Off")
        self.engine.record_start()

    def _on_record_stop(self) -> None:
        self.engine.record_stop()

    def _on_monitor_toggle(self) -> None:
        self.engine.set_param("monitor", self.monitor_var.get())

    def _on_play(self) -> None:
        # Replay the live processed take exactly as it was captured -- no
        # re-render, no params re-applied.
        try:
            self.engine.play("live")
        except Exception as exc:
            self.error_message = f"Play failed: {exc}"

    def _on_rerender_play(self) -> None:
        try:
            self.engine.render_current()
            self.engine.play("rendered")
        except Exception as exc:
            self.error_message = f"Re-render/play failed: {exc}"

    # -- status polling ------------------------------------------------

    def _refresh_status(self) -> None:
        status = self.engine.get_status()
        snapshot = self.engine.snapshot()

        # Reflect any out-of-band (API-driven) param changes in the
        # widgets, without re-triggering their commands and without
        # yanking a slider the user is currently dragging.
        self._suppress_commands = True
        try:
            for name, widget in self._param_widgets.items():
                if name not in snapshot:
                    continue
                value = snapshot[name]
                if widget["kind"] == "bool":
                    if bool(widget["var"].get()) != bool(value):
                        widget["var"].set(bool(value))
                else:
                    if self._dragging.get(name):
                        continue
                    try:
                        changed = abs(float(widget["var"].get()) - float(value)) > 1e-9
                    except (TypeError, ValueError):
                        changed = True
                    if changed:
                        widget["var"].set(value)
                        widget["readout"].config(text=self._format_value(widget["spec"], value))
            # Keep the dedicated Live monitor checkbox in sync with
            # API-driven changes (it's not in the generated grid).
            if bool(self.monitor_var.get()) != bool(status.get("monitor", False)):
                self.monitor_var.set(bool(status.get("monitor", False)))
        finally:
            self._suppress_commands = False

        level = status.get("level", {})
        error_suffix = f"\nError: {self.error_message}" if self.error_message else ""
        take_state = "recording" if status.get("recording") else "stopped"
        monitor_state = "on" if status.get("monitor") else "off"
        text = (
            f"Running: {status['running']}   Enabled: {status['enabled']}   "
            f"Monitor: {monitor_state}   Latency: {status.get('latency_ms', 0.0):.0f}ms\n"
            f"Underruns: {status['underruns']}   Overruns: {status['overruns']}\n"
            f"Level  dry: {level.get('dry_rms', 0.0):.3f} rms / {level.get('dry_peak', 0.0):.3f} pk   "
            f"out: {level.get('processed_rms', 0.0):.3f} rms / {level.get('processed_peak', 0.0):.3f} pk\n"
            f"Take: {take_state}, {status.get('take_seconds', 0.0):.1f}s   "
            f"live={'yes' if status.get('has_live_take') else 'no'}   "
            f"raw={'yes' if status.get('has_raw_take') else 'no'}   "
            f"rendered={'yes' if status.get('has_rendered_take') else 'no'}"
            f"{error_suffix}"
        )
        self.status_label.config(text=text)
        self.root.after(STATUS_REFRESH_MS, self._refresh_status)

    # -- shutdown ------------------------------------------------------

    def _on_close(self) -> None:
        try:
            if self.control_server is not None:
                self.control_server.stop()
            if self.engine.running:
                self.engine.stop()
        finally:
            self.root.destroy()


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
