"""Time-based intensity ramp (pure time math, no audio I/O)."""
from __future__ import annotations

import threading
import time


class IntensityRamp:
    """Produces a 0..1 ramp value over `duration_s` seconds after start()."""

    def __init__(self, duration_s: float = 1.2) -> None:
        self.duration_s = float(duration_s)
        self._lock = threading.Lock()
        self._start_time: float | None = None

    def start(self) -> None:
        with self._lock:
            self._start_time = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._start_time = None

    def reset(self) -> None:
        self.stop()

    def current(self) -> float:
        with self._lock:
            start_time = self._start_time
        if start_time is None:
            return 0.0
        if self.duration_s <= 0.0:
            return 1.0
        elapsed = time.monotonic() - start_time
        t = elapsed / self.duration_s
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t
