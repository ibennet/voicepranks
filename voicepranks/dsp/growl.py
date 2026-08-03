"""Ring / amplitude modulation -- adds a gnarly, gravelly "growl".

Multiplies the signal by a low-frequency carrier. At a fast carrier rate
(~40-100 Hz) this produces auditory *roughness* -- the buzzy, broken-up
texture the ear reads as a growl/rasp -- plus inharmonic sidebands that give a
metallic, demonic edge. It's the classic monster/demon-voice trick.

`depth` blends between clean and full ring modulation:
  * depth = 0.0  -> modulator is constant 1.0 (signal untouched)
  * depth = 0.5  -> full-depth *amplitude* modulation (unipolar 0..1 carrier):
                    strong roughness, carrier not fully suppressed
  * depth = 1.0  -> true *ring* modulation (bipolar -1..1 carrier): maximum
                    metallic/inharmonic character, carrier fully suppressed

The carrier phase persists across `process()` calls so the modulation is
continuous (no clicks at block boundaries); `reset()` returns it to zero.
"""
from __future__ import annotations

import math

import numpy as np

_TWO_PI = 2.0 * math.pi


class Growl:
    """Phase-continuous ring/amplitude modulator."""

    def __init__(self, sample_rate: int, rate_hz: float = 70.0, depth: float = 0.0) -> None:
        self.sample_rate = int(sample_rate)
        self.rate_hz = max(float(rate_hz), 0.0)
        self.depth = min(max(float(depth), 0.0), 1.0)
        self._phase = 0.0

    def set_rate_hz(self, rate_hz: float) -> None:
        """Carrier frequency (Hz). Higher = buzzier/rougher."""
        self.rate_hz = max(float(rate_hz), 0.0)

    def set_depth(self, depth: float) -> None:
        """0 = clean, 0.5 = full AM roughness, 1 = full ring mod."""
        self.depth = min(max(float(depth), 0.0), 1.0)

    def reset(self) -> None:
        self._phase = 0.0

    def process(self, mono: np.ndarray) -> np.ndarray:
        x = np.asarray(mono, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        n = x.shape[0]
        w = _TWO_PI * self.rate_hz / self.sample_rate
        phases = self._phase + w * np.arange(1, n + 1)
        carrier = np.sin(phases)  # bipolar -1..1

        # depth blends clean (constant 1) with the bipolar carrier.
        mod = (1.0 - self.depth) + self.depth * carrier
        out = (x * mod).astype(np.float32)

        # Carry the phase forward, wrapped to keep it bounded.
        self._phase = float((self._phase + w * n) % _TWO_PI)

        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out
