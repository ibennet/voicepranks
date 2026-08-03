"""Waveshaping distortion -- adds harsh, gritty, gravelly harmonics.

A memoryless (stateless) waveshaper. The transfer curve blends a soft knee
(`tanh`) with a *hard* clip and then pushes the sum further into clipping, so
higher `drive` produces a buzzy, aggressive, flat-topped edge (rich odd
harmonics) rather than a gentle warm saturation. Stacked on a downward pitch
shift + ring-mod growl this reads as a harsh, gravelly monster voice.

The curve is odd-symmetric (no DC offset) and the final hard clip keeps the
output bounded in [-1, 1] regardless of drive, so no DC blocker or makeup
stage is needed. `drive == 1.0` is the gentlest setting, so the same knob
doubles as an amount control.

Stateless: `process()` depends only on the current block, so `reset()` is a
no-op and the output is inherently click-free across block boundaries.
"""
from __future__ import annotations

import numpy as np

_MIN_DRIVE = 1.0


class Distortion:
    """Memoryless tanh waveshaper, driven by `drive` (>= 1.0)."""

    def __init__(self, sample_rate: int, drive: float = 1.0) -> None:
        self.sample_rate = int(sample_rate)
        self.drive = max(float(drive), _MIN_DRIVE)

    def set_drive(self, drive: float) -> None:
        """Saturation amount. 1.0 is nearly clean; larger is grittier."""
        self.drive = max(float(drive), _MIN_DRIVE)

    def reset(self) -> None:
        # Memoryless -- nothing to clear.
        pass

    def process(self, mono: np.ndarray) -> np.ndarray:
        x = np.asarray(mono, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        drv = self.drive * x
        # Blend a soft knee (tanh) with a hard clip, then drive the sum
        # further into clipping (*1.4) for a buzzy, aggressive, flat-topped
        # edge. Odd-symmetric (no DC); the final clip bounds output to +/-1.
        soft = np.tanh(drv)
        hard = np.clip(drv, -1.0, 1.0)
        shaped = np.clip((0.5 * soft + 0.5 * hard) * 1.4, -1.0, 1.0)

        out = shaped.astype(np.float32)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out
