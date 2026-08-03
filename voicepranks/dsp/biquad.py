"""RBJ Audio EQ Cookbook peaking (bell) filter, stateful across blocks."""
from __future__ import annotations

import math

import numpy as np


class PeakingEQ:
    """Stateful biquad peaking EQ (RBJ cookbook formulas)."""

    def __init__(
        self,
        sample_rate: int,
        center_hz: float = 2000.0,
        q: float = 1.0,
        gain_db: float = 0.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.center_hz = float(center_hz)
        self.q = float(q)
        self.gain_db = float(gain_db)

        # Direct Form II transposed state.
        self.z1 = 0.0
        self.z2 = 0.0

        self._compute_coeffs()

    def set_gain_db(self, gain_db: float) -> None:
        self.gain_db = float(gain_db)
        self._compute_coeffs()

    def set_center_hz(self, center_hz: float) -> None:
        self.center_hz = float(center_hz)
        self._compute_coeffs()

    def set_q(self, q: float) -> None:
        self.q = float(q)
        self._compute_coeffs()

    def reset(self) -> None:
        self.z1 = 0.0
        self.z2 = 0.0

    def _compute_coeffs(self) -> None:
        a = 10.0 ** (self.gain_db / 40.0)
        w0 = 2.0 * math.pi * self.center_hz / self.sample_rate
        # Clamp to valid digital range in case of a pathological center_hz.
        w0 = min(max(w0, 1e-6), math.pi - 1e-6)
        alpha = math.sin(w0) / (2.0 * self.q)
        cos_w0 = math.cos(w0)

        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def process(self, mono: np.ndarray) -> np.ndarray:
        x = np.asarray(mono, dtype=np.float64)
        out = np.empty_like(x)

        b0, b1, b2 = self.b0, self.b1, self.b2
        a1, a2 = self.a1, self.a2
        z1, z2 = self.z1, self.z2

        # Direct Form II transposed recursion. Cheap enough per-sample for
        # a single biquad; kept as a tight Python loop for clarity.
        for i in range(x.shape[0]):
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            out[i] = yi

        self.z1 = z1
        self.z2 = z2

        result = out.astype(np.float32)
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return result
