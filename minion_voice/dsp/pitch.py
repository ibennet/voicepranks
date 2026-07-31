"""Streaming pitch shifter using a two-tap cross-fading delay line.

This is the classic time-domain pitch-shifting technique: we keep a rolling
history of recently seen samples and read back from it at a rate that is
slightly faster (to raise pitch) or slower (to lower pitch) than we write to
it. Reading faster than real time means the read pointer would eventually
run off the front of the available history, so once a "read window" fills
up we wrap back around and start again from a fresh point in the buffer.
Wrapping abruptly would click, so instead of a single read tap we keep two
taps, half a window out of phase with each other, and cross-fade between
them with a triangular window that fades each tap out right as it wraps.

Formants move up together with pitch here -- that's intentional (desired
"chipmunk" character), not preserved.
"""
from __future__ import annotations

import numpy as np


class PitchShifter:
    """Real-time, block-based pitch shifter (numpy-vectorized per block)."""

    def __init__(self, sample_rate: int, window_ms: float = 40.0) -> None:
        self.sample_rate = int(sample_rate)
        self.window_ms = float(window_ms)

        self.window = max(4, int(self.sample_rate * self.window_ms / 1000.0))
        # History buffer must be able to serve reads up to `window` samples
        # behind the write pointer at all times.
        self.bufsize = 2 * self.window

        self.ratio = 1.0
        self.phase = 0.0

        self.history = np.zeros(self.bufsize, dtype=np.float32)

    def set_ratio(self, ratio: float) -> None:
        self.ratio = float(ratio)

    def set_semitones(self, semitones: float) -> None:
        self.ratio = 2.0 ** (float(semitones) / 12.0)

    def reset(self) -> None:
        self.phase = 0.0
        self.history[:] = 0.0

    def process(self, mono: np.ndarray) -> np.ndarray:
        block = np.asarray(mono, dtype=np.float32)
        n_samples = block.shape[0]
        if n_samples == 0:
            return block.copy()

        window = self.window
        ratio = self.ratio
        rate = (ratio - 1.0) / window  # phase change per output sample

        # Extended buffer: existing history followed by this block, so that
        # sample index (bufsize + n) in `ext` corresponds to input sample
        # `n` of the current block, and everything before it is genuine
        # look-back history.
        ext = np.concatenate([self.history, block])
        ext_len = ext.shape[0]

        n = np.arange(n_samples, dtype=np.float64)
        phase_n = np.mod(self.phase - rate * n, 1.0)
        phase2_n = np.mod(phase_n + 0.5, 1.0)

        d1 = phase_n * window
        d2 = phase2_n * window

        write_pos = self.bufsize + n
        read_pos1 = write_pos - d1
        read_pos2 = write_pos - d2

        s1 = _interp_gather(ext, read_pos1)
        s2 = _interp_gather(ext, read_pos2)

        w1 = 1.0 - np.abs(2.0 * phase_n - 1.0)
        w2 = 1.0 - np.abs(2.0 * phase2_n - 1.0)

        out = (w1 * s1 + w2 * s2) / (w1 + w2 + 1e-9)
        out = out.astype(np.float32)

        # Advance state for next call.
        self.phase = float(np.mod(self.phase - rate * n_samples, 1.0))
        if n_samples >= self.bufsize:
            self.history = ext[-self.bufsize:].copy()
        else:
            self.history = ext[-self.bufsize:]

        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out


def _interp_gather(ext: np.ndarray, read_pos: np.ndarray) -> np.ndarray:
    """Linearly interpolated gather of `ext` at fractional positions."""
    max_idx = ext.shape[0] - 2
    clipped = np.clip(read_pos, 0.0, max_idx)
    idx0 = np.floor(clipped).astype(np.int64)
    frac = clipped - idx0
    idx1 = idx0 + 1
    return ext[idx0] * (1.0 - frac) + ext[idx1] * frac
