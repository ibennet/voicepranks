"""Streaming Schroeder/Freeverb-style reverb.

A lighter mono take on Freeverb: a bank of parallel feedback comb filters
(each with a one-pole lowpass in its feedback loop for high-frequency
"damping") feeding a short series of allpass filters for diffusion, then a
wet/dry mix. Stacked under a downward pitch shift + grit it gives the big,
dark, cavernous space that reads as "scary".

Streaming contract (matches the other DSP blocks):
  * `process()` accepts a mono float block of any length (including 0) and
    returns the *same number* of samples -- the reverb is a feedback network
    whose output is time-aligned with its input, so it adds NO algorithmic
    front latency (the engine's output pre-fill needs no change).
  * All delay-line buffers and indices persist across calls, so output is
    continuous and click-free regardless of how the input is chunked.
  * `reset()` zeroes every buffer.

Implementation note: the block is processed in internal sub-chunks no longer
than the shortest delay line. Within such a sub-chunk every delay tap reads
only already-written history, so the comb/allpass delay reads and writes
vectorize cleanly in numpy. The only genuinely recursive part is each comb's
one-pole damping filter, kept as a tight scalar loop (as in `biquad.py`); it
decays fast (damp coefficient <= 0.4) and is cheap.
"""
from __future__ import annotations

import numpy as np

# Freeverb's classic delay tunings (in samples at 44.1kHz): the full bank of
# 8 combs + 4 allpass gives a dense, smooth tail (fewer discrete "echoes" than
# a smaller bank). Lengths are rescaled to the actual sample rate in __init__.
_COMB_TUNING = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS_TUNING = (556, 441, 341, 225)
_REF_SR = 44100

_ALLPASS_FEEDBACK = 0.5
_FIXED_GAIN = 0.015  # input scale into the comb bank (Freeverb's 8-comb value)
_SCALE_ROOM = 0.28
_OFFSET_ROOM = 0.7  # room_size 0..1 -> comb feedback 0.70..0.98
_SCALE_DAMP = 0.4   # damp 0..1 -> lowpass coefficient 0.0..0.4


class Reverb:
    """Streaming feedback-comb + allpass reverb with wet/dry mix."""

    def __init__(
        self,
        sample_rate: int,
        mix: float = 0.0,
        room_size: float = 0.5,
        damp: float = 0.5,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.mix = min(max(float(mix), 0.0), 1.0)
        self.room_size = min(max(float(room_size), 0.0), 1.0)
        self.damp = min(max(float(damp), 0.0), 1.0)

        scale = self.sample_rate / _REF_SR
        self._comb_len = [max(1, int(round(t * scale))) for t in _COMB_TUNING]
        self._allpass_len = [max(1, int(round(t * scale))) for t in _ALLPASS_TUNING]
        # Longest sub-chunk for which every delay read is pure history.
        self._min_delay = min(self._comb_len + self._allpass_len)

        self._init_state()
        self._update_coeffs()

    # -- configuration ---------------------------------------------------

    def set_mix(self, mix: float) -> None:
        """Wet/dry balance, 0 (dry) .. 1 (fully wet)."""
        self.mix = min(max(float(mix), 0.0), 1.0)

    def set_room_size(self, room_size: float) -> None:
        """Tail length / size, 0 (small) .. 1 (huge). Sets comb feedback."""
        self.room_size = min(max(float(room_size), 0.0), 1.0)
        self._update_coeffs()

    def set_damp(self, damp: float) -> None:
        """High-frequency damping in the tail, 0 (bright) .. 1 (dark)."""
        self.damp = min(max(float(damp), 0.0), 1.0)
        self._update_coeffs()

    def reset(self) -> None:
        self._init_state()

    def _init_state(self) -> None:
        self._comb_buf = [np.zeros(n, dtype=np.float64) for n in self._comb_len]
        self._comb_idx = [0 for _ in self._comb_len]
        self._comb_store = [0.0 for _ in self._comb_len]  # one-pole lowpass state
        self._ap_buf = [np.zeros(n, dtype=np.float64) for n in self._allpass_len]
        self._ap_idx = [0 for _ in self._allpass_len]

    def _update_coeffs(self) -> None:
        self._feedback = self.room_size * _SCALE_ROOM + _OFFSET_ROOM
        self._damp1 = self.damp * _SCALE_DAMP
        self._damp2 = 1.0 - self._damp1

    # -- processing ------------------------------------------------------

    def process(self, mono: np.ndarray) -> np.ndarray:
        x = np.asarray(mono, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        n_total = x.shape[0]
        out = np.empty(n_total, dtype=np.float64)
        step = self._min_delay
        pos = 0
        while pos < n_total:
            n = min(step, n_total - pos)
            out[pos:pos + n] = self._process_chunk(x[pos:pos + n], n)
            pos += n

        result = out.astype(np.float32)
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _process_chunk(self, chunk: np.ndarray, n: int) -> np.ndarray:
        """Process one sub-chunk (n <= shortest delay line, so every delay
        read is pure history and vectorizes)."""
        feedback = self._feedback
        damp1, damp2 = self._damp1, self._damp2
        offsets = np.arange(n)

        comb_in = chunk * _FIXED_GAIN
        wet = np.zeros(n, dtype=np.float64)

        # Parallel feedback comb bank.
        for c in range(len(self._comb_len)):
            buf = self._comb_buf[c]
            L = self._comb_len[c]
            idx = self._comb_idx[c]
            read_idx = (idx + offsets) % L

            y = buf[read_idx]  # comb output = delayed sample (history)
            wet += y

            # One-pole lowpass in the feedback path (the only recursion).
            fs = np.empty(n, dtype=np.float64)
            store = self._comb_store[c]
            for k in range(n):
                store = y[k] * damp2 + store * damp1
                fs[k] = store
            self._comb_store[c] = store

            buf[read_idx] = comb_in + fs * feedback
            self._comb_idx[c] = (idx + n) % L

        # Series allpass diffusers.
        for a in range(len(self._allpass_len)):
            buf = self._ap_buf[a]
            L = self._allpass_len[a]
            idx = self._ap_idx[a]
            read_idx = (idx + offsets) % L

            bufout = buf[read_idx]
            output = -wet + bufout
            buf[read_idx] = wet + bufout * _ALLPASS_FEEDBACK
            self._ap_idx[a] = (idx + n) % L
            wet = output

        mix = self.mix
        return chunk * (1.0 - mix) + wet * mix
