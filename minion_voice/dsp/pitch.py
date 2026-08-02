"""Streaming pitch shifter using WSOLA (waveform-similarity overlap-add)
time-stretch followed by linear resampling.

Why this over granular / phase-vocoder: by ear, granular pitch-shifting
(the previous implementation here, a two-tap cross-fading delay line)
sounded robotic/echoey at the amounts of shift needed for the Minion
effect, and phase-vocoder approaches sounded airy/tunnel-y (phasiness from
the STFT resynthesis). WSOLA works directly on the waveform: it stretches
the signal in time while preserving pitch by re-using near-identical
looking grains (found via a windowed cross-correlation search, so seams
land on matching waveform cycles instead of arbitrary sample boundaries),
then plays the stretched signal back faster/slower via resampling to
restore the original duration while shifting pitch. This gives a natural,
formant-scaled "chipmunk" character that reads correctly to the ear.

Everything below is a **streaming** re-implementation of this well-known
offline algorithm:

    def wsola(x, S, L=1024, tol=256):
        Hs=L//2; Ha=max(1,int(round(Hs/S))); win=np.hanning(L)
        out=np.zeros(int(len(x)*S)+2*L); ow=np.zeros_like(out)
        syn=0; m=0; target=x[:L].copy()
        while syn+L<len(out):
            center=m*Ha
            if center+L+tol>=len(x): break
            lo=max(0,center-tol); hi=min(len(x)-L,center+tol)
            corr=np.correlate(x[lo:hi+L], target, mode='valid')
            best=lo+int(np.argmax(corr))
            out[syn:syn+L]+=x[best:best+L]*win; ow[syn:syn+L]+=win
            ts=best+Hs; target=x[ts:ts+L] if ts+L<=len(x) else pad(x[ts:])
            syn+=Hs; m+=1
        return (out/np.maximum(ow,1e-6))[:int(len(x)*S)]

    def wsola_pitch(x, ratio):
        st=wsola(x, ratio)                          # stretch longer...
        idx=np.clip(np.arange(len(x))*ratio,0,len(st)-2)
        ...linear-interpolate st at idx...            # ...then decimate
        return result

Two pieces of persistent state make this streaming:

1. **WSOLA time-stretch**, running against a small input FIFO. The
   synthesis hop `Hs = L // 2` never changes, so at any moment at most two
   synthesis frames overlap; that means the overlap-add accumulator only
   ever needs to be `L` samples long. Each time a new frame is placed, the
   first `Hs` samples of the accumulator are *finished* (no future frame
   can touch them), so we immediately normalize and flush them into a
   "stretched-domain" buffer, then shift the accumulator left by `Hs`.
   This produces byte-for-byte the same values the offline reference
   would produce for the same region, just incrementally instead of in
   one batch pass at the very end. The analysis search position
   (`_center`), current search target frame (`_target`), and un-flushed
   overlap-add tail (`_ola`/`_ow`) all persist across `process()` calls,
   so a signal fed in small blocks stretches identically to the same
   signal fed in one shot.

2. **Streaming linear resampler**, which reads the stretched-domain
   buffer at a fractional cursor that advances by `ratio` per output
   sample (vectorized: build all read positions for this call at once,
   gather + interpolate, then carry the leftover fractional cursor
   position to the next call). This is the "decimate" half of
   `wsola_pitch`, done continuously instead of once over a
   known-length buffer.

`ratio = 2 ** (semitones / 12)`. Formants move up together with pitch
here -- that's intentional (desired "chipmunk" character), not preserved.
"""
from __future__ import annotations

import numpy as np

_MIN_RATIO = 1e-3


class PitchShifter:
    """Real-time, block-based WSOLA pitch shifter.

    `process()` may be called with arbitrarily small blocks (as small as a
    single sample) and may return a variable number of samples -- possibly
    zero -- per call. All WSOLA and resampling state is carried across
    calls, so output is continuous (no clicks/discontinuities at block
    boundaries) regardless of how the input is chunked.
    """

    def __init__(self, sample_rate: int, frame: int = 1024, tol: int = 256) -> None:
        self.sample_rate = int(sample_rate)
        self.L = int(frame)
        self.tol = int(tol)
        self.Hs = self.L // 2
        self.win = np.hanning(self.L).astype(np.float64)

        self.ratio = 1.0
        self._init_state()

    # -- configuration ---------------------------------------------------

    def set_ratio(self, ratio: float) -> None:
        self.ratio = max(float(ratio), _MIN_RATIO)

    def set_semitones(self, semitones: float) -> None:
        self.set_ratio(2.0 ** (float(semitones) / 12.0))

    def set_frame(self, frame: int) -> None:
        """Change the WSOLA analysis/synthesis window length. Rebuilds
        the Hann window and resets all WSOLA/resample state."""
        self.L = int(frame)
        self.Hs = self.L // 2
        self.win = np.hanning(self.L).astype(np.float64)
        self._init_state()

    def set_tol(self, tol: int) -> None:
        """Change the WSOLA search tolerance (samples). Resets state so
        the search window doesn't straddle stale buffered geometry."""
        self.tol = int(tol)
        self._init_state()

    def reset(self) -> None:
        self._init_state()

    def _init_state(self) -> None:
        L, Hs = self.L, self.Hs

        # Input FIFO: raw (unshifted) samples awaiting analysis. `_in_base`
        # is the absolute input-sample index that `_in_buf[0]` corresponds
        # to, so we can discard consumed history while still reasoning
        # about positions in one continuous coordinate space.
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._in_base = 0

        # WSOLA analysis/search state.
        self._center = 0
        self._target = np.zeros(L, dtype=np.float32)
        self._have_target = False

        # WSOLA overlap-add accumulator (synthesis side). Only ever needs
        # to hold `L` samples since hop is fixed at `Hs = L // 2` (2x
        # overlap).
        self._ola = np.zeros(L, dtype=np.float64)
        self._ow = np.zeros(L, dtype=np.float64)

        # Stretched-domain samples, finished (normalized) but not yet
        # consumed by the resampler.
        self._stretch_buf = np.zeros(0, dtype=np.float32)
        self._resample_pos = 0.0

    # -- processing --------------------------------------------------

    def process(self, mono: np.ndarray) -> np.ndarray:
        block = np.asarray(mono, dtype=np.float32)
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)

        L, Hs, tol = self.L, self.Hs, self.tol

        self._in_buf = np.concatenate([self._in_buf, block])

        # Bootstrap the very first search target once enough input exists.
        if not self._have_target and self._in_base + self._in_buf.shape[0] >= L:
            self._target = self._in_buf[:L].copy()
            self._have_target = True
            self._center = self._in_base

        self._run_wsola(L, Hs, tol)
        out = self._run_resample()

        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _run_wsola(self, L: int, Hs: int, tol: int) -> None:
        """Advance the WSOLA time-stretch as far as buffered input allows."""
        while self._have_target:
            avail_end = self._in_base + self._in_buf.shape[0]
            center = self._center
            hi_bound = center + tol
            # We need enough input to pick a synthesis frame anywhere in
            # [center-tol, center+tol] *and* to form the next search target
            # (up to Hs samples past the chosen frame), all without
            # zero-padding mid-stream.
            if avail_end < hi_bound + Hs + L:
                break

            ratio = max(self.ratio, _MIN_RATIO)
            Ha = max(1, int(round(Hs / ratio)))

            lo = max(self._in_base, center - tol)
            hi = min(avail_end - L, hi_bound)
            lo_rel = lo - self._in_base
            hi_rel = hi - self._in_base

            if hi_rel > lo_rel:
                seg = self._in_buf[lo_rel: hi_rel + L]
                corr = np.correlate(seg, self._target, mode="valid")
                best_rel = lo_rel + int(np.argmax(corr))
            else:
                best_rel = lo_rel

            frame = self._in_buf[best_rel: best_rel + L]
            if frame.shape[0] < L:
                frame = np.pad(frame, (0, L - frame.shape[0]))
            windowed = frame.astype(np.float64) * self.win

            self._ola += windowed
            self._ow += self.win

            denom = np.maximum(self._ow[:Hs], 1e-6)
            flushed = (self._ola[:Hs] / denom).astype(np.float32)
            self._stretch_buf = np.concatenate([self._stretch_buf, flushed])

            # Shift the OLA accumulator left by one hop.
            self._ola[: L - Hs] = self._ola[Hs:]
            self._ola[L - Hs:] = 0.0
            self._ow[: L - Hs] = self._ow[Hs:]
            self._ow[L - Hs:] = 0.0

            ts_rel = best_rel + Hs
            tgt = self._in_buf[ts_rel: ts_rel + L]
            if tgt.shape[0] < L:
                tgt = np.pad(tgt, (0, L - tgt.shape[0]))
            self._target = tgt.astype(np.float32).copy()

            self._center += Ha

            # Discard input we can no longer need (everything before the
            # next iteration's earliest possible `lo`).
            next_lo = max(self._in_base, self._center - tol)
            trim = next_lo - self._in_base
            if trim > 0:
                self._in_buf = self._in_buf[trim:]
                self._in_base += trim

    def _run_resample(self) -> np.ndarray:
        """Drain as much of the stretched-domain buffer as the current
        fractional read cursor allows, at step size `ratio` (vectorized)."""
        ratio = max(self.ratio, _MIN_RATIO)
        sbuf = self._stretch_buf
        n = sbuf.shape[0]
        pos = self._resample_pos

        if n > 1 and pos + 1.0 < n:
            count = int(np.floor((n - 1 - pos) / ratio)) + 1
        else:
            count = 0

        if count > 0:
            positions = pos + np.arange(count, dtype=np.float64) * ratio
            idx0 = positions.astype(np.int64)
            idx0 = np.clip(idx0, 0, n - 2)
            frac = (positions - idx0).astype(np.float32)
            out = (sbuf[idx0] * (1.0 - frac) + sbuf[idx0 + 1] * frac).astype(np.float32)
            pos = pos + count * ratio
        else:
            out = np.zeros(0, dtype=np.float32)

        consumed = int(pos)
        if consumed > 0:
            self._stretch_buf = sbuf[consumed:]
            pos -= consumed
        self._resample_pos = pos

        return out
