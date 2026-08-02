"""Streaming "Minionese" gibberish voice effect (time-domain shuffle).

An alternative gibberish engine to the STFT formant-replacement one in
`dsp/minionese.py`. That approach rebuilt the spectral envelope frame by
frame while keeping the original phase, and the resulting magnitude/phase
mismatch produced constant broadband static -- the same class of
"airy/phasy" artifact that got the phase-vocoder pitch shifter rejected in
`dsp/pitch.py`. This engine stays entirely in the time domain and is built
on top of the clean WSOLA pitch shifter, so it inherits that path's clean
tone instead of fighting it.

The chain, all streaming and click-free:

1. Pitch-up: the WSOLA `PitchShifter` from `dsp/pitch.py`. Formants ride up
   with pitch (the "chipmunk"/minion timbre).

2. Pitch wobble (`_PitchWobble`): a modulated fractional-delay read whose
   delay is driven by a sum of a few slow, incommensurate LFOs. The
   derivative of that wandering delay is a wandering pitch, giving the
   sing-song minion melody. Because it is fractional-delay interpolation of
   an already-clean signal, it adds no spectral artifacts.

3. Chunk shuffle (`_ChunkShuffle`): the part that actually scrambles the
   words. The stream is chopped into syllable-sized chunks (`chunk_ms`) and,
   in windows of `shuffle_k` chunks, the chunk *order* is randomly permuted
   before playback. Reordering syllable-sized pieces makes the speech come
   out as new nonsense sequences ("bello / poka / tank yu"-style babble)
   while each chunk keeps its clean local timbre. Chunks are stitched with
   an equal-power crossfade (`fade_ms`) so reordered boundaries don't click,
   and each chunk is played exactly once per window so there is no net time
   drift.

`set_intensity(t)` scales the wobble depth and the scramble jitter, so the
effect ramps in smoothly (t=0 -> a clean pitched-up voice in original order;
t=1 -> full wobble + scramble).

Latency is roughly the pitch pipeline plus a shuffle window of grain
lookback -- fine for a voiceprank, not a phone call.
"""
from __future__ import annotations

import numpy as np

from .pitch import PitchShifter

# -- params (keep these easy to find/tweak) --------------------------------

SEMITONES = 5.0        # base pitch-up amount

# Pitch wobble (the wandering minion melody).
WOBBLE_MS = 4.0        # peak delay excursion, ms (bigger -> wider wobble)
WOBBLE_RATES = (       # (Hz, weight) LFOs summed for a non-periodic wander
    (2.7, 1.0),
    (4.3, 0.7),
    (6.1, 0.4),
)

# Chunk shuffle (the word-scrambler): chop the stream into syllable-sized
# chunks and reorder them so they form new nonsense sequences.
CHUNK_MS = 150.0       # chunk (~syllable) length, ms
SHUFFLE_K = 3          # chunks per shuffle window (window = K * CHUNK_MS).
                       # Bigger -> syllables travel further when reordered
                       # (more scrambled) but more latency.
FADE_MS = 25.0         # equal-power crossfade between reordered chunks, ms

# Final safety net (not normally engaged now that the chain is time-domain
# and crossfaded): a causal per-sample slew clamp that only catches gross
# discontinuities. Set high enough not to touch normal bright speech.
MAX_SLEW = 0.25        # max sample-to-sample output change


def _slew_limit(x: np.ndarray, max_step: float, prev: float):
    """Causal slew-rate limiter. Returns (limited, new_prev)."""
    if x.size == 0:
        return x, prev
    y = x.astype(np.float64).copy()
    p = prev
    for i in range(y.shape[0]):
        lo = p - max_step
        hi = p + max_step
        if y[i] > hi:
            y[i] = hi
        elif y[i] < lo:
            y[i] = lo
        p = y[i]
    return y.astype(np.float32), p


class _PitchWobble:
    """Modulated fractional delay: a wandering pitch made by reading an
    input FIFO at a delay that drifts along a sum of slow LFOs.

    Like a vibrato, but with several incommensurate rates summed so the
    pitch wanders instead of oscillating periodically. Everything is delayed
    by a fixed amount just past the LFO's peak excursion and the read point
    moves around that fixed delay, which keeps it causal (a few ms of extra
    latency) and click-free.
    """

    def __init__(self, sample_rate: int, wobble_ms: float = WOBBLE_MS, rates=WOBBLE_RATES) -> None:
        self.sr = float(sample_rate)
        self.depth = float(wobble_ms) / 1000.0 * self.sr       # peak excursion
        self.delay = int(np.ceil(self.depth)) + 4
        self.rates = rates
        self._wsum = float(sum(w for _, w in self.rates)) or 1.0
        self._depth_scale = 1.0
        self.reset()

    def set_depth_scale(self, s: float) -> None:
        self._depth_scale = min(max(float(s), 0.0), 1.0)

    def reset(self) -> None:
        self._n = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._origin = -(self.delay + 4)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        m = x.shape[0]
        if m == 0:
            return np.zeros(0, dtype=np.float32)

        if self._buf.shape[0] == 0:
            self._buf = np.zeros(self.delay + 4, dtype=np.float32)
            self._origin = -(self.delay + 4)

        self._buf = np.concatenate([self._buf, x])
        n0 = self._n
        self._n += m

        idx = n0 + np.arange(m, dtype=np.float64)
        lfo = np.zeros(m, dtype=np.float64)
        for f, w in self.rates:
            lfo += w * np.sin(2.0 * np.pi * f * idx / self.sr)
        lfo /= self._wsum

        offs = self.depth * self._depth_scale * lfo
        read_pos = idx - self.delay + offs
        rel = read_pos - self._origin
        rel = np.clip(rel, 0.0, self._buf.shape[0] - 1.001)
        i0 = rel.astype(np.int64)
        frac = (rel - i0).astype(np.float32)
        out = self._buf[i0] * (1.0 - frac) + self._buf[i0 + 1] * frac

        keep_margin = self.delay + 8
        newest_rel = (self._n - 1) - self._origin
        trim = newest_rel - keep_margin
        if trim > 0:
            self._buf = self._buf[trim:]
            self._origin += trim

        return out.astype(np.float32)


class _ChunkShuffle:
    """Block-shuffle syllable scrambler.

    Buffers `shuffle_k` chunks of `chunk_ms` each, randomly permutes their
    order, and plays them back stitched with an equal-power crossfade. Each
    chunk is used exactly once per window, so the output tracks the input in
    length (no drift). Each emitted chunk reads `fade` extra samples past its
    end for the crossfade tail, and a persistent tail carries the crossfade
    across chunk and window boundaries so nothing clicks -- even though
    neighbours in the output were not neighbours in the input.

    `set_strength(s)` is the probability that any given window is actually
    shuffled (vs. played in order), so intensity 0 -> a clean in-order copy.
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_ms: float = CHUNK_MS,
        shuffle_k: int = SHUFFLE_K,
        fade_ms: float = FADE_MS,
        reverse_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.sr = int(sample_rate)
        self.C = max(2, int(float(chunk_ms) / 1000.0 * self.sr))
        self.K = max(1, int(shuffle_k))
        self.fade = max(0, int(float(fade_ms) / 1000.0 * self.sr))
        self.reverse_prob = min(max(float(reverse_prob), 0.0), 1.0)
        # Equal-power crossfade ramp, 0 -> 1 over `fade` samples. Uses
        # sin (NOT sin^2) so the fade-in and its reverse (a cos) satisfy
        # sin^2 + cos^2 = 1 -> constant *power*. Reordered chunks are
        # uncorrelated, so amplitude-summing (sin^2/cos^2) weights would dip
        # ~3 dB at each crossfade midpoint -> an audible ~6.7 Hz pulsing that
        # sounds like inserted pauses.
        f = self.fade
        self._ramp_up = np.sin(0.5 * np.pi * (np.arange(f) + 0.5) / f) if f > 0 else np.zeros(0)
        self._seed = seed
        self._strength = 1.0
        self.reset()

    def set_strength(self, s: float) -> None:
        self._strength = min(max(float(s), 0.0), 1.0)

    def set_reverse_prob(self, p: float) -> None:
        self.reverse_prob = min(max(float(p), 0.0), 1.0)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._in = np.zeros(0, dtype=np.float32)         # input FIFO
        self._tail = np.zeros(self.fade, dtype=np.float64)  # crossfade carry

    def _faded(self, g: np.ndarray) -> np.ndarray:
        """Window a (C + fade)-long chunk with fade-in/out at its edges."""
        g = g.astype(np.float64).copy()
        f = self.fade
        if f > 0 and g.shape[0] >= 2 * f:
            g[:f] *= self._ramp_up
            g[-f:] *= self._ramp_up[::-1]
        return g

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        self._in = np.concatenate([self._in, x])
        C, K, f = self.C, self.K, self.fade
        window = K * C
        need = window + f  # +f lookahead for the last chunk's crossfade tail

        out_chunks = []
        while self._in.shape[0] >= need:
            if self._rng.random() < self._strength:
                perm = self._rng.permutation(K)
            else:
                perm = np.arange(K)

            for p in perm:
                seg = self._in[p * C: p * C + C + f]
                # Optionally time-reverse the whole grabbed segment. Reversal
                # turns a chunk into a phonetic non-word (so the output stops
                # sounding like reordered real words) while keeping the local
                # timbre. The equal-power crossfade at every chunk boundary is
                # applied after, so reversed chunks still stitch click-free.
                if self.reverse_prob > 0.0 and self._rng.random() < self.reverse_prob:
                    seg = seg[::-1]
                g = self._faded(seg)
                if f > 0:
                    g[:f] += self._tail            # crossfade with prior tail
                    out_chunks.append(g[:C].astype(np.float32))
                    self._tail = g[C: C + f].copy()  # carry tail forward
                else:
                    out_chunks.append(g[:C].astype(np.float32))

            self._in = self._in[window:]  # consume one window (keep the +f overlap)

        if out_chunks:
            return np.concatenate(out_chunks)
        return np.zeros(0, dtype=np.float32)


class MinioneseShuffle:
    """Streaming time-domain "Minionese" gibberish effect (shuffle engine).

    `process()` may be called with arbitrarily small blocks and returns a
    variable number of samples per call (possibly zero while the pitch and
    grain buffers prime). All stage state persists across calls, so output
    is continuous and click-free.
    """

    def __init__(self, sample_rate: int, seed: int = 0) -> None:
        self.sample_rate = int(sample_rate)
        self._seed = seed

        # Instance copies of the module-level defaults so a single instance
        # can be tuned live without touching the module constants.
        self.semitones = SEMITONES
        self.wobble_ms = WOBBLE_MS
        self.chunk_ms = CHUNK_MS
        self.shuffle_k = SHUFFLE_K
        self.fade_ms = FADE_MS
        self.reverse_prob = 0.0
        self.max_slew = MAX_SLEW

        self._pitch = PitchShifter(self.sample_rate)
        self._pitch.set_semitones(self.semitones)
        self._wobble = _PitchWobble(self.sample_rate, self.wobble_ms)
        self._scramble = _ChunkShuffle(
            self.sample_rate, self.chunk_ms, self.shuffle_k, self.fade_ms,
            reverse_prob=self.reverse_prob, seed=seed,
        )

        self.intensity = 1.0
        self._slew_prev = 0.0

    # -- configuration -----------------------------------------------

    def set_intensity(self, t: float) -> None:
        self.intensity = min(max(float(t), 0.0), 1.0)
        self._wobble.set_depth_scale(self.intensity)
        self._scramble.set_strength(self.intensity)

    def set_semitones(self, semitones: float) -> None:
        """Live-adjustable pitch-up amount; no rebuild needed."""
        self.semitones = float(semitones)
        self._pitch.set_semitones(self.semitones)

    def set_wobble_ms(self, wobble_ms: float) -> None:
        """Pitch-wobble excursion (ms). Rebuilds the wobble delay line
        (small reset, inaudible as a click)."""
        self.wobble_ms = float(wobble_ms)
        self._wobble = _PitchWobble(self.sample_rate, self.wobble_ms)
        self._wobble.set_depth_scale(self.intensity)

    def _rebuild_scramble(self) -> None:
        self._scramble = _ChunkShuffle(
            self.sample_rate, self.chunk_ms, self.shuffle_k, self.fade_ms,
            reverse_prob=self.reverse_prob, seed=self._seed,
        )
        self._scramble.set_strength(self.intensity)

    def set_reverse_prob(self, p: float) -> None:
        """Probability each emitted chunk is time-reversed (0..1). The main
        lever against 'sounds like reordered real words' -- reversed chunks
        are phonetic non-words. No rebuild needed."""
        self.reverse_prob = min(max(float(p), 0.0), 1.0)
        self._scramble.set_reverse_prob(self.reverse_prob)

    def set_chunk_ms(self, chunk_ms: float) -> None:
        """Syllable-chunk length (ms). Rebuilds the scrambler (reset)."""
        self.chunk_ms = float(chunk_ms)
        self._rebuild_scramble()

    def set_shuffle_k(self, k: int) -> None:
        """Chunks per shuffle window (bigger -> more scrambled, more
        latency). Rebuilds the scrambler (reset)."""
        self.shuffle_k = max(1, int(k))
        self._rebuild_scramble()

    def set_fade_ms(self, fade_ms: float) -> None:
        """Equal-power crossfade between reordered chunks (ms). Rebuilds
        the scrambler (reset)."""
        self.fade_ms = float(fade_ms)
        self._rebuild_scramble()

    def set_max_slew(self, max_slew: float) -> None:
        """Output slew-rate ceiling; no rebuild needed."""
        self.max_slew = float(max_slew)

    def latency_ms(self) -> float:
        """Algorithmic buffering latency, in ms. The scrambler must
        accumulate a full shuffle window (`shuffle_k * chunk_ms`) before it
        can emit any output, plus the crossfade lookahead and the WSOLA
        pitch frame. The engine sizes its output pre-fill from this so the
        live path doesn't starve while the window fills."""
        window_ms = (self._scramble.K * self._scramble.C) / float(self.sample_rate) * 1000.0
        fade_ms = self._scramble.fade / float(self.sample_rate) * 1000.0
        pitch_ms = self._pitch.L / float(self.sample_rate) * 1000.0
        return window_ms + fade_ms + pitch_ms

    def reset(self) -> None:
        self._pitch.reset()
        self._wobble.reset()
        self._scramble.reset()
        self._slew_prev = 0.0

    # -- processing --------------------------------------------------

    def process(self, mono: np.ndarray) -> np.ndarray:
        block = np.asarray(mono, dtype=np.float32)
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)

        pitched = self._pitch.process(block)
        wobbled = self._wobble.process(pitched)
        out = self._scramble.process(wobbled)

        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        out, self._slew_prev = _slew_limit(out, self.max_slew, self._slew_prev)
        return out
