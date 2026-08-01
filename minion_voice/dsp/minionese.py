"""Streaming "Minionese" gibberish voice effect.

This is a streaming re-implementation of an offline algorithm that was
dialed in by ear (internally called "C1"). The offline version, in full,
is:

    # dry = mono float32 input @ 48000
    def pitchup(x, semis, vib_ms, vib_hz):
        R = 2.0**(semis/12.0); st = wsola(x, R); n = np.arange(len(x))
        idx = np.clip(n*R + (vib_ms/1000*SR)*np.sin(2*np.pi*vib_hz*n/SR), 0, len(st)-2)
        i0 = np.floor(idx).astype(int); f = idx-i0
        return st[i0]*(1-f) + st[i0+1]*f

    N=1024; hop=128; Q=40; win=np.hanning(N); freqs=np.fft.rfftfreq(N,1/SR)
    VOW=[[800,1200,2800],[500,1800,2700],[320,2300,3200],[600,1000,2600],[400,900,2500]]
    NAS=[330,1200,2450]
    ss = lambda z:(lambda q:q*q*(3-2*q))(np.clip(z,0,1))

    def gib():
        p = pitchup(dry, semis, vib_ms, vib_hz)
        SYLL = int(syll_ms/1000*SR); nfr = 1+(len(p)-N)//hop
        rmsf = np.array([np.sqrt(np.mean(p[k*hop:k*hop+N]**2)) for k in range(nfr)])
        kv = int(0.25*SR/hop); vad = np.convolve(rmsf, np.ones(kv)/kv, 'same') > 0.04*rmsf.max()
        speak = lambda t: vad[min(len(vad)-1, max(0, t//hop))]
        onsets=[]; vw=[]; nasf=[]; t=0; prevv=rng.integers(5)
        while t < len(p)-N:
            if not speak(t): t += hop; continue
            v = rng.integers(5)
            while v==prevv: v = rng.integers(5)
            onsets.append(t); vw.append(v); nasf.append(rng.random()<nasal_frac); prevv=v; t += SYLL
        onsets = np.array(onsets)
        def venv(F):
            e = -1.0 + bright*ss((freqs-1500)/2500) - tilt*freqs
            for f in F: e = e + gain*np.exp(-0.5*((freqs-f*scale)/bw)**2)
            return e
        out=np.zeros(len(p)+N); ow=np.zeros(len(p)+N); gsm=1.0; dur=SYLL/SR
        for k in range(nfr):
            i=k*hop; center=i+N//2; j=int(np.searchsorted(onsets,center,'right')-1)
            if j<0: ow[i:i+N]+=win**2; continue
            dt=(center-onsets[j])/SR; u=dt/dur
            if u>1.05: ow[i:i+N]+=win**2; continue
            gate = floor + (1-floor)*ss(u/atk)*ss((1-u)/rel)
            curF = VOW[vw[j]]
            if nasf[j] and dt<0.028:
                mm=ss(dt/0.028); F=[(1-mm)*nf+mm*cf for nf,cf in zip(NAS,curF)]; gate*=(0.7+0.3*mm)
            else:
                gv=ss(dt/0.018); pf=VOW[vw[j-1] if j>0 else vw[j]]; F=[(1-gv)*a+gv*b for a,b in zip(pf,curF)]
            fr=p[i:i+N]*win; X=np.fft.rfft(fr); mag=np.abs(X)+1e-6; ph=np.angle(X); logm=np.log(mag)
            c=np.fft.irfft(logm); c[Q:len(c)-Q]=0; env=np.fft.rfft(c).real; resid=(logm-env)*resid_f
            delta=np.clip((venv(F)+resid)-logm, -3.0, clamp)
            fo=np.fft.irfft(np.exp(logm+delta)*np.exp(1j*ph), N)
            e0=np.sqrt(np.mean(fr**2))+1e-9; e1=np.sqrt(np.mean(fo**2))+1e-9
            gsm=float(np.clip(0.7*gsm+0.3*(e0/e1), 0.3, 3.0)); fo=fo*gsm*gate*win
            out[i:i+N]+=fo; ow[i:i+N]+=win**2
        return out/np.maximum(ow,1e-6)

It pitches the voice up ~5 semitones (with a subtle vibrato wobble), then
per STFT frame (N=1024, hop=128) replaces the spectral envelope (cepstral
split at lifter Q=40) with a synthetic vowel-formant target chosen by a
syllable sequencer, keeping the original excitation residual, and
overlap-adds the result. The sequencer places syllables on a continuous
grid while the speaker is voiced (VAD-gated), picking random
non-repeating vowels with occasional nasal-murmur onsets.

Streaming structure (see `Minionese` docstring below for the persistent
state that makes each stage continuous across `process()` calls):

1. Pitch-up: the existing streaming `PitchShifter` (WSOLA) from
   `dsp/pitch.py`, at +5 semitones.
2. Vibrato: a small streaming modulated fractional delay line applied
   after the pitch shifter (see `_VibratoDelay`).
3. STFT overlap-add: buffered analysis/synthesis at hop=128 over an
   N=1024 window, flushed incrementally the same way `PitchShifter`
   flushes its own WSOLA overlap-add (see module docstring in
   `dsp/pitch.py`) -- an N-sample accumulator is enough since frames are
   placed in increasing order at a fixed hop, so once a frame lands, the
   oldest `hop` samples of the accumulator can never be touched again and
   are flushed immediately.
4. Sequencer: causal port of the offline while-loop -- persistent
   "next check time", current/previous vowel, nasal flag, and current
   onset time, advanced in lockstep with the STFT frame loop.
5. VAD: causal replacement for the offline algorithm's centered
   box-smoothing + *global* peak-normalized threshold (both of which are
   non-causal / need the whole signal up front). Streaming version uses a
   causal ~250ms moving average of per-frame RMS, gated against an
   adaptive threshold derived from a slow-decay running peak of that
   smoothed RMS (plus a small absolute floor so long silence doesn't
   self-trigger on noise once the peak has decayed close to zero).

Known perceptual divergences from the offline reference (see module
docstring for detail as each piece is implemented below):
  - Onset/gate timing is computed from each STFT frame's *start* sample
    rather than its center (offline uses `center = i + N//2`). Avoiding
    the center would otherwise require the sequencer to look up to
    `N//2` samples of VAD state ahead of the frame currently being
    synthesized. Using the frame start instead keeps the sequencer fully
    causal with zero lookahead, at the cost of a static ~10.7ms
    (`N//2` samples) shift in exactly when syllable transitions/gating
    land relative to the offline reference -- inaudible on its own.
  - VAD is causal and adaptive (running peak + floor) instead of
    offline's non-causal centered smoothing + whole-signal global peak.
    This means the streaming VAD's threshold adapts over time rather
    than being fixed by the loudest moment in the whole clip, and there
    is a small amount of the same latency as (2) baked into how quickly
    it reacts.
"""
from __future__ import annotations

import numpy as np

from .pitch import PitchShifter

# -- C1 params (dialed in by ear; keep these easy to find/tweak) -----------

SEMITONES = 5.0        # pitchup amount
VIB_MS = 1.2           # vibrato depth, ms
VIB_HZ = 5.5           # vibrato rate, Hz

SCALE = 1.22           # formant frequency scale
GAIN = 1.4             # formant bump gain (log-magnitude domain)
BW = 215.0             # formant bump bandwidth, Hz
BRIGHT = 0.3           # high-frequency brightness tilt-up amount
TILT = 0.00025         # linear spectral tilt (per Hz)

FLOOR = 0.55           # syllable envelope gate floor (0..1)
ATK = 0.30             # syllable attack fraction (of syllable duration)
REL = 0.30             # syllable release fraction (of syllable duration)
NASAL_FRAC = 0.15      # probability a syllable onset is a nasal murmur
CLAMP = 1.1            # max upward log-magnitude delta
SYLL_MS = 150.0        # syllable duration, ms
RESID_F = 0.7          # excitation residual mix

N_FFT = 1024
HOP = 128
Q_LIFTER = 40

VOW = [
    [800, 1200, 2800],   # a
    [500, 1800, 2700],   # e
    [320, 2300, 3200],   # i
    [600, 1000, 2600],   # o
    [400, 900, 2500],    # u
]
NAS = [330, 1200, 2450]  # nasal murmur formants

VAD_SMOOTH_MS = 250.0    # causal moving-average window for VAD RMS
VAD_REL_THRESH = 0.04    # threshold as a fraction of the adaptive peak
VAD_PEAK_DECAY = 0.999   # per-STFT-frame decay of the running peak
VAD_ABS_FLOOR = 0.003    # absolute RMS floor so decayed-to-zero peaks
                          # don't self-trigger on noise during silence

# Safety net, NOT part of the offline C1 spec: the `gsm` auto-gain-match
# (see `_process_frame`) is a slow IIR-smoothed ratio with a hard 3.0x
# ceiling. Whenever the envelope-replaced spectrum runs quiet relative to
# the input for several frames in a row (common with these params -- the
# offline reference itself spends over a third of its frames pinned at
# the same 3.0x ceiling on the reference test clip), `gsm` saturates at
# 3.0x and *stays* there across several frames. If the resonance-boosted
# signal then suddenly gets louder while gsm is still pinned at max, the
# result is briefly amplified ~3x more than it should be -- an audible
# pop. This is inherent to the C1 params on real speech, not a streaming
# bug (confirmed by replicating the offline algorithm verbatim and
# observing the same ceiling-pinning behavior on the same input); the
# *specific* onset/vowel draws differ between the causal streaming
# sequencer and the offline batch one (different VAD timing shifts which
# RNG draws land where), so how bad any single overshoot gets varies
# per-run. A causal per-sample slew-rate limiter keeps worst-case jumps
# within the range the offline reference itself exhibits, without
# audibly touching normal-level output.
MAX_SLEW = 0.09           # max sample-to-sample output change


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


def _smootherstep(z):
    q = np.clip(z, 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q)


class _VibratoDelay:
    """Streaming modulated fractional delay: the "rubber" pitch wobble from
    the offline `pitchup()`, applied as a separate stage after the WSOLA
    pitch shift.

    The offline version reads `st[idx]` where `idx = n*R + wobble` directly
    into a fully-known array -- effectively non-causal by up to the
    vibrato's peak excursion (a few dozen samples). To make this
    realizable in real time, everything is instead delayed by a fixed
    amount just past the vibrato's peak excursion, and the read point
    wobbles around that fixed delay. This adds an inaudible ~1.2ms of
    extra latency and reproduces the same audible wobble character.
    """

    def __init__(self, sample_rate: int, vib_ms: float, vib_hz: float) -> None:
        self.sr = float(sample_rate)
        self.vib_ms = float(vib_ms)
        self.vib_hz = float(vib_hz)
        self.max_offset = self.vib_ms / 1000.0 * self.sr
        self.delay = int(np.ceil(self.max_offset)) + 4
        self._n = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._origin = -(self.delay + 4)

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
            # Prime with silence so the very first real samples can be
            # read through the delay/modulation immediately.
            self._buf = np.zeros(self.delay + 4, dtype=np.float32)
            self._origin = -(self.delay + 4)

        self._buf = np.concatenate([self._buf, x])
        n0 = self._n
        self._n += m

        idx = n0 + np.arange(m, dtype=np.float64)
        offs = self.max_offset * np.sin(2.0 * np.pi * self.vib_hz * idx / self.sr)
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


class Minionese:
    """Streaming "Minionese" gibberish effect.

    `process()` may be called with arbitrarily small blocks (spec'd around
    ~256 samples) and returns a variable number of samples per call
    (possibly zero, notably while the internal N=1024 analysis buffer
    fills on the first few calls). All pitch/vibrato/STFT/sequencer/VAD
    state persists across calls so output is continuous and click-free.
    """

    def __init__(self, sample_rate: int, seed: int = 0) -> None:
        self.sample_rate = int(sample_rate)
        self.N = N_FFT
        self.hop = HOP
        self.win = np.hanning(self.N).astype(np.float64)
        self.freqs = np.fft.rfftfreq(self.N, 1.0 / self.sample_rate)

        # Instance copies of the module-level "C1" defaults so a single
        # Minionese instance can be tuned live without touching the
        # module constants (kept as defaults read here at construction).
        self.semitones = SEMITONES
        self.wobble_ms = VIB_MS
        self.chunk_ms = SYLL_MS
        self.shuffle_k = 5  # number of vowel candidates (len(VOW))
        self.fade_ms = ATK * SYLL_MS  # syllable attack/release time, ms
        self.max_slew = MAX_SLEW
        self.floor = FLOOR          # inter-syllable gate floor (0..1)
        self.resid_f = RESID_F      # original-excitation residual mix
        self.vad_thresh = VAD_REL_THRESH  # VAD gate: fraction of adaptive peak

        self.SYLL = max(1, int(self.chunk_ms / 1000.0 * self.sample_rate))
        self.dur = self.SYLL / self.sample_rate
        self._vad_kv = max(1, int(VAD_SMOOTH_MS / 1000.0 * self.sample_rate / self.hop))

        self._seed = seed
        self._pitch = PitchShifter(self.sample_rate)
        self._pitch.set_semitones(self.semitones)
        self._vibrato = _VibratoDelay(self.sample_rate, self.wobble_ms, VIB_HZ)

        self.intensity = 1.0

        self._init_state()

    # -- configuration -----------------------------------------------

    def set_intensity(self, t: float) -> None:
        self.intensity = min(max(float(t), 0.0), 1.0)

    def set_semitones(self, semitones: float) -> None:
        """Live-adjustable pitch-up amount; no rebuild needed."""
        self.semitones = float(semitones)
        self._pitch.set_semitones(self.semitones)

    def set_wobble_ms(self, wobble_ms: float) -> None:
        """Vibrato depth (ms). Rebuilds the vibrato delay stage (small
        reset -- a few ms of latency change, inaudible as a click)."""
        self.wobble_ms = float(wobble_ms)
        self._vibrato = _VibratoDelay(self.sample_rate, self.wobble_ms, VIB_HZ)

    def set_chunk_ms(self, chunk_ms: float) -> None:
        """Syllable duration (ms). Rebuilds the sequencer's chunking, so
        the whole streaming pipeline state is reset."""
        self.chunk_ms = float(chunk_ms)
        self.SYLL = max(1, int(self.chunk_ms / 1000.0 * self.sample_rate))
        self.dur = self.SYLL / self.sample_rate
        self._init_state()

    def set_shuffle_k(self, k: int) -> None:
        """Number of candidate vowels the sequencer shuffles between.
        Rebuilds sequencer state (reset)."""
        self.shuffle_k = max(2, int(k))
        self._init_state()

    def set_fade_ms(self, fade_ms: float) -> None:
        """Syllable attack/release time (ms), converted internally to the
        `gate` envelope's attack/release fraction of the syllable
        duration. Rebuilds sequencer state (reset) for parity with the
        other syllable-timing params."""
        self.fade_ms = float(fade_ms)
        self._init_state()

    def set_max_slew(self, max_slew: float) -> None:
        """Output slew-rate ceiling; no rebuild needed."""
        self.max_slew = float(max_slew)

    def latency_ms(self) -> float:
        """Algorithmic buffering latency, in ms: the STFT analysis window
        must fill before any frame is emitted, on top of the WSOLA pitch
        frame. Small and fixed (unlike the shuffle engine, whose latency
        grows with its shuffle window)."""
        return (self.N + self._pitch.L) / float(self.sample_rate) * 1000.0

    def set_floor(self, floor: float) -> None:
        """Inter-syllable gate floor (0..1). Lower -> the gate closes
        harder between syllables (quieter gaps, less continuous drone),
        higher -> a more connected babble. No rebuild needed."""
        self.floor = min(max(float(floor), 0.0), 1.0)

    def set_resid_f(self, resid_f: float) -> None:
        """Mix of the *original* excitation residual kept under the
        synthetic vowel formants (0..1). Lower -> the synthetic vowels
        dominate (less of the real speech bleeds through, so it reads as
        babble rather than intelligible words), higher -> more of the
        source timbre/consonants survive. No rebuild needed."""
        self.resid_f = min(max(float(resid_f), 0.0), 1.0)

    def set_vad_thresh(self, vad_thresh: float) -> None:
        """Voice-activity gate threshold as a fraction of the adaptive
        running peak. Higher -> only louder speech triggers syllable
        synthesis (background noise during pauses stays silent), lower ->
        the gate opens more easily. No rebuild needed."""
        self.vad_thresh = max(float(vad_thresh), 0.0)

    def reset(self) -> None:
        self._pitch.reset()
        self._vibrato.reset()
        self._init_state()

    def _init_state(self) -> None:
        self._rng = np.random.default_rng(self._seed)

        # STFT input FIFO (post pitch+vibrato). `_in_base` is the absolute
        # sample index that `_in_buf[0]` corresponds to.
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._in_base = 0
        self._frame_i = 0  # absolute index of the next frame to analyze

        # STFT overlap-add accumulator/normalizer (synthesis side). Only
        # ever needs to hold N samples since hop is fixed.
        self._out_acc = np.zeros(self.N, dtype=np.float64)
        self._out_ow = np.zeros(self.N, dtype=np.float64)

        self._gsm = 1.0

        # Output safety net (see `MAX_SLEW` above).
        self._slew_prev = 0.0

        # Causal VAD state.
        self._vad_ring = np.zeros(self._vad_kv, dtype=np.float64)
        self._vad_ring_pos = 0
        self._vad_sum = 0.0
        self._vad_count = 0
        self._vad_peak = 0.0
        self._vad_hist: list[bool] = []
        self._vad_hist_base = 0

        # Sequencer state.
        self._seq_t = 0
        self._vowel_pool = min(max(int(self.shuffle_k), 2), len(VOW))
        self._seq_prevv = int(self._rng.integers(self._vowel_pool))
        self._have_onset = False
        self._onset_t = 0
        self._vw_cur = 0
        self._vw_prev = 0
        self._nasal_cur = False

    # -- VAD -----------------------------------------------------------

    def _vad_update(self, rms: float) -> bool:
        """Advance the causal VAD by one STFT frame; return speak/no-speak."""
        kv = self._vad_kv
        old = self._vad_ring[self._vad_ring_pos]
        self._vad_ring[self._vad_ring_pos] = rms
        self._vad_ring_pos = (self._vad_ring_pos + 1) % kv
        self._vad_sum += rms - old
        self._vad_count = min(self._vad_count + 1, kv)
        smoothed = self._vad_sum / self._vad_count

        # Peak tracks the *raw* per-frame RMS (matching the offline
        # reference's un-smoothed `rmsf.max()`), not the smoothed value --
        # smoothing damps transient peaks, which would otherwise drag the
        # adaptive threshold down and make the gate stricter than intended.
        self._vad_peak = max(rms, self._vad_peak * VAD_PEAK_DECAY)
        threshold = max(self.vad_thresh * self._vad_peak, VAD_ABS_FLOOR)
        vad = smoothed > threshold

        self._vad_hist.append(bool(vad))
        # Trim old history we'll never query again (sequencer only ever
        # looks back to roughly the current frame, never further).
        if len(self._vad_hist) > 512:
            drop = len(self._vad_hist) - 256
            self._vad_hist = self._vad_hist[drop:]
            self._vad_hist_base += drop

        return vad

    def _speak_at(self, t: int) -> bool:
        k = int(t) // self.hop
        idx = k - self._vad_hist_base
        if idx < 0:
            return False
        if idx >= len(self._vad_hist):
            idx = len(self._vad_hist) - 1
        if idx < 0:
            return False
        return self._vad_hist[idx]

    # -- sequencer -------------------------------------------------------

    def _advance_sequencer(self, up_to: int) -> None:
        """Advance the causal syllable sequencer up to (and including)
        `up_to` (an absolute sample index), mirroring the offline
        while-loop but gated by causal VAD instead of a whole-signal one."""
        while self._seq_t <= up_to:
            if not self._speak_at(self._seq_t):
                self._seq_t += self.hop
                continue
            v = int(self._rng.integers(self._vowel_pool))
            while v == self._seq_prevv:
                v = int(self._rng.integers(self._vowel_pool))
            nasal = bool(self._rng.random() < NASAL_FRAC)

            if not self._have_onset:
                self._vw_prev = v
                self._have_onset = True
            else:
                self._vw_prev = self._seq_prevv

            self._vw_cur = v
            self._nasal_cur = nasal
            self._onset_t = self._seq_t
            self._seq_prevv = v
            self._seq_t += self.SYLL

    # -- spectral envelope -------------------------------------------------

    def _venv(self, formants) -> np.ndarray:
        e = -1.0 + BRIGHT * _smootherstep((self.freqs - 1500.0) / 2500.0) - TILT * self.freqs
        for f in formants:
            e = e + GAIN * np.exp(-0.5 * ((self.freqs - f * SCALE) / BW) ** 2)
        return e

    # -- per-frame processing ---------------------------------------------

    def _process_frame(self, frame32: np.ndarray, i: int) -> None:
        rms = float(np.sqrt(np.mean(frame32.astype(np.float64) ** 2)))
        self._vad_update(rms)
        self._advance_sequencer(i)

        if not self._have_onset:
            self._out_ow[: self.N] += self.win ** 2
            return

        dt = (i - self._onset_t) / self.sample_rate
        u = dt / self.dur
        if u > 1.05:
            self._out_ow[: self.N] += self.win ** 2
            return

        fade_frac = min(max((self.fade_ms / 1000.0) / max(self.dur, 1e-9), 0.01), 0.9)
        gate = self.floor + (1.0 - self.floor) * _smootherstep(u / fade_frac) * _smootherstep((1.0 - u) / fade_frac)
        cur_f = VOW[self._vw_cur]
        if self._nasal_cur and dt < 0.028:
            mm = _smootherstep(dt / 0.028)
            formants = [(1.0 - mm) * nf + mm * cf for nf, cf in zip(NAS, cur_f)]
            gate *= 0.7 + 0.3 * mm
        else:
            gv = _smootherstep(dt / 0.018)
            pf = VOW[self._vw_prev]
            formants = [(1.0 - gv) * a + gv * b for a, b in zip(pf, cur_f)]

        fr = frame32.astype(np.float64) * self.win
        X = np.fft.rfft(fr)
        mag = np.abs(X) + 1e-6
        ph = np.angle(X)
        logm = np.log(mag)

        c = np.fft.irfft(logm)
        c[Q_LIFTER: len(c) - Q_LIFTER] = 0.0
        env = np.fft.rfft(c).real
        resid = (logm - env) * self.resid_f

        delta = np.clip((self._venv(formants) + resid) - logm, -3.0, CLAMP)
        fo = np.fft.irfft(np.exp(logm + delta) * np.exp(1j * ph), self.N)

        e0 = np.sqrt(np.mean(fr ** 2)) + 1e-9
        e1 = np.sqrt(np.mean(fo ** 2)) + 1e-9
        self._gsm = float(np.clip(0.7 * self._gsm + 0.3 * (e0 / e1), 0.3, 3.0))
        fo = fo * self._gsm * gate * self.win

        self._out_acc[: self.N] += fo
        self._out_ow[: self.N] += self.win ** 2

    # -- processing --------------------------------------------------------

    def process(self, mono: np.ndarray) -> np.ndarray:
        block = np.asarray(mono, dtype=np.float32)
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)

        pitched = self._pitch.process(block)
        vibrated = self._vibrato.process(pitched)

        self._in_buf = np.concatenate([self._in_buf, vibrated])

        N, hop = self.N, self.hop
        out_chunks = []

        while True:
            avail_end = self._in_base + self._in_buf.shape[0]
            i = self._frame_i
            if i + N > avail_end:
                break

            start_rel = i - self._in_base
            frame32 = self._in_buf[start_rel: start_rel + N]

            self._process_frame(frame32, i)

            denom = np.maximum(self._out_ow[:hop], 1e-6)
            flushed = (self._out_acc[:hop] / denom).astype(np.float32)
            out_chunks.append(flushed)

            self._out_acc[: N - hop] = self._out_acc[hop:]
            self._out_acc[N - hop:] = 0.0
            self._out_ow[: N - hop] = self._out_ow[hop:]
            self._out_ow[N - hop:] = 0.0

            self._frame_i += hop
            trim = self._frame_i - self._in_base
            if trim > 0:
                self._in_buf = self._in_buf[trim:]
                self._in_base += trim

        if out_chunks:
            out = np.concatenate(out_chunks)
        else:
            out = np.zeros(0, dtype=np.float32)

        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        out, self._slew_prev = _slew_limit(out, self.max_slew, self._slew_prev)
        return out
