"""The "Minion" voice effect: pitch shift only (flat, no EQ).

An EQ presence-boost stage was previously stacked on top of the pitch
shift, but chosen-by-ear testing showed the pitch shift alone reads as
correct/desired, and the EQ boost was not needed (and slightly hurt) once
the pitch shifter itself changed to WSOLA. The EQ stage (`PeakingEQ` in
`biquad.py`) is left intact and wired up (but at 0 dB / unused in the
process chain) so it can be re-enabled later without touching
`engine.py`/`ui/app.py`.

A second, alternate mode -- "Minionese" gibberish -- can be toggled on via
`set_gibberish()`. When enabled, `process()` routes through a gibberish
engine instead of the plain WSOLA pitch shift. There are two gibberish
engines and `set_use_shuffle()` picks between them:
  - `Minionese` (`dsp/minionese.py`): STFT formant-replacement.
  - `MinioneseShuffle` (`dsp/minionese_shuffle.py`): time-domain WSOLA +
    syllable-chunk shuffle (cleaner tone, scrambles real syllables).
"""
from __future__ import annotations

import numpy as np

from .biquad import PeakingEQ
from .minionese import Minionese
from .minionese_shuffle import MinioneseShuffle
from .pitch import PitchShifter


class MinionEffect:
    """Pitch-shift-up voice effect, driven by intensity. Flat (no EQ) by
    default; the EQ stage can be re-enabled live via `set_eq_enabled`."""

    MAX_SEMITONES = 8.5
    MAX_EQ_GAIN_DB = 0.0
    EQ_CENTER_HZ = 2000.0
    EQ_Q = 1.0

    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)

        # Instance copies of the class-level defaults above, so each
        # MinionEffect can be tuned independently without touching the
        # module constants (which existing tests/selftest still rely on
        # as defaults).
        self.max_semitones = self.MAX_SEMITONES
        self.max_eq_gain_db = self.MAX_EQ_GAIN_DB
        self.eq_center_hz = self.EQ_CENTER_HZ
        self.eq_q = self.EQ_Q
        self.eq_enabled = False

        self.pitch = PitchShifter(sample_rate)
        self.eq = PeakingEQ(sample_rate, center_hz=self.eq_center_hz, q=self.eq_q, gain_db=0.0)
        self.minionese = Minionese(sample_rate)
        self.shuffle = MinioneseShuffle(sample_rate)
        self.gibberish = False
        self.use_shuffle = True  # which gibberish engine: shuffle (True) or formant

        self.intensity = 0.0

    def _gibberish_engine(self):
        return self.shuffle if self.use_shuffle else self.minionese

    def set_intensity(self, t: float) -> None:
        t = min(max(float(t), 0.0), 1.0)
        self.intensity = t
        self.pitch.set_semitones(self.max_semitones * t)
        self.eq.set_gain_db(self.max_eq_gain_db * t)
        self.minionese.set_intensity(t)
        self.shuffle.set_intensity(t)

    def set_gibberish(self, b: bool) -> None:
        self.gibberish = bool(b)

    def set_use_shuffle(self, b: bool) -> None:
        """Pick the gibberish engine: True -> time-domain shuffle
        (`MinioneseShuffle`), False -> STFT formant (`Minionese`)."""
        self.use_shuffle = bool(b)

    def set_max_semitones(self, semitones: float) -> None:
        self.max_semitones = float(semitones)
        self.set_intensity(self.intensity)

    def set_eq_enabled(self, b: bool) -> None:
        self.eq_enabled = bool(b)

    def set_eq_gain_db(self, gain_db: float) -> None:
        self.max_eq_gain_db = float(gain_db)
        self.set_intensity(self.intensity)

    def set_eq_center_hz(self, center_hz: float) -> None:
        self.eq_center_hz = float(center_hz)
        self.eq.set_center_hz(self.eq_center_hz)

    def set_eq_q(self, q: float) -> None:
        self.eq_q = float(q)
        self.eq.set_q(self.eq_q)

    def reset(self) -> None:
        self.pitch.reset()
        self.eq.reset()
        self.minionese.reset()
        self.shuffle.reset()

    def process(self, mono: np.ndarray) -> np.ndarray:
        if self.gibberish:
            return self._gibberish_engine().process(mono)
        out = self.pitch.process(mono)
        if self.eq_enabled:
            out = self.eq.process(out)
        return out
