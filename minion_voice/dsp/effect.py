"""The "Minion" voice effect: pitch shift only (flat, no EQ).

An EQ presence-boost stage was previously stacked on top of the pitch
shift, but chosen-by-ear testing showed the pitch shift alone reads as
correct/desired, and the EQ boost was not needed (and slightly hurt) once
the pitch shifter itself changed to WSOLA. The EQ stage (`PeakingEQ` in
`biquad.py`) is left intact and wired up (but at 0 dB / unused in the
process chain) so it can be re-enabled later without touching
`engine.py`/`ui/app.py`.
"""
from __future__ import annotations

import numpy as np

from .biquad import PeakingEQ
from .pitch import PitchShifter


class MinionEffect:
    """Pitch-shift-up voice effect, driven by intensity. Flat (no EQ)."""

    MAX_SEMITONES = 8.5
    MAX_EQ_GAIN_DB = 0.0
    EQ_CENTER_HZ = 2000.0

    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)

        self.pitch = PitchShifter(sample_rate)
        self.eq = PeakingEQ(sample_rate, center_hz=self.EQ_CENTER_HZ, q=1.0, gain_db=0.0)

        self.intensity = 0.0

    def set_intensity(self, t: float) -> None:
        t = min(max(float(t), 0.0), 1.0)
        self.intensity = t
        self.pitch.set_semitones(self.MAX_SEMITONES * t)
        self.eq.set_gain_db(self.MAX_EQ_GAIN_DB * t)

    def reset(self) -> None:
        self.pitch.reset()
        self.eq.reset()

    def process(self, mono: np.ndarray) -> np.ndarray:
        return self.pitch.process(mono)
