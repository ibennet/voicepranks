"""Single source of truth for every tunable parameter in the app.

`PARAM_SPECS` describes the full set of knobs (name, group, UI label,
kind, range, default, whether applying it causes a brief DSP-stage
rebuild/glitch). Both the Tkinter UI (`ui/app.py`) and the HTTP control
server (`control_server.py`) drive the engine exclusively through this
registry (`VoiceEngine.set_param`/`snapshot`) so neither one ever
hardcodes the param list -- add a knob here and it shows up everywhere.

Params are named with a dotted `group.field` convention, e.g.
`"minionese.chunk_ms"`. The `global` group has no dot prefix (`enabled`,
`gibberish`, `intensity`, `ramp.duration_s`) since it isn't tied to one
DSP stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .dsp import minionese as minionese_mod
from .dsp.effect import MinionEffect
from .dsp.pitch import PitchShifter
from .dsp.ramp import IntensityRamp


@dataclass(frozen=True)
class ParamSpec:
    """Describes one adjustable parameter, independent of any live object."""

    name: str
    group: str
    label: str
    kind: str  # "float" | "int" | "bool"
    min: float
    max: float
    step: float
    default: float
    needs_reset: bool = False  # applying it rebuilds a DSP stage (brief glitch)


PARAM_SPECS = [
    # -- global ----------------------------------------------------------
    ParamSpec("enabled", "global", "Enabled", "bool", 0, 1, 1, False),
    ParamSpec("gibberish", "global", "Minionese (gibberish)", "bool", 0, 1, 1, False),
    ParamSpec("intensity", "global", "Intensity (manual)", "float", 0.0, 1.0, 0.01, 1.0),
    ParamSpec("ramp.duration_s", "global", "Ramp duration (s)", "float", 0.0, 5.0, 0.1, IntensityRamp().duration_s),

    # -- plain mode (effect.py, flat pitch shift + optional EQ) ----------
    ParamSpec("effect.max_semitones", "effect", "Max semitones", "float", 0.0, 12.0, 0.1, MinionEffect.MAX_SEMITONES),
    ParamSpec("effect.eq_enabled", "effect", "EQ enabled", "bool", 0, 1, 1, False),
    ParamSpec("effect.eq_gain_db", "effect", "EQ gain (dB)", "float", -12.0, 12.0, 0.5, MinionEffect.MAX_EQ_GAIN_DB),
    ParamSpec("effect.eq_center_hz", "effect", "EQ center (Hz)", "float", 200.0, 6000.0, 50.0, MinionEffect.EQ_CENTER_HZ),
    ParamSpec("effect.eq_q", "effect", "EQ Q", "float", 0.3, 4.0, 0.1, MinionEffect.EQ_Q),

    # -- minionese (gibberish mode) ---------------------------------------
    ParamSpec("minionese.semitones", "minionese", "Semitones", "float", 0.0, 12.0, 0.1, minionese_mod.SEMITONES),
    ParamSpec("minionese.wobble_ms", "minionese", "Wobble depth (ms)", "float", 0.0, 12.0, 0.1, minionese_mod.VIB_MS),
    ParamSpec("minionese.chunk_ms", "minionese", "Syllable length (ms)", "float", 40.0, 400.0, 5.0, minionese_mod.SYLL_MS, needs_reset=True),
    ParamSpec("minionese.shuffle_k", "minionese", "Vowel pool size", "int", 1, 8, 1, 5, needs_reset=True),
    ParamSpec(
        "minionese.fade_ms",
        "minionese",
        "Syllable fade (ms)",
        "float",
        0.0,
        60.0,
        1.0,
        minionese_mod.ATK * minionese_mod.SYLL_MS,
        needs_reset=True,
    ),
    ParamSpec("minionese.max_slew", "minionese", "Max output slew", "float", 0.02, 1.0, 0.01, minionese_mod.MAX_SLEW),

    # -- pitch engine (advanced, shared WSOLA implementation) -------------
    ParamSpec("pitch.frame", "pitch", "WSOLA frame size", "int", 256, 4096, 64, 1024, needs_reset=True),
    ParamSpec("pitch.tol", "pitch", "WSOLA search tolerance", "int", 32, 512, 8, 256, needs_reset=True),

    # -- I/O (apply on (re)start) -----------------------------------------
    ParamSpec("io.sample_rate", "io", "Sample rate (Hz)", "int", 8000, 192000, 1000, 48000, needs_reset=True),
    ParamSpec("io.blocksize", "io", "Block size (samples)", "int", 64, 4096, 64, 256, needs_reset=True),
    ParamSpec("io.ring_ms", "io", "Output ring buffer (ms)", "float", 50.0, 1000.0, 10.0, 200.0, needs_reset=True),
    ParamSpec("io.input_device", "io", "Input device", "int", -1, 999, 1, -1, needs_reset=True),
    ParamSpec("io.output_device", "io", "Output device", "int", -1, 999, 1, -1, needs_reset=True),
]

PARAM_SPECS_BY_NAME: Dict[str, ParamSpec] = {spec.name: spec for spec in PARAM_SPECS}


@dataclass(frozen=True)
class ParamHandlers:
    """Getter/setter closures bound to one live object (engine or effect)."""

    get: Callable[[], Any]
    set: Callable[[Any], None]


def build_effect_registry(effect: MinionEffect) -> Dict[str, ParamHandlers]:
    """Registry for the params that live entirely on a `MinionEffect`
    instance (`effect.*`, `minionese.*`, `pitch.*`). Used both by
    `VoiceEngine` (bound to its live `effect`) and by the recorder's
    `render_current()`, which configures a scratch `MinionEffect` from a
    snapshot without needing a running engine.
    """
    pitch: PitchShifter = effect.pitch
    m = effect.minionese
    reg: Dict[str, ParamHandlers] = {
        "effect.max_semitones": ParamHandlers(
            get=lambda: effect.max_semitones,
            set=lambda v: effect.set_max_semitones(float(v)),
        ),
        "effect.eq_enabled": ParamHandlers(
            get=lambda: effect.eq_enabled,
            set=lambda v: effect.set_eq_enabled(bool(v)),
        ),
        "effect.eq_gain_db": ParamHandlers(
            get=lambda: effect.max_eq_gain_db,
            set=lambda v: effect.set_eq_gain_db(float(v)),
        ),
        "effect.eq_center_hz": ParamHandlers(
            get=lambda: effect.eq_center_hz,
            set=lambda v: effect.set_eq_center_hz(float(v)),
        ),
        "effect.eq_q": ParamHandlers(
            get=lambda: effect.eq_q,
            set=lambda v: effect.set_eq_q(float(v)),
        ),
        "minionese.semitones": ParamHandlers(
            get=lambda: m.semitones,
            set=lambda v: m.set_semitones(float(v)),
        ),
        "minionese.wobble_ms": ParamHandlers(
            get=lambda: m.wobble_ms,
            set=lambda v: m.set_wobble_ms(float(v)),
        ),
        "minionese.chunk_ms": ParamHandlers(
            get=lambda: m.chunk_ms,
            set=lambda v: m.set_chunk_ms(float(v)),
        ),
        "minionese.shuffle_k": ParamHandlers(
            get=lambda: m.shuffle_k,
            set=lambda v: m.set_shuffle_k(int(v)),
        ),
        "minionese.fade_ms": ParamHandlers(
            get=lambda: m.fade_ms,
            set=lambda v: m.set_fade_ms(float(v)),
        ),
        "minionese.max_slew": ParamHandlers(
            get=lambda: m.max_slew,
            set=lambda v: m.set_max_slew(float(v)),
        ),
        "pitch.frame": ParamHandlers(
            get=lambda: pitch.L,
            set=lambda v: pitch.set_frame(int(v)),
        ),
        "pitch.tol": ParamHandlers(
            get=lambda: pitch.tol,
            set=lambda v: pitch.set_tol(int(v)),
        ),
    }
    return reg


def build_engine_registry(engine) -> Dict[str, ParamHandlers]:
    """Full registry for a live `VoiceEngine`: global + effect + I/O."""
    reg: Dict[str, ParamHandlers] = {
        "enabled": ParamHandlers(
            get=lambda: engine.enabled,
            set=lambda v: engine.set_enabled(bool(v)),
        ),
        "gibberish": ParamHandlers(
            get=lambda: engine.effect.gibberish,
            set=lambda v: engine.set_gibberish(bool(v)),
        ),
        "intensity": ParamHandlers(
            # `effect.intensity` is only refreshed inside the audio
            # callback (from the manual override or the auto-ramp), so
            # while stopped -- or before the next block runs -- it can lag
            # a manual override that was just set. Prefer the manual
            # override itself when one is pending so `snapshot()` (and
            # thus `render_current()`) reflects the *intended* value
            # immediately, not just whatever the last processed block saw.
            get=lambda: (
                engine._manual_intensity if engine._manual_intensity is not None else engine.effect.intensity
            ),
            set=lambda v: engine.set_manual_intensity(float(v)),
        ),
        "ramp.duration_s": ParamHandlers(
            get=lambda: engine.ramp.duration_s,
            set=lambda v: engine.ramp.set_duration(float(v)),
        ),
        "io.sample_rate": ParamHandlers(
            get=lambda: engine.sample_rate,
            set=lambda v: engine.set_io_param("io.sample_rate", int(v)),
        ),
        "io.blocksize": ParamHandlers(
            get=lambda: engine.blocksize,
            set=lambda v: engine.set_io_param("io.blocksize", int(v)),
        ),
        "io.ring_ms": ParamHandlers(
            get=lambda: engine.ring_ms,
            set=lambda v: engine.set_io_param("io.ring_ms", float(v)),
        ),
        "io.input_device": ParamHandlers(
            get=lambda: -1 if engine.input_device is None else int(engine.input_device),
            set=lambda v: engine.set_io_param("io.input_device", None if int(v) < 0 else int(v)),
        ),
        "io.output_device": ParamHandlers(
            get=lambda: -1 if engine.output_device is None else int(engine.output_device),
            set=lambda v: engine.set_io_param("io.output_device", None if int(v) < 0 else int(v)),
        ),
    }
    reg.update(build_effect_registry(engine.effect))
    return reg
