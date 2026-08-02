"""Built-in voice presets: named bundles of sound-character params.

A preset is a curated `{param_name: value}` dict drawn from
`params.PARAM_SPECS`. It intentionally covers only *sound-character* params
(the gibberish/shuffle knobs) and leaves out `io.*`, `enabled`, `monitor`,
and `intensity` -- so applying a preset changes the voice without disturbing
your devices, on/off state, live monitor, or your intensity setting.

`intensity` is deliberately excluded so it stays a live master control: it
scales the applied preset's character (wobble depth + scramble strength), and
applying a preset never resets it.

Applying is done by `VoiceEngine.apply_preset`, which just feeds each entry
through `set_param`. The two presets below were dialed in by ear during live
tuning; edit the values here (or add entries) to change/extend them.
"""
from __future__ import annotations

from typing import Dict, List

from .params import PARAM_SPECS_BY_NAME

# Params a preset is never allowed to set: environment/session state, not
# voice character.
_EXCLUDED_PREFIXES = ("io.",)
# `intensity` is a live master control the user owns -- presets must not
# reset it (they scale under it instead).
_EXCLUDED_NAMES = {"enabled", "monitor", "intensity"}


PRESETS: Dict[str, Dict] = {
    # Chopped / blippy / scrambled -- discrete staccato syllable blips, the
    # Animal-Crossing-villager sound.
    "animalese": {
        "gibberish": True,
        "minionese.use_shuffle": True,
        "shuffle.semitones": 5.5,
        "shuffle.wobble_ms": 1.0,
        "shuffle.chunk_ms": 80.0,
        "shuffle.shuffle_k": 6,
        "shuffle.fade_ms": 18.0,
        "shuffle.reverse_prob": 0.5,
        "shuffle.nasality": 0.0,
        "shuffle.max_slew": 1.0,
    },
    # Connected / pitched / lilting -- no scramble, so speech stays flowing;
    # a moderate pitch-up plus sing-song wobble reads as "Minion".
    "minion": {
        "gibberish": True,
        "minionese.use_shuffle": True,
        "shuffle.semitones": 10.0,
        "shuffle.wobble_ms": 4.0,
        "shuffle.chunk_ms": 150.0,
        "shuffle.shuffle_k": 1,
        "shuffle.fade_ms": 25.0,
        "shuffle.reverse_prob": 0.0,
        "shuffle.nasality": 0.7,
        "shuffle.max_slew": 1.0,
    },
}


def preset_names() -> List[str]:
    """Preset names in definition order."""
    return list(PRESETS.keys())


def get_preset(name: str) -> Dict:
    """Return a copy of the named preset's param dict.

    Raises KeyError if there is no such preset.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r} (have {preset_names()})")
    return dict(PRESETS[name])


def _validate() -> None:
    """Fail loudly at import if a preset references an unknown or excluded
    param -- guards against typos and against a preset reaching into
    device/session state it shouldn't touch."""
    for preset_name, values in PRESETS.items():
        for key in values:
            if key not in PARAM_SPECS_BY_NAME:
                raise ValueError(f"preset {preset_name!r} sets unknown param {key!r}")
            if key in _EXCLUDED_NAMES or key.startswith(_EXCLUDED_PREFIXES):
                raise ValueError(
                    f"preset {preset_name!r} must not set {key!r} "
                    "(io/enabled/monitor are session state, not voice character)"
                )


_validate()
