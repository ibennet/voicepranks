"""The shuffle gibberish engine buffers a full shuffle window before it can
emit audio, so its latency grows with the scramble amount. These tests check
that (a) each engine reports a sane latency, and (b) `VoiceEngine` sizes and
pre-fills its output ring to cover that latency so the live path can't starve
on a big scramble window. Driven entirely without audio hardware -- the ring
priming/sizing helpers are called directly.
"""
from __future__ import annotations

import numpy as np

from voicepranks.audio.engine import VoiceEngine
from voicepranks.dsp.effect import MinionEffect

SAMPLE_RATE = 48000


def test_shuffle_latency_grows_with_scramble_window():
    eff = MinionEffect(SAMPLE_RATE)
    eff.set_gibberish(True)
    eff.set_use_shuffle(True)

    eff.shuffle.set_chunk_ms(100.0)
    eff.shuffle.set_shuffle_k(2)
    small = eff.latency_ms()

    eff.shuffle.set_chunk_ms(160.0)
    eff.shuffle.set_shuffle_k(6)
    big = eff.latency_ms()

    # 6*160 window is far larger than 2*100, so latency must grow with it.
    assert big > small
    assert big > 6 * 160.0 * 0.9  # dominated by the K*C shuffle window


def test_plain_pitch_path_has_zero_reported_latency():
    eff = MinionEffect(SAMPLE_RATE)
    eff.set_gibberish(False)
    assert eff.latency_ms() == 0.0


def test_prime_for_effect_covers_and_grows_ring_for_big_window():
    # A tiny default ring (100ms) can't hold a ~1s shuffle window; the engine
    # must both grow capacity and pre-fill enough cushion to cover it.
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=256, ring_ms=100.0)
    engine.set_gibberish(True)
    engine.set_param("minionese.use_shuffle", True)
    engine.set_param("shuffle.chunk_ms", 160.0)
    engine.set_param("shuffle.shuffle_k", 6)

    cushion_ms = engine._cushion_ms()
    window_ms = 6 * 160.0
    assert cushion_ms > window_ms  # cushion must exceed the shuffle window

    engine._prime_for_effect()

    # Ring grew to hold the cushion, and was actually pre-filled to it.
    cushion_samples = int(SAMPLE_RATE * cushion_ms / 1000.0)
    assert engine.ring.capacity >= cushion_samples
    assert engine.ring.count >= cushion_samples * 0.99


def test_prime_for_effect_is_cheap_for_plain_path():
    engine = VoiceEngine(sample_rate=SAMPLE_RATE, blocksize=256, ring_ms=200.0)
    engine.set_gibberish(False)
    engine._prime_for_effect()
    # Plain path reports zero latency, so the cushion is just the headroom
    # floor (60ms) -- well under the default 200ms ring, no growth needed.
    assert engine.ring.count > 0
    assert engine.ring.count < int(SAMPLE_RATE * 200.0 / 1000.0)
