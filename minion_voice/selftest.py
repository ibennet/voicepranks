"""Audition the Minion DSP effect on a WAV file with no audio hardware.

Usage:
    python -m minion_voice.selftest OUTPUT.wav
    python -m minion_voice.selftest INPUT.wav OUTPUT.wav [--semitones 8] [--eq-db 8]
    python -m minion_voice.selftest INPUT.wav OUTPUT.wav --gibberish

If INPUT.wav is omitted, a 2-second synthetic test tone (200 Hz with a
little vibrato) is generated so the tool is runnable with zero inputs.

Pass --gibberish to route through Minionese mode instead of the plain
pitch shift (--semitones/--eq-db are ignored in that mode).
"""
from __future__ import annotations

import argparse
import wave
from typing import Optional, Tuple

import numpy as np

from .dsp.effect import MinionEffect

DEFAULT_SAMPLE_RATE = 48000


def _synthesize_test_signal(sample_rate: int = DEFAULT_SAMPLE_RATE, duration_s: float = 2.0) -> np.ndarray:
    """200 Hz tone with a bit of vibrato, so selftest is runnable with no inputs."""
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate

    base_freq = 200.0
    vibrato_depth_hz = 6.0
    vibrato_rate_hz = 5.0
    inst_freq = base_freq + vibrato_depth_hz * np.sin(2.0 * np.pi * vibrato_rate_hz * t)

    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate
    signal = 0.6 * np.sin(phase)

    # Gentle fade in/out to avoid a click at the file edges.
    fade_len = min(int(0.02 * sample_rate), n // 2)
    if fade_len > 0:
        fade = np.linspace(0.0, 1.0, fade_len)
        signal[:fade_len] *= fade
        signal[-fade_len:] *= fade[::-1]

    return signal.astype(np.float32)


def read_wav(path: str) -> Tuple[int, np.ndarray]:
    """Read a mono (or first-channel-of-multi) 16-bit PCM WAV as float32 [-1, 1]."""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sampwidth={sampwidth}")

    data = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]

    float_data = (data.astype(np.float32)) / 32768.0
    return sample_rate, float_data


def write_wav(path: str, sample_rate: int, mono: np.ndarray) -> None:
    """Write a mono float32 [-1, 1] array as a 16-bit PCM WAV file."""
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def _peak_freq(signal: np.ndarray, sample_rate: int) -> float:
    if signal.shape[0] < 2:
        return 0.0
    windowed = signal.astype(np.float64) * np.hanning(len(signal))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    mag = np.abs(spectrum)
    peak_idx = int(np.argmax(mag))
    return float(freqs[peak_idx])


def _process_in_blocks(effect: MinionEffect, signal: np.ndarray, blocksize: int = 1024) -> np.ndarray:
    out_chunks = []
    n = signal.shape[0]
    for start in range(0, n, blocksize):
        chunk = signal[start:start + blocksize]
        out_chunks.append(effect.process(chunk))
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audition the Minion voice DSP effect on a WAV file (no audio hardware needed)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="[INPUT.wav] OUTPUT.wav -- if only one path is given, it is treated as OUTPUT.wav "
        "and a synthetic test tone is used as input.",
    )
    parser.add_argument("--semitones", type=float, default=MinionEffect.MAX_SEMITONES, help="Pitch shift in semitones.")
    parser.add_argument("--eq-db", type=float, default=MinionEffect.MAX_EQ_GAIN_DB, help="Presence EQ boost in dB.")
    parser.add_argument(
        "--gibberish",
        action="store_true",
        help="Route through Minionese gibberish mode instead of the plain pitch shift.",
    )
    args = parser.parse_args(argv)

    if len(args.paths) == 1:
        input_path = None
        output_path = args.paths[0]
    elif len(args.paths) == 2:
        input_path, output_path = args.paths
    else:
        parser.error("expected 1 or 2 positional paths: [INPUT.wav] OUTPUT.wav")
        return

    if input_path is None:
        sample_rate = DEFAULT_SAMPLE_RATE
        input_signal = _synthesize_test_signal(sample_rate)
        print(f"No input given -- synthesized a {len(input_signal) / sample_rate:.1f}s 200 Hz test tone with vibrato.")
    else:
        sample_rate, input_signal = read_wav(input_path)
        print(f"Read input '{input_path}': {len(input_signal) / sample_rate:.2f}s @ {sample_rate} Hz")

    effect = MinionEffect(sample_rate)
    effect.pitch.set_semitones(args.semitones)
    effect.eq.set_gain_db(args.eq_db)
    effect.set_gibberish(args.gibberish)

    output_signal = _process_in_blocks(effect, input_signal)

    write_wav(output_path, sample_rate, output_signal)

    in_rms = _rms(input_signal)
    out_rms = _rms(output_signal)
    in_peak_freq = _peak_freq(input_signal, sample_rate)
    out_peak_freq = _peak_freq(output_signal, sample_rate)

    print(f"Wrote output '{output_path}'")
    if args.gibberish:
        print("Mode: Minionese (gibberish) -- --semitones/--eq-db do not apply")
    else:
        print(f"Mode: plain pitch shift -- semitones: {args.semitones}, EQ gain: {args.eq_db} dB")
    print(f"Input  RMS: {in_rms:.4f}   peak freq: {in_peak_freq:.1f} Hz")
    print(f"Output RMS: {out_rms:.4f}   peak freq: {out_peak_freq:.1f} Hz")


if __name__ == "__main__":
    main()
