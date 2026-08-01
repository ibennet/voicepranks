"""Real-time voice engine: mic capture -> MinionEffect -> virtual output."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from ..dsp.effect import MinionEffect
from ..dsp.ramp import IntensityRamp
from .devices import find_virtual_output


class _RingBuffer:
    """Thread-safe circular float32 sample buffer.

    Overrun (writer outrunning reader): drops the oldest samples to make
    room for the new ones. Underrun (reader outrunning writer): zero-fills
    the missing tail of the read.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buf = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        self.lock = threading.Lock()
        self.underruns = 0
        self.overruns = 0

    def write(self, data: np.ndarray) -> None:
        with self.lock:
            n = data.shape[0]
            if n > self.capacity:
                data = data[-self.capacity:]
                n = data.shape[0]

            free = self.capacity - self.count
            if n > free:
                overflow = n - free
                self.read_idx = (self.read_idx + overflow) % self.capacity
                self.count -= overflow
                self.overruns += 1

            end_space = self.capacity - self.write_idx
            if n <= end_space:
                self.buf[self.write_idx:self.write_idx + n] = data
            else:
                self.buf[self.write_idx:] = data[:end_space]
                self.buf[:n - end_space] = data[end_space:]
            self.write_idx = (self.write_idx + n) % self.capacity
            self.count = min(self.capacity, self.count + n)

    def read(self, n: int) -> np.ndarray:
        with self.lock:
            out = np.zeros(n, dtype=np.float32)
            avail = min(n, self.count)
            if avail > 0:
                end_space = self.capacity - self.read_idx
                if avail <= end_space:
                    out[:avail] = self.buf[self.read_idx:self.read_idx + avail]
                else:
                    out[:end_space] = self.buf[self.read_idx:]
                    out[end_space:avail] = self.buf[:avail - end_space]
                self.read_idx = (self.read_idx + avail) % self.capacity
                self.count -= avail
            if avail < n:
                self.underruns += 1
            return out


class VoiceEngine:
    """Owns the DSP effect, the intensity ramp, and the two audio streams."""

    def __init__(
        self,
        sample_rate: int = 48000,
        blocksize: int = 256,
        ring_ms: float = 200.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)

        self.effect = MinionEffect(self.sample_rate, channels=1)
        self.ramp = IntensityRamp()

        ring_capacity = max(self.blocksize * 2, int(self.sample_rate * ring_ms / 1000.0))
        self.ring = _RingBuffer(ring_capacity)

        self.enabled = False
        self._manual_intensity: Optional[float] = None

        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.running = False
        self.last_error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def start(self, input_device: Optional[int] = None, output_device: Optional[int] = None) -> None:
        if output_device is None:
            output_device = find_virtual_output()

        self.input_device = input_device
        self.output_device = output_device

        out_info = sd.query_devices(output_device)
        out_channels = max(1, int(out_info.get("max_output_channels", 1)))
        self._out_channels = out_channels

        self.input_stream = sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._input_callback,
        )
        self.output_stream = sd.OutputStream(
            device=output_device,
            channels=out_channels,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._output_callback,
        )
        # Pre-fill the ring with a short silence cushion so the output stream
        # doesn't starve while the effect's internal pipeline latency (WSOLA +
        # Minionese STFT buffering, up to ~60ms) fills in after start. Without
        # this the first ~second of Minionese mode underruns while it primes.
        self._prime_ring(90.0)
        self.input_stream.start()
        self.output_stream.start()
        self.running = True

    def stop(self) -> None:
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
        self.running = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            self._manual_intensity = None
            self.ramp.start()
        else:
            self.ramp.stop()
            self.effect.set_intensity(0.0)

    def set_manual_intensity(self, t: float) -> None:
        """Override the auto-ramp with a fixed intensity (e.g. slider drag)."""
        self._manual_intensity = min(max(float(t), 0.0), 1.0)

    def set_gibberish(self, b: bool) -> None:
        self.effect.set_gibberish(b)
        # Switching into Minionese mid-run resets its internal pipeline, so
        # give the ring the same cushion to avoid a toggle-on dropout.
        if b and self.running:
            self._prime_ring(90.0)

    def _prime_ring(self, ms: float) -> None:
        """Top the output ring up to `ms` of buffered silence — a startup /
        mode-switch cushion against the effect pipeline's latency."""
        target = int(self.sample_rate * ms / 1000.0)
        need = target - self.ring.count
        if need > 0:
            self.ring.write(np.zeros(need, dtype=np.float32))

    def get_status(self) -> dict:
        return {
            "intensity": self.effect.intensity,
            "underruns": self.ring.underruns,
            "overruns": self.ring.overruns,
            "running": self.running,
        }

    # -- stream callbacks ------------------------------------------------

    def _input_callback(self, indata, frames, time_info, status) -> None:  # noqa: D401
        try:
            mono = indata[:, 0] if indata.ndim > 1 else indata
            mono = np.asarray(mono, dtype=np.float32)

            if self.enabled:
                t = self._manual_intensity if self._manual_intensity is not None else self.ramp.current()
                self.effect.set_intensity(t)
                processed = self.effect.process(mono)
            else:
                processed = mono

            self.ring.write(processed)
        except Exception as exc:  # keep the audio thread alive
            self.last_error = str(exc)

    def _output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: D401
        try:
            mono = self.ring.read(frames)
            channels = outdata.shape[1]
            if channels == 1:
                outdata[:, 0] = mono
            else:
                outdata[:, :] = np.repeat(mono.reshape(-1, 1), channels, axis=1)
        except Exception as exc:
            self.last_error = str(exc)
            outdata.fill(0.0)
