"""Real-time voice engine: mic capture -> MinionEffect -> virtual output."""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import sounddevice as sd

from .. import params
from .. import presets as presets_mod
from ..dsp.effect import MinionEffect
from ..dsp.ramp import IntensityRamp
from . import wavio
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


def _process_in_blocks(effect: MinionEffect, signal: np.ndarray, blocksize: int = 1024) -> np.ndarray:
    """Run a full buffer through `effect` in fixed-size chunks, mirroring
    how the real-time callback feeds it -- used by `render_current()` so
    a recorded take is rendered the same way it would sound live."""
    out_chunks = []
    n = signal.shape[0]
    for start in range(0, n, blocksize):
        chunk = signal[start:start + blocksize]
        out_chunks.append(effect.process(chunk))
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)


class VoiceEngine:
    """Owns the DSP effect, the intensity ramp, the two audio streams, the
    live param registry, and the raw/rendered take recorder."""

    def __init__(
        self,
        sample_rate: int = 48000,
        blocksize: int = 256,
        ring_ms: float = 200.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.ring_ms = float(ring_ms)

        self.effect = MinionEffect(self.sample_rate, channels=1)
        self.ramp = IntensityRamp()

        ring_capacity = max(self.blocksize * 2, int(self.sample_rate * self.ring_ms / 1000.0))
        self.ring = _RingBuffer(ring_capacity)

        self.enabled = False
        self._manual_intensity: Optional[float] = None
        # Last intensity pushed to the effect from the callback, so a constant
        # intensity (e.g. after the ramp settles) doesn't re-derive every
        # sub-engine's coefficients every block. Param changes go through their
        # own setters, so they still apply immediately regardless of this.
        self._last_applied_intensity: Optional[float] = None

        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.running = False
        self.last_error: Optional[str] = None

        # -- live monitor: a second output stream to an audible device
        # (speakers/headphones) so the user can HEAR the processed effect in
        # real time while the main output feeds the virtual cable. Fed the
        # same processed audio via its own ring so the two sinks don't fight
        # over one buffer. The listening/playback device is also what the
        # Play button routes takes to. `playback_device=None` -> system
        # default output.
        self.monitor_enabled = False
        self.playback_device: Optional[int] = None
        self.monitor_stream: Optional[sd.OutputStream] = None
        self.monitor_ring = _RingBuffer(ring_capacity)

        # -- live param registry (single source of truth; see params.py) --
        self._registry = params.build_engine_registry(self)
        self._pending: dict = {}
        self._param_lock = threading.Lock()

        # -- level meter (updated every input callback block) --------------
        self._dry_rms = 0.0
        self._dry_peak = 0.0
        self._proc_rms = 0.0
        self._proc_peak = 0.0

        # -- recorder ------------------------------------------------------
        # Two takes are captured per recording:
        #  - "live": the processed effect output, captured block-by-block as
        #    it is generated in real time (what you'd have heard live). Play
        #    replays this verbatim -- no re-render, nothing re-applied.
        #  - "raw": the dry (pre-effect) mic signal, kept so the effect can
        #    be re-rendered onto the same take with tweaked params (the A/B
        #    tuning flow). Optional; the primary flow is the live take.
        self._record_lock = threading.Lock()
        self._recording = False
        self._raw_chunks: List[np.ndarray] = []
        self._raw_take: Optional[np.ndarray] = None
        self._live_chunks: List[np.ndarray] = []
        self._live_take: Optional[np.ndarray] = None
        self._rendered_take: Optional[np.ndarray] = None

    # -- param registry (shared by Tkinter UI + HTTP control server) -----

    def set_param(self, name: str, value) -> None:
        """Set one live param by its dotted name (see `params.PARAM_SPECS`).

        I/O params (`io.*`) restart the streams immediately (can't be
        applied from inside the audio callback they'd be tearing down).
        Everything else is stashed and applied at the top of the next
        `_input_callback` while the engine is running, so DSP-stage
        rebuilds never race the audio thread that's using them. If the
        engine isn't running, there is no audio thread to race, so the
        value is applied immediately instead of waiting for a callback
        that will never come.
        """
        if name == "io.playback_device":
            # The listening device manages its own (monitor) stream and must
            # NOT restart the main input/output streams. Apply immediately on
            # the caller thread (opening a stream from the audio callback is
            # unsafe), whether running or not.
            self.set_playback_device(None if int(value) < 0 else int(value))
            return
        if name.startswith("io."):
            self.set_io_param(name, value)
            return
        if self.running:
            with self._param_lock:
                self._pending[name] = value
        else:
            handlers = self._registry.get(name)
            if handlers is None:
                raise KeyError(f"unknown param: {name}")
            handlers.set(value)

    def snapshot(self) -> dict:
        """Return the current value of every registered param."""
        return {name: handlers.get() for name, handlers in self._registry.items()}

    def apply_preset(self, name: str) -> dict:
        """Apply a built-in preset (see `presets.PRESETS`): feed each of its
        sound-character params through `set_param`. Works whether the engine
        is running or stopped. Returns the resulting full snapshot. Raises
        KeyError for an unknown preset name."""
        preset = presets_mod.get_preset(name)
        for pname, value in preset.items():
            self.set_param(pname, value)
        return self.snapshot()

    def _drain_pending(self) -> None:
        """Apply any params queued by `set_param` since the last block.
        Called at the top of `_input_callback`, i.e. on the audio thread."""
        with self._param_lock:
            if not self._pending:
                return
            pending, self._pending = self._pending, {}
        reprime = False
        for name, value in pending.items():
            handlers = self._registry.get(name)
            if handlers is None:
                continue
            try:
                handlers.set(value)
            except Exception as exc:  # keep the audio thread alive
                self.last_error = str(exc)
                continue
            # Params that rebuild a DSP stage (`needs_reset`) or switch the
            # gibberish engine can change the active path's buffering latency
            # -- notably the shuffle window. Re-prime the output cushion so a
            # freshly-enlarged window can't starve the stream mid-run.
            spec = params.PARAM_SPECS_BY_NAME.get(name)
            if name == "minionese.use_shuffle" or (spec is not None and spec.needs_reset):
                reprime = True
        if reprime and self.running:
            self._prime_for_effect()

    def set_io_param(self, name: str, value) -> None:
        """Apply an `io.*` param, restarting the streams (stop then start)
        if the engine is currently running."""
        was_running = self.running
        if was_running:
            self.stop()

        rebuild_effect = False
        if name == "io.sample_rate":
            self.sample_rate = int(value)
            rebuild_effect = True
        elif name == "io.blocksize":
            self.blocksize = int(value)
        elif name == "io.ring_ms":
            self.ring_ms = float(value)
        elif name == "io.input_device":
            self.input_device = None if value is None or int(value) < 0 else int(value)
        elif name == "io.output_device":
            self.output_device = None if value is None or int(value) < 0 else int(value)
        else:
            raise KeyError(f"unknown io param: {name}")

        if rebuild_effect:
            # Pitch/Minionese internal windows are sized in samples off the
            # sample rate, so a rate change needs a fresh DSP chain.
            self.effect = MinionEffect(self.sample_rate, channels=1)

        # Devices/blocksize/ring_ms don't change effect identity, but
        # rebuilding the registry unconditionally is cheap and keeps every
        # closure bound to the current `self.effect`.
        self._registry = params.build_engine_registry(self)

        ring_capacity = max(self.blocksize * 2, int(self.sample_rate * self.ring_ms / 1000.0))
        self.ring = _RingBuffer(ring_capacity)

        if was_running:
            self.start(input_device=self.input_device, output_device=self.output_device)

    # -- lifecycle -----------------------------------------------------

    def start(self, input_device: Optional[int] = None, output_device: Optional[int] = None) -> None:
        if input_device is None:
            input_device = self.input_device
        if output_device is None:
            output_device = self.output_device
        if output_device is None:
            output_device = find_virtual_output()

        self.input_device = input_device
        self.output_device = output_device
        # A restart may follow an effect rebuild (io.sample_rate); force the
        # first callback to re-apply intensity rather than trust the cache.
        self._last_applied_intensity = None

        out_info = sd.query_devices(output_device)
        out_channels = max(1, int(out_info.get("max_output_channels", 1)))

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
        # Pre-fill the ring with a silence cushion sized to the active
        # effect path's internal buffering latency, so the output stream
        # doesn't starve while that latency fills in after start. Critical
        # for the shuffle gibberish engine, whose latency grows with the
        # scramble amount (a full shuffle window can be ~1s).
        self._prime_for_effect()
        self.input_stream.start()
        self.output_stream.start()
        self.running = True
        if self.monitor_enabled:
            self._open_monitor_stream()

    def stop(self) -> None:
        self._close_monitor_stream()
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
        self.running = False

    # -- live monitor ----------------------------------------------------

    def set_monitor_enabled(self, b: bool) -> None:
        """Turn the live monitor (audible processed-audio output) on/off.
        Opens/closes the monitor stream immediately if the engine is
        running; otherwise it takes effect on the next `start()`."""
        self.monitor_enabled = bool(b)
        if not self.running:
            return
        if self.monitor_enabled:
            self._open_monitor_stream()
        else:
            self._close_monitor_stream()

    def set_playback_device(self, device: Optional[int]) -> None:
        """Pick the device used for *listening* -- both the live monitor and
        the Play button route here (None = system default output). Re-opens
        the monitor stream if it's currently active."""
        self.playback_device = None if device is None else int(device)
        if self.running and self.monitor_enabled:
            self._close_monitor_stream()
            self._open_monitor_stream()

    def _open_monitor_stream(self) -> None:
        if self.monitor_stream is not None:
            return
        try:
            device = self.playback_device  # None -> system default output
            # `query_devices(kind='output')` resolves the system default when
            # no explicit device is given, without assuming the shape of
            # `sd.default.device`.
            info = sd.query_devices(device) if device is not None else sd.query_devices(kind="output")
            channels = max(1, int(info.get("max_output_channels", 1)))
            self.monitor_ring = _RingBuffer(self.ring.capacity)
            self._prime_ring_target(self.monitor_ring, self._cushion_ms())
            self.monitor_stream = sd.OutputStream(
                device=device,
                channels=channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._monitor_output_callback,
            )
            self.monitor_stream.start()
        except Exception as exc:
            self.last_error = f"Live monitor failed: {exc}"
            self.monitor_stream = None

    def _close_monitor_stream(self) -> None:
        if self.monitor_stream is not None:
            try:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None

    def _monitor_output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: D401
        try:
            mono = self.monitor_ring.read(frames)
            channels = outdata.shape[1]
            if channels == 1:
                outdata[:, 0] = mono
            else:
                outdata[:, :] = mono[:, None]  # broadcast mono across channels (no temp)
        except Exception as exc:
            self.last_error = str(exc)
            outdata.fill(0.0)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        # Both branches change the effect's intensity out-of-band (the ramp
        # restart, or the direct set_intensity(0) below), so invalidate the
        # callback's cache -- otherwise a re-enable with no ramp could match
        # the stale cached value and skip re-applying, leaving the effect at 0.
        self._last_applied_intensity = None
        if self.enabled:
            self._manual_intensity = None
            self.ramp.start()
        else:
            self.ramp.stop()
            self.effect.set_intensity(0.0)

    def set_manual_intensity(self, t: float) -> None:
        """Override the auto-ramp with a fixed intensity (e.g. slider drag)."""
        self._manual_intensity = min(max(float(t), 0.0), 1.0)

    def restart_ramp(self, duration_s: Optional[float] = None) -> None:
        """Restart the intensity ramp from 0 with the current (or given)
        duration, handing intensity control back to the ramp (clears any
        manual override). `duration_s` is set directly so the new value takes
        effect immediately, without waiting for the pending-param drain."""
        if duration_s is not None:
            self.ramp.set_duration(float(duration_s))
        self._manual_intensity = None
        self.ramp.start()

    def set_gibberish(self, b: bool) -> None:
        self.effect.set_gibberish(b)
        # Switching into a gibberish engine mid-run resets its internal
        # pipeline, so re-prime the ring to that engine's latency to avoid a
        # toggle-on dropout.
        if b and self.running:
            self._prime_for_effect()

    def _prime_ring_target(self, ring: "_RingBuffer", ms: float) -> None:
        """Top a ring up to `ms` of buffered silence — a startup / mode-switch
        cushion against the effect pipeline's latency."""
        target = int(self.sample_rate * ms / 1000.0)
        need = target - ring.count
        if need > 0:
            ring.write(np.zeros(need, dtype=np.float32))

    def _cushion_ms(self) -> float:
        """Output pre-fill needed to cover the active effect path's internal
        buffering latency, plus headroom. The shuffle gibberish engine must
        accumulate a full shuffle window before emitting, so its latency (and
        thus the cushion) grows with the scramble amount; if we primed a
        fixed cushion smaller than that window the output would starve every
        window. Headroom (1.5x + 60ms) absorbs block jitter."""
        try:
            eff_latency = self.effect.latency_ms()
        except Exception:
            eff_latency = 0.0
        return eff_latency * 1.5 + 60.0

    def _grown_ring(self, ring: "_RingBuffer", ms: float) -> "_RingBuffer":
        """Return a ring with capacity for at least `ms`, carrying over any
        buffered audio and the cumulative counters. Returns `ring` unchanged
        if it is already big enough. Growing on the audio thread is safe here
        because `_RingBuffer` is only read/written under its own lock and we
        swap in a pre-filled replacement."""
        target = int(self.sample_rate * ms / 1000.0)
        want_capacity = max(self.blocksize * 2, target + self.blocksize * 2)
        if want_capacity <= ring.capacity:
            return ring
        new = _RingBuffer(want_capacity)
        # `read`/`write` each lock internally; don't hold `ring.lock` here
        # (it's non-reentrant -> would deadlock). Carry over buffered audio
        # and the cumulative under/overrun counters.
        carried = ring.read(ring.count) if ring.count > 0 else np.zeros(0, dtype=np.float32)
        new.underruns = ring.underruns
        new.overruns = ring.overruns
        if carried.size > 0:
            new.write(carried)
        return new

    def _prime_for_effect(self) -> None:
        """Size the ring(s) and pre-fill to cover the active effect path's
        latency. Called on start, on gibberish toggle, and after a scramble
        param reset — anywhere the effect's internal latency may have grown.
        The monitor ring gets the same treatment so the live monitor doesn't
        starve on a big shuffle window either."""
        cushion = self._cushion_ms()
        capacity_ms = cushion + self.blocksize / self.sample_rate * 1000.0
        self.ring = self._grown_ring(self.ring, capacity_ms)
        self._prime_ring_target(self.ring, cushion)
        if self.monitor_enabled and self.monitor_stream is not None:
            self.monitor_ring = self._grown_ring(self.monitor_ring, capacity_ms)
            self._prime_ring_target(self.monitor_ring, cushion)

    def get_status(self) -> dict:
        return {
            "intensity": self.effect.intensity,
            "underruns": self.ring.underruns,
            "overruns": self.ring.overruns,
            "running": self.running,
            "enabled": self.enabled,
            "gibberish": self.effect.gibberish,
            "monitor": self.monitor_enabled,
            "latency_ms": round(self.effect.latency_ms(), 1),
            "last_error": self.last_error,
            "recording": self._recording,
            "take_seconds": self._current_take_seconds(),
            "has_live_take": self._live_take is not None and self._live_take.size > 0,
            "has_raw_take": self._raw_take is not None and self._raw_take.size > 0,
            "has_rendered_take": self._rendered_take is not None and self._rendered_take.size > 0,
            "level": {
                "dry_rms": self._dry_rms,
                "dry_peak": self._dry_peak,
                "processed_rms": self._proc_rms,
                "processed_peak": self._proc_peak,
            },
        }

    # -- recorder --------------------------------------------------------

    def record_start(self) -> None:
        """Begin a new take: capture the live processed output (and the dry
        mic for optional re-render)."""
        with self._record_lock:
            self._raw_chunks = []
            self._raw_take = None
            self._live_chunks = []
            self._live_take = None
            self._rendered_take = None
            self._recording = True

    def record_stop(self) -> np.ndarray:
        """Stop recording; finalize both takes and return the live processed
        take (what Play replays)."""
        with self._record_lock:
            self._recording = False
            self._raw_take = (
                np.concatenate(self._raw_chunks) if self._raw_chunks else np.zeros(0, dtype=np.float32)
            )
            self._live_take = (
                np.concatenate(self._live_chunks) if self._live_chunks else np.zeros(0, dtype=np.float32)
            )
            self._raw_chunks = []
            self._live_chunks = []
        return self._live_take

    def _current_take_seconds(self) -> float:
        with self._record_lock:
            if self._recording:
                total = sum(c.shape[0] for c in self._raw_chunks)
            elif self._raw_take is not None:
                total = self._raw_take.shape[0]
            else:
                total = 0
        return total / self.sample_rate if self.sample_rate else 0.0

    def render_current(self) -> np.ndarray:
        """Build a fresh `MinionEffect` from the current param snapshot and
        run the raw take through it, block by block -- an A/B "re-render"
        of the last take with whatever settings are dialed in *now*,
        without disturbing the live engine's own effect/state."""
        if self._raw_take is None or self._raw_take.size == 0:
            raise RuntimeError("No raw take recorded yet -- call record_start()/record_stop() first.")

        values = self.snapshot()
        fresh = MinionEffect(self.sample_rate, channels=1)
        fresh.set_gibberish(bool(values.get("gibberish", False)))

        effect_registry = params.build_effect_registry(fresh)
        for name, handlers in effect_registry.items():
            if name in values:
                handlers.set(values[name])

        fresh.set_intensity(float(values.get("intensity", 1.0)))

        out = _process_in_blocks(fresh, self._raw_take, blocksize=self.blocksize)
        self._rendered_take = out
        return out

    def _get_take(self, which: str) -> np.ndarray:
        if which == "live":
            buf = self._live_take
        elif which == "raw":
            buf = self._raw_take
        elif which == "rendered":
            buf = self._rendered_take
        else:
            raise ValueError(f"unknown take '{which}' (expected 'live', 'raw' or 'rendered')")
        if buf is None or buf.size == 0:
            raise RuntimeError(f"no {which} take available")
        return buf

    def play(self, which: str = "live") -> None:
        # Route playback to the listening device (speakers/headphones), NOT
        # the virtual cable -- Play is for hearing the take yourself.
        buf = self._get_take(which)
        sd.play(buf, self.sample_rate, device=self.playback_device)

    def save(self, path: str, which: str = "live") -> None:
        buf = self._get_take(which)
        wavio.write_wav(path, self.sample_rate, buf)

    def take_bytes(self, which: str = "live") -> bytes:
        buf = self._get_take(which)
        return wavio.wav_bytes(self.sample_rate, buf)

    # -- stream callbacks ------------------------------------------------

    def _input_callback(self, indata, frames, time_info, status) -> None:  # noqa: D401
        try:
            self._drain_pending()

            mono = indata[:, 0] if indata.ndim > 1 else indata
            mono = np.asarray(mono, dtype=np.float32)

            if mono.size:
                dry64 = mono.astype(np.float64)
                self._dry_rms = float(np.sqrt(np.mean(dry64 ** 2)))
                self._dry_peak = float(np.max(np.abs(dry64)))
            else:
                self._dry_rms = 0.0
                self._dry_peak = 0.0

            if self.enabled:
                t = self._manual_intensity if self._manual_intensity is not None else self.ramp.current()
                if t != self._last_applied_intensity:
                    self.effect.set_intensity(t)
                    self._last_applied_intensity = t
                processed = self.effect.process(mono)
            else:
                processed = mono

            # Capture both takes under one lock: the live processed output
            # (the real-time effect result, replayed verbatim by Play) and
            # the dry mic (for optional re-render). The effect emits a
            # variable number of samples per block, so the live take is the
            # concatenation of exactly what went out.
            if self._recording:
                with self._record_lock:
                    if self._recording:
                        self._raw_chunks.append(mono.copy())
                        if processed.size:
                            self._live_chunks.append(np.asarray(processed, dtype=np.float32).copy())

            if processed.size:
                proc64 = processed.astype(np.float64)
                self._proc_rms = float(np.sqrt(np.mean(proc64 ** 2)))
                self._proc_peak = float(np.max(np.abs(proc64)))
            else:
                self._proc_rms = 0.0
                self._proc_peak = 0.0

            self.ring.write(processed)
            # Feed the live monitor the same processed audio (its own ring so
            # the two output streams don't consume each other's samples).
            if self.monitor_enabled and self.monitor_stream is not None:
                self.monitor_ring.write(processed)
        except Exception as exc:  # keep the audio thread alive
            self.last_error = str(exc)

    def _output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: D401
        try:
            mono = self.ring.read(frames)
            channels = outdata.shape[1]
            if channels == 1:
                outdata[:, 0] = mono
            else:
                outdata[:, :] = mono[:, None]  # broadcast mono across channels (no temp)
        except Exception as exc:
            self.last_error = str(exc)
            outdata.fill(0.0)
