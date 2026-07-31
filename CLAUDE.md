# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`voicepranks` — a Windows tray app (WPF) that sits between a physical microphone and
a virtual microphone (VB-Audio VB-CABLE), gradually pitch-shifting the user's voice
toward a "minion" register the longer the effect is toggled on. It works at the OS
audio-device level rather than as a plugin for one specific call app, so any app that
selects "CABLE Input (VB-Audio Virtual Cable)" as its microphone hears the effect.

Signal path: `Physical Mic → WASAPI capture → pitch/EQ shift (BASS_FX) → WASAPI render → VB-CABLE "CABLE Input"`.

Note: the repository root also contains `izzybennett.com/` and `izzybennett.com-worktrees/`
— these are separate, unrelated git repositories (a personal website) nested in this
directory, not part of this project. Ignore them unless explicitly asked to work there.

## Build & run

Windows + .NET 8 SDK only (WPF and WASAPI are Windows-only APIs) — this cannot be
built or run on macOS/Linux.

```
dotnet build MinionVoiceChanger.sln
dotnet run --project src/App
```

Requires VB-Audio VB-CABLE installed (https://vb-audio.com/Cable/) so the
"CABLE Input" render device exists; `VoiceEngine.Initialize()` throws with a
user-facing message if it isn't found.

There are no automated tests in this repo currently.

## Untested / unverified code — read before touching AudioEngine

This codebase was developed on a machine without Windows/.NET, so **none of the
`AudioEngine` code has been compiled or run against real hardware.** Several spots
are explicitly called out in code comments as the most likely first-build breakage
points — check these before assuming a bug elsewhere:

- `MinionEffect.cs`: `PeakEQParameters` field names/namespace have shifted across
  ManagedBass versions — if this doesn't compile, check `ManagedBass.Dx8`/
  `ManagedBass.Effects` for the current type.
- `DeviceRouter.Start`: assumes the capture device's WASAPI shared-mode format is
  IEEE float and throws `NotSupportedException` if not — a PCM16→float path isn't
  implemented.
- Real-time latency/choppiness: first things to try are `BufferDuration` in
  `DeviceRouter` and the tempo/sequence overlap attributes in `MinionEffect.Init`.
- `DeviceRouter` auto-inserts a `MediaFoundationResampler` when the mic's format and
  VB-CABLE's mix format differ — logic is present but unexercised against real
  hardware.

## Architecture

Two projects, referenced as `AudioEngine → App` (one-directional; AudioEngine has no
UI dependency and doesn't reference WPF):

- **`src/AudioEngine`** — device I/O and DSP, framework-agnostic:
  - `VoiceEngine` — the facade the UI talks to. Owns a `DeviceRouter`, a
    `MinionEffect`, and a `RampController`; exposes `Initialize()` / `Toggle()` and
    an `IntensityChanged` event. Devices are opened once for the app's lifetime;
    `Toggle()` only flips a bool so on/off is instant with no device
    re-negotiation. `IntensityChanged` fires from a background `Timer` thread —
    subscribers must marshal to the UI thread themselves (see
    `MainWindow.OnIntensityChanged`'s `Dispatcher.Invoke`).
  - `DeviceRouter` — opens the physical mic (WASAPI capture, `Role.Communications`)
    and VB-CABLE's "CABLE Input" (WASAPI render) via NAudio, and pipes audio between
    them. When `ProcessingEnabled` is false, capture audio is passed straight to the
    render buffer unmodified (not muted) — this is what makes toggle-off transparent.
  - `MinionEffect` — the actual DSP, built on BASS_FX: a decode-only BASS push
    stream feeds a `BassFx.TempoCreate` tempo/pitch handle with a `PeakEQ` DX8
    effect attached. `SetIntensity(0..1)` scales both pitch shift (semitones) and EQ
    gain linearly. BASS is initialized once process-wide in "no sound" device mode
    (device 0) — it's used purely for DSP, never for playback; NAudio/WASAPI
    handles all actual audio I/O.
  - `RampController` — pure elapsed-time → 0..1 intensity math, no I/O. Resets to 0
    whenever stopped.
- **`src/App`** — WPF UI: a single `MainWindow` (toggle button, ramp-duration
  textbox, live intensity bar) plus a `System.Windows.Forms.NotifyIcon` tray icon.
  Closing the window hides it instead of exiting (`Closing` handler cancels unless
  `_isExiting` was set via the tray menu's "Exit"), so the app keeps running in the
  tray. `Settings` persists `RampMinutes` as JSON to
  `%AppData%\MinionVoiceChanger\settings.json`, loaded/saved defensively (falls back
  to defaults on any I/O/JSON error rather than blocking startup).

## Dependencies

`NAudio`, `ManagedBass` + `ManagedBass.Fx` (BASS_FX is free for non-commercial use
only — confirm licensing before distributing further), and `System.Windows.Forms`
(Windows Desktop SDK, used solely for the tray icon).
