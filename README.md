# voicepranks — Minion Voice Changer

Real-time "Minion voice" microphone filter for macOS and Windows, written in
pure Python (numpy + sounddevice, hand-rolled DSP — no scipy/librosa).

It captures your physical microphone, pitch-shifts it up (chipmunk-style,
formants move up too) with a presence-EQ boost that ramps in over ~1.2s, and
routes the result to a virtual audio device so apps like Discord, Zoom, or
OBS can pick it up as their microphone input.

## How it works

```
physical mic -> capture -> pitch shift up + presence EQ (ramped in) -> virtual output device
```

The virtual device is [VB-CABLE](https://vb-audio.com/Cable/) ("CABLE Input")
on Windows, or [BlackHole](https://existential.audio/blackhole/) on macOS.
Install one of those first, then point your voice/video app's microphone
input at it.

## Download & run (no Python)

Not a developer? Grab a prebuilt bundle and double-click it — no Python, no
`pip`. You only need to install one small free driver first.

1. **Install the free virtual audio cable** (this is *not* bundled — it's a
   system driver): [BlackHole](https://existential.audio/blackhole/) on macOS,
   [VB-CABLE](https://vb-audio.com/Cable/) on Windows.
2. **Unzip and open the app.** It's unsigned, so the OS warns you the first time:
   - macOS: right-click `Minion Voice.app` → **Open** → **Open**.
   - Windows: on the SmartScreen prompt, **More info → Run anyway**.
3. In Discord / Zoom / OBS, set the **microphone** to the virtual cable
   (BlackHole / CABLE Output).

Full end-user steps ship inside each zip as `INSTALL.txt`. To *build* these
bundles yourself, see [`DISTRIBUTING.md`](DISTRIBUTING.md).

## Requirements (running from source)

- Python 3.9+
- `numpy`, `sounddevice` (see `requirements.txt`)
- A virtual audio cable driver (VB-CABLE on Windows, BlackHole on macOS)

## Install

```
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run the GUI

```
python -m minion_voice
```

Toggle the effect on/off, drag the intensity slider, and pick your input
device and the detected virtual output device.

## Audition the effect with no audio hardware

`selftest.py` reads/writes WAV files directly, so you can hear the effect
without any mic or virtual cable set up:

```
python -m minion_voice.selftest out.wav
python -m minion_voice.selftest in.wav out.wav --semitones 8 --eq-db 8
```

If no input WAV is given, a 2-second synthetic test tone (200 Hz with a
little vibrato) is generated and processed instead.

## Troubleshooting

**Blank / empty window on macOS.** Apple's *system* Python (`/usr/bin/python3`)
ships with Tk 8.5, which renders blank windows on modern macOS. Use a Python
built against a newer Tk:

```
brew install python-tk
python3 -m venv .venv          # uses Homebrew's python3 (Tk 8.6+/9.0)
source .venv/bin/activate
pip install -r requirements.txt
```

The app prints a warning at startup if it detects a broken Tk.

**BlackHole not detected right after installing it.** macOS only scans for
audio drivers when `coreaudiod` starts, so a fresh install may be invisible
until you restart the audio daemon (`sudo killall coreaudiod`) or reboot.

## Run the tests

```
pytest -q
```

## Project layout

```
minion_voice/
  dsp/        pitch shifting, peaking EQ, intensity ramp, combined effect
  audio/      device discovery + the real-time capture/output engine
  ui/         tkinter GUI
  selftest.py CLI WAV-file audition tool
```
