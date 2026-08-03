# Distributing Minion Voice (standalone builds)

How to produce the double-clickable bundles handed to non-technical users.
End users need **no Python** — just the zip plus a free virtual audio cable.

## What ships

- **macOS:** `dist/minion-voice-macos.zip` → contains `Minion Voice.app` + `INSTALL.txt`
- **Windows:** `dist/minion-voice-windows.zip` → contains `minion-voice\` (with
  `minion-voice.exe`) + `INSTALL.txt`

Builds are **unsigned** (casual distribution). Users click past the OS
"unidentified developer" warning — this is documented in `INSTALL.txt`.

## The hard limit: the virtual audio cable is NOT bundled

Minion Voice routes processed audio into a virtual audio driver — **BlackHole**
(macOS) / **VB-CABLE** (Windows). That's a system/CoreAudio driver with its own
installer and **cannot** be packaged inside our executable. Every end user
installs it once themselves (covered in Step 1 of `INSTALL.txt`).

## Building

PyInstaller **cannot cross-compile** — build each OS on that OS.

### macOS (`dist/Minion Voice.app`)
Use a **Homebrew Python** (Tk 8.6+), not Apple's `/usr/bin/python3` (Tk 8.5 →
blank window). The build script asserts this and refuses otherwise.

```
brew install python-tk
PYTHON=$(brew --prefix)/bin/python3 ./build-macos.sh
```

Output: `dist/Minion Voice.app` and the zipped `dist/minion-voice-macos.zip`.
Apple Silicon and Intel are separate builds — a bundle built on one arch runs on
that arch (plus Rosetta for Intel-on-Apple-Silicon). Build on each, or on Apple
Silicon target a universal Python if you need one binary for both.

### Windows (`dist\minion-voice\`)
On a Windows machine (or CI runner), from PowerShell:

```
.\build-windows.ps1
```

Output: `dist\minion-voice\minion-voice.exe` and `dist\minion-voice-windows.zip`.

## How it works under the hood

- Entry point: `launcher.py` (mirrors `python -m minion_voice`).
- `minion-voice.spec` is one cross-platform spec (branches on `sys.platform`):
  - `collect_dynamic_libs("sounddevice")` bundles PortAudio.
  - `console=False` — GUI, no terminal window.
  - macOS `BUNDLE` declares `NSMicrophoneUsageDescription` so mic access is
    granted (without it macOS denies the input stream).
- Build outputs (`build/`, `dist/`) are gitignored.

## Verifying a build

1. Launch the bundle; the GUI must appear **non-blank** and list devices.
2. With the cable installed, toggle the effect on and confirm audio reaches the
   virtual device (select it as mic in another app). On macOS confirm the
   microphone permission prompt appears.
3. Best proof: run the zip on a machine **without Python** installed.
