#!/usr/bin/env bash
# Build the macOS standalone bundle (dist/VoicePranks.app) and zip it.
#
# Must run on a Mac. Use a Homebrew Python (Tk 8.6+), NOT Apple's system
# /usr/bin/python3 (Tk 8.5), which produces a blank GUI window.
#
#   ./build-macos.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv-build"
# The .app name is owned solely by voicepranks.spec; this script discovers
# whatever bundle PyInstaller produced rather than re-hardcoding the name.

# Guard against the Apple-system-Python / Tk 8.5 blank-window trap before we
# spend time building an unusable bundle.
"$PYTHON" - <<'PY'
import sys, tkinter
if tkinter.TkVersion < 8.6:
    sys.exit(
        f"Refusing to build: Tk {tkinter.TkVersion} renders blank windows on modern "
        "macOS. Use a Homebrew Python:\n"
        "  brew install python-tk\n"
        "  PYTHON=$(brew --prefix)/bin/python3 ./build-macos.sh"
    )
print(f"Tk {tkinter.TkVersion} OK")
PY

echo "==> Creating clean build venv ($VENV)"
rm -rf "$VENV"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt -r requirements-build.txt

echo "==> Running PyInstaller"
rm -rf build dist  # clean slate so the *.app glob below is unambiguous
pyinstaller voicepranks.spec --noconfirm

# Locate the bundle PyInstaller produced (name comes from the spec).
shopt -s nullglob
apps=(dist/*.app)
shopt -u nullglob
if [ "${#apps[@]}" -ne 1 ]; then
  echo "Expected exactly one dist/*.app, found ${#apps[@]}" >&2
  exit 1
fi

echo "==> Staging zip (app + INSTALL.txt)"
STAGE="dist/voicepranks-macos"
mkdir -p "$STAGE"
mv "${apps[0]}" "$STAGE/"  # move (not copy) — the zip is the deliverable
cp INSTALL.txt "$STAGE/"
# ditto preserves the .app bundle structure and resource forks in the zip.
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$STAGE.zip"

echo "==> Done: $STAGE.zip"
