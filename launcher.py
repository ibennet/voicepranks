"""PyInstaller entry point for the VoicePranks GUI.

This lives at the repo root (rather than pointing PyInstaller at
``voicepranks/__main__.py``) on purpose: PyInstaller puts the entry script's
own directory on ``sys.path[0]``, so a script inside the package would make the
``import voicepranks`` line fail to resolve. A root-level launcher keeps the
package importable. It intentionally mirrors ``voicepranks/__main__.py``.
"""

from voicepranks.ui.app import run

if __name__ == "__main__":
    run()
