"""PyInstaller entry point for the Minion Voice GUI.

This lives at the repo root (rather than pointing PyInstaller at
``minion_voice/__main__.py``) on purpose: PyInstaller puts the entry script's
own directory on ``sys.path[0]``, so a script inside the package would make the
``import minion_voice`` line fail to resolve. A root-level launcher keeps the
package importable. It intentionally mirrors ``minion_voice/__main__.py``.
"""

from minion_voice.ui.app import run

if __name__ == "__main__":
    run()
