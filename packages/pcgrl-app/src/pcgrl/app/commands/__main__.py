from __future__ import annotations

from pcgrl.app.commands.app import app

# Avoid re-executing the CLI when multiprocessing children import this module.
if __name__ == "__main__":
    app()
