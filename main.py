from __future__ import annotations

from pcgrl.app.commands.app import app


def main() -> None:
    """Run the project command-line interface.

    Description:
        Entry point for ``python main.py`` and console scripts that delegate
        to the Typer application in ``pcgrl.app.commands``.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    app()


if __name__ == "__main__":
    main()
