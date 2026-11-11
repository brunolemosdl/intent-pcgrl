from __future__ import annotations

import logging

from pcgrl.core.helpers.log import get_logger, setup_logging


def test_setup_logging_returns_logger() -> None:
    """Verify setup logging returns logger."""
    logger = setup_logging("info")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "pcgrl"


def test_setup_logging_level_info() -> None:
    """Verify setup logging level info."""
    logger = setup_logging("info")
    assert logger.level == logging.INFO


def test_setup_logging_level_debug() -> None:
    """Verify setup logging level debug."""
    logger = setup_logging("debug")
    assert logger.level == logging.DEBUG


def test_setup_logging_level_warning() -> None:
    """Verify setup logging level warning."""
    logger = setup_logging("warning")
    assert logger.level == logging.WARNING


def test_setup_logging_stream_handler() -> None:
    """Verify setup logging clears existing handlers."""
    logger = setup_logging("info")
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_setup_logging_clears_existing_handlers() -> None:
    """Verify setup logging clears existing handlers."""
    setup_logging("info")
    setup_logging("info")
    logger = setup_logging("info")
    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1


def test_setup_logging_with_file(tmp_path) -> None:
    """Verify setup logging with file."""
    log_file = tmp_path / "logs" / "run.log"
    logger = setup_logging("info", log_file=log_file)
    assert log_file.is_file() or log_file.parent.is_dir()
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    for h in logger.handlers:
        h.close()


def test_get_logger_returns_same_instance() -> None:
    """Verify get logger returns same instance."""
    setup_logging("info")
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_setup_logging_propagate_false() -> None:
    """Verify setup logging propagate false."""
    logger = setup_logging("info")
    assert not logger.propagate
