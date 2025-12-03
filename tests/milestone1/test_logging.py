"""
Issue 1.2.3: Tests for `modeling_studio/utils/logging.py`

Tests:
- test_setup_logging_returns_logger - Returns `logging.Logger` instance
- test_setup_logging_level_info - Logger level is `INFO` by default
- test_setup_logging_level_debug - Can set level to `DEBUG`
- test_setup_logging_with_file - Creates log file when path specified
- test_setup_logging_rich_handler - Uses `RichHandler` when `use_rich=True`
- test_setup_logging_no_rich - Uses `StreamHandler` when `use_rich=False`
- test_get_logger_named - `get_logger("test")` returns logger named `modeling_studio.test`
- test_get_logger_unnamed - `get_logger()` returns logger named `modeling_studio`
"""

import logging


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_returns_logger(self):
        """Returns `logging.Logger` instance."""
        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging()

        assert isinstance(logger, logging.Logger)

    def test_setup_logging_level_info(self):
        """Logger level is `INFO` by default."""
        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging()

        assert logger.level == logging.INFO

    def test_setup_logging_level_debug(self):
        """Can set level to `DEBUG`."""
        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging(log_level="DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logging_level_warning(self):
        """Can set level to `WARNING`."""
        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging(log_level="WARNING")

        assert logger.level == logging.WARNING

    def test_setup_logging_level_error(self):
        """Can set level to `ERROR`."""
        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging(log_level="ERROR")

        assert logger.level == logging.ERROR

    def test_setup_logging_with_file(self, tmp_path):
        """Creates log file when path specified."""
        from modeling_studio.utils.logging import setup_logging

        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file)

        # Log a message
        logger.info("Test message")

        assert log_file.exists()

    def test_setup_logging_file_creates_parent_dirs(self, tmp_path):
        """Log file path creates parent directories."""
        from modeling_studio.utils.logging import setup_logging

        log_file = tmp_path / "nested" / "dir" / "test.log"
        logger = setup_logging(log_file=log_file)

        logger.info("Test message")

        assert log_file.exists()

    def test_setup_logging_rich_handler(self):
        """Uses `RichHandler` when `use_rich=True`."""
        from rich.logging import RichHandler

        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging(use_rich=True)

        # Check that at least one handler is RichHandler
        has_rich = any(isinstance(h, RichHandler) for h in logger.handlers)
        assert has_rich

    def test_setup_logging_no_rich(self):
        """Uses `StreamHandler` when `use_rich=False`."""
        from rich.logging import RichHandler

        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging(use_rich=False)

        # Should not have RichHandler
        has_rich = any(isinstance(h, RichHandler) for h in logger.handlers)
        assert not has_rich

        # Should have StreamHandler
        has_stream = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RichHandler)
            for h in logger.handlers
        )
        assert has_stream

    def test_setup_logging_clears_handlers(self):
        """setup_logging clears existing handlers."""
        from modeling_studio.utils.logging import setup_logging

        # Call twice
        logger1 = setup_logging()
        initial_count = len(logger1.handlers)

        logger2 = setup_logging()

        # Should not accumulate handlers
        assert len(logger2.handlers) <= initial_count

    def test_setup_logging_logger_name(self):
        """Logger has correct name."""
        from modeling_studio.utils.logging import setup_logging

        logger = setup_logging()

        assert logger.name == "modeling_studio"


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_named(self):
        """get_logger('test') returns logger named 'modeling_studio.test'."""
        from modeling_studio.utils.logging import get_logger

        logger = get_logger("test")

        assert logger.name == "modeling_studio.test"

    def test_get_logger_unnamed(self):
        """get_logger() returns logger named 'modeling_studio'."""
        from modeling_studio.utils.logging import get_logger

        logger = get_logger()

        assert logger.name == "modeling_studio"

    def test_get_logger_returns_logger_instance(self):
        """get_logger returns logging.Logger instance."""
        from modeling_studio.utils.logging import get_logger

        logger = get_logger("any_name")

        assert isinstance(logger, logging.Logger)

    def test_get_logger_nested_name(self):
        """get_logger with nested name works correctly."""
        from modeling_studio.utils.logging import get_logger

        logger = get_logger("module.submodule")

        assert logger.name == "modeling_studio.module.submodule"

    def test_get_logger_same_name_same_instance(self):
        """Calling get_logger with same name returns same logger."""
        from modeling_studio.utils.logging import get_logger

        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")

        assert logger1 is logger2
