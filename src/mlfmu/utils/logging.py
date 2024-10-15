import logging
import sys
from pathlib import Path

__all__ = ["configure_logging"]

logger = logging.getLogger(__name__)


def configure_logging(
    log_level_console: str = "WARNING",
    log_file: Path | None = None,
    log_level_file: str = "WARNING",
):  # sourcery skip: extract-duplicate-method, extract-method
    """
    Configure the logging settings for the application.

    Args
    ----
        log_level_console (str): The log level for console output. Defaults to "WARNING".
        log_file (Union[Path, None]): The path to the log file. Defaults to None.
        log_level_file (str): The log level for file output. Defaults to "WARNING".

    Raises
    ------
        ValueError: If an invalid log level is provided for console or file.
    """

    log_level_console_numeric = getattr(logging, log_level_console.upper(), None)
    if not isinstance(log_level_console_numeric, int):
        raise ValueError(f"Invalid log level to console: {log_level_console_numeric}")

    log_level_file_numeric = getattr(logging, log_level_file.upper(), None)
    if not isinstance(log_level_file_numeric, int):
        raise ValueError(f"Invalid log level to file: {log_level_file_numeric}")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_console_numeric)
    console_formatter = logging.Formatter("%(levelname)-8s %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file.absolute()), "a")
        print(f"Logging to: {log_file.absolute()}")  # noqa: T201
        file_handler.setLevel(log_level_file_numeric)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return
