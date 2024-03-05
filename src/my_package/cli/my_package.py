#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path
from typing import Union

from my_package.api import run
from my_package.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my-package",
        usage="%(prog)s config_file [options [args]]",
        epilog="_________________my-package___________________",
        prefix_chars="-",
        add_help=True,
        description=("my-package config_file --option"),
    )

    _ = parser.add_argument(
        "config_file",
        metavar="config_file",
        type=str,
        help="name of the file containing the my-package configuration.",
    )

    _ = parser.add_argument(
        "--option",
        action="store_true",
        help="example option.",
        default=False,
        required=False,
    )

    console_verbosity = parser.add_mutually_exclusive_group(required=False)

    _ = console_verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help=("console output will be quiet."),
        default=False,
    )

    _ = console_verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=("console output will be verbose."),
        default=False,
    )

    _ = parser.add_argument(
        "--log",
        action="store",
        type=str,
        help="name of log file. If specified, this will activate logging to file.",
        default=None,
        required=False,
    )

    _ = parser.add_argument(
        "--log-level",
        action="store",
        type=str,
        help="log level applied to logging to file.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        required=False,
    )

    return parser


def main():
    """Entry point for console script as configured in setup.cfg.

    Runs the command line interface and parses arguments and options entered on the console.
    """

    parser = _argparser()
    args = parser.parse_args()

    # Configure Logging
    # ..to console
    log_level_console: str = "WARNING"
    if any([args.quiet, args.verbose]):
        log_level_console = "ERROR" if args.quiet else log_level_console
        log_level_console = "INFO" if args.verbose else log_level_console
    # ..to file
    log_file: Union[Path, None] = Path(args.log) if args.log else None
    log_level_file: str = args.log_level
    configure_logging(log_level_console, log_file, log_level_file)

    config_file: Path = Path(args.config_file)
    option: bool = args.option

    # Check whether my-package config file exists
    if not config_file.is_file():
        logger.error(f"my-package.py: File {config_file} not found.")
        return

    logger.info(
        f"Start my-package.py with following arguments:\n"
        f"\t config_file: \t{config_file}\n"
        f"\t option: \t\t\t{option}\n"
    )

    # Invoke API
    run(
        config_file=config_file,
        option=option,
    )


if __name__ == "__main__":
    main()
