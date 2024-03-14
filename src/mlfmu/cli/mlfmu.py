#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path
from typing import Union

from mlfmu.api import MlFmuCommand, run
from mlfmu.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlfmu",
        # usage="%(prog)s [options [args]]",
        epilog="_________________mlfmu___________________",
        prefix_chars="-",
        add_help=True,
        # description=("mlfmu config_file --option"),
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

    # Create a sub parser for each command
    sub_parsers = parser.add_subparsers(dest="command", title="Available commands", metavar="command", required=True)

    # Main command
    # build command to go from config to compiled fmu
    build_parser = sub_parsers.add_parser(MlFmuCommand.BUILD.value, help="Build FMU from interface and model files")

    # Add options for build command
    _ = build_parser.add_argument("--interface-file", type=str, help="JSON file describing the FMU following schema")
    _ = build_parser.add_argument("--model-file", type=str, help="ONNX file containing the ML Model")
    _ = build_parser.add_argument("--fmu-path", type=str, help="Path to where the built FMU should be saved")

    # Split the main build command into steps for customization
    # generate-code command to go from config to generated fmu source code
    code_generation_parser = sub_parsers.add_parser(
        MlFmuCommand.GENERATE.value, help="Generate FMU source code from interface and model files"
    )

    # Add options for code generation command
    _ = code_generation_parser.add_argument(
        "--interface-file", type=str, help="json file describing the FMU following schema"
    )
    _ = code_generation_parser.add_argument("--model-file", type=str, help="onnx file containing the ML Model")
    _ = code_generation_parser.add_argument(
        "--fmu-source-path",
        help="Path to where the generated FMU source code should be saved. Given path/to/folder the files can be found in path/to/folder/[FmuName]",
    )

    # build-code command to go from fmu source code to compiled fmu
    build_code_parser = sub_parsers.add_parser(MlFmuCommand.COMPILE.value, help="Build FMU from FMU source code")

    # Add option for fmu compilation
    _ = build_code_parser.add_argument(
        "--fmu-source-path",
        type=str,
        help="Path to the folder where the FMU source code is located. The folder needs to have the same name as the FMU. E.g. path/to/folder/[FmuName]",
    )
    _ = build_code_parser.add_argument(
        "--fmu-path", type=str, help="Path to where the where the built FMU should be saved"
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

    command: MlFmuCommand = args.command
    interface_file = args.interface_file if "interface_file" in args else None
    model_file = args.model_file if "model_file" in args else None
    fmu_path = args.fmu_path if "fmu_path" in args else None
    source_folder = args.fmu_source_path if "fmu_source_path" in args else None

    # Invoke API
    run(
        command=command,
        logger=logger,
        interface_file=interface_file,
        model_file=model_file,
        fmu_path=fmu_path,
        source_folder=source_folder,
    )


if __name__ == "__main__":
    main()
