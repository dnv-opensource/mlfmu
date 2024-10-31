#!/usr/bin/env python

import argparse
import logging
import textwrap
from pathlib import Path

from mlfmu.api import MlFmuCommand, run
from mlfmu.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _argparser() -> argparse.ArgumentParser:
    """
    Create and return an ArgumentParser object for parsing command line arguments.

    Returns
    -------
        argparse.ArgumentParser: The ArgumentParser object.
    """
    parser = argparse.ArgumentParser(
        prog="mlfmu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
                                    mlfmu is a command line tool to build FMUs from ONNX ML models.
                                    Check the README and docs for more info.
                                    You can also run `mlfmu <command> --help` for more info on a specific command."""),
        epilog=textwrap.dedent("""\
                               This tool utilizes cppfmu, source code is available at: https://github.com/viproma/cppfmu
                               _________________mlfmu___________________
                               """),
        prefix_chars="-",
        add_help=True,
    )

    common_args_parser = argparse.ArgumentParser(add_help=False)

    console_verbosity = common_args_parser.add_mutually_exclusive_group(required=False)

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

    _ = common_args_parser.add_argument(
        "-l",
        "--log",
        action="store",
        type=str,
        help=(
            "name of log file. If specified, this will activate logging to file. "
            "If not, it does not log to file, only console."
        ),
        default=None,
        required=False,
    )

    _ = common_args_parser.add_argument(
        "-ll",
        "--log-level",
        action="store",
        type=str,
        help="log level applied to logging to file. Default: WARNING.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        required=False,
    )

    # Create a sub parser for each command
    sub_parsers = parser.add_subparsers(
        dest="command",
        title="Available commands",
        metavar="command",
        required=True,
    )

    # Main command
    # build command to go from config to compiled fmu
    build_parser = sub_parsers.add_parser(
        MlFmuCommand.BUILD.value,
        help="Build FMU from interface and model files",
        parents=[common_args_parser],
        add_help=True,
    )

    # Add options for build command
    _ = build_parser.add_argument(
        "-i",
        "--interface-file",
        type=str,
        help="JSON file describing the FMU following schema",
    )
    _ = build_parser.add_argument(
        "-m",
        "--model-file",
        type=str,
        help="ONNX file containing the ML Model",
    )
    _ = build_parser.add_argument(
        "-f",
        "--fmu-path",
        type=str,
        help="Path to where the built FMU should be saved",
    )

    # Split the main build command into steps for customization
    # generate-code command to go from config to generated fmu source code
    code_generation_parser = sub_parsers.add_parser(
        MlFmuCommand.GENERATE.value,
        help="Generate FMU source code from interface and model files",
        parents=[common_args_parser],
        add_help=True,
    )

    # Add options for code generation command
    _ = code_generation_parser.add_argument(
        "--interface-file",
        type=str,
        help="json file describing the FMU following schema (e.g. interface.json).",
    )
    _ = code_generation_parser.add_argument(
        "--model-file",
        type=str,
        help="onnx file containing the ML Model (e.g. example.onnx).",
    )
    _ = code_generation_parser.add_argument(
        "--fmu-source-path",
        help=(
            "Path to where the generated FMU source code should be saved. "
            "Given path/to/folder the files can be found in path/to/folder/[FmuName]"
        ),
    )

    # build-code command to go from fmu source code to compiled fmu
    build_code_parser = sub_parsers.add_parser(
        MlFmuCommand.COMPILE.value,
        help="Build FMU from FMU source code",
        parents=[common_args_parser],
        add_help=True,
    )

    # Add option for fmu compilation
    _ = build_code_parser.add_argument(
        "--fmu-source-path",
        type=str,
        help=(
            "Path to the folder where the FMU source code is located. "
            "The folder needs to have the same name as the FMU. E.g. path/to/folder/[FmuName]"
        ),
    )
    _ = build_code_parser.add_argument(
        "--fmu-path",
        type=str,
        help="Path to where the built FMU should be saved.",
    )

    return parser


def main() -> None:
    """
    Entry point for console script as configured in setup.cfg.

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
    log_file: Path | None = Path(args.log) if args.log else None
    log_level_file: str = args.log_level
    configure_logging(log_level_console, log_file, log_level_file)
    logger.info("Logging to file: %s", log_file)

    command: MlFmuCommand | None = MlFmuCommand.from_string(args.command)

    if command is None:
        raise ValueError(
            f"The given command (={args.command}) does not match any of the existing commands "
            f"(={[command.value for command in MlFmuCommand]})."
        )

    interface_file = args.interface_file if "interface_file" in args else None
    model_file = args.model_file if "model_file" in args else None
    fmu_path = args.fmu_path if "fmu_path" in args else None
    source_folder = args.fmu_source_path if "fmu_source_path" in args else None

    # Invoke API
    try:
        run(
            command=command,
            interface_file=interface_file,
            model_file=model_file,
            fmu_path=fmu_path,
            source_folder=source_folder,
        )
    except Exception:
        logger.exception("Unhandled exception in run: %s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception in main: %s")
