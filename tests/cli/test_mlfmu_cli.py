# pyright: reportPrivateUsage=false
import sys
from argparse import ArgumentError
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import pytest
from pytest import MonkeyPatch

from mlfmu.api import MlFmuCommand
from mlfmu.cli import mlfmu
from mlfmu.cli.mlfmu import _argparser, main

# *****Test commandline interface (CLI)************************************************************


@dataclass()
class CliArgs:
    # Expected default values for the CLI arguments when mlfmu gets called via the commandline
    quiet: bool = False
    verbose: bool = False
    log: Union[str, None] = None
    log_level: str = field(default_factory=lambda: "WARNING")
    command: str = ""


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([], ArgumentError),
        (["asd"], ArgumentError),
        (["build", "-q"], CliArgs(quiet=True, command="build")),
        (["build", "--quiet"], CliArgs(quiet=True, command="build")),
        (["build", "-v"], CliArgs(verbose=True, command="build")),
        (["build", "--verbose"], CliArgs(verbose=True, command="build")),
        (["build", "-qv"], ArgumentError),
        (["build", "--log", "logFile"], CliArgs(log="logFile", command="build")),
        (["build", "--log"], ArgumentError),
        (["build", "--log-level", "INFO"], CliArgs(log_level="INFO", command="build")),
        (["build", "--log-level"], ArgumentError),
        (["build", "-o"], ArgumentError),
    ],
)
def test_cli(
    inputs: List[str],
    expected: Union[CliArgs, type],
    monkeypatch: MonkeyPatch,
):
    # sourcery skip: no-conditionals-in-tests
    # sourcery skip: no-loop-in-tests
    # Prepare
    monkeypatch.setattr(sys, "argv", ["mlfmu"] + inputs)
    parser = _argparser()
    # Execute
    if isinstance(expected, CliArgs):
        args_expected: CliArgs = expected
        args = parser.parse_args()
        # Assert args
        print(args)
        print(args_expected)
        for key in args_expected.__dataclass_fields__:
            assert args.__getattribute__(key) == args_expected.__getattribute__(key)
    elif issubclass(expected, Exception):
        exception: type = expected
        # Assert that expected exception is raised
        with pytest.raises((exception, SystemExit)):
            args = parser.parse_args()
    else:
        raise AssertionError()


# *****Ensure the CLI correctly configures logging*************************************************


@dataclass()
class ConfigureLoggingArgs:
    # Values that main() is expected to pass to ConfigureLogging() by default when configuring the logging
    log_level_console: str = field(default_factory=lambda: "WARNING")
    log_file: Union[Path, None] = None
    log_level_file: str = field(default_factory=lambda: "WARNING")


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([], ArgumentError),
        (["build"], ConfigureLoggingArgs()),
        (["build", "-q"], ConfigureLoggingArgs(log_level_console="ERROR")),
        (
            ["build", "--quiet"],
            ConfigureLoggingArgs(log_level_console="ERROR"),
        ),
        (["build", "-v"], ConfigureLoggingArgs(log_level_console="INFO")),
        (
            ["build", "--verbose"],
            ConfigureLoggingArgs(log_level_console="INFO"),
        ),
        (["build", "-qv"], ArgumentError),
        (
            ["build", "--log", "logFile"],
            ConfigureLoggingArgs(log_file=Path("logFile")),
        ),
        (["build", "--log"], ArgumentError),
        (
            ["build", "--log-level", "INFO"],
            ConfigureLoggingArgs(log_level_file="INFO"),
        ),
        (["build", "--log-level"], ArgumentError),
    ],
)
def test_logging_configuration(
    inputs: List[str],
    expected: Union[ConfigureLoggingArgs, type],
    monkeypatch: MonkeyPatch,
):
    # sourcery skip: no-conditionals-in-tests
    # sourcery skip: no-loop-in-tests
    # Prepare
    monkeypatch.setattr(sys, "argv", ["mlfmu"] + inputs)
    args: ConfigureLoggingArgs = ConfigureLoggingArgs()

    def fake_configure_logging(
        log_level_console: str,
        log_file: Union[Path, None],
        log_level_file: str,
    ):
        args.log_level_console = log_level_console
        args.log_file = log_file
        args.log_level_file = log_level_file

    def fake_run(
        command: str,
        interface_file: Optional[str],
        model_file: Optional[str],
        fmu_path: Optional[str],
        source_folder: Optional[str],
    ):
        pass

    monkeypatch.setattr(mlfmu, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(mlfmu, "run", fake_run)
    # Execute
    if isinstance(expected, ConfigureLoggingArgs):
        args_expected: ConfigureLoggingArgs = expected
        main()
        # Assert args
        for key in args_expected.__dataclass_fields__:
            assert args.__getattribute__(key) == args_expected.__getattribute__(key)
    elif issubclass(expected, Exception):
        exception: type = expected
        # Assert that expected exception is raised
        with pytest.raises((exception, SystemExit)):
            main()
    else:
        raise AssertionError()


# *****Ensure the CLI correctly invokes the API****************************************************


@dataclass()
class ApiArgs:
    # Values that main() is expected to pass to run() by default when invoking the API
    command: Optional[MlFmuCommand] = None
    interface_file: Optional[str] = None
    model_file: Optional[str] = None
    fmu_path: Optional[str] = None
    source_folder: Optional[str] = None


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([], ArgumentError),
        (["build"], ApiArgs()),
    ],
)
def test_api_invokation(
    inputs: List[str],
    expected: Union[ApiArgs, type],
    monkeypatch: MonkeyPatch,
):
    # sourcery skip: no-conditionals-in-tests
    # sourcery skip: no-loop-in-tests
    # Prepare
    monkeypatch.setattr(sys, "argv", ["mlfmu"] + inputs)
    args: ApiArgs = ApiArgs()

    def fake_run(
        command: str,
        interface_file: Optional[str],
        model_file: Optional[str],
        fmu_path: Optional[str],
        source_folder: Optional[str],
    ):
        args.command = MlFmuCommand.from_string(command)
        args.interface_file = interface_file
        args.model_file = model_file
        args.fmu_path = fmu_path
        args.source_folder = source_folder

    monkeypatch.setattr(mlfmu, "run", fake_run)
    # Execute
    if isinstance(expected, ApiArgs):
        args_expected: ApiArgs = expected
        main()
        # Assert args
        for key in args_expected.__dataclass_fields__:
            assert args.__getattribute__(key) == args_expected.__getattribute__(key)
    elif issubclass(expected, Exception):
        exception: type = expected
        # Assert that expected exception is raised
        with pytest.raises((exception, SystemExit)):
            main()
    else:
        raise AssertionError()
