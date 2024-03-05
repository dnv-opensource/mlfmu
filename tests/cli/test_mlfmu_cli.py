# pyright: reportPrivateUsage=false
import sys
from argparse import ArgumentError
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import pytest
from pytest import MonkeyPatch

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
    config_file: Union[str, None] = field(
        default_factory=lambda: "test_config_file"
    )  # noqa: N815
    option: bool = False


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([], ArgumentError),
        (["test_config_file"], CliArgs()),
        (["test_config_file", "-q"], CliArgs(quiet=True)),
        (["test_config_file", "--quiet"], CliArgs(quiet=True)),
        (["test_config_file", "-v"], CliArgs(verbose=True)),
        (["test_config_file", "--verbose"], CliArgs(verbose=True)),
        (["test_config_file", "-qv"], ArgumentError),
        (["test_config_file", "--log", "logFile"], CliArgs(log="logFile")),
        (["test_config_file", "--log"], ArgumentError),
        (["test_config_file", "--log-level", "INFO"], CliArgs(log_level="INFO")),
        (["test_config_file", "--log-level"], ArgumentError),
        (["test_config_file", "--option"], CliArgs(option=True)),
        (["test_config_file", "-o"], ArgumentError),
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
        (["test_config_file"], ConfigureLoggingArgs()),
        (["test_config_file", "-q"], ConfigureLoggingArgs(log_level_console="ERROR")),
        (
            ["test_config_file", "--quiet"],
            ConfigureLoggingArgs(log_level_console="ERROR"),
        ),
        (["test_config_file", "-v"], ConfigureLoggingArgs(log_level_console="INFO")),
        (
            ["test_config_file", "--verbose"],
            ConfigureLoggingArgs(log_level_console="INFO"),
        ),
        (["test_config_file", "-qv"], ArgumentError),
        (
            ["test_config_file", "--log", "logFile"],
            ConfigureLoggingArgs(log_file=Path("logFile")),
        ),
        (["test_config_file", "--log"], ArgumentError),
        (
            ["test_config_file", "--log-level", "INFO"],
            ConfigureLoggingArgs(log_level_file="INFO"),
        ),
        (["test_config_file", "--log-level"], ArgumentError),
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
        config_file: Path,
        option: bool,
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
    config_file: Path = field(default_factory=lambda: Path("test_config_file"))
    option: bool = False


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([], ArgumentError),
        (["test_config_file"], ApiArgs()),
        (["test_config_file", "--option"], ApiArgs(option=True)),
        (["test_config_file", "-o"], ArgumentError),
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
        config_file: Path,
        option: bool = False,
    ):
        args.config_file = config_file
        args.option = option

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
