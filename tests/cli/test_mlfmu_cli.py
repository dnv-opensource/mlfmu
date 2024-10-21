# pyright: reportPrivateUsage=false
import sys
from argparse import ArgumentError
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mlfmu.api import MlFmuCommand
from mlfmu.cli import mlfmu
from mlfmu.cli.mlfmu import _argparser, main

# *****Test commandline interface (CLI)************************************************************


@dataclass()
class CliArgs:
    # Expected default values for the CLI arguments when mlfmu gets called via the commandline
    quiet: bool = False
    verbose: bool = False
    log: str | None = None
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
    inputs: list[str],
    expected: CliArgs | type,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test the command-line interface (CLI) of the 'mlfmu' program.

    Args
    ----
        inputs (List[str]): A list of input arguments to be passed to the CLI.
        expected (Union[CliArgs, type]): The expected output of the CLI.
            It can be either an instance of the `CliArgs` class or a subclass of `Exception`.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture that allows patching of objects at runtime.

    Raises
    ------
        AssertionError: If the `expected` argument is neither an instance of `CliArgs`
            nor a subclass of `Exception`.
    """

    # sourcery skip: no-conditionals-in-tests
    # sourcery skip: no-loop-in-tests

    # Prepare
    monkeypatch.setattr(sys, "argv", ["mlfmu", *inputs])
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
        raise AssertionError


# *****Ensure the CLI correctly configures logging*************************************************


@dataclass()
class ConfigureLoggingArgs:
    # Values that main() is expected to pass to ConfigureLogging() by default when configuring the logging
    log_level_console: str = field(default_factory=lambda: "WARNING")
    log_file: Path | None = None
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
    inputs: list[str],
    expected: ConfigureLoggingArgs | type,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test the logging configuration of the `main` function in the `mlfmu` module.

    Args
    ----
        inputs (List[str]): The list of input arguments to be passed to the `main` function.
        expected (Union[ConfigureLoggingArgs, type]): The expected output of the `main` function.
        It can be an instance of `ConfigureLoggingArgs` or a subclass of `Exception`.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture provided by pytest.

    Raises
    ----
        AssertionError: If the `expected` argument is neither an instance of `ConfigureLoggingArgs`
        nor a subclass of `Exception`.
    """

    # sourcery skip: no-conditionals-in-tests
    # sourcery skip: no-loop-in-tests

    # Prepare
    monkeypatch.setattr(sys, "argv", ["mlfmu", *inputs])
    args: ConfigureLoggingArgs = ConfigureLoggingArgs()

    def fake_configure_logging(
        log_level_console: str,
        log_file: Path | None,
        log_level_file: str,
    ):
        args.log_level_console = log_level_console
        args.log_file = log_file
        args.log_level_file = log_level_file

    def fake_run(
        command: str,
        interface_file: str | None,
        model_file: str | None,
        fmu_path: str | None,
        source_folder: str | None,
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
        raise AssertionError


# *****Ensure the CLI correctly invokes the API****************************************************


@dataclass()
class ApiArgs:
    # Values that main() is expected to pass to run() by default when invoking the API
    command: MlFmuCommand | None = None
    interface_file: str | None = None
    model_file: str | None = None
    fmu_path: str | None = None
    source_folder: str | None = None


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([], ArgumentError),
        (["build"], ApiArgs()),
    ],
)
def test_api_invokation(
    inputs: list[str],
    expected: ApiArgs | type,
    monkeypatch: pytest.MonkeyPatch,
):
    # sourcery skip: no-conditionals-in-tests
    # sourcery skip: no-loop-in-tests
    """
    Test the invocation of the API function.

    Args
    ----
        inputs (List[str]): The list of input arguments.
        expected (Union[ApiArgs, type]): The expected output, either an instance of ApiArgs or an exception type.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch object for patching sys.argv.

    Raises
    ----
        AssertionError: If the expected output is neither an instance of ApiArgs nor an exception type.
    """

    # Prepare
    monkeypatch.setattr(sys, "argv", ["mlfmu", *inputs])
    args: ApiArgs = ApiArgs()

    def fake_run(
        command: str,
        interface_file: str | None,
        model_file: str | None,
        fmu_path: str | None,
        source_folder: str | None,
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
        raise AssertionError
