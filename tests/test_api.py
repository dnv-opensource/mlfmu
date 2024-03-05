# pyright: reportPrivateUsage=false
from pathlib import Path

import pytest
from pytest import LogCaptureFixture

from mlfmu.api import MlFmuProcess, run


def test_file_not_found_exception():
    # Prepare
    config_file = Path("this_file_does_not_exist")
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        _ = run(config_file)


def test_run():
    # Prepare
    config_file = Path("test_config_file")
    # Execute
    run(config_file=config_file)
    # Assert
    # (nothin to assert. Assertion is that no exception is thrown.)


def test_run_with_option(caplog: LogCaptureFixture):
    # Prepare
    config_file = Path("test_config_file")
    log_level_expected = "INFO"
    log_message_expected = "option is True. mlfmu process will do something differently."
    caplog.clear()
    # Execute
    run(config_file=config_file, option=True)
    # Assert
    assert len(caplog.records) > 0
    assert caplog.records[0].levelname == log_level_expected
    assert caplog.records[0].message == log_message_expected


class TestMlFmuProcess:
    def test_init(self):
        # Prepare
        config_file = Path("test_config_file.json")
        # Execute
        process = MlFmuProcess(config_file=config_file)
        # Assert
        assert process.config_file is config_file
        assert process.max_number_of_runs == 3
        assert process.run_number == 0
        assert process.terminate is False

    def test_init_with_empty_config_file(self):
        # sourcery skip: class-extract-method
        # Prepare
        config_file = Path("test_config_file_empty.json")
        # Execute
        process = MlFmuProcess(config_file=config_file)
        # Assert
        assert process.config_file is config_file
        assert process.max_number_of_runs == 1
        assert process.run_number == 0
        assert process.terminate is False


# @TODO: To be implemented
@pytest.mark.skip(reason="To be implemented")
def test_example_skip():
    """Example of a test skipped because it is not yet implemented."""
    pass
