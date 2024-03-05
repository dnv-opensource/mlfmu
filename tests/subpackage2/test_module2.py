# pyright: reportPrivateUsage=false
import contextlib

import pytest
from pytest import LogCaptureFixture

from mlfmu.subpackage2.module2 import str_to_int


@pytest.mark.parametrize(
    "str_in, int_expected",
    [
        ("1", 1),
        ("2", 2),
        ("-1", -1),
    ],
)
def test_str_to_int(str_in: str, int_expected: int):
    """Example of a test that uses the pytest.mark.parametrize decorator.

    Parameters
    ----------
    str_in : str
        the string passed in to str_to_int()
    int_expected : int
        the integer expected as return value from the call to str_to_int()
    """
    # Prepare
    # (nothing to prepare, as we use the pytest.mark.parametrize decorator
    #  to set the input and expected output.)
    # Execute
    int_out = str_to_int(str_in)
    # Assert
    assert isinstance(int_out, int)
    assert int_out == int_expected


def test_str_to_int_exception():
    # Prepare
    str_in: str = "3.0"
    # Execute
    with pytest.raises(ValueError):
        _ = str_to_int(str_in)


def test_str_to_int_exception_log_message(caplog: LogCaptureFixture):
    # Prepare
    str_in: str = "3.0"
    log_level_expected = "ERROR"
    log_message_expected = f"ValueError raised in function str_to_int() in subpackage2. input was: {str_in}"
    caplog.clear()
    # Execute
    with contextlib.suppress(Exception):
        _ = str_to_int(str_in)
    # Assert
    assert len(caplog.records) > 0
    assert caplog.records[0].levelname == log_level_expected
    assert caplog.records[0].message == log_message_expected
