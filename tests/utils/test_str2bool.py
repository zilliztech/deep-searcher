"""Tests for the str2bool CLI helper used by main.py's --enable-cors flag."""

import argparse

import pytest

from deepsearcher.utils.str2bool import str2bool


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("t", True),
        ("yes", True),
        ("y", True),
        ("1", True),
        (True, True),
        ("false", False),
        ("False", False),
        ("f", False),
        ("no", False),
        ("n", False),
        ("0", False),
        (False, False),
        (" true ", True),
    ],
)
def test_str2bool(value, expected):
    assert str2bool(value) is expected


@pytest.mark.parametrize("value", ["maybe", "", "2", "on", "off"])
def test_str2bool_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool(value)


def test_enable_cors_flag_parses_explicit_false():
    # Regression test for `python main.py --enable-cors false`, which argparse's
    # type=bool evaluated to True (bool("false")). main.py itself is not imported
    # here because importing it initializes all configured providers.
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-cors", type=str2bool, default=False)
    assert parser.parse_args(["--enable-cors", "false"]).enable_cors is False
    assert parser.parse_args(["--enable-cors", "true"]).enable_cors is True
    assert parser.parse_args([]).enable_cors is False
