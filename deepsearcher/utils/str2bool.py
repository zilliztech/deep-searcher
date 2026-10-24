"""Shared CLI helpers."""

import argparse


def str2bool(value):
    """Parse a CLI boolean flag that accepts explicit true/false values.

    ``argparse``'s built-in ``type=bool`` maps any non-empty string (including
    ``"false"`` and ``"no"``) to ``True``, which silently inverts the user's
    intent. This parser accepts the common truthy/falsy spellings instead and
    raises ``ArgumentTypeError`` on anything else.
    """
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")
