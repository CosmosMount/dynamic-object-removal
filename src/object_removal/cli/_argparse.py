from __future__ import annotations

import argparse


def parse_bool_arg(value: str) -> bool:
    """@brief Parse a human-friendly boolean CLI argument.

    @param value Input string such as `true`, `false`, `1`, `0`, `yes`, or `no`.
    @return The parsed boolean value.
    @raises argparse.ArgumentTypeError If the value is not a supported boolean literal.
    """
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "t", "yes", "y"):
        return True
    if normalized in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")
