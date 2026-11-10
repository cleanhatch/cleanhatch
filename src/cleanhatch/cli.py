"""Command-line entry point for Cleanhatch."""

import argparse
from collections.abc import Sequence

from cleanhatch import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleanhatch",
        description=(
            "Cleanhatch is in early development. "
            "This release provides version information only; "
            "data validation is not implemented."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
