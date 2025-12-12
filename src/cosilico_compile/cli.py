"""
Command-line interface for cosilico-compile.

Usage:
    cosilico-compile eitc -o eitc.js
    cosilico-compile eitc --typescript -o eitc.ts
"""

import argparse
import sys
from pathlib import Path

from .js_generator import generate_eitc_calculator, JSCodeGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cosilico-compile",
        description="Compile Cosilico .cos files to standalone JavaScript",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # EITC command
    eitc_parser = subparsers.add_parser(
        "eitc",
        help="Generate EITC calculator (26 USC 32)",
    )
    eitc_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    eitc_parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Tax year (default: 2025)",
    )
    eitc_parser.add_argument(
        "--typescript",
        action="store_true",
        help="Generate TypeScript instead of JavaScript",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    if args.command == "eitc":
        code = generate_eitc_calculator(tax_year=args.year)

        if args.output:
            args.output.write_text(code)
            print(f"Generated {args.output}", file=sys.stderr)
        else:
            print(code)

    elif args.command is None:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
