"""
Command-line interface for rac-compile.

Usage:
    rac-compile compile input.rac -o output.js
    rac-compile compile input.cos -o output.js
    rac-compile eitc -o eitc.js
"""

import argparse
import sys
from pathlib import Path

from .js_generator import generate_eitc_calculator, JSCodeGenerator
from .python_generator import generate_eitc_calculator as generate_eitc_calculator_py
from .parser import parse_rac


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="rac-compile",
        description="Compile RAC .rac/.cos files to standalone JavaScript or Python",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compile command
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a .rac or .cos file to JavaScript",
    )
    compile_parser.add_argument(
        "input",
        type=Path,
        help="Input .rac or .cos file",
    )
    compile_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )

    # EITC command (pre-built)
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
        "--python",
        action="store_true",
        help="Generate Python code instead of JavaScript (default: JavaScript)",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    args = parser.parse_args()

    if args.command == "compile":
        if not args.input.exists():
            print(f"Error: {args.input} not found", file=sys.stderr)
            sys.exit(1)

        content = args.input.read_text()
        rac_file = parse_rac(content)
        gen = rac_file.to_js_generator()
        code = gen.generate()

        if args.output:
            args.output.write_text(code)
            print(f"Compiled {args.input} -> {args.output}", file=sys.stderr)
        else:
            print(code)

    elif args.command == "eitc":
        if hasattr(args, 'python') and args.python:
            code = generate_eitc_calculator_py(tax_year=args.year)
            lang = "Python"
        else:
            code = generate_eitc_calculator(tax_year=args.year)
            lang = "JavaScript"

        if args.output:
            args.output.write_text(code)
            print(f"Generated {lang} EITC calculator -> {args.output}", file=sys.stderr)
        else:
            print(code)

    elif args.command is None:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
