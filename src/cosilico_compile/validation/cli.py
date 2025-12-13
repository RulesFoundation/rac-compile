"""
CLI for running validation pipeline.

Usage:
    python -m cosilico_compile.validation.cli [options]
    cosilico-validate [options]  # if installed
"""

import argparse
import sys
from pathlib import Path

from .comparator import validate, ComparisonConfig


def main():
    parser = argparse.ArgumentParser(
        description="Validate cosilico calculators against PolicyEngine-US on CPS microdata"
    )

    parser.add_argument(
        "--source",
        choices=["policyengine", "csv"],
        default="policyengine",
        help="Data source (default: policyengine)",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to CSV file (required if source=csv)",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Tax year (default: 2025)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size for faster validation (default: all households)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results",
    )

    # Tolerance overrides
    parser.add_argument(
        "--eitc-tolerance",
        type=float,
        default=1.0,
        help="EITC tolerance in dollars (default: 1)",
    )

    parser.add_argument(
        "--ctc-tolerance",
        type=float,
        default=1.0,
        help="CTC tolerance in dollars (default: 1)",
    )

    parser.add_argument(
        "--snap-tolerance",
        type=float,
        default=50.0,
        help="SNAP tolerance in dollars (default: 50)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == "csv" and not args.csv_path:
        parser.error("--csv-path required when source=csv")

    # Build config
    config = ComparisonConfig(
        eitc_tolerance=args.eitc_tolerance,
        ctc_tolerance=args.ctc_tolerance,
        actc_tolerance=args.ctc_tolerance,  # Same as CTC
        snap_tolerance=args.snap_tolerance,
    )

    # Run validation
    try:
        results = validate(
            source=args.source,
            year=args.year,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            config=config,
            csv_path=args.csv_path,
        )

        # Exit with error if match rates are too low
        min_match_rate = min(results.match_rates.values()) if results.match_rates else 0
        if min_match_rate < 90:
            print(f"\nWarning: Lowest match rate is {min_match_rate:.1f}%")
            sys.exit(1)

    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo run validation, install policyengine-us:")
        print("  pip install policyengine-us")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
