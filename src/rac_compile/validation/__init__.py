"""
Validation module: Compare RAC calculators against PolicyEngine-US on CPS microdata.

This follows the policyengine-taxsim pattern:
- No hand-built test cases
- Validation across full enhanced CPS (~200k households)
- Tolerance-based comparison with detailed mismatch reporting
"""

from .comparator import Comparator, ComparisonConfig, ComparisonResults
from .cps_loader import CPSHousehold, load_cps_data
from .runners import run_policyengine, run_rac

__all__ = [
    "Comparator",
    "ComparisonConfig",
    "ComparisonResults",
    "run_rac",
    "run_policyengine",
    "load_cps_data",
    "CPSHousehold",
]
