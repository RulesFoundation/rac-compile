"""
Python reference implementations of tax/benefit calculators.

These serve as the ground truth for validating:
1. Against external sources (IRS, USDA, etc.)
2. Against compiled outputs (JS, WASM, etc.)
"""

from .eitc import calculate_eitc, EITC_PARAMS
from .ctc import calculate_ctc, calculate_actc, CTC_PARAMS
from .snap import calculate_snap_benefit, calculate_snap_eligible, SNAP_PARAMS

__all__ = [
    "calculate_eitc",
    "EITC_PARAMS",
    "calculate_ctc",
    "calculate_actc",
    "CTC_PARAMS",
    "calculate_snap_benefit",
    "calculate_snap_eligible",
    "SNAP_PARAMS",
]
