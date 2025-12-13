"""
Runners for validation: execute calculators on CPS microdata.

Provides functions to run both cosilico calculators and PolicyEngine-US
on the same household data for comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from ..calculators import calculate_eitc, calculate_ctc, calculate_actc, calculate_snap_benefit


def run_cosilico(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """
    Run cosilico calculators on CPS household data.

    Args:
        df: DataFrame with household data (from load_cps_data)
        show_progress: Show progress bar

    Returns:
        DataFrame with household_id and calculated values
    """
    results = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Cosilico") if show_progress else df.iterrows()

    for _, row in iterator:
        # EITC
        eitc_result = calculate_eitc(
            earned_income=row["earned_income"],
            agi=row["agi"],
            n_children=row["n_children"],
            is_joint=row["is_joint"],
        )

        # CTC
        ctc_result = calculate_ctc(
            n_qualifying_children=row["n_children"],
            agi=row["agi"],
            is_joint=row["is_joint"],
        )

        # ACTC
        actc_result = calculate_actc(
            n_qualifying_children=row["n_children"],
            earned_income=row["earned_income"],
        )

        # SNAP
        snap_result = calculate_snap_benefit(
            household_size=row["household_size"],
            gross_income=row["gross_monthly_income"],
        )

        results.append({
            "household_id": row["household_id"],
            "cosilico_eitc": eitc_result.eitc,
            "cosilico_ctc": ctc_result.ctc,
            "cosilico_actc": actc_result.actc,
            "cosilico_snap": snap_result.benefit,
        })

    return pd.DataFrame(results)


def run_policyengine(df: pd.DataFrame, year: int = 2025, show_progress: bool = True) -> pd.DataFrame:
    """
    Run PolicyEngine-US on CPS household data.

    Args:
        df: DataFrame with household data (from load_cps_data)
        year: Tax year
        show_progress: Show progress bar

    Returns:
        DataFrame with household_id and PolicyEngine calculated values
    """
    try:
        from policyengine_us import Simulation
    except ImportError:
        raise ImportError(
            "policyengine-us required for validation. "
            "Install with: pip install policyengine-us"
        )

    results = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="PolicyEngine") if show_progress else df.iterrows()

    for _, row in iterator:
        try:
            pe_values = _run_single_pe_simulation(row, year)
            pe_values["household_id"] = row["household_id"]
            results.append(pe_values)
        except Exception as e:
            # Log error but continue with other households
            results.append({
                "household_id": row["household_id"],
                "pe_eitc": np.nan,
                "pe_ctc": np.nan,
                "pe_actc": np.nan,
                "pe_snap": np.nan,
                "pe_error": str(e),
            })

    return pd.DataFrame(results)


def _run_single_pe_simulation(row: pd.Series, year: int) -> Dict:
    """Run PolicyEngine simulation for a single household."""
    from policyengine_us import Simulation

    # Build people dict
    people = {
        "adult": {
            "age": {year: 30},
            "employment_income": {year: row["earned_income"]},
        }
    }

    members = ["adult"]

    # Add spouse if joint
    if row["is_joint"]:
        people["spouse"] = {
            "age": {year: 30},
            "employment_income": {year: 0},
        }
        members.append("spouse")

    # Add children
    for i in range(row["n_children"]):
        child_id = f"child_{i}"
        people[child_id] = {
            "age": {year: 5},
            "is_tax_unit_dependent": {year: True},
        }
        members.append(child_id)

    # Add additional household members for SNAP (beyond tax unit)
    extra_members = row["household_size"] - len(members)
    for i in range(extra_members):
        extra_id = f"extra_{i}"
        people[extra_id] = {
            "age": {year: 25},
        }
        members.append(extra_id)

    filing_status = "JOINT" if row["is_joint"] else "SINGLE"

    situation = {
        "people": people,
        "tax_units": {
            "tax_unit": {
                "members": members,
                "filing_status": {year: filing_status},
            }
        },
        "families": {"family": {"members": members}},
        "spm_units": {"spm_unit": {"members": members}},
        "households": {
            "household": {
                "members": members,
                "state_code": {year: row["state_code"]},
            }
        },
    }

    sim = Simulation(situation=situation)

    return {
        "pe_eitc": float(sim.calculate("eitc", year)[0]),
        "pe_ctc": float(sim.calculate("ctc", year)[0]),
        "pe_actc": float(sim.calculate("refundable_ctc", year)[0]),
        "pe_snap": float(sim.calculate("snap", year)[0]) / 12,  # Monthly
    }


def run_both(
    df: pd.DataFrame,
    year: int = 2025,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run both cosilico and PolicyEngine on the same data.

    Args:
        df: DataFrame with household data
        year: Tax year
        show_progress: Show progress bars

    Returns:
        Merged DataFrame with both sets of results
    """
    cosilico_results = run_cosilico(df, show_progress)
    pe_results = run_policyengine(df, year, show_progress)

    # Merge results
    merged = df.merge(cosilico_results, on="household_id")
    merged = merged.merge(pe_results, on="household_id")

    return merged
