"""
Runners for validation: execute calculators on CPS microdata.

Provides functions to run both cosilico calculators and PolicyEngine-US
on the same household data for comparison.

Two modes:
1. Vectorized (fast): Use PE Microsimulation on full CPS, vectorized cosilico
2. Individual (slow): Build individual situations for each household
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from ..calculators import calculate_eitc, calculate_ctc, calculate_actc, calculate_snap_benefit
from ..calculators.eitc import EITC_PARAMS
from ..calculators.ctc import CTC_PARAMS
from ..calculators.snap import SNAP_PARAMS


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


# =============================================================================
# VECTORIZED RUNNERS (Fast - for full CPS)
# =============================================================================


def run_cosilico_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run cosilico calculators vectorized on full DataFrame.

    Much faster than row-by-row iteration.
    """
    n = len(df)

    # Vectorized EITC
    earned = df["earned_income"].values
    agi = df["agi"].values
    n_children = np.minimum(df["n_children"].values, 3)  # Cap at 3
    is_joint = df["is_joint"].values

    # Get parameters per household based on n_children
    credit_pct = np.array([EITC_PARAMS["credit_pct"][nc] / 100 for nc in n_children])
    phaseout_pct = np.array([EITC_PARAMS["phaseout_pct"][nc] / 100 for nc in n_children])
    earned_amount = np.array([EITC_PARAMS["earned_income_amount"][nc] for nc in n_children])
    phaseout_start = np.where(
        is_joint,
        np.array([EITC_PARAMS["phaseout_joint"][nc] for nc in n_children]),
        np.array([EITC_PARAMS["phaseout_single"][nc] for nc in n_children]),
    )

    credit_base = credit_pct * np.minimum(earned, earned_amount)
    income_for_phaseout = np.maximum(agi, earned)
    excess = np.maximum(0, income_for_phaseout - phaseout_start)
    phaseout = phaseout_pct * excess
    eitc = np.round(np.maximum(0, credit_base - phaseout)).astype(int)

    # Vectorized CTC
    n_qual_children = df["n_children"].values
    ctc_base = n_qual_children * CTC_PARAMS["credit_per_child"]
    ctc_phaseout_start = np.where(
        is_joint,
        CTC_PARAMS["phaseout_joint"],
        CTC_PARAMS["phaseout_single"],
    )
    ctc_excess = np.maximum(0, agi - ctc_phaseout_start)
    ctc_reduction = np.ceil(ctc_excess / 1000) * CTC_PARAMS["phaseout_rate"]
    ctc = np.round(np.maximum(0, ctc_base - ctc_reduction)).astype(int)

    # Vectorized ACTC
    earned_above = np.maximum(0, earned - CTC_PARAMS["refundable_threshold"])
    refundable_by_earnings = earned_above * CTC_PARAMS["refundable_rate"] / 100
    max_refundable = n_qual_children * CTC_PARAMS["refundable_max"]
    actc = np.round(np.minimum(refundable_by_earnings, max_refundable)).astype(int)

    # Vectorized SNAP
    hh_size = np.minimum(df["household_size"].values, 8)
    gross_monthly = df["gross_monthly_income"].values

    max_allotment = np.array([SNAP_PARAMS["max_allotment"][h] for h in hh_size])
    std_deduction = np.array([SNAP_PARAMS["standard_deduction"][h] for h in hh_size])
    net_income = np.maximum(0, gross_monthly - std_deduction)
    net_limit = np.array([SNAP_PARAMS["net_income_limit"][h] for h in hh_size])
    is_eligible = net_income <= net_limit

    reduction = net_income * SNAP_PARAMS["benefit_reduction_rate"] / 100
    benefit = np.maximum(0, max_allotment - reduction)
    min_benefit = np.where(hh_size <= 2, SNAP_PARAMS["min_benefit"], 0)
    snap = np.where(is_eligible, np.round(np.maximum(benefit, min_benefit)), 0).astype(int)

    return pd.DataFrame({
        "household_id": df["household_id"],
        "cosilico_eitc": eitc,
        "cosilico_ctc": ctc,
        "cosilico_actc": actc,
        "cosilico_snap": snap,
    })


def run_policyengine_microsim(year: int = 2025) -> pd.DataFrame:
    """
    Run PolicyEngine on full enhanced CPS using Microsimulation (vectorized).

    This is MUCH faster than individual simulations - runs in ~30 seconds
    instead of ~22 hours.

    Returns DataFrame with tax_unit_id and PE calculated values.
    """
    try:
        from policyengine_us import Microsimulation
    except ImportError:
        raise ImportError(
            "policyengine-us required for validation. "
            "Install with: pip install policyengine-us"
        )

    print("Loading PolicyEngine Microsimulation (this may take a minute)...")
    sim = Microsimulation()  # Uses enhanced CPS by default

    print("Calculating EITC...")
    eitc = sim.calculate("eitc", year)

    print("Calculating CTC...")
    ctc = sim.calculate("ctc", year)

    print("Calculating refundable CTC...")
    actc = sim.calculate("refundable_ctc", year)

    print("Calculating SNAP...")
    snap = sim.calculate("snap", year) / 12  # Monthly

    # Get tax unit IDs
    tax_unit_id = sim.calculate("tax_unit_id", year)

    # Aggregate to tax unit level (sum over members)
    # For tax credits, we take the tax unit value (same for all members)
    unique_tu = np.unique(tax_unit_id)

    results = []
    for tu_id in unique_tu:
        mask = tax_unit_id == tu_id
        results.append({
            "tax_unit_id": int(tu_id),
            "pe_eitc": float(eitc[mask].iloc[0]),
            "pe_ctc": float(ctc[mask].iloc[0]),
            "pe_actc": float(actc[mask].iloc[0]),
            "pe_snap": float(snap[mask].iloc[0]),
        })

    return pd.DataFrame(results)


def run_both_vectorized(year: int = 2025) -> pd.DataFrame:
    """
    Run full CPS validation using vectorized operations.

    Extracts inputs and PE outputs from Microsimulation, then runs
    cosilico on the same inputs for comparison.
    """
    try:
        from policyengine_us import Microsimulation
    except ImportError:
        raise ImportError(
            "policyengine-us required for validation. "
            "Install with: pip install policyengine-us"
        )

    print("Loading PolicyEngine Microsimulation...")
    sim = Microsimulation()

    # Get tax unit level data
    print("Extracting tax unit data...")
    tax_unit_id = sim.calculate("tax_unit_id", year)
    unique_tu = np.unique(tax_unit_id)
    print(f"Found {len(unique_tu):,} tax units")

    # Calculate PE outputs (tax unit level only - SNAP is SPM unit level)
    print("Calculating PolicyEngine outputs...")
    pe_eitc = sim.calculate("eitc", year)
    pe_ctc = sim.calculate("ctc", year)
    pe_actc = sim.calculate("refundable_ctc", year)
    # Note: SNAP is at SPM unit level, not tax unit - skip for now

    # Get inputs for cosilico
    print("Extracting input variables...")
    earned_income = sim.calculate("tax_unit_earned_income", year)
    agi = sim.calculate("adjusted_gross_income", year)
    n_children = sim.calculate("tax_unit_children", year)
    filing_status = sim.calculate("filing_status", year)
    tax_unit_size = sim.calculate("tax_unit_size", year)

    # Build tax-unit level DataFrame
    print("Building comparison dataset...")
    records = []
    for i, tu_id in enumerate(tqdm(unique_tu, desc="Tax units")):
        mask = tax_unit_id == tu_id
        idx = np.where(mask)[0][0]  # First person in tax unit

        is_joint = filing_status.values[idx] == "JOINT"
        n_child = int(n_children.values[idx])
        tu_size = int(tax_unit_size.values[idx])

        records.append({
            "household_id": int(tu_id),
            "earned_income": float(earned_income.values[idx]),
            "agi": float(agi.values[idx]),
            "n_children": n_child,
            "is_joint": is_joint,
            "household_size": tu_size,
            "gross_monthly_income": float(agi.values[idx]) / 12,
            "pe_eitc": float(pe_eitc.values[idx]),
            "pe_ctc": float(pe_ctc.values[idx]),
            "pe_actc": float(pe_actc.values[idx]),
            "pe_snap": np.nan,  # SNAP is at different entity level
        })

    df = pd.DataFrame(records)

    # Run cosilico vectorized
    print("\nRunning cosilico (vectorized)...")
    cosilico_results = run_cosilico_vectorized(df)

    # Merge
    merged = df.merge(cosilico_results, on="household_id")
    print(f"Comparison dataset: {len(merged):,} tax units")

    return merged
