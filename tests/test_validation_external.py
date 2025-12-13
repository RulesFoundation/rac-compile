"""
Validation tests: Python implementations vs PolicyEngine (external oracle).

These tests validate our Python reference implementations against PolicyEngine-US,
which serves as the authoritative external calculator.

Two-layer validation strategy:
1. Our Python calculators vs PolicyEngine-US (this file)
2. Our compiled JS vs our Python calculators (test_validation_js.py)
"""

import pytest
from src.cosilico_compile.calculators import (
    calculate_eitc,
    calculate_ctc,
    calculate_actc,
    calculate_snap_benefit,
)

# Try to import PolicyEngine - skip tests if not available
try:
    from policyengine_us import Simulation
    HAS_POLICYENGINE = True
except ImportError:
    HAS_POLICYENGINE = False


def run_policyengine_tax(
    earned_income: float = 0,
    n_children: int = 0,
    is_joint: bool = False,
    year: int = 2025,
) -> dict:
    """Run PolicyEngine-US simulation for tax credits."""
    # Build people
    people = {
        "adult": {
            "age": {year: 30},
            "employment_income": {year: earned_income},
        }
    }

    tax_unit_members = ["adult"]

    # Add spouse if joint
    if is_joint:
        people["spouse"] = {
            "age": {year: 30},
            "employment_income": {year: 0},
        }
        tax_unit_members.append("spouse")

    # Add children
    for i in range(n_children):
        child_id = f"child_{i}"
        people[child_id] = {
            "age": {year: 5},
            "is_tax_unit_dependent": {year: True},
        }
        tax_unit_members.append(child_id)

    filing_status = "JOINT" if is_joint else "SINGLE"

    situation = {
        "people": people,
        "tax_units": {
            "tax_unit": {
                "members": tax_unit_members,
                "filing_status": {year: filing_status},
            }
        },
        "families": {"family": {"members": tax_unit_members}},
        "spm_units": {"spm_unit": {"members": tax_unit_members}},
        "households": {
            "household": {
                "members": tax_unit_members,
                "state_code": {year: "CA"},
            }
        },
    }

    sim = Simulation(situation=situation)

    return {
        "eitc": float(sim.calculate("eitc", year)[0]),
        "ctc": float(sim.calculate("ctc", year)[0]),
        "refundable_ctc": float(sim.calculate("refundable_ctc", year)[0]),
    }


def run_policyengine_snap(
    household_size: int = 1,
    gross_income: float = 0,
    year: int = 2025,
) -> dict:
    """Run PolicyEngine-US simulation for SNAP benefits."""
    # Build people
    people = {}
    member_ids = []
    for i in range(household_size):
        person_id = f"person_{i}"
        member_ids.append(person_id)
        people[person_id] = {
            "age": {year: 30 if i == 0 else 25},
            "employment_income": {year: gross_income * 12 if i == 0 else 0},  # Annual
        }

    situation = {
        "people": people,
        "tax_units": {"tax_unit": {"members": member_ids}},
        "families": {"family": {"members": member_ids}},
        "spm_units": {"spm_unit": {"members": member_ids}},
        "households": {
            "household": {
                "members": member_ids,
                "state_code": {year: "CA"},
            }
        },
    }

    sim = Simulation(situation=situation)

    # SNAP is calculated monthly in PE
    snap_annual = float(sim.calculate("snap", year)[0])
    return {
        "snap": snap_annual / 12,  # Convert to monthly
    }


@pytest.mark.skipif(not HAS_POLICYENGINE, reason="PolicyEngine-US not installed")
class TestEITCAgainstPolicyEngine:
    """Validate our EITC calculator against PolicyEngine-US."""

    # Test cases covering different scenarios
    # Format: (earned_income, n_children, is_joint)
    CASES = [
        (0, 0, False),
        (8000, 0, False),
        (15000, 0, False),
        (10000, 1, False),
        (20000, 1, False),
        (15000, 2, False),
        (30000, 2, False),
        (20000, 3, False),
        (40000, 3, True),
        (50000, 2, True),
    ]

    @pytest.mark.parametrize("earned_income,n_children,is_joint", CASES)
    def test_eitc_matches_policyengine(self, earned_income, n_children, is_joint):
        """Our EITC matches PolicyEngine-US calculation."""
        # Our calculation
        our_result = calculate_eitc(
            earned_income=earned_income,
            agi=earned_income,  # Simplified: AGI = earned income
            n_children=n_children,
            is_joint=is_joint,
        )

        # PolicyEngine calculation
        pe_result = run_policyengine_tax(
            earned_income=earned_income,
            n_children=n_children,
            is_joint=is_joint,
        )

        # Allow $1 tolerance for rounding
        assert abs(our_result.eitc - pe_result["eitc"]) <= 1, (
            f"EITC mismatch: ours={our_result.eitc}, PE={pe_result['eitc']} "
            f"for earned_income={earned_income}, n_children={n_children}, joint={is_joint}"
        )


@pytest.mark.skipif(not HAS_POLICYENGINE, reason="PolicyEngine-US not installed")
class TestCTCAgainstPolicyEngine:
    """Validate our CTC calculator against PolicyEngine-US."""

    CASES = [
        (1, 50000, False),
        (2, 100000, True),
        (3, 150000, False),
        (2, 250000, False),  # In phaseout
        (1, 300000, True),
    ]

    @pytest.mark.parametrize("n_children,agi,is_joint", CASES)
    def test_ctc_matches_policyengine(self, n_children, agi, is_joint):
        """Our CTC matches PolicyEngine-US calculation."""
        our_result = calculate_ctc(
            n_qualifying_children=n_children,
            agi=agi,
            is_joint=is_joint,
        )

        pe_result = run_policyengine_tax(
            earned_income=agi,
            n_children=n_children,
            is_joint=is_joint,
        )

        # Allow $50 tolerance (PE may have more complex logic)
        assert abs(our_result.ctc - pe_result["ctc"]) <= 50, (
            f"CTC mismatch: ours={our_result.ctc}, PE={pe_result['ctc']}"
        )


@pytest.mark.skipif(not HAS_POLICYENGINE, reason="PolicyEngine-US not installed")
class TestSNAPAgainstPolicyEngine:
    """Validate our SNAP calculator against PolicyEngine-US."""

    CASES = [
        (1, 0),
        (1, 500),
        (2, 1000),
        (4, 2000),
    ]

    @pytest.mark.parametrize("household_size,gross_income", CASES)
    def test_snap_matches_policyengine(self, household_size, gross_income):
        """Our SNAP matches PolicyEngine-US calculation."""
        our_result = calculate_snap_benefit(
            household_size=household_size,
            gross_income=gross_income,
        )

        pe_result = run_policyengine_snap(
            household_size=household_size,
            gross_income=gross_income,
        )

        # SNAP has many deductions we don't model (earned income deduction,
        # shelter deduction, dependent care) - allow larger tolerance.
        # This test is more about catching major bugs than exact matching.
        assert abs(our_result.benefit - pe_result["snap"]) <= 150, (
            f"SNAP mismatch: ours={our_result.benefit}, PE={pe_result['snap']}"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions (no external dependency)."""

    def test_eitc_zero_income(self):
        """EITC is 0 with no earned income."""
        result = calculate_eitc(earned_income=0, agi=0, n_children=2)
        assert result.eitc == 0

    def test_eitc_negative_not_possible(self):
        """EITC can never be negative."""
        result = calculate_eitc(earned_income=100000, agi=100000, n_children=0)
        assert result.eitc >= 0

    def test_eitc_four_children_same_as_three(self):
        """4+ children uses same parameters as 3 (capped)."""
        result3 = calculate_eitc(
            earned_income=20000, agi=20000, n_children=3, is_joint=True
        )
        result4 = calculate_eitc(
            earned_income=20000, agi=20000, n_children=4, is_joint=True
        )
        assert result3.eitc == result4.eitc

    def test_snap_large_household_uses_8(self):
        """Households > 8 use the 8-person values."""
        result8 = calculate_snap_benefit(household_size=8, gross_income=0)
        result10 = calculate_snap_benefit(household_size=10, gross_income=0)
        assert result10.benefit == result8.benefit

    def test_ctc_below_phaseout(self):
        """CTC at full value below phaseout threshold."""
        result = calculate_ctc(n_qualifying_children=2, agi=100000, is_joint=False)
        assert result.ctc == 4400  # $2,200 * 2 for TY2025

    def test_actc_below_threshold(self):
        """ACTC is 0 when earned income below $2,500."""
        result = calculate_actc(n_qualifying_children=2, earned_income=2000)
        assert result.actc == 0
