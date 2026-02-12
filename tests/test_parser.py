"""
Tests for RAC DSL parser.

TDD: Tests define the DSL syntax we want to support.
Covers both legacy .cos syntax and unified .rac syntax.
"""

import subprocess
from pathlib import Path

import pytest
from src.rac_compile.parser import (
    parse_cos,
    parse_rac,
    CosFile,
    RacFile,
    SourceBlock,
    VariableBlock,
    TemporalEntry,
)


# ============================================================
# Legacy .cos syntax tests (backward compatibility)
# ============================================================


class TestParseSource:
    """Tests for source block parsing."""

    def test_parse_source_block(self):
        """Can parse source block with lawarchive reference."""
        cos = """
source {
  lawarchive: us/statute/26/32/2025-01-01
  citation: "26 USC 32"
  accessed: 2025-12-12
}
"""
        result = parse_cos(cos)
        assert result.source is not None
        assert result.source.lawarchive == "us/statute/26/32/2025-01-01"
        assert result.source.citation == "26 USC 32"
        assert result.source.accessed == "2025-12-12"

    def test_source_block_optional(self):
        """Source block is optional."""
        cos = """
variable foo {
  formula { return 0 }
}
"""
        result = parse_cos(cos)
        assert result.source is None


class TestParseParameters:
    """Tests for parameters block parsing."""

    def test_parse_parameters_block(self):
        """Can parse parameters with references."""
        cos = """
parameters {
  credit_pct: statute/26/32/b/1/credit_pct
  phaseout_pct: statute/26/32/b/1/phaseout_pct
}
"""
        result = parse_cos(cos)
        assert "credit_pct" in result.parameters
        assert result.parameters["credit_pct"].source == "statute/26/32/b/1/credit_pct"

    def test_parse_parameters_with_comments(self):
        """Comments are ignored in parameters block."""
        cos = """
parameters {
  # This is a comment
  rate: some/path  # inline comment
}
"""
        result = parse_cos(cos)
        assert "rate" in result.parameters
        assert result.parameters["rate"].source == "some/path"

    def test_parse_parameter_with_values(self):
        """Can parse parameter block with inline values."""
        cos = """
parameter credit_pct {
  source: "26 USC 32(b)(1)"
  values {
    0: 7.65
    1: 34
    2: 40
    3: 45
  }
}
"""
        result = parse_cos(cos)
        assert "credit_pct" in result.parameters
        param = result.parameters["credit_pct"]
        assert param.source == "26 USC 32(b)(1)"
        assert param.values == {0: 7.65, 1: 34, 2: 40, 3: 45}

    def test_parse_mixed_parameters(self):
        """Can mix parameter blocks and simple parameters."""
        cos = """
parameters {
  simple_rate: some/path
}

parameter credit_pct {
  source: "26 USC 32(b)(1)"
  values {
    0: 7.65
    1: 34
  }
}
"""
        result = parse_cos(cos)
        assert "simple_rate" in result.parameters
        assert "credit_pct" in result.parameters
        assert result.parameters["credit_pct"].values == {0: 7.65, 1: 34}


class TestParseVariable:
    """Tests for variable block parsing."""

    def test_parse_variable_metadata(self):
        """Can parse variable metadata."""
        cos = """
variable eitc {
  entity TaxUnit
  period Year
  dtype Money
  label "Earned Income Tax Credit"

  formula {
    return 0
  }
}
"""
        result = parse_cos(cos)
        assert len(result.variables) == 1
        var = result.variables[0]
        assert var.name == "eitc"
        assert var.entity == "TaxUnit"
        assert var.period == "Year"
        assert var.dtype == "Money"
        assert var.label == "Earned Income Tax Credit"

    def test_parse_variable_formula(self):
        """Can parse formula block."""
        cos = """
variable tax {
  formula {
    let rate = 0.2
    return income * rate
  }
}
"""
        result = parse_cos(cos)
        var = result.variables[0]
        assert "let rate = 0.2" in var.formula
        assert "return income * rate" in var.formula

    def test_parse_multiple_variables(self):
        """Can parse multiple variables."""
        cos = """
variable a {
  formula { return 1 }
}

variable b {
  formula { return 2 }
}
"""
        result = parse_cos(cos)
        assert len(result.variables) == 2
        assert result.variables[0].name == "a"
        assert result.variables[1].name == "b"


class TestParseComments:
    """Tests for comment handling."""

    def test_line_comments(self):
        """Line comments starting with # are ignored."""
        cos = """
# This is a file comment
variable foo {
  # This describes the formula
  formula { return 0 }
}
"""
        result = parse_cos(cos)
        assert len(result.variables) == 1

    def test_formula_comments_preserved(self):
        """Comments inside formula are preserved (for JS output)."""
        cos = """
variable foo {
  formula {
    # 32(a)(1): Credit base
    let base = income * rate
    return base
  }
}
"""
        result = parse_cos(cos)
        assert "# 32(a)(1)" in result.variables[0].formula


class TestFormulaConversion:
    """Tests for formula DSL to JS conversion."""

    def test_converts_min_to_math_min(self):
        """min() is converted to Math.min()."""
        cos = """
variable x {
  formula { return min(a, b) }
}
"""
        result = parse_cos(cos)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "Math.min(a, b)" in code

    def test_converts_max_to_math_max(self):
        """max() is converted to Math.max()."""
        cos = """
variable x {
  formula { return max(a, b) }
}
"""
        result = parse_cos(cos)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "Math.max(a, b)" in code

    def test_converts_round_to_math_round(self):
        """round() is converted to Math.round()."""
        cos = """
variable x {
  formula { return round(income) }
}
"""
        result = parse_cos(cos)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "Math.round(income)" in code

    def test_converts_parameter_references(self):
        """Parameter references are converted to PARAMS.name[]."""
        cos = """
parameters {
  rate: test/path
}

variable x {
  formula { return rate[0] * income }
}
"""
        result = parse_cos(cos)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "PARAMS.rate[0]" in code

    def test_nested_math_functions(self):
        """Nested math functions work correctly."""
        cos = """
variable x {
  formula { return max(0, round(min(a, b))) }
}
"""
        result = parse_cos(cos)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "Math.max(0, Math.round(Math.min(a, b)))" in code


class TestFullFile:
    """Tests for complete .cos file parsing."""

    def test_parse_complete_file(self):
        """Can parse a complete .cos file."""
        cos = """
# 26 USC 32 - Earned Income Tax Credit

source {
  lawarchive: us/statute/26/32/2025-01-01
  citation: "26 USC 32"
  accessed: 2025-12-12
}

parameters {
  credit_pct: statute/26/32/b/1/credit_pct
  earned_income_amount: guidance/irs/rp-24-40/eitc/earned_income_amount
}

variable eitc {
  entity TaxUnit
  period Year
  dtype Money
  label "Earned Income Tax Credit"

  formula {
    let credit_base = credit_pct[n_children] * min(earned_income, earned_income_amount[n_children])
    return max(0, credit_base - phaseout)
  }
}
"""
        result = parse_cos(cos)
        assert result.source.citation == "26 USC 32"
        assert "credit_pct" in result.parameters
        assert len(result.variables) == 1
        assert result.variables[0].name == "eitc"

    def test_cos_file_to_js_generator(self):
        """CosFile can be converted to JSCodeGenerator."""
        cos = """
source {
  citation: "Test"
  accessed: 2025-01-01
}

parameters {
  rate: test/path
}

variable tax {
  label "Tax"
  formula {
    return income * 0.2
  }
}
"""
        result = parse_cos(cos)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "function calculate(" in code


class TestExampleFiles:
    """Tests for example .cos files."""

    def test_eitc_example_parses(self):
        """examples/eitc.cos parses correctly with parameter values."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.cos"
        if not eitc_path.exists():
            pytest.skip("examples/eitc.cos not found")

        content = eitc_path.read_text()
        result = parse_cos(content)

        assert result.source is not None
        assert result.source.citation == "26 USC 32"
        assert len(result.parameters) == 5
        assert "credit_pct" in result.parameters
        # Check parameter has actual values
        credit_pct = result.parameters["credit_pct"]
        assert credit_pct.source == "26 USC 32(b)(1)"
        assert credit_pct.values == {0: 7.65, 1: 34.0, 2: 40.0, 3: 45.0}
        assert len(result.variables) == 1
        assert result.variables[0].name == "eitc"

    def test_eitc_example_compiles_to_valid_js(self):
        """examples/eitc.cos compiles to syntactically valid JS."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.cos"
        if not eitc_path.exists():
            pytest.skip("examples/eitc.cos not found")

        content = eitc_path.read_text()
        result = parse_cos(content)
        gen = result.to_js_generator()
        code = gen.generate()

        # Check JS syntax with Node.js (use --input-type=module for ESM)
        proc = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=code,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"

    def test_simple_tax_example_compiles(self):
        """examples/simple_tax.cos compiles to valid JS."""
        simple_path = Path(__file__).parent.parent / "examples" / "simple_tax.cos"
        if not simple_path.exists():
            pytest.skip("examples/simple_tax.cos not found")

        content = simple_path.read_text()
        result = parse_cos(content)
        gen = result.to_js_generator()
        code = gen.generate()

        # Check JS syntax with Node.js (use --input-type=module for ESM)
        proc = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=code,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"

    def test_ctc_example_parses(self):
        """examples/ctc.cos parses correctly with parameter values."""
        ctc_path = Path(__file__).parent.parent / "examples" / "ctc.cos"
        if not ctc_path.exists():
            pytest.skip("examples/ctc.cos not found")

        content = ctc_path.read_text()
        result = parse_cos(content)

        assert result.source is not None
        assert result.source.citation == "26 USC 24"
        assert len(result.parameters) == 7
        assert "credit_per_child" in result.parameters
        assert result.parameters["credit_per_child"].values == {0: 2200.0}  # TY2025
        assert len(result.variables) == 2
        assert result.variables[0].name == "ctc"
        assert result.variables[1].name == "actc"

    def test_ctc_example_compiles_to_valid_js(self):
        """examples/ctc.cos compiles to syntactically valid JS."""
        ctc_path = Path(__file__).parent.parent / "examples" / "ctc.cos"
        if not ctc_path.exists():
            pytest.skip("examples/ctc.cos not found")

        content = ctc_path.read_text()
        result = parse_cos(content)
        gen = result.to_js_generator()
        code = gen.generate()

        # Check JS syntax with Node.js (use --input-type=module for ESM)
        proc = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=code,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"

    def test_snap_example_parses(self):
        """examples/snap.cos parses correctly with parameter values."""
        snap_path = Path(__file__).parent.parent / "examples" / "snap.cos"
        if not snap_path.exists():
            pytest.skip("examples/snap.cos not found")

        content = snap_path.read_text()
        result = parse_cos(content)

        assert result.source is not None
        assert result.source.citation == "7 USC 2017"
        assert len(result.parameters) == 5
        assert "max_allotment" in result.parameters
        assert result.parameters["max_allotment"].values[1] == 292.0
        assert len(result.variables) == 2
        assert result.variables[0].name == "snap_eligible"
        assert result.variables[1].name == "snap_benefit"

    def test_snap_example_compiles_to_valid_js(self):
        """examples/snap.cos compiles to syntactically valid JS."""
        snap_path = Path(__file__).parent.parent / "examples" / "snap.cos"
        if not snap_path.exists():
            pytest.skip("examples/snap.cos not found")

        content = snap_path.read_text()
        result = parse_cos(content)
        gen = result.to_js_generator()
        code = gen.generate()

        # Check JS syntax with Node.js (use --input-type=module for ESM)
        proc = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=code,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"


# ============================================================
# Unified .rac syntax tests
# ============================================================


class TestUnifiedStatuteText:
    """Tests for top-level triple-quoted statute text."""

    def test_parse_statute_text(self):
        """Can parse top-level triple-quoted statute text."""
        rac = '''
"""
In the case of an individual, there shall be imposed
a tax equal to 3.8 percent of the lesser of net investment
income or excess MAGI over the threshold amount.
"""
'''
        result = parse_rac(rac)
        assert result.statute_text is not None
        assert "3.8 percent" in result.statute_text
        assert "net investment" in result.statute_text

    def test_statute_text_optional(self):
        """Statute text is optional."""
        rac = """
rate:
  from 2024-01-01: 0.30
"""
        result = parse_rac(rac)
        assert result.statute_text is None


class TestUnifiedParameterDef:
    """Tests for unified parameter definitions (name: with from dates)."""

    def test_parse_scalar_parameter(self):
        """Can parse parameter with scalar temporal values."""
        rac = """
niit_rate:
  source: "26 USC 1411"
  from 2013-01-01: 0.038
"""
        result = parse_rac(rac)
        assert "niit_rate" in result.parameters
        param = result.parameters["niit_rate"]
        assert param.source == "26 USC 1411"
        assert len(param.temporal) == 1
        assert param.temporal[0].from_date == "2013-01-01"
        assert param.temporal[0].value == 0.038

    def test_parse_multiple_temporal_values(self):
        """Can parse parameter with multiple temporal entries."""
        rac = """
threshold:
  source: "Rev. Proc. 2024-40"
  from 2024-01-01: 250000
  from 2023-01-01: 220000
  from 2022-01-01: 200000
"""
        result = parse_rac(rac)
        assert "threshold" in result.parameters
        param = result.parameters["threshold"]
        assert len(param.temporal) == 3
        assert param.temporal[0].value == 250000
        assert param.temporal[1].value == 220000
        assert param.temporal[2].value == 200000

    def test_parameter_values_in_values_dict(self):
        """Temporal scalar values are also stored in the values dict."""
        rac = """
rate:
  from 2024-01-01: 7.65
  from 2023-01-01: 7.65
"""
        result = parse_rac(rac)
        param = result.parameters["rate"]
        assert param.values == {0: 7.65, 1: 7.65}

    def test_parameter_with_description(self):
        """Can parse parameter with description attribute."""
        rac = """
contribution_rate:
  description: "Household contribution as share of net income"
  unit: rate
  source: "USDA FNS"
  from 2024-01-01: 0.30
"""
        result = parse_rac(rac)
        param = result.parameters["contribution_rate"]
        assert param.description == "Household contribution as share of net income"
        assert param.unit == "rate"
        assert param.source == "USDA FNS"


class TestUnifiedVariableDef:
    """Tests for unified variable definitions (name: with entity/period/dtype)."""

    def test_parse_variable_with_temporal_formula(self):
        """Can parse variable with formula under from-date."""
        rac = """
niit:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2013-01-01:
    magi = agi + foreign_earned_income_exclusion
    threshold = 200000
    excess = max(0, magi - threshold)
    return min(net_investment_income, excess) * 0.038
"""
        result = parse_rac(rac)
        assert len(result.variables) == 1
        var = result.variables[0]
        assert var.name == "niit"
        assert var.entity == "TaxUnit"
        assert var.period == "Year"
        assert var.dtype == "Money"
        assert "0.038" in var.formula
        assert "max(0, magi - threshold)" in var.formula

    def test_variable_type_inference(self):
        """Definitions with entity/period/dtype are variables, others are parameters."""
        rac = """
rate:
  from 2024-01-01: 0.038

tax:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2024-01-01:
    return income * 0.038
"""
        result = parse_rac(rac)
        assert "rate" in result.parameters
        assert len(result.variables) == 1
        assert result.variables[0].name == "tax"

    def test_variable_with_label(self):
        """Can parse variable with label attribute."""
        rac = """
eitc:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Earned Income Tax Credit"
  from 2025-01-01:
    return max(0, earned_income * 0.34)
"""
        result = parse_rac(rac)
        var = result.variables[0]
        assert var.label == "Earned Income Tax Credit"

    def test_variable_with_multiple_temporal_formulas(self):
        """Can parse variable with different formulas for different dates."""
        rac = """
credit:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2026-01-01:
    return income * 0.10
  from 2018-01-01:
    return income * 0.15
"""
        result = parse_rac(rac)
        var = result.variables[0]
        assert len(var.temporal) == 2
        assert var.temporal[0].from_date == "2026-01-01"
        assert var.temporal[1].from_date == "2018-01-01"
        # Most recent temporal entry becomes the formula
        assert "0.10" in var.formula


class TestUnifiedMixedFile:
    """Tests for complete files using unified syntax."""

    def test_parse_complete_unified_file(self):
        """Can parse a complete file with unified syntax."""
        rac = '''
# 26 USC 1411 - Net Investment Income Tax

"""
In the case of an individual, there shall be imposed
a tax equal to 3.8 percent of the lesser of NII or excess MAGI.
"""

niit_rate:
  source: "26 USC 1411(a)"
  from 2013-01-01: 0.038

threshold_joint:
  source: "26 USC 1411(b)"
  from 2013-01-01: 250000

niit:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Net Investment Income Tax"
  from 2013-01-01:
    excess = max(0, agi - 250000)
    return min(net_investment_income, excess) * 0.038
'''
        result = parse_rac(rac)
        assert result.statute_text is not None
        assert "3.8 percent" in result.statute_text
        assert "niit_rate" in result.parameters
        assert "threshold_joint" in result.parameters
        assert len(result.variables) == 1
        assert result.variables[0].name == "niit"

    def test_unified_file_compiles_to_js(self):
        """Unified syntax file can be compiled to JS."""
        rac = """
rate:
  source: "Test"
  from 2024-01-01: 20

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return income * 0.2
"""
        result = parse_rac(rac)
        gen = result.to_js_generator()
        code = gen.generate()
        assert "function calculate(" in code
        assert "PARAMS" in code


class TestUnifiedExampleFiles:
    """Tests for example .rac files in unified syntax."""

    def test_eitc_unified_example_parses(self):
        """examples/eitc.rac parses correctly with unified syntax."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.rac"
        if not eitc_path.exists():
            pytest.skip("examples/eitc.rac not found")

        content = eitc_path.read_text()
        result = parse_rac(content)

        assert len(result.parameters) >= 1
        assert len(result.variables) >= 1


class TestBackwardCompatibility:
    """Ensure parse_rac handles legacy .cos syntax too."""

    def test_parse_rac_handles_legacy_cos(self):
        """parse_rac can parse legacy .cos format."""
        cos = """
source {
  citation: "26 USC 32"
  accessed: 2025-01-01
}

parameter rate {
  source: "Test"
  values {
    0: 10
    1: 20
  }
}

variable tax {
  entity Person
  period Year
  dtype Money
  formula {
    return income * rate[0] / 100
  }
}
"""
        result = parse_rac(cos)
        assert result.source.citation == "26 USC 32"
        assert result.parameters["rate"].values == {0: 10.0, 1: 20.0}
        assert len(result.variables) == 1
        assert result.variables[0].entity == "Person"

    def test_cosfile_alias(self):
        """CosFile is an alias for RacFile."""
        assert CosFile is RacFile
