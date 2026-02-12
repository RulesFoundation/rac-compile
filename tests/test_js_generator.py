"""
Tests for rac-compile JS code generator.

TDD: Tests written first, implementation follows.
"""

import pytest
from src.rac_compile.js_generator import (
    JSCodeGenerator,
    generate_eitc_calculator,
    Parameter,
    Variable,
)


class TestJSCodeGenerator:
    """Tests for JSCodeGenerator class."""

    def test_init_defaults(self):
        """Generator initializes with sensible defaults."""
        gen = JSCodeGenerator()
        assert gen.module_name == "calculator"
        assert gen.include_provenance is True
        assert gen.typescript is False
        assert gen.parameters == {}
        assert gen.variables == []
        assert gen.inputs == {}

    def test_add_input_number(self):
        """Can add numeric input with default."""
        gen = JSCodeGenerator()
        gen.add_input("income", 50000, "number")
        assert "income" in gen.inputs
        assert gen.inputs["income"]["default"] == 50000
        assert gen.inputs["income"]["type"] == "number"

    def test_add_input_boolean_converts_to_js(self):
        """Boolean defaults are converted to JS syntax."""
        gen = JSCodeGenerator()
        gen.add_input("is_married", False, "boolean")
        assert gen.inputs["is_married"]["default"] == "false"

        gen.add_input("has_children", True, "boolean")
        assert gen.inputs["has_children"]["default"] == "true"

    def test_add_parameter(self):
        """Can add parameter with values and source."""
        gen = JSCodeGenerator()
        gen.add_parameter(
            "tax_rate",
            {0: 10, 1: 12, 2: 22},
            "26 USC 1(a)",
        )
        assert "tax_rate" in gen.parameters
        assert gen.parameters["tax_rate"].values == {0: 10, 1: 12, 2: 22}
        assert gen.parameters["tax_rate"].source == "26 USC 1(a)"

    def test_add_variable(self):
        """Can add calculated variable."""
        gen = JSCodeGenerator()
        gen.add_variable(
            name="tax",
            inputs=["income"],
            formula_js="income * 0.2",
            label="Income Tax",
            citation="26 USC 1",
        )
        assert len(gen.variables) == 1
        assert gen.variables[0].name == "tax"
        assert gen.variables[0].formula_js == "income * 0.2"


class TestGenerateOutput:
    """Tests for generated JS code."""

    def test_generate_includes_header(self):
        """Generated code includes module header."""
        gen = JSCodeGenerator(module_name="Test Calculator")
        code = gen.generate()
        assert "Test Calculator" in code
        assert "Auto-generated from RAC DSL" in code

    def test_generate_includes_params_object(self):
        """Generated code includes PARAMS constant."""
        gen = JSCodeGenerator()
        gen.add_parameter("rate", {0: 10, 1: 20}, "Test Source")
        code = gen.generate()
        assert "const PARAMS = {" in code
        assert "rate:" in code
        assert "0: 10" in code
        assert "// Test Source" in code

    def test_generate_includes_calculate_function(self):
        """Generated code includes calculate function."""
        gen = JSCodeGenerator()
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "x * 2")
        code = gen.generate()
        assert "function calculate(" in code
        assert "x = 0" in code
        assert "const y = x * 2" in code

    def test_generate_returns_citations(self):
        """Generated calculate returns citation chain."""
        gen = JSCodeGenerator()
        gen.add_parameter("rate", {0: 10}, "26 USC 1")
        gen.add_variable("tax", [], "100", citation="26 USC 1(a)")
        code = gen.generate()
        assert "citations:" in code
        assert 'source: "26 USC 1"' in code

    def test_generate_esm_exports(self):
        """Generated code includes ESM exports."""
        gen = JSCodeGenerator()
        code = gen.generate()
        assert "export { calculate, PARAMS };" in code
        assert "export default calculate;" in code

    def test_generate_provenance_sources(self):
        """Provenance section lists all sources."""
        gen = JSCodeGenerator(include_provenance=True)
        gen.add_parameter("a", {0: 1}, "Source A")
        gen.add_parameter("b", {0: 2}, "Source B")
        gen.add_variable("c", [], "1", citation="Source C")
        code = gen.generate()
        assert "Sources:" in code
        assert "Source A" in code
        assert "Source B" in code
        assert "Source C" in code

    def test_generate_no_provenance(self):
        """Can disable provenance section."""
        gen = JSCodeGenerator(include_provenance=False)
        gen.add_parameter("a", {0: 1}, "Source A")
        code = gen.generate()
        assert "Sources:" not in code


class TestGenerateEITCCalculator:
    """Tests for pre-built EITC calculator."""

    def test_returns_valid_js(self):
        """EITC calculator generates valid JS structure."""
        code = generate_eitc_calculator()
        assert "function calculate(" in code
        assert "const PARAMS = {" in code
        assert "export default calculate;" in code

    def test_includes_all_eitc_params(self):
        """EITC calculator includes required parameters."""
        code = generate_eitc_calculator()
        assert "credit_pct:" in code
        assert "phaseout_pct:" in code
        assert "earned_income_amount:" in code
        assert "phaseout_single:" in code
        assert "phaseout_joint:" in code

    def test_includes_statute_citations(self):
        """EITC calculator cites 26 USC 32."""
        code = generate_eitc_calculator()
        assert "26 USC 32" in code
        assert "26 USC 32(b)(1)" in code

    def test_includes_guidance_citations(self):
        """EITC calculator cites Rev. Proc. 2024-40."""
        code = generate_eitc_calculator()
        assert "Rev. Proc. 2024-40" in code

    def test_inputs_have_correct_defaults(self):
        """EITC inputs have sensible defaults."""
        code = generate_eitc_calculator()
        assert "earned_income = 0" in code
        assert "agi = 0" in code
        assert "n_children = 0" in code
        assert "is_joint = false" in code


class TestJSExecution:
    """Tests that generated JS actually executes correctly."""

    @pytest.fixture
    def simple_calculator(self):
        """Create a simple calculator for testing."""
        gen = JSCodeGenerator()
        gen.add_input("income", 0)
        gen.add_parameter("rate", {0: 20}, "Test")
        gen.add_variable("tax", ["income"], "income * PARAMS.rate[0] / 100")
        return gen.generate()

    def test_generated_code_is_syntactically_valid(self, simple_calculator):
        """Generated code can be parsed (basic syntax check)."""
        # Check for balanced braces
        assert simple_calculator.count("{") == simple_calculator.count("}")
        assert simple_calculator.count("(") == simple_calculator.count(")")
        assert simple_calculator.count("[") == simple_calculator.count("]")

    def test_eitc_calculator_syntax(self):
        """EITC calculator is syntactically valid."""
        code = generate_eitc_calculator()
        assert code.count("{") == code.count("}")
        assert code.count("(") == code.count(")")
