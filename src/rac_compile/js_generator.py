"""
JavaScript code generation from RAC DSL.

Generates standalone JS calculators that can run in browsers
without any backend - perfect for static sites like rules.foundation/demo.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    """A parameter value from statute or guidance."""

    name: str
    values: dict[int, float]  # key by number of children or bracket index
    source: str  # e.g., "26 USC 32(b)(1)" or "Rev. Proc. 2024-40"


@dataclass
class Variable:
    """A compiled variable ready for JS generation."""

    name: str
    inputs: list[str]
    formula_js: str
    label: str = ""
    citation: str = ""


class JSCodeGenerator:
    """
    Generate standalone JavaScript calculators from RAC DSL.

    Usage:
        gen = JSCodeGenerator()
        gen.add_parameter("credit_pct", {0: 7.65, 1: 34, ...}, "26 USC 32(b)(1)")
        gen.add_variable("eitc", ["earned_income", "agi", ...], formula_js)
        code = gen.generate()
    """

    def __init__(
        self,
        module_name: str = "calculator",
        include_provenance: bool = True,
        typescript: bool = False,
    ):
        self.module_name = module_name
        self.include_provenance = include_provenance
        self.typescript = typescript
        self.parameters: dict[str, Parameter] = {}
        self.variables: list[Variable] = []
        self.inputs: dict[str, Any] = {}  # name -> default value

    def add_input(self, name: str, default: Any = 0, type_hint: str = "number") -> None:
        """Add an input variable."""
        # Convert Python booleans to JS
        if isinstance(default, bool):
            default = "true" if default else "false"
        self.inputs[name] = {"default": default, "type": type_hint}

    def add_parameter(
        self, name: str, values: dict[int, float], source: str = ""
    ) -> None:
        """Add a parameter with values indexed by bracket."""
        self.parameters[name] = Parameter(name=name, values=values, source=source)

    def add_variable(
        self,
        name: str,
        inputs: list[str],
        formula_js: str,
        label: str = "",
        citation: str = "",
    ) -> None:
        """Add a calculated variable with its JS formula."""
        self.variables.append(
            Variable(
                name=name,
                inputs=inputs,
                formula_js=formula_js,
                label=label,
                citation=citation,
            )
        )

    def generate(self) -> str:
        """Generate the complete JavaScript module."""
        lines = []

        # Header with provenance
        lines.append("/**")
        lines.append(f" * {self.module_name} - Auto-generated from RAC DSL")
        lines.append(" * ")
        lines.append(" * This code runs entirely in the browser with full citation")
        lines.append(" * chain - every value traces back to authoritative law.")
        lines.append(" * ")
        if self.include_provenance:
            lines.append(" * Sources:")
            sources = set()
            for param in self.parameters.values():
                if param.source:
                    sources.add(param.source)
            for var in self.variables:
                if var.citation:
                    sources.add(var.citation)
            for src in sorted(sources):
                lines.append(f" *   - {src}")
        lines.append(" */")
        lines.append("")

        # Parameters as const objects
        if self.parameters:
            lines.append("// Parameters from statute and guidance")
            lines.append("const PARAMS = {")
            for param in self.parameters.values():
                values_js = ", ".join(
                    f"{k}: {v}" for k, v in sorted(param.values.items())
                )
                comment = f"  // {param.source}" if param.source else ""
                lines.append(f"  {param.name}: {{ {values_js} }},{comment}")
            lines.append("};")
            lines.append("")

        # Main calculate function
        if self.typescript:
            self._generate_typescript_function(lines)
        else:
            self._generate_js_function(lines)

        # ESM exports
        lines.append("")
        lines.append("export { calculate, PARAMS };")
        lines.append("export default calculate;")

        return "\n".join(lines)

    def _generate_js_function(self, lines: list[str]) -> None:
        """Generate JavaScript function."""
        # JSDoc
        lines.append("/**")
        lines.append(" * Calculate tax/benefit values with full citation chain.")
        lines.append(" *")
        for name, info in self.inputs.items():
            lines.append(f" * @param {{{info['type']}}} {name}")
        lines.append(" * @returns {{result: number, citations: Array}}")
        lines.append(" */")

        # Function signature
        params = ", ".join(
            f"{name} = {info['default']}" for name, info in self.inputs.items()
        )
        lines.append(f"function calculate({{ {params} }}) {{")

        # Add calculations
        for var in self.variables:
            lines.append(f"  // {var.citation}" if var.citation else "")
            lines.append(f"  const {var.name} = {var.formula_js};")
            lines.append("")

        # Return with citations
        lines.append("  return {")
        for var in self.variables:
            lines.append(f"    {var.name},")
        lines.append("    citations: [")
        for param in self.parameters.values():
            if param.source:
                lines.append(f'      {{ param: "{param.name}", source: "{param.source}" }},')
        for var in self.variables:
            if var.citation:
                lines.append(f'      {{ variable: "{var.name}", source: "{var.citation}" }},')
        lines.append("    ],")
        lines.append("  };")
        lines.append("}")

    def _generate_typescript_function(self, lines: list[str]) -> None:
        """Generate TypeScript function with proper types."""
        # Interface for inputs
        lines.append("interface CalculatorInputs {")
        for name, info in self.inputs.items():
            lines.append(f"  {name}?: {info['type']};")
        lines.append("}")
        lines.append("")

        # Interface for result
        lines.append("interface CalculatorResult {")
        for var in self.variables:
            lines.append(f"  {var.name}: number;")
        lines.append("  citations: Array<{param?: string; variable?: string; source: string}>;")
        lines.append("}")
        lines.append("")

        # Function
        lines.append("function calculate(inputs: CalculatorInputs = {}): CalculatorResult {")

        # Destructure with defaults
        destructure = ", ".join(
            f"{name} = {info['default']}" for name, info in self.inputs.items()
        )
        lines.append(f"  const {{ {destructure} }} = inputs;")
        lines.append("")

        # Calculations
        for var in self.variables:
            if var.citation:
                lines.append(f"  // {var.citation}")
            lines.append(f"  const {var.name}: number = {var.formula_js};")
            lines.append("")

        # Return
        lines.append("  return {")
        for var in self.variables:
            lines.append(f"    {var.name},")
        lines.append("    citations: [")
        for param in self.parameters.values():
            if param.source:
                lines.append(f'      {{ param: "{param.name}", source: "{param.source}" }},')
        for var in self.variables:
            if var.citation:
                lines.append(f'      {{ variable: "{var.name}", source: "{var.citation}" }},')
        lines.append("    ],")
        lines.append("  };")
        lines.append("}")


def generate_eitc_calculator(tax_year: int = 2025) -> str:
    """
    Generate a standalone EITC calculator for the specified tax year.

    This is a pre-built calculator based on 26 USC 32.
    """
    gen = JSCodeGenerator(module_name=f"EITC Calculator (TY {tax_year})")

    # Inputs
    gen.add_input("earned_income", 0, "number")
    gen.add_input("agi", 0, "number")
    gen.add_input("n_children", 0, "number")
    gen.add_input("is_joint", False, "boolean")

    # Parameters from statute (fixed percentages)
    gen.add_parameter(
        "credit_pct",
        {0: 7.65, 1: 34, 2: 40, 3: 45},
        "26 USC 32(b)(1)",
    )
    gen.add_parameter(
        "phaseout_pct",
        {0: 7.65, 1: 15.98, 2: 21.06, 3: 21.06},
        "26 USC 32(b)(1)",
    )

    # Parameters from IRS guidance (inflation-adjusted for TY 2025)
    gen.add_parameter(
        "earned_income_amount",
        {0: 8260, 1: 12730, 2: 17880, 3: 17880},
        "Rev. Proc. 2024-40",
    )
    gen.add_parameter(
        "phaseout_single",
        {0: 10620, 1: 23350, 2: 23350, 3: 23350},
        "Rev. Proc. 2024-40",
    )
    gen.add_parameter(
        "phaseout_joint",
        {0: 17730, 1: 30470, 2: 30470, 3: 30470},
        "Rev. Proc. 2024-40",
    )

    # EITC formula - follows 26 USC 32(a)
    eitc_formula = """(() => {
    const n = Math.min(n_children, 3);
    const creditPct = PARAMS.credit_pct[n] / 100;
    const phaseoutPct = PARAMS.phaseout_pct[n] / 100;
    const earnedAmount = PARAMS.earned_income_amount[n];
    const phaseoutStart = is_joint
      ? PARAMS.phaseout_joint[n]
      : PARAMS.phaseout_single[n];

    // 32(a)(1): Credit base
    const creditBase = creditPct * Math.min(earned_income, earnedAmount);

    // 32(a)(2): Phaseout
    const incomeForPhaseout = Math.max(agi, earned_income);
    const excess = Math.max(0, incomeForPhaseout - phaseoutStart);
    const phaseout = phaseoutPct * excess;

    return Math.max(0, Math.round(creditBase - phaseout));
  })()"""

    gen.add_variable(
        "eitc",
        ["earned_income", "agi", "n_children", "is_joint"],
        eitc_formula,
        label="Earned Income Tax Credit",
        citation="26 USC 32",
    )

    return gen.generate()


if __name__ == "__main__":
    # Generate and print EITC calculator
    print(generate_eitc_calculator())
