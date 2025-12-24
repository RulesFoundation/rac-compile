"""
Python code generation from Cosilico DSL.

Generates standalone Python calculators that can be imported
and used in Python applications without any dependencies.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Parameter:
    """A parameter value from statute or guidance."""

    name: str
    values: dict[int, float]  # key by number of children or bracket index
    source: str  # e.g., "26 USC 32(b)(1)" or "Rev. Proc. 2024-40"


@dataclass
class Variable:
    """A compiled variable ready for Python generation."""

    name: str
    inputs: list[str]
    formula_python: str
    label: str = ""
    citation: str = ""


class PythonCodeGenerator:
    """
    Generate standalone Python calculators from Cosilico DSL.

    Usage:
        gen = PythonCodeGenerator()
        gen.add_parameter("credit_pct", {0: 7.65, 1: 34, ...}, "26 USC 32(b)(1)")
        gen.add_variable("eitc", ["earned_income", "agi", ...], formula_python)
        code = gen.generate()
    """

    def __init__(
        self,
        module_name: str = "calculator",
        include_provenance: bool = True,
        type_hints: bool = True,
    ):
        self.module_name = module_name
        self.include_provenance = include_provenance
        self.type_hints = type_hints
        self.parameters: dict[str, Parameter] = {}
        self.variables: list[Variable] = []
        self.inputs: dict[str, Any] = {}  # name -> default value

    def add_input(self, name: str, default: Any = 0, type_hint: str = "float") -> None:
        """Add an input variable."""
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
        formula_python: str,
        label: str = "",
        citation: str = "",
    ) -> None:
        """Add a calculated variable with its Python formula."""
        self.variables.append(
            Variable(
                name=name,
                inputs=inputs,
                formula_python=formula_python,
                label=label,
                citation=citation,
            )
        )

    def generate(self) -> str:
        """Generate the complete Python module."""
        lines = []

        # Header with provenance
        lines.append('"""')
        lines.append(f"{self.module_name} - Auto-generated from Cosilico DSL")
        lines.append("")
        lines.append("This code runs standalone with full citation chain -")
        lines.append("every value traces back to authoritative law.")

        if self.include_provenance:
            lines.append("")
            lines.append("Sources:")
            sources = set()
            for param in self.parameters.values():
                if param.source:
                    sources.add(param.source)
            for var in self.variables:
                if var.citation:
                    sources.add(var.citation)
            for src in sorted(sources):
                lines.append(f"  - {src}")
        lines.append('"""')
        lines.append("")

        # Type imports if needed
        if self.type_hints:
            lines.append("from typing import Any")
            lines.append("")

        # Parameters as constant dict
        if self.parameters:
            lines.append("# Parameters from statute and guidance")
            lines.append("PARAMS = {")
            for param in self.parameters.values():
                values_str = "{" + ", ".join(
                    f"{k}: {v}" for k, v in sorted(param.values.items())
                ) + "}"
                comment = f"  # {param.source}" if param.source else ""
                lines.append(f'    "{param.name}": {values_str},{comment}')
            lines.append("}")
            lines.append("")

        # Main calculate function
        self._generate_function(lines)

        return "\n".join(lines)

    def _generate_function(self, lines: list[str]) -> None:
        """Generate Python function."""
        # Function signature
        if self.type_hints:
            # Build parameter list with type hints
            params = []
            for name, info in self.inputs.items():
                type_str = info["type"]
                default = info["default"]
                if isinstance(default, bool):
                    default_str = str(default)
                elif isinstance(default, str):
                    default_str = f'"{default}"'
                else:
                    default_str = str(default)
                params.append(f"{name}: {type_str} = {default_str}")

            params_str = ", ".join(params) if params else ""
            lines.append(f"def calculate({params_str}) -> dict[str, Any]:")
        else:
            # No type hints
            params = []
            for name, info in self.inputs.items():
                default = info["default"]
                if isinstance(default, bool):
                    default_str = str(default)
                elif isinstance(default, str):
                    default_str = f'"{default}"'
                else:
                    default_str = str(default)
                params.append(f"{name}={default_str}")

            params_str = ", ".join(params) if params else ""
            lines.append(f"def calculate({params_str}):")

        # Docstring
        lines.append('    """')
        lines.append("    Calculate tax/benefit values with full citation chain.")
        lines.append("")
        for name, info in self.inputs.items():
            lines.append(f"    Args:")
            break
        for name, info in self.inputs.items():
            lines.append(f"        {name}: {info['type']}")
        lines.append("")
        lines.append("    Returns:")
        lines.append("        Dictionary with calculated values and citations")
        lines.append('    """')

        # Add calculations
        for var in self.variables:
            if var.citation:
                lines.append(f"    # {var.citation}")
            lines.append(f"    {var.name} = {var.formula_python}")
            lines.append("")

        # Return dictionary with citations
        lines.append("    return {")
        for var in self.variables:
            lines.append(f'        "{var.name}": {var.name},')
        lines.append('        "citations": [')
        for param in self.parameters.values():
            if param.source:
                lines.append(
                    f'            {{"param": "{param.name}", "source": "{param.source}"}},'
                )
        for var in self.variables:
            if var.citation:
                lines.append(
                    f'            {{"variable": "{var.name}", "source": "{var.citation}"}},'
                )
        lines.append("        ],")
        lines.append("    }")


def generate_eitc_calculator(tax_year: int = 2025) -> str:
    """
    Generate a standalone EITC calculator for the specified tax year.

    This is a pre-built calculator based on 26 USC 32.
    """
    gen = PythonCodeGenerator(module_name=f"EITC Calculator (TY {tax_year})")

    # Inputs
    gen.add_input("earned_income", 0, "float")
    gen.add_input("agi", 0, "float")
    gen.add_input("n_children", 0, "int")
    gen.add_input("is_joint", False, "bool")

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
    eitc_formula = """min(n_children, 3) and (
        lambda n: (
            lambda credit_pct, phaseout_pct, earned_amount, phaseout_start: (
                lambda credit_base: (
                    lambda income_for_phaseout: (
                        lambda excess: (
                            lambda phaseout: max(0, round(credit_base - phaseout))
                        )(phaseout_pct * excess)
                    )(max(0, income_for_phaseout - phaseout_start))
                )(max(agi, earned_income))
            )(credit_pct * min(earned_income, earned_amount))
        )(
            PARAMS['credit_pct'][n] / 100,
            PARAMS['phaseout_pct'][n] / 100,
            PARAMS['earned_income_amount'][n],
            PARAMS['phaseout_joint'][n] if is_joint else PARAMS['phaseout_single'][n]
        )
    )(min(n_children, 3))"""

    # Simpler, more readable version
    eitc_formula_simple = """(lambda n: (
    lambda credit_base, income_for_phaseout, phaseout_start, phaseout_pct:
        max(0, round(credit_base - phaseout_pct * max(0, income_for_phaseout - phaseout_start)))
    )(
        PARAMS['credit_pct'][n] / 100 * min(earned_income, PARAMS['earned_income_amount'][n]),
        max(agi, earned_income),
        PARAMS['phaseout_joint'][n] if is_joint else PARAMS['phaseout_single'][n],
        PARAMS['phaseout_pct'][n] / 100
    )
)(min(n_children, 3))"""

    gen.add_variable(
        "eitc",
        ["earned_income", "agi", "n_children", "is_joint"],
        eitc_formula_simple,
        label="Earned Income Tax Credit",
        citation="26 USC 32",
    )

    return gen.generate()


if __name__ == "__main__":
    # Generate and print EITC calculator
    print(generate_eitc_calculator())
