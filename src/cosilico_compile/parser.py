"""
Parser for .cos DSL files.

Parses Cosilico policy encoding files into structured data
that can be compiled to JavaScript.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .js_generator import JSCodeGenerator


@dataclass
class SourceBlock:
    """Parsed source block."""

    lawarchive: Optional[str] = None
    citation: Optional[str] = None
    accessed: Optional[str] = None


@dataclass
class VariableBlock:
    """Parsed variable block."""

    name: str
    entity: Optional[str] = None
    period: Optional[str] = None
    dtype: Optional[str] = None
    label: Optional[str] = None
    formula: str = ""


@dataclass
class CosFile:
    """Parsed .cos file."""

    source: Optional[SourceBlock] = None
    parameters: dict[str, str] = field(default_factory=dict)
    variables: list[VariableBlock] = field(default_factory=list)

    def to_js_generator(self) -> JSCodeGenerator:
        """Convert to JSCodeGenerator for JS output."""
        gen = JSCodeGenerator()

        # Add citation from source block
        citation = self.source.citation if self.source else None

        # Parameters become PARAMS entries
        # Note: In a full implementation, we'd resolve the paths to actual values
        # For now, we just note them as placeholders
        for name, path in self.parameters.items():
            gen.add_parameter(name, {0: 0}, path)  # Placeholder values

        # Variables become calculations
        for var in self.variables:
            # Extract inputs from formula (simple heuristic)
            inputs = self._extract_inputs(var.formula)
            for inp in inputs:
                if inp not in gen.inputs:
                    gen.add_input(inp, 0)

            gen.add_variable(
                name=var.name,
                inputs=inputs,
                formula_js=self._formula_to_js(var.formula),
                label=var.label or "",
                citation=citation or "",
            )

        return gen

    def _extract_inputs(self, formula: str) -> list[str]:
        """Extract likely input variable names from formula."""
        # Common input patterns
        common_inputs = [
            "income", "earned_income", "agi", "wages",
            "n_children", "num_children", "is_joint", "is_married",
        ]
        found = []
        for inp in common_inputs:
            if inp in formula:
                found.append(inp)
        return found

    def _formula_to_js(self, formula: str) -> str:
        """Convert formula DSL to JavaScript."""
        js = formula.strip()

        # Convert let statements
        js = re.sub(r"^let\s+", "const ", js, flags=re.MULTILINE)

        # Convert comments (# to //)
        js = re.sub(r"#\s*", "// ", js)

        # Wrap in IIFE if multi-line
        if "\n" in js or "const " in js:
            lines = js.split("\n")
            indented = "\n".join(f"    {line}" for line in lines)
            js = f"(() => {{\n{indented}\n  }})()"

        return js


def parse_cos(content: str) -> CosFile:
    """
    Parse a .cos file content string.

    Args:
        content: The .cos file content

    Returns:
        CosFile with parsed blocks
    """
    result = CosFile()

    # Remove full-line comments (but preserve comments in formulas)
    lines = content.split("\n")
    cleaned_lines = []
    in_formula = False

    for line in lines:
        stripped = line.strip()

        # Track formula blocks
        if "formula {" in line or "formula{" in line:
            in_formula = True
        if in_formula and stripped == "}":
            in_formula = False

        # Keep comments inside formulas, remove outside
        if stripped.startswith("#") and not in_formula:
            continue

        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)

    # Parse source block
    source_match = re.search(
        r"source\s*\{([^}]+)\}",
        content,
        re.DOTALL,
    )
    if source_match:
        result.source = _parse_source_block(source_match.group(1))

    # Parse parameters block
    params_match = re.search(
        r"parameters\s*\{([^}]+)\}",
        content,
        re.DOTALL,
    )
    if params_match:
        result.parameters = _parse_parameters_block(params_match.group(1))

    # Parse variable blocks
    variable_pattern = re.compile(
        r"variable\s+(\w+)\s*\{(.*?)\n\}",
        re.DOTALL,
    )
    for match in variable_pattern.finditer(content):
        name = match.group(1)
        body = match.group(2)
        result.variables.append(_parse_variable_block(name, body))

    return result


def _parse_source_block(content: str) -> SourceBlock:
    """Parse source block content."""
    source = SourceBlock()

    # Parse key: value pairs
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')

            if key == "lawarchive":
                source.lawarchive = value
            elif key == "citation":
                source.citation = value
            elif key == "accessed":
                source.accessed = value

    return source


def _parse_parameters_block(content: str) -> dict[str, str]:
    """Parse parameters block content."""
    params = {}

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Remove inline comments
        if "#" in line:
            line = line.split("#")[0].strip()

        if ":" in line:
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()

    return params


def _parse_variable_block(name: str, content: str) -> VariableBlock:
    """Parse variable block content."""
    var = VariableBlock(name=name)

    # Extract formula block first
    formula_match = re.search(
        r"formula\s*\{(.*)\}",
        content,
        re.DOTALL,
    )
    if formula_match:
        var.formula = formula_match.group(1).strip()

    # Parse metadata (everything before formula)
    pre_formula = content
    if formula_match:
        pre_formula = content[: formula_match.start()]

    for line in pre_formula.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # entity Type
        if line.startswith("entity "):
            var.entity = line.split(" ", 1)[1].strip()
        # period Year/Month
        elif line.startswith("period "):
            var.period = line.split(" ", 1)[1].strip()
        # dtype Money/Rate/Boolean
        elif line.startswith("dtype "):
            var.dtype = line.split(" ", 1)[1].strip()
        # label "..."
        elif line.startswith("label "):
            label_match = re.search(r'label\s+"([^"]+)"', line)
            if label_match:
                var.label = label_match.group(1)

    return var
