"""
Parser for RAC DSL files (.rac and legacy .cos).

Parses RAC policy encoding files into structured data
that can be compiled to JavaScript and Python.

Supports both:
- Legacy .cos syntax: parameter/variable keywords with brace blocks
- Unified .rac v3 syntax: name: definitions with from yyyy-mm-dd: temporal entries
"""

import re
import textwrap
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
class TemporalEntry:
    """A single temporal entry with a from-date and either a scalar value or code."""

    from_date: str  # yyyy-mm-dd
    value: Optional[float] = None  # scalar value
    code: Optional[str] = None  # formula code


@dataclass
class ParameterDef:
    """Parsed parameter definition."""

    source: str = ""
    values: dict[int, float] = field(default_factory=dict)
    # Unified syntax: temporal entries keyed by date
    temporal: list[TemporalEntry] = field(default_factory=list)
    description: Optional[str] = None
    unit: Optional[str] = None
    reference: Optional[str] = None


@dataclass
class VariableBlock:
    """Parsed variable block."""

    name: str
    entity: Optional[str] = None
    period: Optional[str] = None
    dtype: Optional[str] = None
    label: Optional[str] = None
    formula: str = ""
    # Unified syntax: temporal formula entries
    temporal: list[TemporalEntry] = field(default_factory=list)


@dataclass
class RacFile:
    """Parsed .rac or .cos file."""

    source: Optional[SourceBlock] = None
    statute_text: Optional[str] = None
    parameters: dict[str, ParameterDef] = field(default_factory=dict)
    variables: list[VariableBlock] = field(default_factory=list)

    def to_js_generator(self) -> JSCodeGenerator:
        """Convert to JSCodeGenerator for JS output."""
        gen = JSCodeGenerator()

        # Add citation from source block
        citation = self.source.citation if self.source else None

        # Parameters become PARAMS entries
        for name, param_def in self.parameters.items():
            values = param_def.values if param_def.values else {0: 0}
            gen.add_parameter(name, values, param_def.source)

        # Variables become calculations
        for var in self.variables:
            # Extract inputs from formula (simple heuristic)
            formula = var.formula
            # For temporal variables, use the formula with the latest date
            if not formula and var.temporal:
                code_entries = [t for t in var.temporal if t.code]
                if code_entries:
                    latest = max(code_entries, key=lambda t: t.from_date)
                    formula = latest.code

            inputs = self._extract_inputs(formula)
            for inp in inputs:
                if inp not in gen.inputs:
                    gen.add_input(inp, 0)

            gen.add_variable(
                name=var.name,
                inputs=inputs,
                formula_js=self._formula_to_js(formula),
                label=var.label or "",
                citation=citation or "",
            )

        return gen

    def _extract_inputs(self, formula: str) -> list[str]:
        """Extract likely input variable names from formula."""
        # Common input patterns for tax/benefit calculations
        common_inputs = [
            # Income types
            "income", "earned_income", "agi", "wages",
            "gross_income", "net_income", "tax_liability",
            # Family/household composition
            "n_children", "num_children", "n_qualifying_children",
            "household_size", "family_size",
            # Filing status
            "is_joint", "is_married", "filing_status",
        ]
        found = []
        for inp in common_inputs:
            if inp in formula:
                found.append(inp)
        return found

    def _formula_to_js(self, formula: str) -> str:
        """Convert formula DSL to JavaScript."""
        # Normalize indentation using textwrap.dedent
        js = textwrap.dedent(formula).strip()

        # Convert let statements
        js = re.sub(r"^let\s+", "const ", js, flags=re.MULTILINE)

        # Convert comments (# to //)
        js = re.sub(r"#\s*", "// ", js)

        # Convert DSL functions to JS Math functions
        # Must do these before adding PARAMS. prefix to avoid double-converting
        js = re.sub(r"\bmin\(", "Math.min(", js)
        js = re.sub(r"\bmax\(", "Math.max(", js)
        js = re.sub(r"\bround\(", "Math.round(", js)
        js = re.sub(r"\bfloor\(", "Math.floor(", js)
        js = re.sub(r"\bceil\(", "Math.ceil(", js)
        js = re.sub(r"\babs\(", "Math.abs(", js)

        # Convert parameter references to PARAMS.name
        for param_name in self.parameters.keys():
            # Match param_name followed by [ (for bracket access)
            js = re.sub(
                rf"\b{param_name}\[",
                f"PARAMS.{param_name}[",
                js
            )

        # Wrap in IIFE if multi-line
        if "\n" in js or "const " in js:
            lines = js.split("\n")
            indented = "\n".join(f"    {line}" for line in lines)
            js = f"(() => {{\n{indented}\n  }})()"

        return js


# Backward compatibility alias
CosFile = RacFile


# Date pattern for temporal entries: from yyyy-mm-dd:
_DATE_PATTERN = re.compile(r"^from\s+(\d{4}-\d{2}-\d{2})\s*:\s*$")

# Detect if a "from" line has an inline scalar value: from yyyy-mm-dd: 123.45
_DATE_SCALAR_PATTERN = re.compile(
    r"^from\s+(\d{4}-\d{2}-\d{2})\s*:\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_rac(content: str) -> RacFile:
    """
    Parse a .rac file using unified syntax.

    Unified syntax uses:
    - name: for all definitions (no parameter/variable/input keywords)
    - from yyyy-mm-dd: for temporal entries (scalar values or code blocks)
    - Top-level triple-quoted strings for statute text
    - Type inferred from attributes (entity/period/dtype = variable)

    Args:
        content: The .rac file content

    Returns:
        RacFile with parsed definitions
    """
    result = RacFile()

    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Top-level triple-quoted statute text
        if stripped.startswith('"""'):
            text_lines = []
            # Check if it closes on the same line
            rest = stripped[3:]
            if '"""' in rest:
                result.statute_text = rest[:rest.index('"""')]
                i += 1
                continue
            text_lines.append(rest)
            i += 1
            while i < len(lines):
                if '"""' in lines[i]:
                    before_close = lines[i][:lines[i].index('"""')]
                    text_lines.append(before_close)
                    break
                text_lines.append(lines[i])
                i += 1
            result.statute_text = textwrap.dedent("\n".join(text_lines)).strip()
            i += 1
            continue

        # Legacy source block
        if stripped.startswith("source") and "{" in stripped:
            block_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "}":
                block_lines.append(lines[i])
                i += 1
            result.source = _parse_source_block("\n".join(block_lines))
            i += 1
            continue

        # Legacy parameters block
        if stripped.startswith("parameters") and "{" in stripped:
            block_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "}":
                block_lines.append(lines[i])
                i += 1
            result.parameters = _parse_parameters_block("\n".join(block_lines))
            i += 1
            continue

        # Legacy parameter block (parameter name { ... })
        legacy_param = re.match(r"parameter\s+(\w+)\s*\{", stripped)
        if legacy_param:
            name = legacy_param.group(1)
            block_lines = []
            brace_depth = 1
            i += 1
            while i < len(lines) and brace_depth > 0:
                l = lines[i]
                brace_depth += l.count("{") - l.count("}")
                if brace_depth > 0:
                    block_lines.append(l)
                i += 1
            result.parameters[name] = _parse_parameter_block(
                name, "\n".join(block_lines)
            )
            continue

        # Legacy variable block (variable name { ... })
        legacy_var = re.match(r"variable\s+(\w+)\s*\{", stripped)
        if legacy_var:
            name = legacy_var.group(1)
            block_lines = []
            brace_depth = 1
            i += 1
            while i < len(lines) and brace_depth > 0:
                l = lines[i]
                brace_depth += l.count("{") - l.count("}")
                if brace_depth > 0:
                    block_lines.append(l)
                i += 1
            result.variables.append(_parse_variable_block(name, "\n".join(block_lines)))
            continue

        # Unified syntax: name: (top-level definition)
        # Must be at column 0 (no leading whitespace) and match word:
        unified_match = re.match(r"^(\w+)\s*:\s*$", line)
        if unified_match:
            name = unified_match.group(1)
            i += 1
            i, definition = _parse_unified_definition(name, lines, i)
            if isinstance(definition, ParameterDef):
                result.parameters[name] = definition
            elif isinstance(definition, VariableBlock):
                result.variables.append(definition)
            continue

        i += 1

    return result


def _parse_unified_definition(
    name: str, lines: list[str], start: int
) -> tuple[int, "ParameterDef | VariableBlock"]:
    """
    Parse a unified definition block starting after the 'name:' line.

    Reads indented lines to extract attributes and temporal entries.
    Determines whether this is a parameter or variable based on attributes.

    Returns:
        Tuple of (next line index, parsed definition)
    """
    attrs: dict[str, str] = {}
    temporal: list[TemporalEntry] = []
    i = start

    while i < len(lines):
        line = lines[i]

        # End of block: non-indented, non-empty line
        if line and not line[0].isspace() and line.strip():
            break

        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Check for "from yyyy-mm-dd: value" (inline scalar)
        scalar_match = _DATE_SCALAR_PATTERN.match(stripped)
        if scalar_match:
            date = scalar_match.group(1)
            value = float(scalar_match.group(2))
            temporal.append(TemporalEntry(from_date=date, value=value))
            i += 1
            continue

        # Check for "from yyyy-mm-dd:" (start of temporal block)
        date_match = _DATE_PATTERN.match(stripped)
        if date_match:
            date = date_match.group(1)
            i += 1
            # Collect the indented code block under this date
            code_lines = []
            # Determine the indent level of the code block
            code_indent = None
            while i < len(lines):
                code_line = lines[i]
                code_stripped = code_line.strip()

                # Empty lines within code are ok
                if not code_stripped:
                    code_lines.append("")
                    i += 1
                    continue

                # Determine indent level from first non-empty line
                if code_indent is None:
                    code_indent = len(code_line) - len(code_line.lstrip())

                current_indent = len(code_line) - len(code_line.lstrip())

                # If less indented than the code block, we're done
                # But allow lines at the "from" level (2 spaces typically)
                # to be new temporal entries or attributes
                if current_indent < code_indent:
                    break

                code_lines.append(code_line)
                i += 1

            # Strip trailing empty lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()

            code = textwrap.dedent("\n".join(code_lines)).strip()

            # Determine if this is a scalar or code
            try:
                value = float(code)
                temporal.append(TemporalEntry(from_date=date, value=value))
            except ValueError:
                temporal.append(TemporalEntry(from_date=date, code=code))
            continue

        # Regular attribute: key: value
        attr_match = re.match(r"(\w+)\s*:\s*(.+)", stripped)
        if attr_match:
            key = attr_match.group(1)
            value = attr_match.group(2).strip().strip('"')
            attrs[key] = value
            i += 1
            continue

        i += 1

    # Determine type: variable if has entity/period/dtype, otherwise parameter
    is_variable = any(k in attrs for k in ("entity", "period", "dtype", "formula"))

    if is_variable:
        var = VariableBlock(name=name)
        var.entity = attrs.get("entity")
        var.period = attrs.get("period")
        var.dtype = attrs.get("dtype")
        var.label = attrs.get("label")
        var.temporal = temporal

        # If there are temporal code entries, use the one with the latest date
        code_entries = [t for t in temporal if t.code]
        if code_entries:
            latest = max(code_entries, key=lambda t: t.from_date)
            var.formula = latest.code

        return i, var
    else:
        param = ParameterDef()
        param.source = attrs.get("source", attrs.get("reference", ""))
        param.description = attrs.get("description")
        param.unit = attrs.get("unit")
        param.reference = attrs.get("reference")
        param.temporal = temporal

        # Convert temporal scalar entries to values dict
        # For backward compat, use integer keys (0-indexed)
        if temporal:
            for idx, entry in enumerate(temporal):
                if entry.value is not None:
                    param.values[idx] = entry.value

        return i, param


# Keep parse_cos as an alias for backward compatibility
def parse_cos(content: str) -> CosFile:
    """
    Parse a .cos file content string.

    This is an alias for parse_rac that supports both legacy .cos syntax
    and the new unified .rac syntax.

    Args:
        content: The .cos or .rac file content

    Returns:
        RacFile with parsed blocks
    """
    return parse_rac(content)


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


def _parse_parameters_block(content: str) -> dict[str, ParameterDef]:
    """Parse parameters block content (simple key: source format)."""
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
            params[key.strip()] = ParameterDef(source=value.strip())

    return params


def _parse_parameter_block(name: str, content: str) -> ParameterDef:
    """Parse a single parameter block with source and values."""
    param = ParameterDef()

    # Extract values block first
    values_match = re.search(
        r"values\s*\{([^}]+)\}",
        content,
        re.DOTALL,
    )
    if values_match:
        values_content = values_match.group(1)
        for line in values_content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    param.values[int(key.strip())] = float(value.strip())
                except ValueError:
                    pass  # Skip invalid entries

    # Parse source
    source_match = re.search(r'source:\s*"([^"]+)"', content)
    if source_match:
        param.source = source_match.group(1)
    else:
        # Try unquoted source
        source_match = re.search(r"source:\s*(\S+)", content)
        if source_match:
            param.source = source_match.group(1)

    return param


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
        # Use dedent to normalize indentation, then strip trailing whitespace
        var.formula = textwrap.dedent(formula_match.group(1)).strip()

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
