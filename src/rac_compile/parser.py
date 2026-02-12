"""Parser for RAC DSL files (.rac and legacy .cos).

Supports legacy .cos brace syntax and unified .rac v3 temporal syntax.
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
    """A temporal entry: a from-date with either a scalar value or code."""

    from_date: str
    value: Optional[float] = None
    code: Optional[str] = None


@dataclass
class ParameterDef:
    """Parsed parameter definition."""

    source: str = ""
    values: dict[int, float] = field(default_factory=dict)
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
    temporal: list[TemporalEntry] = field(default_factory=list)

    @property
    def effective_formula(self) -> str:
        """Return formula, falling back to the latest temporal code entry."""
        if self.formula:
            return self.formula
        code_entries = [t for t in self.temporal if t.code]
        if code_entries:
            return max(code_entries, key=lambda t: t.from_date).code
        return ""


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
        citation = self.source.citation if self.source else None

        for name, param_def in self.parameters.items():
            values = param_def.values or {0: 0}
            gen.add_parameter(name, values, param_def.source)

        for var in self.variables:
            formula = var.effective_formula
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

    _COMMON_INPUTS = [
        "income", "earned_income", "agi", "wages",
        "gross_income", "net_income", "tax_liability",
        "n_children", "num_children", "n_qualifying_children",
        "household_size", "family_size",
        "is_joint", "is_married", "filing_status",
    ]

    def _extract_inputs(self, formula: str) -> list[str]:
        """Extract likely input variable names from formula."""
        return [inp for inp in self._COMMON_INPUTS if inp in formula]

    def _formula_to_js(self, formula: str) -> str:
        """Convert formula DSL to JavaScript."""
        js = textwrap.dedent(formula).strip()
        js = re.sub(r"^let\s+", "const ", js, flags=re.MULTILINE)
        js = re.sub(r"#\s*", "// ", js)

        for func in ("min", "max", "round", "floor", "ceil", "abs"):
            js = re.sub(rf"\b{func}\(", f"Math.{func}(", js)

        for param_name in self.parameters:
            js = re.sub(rf"\b{param_name}\[", f"PARAMS.{param_name}[", js)

        if "\n" in js or "const " in js:
            lines = js.split("\n")
            indented = "\n".join(f"    {line}" for line in lines)
            js = f"(() => {{\n{indented}\n  }})()"

        return js


CosFile = RacFile

_DATE_PATTERN = re.compile(r"^from\s+(\d{4}-\d{2}-\d{2})\s*:\s*$")
_DATE_SCALAR_PATTERN = re.compile(
    r"^from\s+(\d{4}-\d{2}-\d{2})\s*:\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_rac(content: str) -> RacFile:
    """Parse .rac or .cos file content into a RacFile."""
    result = RacFile()

    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if stripped.startswith('"""'):
            text_lines = []
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

        if stripped.startswith("source") and "{" in stripped:
            block_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "}":
                block_lines.append(lines[i])
                i += 1
            result.source = _parse_source_block("\n".join(block_lines))
            i += 1
            continue

        if stripped.startswith("parameters") and "{" in stripped:
            block_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "}":
                block_lines.append(lines[i])
                i += 1
            result.parameters = _parse_parameters_block("\n".join(block_lines))
            i += 1
            continue

        legacy_param = re.match(r"parameter\s+(\w+)\s*\{", stripped)
        if legacy_param:
            name = legacy_param.group(1)
            i, body = _collect_brace_block(lines, i + 1)
            result.parameters[name] = _parse_parameter_block(name, body)
            continue

        legacy_var = re.match(r"variable\s+(\w+)\s*\{", stripped)
        if legacy_var:
            name = legacy_var.group(1)
            i, body = _collect_brace_block(lines, i + 1)
            result.variables.append(_parse_variable_block(name, body))
            continue

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


def _collect_brace_block(lines: list[str], start: int) -> tuple[int, str]:
    """Collect lines inside a brace-delimited block, handling nesting."""
    block_lines = []
    brace_depth = 1
    i = start
    while i < len(lines) and brace_depth > 0:
        line = lines[i]
        brace_depth += line.count("{") - line.count("}")
        if brace_depth > 0:
            block_lines.append(line)
        i += 1
    return i, "\n".join(block_lines)


def _parse_unified_definition(
    name: str, lines: list[str], start: int
) -> tuple[int, "ParameterDef | VariableBlock"]:
    """Parse a unified definition block starting after the 'name:' line.

    Returns (next line index, parsed ParameterDef or VariableBlock).
    """
    attrs: dict[str, str] = {}
    temporal: list[TemporalEntry] = []
    i = start

    while i < len(lines):
        line = lines[i]
        if line and not line[0].isspace() and line.strip():
            break

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        scalar_match = _DATE_SCALAR_PATTERN.match(stripped)
        if scalar_match:
            temporal.append(TemporalEntry(
                from_date=scalar_match.group(1),
                value=float(scalar_match.group(2)),
            ))
            i += 1
            continue

        date_match = _DATE_PATTERN.match(stripped)
        if date_match:
            date = date_match.group(1)
            i += 1
            i, entry = _collect_temporal_block(date, lines, i)
            temporal.append(entry)
            continue

        attr_match = re.match(r"(\w+)\s*:\s*(.+)", stripped)
        if attr_match:
            attrs[attr_match.group(1)] = attr_match.group(2).strip().strip('"')
            i += 1
            continue

        i += 1

    if any(k in attrs for k in ("entity", "period", "dtype", "formula")):
        return i, VariableBlock(
            name=name,
            entity=attrs.get("entity"),
            period=attrs.get("period"),
            dtype=attrs.get("dtype"),
            label=attrs.get("label"),
            temporal=temporal,
        )

    return i, ParameterDef(
        source=attrs.get("source", attrs.get("reference", "")),
        description=attrs.get("description"),
        unit=attrs.get("unit"),
        reference=attrs.get("reference"),
        temporal=temporal,
        values={
            idx: entry.value
            for idx, entry in enumerate(temporal)
            if entry.value is not None
        },
    )


def _collect_temporal_block(
    date: str, lines: list[str], start: int
) -> tuple[int, TemporalEntry]:
    """Collect an indented code block under a 'from date:' line."""
    code_lines = []
    code_indent = None
    i = start

    while i < len(lines):
        code_line = lines[i]
        if not code_line.strip():
            code_lines.append("")
            i += 1
            continue

        if code_indent is None:
            code_indent = len(code_line) - len(code_line.lstrip())

        if len(code_line) - len(code_line.lstrip()) < code_indent:
            break

        code_lines.append(code_line)
        i += 1

    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    code = textwrap.dedent("\n".join(code_lines)).strip()

    try:
        return i, TemporalEntry(from_date=date, value=float(code))
    except ValueError:
        return i, TemporalEntry(from_date=date, code=code)


# Backward compatibility alias
parse_cos = parse_rac


def _parse_source_block(content: str) -> SourceBlock:
    """Parse source block content."""
    source = SourceBlock()
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
        if "#" in line:
            line = line.split("#")[0].strip()

        if ":" in line:
            key, value = line.split(":", 1)
            params[key.strip()] = ParameterDef(source=value.strip())

    return params


def _parse_parameter_block(name: str, content: str) -> ParameterDef:
    """Parse a single parameter block with source and values."""
    param = ParameterDef()

    values_match = re.search(r"values\s*\{([^}]+)\}", content, re.DOTALL)
    if values_match:
        for line in values_match.group(1).split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                param.values[int(key.strip())] = float(value.strip())
            except ValueError:
                pass

    source_match = re.search(r'source:\s*"([^"]+)"', content)
    if not source_match:
        source_match = re.search(r"source:\s*(\S+)", content)
    if source_match:
        param.source = source_match.group(1)

    return param


def _parse_variable_block(name: str, content: str) -> VariableBlock:
    """Parse variable block content."""
    var = VariableBlock(name=name)

    formula_match = re.search(r"formula\s*\{(.*)\}", content, re.DOTALL)
    if formula_match:
        var.formula = textwrap.dedent(formula_match.group(1)).strip()

    pre_formula = content[:formula_match.start()] if formula_match else content

    for line in pre_formula.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("entity "):
            var.entity = line.split(" ", 1)[1].strip()
        elif line.startswith("period "):
            var.period = line.split(" ", 1)[1].strip()
        elif line.startswith("dtype "):
            var.dtype = line.split(" ", 1)[1].strip()
        elif line.startswith("label "):
            label_match = re.search(r'label\s+"([^"]+)"', line)
            if label_match:
                var.label = label_match.group(1)

    return var
