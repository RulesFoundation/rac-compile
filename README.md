# cosilico-compile

Compile Cosilico `.cos` files to standalone JavaScript calculators.

## Overview

`cosilico-compile` generates JS code from Cosilico policy encodings that runs entirely in the browser - no server required. Every calculation includes a citation chain tracing values back to authoritative law.

## Installation

```bash
pip install cosilico-compile
```

## Quick Start

### Command Line

```bash
# Generate EITC calculator
cosilico-compile eitc -o eitc.js

# Output to stdout
cosilico-compile eitc
```

### Python API

```python
from cosilico_compile import JSCodeGenerator, generate_eitc_calculator

# Pre-built EITC calculator
code = generate_eitc_calculator()
print(code)

# Custom calculator
gen = JSCodeGenerator(module_name="My Calculator")
gen.add_input("income", 0, "number")
gen.add_parameter("rate", {0: 20, 1: 30}, "26 USC 1")
gen.add_variable("tax", ["income"], "income * PARAMS.rate[0] / 100", citation="26 USC 1(a)")
code = gen.generate()
```

### Generated Output

```javascript
const PARAMS = {
  credit_pct: { 0: 7.65, 1: 34, 2: 40, 3: 45 },  // 26 USC 32(b)(1)
  // ...
};

function calculate({ earned_income = 0, agi = 0, n_children = 0, is_joint = false }) {
  const eitc = /* formula */;

  return {
    eitc,
    citations: [
      { param: "credit_pct", source: "26 USC 32(b)(1)" },
      { variable: "eitc", source: "26 USC 32" },
    ],
  };
}

export { calculate, PARAMS };
export default calculate;
```

## Features

- **Citation chains**: Every calculation traces back to statute/guidance
- **Zero dependencies**: Generated JS runs standalone in any browser
- **ESM exports**: Works with modern bundlers and `<script type="module">`
- **TypeScript support**: Optional `.ts` output with full type annotations

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=cosilico_compile
```

## See Also

- [Cosilico](https://cosilico.ai) - Society, in silico
- [pe-compile](https://github.com/PolicyEngine/pe-compile) - Similar tool for PolicyEngine
