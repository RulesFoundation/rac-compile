# rac-compile

Compile RAC `.cos` files to standalone JavaScript and Python calculators.

## Overview

`rac-compile` generates JS and Python code from RAC policy encodings. JavaScript output runs entirely in the browser with no server required. Python output can be imported and used in any Python application. Every calculation includes a citation chain tracing values back to authoritative law.

## Installation

```bash
pip install rac-compile
```

## Quick start

### Command line

```bash
# Generate JavaScript EITC calculator
rac-compile eitc -o eitc.js

# Generate Python EITC calculator
rac-compile eitc --python -o eitc.py

# Output to stdout
rac-compile eitc           # JavaScript
rac-compile eitc --python  # Python
```

### Python API

```python
from rac_compile import (
    JSCodeGenerator,
    PythonCodeGenerator,
    generate_eitc_calculator_js,
    generate_eitc_calculator_py,
)

# Pre-built EITC calculator (JavaScript)
js_code = generate_eitc_calculator_js()
print(js_code)

# Pre-built EITC calculator (Python)
py_code = generate_eitc_calculator_py()
print(py_code)

# Custom JavaScript calculator
js_gen = JSCodeGenerator(module_name="My Calculator")
js_gen.add_input("income", 0, "number")
js_gen.add_parameter("rate", {0: 20, 1: 30}, "26 USC 1")
js_gen.add_variable("tax", ["income"], "income * PARAMS.rate[0] / 100", citation="26 USC 1(a)")
js_code = js_gen.generate()

# Custom Python calculator
py_gen = PythonCodeGenerator(module_name="My Calculator")
py_gen.add_input("income", 0, "float")
py_gen.add_parameter("rate", {0: 20, 1: 30}, "26 USC 1")
py_gen.add_variable("tax", ["income"], "income * PARAMS['rate'][0] / 100", citation="26 USC 1(a)")
py_code = py_gen.generate()
```

### Generated output

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

- **Multi-target compilation**: Generate JavaScript or Python from the same DSL
- **Citation chains**: Every calculation traces back to statute/guidance
- **Zero dependencies**: Generated code runs standalone (JS in browsers, Python anywhere)
- **ESM exports**: JavaScript works with modern bundlers and `<script type="module">`
- **Type hints**: Python output includes full type annotations, TypeScript support coming soon

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=rac_compile
```

## See also

- [Rules Foundation](https://rules.foundation) - Open infrastructure for encoded law
- [pe-compile](https://github.com/PolicyEngine/pe-compile) - Similar tool for PolicyEngine
