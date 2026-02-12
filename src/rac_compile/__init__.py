"""
rac-compile: Compile RAC DSL to standalone JavaScript and Python.

This module generates JS and Python code from .cos files for use in browsers,
Node.js, and Python applications without any dependencies.
"""

from .js_generator import JSCodeGenerator
from .js_generator import Parameter as JSParameter
from .js_generator import Variable as JSVariable
from .js_generator import generate_eitc_calculator as generate_eitc_calculator_js
from .parser import CosFile, ParameterDef, SourceBlock, VariableBlock, parse_cos
from .python_generator import PythonCodeGenerator
from .python_generator import Parameter as PythonParameter
from .python_generator import Variable as PythonVariable
from .python_generator import generate_eitc_calculator as generate_eitc_calculator_py

__version__ = "0.1.0"
__all__ = [
    "JSCodeGenerator",
    "PythonCodeGenerator",
    "generate_eitc_calculator_js",
    "generate_eitc_calculator_py",
    "JSParameter",
    "JSVariable",
    "PythonParameter",
    "PythonVariable",
    "parse_cos",
    "CosFile",
    "SourceBlock",
    "VariableBlock",
    "ParameterDef",
]
