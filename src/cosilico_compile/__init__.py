"""
cosilico-compile: Compile Cosilico DSL to standalone JavaScript.

This module generates JS code from .cos files for use in browsers
and Node.js without any Python dependencies.
"""

from .js_generator import JSCodeGenerator, generate_eitc_calculator, Parameter, Variable
from .parser import parse_cos, CosFile, SourceBlock, VariableBlock, ParameterDef

__version__ = "0.1.0"
__all__ = [
    "JSCodeGenerator",
    "generate_eitc_calculator",
    "Parameter",
    "Variable",
    "parse_cos",
    "CosFile",
    "SourceBlock",
    "VariableBlock",
    "ParameterDef",
]
