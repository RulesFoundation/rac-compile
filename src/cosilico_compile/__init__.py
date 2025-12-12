"""
cosilico-compile: Compile Cosilico DSL to standalone JavaScript.

This module generates JS code from .cosilico files for use in browsers
and Node.js without any Python dependencies.
"""

from .js_generator import JSCodeGenerator, generate_eitc_calculator

__all__ = ["JSCodeGenerator", "generate_eitc_calculator"]
