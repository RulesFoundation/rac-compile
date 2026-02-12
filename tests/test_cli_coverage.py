"""Tests for cli.py to achieve 100% coverage.

Tests the main() CLI entry point by mocking sys.argv and verifying
all branches: compile (success, file-not-found, stdout), eitc (JS, Python,
to file, to stdout), no-command.
"""

from unittest.mock import patch

import pytest

from src.rac_compile.cli import main


class TestCLIMainCompile:
    """Test compile command branches."""

    def test_compile_file_not_found(self, tmp_path):
        """Compile with non-existent file prints error and exits 1."""
        fake_input = str(tmp_path / "nonexistent.rac")
        with patch("sys.argv", ["rac-compile", "compile", fake_input]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_to_stdout(self, tmp_path):
        """Compile with no -o prints JS to stdout."""
        input_file = tmp_path / "test.cos"
        input_file.write_text(
            """
variable x {
  formula { return 42 }
}
"""
        )
        with patch("sys.argv", ["rac-compile", "compile", str(input_file)]):
            # Should not raise, just prints to stdout
            with patch("builtins.print") as mock_print:
                main()
                # Should have printed the generated code
                output = mock_print.call_args_list[0][0][0]
                assert "calculate" in output

    def test_compile_to_file(self, tmp_path):
        """Compile with -o writes JS to file."""
        input_file = tmp_path / "test.cos"
        input_file.write_text(
            """
variable x {
  formula { return 42 }
}
"""
        )
        output_file = tmp_path / "output.js"
        with patch(
            "sys.argv",
            ["rac-compile", "compile", str(input_file), "-o", str(output_file)],
        ):
            main()
            assert output_file.exists()
            content = output_file.read_text()
            assert "calculate" in content


class TestCLIMainEitc:
    """Test eitc command branches."""

    def test_eitc_js_to_stdout(self):
        """eitc command outputs JS to stdout by default."""
        with patch("sys.argv", ["rac-compile", "eitc"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "function calculate(" in output

    def test_eitc_js_to_file(self, tmp_path):
        """eitc command writes JS to file with -o."""
        output_file = tmp_path / "eitc.js"
        with patch("sys.argv", ["rac-compile", "eitc", "-o", str(output_file)]):
            main()
            assert output_file.exists()
            content = output_file.read_text()
            assert "function calculate(" in content

    def test_eitc_python_to_stdout(self):
        """eitc --python outputs Python to stdout."""
        with patch("sys.argv", ["rac-compile", "eitc", "--python"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "def calculate(" in output

    def test_eitc_python_to_file(self, tmp_path):
        """eitc --python -o writes Python to file."""
        output_file = tmp_path / "eitc.py"
        with patch(
            "sys.argv",
            ["rac-compile", "eitc", "--python", "-o", str(output_file)],
        ):
            main()
            assert output_file.exists()
            content = output_file.read_text()
            assert "def calculate(" in content

    def test_eitc_custom_year(self):
        """eitc --year 2024 passes correct year."""
        with patch("sys.argv", ["rac-compile", "eitc", "--year", "2024"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "2024" in output


class TestCLIMainNoCommand:
    """Test no-command case."""

    def test_no_command_exits_1(self):
        """No command prints help and exits 1."""
        with patch("sys.argv", ["rac-compile"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
