import contextlib
import io
import subprocess
import sys
import unittest

from cleanhatch import __version__
from cleanhatch.cli import main


class CliTests(unittest.TestCase):
    def test_alpha_version(self) -> None:
        self.assertEqual(__version__, "0.0.1a0")

    def test_help_discloses_missing_validation(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("data validation is not implemented", output.getvalue())

    def test_module_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cleanhatch", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "cleanhatch 0.0.1a0")
        self.assertEqual(result.stderr, "")

    def test_validation_is_not_a_successful_noop(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cleanhatch", "validate"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: validate", result.stderr)


if __name__ == "__main__":
    unittest.main()
