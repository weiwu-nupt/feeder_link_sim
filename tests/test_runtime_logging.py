"""Only the optional Triton FLOP-counter warning is filtered."""

import logging
import subprocess
import sys
import unittest
from modules.ber_core.diagnostics import configure_runtime_logging


class RuntimeLoggingTests(unittest.TestCase):
    def test_real_sionna_startup_omits_optional_counter_warning(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "from modules.ber_core import configure_runtime_logging; from sionna.phy.nr import decode_mcs_index; print('ready')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ready")
        self.assertNotIn("triton not found; flop counting", result.stderr)

    def test_other_warnings_and_errors_remain_visible(self):
        configure_runtime_logging()
        logger = logging.getLogger("torch.utils.flop_counter")
        with self.assertLogs(logger, level="WARNING") as records:
            logger.warning("triton not found; flop counting will not work for triton kernels")
            logger.warning("other warning")
            logger.error("decoder failure")
        self.assertEqual(len(records.output), 2)
        self.assertIn("other warning", records.output[0])
        self.assertIn("decoder failure", records.output[1])


if __name__ == "__main__":
    unittest.main()
