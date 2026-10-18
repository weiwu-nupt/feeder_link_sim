"""Cold-open options must not initialize Torch/Sionna or change NR configurations."""

import json
import math
import os
import subprocess
import sys
import unittest
from modules.ber_core.catalog import load_catalog


class CatalogStartupTests(unittest.TestCase):
    def test_cold_dialog_has_options_before_events_without_compute_imports(self):
        script = """
import sys,json,time
from PyQt6.QtWidgets import QApplication
from modules.ber_analysis import BERAnalysisDialog
app=QApplication([])
begin=time.perf_counter();window=BERAnalysisDialog();elapsed=time.perf_counter()-begin
assert window.ready and len(window.codes)==60 and window.modulation.count()==7
assert 'torch' not in sys.modules and 'sionna' not in sys.modules
window.close()
print(json.dumps(dict(ready_ms=elapsed*1000)))
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(json.loads(result.stdout)["ready_ms"], 0)

    def test_fixed_nr_profiles_equal_previous_live_mcs_definitions(self):
        from sionna.phy.nr import decode_mcs_index

        numerators = set()
        for table in (1, 2, 3, 4):
            for index in range(29):
                try:
                    _, rate = decode_mcs_index(
                        index,
                        table_index=table,
                        is_pusch=True,
                        check_index_validity=True,
                        device="cpu",
                    )
                    value = round(float(rate) * 1024)
                    if value / 1024 >= 0.2:
                        numerators.add(value)
                except (ValueError, AssertionError):
                    pass
        expected = [
            dict(
                id=f"nr_512_{math.ceil(512*1024/rate)}",
                name=f"K512 · N{math.ceil(512*1024/rate)} · {rate}/1024",
                family="nr",
                k=512,
                n=math.ceil(512 * 1024 / rate),
            )
            for rate in sorted(numerators)
        ]
        actual = [code for code in load_catalog()["codes"] if code["family"] == "nr"]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
