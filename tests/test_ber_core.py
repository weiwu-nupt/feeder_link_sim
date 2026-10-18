"""Independent BER regression checks against actual transmitted bits and H."""

import math
import unittest
from collections import Counter
import numpy as np
import torch
from modules.ber_core.catalog import load_catalog, apsk_definition
from modules.ber_core.simulation import make_modem, make_codec, simulate, validate_code
from modules.ber_core.matrices import read_matrix


class BERCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.catalog = load_catalog()

    def request(self, code=None, **changes):
        req = dict(
            code=code or dict(family="none", k=1024),
            mod=self.catalog["modulations"][1],
            start=20,
            stop=20,
            step=1,
            budget=73,
            errors=0,
            iterations=4,
            batch=2,
            device="cpu",
        )
        req.update(changes)
        return req

    def run_link(self, request):
        events = []
        simulate(request, events.append, lambda: False)
        self.assertEqual(events[-1], dict(type="done", cancelled=False))
        return [e["row"] for e in events if e["type"] == "point"]

    def test_reference_data_catalog(self):
        counts = Counter(c["family"] for c in self.catalog["codes"])
        self.assertEqual(counts, dict(dvb=21, nr=35, qc=3, none=1))
        short = next(
            c
            for c in self.catalog["codes"]
            if c["family"] == "dvb" and c["rate"] == "1/2" and "short" in c["id"]
        )
        self.assertEqual((short["k"], short["n"]), (7200, 16200))

    def test_modems_recover_all_binary_labels(self):
        mods = self.catalog["modulations"] + [apsk_definition(16, [4, 12], [1, 2.85], [45, 15])]
        for mod in mods:
            with self.subTest(modulation=mod["name"]):
                mapper, demapper, points = make_modem(mod)
                m = mod["bits"]
                labels = torch.arange(1 << m)
                bits = ((labels[:, None] >> torch.arange(m - 1, -1, -1)) & 1).float()
                received = mapper(bits)
                llr = demapper(received, torch.tensor(1e-6))
                self.assertTrue(torch.equal(llr > 0, bits.bool()))
                self.assertAlmostEqual(float(points.abs().square().mean()), 1, places=5)

    def test_dvb_every_matrix_satisfies_original_checks(self):
        for code in self.catalog["codes"]:
            if code["family"] != "dvb":
                continue
            with self.subTest(matrix=code["id"]):
                encoder, _, k, n, h = make_codec(code, iterations=1)
                word = encoder(torch.randint(0, 2, (1, k)).float()).numpy().T
                self.assertEqual(word.shape, (n, 1))
                self.assertFalse(np.any(np.asarray(h @ word, dtype=np.int64) % 2))

    def test_actual_ldpc_decode_and_partial_frame(self):
        for family in ("nr", "dvb"):
            code = next(c for c in self.catalog["codes"] if c["family"] == family)
            row = self.run_link(self.request(code))[0]
            self.assertEqual(row["bits"], 73)
            self.assertEqual(row["bit_errors"], 0)

    def test_qpsk_awgn_zero_db_matches_information_bit_theory(self):
        row = self.run_link(self.request(start=0, stop=0, budget=100000))[0]
        self.assertLess(abs(row["ber"] - math.erfc(1) / 2), 0.005)
        self.assertEqual(row["bits"], 100000)

    def test_trellis_codes_and_custom_interleaver(self):
        for family, interleaver in [
            ("conv", "qpp"),
            ("turbo", "qpp"),
            ("turbo", "random"),
            ("turbo", "custom"),
        ]:
            code = dict(
                family=family,
                k=40,
                length=3,
                polys=["7", "5"],
                terminate=True,
                rate="1/3",
                interleaver=interleaver,
            )
            if interleaver == "custom":
                code["permutation"] = list(range(39, -1, -1))
            with self.subTest(family=family, interleaver=interleaver):
                checked = validate_code(code)
                self.assertGreater(checked["n"], 40)
                row = self.run_link(self.request(code))[0]
                self.assertEqual(row["bit_errors"], 0)
                self.assertEqual(row["bits"], 73)

    def test_custom_h_and_qc_rank_and_noiseless_recovery(self):
        for code in [
            dict(
                family="h",
                format="h",
                text="1 0 1 0 1 0 1\n0 1 1 0 0 1 1\n0 0 0 1 1 1 1\n1 0 1 0 1 0 1",
            ),
            dict(family="qc", format="qc", text="0 1 0 -1\n1 0 -1 0", z=3, direction=1),
        ]:
            checked = validate_code(code)
            if code["family"] == "h":
                self.assertEqual((checked["k"], checked["n"]), (4, 7))
            row = self.run_link(self.request(code))[0]
            self.assertEqual(row["bit_errors"], 0)

    def test_cancel_does_not_emit_fabricated_point(self):
        events = []
        simulate(self.request(), events.append, lambda: True)
        self.assertEqual(events, [dict(type="done", cancelled=True)])
        events = []
        stop = [False]

        def emit(event):
            events.append(event)
            if event["type"] == "progress":
                stop[0] = True

        simulate(self.request(budget=10000, batch=1), emit, lambda: stop[0])
        row = next(e["row"] for e in events if e["type"] == "point")
        self.assertEqual(row["bits"], 1024)
        self.assertTrue(events[-1]["cancelled"])

    def test_malformed_matrix_is_rejected(self):
        for text in ("1,,0", "1 2\n0 1", "1 0\n1"):
            with self.assertRaises(ValueError):
                read_matrix(text, "h")


if __name__ == "__main__":
    unittest.main()
