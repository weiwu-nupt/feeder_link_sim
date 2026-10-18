"""ILA accuracy, spectral metrics, boundaries, OFDM references and GUI lifecycle."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from modules.pa_simulation_core import PAModel, fit_model, save_model, load_model
from modules.pa_dpd_core import (
    train_ila, evaluate_dpd, cascade, channel_acpr, align_pair,
    partition_capture, ofdm_evm, save_dpd, load_dpd,
)


class DPDCoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20)
        self.x = (rng.normal(size=12000) + 1j*rng.normal(size=12000))*.13
        self.pa = PAModel([[3, 0, -.8]], "memPoly", 3, sample_rate=10e6)

    def evaluate(self, pa, dpd, x, gain):
        return evaluate_dpd(pa, dpd, x, 10e6, gain, main_bandwidth=2e6,
                            adjacent_bandwidth=2e6, adjacent_offset=2e6)

    def test_cross_term_formula_fit_and_interchange(self):
        # y[n] = 2*x[n] + j*x[n-1] + .4*x[n]*|x[n-1]|^2
        coefficients = np.zeros((2, 5), complex)
        coefficients[0, 0], coefficients[1, 0], coefficients[0, 4] = 2, 1j, .4
        original = PAModel(coefficients, "ctMemPoly", 3, sample_rate=10e6)
        y = original.predict(self.x)
        expected = 2*self.x[1:] + 1j*self.x[:-1] + .4*self.x[1:]*abs(self.x[:-1])**2
        np.testing.assert_allclose(y[1:], expected)
        fitted = fit_model(self.x[:4000], y[:4000], "ctMemPoly", 2, 3, sample_rate=10e6)
        np.testing.assert_allclose(fitted.predict(self.x[4000:]), original.predict(self.x[4000:]), atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("npz", "csv", "txt"):
                path = Path(directory)/f"cross.{suffix}"
                save_model(path, fitted)
                restored = load_model(path)
                self.assertEqual(restored.name, "Cross-Term Memory")
                self.assertEqual(restored.coefficients.shape, (2, 5))
                np.testing.assert_allclose(restored.predict(self.x), fitted.predict(self.x))

    def test_ila_inverts_complex_gain_without_losing_amplification(self):
        pa = PAModel([[3j]], "memPoly", 1, sample_rate=10e6)
        trained = train_ila(pa, self.x[:4000], self.x[4000:8000], 10e6, 3,
                            memory=1, degree=1, iterations=1)
        result = self.evaluate(pa, trained.model, self.x[8000:], 3)
        np.testing.assert_allclose(result["y1"], 3*self.x[8000:], atol=1e-12)
        self.assertGreater(result["metrics"]["before"]["rmsPercent"], 100)
        self.assertLess(result["metrics"]["after"]["rmsPercent"], 1e-10)
        self.assertEqual(trained.model.role, "predistorter")

    def test_nonlinear_ila_improves_independent_test_for_all_structures(self):
        for kind in ("memPoly", "gmp", "ctMemPoly"):
            with self.subTest(kind=kind):
                trained = train_ila(self.pa, self.x[:5000], self.x[5000:8000], 10e6, 3,
                    model_type=kind, memory=2, degree=7, cross_depth=1, iterations=2)
                result = self.evaluate(self.pa, trained.model, self.x[8000:], 3)
                before, after = result["metrics"]["before"], result["metrics"]["after"]
                self.assertLess(after["rmsPercent"], before["rmsPercent"] / 100)
                self.assertGreater(after["powerDbm"], before["powerDbm"])
                self.assertIsNone(after["evmPercent"])
                self.assertEqual(trained.model.fit_info["trainingSamples"], 5000)

    def test_combined_memory_guards_exclude_both_models_zero_padding(self):
        pa = PAModel(np.ones((3, 7)), "gmp", 3, 1)
        dpd = PAModel(np.ones((2, 7)), "gmp", 3, 1, role="predistorter")
        u, y, valid = cascade(pa, dpd, self.x)
        self.assertEqual(valid.start, 5)  # (2-1+1) + (3-1+1)
        self.assertEqual(valid.stop, len(self.x)-2)
        np.testing.assert_allclose(y[valid], pa.predict(u)[valid])
        # At every scored PA tap, both signal and envelope indices belong to P's valid interval.
        p_valid = dpd.valid_slice(len(self.x))
        for _, _, signal_delay, envelope_delay, _ in pa.terms():
            self.assertGreaterEqual(valid.start-max(signal_delay, envelope_delay), p_valid.start)
            self.assertLessEqual(valid.stop-min(signal_delay, envelope_delay), p_valid.stop)

    def test_acpr_integrates_linear_power_in_separate_channels(self):
        fs = 4096
        n = np.arange(32768)
        signal = np.ones(len(n), complex) + .1*np.exp(2j*np.pi*(-480)*n/fs) + .0316227766*np.exp(2j*np.pi*520*n/fs)
        lower, upper = channel_acpr(signal, fs, 200, 200, 500, 4096)
        self.assertAlmostEqual(lower, -20, places=5)
        self.assertAlmostEqual(upper, -30, places=5)
        for b, a, offset in ((200, 200, 100), (200, 200, 2000)):
            with self.assertRaises(ValueError):
                channel_acpr(signal, fs, b, a, offset)

    def test_measured_alignment_does_not_remove_gain_or_phase(self):
        y = np.r_[np.zeros(9), 2j*self.x[:-9]]
        u, aligned, delay = align_pair(self.x, y)
        self.assertEqual(delay, 9)
        np.testing.assert_allclose(aligned, 2j*u)
        trained = train_ila(PAModel([[2j]], "memPoly", 1, sample_rate=10e6),
            self.x[:3000], self.x[3000:6000], 10e6, 2j,
            memory=1, degree=1, measured_pair=(self.x[:3000], y[:3000]))
        self.assertEqual(trained.model.fit_info["acquisitionDelay"], 9)
        self.assertEqual(trained.model.fit_info["trainingSource"], "measured")

    def test_out_of_range_rejection_and_model_roles_roundtrip(self):
        with self.assertRaisesRegex(ValueError, "峰值|幅度"):
            train_ila(self.pa, self.x[:5000], self.x[5000:8000], 10e6, 3, peak_limit=.01)
        trained = train_ila(self.pa, self.x[:5000], self.x[5000:8000], 10e6, 3, memory=1, degree=5)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("npz", "csv", "txt"):
                path = Path(directory)/f"dpd.{suffix}"
                save_dpd(path, trained.model)
                restored = load_dpd(path)
                np.testing.assert_allclose(restored.predict(self.x), trained.model.predict(self.x))
                self.assertEqual(restored.fit_info, trained.model.fit_info)
                with self.assertRaisesRegex(ValueError, "用途"):
                    load_model(path)
            path = Path(directory)/"pa.npz"
            save_model(path, self.pa)
            with self.assertRaisesRegex(ValueError, "用途"):
                load_dpd(path)
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.evaluate(PAModel([[4]], "memPoly", 1, sample_rate=10e6), trained.model, self.x, 3)

    def test_unsafe_candidate_is_rejected_and_passthrough_is_reported(self):
        trained = train_ila(self.pa, self.x[:5000], self.x[5000:8000], 10e6, 9,
                            memory=1, degree=3, iterations=1, peak_limit=1)
        self.assertFalse(trained.history[0]["accepted"])
        self.assertFalse(trained.history[0]["inRange"])
        self.assertFalse(trained.model.fit_info["improved"])
        np.testing.assert_array_equal(trained.model.predict(self.x), self.x)

    def test_sionna_reference_partition_and_fixed_gain_evm(self):
        from modules.pa_ofdm import generate_ofdm
        x, fs, reference = generate_ofdm(5e6, 7, return_reference=True)
        blocks = partition_capture(x, reference)
        np.testing.assert_array_equal(np.concatenate([b[0] for b in blocks]), x)
        self.assertEqual(sum(b[1]["numSymbols"] for b in blocks), reference["numSymbols"])
        test, ref = blocks[-1]
        evm, symbols = ofdm_evm(2j*test, ref, 2j, slice(0, len(test)))
        self.assertLess(evm, 1e-10)
        self.assertEqual(symbols, ref["numSymbols"])
        evm, _ = ofdm_evm(2j*test*.9, ref, 2j, slice(0, len(test)))
        self.assertAlmostEqual(evm, 10, places=10)


class DPDWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_cross_model_selection_and_async_dpd_export_invalidation(self):
        from PyQt6.QtCore import QEventLoop, QTimer
        from modules.pa_model import PAModelDialog
        window = PAModelDialog()
        self.addCleanup(window.close)
        self.assertEqual([window.tabs.tabText(i) for i in range(3)],
                         ["实测数据建模与导出", "导入模型与信号仿真", "数字预失真"])
        window.combo_model.setCurrentIndex(2)
        self.assertEqual(window._MODELS[2][1], "ctMemPoly")
        self.assertFalse(window.e_crossdepth.isEnabled())
        page = window.dpd_page
        window.tabs.setCurrentIndex(2)
        rng = np.random.default_rng(3)
        x = (rng.normal(size=12000)+1j*rng.normal(size=12000))*.12
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_model(root/"pa.npz", PAModel([[3, 0, -.8]], "memPoly", 3, sample_rate=10e6))
            np.savez(root/"x.npz", x=x, sampleRate=10e6)
            page.set_model_file(root/"pa.npz")
            page.set_wave_file(root/"x.npz")
            page.gain_db.setValue(20*np.log10(3))
            page.memory.setValue(1)
            page.degree.setValue(7)
            page.main_bw.setValue(2)
            page.adj_bw.setValue(2)
            page.adj_offset.setValue(2)
            with patch("modules.pa_dpd.QMessageBox.warning") as warning:
                page.run_button.click()
                self.assertTrue(page.busy)
                self.assertFalse(page.controls_scroll.isEnabled())
                loop = QEventLoop()
                page.worker.finished.connect(loop.quit)
                QTimer.singleShot(60000, loop.quit)
                loop.exec()
                self.app.processEvents()
                warning.assert_not_called()
            self.assertFalse(page.busy)
            self.assertIsNotNone(page.result)
            self.assertEqual(len(page.figure.axes[0].lines), 3)
            self.assertLess(page.result["metrics"]["after"]["rmsPercent"], .01)
            self.assertEqual(page.result["trainingSourceSamples"], 6000)
            self.assertEqual(len(page.result["x"]), 3000)
            self.assertEqual(page.table.item(4, 1).text(), "无符号参考")
            for mode in (1, 2):
                page.plot_mode.setCurrentIndex(mode)
                self.assertEqual(len(page.figure.axes[0].collections), 3)
            with patch("modules.pa_dpd.QFileDialog.getSaveFileName", return_value=(str(root/"result.npz"), "")):
                page.export_result()
            with np.load(root/"result.npz", allow_pickle=False) as data:
                self.assertIn("u", data)
                metrics = json.loads(data["metricsJson"].item())
                self.assertEqual(metrics, page.result["metrics"])
            with patch("modules.pa_dpd.QFileDialog.getSaveFileName", return_value=(str(root/"dpd.npz"), "")):
                page.export_dpd()
            self.assertEqual(load_dpd(root/"dpd.npz").role, "predistorter")
            page.level.setValue(-1)
            self.assertIsNone(page.result)
            self.assertFalse(page.export_button.isEnabled())
            self.assertIsNotNone(page.dpd)


if __name__ == "__main__":
    unittest.main()
