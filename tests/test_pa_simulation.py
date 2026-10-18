"""PA model interchange, analytic behavior, spectral scaling, and GUI workflow."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from modules.pa_simulation_core import (
    PAModel, fit_model, load_model, save_model, load_waveform, simulate, spectrum_pair, power_spectrum,
)


class PACoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = (rng.normal(size=4096) + 1j * rng.normal(size=4096)) / 4

    def test_mp_gain_and_delay(self):
        model = PAModel(np.array([[0, 0, 0], [2j, 0, -0.3]]), "memPoly", 3)
        y = model.predict(self.x)
        expected = 2j * self.x[:-1] - 0.3 * self.x[:-1] * abs(self.x[:-1]) ** 2
        np.testing.assert_allclose(y[1:], expected)
        self.assertEqual(y[0], 0)

    def test_gmp_cross_terms_and_edges(self):
        # D=3,L=1: [x,x|x|,x|x|², lag1 power1,lag1 power2,lead1 power1,lead1 power2]
        coef = np.zeros((2, 7), dtype=complex)
        coef[0, 0] = 2
        coef[0, 4] = 0.4j
        coef[1, 6] = -0.2
        model = PAModel(coef, "gmp", 3, 1)
        y = model.predict(self.x)
        n = np.arange(2, len(self.x) - 1)
        expected = 2 * self.x[n] + 0.4j * self.x[n] * abs(self.x[n-1])**2 - 0.2 * self.x[n-1] * abs(self.x[n])**2
        np.testing.assert_allclose(y[n], expected)
        np.testing.assert_array_equal(y[[0, 1, -1]], 0)

    def test_fit_and_model_roundtrip_predict_new_waveform(self):
        for kind, cross in (("memPoly", 0), ("gmp", 1)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                columns = 3 + 4 * cross
                coefficients = np.arange(2 * columns).reshape(2, columns) * (0.01 + 0.02j)
                coefficients[0, 0] = 1.5
                original = PAModel(coefficients, kind, 3, cross, 10e6)
                y = original.predict(self.x)
                fitted = fit_model(self.x, y, kind, 2, 3, cross, 10e6)
                test_x = self.x[::-1].copy() * 0.8
                np.testing.assert_allclose(fitted.predict(test_x), original.predict(test_x), atol=1e-12)
                for extension in (".npz", ".csv", ".txt"):
                    path = Path(directory) / ("model" + extension)
                    save_model(path, fitted)
                    restored = load_model(path)
                    self.assertEqual(restored.sample_rate, 10e6)
                    self.assertEqual(restored.fit_info, fitted.fit_info)
                    np.testing.assert_allclose(restored.predict(test_x), original.predict(test_x), atol=1e-12)

    def test_correlated_noisy_training_generalizes_without_exploding_coefficients(self):
        from scipy.signal import resample_poly
        rng = np.random.default_rng(18)
        noise = lambda size: rng.normal(size=size) + 1j * rng.normal(size=size)
        x = resample_poly(noise(512) / 3, 8, 1)
        physical_pa = lambda signal: 20 * signal - 2 * signal * abs(signal)**2
        y = physical_pa(x) + 0.002 * noise(len(x))
        fitted = fit_model(x, y, "gmp", 5, 5, 4)
        # Independent excitation includes a small wideband perturbation absent
        # from the narrowband training signal, exposing cancelling coefficients.
        z = resample_poly(noise(512) / 3, 8, 1) + 0.002 * noise(4096)
        valid = fitted.valid_slice(len(z))
        actual, expected = fitted.predict(z)[valid], physical_pa(z)[valid]
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.002)
        self.assertLess(abs(fitted.coefficients).max(), 25)
        self.assertLess(fitted.fit_info["rank"], fitted.coefficients.size)

    def test_legacy_files_and_cm_compatibility(self):
        from modules.pa_model import mp_signal_generator
        with tempfile.TemporaryDirectory() as directory:
            for kind, columns in (("memPoly", 3), ("ctMemPoly", 5)):
                coefficient = np.arange(2 * columns).reshape(2, columns) * (0.01 + 0.03j)
                path = Path(directory) / "legacy.npz"
                np.savez(path, fitCoefMatMem=coefficient, modType=kind, memLen=2, degLen=3)
                restored = load_model(path)
                np.testing.assert_allclose(restored.predict(self.x), mp_signal_generator(self.x, coefficient, kind), atol=1e-12)
                self.assertIsNone(restored.sample_rate)
                text_path = Path(directory) / "legacy.csv"
                with text_path.open("w", encoding="utf-8") as stream:
                    stream.write(f"# 模型类型, CM ({kind})\n# memLen, 2\n# degLen, 3\n")
                    for row in coefficient:
                        stream.write(",".join(str(v) for v in row) + "\n")
                np.testing.assert_allclose(load_model(text_path).predict(self.x), restored.predict(self.x))

    def test_spectra_preserve_gain_and_integrated_power(self):
        rate = 1024
        x = np.exp(2j * np.pi * 128 * np.arange(4096) / rate)
        model = PAModel([[2j]], "memPoly", 1, sample_rate=rate)
        x, y, valid = simulate(model, x, rate)
        frequency, px, py = spectrum_pair(x, y, rate, valid, 1024)
        peak = np.argmax(px)
        self.assertAlmostEqual(frequency[peak], 128)
        self.assertAlmostEqual(py[peak] - px[peak], 20 * np.log10(2), places=10)
        self.assertAlmostEqual(np.sum(10 ** (px / 10)) * (frequency[1] - frequency[0]), 1, places=10)
        _, px_watts, _ = spectrum_pair(x, y, rate, valid, 1024, 100)
        self.assertAlmostEqual(px_watts[peak] - px[peak], 10, places=10)
        _, zero, _ = spectrum_pair(x * 0, y * 0, rate, valid)
        self.assertTrue(np.all(np.isfinite(zero)))

    def test_nonlinear_two_tone_intermodulation(self):
        rate, size = 4096, 8192
        n = np.arange(size)
        x = 0.2 * (np.exp(2j*np.pi*100*n/rate) + np.exp(2j*np.pi*300*n/rate))
        model = PAModel([[1, 0, -0.5]], "memPoly", 3)
        y = model.predict(x)
        f, px, py = spectrum_pair(x, y, rate, slice(0, size), 4096)
        for imd_frequency in (-100, 500):
            bin_idx = np.where(f == imd_frequency)[0][0]
            self.assertGreater(py[bin_idx] - px[bin_idx], 100)

    def test_waveform_formats_and_validation(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.save(base / "wave.npy", self.x)
            np.savez(base / "wave.npz", x=self.x, sampleRate=10e6)
            savemat(base / "wave.mat", {"x": self.x, "sampleRate": 10e6})
            savemat(str(base / "Data100MHz"), {"results": {"InputWaveform": self.x}, "sampleRate": 10e6}, appendmat=False)
            actual, actual_rate = load_waveform(base / "Data100MHz")
            np.testing.assert_allclose(actual, self.x)
            self.assertEqual(actual_rate, 10e6)
            np.savetxt(base / "wave.csv", np.column_stack((self.x.real, self.x.imag)), delimiter=",", header="I,Q", comments="")
            for extension in (".npy", ".npz", ".csv", ".mat"):
                x, rate = load_waveform(base / ("wave" + extension))
                np.testing.assert_allclose(x, self.x)
                self.assertEqual(rate, 10e6 if extension in (".npz", ".mat") else None)
            np.savez(base / "bad.npz", fitCoefMatMem=[[1]], modType="unknown", memLen=1, degLen=1)
            with self.assertRaises(ValueError):
                load_model(base / "bad.npz")
        model = PAModel([[1]], "memPoly", 1, sample_rate=10e6)
        with self.assertRaisesRegex(ValueError, "采样率"):
            simulate(model, self.x, 20e6)
        for bad in ([1, 2], [1, 2, 3, np.nan]):
            with self.assertRaises(ValueError):
                model.predict(bad)
        with self.assertRaises(ValueError):
            PAModel([[1, 2]], "gmp", 3, 1)

    def test_power_spectrum_matches_original_characterization_convention(self):
        size, rate, load = 1024, 10e6, 100
        win = np.hanning(size)
        powers = [abs(np.fft.fftshift(np.fft.fft(self.x[i:i+size]*win)))**2
                  for i in range(0, len(self.x)-size+1, size//2)]
        power = np.mean(powers, axis=0)
        expected = 10*np.log10(power/power.sum()*np.mean(abs(self.x)**2)/load/1e-3 + 1e-30)
        f, actual = power_spectrum(self.x, rate, size, load)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        original = self.x.copy()
        power_spectrum(self.x, rate, size, load, -77)
        np.testing.assert_array_equal(original, self.x)
        _, zero = power_spectrum(self.x*0, rate)
        self.assertTrue(np.isfinite(zero).all())

    def test_sionna_matlab_preset_grid_cp_and_roundtrip(self):
        from modules.pa_ofdm import _modulate
        x, grid, info = _modulate(50e6)
        self.assertEqual(info["sampleRate"], 61.44e6)
        self.assertEqual(info["numSymbols"], 140)
        self.assertEqual(info["activeCarriers"], 270*12)
        self.assertEqual(len(x), 614400)  # Exactly ten 1-ms subframes.
        self.assertFalse(info["windowing"])
        self.assertFalse(info["filtering"])
        self.assertEqual(set(info["cyclicPrefixLengths"]), {288, 320})
        cursor = 0
        recovered = []
        for cp in info["cyclicPrefixLengths"]:
            symbol = x[cursor:cursor+cp+4096]
            np.testing.assert_allclose(symbol[:cp], symbol[-cp:], atol=1e-14)
            recovered.append(np.fft.fftshift(np.fft.fft(symbol[cp:]))/np.sqrt(4096))
            cursor += cp+4096
        np.testing.assert_allclose(recovered, grid, atol=1e-12)
        np.testing.assert_array_equal(np.count_nonzero(grid,axis=1), 3240)

    def test_sionna_oversampling_preserves_grid_without_post_filter(self):
        from modules.pa_ofdm import _modulate,generate_ofdm
        x1, grid1, a = _modulate(5e6, 1)
        x7, grid7, b = _modulate(5e6, 7)
        self.assertEqual(b["sampleRate"], a["sampleRate"]*7)
        self.assertEqual(len(x7), len(x1)*7)
        np.testing.assert_allclose(grid7[:, 768:1024],grid1)
        # Retain natural sidelobes of the serialized OFDM signal. The old
        # record-wide FFT interpolator imposed a hard cutoff at this limit.
        f = np.fft.fftfreq(len(x7),1/b["sampleRate"])
        p = abs(np.fft.fft(x7))**2
        self.assertGreater(p[abs(f)>a["sampleRate"]/2].sum()/p.sum(),1e-10)
        normalized, _, _ = generate_ofdm(5e6,7)
        self.assertAlmostEqual(abs(normalized).max(),1)


class PAWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_pa_numeric_buttons_use_ber_controls_and_edge_hit_areas(self):
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest
        from ui.numeric_inputs import IntegerInput, DecimalInput
        from modules.pa_model import PAModelDialog
        window = PAModelDialog()
        self.addCleanup(window.close)
        window.show()
        page = window.simulation_page
        window.tabs.setCurrentIndex(1)
        page.source.setCurrentIndex(1)
        fields = (window.e_memlen, window.e_deglen, window.e_crossdepth,
                  page.oversampling, page.rate, page.level, page.nfft, page.load)
        window.combo_model.setCurrentIndex(1)
        page.rate.setEnabled(True)
        self.app.processEvents()
        for field in fields:
            with self.subTest(field=type(field).__name__):
                self.assertIsInstance(field, (IntegerInput, DecimalInput))
                if field is page.rate:
                    # Normally locked by the generated waveform's sample rate.
                    field.setEnabled(True)
                # Leave both directions enabled, including the initial sentinel rate.
                field.setValue(min(field.maximum() - field.singleStep(),
                                   max(field.minimum() + field.singleStep(), field.value())))
                before = field.value()
                for button, delta in ((field.up_button, 1), (field.down_button, -1)):
                    self.assertGreaterEqual(button.width(), 30)
                    self.assertGreaterEqual(button.height(), 18)
                    point = QPoint(button.width() - 3, button.height() - 3)
                    self.assertIs(field.childAt(button.pos() + point), button)
                    value = field.value()
                    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=point)
                    self.assertAlmostEqual(field.value(), value + delta * field.singleStep())
                self.assertAlmostEqual(field.value(), before)

    def test_independent_import_simulation_export_and_invalidation(self):
        from modules.pa_model import PAModelDialog
        window = PAModelDialog()
        self.addCleanup(window.close)
        self.assertEqual(window.tabs.count(), 3)
        self.assertIsNone(window.pa_data)
        page = window.simulation_page
        window.tabs.setCurrentIndex(1)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            save_model(base / "model.npz", PAModel([[2j]], "memPoly", 1, sample_rate=1e6))
            x = np.exp(2j * np.pi * np.arange(4096) / 16)
            np.savez(base / "input.npz", x=x, sampleRate=1e6)
            page.set_model_file(base / "model.npz")
            page.set_wave_file(base / "input.npz")
            self.assertTrue(page.run_button.isEnabled())
            with patch("modules.pa_simulation.QMessageBox.warning") as warning:
                page.run_button.click()
                warning.assert_not_called()
            np.testing.assert_allclose(page.result["y"], 2j * x)
            self.assertEqual(len(page.figure.axes), 1)
            self.assertEqual(len(page.figure.axes[0].lines), 2)
            self.assertEqual(page.figure.axes[0].get_ylabel(), "功率 (dBm)")
            self.assertEqual(page.figure.axes[0].lines[0].get_label(), "输入功率谱")
            self.assertNotIn("x[n]", page.figure.axes[0].get_title())
            self.assertEqual(page.controls_scroll.width(), window.fit_controls.width())
            output = base / "result.npz"
            with patch("modules.pa_simulation.QFileDialog.getSaveFileName", return_value=(str(output), "")):
                page.export_result()
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_allclose(data["y"], 2j * x)
                self.assertEqual(data["sampleRate"], 1e6)
            page.level.setValue(-3)
            self.assertIsNone(page.result)
            self.assertFalse(page.export_button.isEnabled())
            self.assertFalse(page.save_button.isEnabled())

    def test_fitting_exports_actual_model_parameters(self):
        from modules.pa_model import PAModelDialog, PAData
        window = PAModelDialog()
        self.addCleanup(window.close)
        rng = np.random.default_rng(3)
        x = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) / 5
        window.pa_data = PAData(input_wave=x, output_wave=2j*x, sample_rate=10e6,
                                reference_power=np.array([-20, -10, 0]),
                                measured_amam=np.array([6.01, 6.02, 6.03]))
        window.combo_model.setCurrentIndex(1)
        window.e_memlen.setValue(2)
        window.e_deglen.setValue(3)
        window._do_fit()
        self.assertEqual(window.fitted_model.name, "GMP")
        self.assertLess(window._last_rms, 1e-8)
        window.e_memlen.setValue(9)  # Editing UI does not relabel the fitted model.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fitted.npz"
            with patch("modules.pa_model.QFileDialog.getSaveFileName", return_value=(str(path), "")), patch("modules.pa_model.QMessageBox.information"):
                window._export()
            restored = load_model(path)
            self.assertEqual(restored.memory, 2)
            self.assertAlmostEqual(restored.fit_info["validationRmsPercent"], window._last_validation_rms)
            window.simulation_page.set_model_file(path)
            np.testing.assert_allclose(restored.predict(x), window.pa_output_fit)
            np.savez(Path(directory)/"x.npz", x=x, sampleRate=10e6)
            window.simulation_page.set_wave_file(Path(directory)/"x.npz")
            with patch("modules.pa_simulation.QMessageBox.warning") as warning:
                window.simulation_page.run_simulation()
                warning.assert_not_called()
            window._show_output_spectrum()
            fit_line = window.canvas.fig.axes[0].lines[1]
            sim_line = window.simulation_page.figure.axes[0].lines[1]
            np.testing.assert_allclose(fit_line.get_xdata(), sim_line.get_xdata())
            np.testing.assert_allclose(fit_line.get_ydata(), sim_line.get_ydata(), atol=1e-12)

    def test_generated_ofdm_and_rate_mismatch(self):
        from modules.pa_model import PAModelDialog
        window = PAModelDialog()
        self.addCleanup(window.close)
        page = window.simulation_page
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_model(path, PAModel([[1]], "memPoly", 1, sample_rate=53.76e6))
            page.set_model_file(path)
            page.bandwidth.setCurrentText("5")
            page.source.setCurrentIndex(1)
            self.assertTrue(page.run_button.isEnabled())
            self.assertFalse(page.rate.isEnabled())
            self.assertFalse(page.wave_button.isEnabled())
            self.assertFalse(page.wave_file.isEnabled())
            self.assertTrue(page.bandwidth.isEnabled())
            self.assertAlmostEqual(page.rate.value(), 53.76)
            with patch("modules.pa_simulation.QMessageBox.warning") as warning:
                page.run_simulation()
                warning.assert_not_called()
                np.testing.assert_allclose(page.result["x"], page.result["y"])
                self.assertGreater(len(page.result["x"]), 1000)
                page.bandwidth.setCurrentIndex(1)
                self.assertIsNone(page.result)
                page.run_simulation()
                warning.assert_called_once()
                self.assertIn("采样率", str(warning.call_args))
                self.assertIsNone(page.result)
                self.assertFalse(page.export_button.isEnabled())
            page.source.setCurrentIndex(0)
            self.assertTrue(page.wave_button.isEnabled())
            self.assertTrue(page.wave_file.isEnabled())
            self.assertFalse(page.bandwidth.isEnabled())

    def test_matlab_preset_input_preview_without_pa(self):
        from modules.pa_model import PAModelDialog
        window = PAModelDialog()
        self.addCleanup(window.close)
        page = window.simulation_page
        page.source.setCurrentIndex(1)
        self.assertEqual(page.bandwidth.currentText(), "50")
        self.assertEqual(page.oversampling.value(), 1)
        self.assertAlmostEqual(page.rate.value(), 61.44)
        self.assertFalse(page.run_button.isEnabled())
        self.assertTrue(page.preview_button.isEnabled())
        with patch("modules.pa_simulation.QMessageBox.warning") as warning:
            page.preview_input()
            warning.assert_not_called()
        self.assertEqual(len(page.figure.axes[0].lines),1)
        self.assertEqual(page.figure.axes[0].lines[0].get_label(),"输入功率谱")
        self.assertTrue(page.save_button.isEnabled())
        self.assertFalse(page.export_button.isEnabled())
        page.oversampling.setValue(14)
        self.assertAlmostEqual(page.rate.value(),860.16)
        self.assertFalse(page.has_plot)

    def test_main_window_opens_pa_with_both_workflows(self):
        from ui.main_window import MainWindow
        from ui.dialogs import PAModelDialog
        main = MainWindow()
        self.addCleanup(main.close)
        main._open("pa", PAModelDialog)
        window = main._open_dialogs["pa"]
        self.addCleanup(window.close)
        self.assertEqual(window.tabs.count(), 3)
        self.assertIsNone(window.pa_data)
        self.assertFalse(window.simulation_page.run_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
