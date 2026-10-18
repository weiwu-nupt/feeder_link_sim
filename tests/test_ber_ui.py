"""Offscreen checks using the platform's real spawned simulation process."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint, QPointF
from PyQt6.QtGui import QFont, QFontDatabase, QWheelEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from modules.ber_analysis import BERAnalysisDialog
from ui.ber_editors import APSKEditor, CodingEditor


class BERUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        for font in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
            if Path(font).exists():
                QFontDatabase.addApplicationFont(font)
        cls.app.setFont(QFont("Microsoft YaHei", 10))

    def wait_until(self, condition, seconds=60):
        until = time.monotonic() + seconds
        while not condition() and time.monotonic() < until:
            loop = QEventLoop()
            QTimer.singleShot(50, loop.quit)
            loop.exec()
        self.assertTrue(condition(), "Timed out")

    def test_process_simulation_export_and_close(self):
        window = BERAnalysisDialog()
        self.addCleanup(window.close)
        window.showMaximized()
        self.wait_until(lambda: window.ready)
        self.assertTrue(window.isMaximized())
        window.family.setCurrentIndex(window.family.findData("none"))
        window.snr_start.setValue(10)
        window.snr_stop.setValue(10)
        window.budget.setValue(1025)
        window.device.setCurrentIndex(window.device.findData("cpu"))
        window.start_run()
        self.assertFalse(window.start_button.isEnabled())
        self.wait_until(lambda: window.session.process is None)
        self.assertEqual(window.table.rowCount(), 1)
        self.assertEqual(window.runs[0]["rows"][0]["bits"], 1025)
        self.assertEqual(window.runs[0]["rows"][0]["bit_errors"], 0)
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "results.csv"
            png = Path(directory) / "curve.png"
            with patch(
                "modules.ber_analysis.QFileDialog.getSaveFileName", return_value=(str(csv), "")
            ):
                window.export_data()
            with patch(
                "modules.ber_analysis.QFileDialog.getSaveFileName", return_value=(str(png), "")
            ):
                window.export_image()
            self.assertIn("1025", csv.read_text(encoding="utf-8-sig"))
            self.assertGreater(png.stat().st_size, 1000)
        window.budget.setValue(1000000)
        window.start_run()
        self.wait_until(lambda: window.session.process is not None)
        window.stop_run()
        self.wait_until(lambda: window.session.process is None)
        self.assertTrue(window.start_button.isEnabled())
        window.clear_results()
        self.assertEqual(window.table.rowCount(), 0)

    def test_code_validation_invalidation_and_apsk_edit(self):
        dialog = CodingEditor(
            definition=dict(family="conv", k=40, length=3, polys=["7", "5"], terminate=True)
        )
        self.addCleanup(dialog.reject)
        dialog.k.setValue(40)
        dialog.length.setValue(3)
        dialog.polys.setText("7,5")
        dialog.run_validation()
        self.wait_until(lambda: dialog.session.process is None)
        self.assertIsNotNone(dialog.verified, dialog.status.text())
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.terminate.setChecked(False)
        self.assertIsNone(dialog.verified)
        self.assertFalse(dialog.save_button.isEnabled())
        apsk = APSKEditor()
        self.addCleanup(apsk.reject)
        values = []
        apsk.saved.connect(values.append)
        apsk.save()
        self.assertEqual(sum(values[0]["counts"]), 16)

    def test_step_buttons_have_matching_hit_regions(self):
        from ui.ber_editors import integer, decimal

        for field in (integer(20, 0, 100), decimal(3, 0, 10, 0.5)):
            with self.subTest(field=type(field).__name__):
                self.addCleanup(field.close)
                field.resize(240, 44)
                field.show()
                self.app.processEvents()
                original = field.value()
                step = field.singleStep()
                for button, delta in ((field.up_button, 1), (field.down_button, -1)):
                    self.assertGreaterEqual(button.width(), 30)
                    self.assertGreaterEqual(button.height(), 18)
                    for point in (QPoint(2, 2), QPoint(button.width() - 3, button.height() - 3)):
                        before = field.value()
                        QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=point)
                        self.assertAlmostEqual(field.value(), before + delta * step)
                self.assertAlmostEqual(field.value(), original)

    def test_live_apsk_drag_zoom_and_validation_share_mapping(self):
        from modules.ber_core.catalog import custom_points

        editor = APSKEditor()
        self.addCleanup(editor.reject)
        editor.show()
        editor.resize(1180, 780)
        self.app.processEvents()
        canvas = editor.canvas
        for order in (4, 8, 16, 32, 64, 128, 256):
            editor.order.setCurrentText(str(order))
            self.app.processEvents()
            self.assertEqual(len(canvas.points), order)
            self.assertAlmostEqual(sum(abs(p) ** 2 for p in canvas.points) / order, 1)
        editor.order.setCurrentText("16")
        self.app.processEvents()
        radius = editor.table.cellWidget(1, 1)
        radius.setValue(3.1)
        self.app.processEvents()
        self.assertEqual(canvas.points, custom_points(editor.draft))
        before = (radius.value(), editor.table.cellWidget(1, 2).value())
        item = next(item for item in canvas.point_geometry() if item[2] == 1 and item[3] == 0)
        pixel = canvas.to_screen(item[0]).toPoint()
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pixel)
        QTest.mouseMove(canvas, pixel + QPoint(24, -18))
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pixel + QPoint(24, -18))
        self.assertIsNone(canvas.drag)
        self.assertNotEqual((radius.value(), editor.table.cellWidget(1, 2).value()), before)
        self.assertEqual(canvas.points, custom_points(editor.draft))
        position = QPointF(canvas.plot_rect().center())
        global_pos = QPointF(canvas.mapToGlobal(position.toPoint()))
        wheel = QWheelEvent(
            position,
            global_pos,
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.app.sendEvent(canvas, wheel)
        self.assertGreater(canvas.zoom, 1)
        QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=position.toPoint())
        self.assertEqual(canvas.zoom, 1)
        mapping = list(canvas.points)
        editor.show_bits.setChecked(False)
        self.assertFalse(canvas.show_bits)
        self.assertEqual(canvas.points, mapping)
        count = editor.table.cellWidget(0, 0)
        count.setValue(5)
        self.assertFalse(editor.save_button.isEnabled())
        self.assertEqual(canvas.points, [])
        count.setValue(4)
        self.assertTrue(editor.save_button.isEnabled())
        self.assertEqual(canvas.points, custom_points(editor.draft))

    def test_curve_connects_zero_error_triangles_and_has_no_constellation_tab(self):
        window = BERAnalysisDialog()
        self.addCleanup(window.close)
        self.wait_until(lambda: window.ready)
        self.assertEqual(
            [window.tabs.tabText(i) for i in range(window.tabs.count())], ["误码率曲线", "逐点统计"]
        )
        window.runs = [
            dict(
                label="sample",
                rows=[
                    dict(ebno=0, ber=0.1, bit_errors=10),
                    dict(ebno=1, ber=0, bit_errors=0, upper=0.03),
                    dict(ebno=2, ber=0.01, bit_errors=1),
                    dict(ebno=3, ber=0, bit_errors=0, upper=0.003),
                    dict(ebno=4, ber=0, bit_errors=0, upper=0.003),
                ],
            )
        ]
        window.draw_curves()
        axes = window.figure.axes[0]
        self.assertEqual(len(axes.lines), 3)
        self.assertEqual(axes.lines[0].get_linestyle(), "-")
        self.assertEqual(list(axes.lines[0].get_ydata()), [0.1, 0.03, 0.01])
        self.assertEqual(axes.lines[1].get_marker(), "o")
        self.assertEqual(axes.lines[2].get_marker(), "v")
        self.assertEqual(axes.lines[2].get_markerfacecolor(), "white")
        self.assertEqual(len(axes.lines[2].get_xdata()), 1)
        self.assertEqual(len(window.runs[0]["rows"]), 5)
        self.assertEqual(window.runs[0]["rows"][1]["ber"], 0)
        window.runs[0]["rows"] = [dict(ebno=i, ber=0, bit_errors=0, upper=0.003) for i in (0, 1, 2)]
        window.draw_curves()
        self.assertEqual(list(window.figure.axes[0].lines[0].get_ydata()), [0.003])
        self.assertEqual(window.figure.axes[0].lines[1].get_marker(), "v")

    def test_ring_count_click_redistributes_points_and_respects_order(self):
        editor = APSKEditor()
        self.addCleanup(editor.reject)
        editor.show()
        self.app.processEvents()
        QTest.mouseClick(editor.ring_count.up_button, Qt.MouseButton.LeftButton)
        self.assertEqual(editor.ring_count.value(), 3)
        self.assertEqual(editor.table.rowCount(), 3)
        self.assertEqual(editor.draft["counts"], [3, 5, 8])
        self.assertTrue(editor.save_button.isEnabled())
        for order in (4, 8, 16, 32, 64, 128, 256):
            editor.order.setCurrentText(str(order))
            self.assertEqual(editor.ring_count.maximum(), min(16, order))
            for count in range(1, min(16, order) + 1):
                editor.ring_count.setValue(count)
                self.assertIsNotNone(editor.draft, editor.validation.text())
                self.assertEqual(len(editor.draft["counts"]), count)
                self.assertEqual(sum(editor.draft["counts"]), order)
                self.assertEqual(len(editor.canvas.points), order)
                self.assertTrue(editor.save_button.isEnabled())
            self.assertFalse(editor.ring_count.up_button.isEnabled())
        editor.order.setCurrentText("16")
        editor.ring_count.lineEdit().setText("3")
        values = []
        editor.saved.connect(values.append)
        editor.save()
        self.assertEqual(len(values[0]["counts"]), 3)

    def test_type_specific_coding_fields_preserve_drafts_and_qpp_lengths(self):
        from ui.coding_editor import QPP_LENGTHS

        dialog = CodingEditor()
        self.addCleanup(dialog.reject)
        dialog.show()
        self.app.processEvents()
        conv = dialog.pages["conv"]
        conv.k.setValue(99)
        conv.polys.setText("171,133")
        self.assertEqual(dialog.definition()["k"], 99)
        dialog.kind.setCurrentIndex(dialog.kind.findData("turbo"))
        turbo = dialog.pages["turbo"]
        self.app.processEvents()
        self.assertFalse(conv.isVisible())
        self.assertTrue(turbo.isVisible())
        self.assertFalse(turbo.custom_panel.isVisible())
        self.assertEqual(len(QPP_LENGTHS), 188)
        self.assertEqual(dialog.definition()["polys"], ["13", "15"])
        turbo.qpp_k.setCurrentText("41")
        self.assertIsNone(dialog.draft)
        self.assertFalse(dialog.validate.isEnabled())
        turbo.qpp_k.setCurrentText("40")
        self.assertEqual(dialog.definition()["k"], 40)
        turbo.interleaver.setCurrentIndex(turbo.interleaver.findData("custom"))
        turbo.k.setValue(8)
        self.assertTrue(turbo.custom_panel.isVisible())
        self.assertFalse(dialog.validate.isEnabled())
        turbo.permutation.setPlainText("2 0 3 1 6 4 7 5")
        self.assertEqual(dialog.definition()["permutation"], [2, 0, 3, 1, 6, 4, 7, 5])
        self.assertTrue(dialog.validate.isEnabled())
        dialog.kind.setCurrentIndex(dialog.kind.findData("qc"))
        dialog.pages["qc"].text.setPlainText("0 1 0 -1\n1 0 -1 0")
        self.assertFalse(turbo.isVisible())
        self.assertTrue(dialog.pages["qc"].isVisible())
        dialog.kind.setCurrentIndex(dialog.kind.findData("h"))
        self.assertFalse(dialog.pages["qc"].isVisible())
        dialog.pages["h"].text.setPlainText("1 1 1")
        self.assertEqual(dialog.definition()["format"], "h")
        dialog.kind.setCurrentIndex(dialog.kind.findData("conv"))
        self.assertEqual(dialog.k.value(), 99)
        dialog.kind.setCurrentIndex(dialog.kind.findData("qc"))
        self.assertEqual(dialog.text.toPlainText(), "0 1 0 -1\n1 0 -1 0")

    def test_all_coding_pages_validate_save_and_reopen(self):
        definitions = [
            dict(family="conv", k=1, length=3, polys=["7", "5"], terminate=True),
            dict(
                family="turbo",
                k=40,
                length=3,
                polys=["7", "5"],
                terminate=True,
                rate="1/3",
                interleaver="qpp",
            ),
            dict(family="qc", text="0 1 0 -1\n1 0 -1 0", z=3, direction=1, format="qc"),
            dict(family="h", format="h", text="1 0 1 0 1 0 1\n0 1 1 0 0 1 1\n0 0 0 1 1 1 1"),
        ]
        for definition in definitions:
            with self.subTest(family=definition["family"]):
                dialog = CodingEditor(definition=definition)
                self.addCleanup(dialog.reject)
                self.assertFalse(dialog.kind.isEnabled())
                dialog.run_validation()
                self.wait_until(lambda: dialog.session.process is None)
                self.assertIsNotNone(dialog.verified, dialog.status.text())
                saved = []
                dialog.saved.connect(saved.append)
                dialog.save()
                self.assertEqual(len(saved), 1)
                reopened = CodingEditor(definition=saved[0])
                self.addCleanup(reopened.reject)
                for key, value in definition.items():
                    self.assertEqual(reopened.definition()[key], value)
                self.assertFalse(reopened.save_button.isEnabled())

    def test_new_plain_codings_save_directly_and_imports_still_require_validation(self):
        for family in ("conv", "turbo"):
            dialog = CodingEditor()
            self.addCleanup(dialog.reject)
            dialog.kind.setCurrentIndex(dialog.kind.findData(family))
            self.assertFalse(dialog.needs_validation())
            self.assertTrue(dialog.save_button.isEnabled())
            values = []
            dialog.saved.connect(values.append)
            dialog.save()
            self.assertEqual(len(values), 1)
            self.assertGreater(values[0]["n"], values[0]["k"])
        dialog = CodingEditor()
        self.addCleanup(dialog.reject)
        dialog.kind.setCurrentIndex(dialog.kind.findData("turbo"))
        turbo = dialog.pages["turbo"]
        turbo.interleaver.setCurrentIndex(turbo.interleaver.findData("custom"))
        turbo.k.setValue(8)
        turbo.permutation.setPlainText("2 0 3 1 6 4 7 5")
        self.assertTrue(dialog.needs_validation())
        self.assertFalse(dialog.save_button.isEnabled())
        values = []
        dialog.saved.connect(values.append)
        dialog.save()
        self.assertEqual(values, [])
        dialog.kind.setCurrentIndex(dialog.kind.findData("qc"))
        self.assertTrue(dialog.needs_validation())


if __name__ == "__main__":
    unittest.main()
