"""Responsive homepage using a credited NASA communications-satellite image."""

from pathlib import Path
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont
from PyQt6.QtWidgets import QWidget


class HomeWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QPixmap(str(Path(__file__).parent / "assets" / "tdrs-earth.jpg"))
        self.setMinimumSize(640, 400)
        self.setToolTip("TDRS 通信中继卫星概念图 · 图片来源 NASA")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#0B1826"))
        if not self.image.isNull():
            scaled = self.image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(
                (self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2, scaled
            )
        gradient = QLinearGradient(0, 0, self.width() * 0.85, 0)
        gradient.setColorAt(0, QColor(5, 15, 29, 205))
        gradient.setColorAt(0.55, QColor(5, 15, 29, 140))
        gradient.setColorAt(1, QColor(5, 15, 29, 0))
        painter.fillRect(self.rect(), gradient)
        bottom = QLinearGradient(0, self.height() * 0.7, 0, self.height())
        bottom.setColorAt(0, QColor(5, 15, 29, 0))
        bottom.setColorAt(1, QColor(5, 15, 29, 200))
        painter.fillRect(self.rect(), bottom)
        margin = max(36, min(80, int(self.width() * 0.05)))
        top = int(self.height() * 0.3)
        painter.setPen(QColor("#9DBED8"))
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(margin, top, "FEEDER LINK SIMULATION")
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Microsoft YaHei", 25, QFont.Weight.DemiBold))
        painter.drawText(margin, top + 66, "馈电链路仿真平台")
        painter.setPen(QColor("#DBE4EB"))
        painter.setFont(QFont("Microsoft YaHei", 11))
        painter.drawText(margin, top + 110, "面向卫星通信的系统设计与链路性能分析")
        painter.setPen(QColor("#AFC4D5"))
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(
            margin,
            self.height() - 86,
            "系统设计   /   信道建模   /   链路仿真   /   大规模星座仿真",
        )
        painter.setPen(QColor("#8FA6B8"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(margin, self.height() - 57, "从上方菜单选择模块开始")
        painter.setPen(QColor("#B6C5D1"))
        painter.setFont(QFont("Microsoft YaHei", 8))
        painter.drawText(
            QRectF(margin, self.height() - 35, self.width() - 2 * margin, 20),
            Qt.AlignmentFlag.AlignRight,
            "TDRS 通信卫星概念图 · NASA",
        )
