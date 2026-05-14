"""模块对话框基类"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFrame, QSizePolicy, QLabel, QPushButton, QHBoxLayout,
)
from PyQt6.QtCore import Qt


class ModuleDialog(QDialog):
    TITLE        = "模块"
    SUBTITLE     = ""
    ACCENT_COLOR = "#378ADD"
    MIN_WIDTH    = 680
    MIN_HEIGHT   = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        # 独立窗口 + 系统原生最小化/最大化/关闭按钮，非模态
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self._build_shell()

    def _build_shell(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        wrapper = QFrame()
        wrapper.setStyleSheet("background:#FAFAF8;")
        wlay = QVBoxLayout(wrapper)
        wlay.setContentsMargins(16, 16, 16, 16)
        wlay.setSpacing(10)

        self.content_layout = wlay
        self.build_content(wlay)

        if wlay.count() == 0:
            self._add_placeholder(wlay)

        root.addWidget(wrapper, stretch=1)

    def _add_placeholder(self, layout):
        ph = QFrame()
        ph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        phl = QVBoxLayout(ph)
        phl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        phl.setSpacing(10)

        icon = QLabel("⚙")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            f"font-size: 21pt;color:{self.ACCENT_COLOR};"
            f"background:{self._lighten(self.ACCENT_COLOR)};"
            "border-radius:14px;padding:14px;")
        icon.setFixedSize(60, 60)

        phl.addStretch()
        phl.addWidget(icon, alignment=Qt.AlignmentFlag.AlignCenter)
        phl.addSpacing(8)
        lbl = QLabel(f"{self.TITLE} — 开发中")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 11pt;font-weight:500;color:#5F5E5A;")
        phl.addWidget(lbl)
        desc = QLabel("该模块正在建设中，敬请期待。")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 10pt;color:#888780;")
        phl.addWidget(desc)
        phl.addStretch()
        layout.addWidget(ph)

    def build_content(self, layout):
        pass

    @staticmethod
    def _lighten(c):
        return {"#378ADD": "#E6F1FB", "#1D9E75": "#E1F5EE",
                "#BA7517": "#FAEEDA", "#7F77DD": "#EEEDFE",
                "#D85A30": "#FAECE7"}.get(c, "#F1EFE8")