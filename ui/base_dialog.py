"""
模块对话框基类
Base Dialog for Module Windows
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont, QIcon, QPixmap


class ModuleDialog(QDialog):
    """
    所有功能模块弹窗的基类。
    继承此类并重写 build_content() 方法来填充具体内容。
    """

    # 子类可覆盖的属性
    TITLE = "模块"
    SUBTITLE = ""
    ACCENT_COLOR = "#378ADD"   # 标题栏强调色
    MIN_WIDTH = 680
    MIN_HEIGHT = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.setModal(False)        # 非模态，允许同时打开多个窗口
        self._build_shell()

    # ── 框架搭建 ──────────────────────────────────────────

    def _build_shell(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 标题栏
        root.addWidget(self._make_header())

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #E8E8E5;")
        root.addWidget(sep)

        # 内容区（子类实现）
        content_wrapper = QFrame()
        content_wrapper.setStyleSheet("background: #FAFAF8;")
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(12)

        self.content_layout = content_layout
        self.build_content(content_layout)

        # 若子类未添加内容，显示占位
        if content_layout.count() == 0:
            self._add_placeholder(content_layout)

        root.addWidget(content_wrapper, stretch=1)

    def _make_header(self):
        header = QFrame()
        header.setFixedHeight(52)
        header.setStyleSheet("background: #FFFFFF;")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # 色块图标
        icon_label = QLabel()
        icon_label.setFixedSize(28, 28)
        icon_label.setStyleSheet(f"""
            background-color: {self._lighten(self.ACCENT_COLOR)};
            border-radius: 7px;
        """)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # 标题
        title_label = QLabel(self.TITLE)
        title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 500;
            color: #2C2C2A;
        """)
        layout.addWidget(title_label)

        if self.SUBTITLE:
            sub_label = QLabel(self.SUBTITLE)
            sub_label.setStyleSheet("font-size: 12px; color: #888780;")
            layout.addWidget(sub_label)

        layout.addStretch()

        # 关闭按钮
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                color: #888780;
            }
            QPushButton:hover {
                background: #F1EFE8;
                color: #2C2C2A;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        return header

    def _add_placeholder(self, layout):
        """默认占位内容"""
        placeholder = QFrame()
        placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        ph_layout = QVBoxLayout(placeholder)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.setSpacing(10)

        icon = QLabel("⚙")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"""
            font-size: 28px;
            color: {self.ACCENT_COLOR};
            background: {self._lighten(self.ACCENT_COLOR)};
            border-radius: 14px;
            padding: 14px;
        """)
        icon.setFixedSize(60, 60)

        title = QLabel(f"{self.TITLE} — 开发中")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 15px; font-weight: 500; color: #5F5E5A;")

        desc = QLabel("该模块正在建设中，敬请期待。")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 13px; color: #888780;")

        ph_layout.addStretch()
        ph_layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignCenter)
        ph_layout.addSpacing(8)
        ph_layout.addWidget(title)
        ph_layout.addWidget(desc)
        ph_layout.addStretch()

        layout.addWidget(placeholder)

    # ── 子类接口 ──────────────────────────────────────────

    def build_content(self, layout: QVBoxLayout):
        """
        子类重写此方法，向 layout 中添加具体内容。
        若不重写，则显示默认占位界面。
        """
        pass

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _lighten(hex_color: str, alpha: float = 0.12) -> str:
        """返回颜色的淡色背景版本（用于图标背景）"""
        COLOR_MAP = {
            "#378ADD": "#E6F1FB",
            "#1D9E75": "#E1F5EE",
            "#BA7517": "#FAEEDA",
            "#7F77DD": "#EEEDFE",
            "#D85A30": "#FAECE7",
        }
        return COLOR_MAP.get(hex_color, "#F1EFE8")