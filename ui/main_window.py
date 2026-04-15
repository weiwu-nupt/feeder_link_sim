"""
主窗口
Main Window
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel,
    QMenuBar, QMenu, QStatusBar, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont, QIcon, QColor

from ui.styles import APP_STYLE
from ui.dialogs import (
    LinkBudgetDialog,
    ChannelModelDialog,
    PAModelDialog,
    ADDAModelDialog,
    FilterModelDialog,
    MixerModelDialog,
    SingleLinkSimDialog,
    MultiLinkSimDialog,
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("馈电链路仿真平台  v0.1.0")
        self.setMinimumSize(960, 640)
        self.resize(1100, 720)
        self.setStyleSheet(APP_STYLE)

        # 记录已打开的对话框（避免重复创建）
        self._open_dialogs: dict = {}

        self._build_menubar()
        self._build_workspace()
        self._build_statusbar()

    # ── 菜单栏 ────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()
        mb.setNativeMenuBar(False)   # 确保在所有平台上显示

        # ① 链路预算
        self._add_action(mb, "链路预算  ", lambda: self._open("link_budget", LinkBudgetDialog))

        # ② 无线信道建模
        self._add_action(mb, "无线信道建模  ", lambda: self._open("channel", ChannelModelDialog))

        # ③ 器件建模（子菜单）
        device_menu = mb.addMenu("器件建模  ")
        device_menu.setObjectName("device_menu")
        self._add_submenu_action(device_menu, "功放模型",    lambda: self._open("pa",     PAModelDialog))
        self._add_submenu_action(device_menu, "AD/DA 模型", lambda: self._open("adda",   ADDAModelDialog))
        self._add_submenu_action(device_menu, "滤波器模型",  lambda: self._open("filter", FilterModelDialog))
        self._add_submenu_action(device_menu, "混频器模型",  lambda: self._open("mixer",  MixerModelDialog))

        # ④ 单链路仿真
        self._add_action(mb, "单链路仿真  ", lambda: self._open("single", SingleLinkSimDialog))

        # ⑤ 多链路仿真
        self._add_action(mb, "多链路仿真  ", lambda: self._open("multi", MultiLinkSimDialog))

    @staticmethod
    def _add_action(menubar: QMenuBar, label: str, slot):
        """向菜单栏直接添加一个顶级可点击项（伪菜单技巧）"""
        menu = menubar.addMenu(label)
        act = QAction(label.strip(), menubar)
        act.triggered.connect(slot)
        # 点击菜单标题本身时触发
        menu.aboutToShow.connect(lambda: (slot(), menu.hide()))

    @staticmethod
    def _add_submenu_action(menu: QMenu, label: str, slot):
        act = QAction(label, menu)
        act.triggered.connect(slot)
        menu.addAction(act)

    # ── 工作区 ────────────────────────────────────────────

    def _build_workspace(self):
        central = QWidget()
        central.setObjectName("workspace")
        central.setStyleSheet("background: #F5F5F3;")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)

        # 图标占位
        icon_frame = QFrame()
        icon_frame.setFixedSize(64, 64)
        icon_frame.setStyleSheet("""
            background: #EEEDFE;
            border-radius: 16px;
        """)
        icon_layout = QVBoxLayout(icon_frame)
        icon_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label = QLabel("◎")
        icon_label.setStyleSheet("font-size: 26px; color: #7F77DD; background: transparent;")
        icon_layout.addWidget(icon_label)

        # 标题
        title = QLabel("馈电链路仿真平台")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 500; color: #444441;")

        # 副标题
        subtitle = QLabel("从上方菜单选择模块开始仿真\n链路预算 · 信道建模 · 器件建模 · 单/多链路仿真")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 13px; color: #888780; line-height: 1.8;")

        layout.addStretch()
        layout.addWidget(icon_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(16)
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(subtitle)
        layout.addStretch()

    # ── 状态栏 ────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setFixedHeight(28)

        self._status_ready = QLabel("●  就绪")
        self._status_ready.setStyleSheet("color: #1D9E75; font-size: 11px; padding: 0 8px;")

        self._status_band = QLabel("频段: —")
        self._status_band.setStyleSheet("color: #888780; font-size: 11px;")

        self._status_links = QLabel("链路数: 0")
        self._status_links.setStyleSheet("color: #888780; font-size: 11px;")

        sb.addWidget(self._status_ready)
        sb.addWidget(self._sep())
        sb.addWidget(self._status_band)
        sb.addWidget(self._sep())
        sb.addWidget(self._status_links)
        sb.addPermanentWidget(QLabel("馈电链路仿真平台  v0.1.0  "))

    @staticmethod
    def _sep():
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #E8E8E5;")
        sep.setFixedHeight(14)
        return sep

    # ── 对话框管理 ────────────────────────────────────────

    def _open(self, key: str, dialog_cls):
        """
        打开或聚焦一个模块对话框。
        同一模块不重复创建，只需 raise / activateWindow。
        """
        if key in self._open_dialogs:
            dlg = self._open_dialogs[key]
            if dlg.isVisible():
                dlg.raise_()
                dlg.activateWindow()
                return
        dlg = dialog_cls(parent=self)
        dlg.finished.connect(lambda: self._open_dialogs.pop(key, None))
        self._open_dialogs[key] = dlg
        dlg.show()