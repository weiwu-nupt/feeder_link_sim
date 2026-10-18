"""主窗口"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel,
    QMenu, QStatusBar, QFrame, QToolBar, QToolButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from ui.home_workspace import HomeWorkspace

from ui.styles import APP_STYLE, light_palette
from ui.dialogs import (
    LinkBudgetDialog, ChannelModelDialog,
    PAModelDialog, ADDAModelDialog, FilterModelDialog, MixerModelDialog,
    SingleLinkSimDialog, MultiLinkSimDialog, BERAnalysisDialog, RainSequenceDialog
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setPalette(light_palette())
        self.setWindowTitle("馈电链路仿真平台  v0.1.0")
        self.setMinimumSize(960, 640)
        self.resize(1100, 720)
        self.setStyleSheet(APP_STYLE)
        self._open_dialogs: dict = {}
        self._build_toolbar()
        self._build_workspace()
        self._build_statusbar()

    # ── 工具栏（替代菜单栏） ──────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        tb.setStyleSheet("""
            QToolBar {
                background: #FFFFFF;
                border-bottom: 1px solid #E8E8E5;
                spacing: 2px;
                padding: 2px 6px;
            }
        """)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        btn_style = """
            QToolButton {
                background: transparent; border: none;
                padding: 5px 14px;
                font-family: "Microsoft YaHei", "SimHei", "Arial", sans-serif;
                font-size: 10pt; color: #5F5E5A;
                border-radius: 4px;
            }
            QToolButton:hover   { background: #F1EFE8; color: #2C2C2A; }
            QToolButton:pressed { background: #E8E6DF; color: #2C2C2A; }
            QToolButton::menu-indicator { image: none; }
        """

        menu_style = """
            QMenu {
                background:#FFFFFF; border:1px solid #D3D1C7;
                border-radius:6px; padding:4px;
                font-size: 10pt; color:#2C2C2A;
            }
            QMenu::item { padding:6px 20px 6px 12px; border-radius:4px; }
            QMenu::item:selected { background:#F1EFE8; }
        """

        # ① 系统设计
        system_menu = QMenu()
        system_menu.setStyleSheet(menu_style)
        for label, key, cls in [
            ("链路预算", "link_budget", LinkBudgetDialog),
            ("可用性分析", "availability", ChannelModelDialog),
            ("雨衰序列生成", "rain_sequence", RainSequenceDialog),
            ("调制编码优化", "modcod", BERAnalysisDialog),
        ]:
            action = QAction(label, system_menu)
            action.triggered.connect(
                lambda _=False, k=key, c=cls: self._open(k, c))
            system_menu.addAction(action)
        self._menu_btn(tb, "系统设计", btn_style, system_menu)

        # ② 信道建模（当前器件建模能力）
        channel_menu = QMenu()
        channel_menu.setStyleSheet(menu_style)
        for label, key, cls in [
            ("功放模型",    "pa",     PAModelDialog),
            ("ADC/DAC 模型",    "adda",   ADDAModelDialog),
            ("滤波器模型",  "filter", FilterModelDialog),
            ("混频器模型", "mixer",  MixerModelDialog),
        ]:
            act = QAction(label, channel_menu)
            act.triggered.connect(
                lambda _=False, k=key, c=cls: self._open(k, c))
            channel_menu.addAction(act)
        self._menu_btn(tb, "信道建模", btn_style, channel_menu)

        # ③ 链路仿真（当前单链路仿真）
        self._plain_btn(tb, "链路仿真", btn_style,
                        lambda: self._open("single", SingleLinkSimDialog))

        # ④ 大规模星座仿真（当前多链路仿真）
        self._plain_btn(tb, "大规模星座仿真", btn_style,
                        lambda: self._open("multi", MultiLinkSimDialog))

    @staticmethod
    def _plain_btn(tb: QToolBar, label: str, style: str, slot):
        """向工具栏添加纯点击按钮，没有菜单，悬停不触发任何业务逻辑"""
        btn = QToolButton()
        btn.setText(label)
        btn.setStyleSheet(style)
        btn.clicked.connect(slot)
        tb.addWidget(btn)

    @staticmethod
    def _menu_btn(tb: QToolBar, label: str, style: str, menu: QMenu):
        """向工具栏添加点击即展开的一级导航菜单。"""
        btn = QToolButton()
        btn.setText(f"{label}  ▼")
        btn.setStyleSheet(style)
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tb.addWidget(btn)

    # ── 工作区 ────────────────────────────────────────────

    def _build_workspace(self):
        self.setCentralWidget(HomeWorkspace(self))

    # ── 状态栏 ────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setFixedHeight(28)
        ready = QLabel("●  就绪")
        ready.setStyleSheet("color:#1D9E75;font-size: 8pt;padding:0 8px;")
        sb.addWidget(ready)
        sb.addWidget(self._vsep())
        sb.addWidget(QLabel("频段: —",
            styleSheet="color:#888780;font-size: 8pt;"))
        sb.addWidget(self._vsep())
        sb.addWidget(QLabel("链路数: 0",
            styleSheet="color:#888780;font-size: 8pt;"))
        sb.addPermanentWidget(QLabel("馈电链路仿真平台  v0.1.0  "))

    @staticmethod
    def _vsep():
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet("color:#E8E8E5;")
        f.setFixedHeight(14)
        return f

    # ── 对话框管理 ────────────────────────────────────────

    def _open(self, key: str, dialog_cls):
        if key in self._open_dialogs:
            dlg = self._open_dialogs[key]
            if dlg.isVisible():
                dlg.raise_()
                dlg.activateWindow()
                return
        dlg = dialog_cls(parent=self)
        dlg.finished.connect(lambda _=None: self._open_dialogs.pop(key, None))
        self._open_dialogs[key] = dlg
        dlg.showMaximized()
