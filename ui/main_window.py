"""主窗口"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel,
    QMenu, QStatusBar, QFrame, QToolBar, QToolButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from ui.styles import APP_STYLE
from ui.dialogs import (
    LinkBudgetDialog, ChannelModelDialog,
    PAModelDialog, ADDAModelDialog, FilterModelDialog, MixerModelDialog,
    SingleLinkSimDialog, MultiLinkSimDialog, BERAnalysisDialog,
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
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

        # ① 链路预算
        self._plain_btn(tb, "链路预算", btn_style,
                        lambda: self._open("link_budget", LinkBudgetDialog))

        # ② 无线信道建模
        self._plain_btn(tb, "无线信道建模", btn_style,
                        lambda: self._open("channel", ChannelModelDialog))

        # ③ 器件建模（下拉子菜单，InstantPopup 只响应点击）
        dev_menu = QMenu()
        dev_menu.setStyleSheet("""
            QMenu {
                background:#FFFFFF; border:1px solid #D3D1C7;
                border-radius:6px; padding:4px;
                font-size: 10pt; color:#2C2C2A;
            }
            QMenu::item { padding:6px 20px 6px 12px; border-radius:4px; }
            QMenu::item:selected { background:#F1EFE8; }
        """)
        for label, key, cls in [
            ("功放模型",    "pa",     PAModelDialog),
            ("AD/DA 模型", "adda",   ADDAModelDialog),
            ("滤波器模型",  "filter", FilterModelDialog),
            ("混频器模型",  "mixer",  MixerModelDialog),
        ]:
            act = QAction(label, dev_menu)
            act.triggered.connect(
                lambda _=False, k=key, c=cls: self._open(k, c))
            dev_menu.addAction(act)

        dev_btn = QToolButton()
        dev_btn.setText("器件建模  ▾")
        dev_btn.setStyleSheet(btn_style)
        dev_btn.setMenu(dev_menu)
        dev_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tb.addWidget(dev_btn)

        # ④ 误码率分析
        self._plain_btn(tb, "误码率分析", btn_style,
                        lambda: self._open("ber", BERAnalysisDialog))

        # ⑤ 单链路仿真
        self._plain_btn(tb, "单链路仿真", btn_style,
                        lambda: self._open("single", SingleLinkSimDialog))

        # ⑤ 多链路仿真
        self._plain_btn(tb, "多链路仿真", btn_style,
                        lambda: self._open("multi", MultiLinkSimDialog))

    @staticmethod
    def _plain_btn(tb: QToolBar, label: str, style: str, slot):
        """向工具栏添加纯点击按钮，没有菜单，悬停不触发任何业务逻辑"""
        btn = QToolButton()
        btn.setText(label)
        btn.setStyleSheet(style)
        btn.clicked.connect(slot)
        tb.addWidget(btn)

    # ── 工作区 ────────────────────────────────────────────

    def _build_workspace(self):
        central = QWidget()
        central.setStyleSheet("background:#F5F5F3;")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)

        icon_frame = QFrame()
        icon_frame.setFixedSize(64, 64)
        icon_frame.setStyleSheet("background:#EEEDFE;border-radius:16px;")
        ilay = QVBoxLayout(icon_frame)
        ilay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ilay.addWidget(QLabel("◎",
            styleSheet="font-size: 20pt;color:#7F77DD;background:transparent;"))

        title = QLabel("馈电链路仿真平台")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14pt;font-weight:500;color:#444441;")

        subtitle = QLabel("从上方菜单选择模块开始仿真\n"
                          "链路预算 · 信道建模 · 器件建模 · 单/多链路仿真")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 10pt;color:#888780;")

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
        dlg.show()