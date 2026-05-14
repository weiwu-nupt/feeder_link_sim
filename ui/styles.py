"""
全局样式表
Global Stylesheet
"""

APP_STYLE = """
QMainWindow {
    background-color: #F5F5F3;
}

/* ── 菜单栏 ── */
QMenuBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E8E8E5;
    padding: 2px 8px;
    font-size: 10pt;
    color: #444441;
    spacing: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 14px;
    border-radius: 5px;
    color: #5F5E5A;
}
QMenuBar::item:selected {
    background-color: #F1EFE8;
    color: #2C2C2A;
}
QMenuBar::item:pressed {
    background-color: #E8E6DF;
    color: #2C2C2A;
}

/* ── 下拉菜单 ── */
QMenu {
    background-color: #FFFFFF;
    border: 1px solid #D3D1C7;
    border-radius: 8px;
    padding: 6px;
    font-size: 10pt;
    color: #444441;
}
QMenu::item {
    padding: 8px 16px 8px 12px;
    border-radius: 5px;
    color: #5F5E5A;
}
QMenu::item:selected {
    background-color: #F1EFE8;
    color: #2C2C2A;
}
QMenu::separator {
    height: 1px;
    background: #E8E8E5;
    margin: 4px 8px;
}

/* ── 状态栏 ── */
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E8E8E5;
    font-size: 8pt;
    color: #888780;
    padding: 0 8px;
}
QStatusBar::item {
    border: none;
}

/* ── 通用对话框 ── */
QDialog {
    background-color: #FFFFFF;
    border-radius: 10px;
}

/* ── 标签 ── */
QLabel {
    color: #444441;
    font-size: 10pt;
}

/* ── 输入框 ── */
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #D3D1C7;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    color: #2C2C2A;
    selection-background-color: #B5D4F4;
}
QLineEdit:focus {
    border: 1.5px solid #378ADD;
    outline: none;
}
QLineEdit:hover {
    border-color: #B4B2A9;
}

/* ── 按钮 ── */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #D3D1C7;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 10pt;
    color: #444441;
}
QPushButton:hover {
    background-color: #F1EFE8;
    border-color: #B4B2A9;
}
QPushButton:pressed {
    background-color: #E8E6DF;
}
QPushButton.primary {
    background-color: #378ADD;
    border-color: #378ADD;
    color: #FFFFFF;
}
QPushButton.primary:hover {
    background-color: #185FA5;
    border-color: #185FA5;
}

/* ── 分组框 ── */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-size: 10pt;
    font-weight: 500;
    color: #444441;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
    color: #5F5E5A;
    font-size: 9pt;
    font-weight: 500;
}

/* ── 下拉框 ── */
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D3D1C7;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    color: #2C2C2A;
    min-width: 120px;
}
QComboBox:focus {
    border-color: #378ADD;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #D3D1C7;
    border-radius: 6px;
    selection-background-color: #E6F1FB;
    selection-color: #2C2C2A;
    padding: 4px;
}

/* ── 表格 ── */
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E8E8E5;
    border-radius: 6px;
    gridline-color: #F1EFE8;
    font-size: 10pt;
    color: #2C2C2A;
}
QTableWidget::item {
    padding: 6px 10px;
}
QTableWidget::item:selected {
    background-color: #E6F1FB;
    color: #2C2C2A;
}
QHeaderView::section {
    background-color: #F8F8F6;
    border: none;
    border-bottom: 1px solid #E8E8E5;
    border-right: 1px solid #E8E8E5;
    padding: 6px 10px;
    font-size: 9pt;
    font-weight: 500;
    color: #5F5E5A;
}

/* ── 滚动条 ── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #D3D1C7;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #B4B2A9;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #D3D1C7;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #B4B2A9;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Tab 组件 ── */
QTabWidget::pane {
    background-color: #FFFFFF;
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background-color: transparent;
    border: none;
    padding: 8px 16px;
    font-size: 10pt;
    color: #888780;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #2C2C2A;
    border-bottom: 2px solid #378ADD;
    font-weight: 500;
}
QTabBar::tab:hover:!selected {
    color: #444441;
    background-color: #F1EFE8;
    border-radius: 6px 6px 0 0;
}

/* ── Splitter ── */
QSplitter::handle {
    background-color: #E8E8E5;
}
QSplitter::handle:horizontal {
    width: 1px;
}
QSplitter::handle:vertical {
    height: 1px;
}
"""