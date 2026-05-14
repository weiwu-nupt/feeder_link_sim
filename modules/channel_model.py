"""
无线信道建模 — 全部效应集成在单张表格中
损耗：雨衰 + 大气气体 + 云雾
群时延：电离层群时延（P.531-16 §4.4）
去极化：雨致 XPD（P.618-14 §4.1）
"""

import math

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QStyledItemDelegate,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush

from ui.base_dialog import ModuleDialog
from modules.rain_attenuation import compute_rain_attenuation
from modules.atmospheric_attenuation import compute_atm_attenuation
from modules.cloud_attenuation import compute_cloud_attenuation
from modules.ionosphere_effects import calc_ionospheric_group_delay
from modules.cross_polarization import calc_xpd_rain


# ══════════════════════════════════════════════════════════
#  行定义
# ══════════════════════════════════════════════════════════

ROWS = [
    # ── 链路参数 ──────────────────────────────────────────
    ("链路参数",             "",        "",               "section"),
    ("工作频率",             "f",       "GHz",            "input"),
    ("地面站仰角",           "θ",       "°",              "input"),
    ("极化倾角",             "τ",       "°",              "input"),
    ("地面站海拔",           "h_s",     "km",             "input"),

    # ── 降雨参数 ──────────────────────────────────────────
    ("降雨参数",             "",        "",               "section"),
    ("降雨强度",             "R",       "mm/h",           "input"),
    ("雨顶高度",             "H_R",     "km",             "input"),

    # ── 雨衰路径 ──────────────────────────────────────────
    ("雨衰路径",             "",        "",               "section"),
    ("雨比衰减",         "γ_R",     "dB/km",          "calc"),
    ("有效路径长度",         "L_eff",   "km",             "calc"),

    # ── 雨衰结果 ──────────────────────────────────────────
    ("雨衰结果",             "",        "",               "section"),
    ("链路雨衰",             "A_rain",  "dB",             "calc"),

    # ── 大气参数 ──────────────────────────────────────────
    ("大气参数",             "",        "",               "section"),
    ("大气压",               "p",       "hPa",            "input"),
    ("温度",                 "T",       "K",              "input"),
    ("水汽分压",             "e",       "hPa",            "input"),

    # ── P.676-13 比衰减 ───────────────────────────────────
    ("大气比衰减",      "",        "",               "section"),
    ("O₂ 比衰减",            "γ_O₂",    "dB/km",          "calc"),
    ("H₂O 比衰减",           "γ_H₂O",   "dB/km",          "calc"),
    ("总气体比衰减",         "γ_gas",   "dB/km",          "calc"),

    # ── 大气衰减路径 ──────────────────────────────────────
    ("大气衰减路径",         "",        "",               "section"),
    ("O₂等效高度",           "H_O₂",    "km",             "fixed"),
    ("H₂O等效高度",          "H_H₂O",   "km",             "fixed"),

    # ── 大气衰减结果 ──────────────────────────────────────
    ("大气衰减结果",         "",        "",               "section"),
    ("O₂ 路径衰减",          "A_O₂",    "dB",             "calc"),
    ("H₂O 路径衰减",         "A_H₂O",   "dB",             "calc"),
    ("总大气气体衰减",       "A_gas",   "dB",             "calc"),

    # ── 云雾参数（P.840-9）───────────────────────────────
    ("云雾参数",   "",        "",               "section"),
    ("液态水柱含量",         "L",       "kg/m²",          "input"),

    # ── 云雾衰减结果 ──────────────────────────────────────
    ("云雾衰减结果",         "",        "",               "section"),
    ("云雾比衰减系数",       "K_L",     "(dB/km)/(g/m³)", "calc"),
    ("云雾路径衰减",         "A_cloud", "dB",             "calc"),

    # ── 综合损耗 ──────────────────────────────────────────
    ("综合损耗",             "",        "",               "section"),
    ("雨衰+大气+云雾",       "A_total", "dB",             "calc"),

    # ── 电离层参数（P.531-16）────────────────────────────
    ("电离层参数","",        "",               "section"),
    ("电子总含量 TEC",       "N_T",     "el/m²",          "input"),

    # ── 电离层群时延（§4.4）──────────────────────────────
    ("电离层群时延",         "",        "",               "section"),
    ("群时延",               "t",       "ns",             "calc"),

    # ── 去极化效应（P.618-14 §4.1）───────────────────────
    ("交叉极化效应",           "",        "",               "section"),
    ("雨滴伪角",       "σ",       "°",              "input"),
    ("雨致 XPD",             "XPD_R",   "dB",             "calc"),
]

DEFAULTS = {
    "工作频率":       "39",
    "地面站仰角":     "10",
    "极化倾角":       "45",
    "地面站海拔":     "0.0",
    "降雨强度":       "30",
    "雨顶高度":       "4.0",
    "大气压":         "1013.25",
    "温度":           "288.15",
    "水汽分压":       "10.0",
    "O₂等效高度":     "6.0",
    "H₂O等效高度":    "2.0",
    "液态水柱含量":   "0.5",
    "电子总含量 TEC": "1e17",
    "雨滴伪角": "5",
}

_RI        = {r[0]: i for i, r in enumerate(ROWS)}
_ROW_RAIN  = _RI["链路雨衰"]
_ROW_GAS   = _RI["总大气气体衰减"]
_ROW_CLOUD = _RI["云雾路径衰减"]
_ROW_TOTAL = _RI["雨衰+大气+云雾"]
_ROW_XPD_R = _RI["雨致 XPD"]

_C_SECTION = QColor("#DCE8F5")
_C_INPUT   = QColor("#FFFFFF")
_C_CALC    = QColor("#FFFBF0")
_C_FIXED   = QColor("#F2F2F0")
_C_HDR_BG  = "#2E6B8A"


# ══════════════════════════════════════════════════════════
#  核心计算
# ══════════════════════════════════════════════════════════

def _fv(s, d=0.0):
    try:    return float(str(s).strip())
    except: return d


def calc_all_columns(vals: dict) -> dict:
    out = dict(vals)

    freq   = _fv(vals.get("工作频率",       "39"))
    elev   = _fv(vals.get("地面站仰角",     "10"))
    pol    = _fv(vals.get("极化倾角",       "45"))
    h_s    = _fv(vals.get("地面站海拔",     "0.0"))
    rain   = _fv(vals.get("降雨强度",       "30"))
    h_rain = _fv(vals.get("雨顶高度",       "4.0"))
    pres   = _fv(vals.get("大气压",         "1013.25"))
    temp   = _fv(vals.get("温度",           "288.15"))
    wv     = _fv(vals.get("水汽分压",       "10.0"))
    L      = _fv(vals.get("液态水柱含量",   "0.5"))
    N_T    = _fv(vals.get("电子总含量 TEC", "1e17"))
    sigma  = _fv(vals.get("雨滴伪角", "5"))

    # ── 雨衰（P.838-3 + P.618-13）────────────────────────
    rr = compute_rain_attenuation(
        freq_ghz=freq, rain_rate=rain,
        rain_height_km=h_rain, elevation_deg=elev,
        station_alt_km=h_s, polarization_deg=pol)

    out["雨比衰减"] = f"{float(rr.gamma_R[0]):.4f}"
    out["有效路径长度"] = f"{float(rr.L_eff[0]):.4f}"
    A_rain = float(rr.A_rain[0])
    out["链路雨衰"]     = f"{A_rain:.4f}"

    # ── 大气气体衰减（P.676-13）──────────────────────────
    ar = compute_atm_attenuation(
        freq_ghz=freq, elevation_deg=elev,
        pressure_hpa=pres, temperature_k=temp,
        water_vapor_hpa=wv, station_alt_km=h_s)

    out["O₂ 比衰减"]      = f"{ar.gamma_o2:.4f}"
    out["H₂O 比衰减"]     = f"{ar.gamma_h2o:.4f}"
    out["总气体比衰减"]   = f"{ar.gamma_total:.4f}"
    out["O₂等效高度"]     = "6.0"
    out["H₂O等效高度"]    = "2.0"
    out["O₂ 路径衰减"]    = f"{ar.A_o2:.4f}"
    out["H₂O 路径衰减"]   = f"{ar.A_h2o:.4f}"
    out["总大气气体衰减"] = f"{ar.A_total:.4f}"

    # ── 云雾衰减（P.840-9）───────────────────────────────
    cr = compute_cloud_attenuation(
        freq_ghz=freq, L_kg_m2=L, elevation_deg=elev)
    out["云雾比衰减系数"] = f"{cr.KL:.4f}"
    out["云雾路径衰减"]   = f"{cr.A_cloud:.4f}"

    # ── 综合损耗 ──────────────────────────────────────────
    A_total = A_rain + ar.A_total + cr.A_cloud
    out["雨衰+大气+云雾"] = f"{A_total:.4f}"

    # ── 电离层群时延（P.531-16 §4.4）────────────────────
    _, t_ns = calc_ionospheric_group_delay(freq, N_T)
    out["群时延"] = f"{t_ns:.4f}"

    # ── 雨致 XPD（P.618-14 §4.1）────────────────────────
    xpd_r = calc_xpd_rain(A_rain, freq, elev, pol, sigma)
    out["雨致 XPD"] = f"{xpd_r['XPD_rain']:.4f}"

    return out


# ══════════════════════════════════════════════════════════
#  编辑委托
# ══════════════════════════════════════════════════════════

class _EditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        ed = QLineEdit(parent)
        ed.setStyleSheet(
            "QLineEdit{background:#FFFFFF;color:#111111;"
            "border:2px solid #2E6B8A;border-radius:2px;"
            "padding:1px 4px;font-size:9pt;"
            "selection-background-color:#BEDAF7;}")
        return ed

    def setEditorData(self, ed, index):
        ed.setText(index.data(Qt.ItemDataRole.EditRole) or "")
        ed.selectAll()

    def setModelData(self, ed, model, index):
        model.setData(index, ed.text(), Qt.ItemDataRole.EditRole)


# ══════════════════════════════════════════════════════════
#  主 Widget
# ══════════════════════════════════════════════════════════

class ChannelTableWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_cols   = 3
        self._updating = False
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 工具栏
        tb = QHBoxLayout()
        tb.setSpacing(6)
        for txt, color, slot in [
            ("＋ 添加场景", "#2E6B8A", self._add_col),
            ("删除选中列",  "#888888", self._del_col),
            ("导出 Excel", "#6B4C2E", self._export_excel),
        ]:
            btn = QPushButton(txt)
            btn.setFixedHeight(28)
            btn.setStyleSheet(self._btn_style(color))
            btn.clicked.connect(slot)
            tb.addWidget(btn)
        tb.addStretch()
        root.addLayout(tb)

        # 表格
        self.table = QTableWidget()
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background:#FFFFFF; border:1px solid #C8D8EC;
                gridline-color:#D8E4F0; font-size:9pt; color:#1A1A1A;
            }}
            QTableWidget::item {{ padding:2px 5px; }}
            QTableWidget::item:selected {{ background:#C5DCF5; color:#1A1A1A; }}
            QHeaderView::section {{
                background:{_C_HDR_BG}; color:#FFFFFF;
                font-size:9pt; font-weight:600;
                padding:4px 5px; border:none;
                border-right:1px solid #4A7EA0;
                border-bottom:1px solid #4A7EA0;
            }}
        """)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.setItemDelegate(_EditDelegate(self.table))
        self.table.itemChanged.connect(self._on_changed)
        root.addWidget(self.table, stretch=1)

        self.status = QLabel("就绪")
        self.status.setStyleSheet("font-size:9pt;color:#888;")
        root.addWidget(self.status)

        self._build_table()

    # ── 表格构建 ──────────────────────────────────────────

    def _build_table(self):
        self._updating = True
        n_rows = len(ROWS)
        n_cols = 3 + self._n_cols
        self.table.setRowCount(n_rows)
        self.table.setColumnCount(n_cols)
        self.table.setHorizontalHeaderLabels(
            ["参数", "符号", "单位"] +
            [f"场景 {i+1}" for i in range(self._n_cols)])

        self.table.setColumnWidth(0, 170)
        self.table.setColumnWidth(1, 58)
        self.table.setColumnWidth(2, 80)
        for ci in range(3, n_cols):
            self.table.setColumnWidth(ci, 110)

        hdr = self.table.horizontalHeader()
        for ci in range(n_cols):
            hdr.setSectionResizeMode(ci, QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)

        for ri, (name, sym, unit, rtype) in enumerate(ROWS):
            self.table.setRowHeight(ri, 22)
            self._set_meta(ri, 0, name, rtype)
            self._set_meta(ri, 1, sym,  rtype)
            self._set_meta(ri, 2, unit, rtype)
            for ci in range(self._n_cols):
                self._init_cell(ri, 3 + ci)

        self._updating = False
        self._calc_all()

    def _set_meta(self, ri, ci, text, rtype):
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        if rtype == "section":
            item.setBackground(QBrush(_C_SECTION))
            f = item.font(); f.setBold(True); item.setFont(f)
            item.setForeground(QBrush(QColor("#1A4A80")))
        else:
            item.setBackground(QBrush(QColor("#F5F8FC")))
            item.setForeground(QBrush(QColor("#444444")))
        self.table.setItem(ri, ci, item)

    def _init_cell(self, ri, ci):
        name, sym, unit, rtype = ROWS[ri]
        if rtype == "section":
            item = QTableWidgetItem("")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setBackground(QBrush(_C_SECTION))
            self.table.setItem(ri, ci, item)
            return
        item = QTableWidgetItem(DEFAULTS.get(name, ""))
        if rtype == "input":
            item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                          Qt.ItemFlag.ItemIsSelectable |
                          Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QBrush(_C_INPUT))
        elif rtype == "fixed":
            item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                          Qt.ItemFlag.ItemIsSelectable)
            item.setBackground(QBrush(_C_FIXED))
            item.setForeground(QBrush(QColor("#888888")))
        else:
            item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                          Qt.ItemFlag.ItemIsSelectable)
            item.setBackground(QBrush(_C_CALC))
        self.table.setItem(ri, ci, item)

    # ── 增/删列 ───────────────────────────────────────────

    def _add_col(self):
        self._n_cols += 1
        ci = self.table.columnCount()
        self.table.insertColumn(ci)
        self.table.setHorizontalHeaderItem(
            ci, QTableWidgetItem(f"场景 {self._n_cols}"))
        self.table.setColumnWidth(ci, 110)
        self.table.horizontalHeader().setSectionResizeMode(
            ci, QHeaderView.ResizeMode.Interactive)
        self._updating = True
        for ri in range(len(ROWS)):
            self._init_cell(ri, ci)
        self._updating = False
        self._calc_col(ci)

    def _del_col(self):
        sel = sorted(
            {idx.column() for idx in self.table.selectedIndexes()
             if idx.column() >= 3}, reverse=True)
        if not sel:
            QMessageBox.information(self, "提示", "请先选中要删除的场景列")
            return
        if self._n_cols - len(sel) < 1:
            QMessageBox.information(self, "提示", "至少保留一列")
            return
        for c in sel:
            self.table.removeColumn(c)
            self._n_cols -= 1
        idx = 1
        for ci in range(3, self.table.columnCount()):
            self.table.setHorizontalHeaderItem(
                ci, QTableWidgetItem(f"场景 {idx}"))
            idx += 1

    # ── 计算 ──────────────────────────────────────────────

    def _on_changed(self, item):
        if self._updating: return
        if item.column() >= 3: self._calc_col(item.column())

    def _calc_all(self):
        for ci in range(3, self.table.columnCount()):
            self._calc_col(ci)

    def _calc_col(self, ci):
        vals = {}
        for ri, (name, sym, unit, rtype) in enumerate(ROWS):
            if rtype in ("input", "fixed"):
                it = self.table.item(ri, ci)
                vals[name] = it.text() if it else DEFAULTS.get(name, "")
        try:
            result = calc_all_columns(vals)
        except Exception as e:
            self.status.setText(f"计算错误：{e}")
            return

        self._updating = True
        for ri, (name, sym, unit, rtype) in enumerate(ROWS):
            if rtype != "calc": continue
            it = self.table.item(ri, ci)
            if it is None:
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemFlag.ItemIsEnabled |
                            Qt.ItemFlag.ItemIsSelectable)
                it.setBackground(QBrush(_C_CALC))
                self.table.setItem(ri, ci, it)
            val_str = result.get(name, "")
            it.setText(val_str)

            # 损耗行着色（越大越红）
            if ri == _ROW_TOTAL:
                try:
                    fv = float(val_str)
                    bg = QColor("#E3F2FD") if fv <= 15 else QColor("#FCE4EC")
                    fg = QColor("#0D47A1") if fv <= 15 else QColor("#880E4F")
                    it.setBackground(QBrush(bg))
                    it.setForeground(QBrush(fg))
                    f = it.font(); f.setBold(True); it.setFont(f)
                except ValueError: pass

            elif ri in (_ROW_RAIN, _ROW_GAS, _ROW_CLOUD):
                try:
                    fv = float(val_str)
                    bg = QColor("#E8F5E9") if fv <= 10 else QColor("#FFF3E0")
                    fg = QColor("#1A6B35") if fv <= 10 else QColor("#B05000")
                    it.setBackground(QBrush(bg))
                    it.setForeground(QBrush(fg))
                    f = it.font(); f.setBold(True); it.setFont(f)
                except ValueError: pass

            # XPD 行着色（越小越红）
            elif ri == _ROW_XPD_R:
                try:
                    fv = float(val_str)
                    if fv >= 25:
                        bg, fg = QColor("#E8F5E9"), QColor("#1A6B35")
                    elif fv >= 10:
                        bg, fg = QColor("#FFF3E0"), QColor("#B05000")
                    else:
                        bg, fg = QColor("#FCE4EC"), QColor("#880E4F")
                    it.setBackground(QBrush(bg))
                    it.setForeground(QBrush(fg))
                    f = it.font(); f.setBold(True); it.setFont(f)
                except ValueError: pass

        self._updating = False

        col_n = ci - 2
        self.status.setText(
            f"场景{col_n} | "
            f"f={vals.get('工作频率','?')}GHz  "
            f"θ={vals.get('地面站仰角','?')}°  "
            f"R={vals.get('降雨强度','?')}mm/h  "
            f"→ A_total={result.get('雨衰+大气+云雾','—')}dB  "
            f"t={result.get('群时延','—')}ns  "
            f"XPD_R={result.get('雨致 XPD','—')}dB")

    # ── 导出 Excel ────────────────────────────────────────

    def _export_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 Excel", "信道建模.xlsx", "Excel (*.xlsx)")
        if not path: return
        try:
            self._write_excel(path)
            QMessageBox.information(self, "导出成功", f"已保存：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _write_excel(self, path):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = Workbook(); ws = wb.active; ws.title = "信道建模"
        thin = Side(style="thin", color="C8D8EC")
        brd  = Border(left=thin, right=thin, top=thin, bottom=thin)
        ctr  = Alignment(horizontal="center", vertical="center")
        lft  = Alignment(horizontal="left",   vertical="center")
        def fill(h): return PatternFill("solid", start_color=h, end_color=h)

        n_data = self._n_cols
        hdrs = (["参数", "符号", "单位"] +
                [self.table.horizontalHeaderItem(3+i).text()
                 for i in range(n_data)])
        ws.row_dimensions[1].height = 24
        for ci_x, h in enumerate(hdrs, 1):
            c = ws.cell(1, ci_x, h)
            c.font = Font(bold=True, color="FFFFFF", size=10)
            c.fill = fill("2E6B8A"); c.alignment = ctr; c.border = brd

        ws.column_dimensions["A"].width = 22
        ws.column_dimensions["B"].width = 8
        ws.column_dimensions["C"].width = 14
        for i in range(n_data):
            ws.column_dimensions[get_column_letter(4+i)].width = 14

        for ri, (name, sym, unit, rtype) in enumerate(ROWS):
            er = ri + 2
            ws.row_dimensions[er].height = 18
            is_sec = (rtype == "section")
            for ci_x, txt in enumerate([name, sym, unit], 1):
                c = ws.cell(er, ci_x, txt)
                c.border = brd
                c.alignment = lft if ci_x == 1 else ctr
                c.font = Font(size=10, bold=is_sec,
                              color="1A4A80" if is_sec else "222222")
                c.fill = fill("DCE8F5") if is_sec else fill("F5F8FC")
            for di in range(n_data):
                it  = self.table.item(ri, 3+di)
                raw = it.text() if it else ""
                try:    val = float(raw)
                except: val = raw
                c = ws.cell(er, 4+di, val)
                c.border = brd; c.alignment = ctr
                c.number_format = "0.0000" if isinstance(val, float) else "@"
                if is_sec:
                    c.fill = fill("DCE8F5")
                    c.font = Font(size=10, bold=True, color="1A4A80")
                elif rtype == "fixed":
                    c.fill = fill("F2F2F0")
                    c.font = Font(size=10, color="888888")
                elif rtype == "calc":
                    if ri == _ROW_TOTAL and isinstance(val, float):
                        clr = "0D47A1" if val <= 15 else "880E4F"
                        c.fill = fill("E3F2FD" if val <= 15 else "FCE4EC")
                        c.font = Font(size=10, bold=True, color=clr)
                    elif ri in (_ROW_RAIN, _ROW_GAS, _ROW_CLOUD) and isinstance(val, float):
                        clr = "1A6B35" if val <= 10 else "B05000"
                        c.fill = fill("E8F5E9" if val <= 10 else "FFF3E0")
                        c.font = Font(size=10, bold=True, color=clr)
                    elif ri == _ROW_XPD_R and isinstance(val, float):
                        if val >= 25:
                            c.fill = fill("E8F5E9")
                            c.font = Font(size=10, bold=True, color="1A6B35")
                        elif val >= 10:
                            c.fill = fill("FFF3E0")
                            c.font = Font(size=10, bold=True, color="B05000")
                        else:
                            c.fill = fill("FCE4EC")
                            c.font = Font(size=10, bold=True, color="880E4F")
                    else:
                        c.fill = fill("FFFBF0"); c.font = Font(size=10)
                else:
                    c.fill = fill("FFFFFF"); c.font = Font(size=10)

        ws.freeze_panes = "A2"
        wb.save(path)

    @staticmethod
    def _btn_style(color):
        from PyQt6.QtGui import QColor as QC
        dark = QC(color).darker(130).name()
        return (f"QPushButton{{background:{color};color:#FFF;border:none;"
                f"border-radius:4px;padding:0 12px;font-size:9pt;}}"
                f"QPushButton:hover{{background:{dark};}}")


# ══════════════════════════════════════════════════════════
#  主对话框
# ══════════════════════════════════════════════════════════

class ChannelModelDialog(ModuleDialog):
    TITLE        = "无线信道建模"
    ACCENT_COLOR = "#1D9E75"
    MIN_WIDTH    = 1050
    MIN_HEIGHT   = 700

    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        layout.addWidget(ChannelTableWidget(), stretch=1)