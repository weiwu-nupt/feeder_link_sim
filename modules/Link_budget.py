"""
链路预算模块
Link Budget Module
"""

import math
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton,
    QGroupBox, QFrame, QScrollArea,
    QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt

from ui.base_dialog import ModuleDialog


# ── 中文字体配置 ──────────────────────────────────────────

def _setup_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
        "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ("CJK", "Chinese", "Hei", "Song", "Noto")):
            plt.rcParams["font.family"] = f.name
            plt.rcParams["axes.unicode_minus"] = False
            return

_setup_chinese_font()


# ── 常量 ──────────────────────────────────────────────────

BOLTZMANN_DBW = -228.6

CODE_RATE = "7/8"

MOD_MODES    = [4,      8,      16,       32,       64,       128,       256,       512    ]
MOD_LABELS   = ["QPSK", "8PSK", "16QAM",  "32QAM",  "64QAM",  "128QAM",  "256QAM",  "512QAM"]
DEMOD_THRESH = [6.58,   11.29,  13.13,    16.02,    19.07,    21.42,     24.27,     26.95  ]

MOD_COLORS   = ["#2196F3", "#4CAF50", "#FF9800", "#9E9D24",
                "#E91E63",  "#9C27B0", "#00BCD4",  "#FF5722"]


# ── 核心计算函数 ──────────────────────────────────────────

def free_space_loss_db(freq_ghz: float, distance_km: float) -> float:
    if distance_km <= 0 or freq_ghz <= 0:
        return 0.0
    return 20 * math.log10(distance_km) + 20 * math.log10(freq_ghz) + 92.45

def slant_range_km(h_km: float, r_km: float, elev_deg: float) -> float:
    if elev_deg <= 0:
        return 0.0
    theta = math.radians(elev_deg)
    return math.sqrt((r_km + h_km)**2 - (r_km * math.cos(theta))**2) - r_km * math.sin(theta)

def earth_central_angle_deg(h_km: float, r_km: float, elev_deg: float) -> float:
    theta = math.radians(elev_deg)
    arg = max(-1.0, min(1.0, r_km * math.cos(theta) / (r_km + h_km)))
    return math.degrees(math.acos(arg)) - elev_deg

def get_bn_db(symbol_rate_mbps: float) -> float:
    return 10 * math.log10(symbol_rate_mbps * 1e6)

def get_cn_db(eirp, l_total, gt, bn_db) -> float:
    return eirp - l_total + gt - BOLTZMANN_DBW - bn_db

def get_margin(cn_db: float, mod_index: int) -> float:
    idx = MOD_MODES.index(mod_index)
    return cn_db - DEMOD_THRESH[idx]

def max_rain_for_zero_margin(eirp, fspl, atm, iono, gt, bn_db, mod_index) -> float:
    idx = MOD_MODES.index(mod_index)
    thresh = DEMOD_THRESH[idx]
    budget = eirp - fspl - atm - iono + gt - BOLTZMANN_DBW - bn_db - thresh
    return max(budget, 0.0)


# ── 输入行组件 ────────────────────────────────────────────

class ParamRow(QWidget):
    def __init__(self, label, symbol="", unit="", default=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        lbl.setStyleSheet("font-size: 13px; color: #2C2C2A;")
        lay.addWidget(lbl)

        sym = QLabel(symbol)
        sym.setFixedWidth(40)
        sym.setStyleSheet("font-size: 12px; color: #888780; font-style: italic;")
        lay.addWidget(sym)

        u = QLabel(unit)
        u.setFixedWidth(72)
        u.setStyleSheet("font-size: 12px; color: #888780;")
        lay.addWidget(u)

        self.edit = QLineEdit()
        self.edit.setFixedWidth(110)
        if default is not None:
            self.edit.setText(str(default))
        self.edit.setStyleSheet("""
            QLineEdit {
                background:#FFFFFF; border:1px solid #D3D1C7;
                border-radius:5px; padding:4px 8px;
                font-size:13px; color:#2C2C2A;
            }
            QLineEdit:focus { border:1.5px solid #378ADD; }
        """)
        lay.addWidget(self.edit)
        lay.addStretch()

    def value(self) -> float:
        try:
            return float(self.edit.text())
        except ValueError:
            return 0.0


# ── 链路预算对话框 ────────────────────────────────────────

class LinkBudgetDialog(ModuleDialog):
    TITLE        = "链路预算"
    SUBTITLE     = "Link Budget"
    ACCENT_COLOR = "#378ADD"
    MIN_WIDTH    = 660
    MIN_HEIGHT   = 680

    def build_content(self, layout: QVBoxLayout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(container)
        vbox.setSpacing(12)
        vbox.setContentsMargins(0, 0, 8, 0)

        # ── 轨道参数 ──────────────────────────────────────
        orb = self._section("轨道参数")
        self.p_r = self._row(orb, "地球半径", "r", "km", 6371)
        self.p_h = self._row(orb, "卫星高度", "h", "km",  508)

        # 仰角行：手动输入，步进固定 1 deg（内部使用，不显示）
        _edit_style = (
            "QLineEdit {"
            "  background:#FFFFFF; border:1px solid #D3D1C7;"
            "  border-radius:5px; padding:4px 8px;"
            "  font-size:13px; color:#2C2C2A;"
            "}"
            "QLineEdit:focus { border:1.5px solid #378ADD; }"
        )

        elev_row = QWidget()
        el = QHBoxLayout(elev_row)
        el.setContentsMargins(0, 1, 0, 1)
        el.setSpacing(6)

        el_lbl = QLabel("地面站仰角")
        el_lbl.setFixedWidth(160)
        el_lbl.setStyleSheet("font-size: 13px; color: #2C2C2A;")
        el.addWidget(el_lbl)

        el_sym = QLabel("θ")
        el_sym.setFixedWidth(40)
        el_sym.setStyleSheet("font-size: 12px; color: #888780; font-style: italic;")
        el.addWidget(el_sym)

        el_unit = QLabel("deg")
        el_unit.setFixedWidth(72)
        el_unit.setStyleSheet("font-size: 12px; color: #888780;")
        el.addWidget(el_unit)

        self.p_elev_start = QLineEdit("10")
        self.p_elev_start.setFixedWidth(72)
        self.p_elev_start.setStyleSheet(_edit_style)
        el.addWidget(self.p_elev_start)

        arr = QLabel("→")
        arr.setStyleSheet("color:#888780; font-size:13px;")
        el.addWidget(arr)

        self.p_elev_end = QLineEdit("90")
        self.p_elev_end.setFixedWidth(72)
        self.p_elev_end.setStyleSheet(_edit_style)
        el.addWidget(self.p_elev_end)

        el.addStretch()
        orb.layout().addWidget(elev_row)
        vbox.addWidget(orb)

        # ── 发射端 ────────────────────────────────────────
        tx = self._section("发射端")
        self.p_eirp = self._row(tx, "等效全向辐射功率", "EIRP", "dBW", 44.8)
        vbox.addWidget(tx)

        # ── 链路损耗 ──────────────────────────────────────
        loss = self._section("链路损耗")
        self.p_freq   = self._row(loss, "工作频率",       "f",     "GHz",    39.0)
        self.p_atm    = self._row(loss, "大气损耗",       "L_atm", "dB",      3.2)
        self.p_iono   = self._row(loss, "电离层损耗",     "L_ion", "dB",      0.5)
        self.p_rain_r = self._row(loss, "雨衰（每公里）", "gR",    "dB/km", 0.01)
        vbox.addWidget(loss)

        # ── 接收端 ────────────────────────────────────────
        rx = self._section("接收端")
        self.p_gt = self._row(rx, "天线品质因数", "G/T", "dB/K", 25.0)
        self.p_rs = self._row(rx, "符号速率",     "Rs",  "Mbps", 750.0)
        vbox.addWidget(rx)

        vbox.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        # ── 底部按钮 ──────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 8, 0, 0)
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.btn_export = QPushButton("导出 Excel")
        self.btn_export.setFixedHeight(34)
        self.btn_export.setStyleSheet(self._btn_style("#1D9E75"))
        self.btn_export.clicked.connect(self.export_excel)
        btn_row.addWidget(self.btn_export)

        self.btn_plot = QPushButton("绘制图像")
        self.btn_plot.setFixedHeight(34)
        self.btn_plot.setStyleSheet(self._btn_style("#378ADD"))
        self.btn_plot.clicked.connect(self.plot_availability)
        btn_row.addWidget(self.btn_plot)

        layout.addLayout(btn_row)

    # ── 数据收集 ──────────────────────────────────────────

    def _collect(self) -> dict:
        def _f(edit):
            try: return float(edit.text())
            except ValueError: return 0.0
        elev_start = _f(self.p_elev_start)
        elev_end   = _f(self.p_elev_end)
        elev_step  = 1.0          # 步进固定 1 deg
        elevs = np.arange(elev_start, elev_end + 1e-9, elev_step)
        return dict(
            r          = self.p_r.value(),
            h          = self.p_h.value(),
            eirp       = self.p_eirp.value(),
            freq       = self.p_freq.value(),
            atm        = self.p_atm.value(),
            iono       = self.p_iono.value(),
            rain_r     = self.p_rain_r.value(),
            gt         = self.p_gt.value(),
            rs         = self.p_rs.value(),
            elev_start = elev_start,
            elev_end   = elev_end,
            elev_step  = elev_step,
            elevs      = elevs,
        )

    # ── 导出 Excel ────────────────────────────────────────

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 Excel", "链路预算.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            self._write_excel(path)
            QMessageBox.information(self, "导出成功", f"已保存到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _write_excel(self, path: str):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        p     = self._collect()
        elevs = p["elevs"]

        wb = Workbook()
        ws = wb.active
        ws.title = "链路预算"

        # ── 样式定义 ──────────────────────────────────────
        hdr_fill  = PatternFill("solid", start_color="3375B7", end_color="3375B7")
        alt_fill  = PatternFill("solid", start_color="F4F8FF", end_color="F4F8FF")
        grn_font  = Font(color="1D6F42", bold=True)
        red_font  = Font(color="C00000", bold=True)
        thin      = Side(style="thin", color="D0D0D0")
        brd       = Border(left=thin, right=thin, top=thin, bottom=thin)
        ctr       = Alignment(horizontal="center", vertical="center")
        ctr_wrap  = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # 每种调制模式对应一种淡色，用于链路余量列表头
        MOD_HDR_COLORS = [
            "BBDEFB",  # QPSK   - 蓝
            "C8E6C9",  # 8PSK   - 绿
            "FFE0B2",  # 16QAM  - 橙
            "F0F4C3",  # 32QAM  - 黄绿
            "FCE4EC",  # 64QAM  - 粉
            "E1BEE7",  # 128QAM - 紫
            "B2EBF2",  # 256QAM - 青
            "FFCCBC",  # 512QAM - 深橙
        ]

        # ── 表头 ──────────────────────────────────────────
        # 固定列：仰角/地心角/斜距/EIRP/频率/FSPL/大气/电离层/雨衰/总损耗/G-T/Rs/BN/C-N
        # 动态列：每种调制模式的 [解调门限, 链路余量]
        fixed_headers = [
            ("仰角\nth (deg)",          8),
            ("地心夹角\ngamma (deg)",   10),
            ("传输距离\nd (km)",        11),
            ("EIRP\n(dBW)",             9),
            ("频率\nf (GHz)",           9),
            ("自由空间损耗\nLfs (dB)",  14),
            ("大气损耗\nLatm (dB)",     11),
            ("电离层损耗\nLion (dB)",   12),
            ("雨衰\nLrain (dB)",        10),
            ("链路总损耗\nL (dB)",      11),
            ("G/T\n(dB/K)",             9),
            ("符号速率\nRs (Mbps)",     11),
            ("带内噪声\nBN (dBHz)",     11),
            ("C/N\n(dB)",               9),
        ]

        n_fixed = len(fixed_headers)
        ws.row_dimensions[1].height = 44

        for ci, (h_text, col_w) in enumerate(fixed_headers, 1):
            c = ws.cell(1, ci)
            c.value     = h_text
            c.font      = Font(bold=True, color="FFFFFF", size=10)
            c.fill      = hdr_fill
            c.alignment = ctr_wrap
            c.border    = brd
            ws.column_dimensions[get_column_letter(ci)].width = col_w

        # 为每种调制模式添加「解调门限」+「链路余量」两列
        for mi, (mod_label, thresh, hdr_color) in enumerate(
                zip(MOD_LABELS, DEMOD_THRESH, MOD_HDR_COLORS)):
            ci_thresh  = n_fixed + mi * 2 + 1
            ci_margin  = n_fixed + mi * 2 + 2
            mod_fill   = PatternFill("solid", start_color=hdr_color, end_color=hdr_color)

            c_thresh = ws.cell(1, ci_thresh)
            c_thresh.value     = f"{mod_label}\n门限 (dB)"
            c_thresh.font      = Font(bold=True, size=10)
            c_thresh.fill      = mod_fill
            c_thresh.alignment = ctr_wrap
            c_thresh.border    = brd
            ws.column_dimensions[get_column_letter(ci_thresh)].width = 11

            c_margin = ws.cell(1, ci_margin)
            c_margin.value     = f"{mod_label}\n余量 (dB)"
            c_margin.font      = Font(bold=True, size=10)
            c_margin.fill      = mod_fill
            c_margin.alignment = ctr_wrap
            c_margin.border    = brd
            ws.column_dimensions[get_column_letter(ci_margin)].width = 11

        # ── 数据行 ────────────────────────────────────────
        bn_db = get_bn_db(p["rs"])

        for ri, elev in enumerate(elevs, 2):
            d       = slant_range_km(p["h"], p["r"], elev)
            gamma   = earth_central_angle_deg(p["h"], p["r"], elev)
            fspl    = free_space_loss_db(p["freq"], d)
            rain    = p["rain_r"] * d          # 雨衰 = 每公里衰减 × 斜距
            l_total = fspl + p["atm"] + p["iono"] + rain
            cn      = get_cn_db(p["eirp"], l_total, p["gt"], bn_db)

            use_alt = (ri % 2 == 0)

            # 固定列数据
            fixed_vals = [
                round(elev,         2),
                round(gamma,        2),
                round(d,            2),
                round(p["eirp"],    2),
                round(p["freq"],    2),
                round(fspl,         2),
                round(p["atm"],     2),
                round(p["iono"],    2),
                round(rain,         2),
                round(l_total,      2),
                round(p["gt"],      2),
                round(p["rs"],      2),
                round(bn_db,        2),
                round(cn,           2),
            ]

            for ci, v in enumerate(fixed_vals, 1):
                c = ws.cell(ri, ci)
                c.value         = v
                c.alignment     = ctr
                c.border        = brd
                c.number_format = "0.00"
                if use_alt:
                    c.fill = alt_fill

            # 每种调制模式的门限列 + 余量列
            for mi, (thresh, hdr_color) in enumerate(zip(DEMOD_THRESH, MOD_HDR_COLORS)):
                ci_thresh = n_fixed + mi * 2 + 1
                ci_margin = n_fixed + mi * 2 + 2
                margin    = round(cn - thresh, 2)
                mod_fill  = PatternFill("solid",
                                        start_color=hdr_color,
                                        end_color=hdr_color)

                # 解调门限列
                ct = ws.cell(ri, ci_thresh)
                ct.value         = round(thresh, 2)
                ct.alignment     = ctr
                ct.border        = brd
                ct.number_format = "0.00"
                if use_alt:
                    ct.fill = alt_fill

                # 链路余量列
                cm = ws.cell(ri, ci_margin)
                cm.value         = margin
                cm.alignment     = ctr
                cm.border        = brd
                cm.number_format = "0.00"
                cm.font          = grn_font if margin >= 0 else red_font
                if use_alt and margin >= 0:
                    cm.fill = alt_fill   # 正值保持交替底色
                # 负值不覆盖 fill，让红色字更醒目

        ws.freeze_panes = "A2"

        # ── 参数配置 Sheet ────────────────────────────────
        ws2 = wb.create_sheet("参数配置")
        hdr2_fill = PatternFill("solid", start_color="E8F0FA", end_color="E8F0FA")
        summary = [
            ("地球半径 r",   p["r"],          "km"),
            ("卫星高度 h",   p["h"],          "km"),
            ("仰角起始",     p["elev_start"], "deg"),
            ("仰角终止",     p["elev_end"],   "deg"),
            ("仰角步进",     p["elev_step"],  "deg"),
            ("EIRP",         p["eirp"],       "dBW"),
            ("工作频率 f",   p["freq"],       "GHz"),
            ("大气损耗",     p["atm"],        "dB"),
            ("电离层损耗",   p["iono"],       "dB"),
            ("雨衰每公里",   p["rain_r"],     "dB/km"),
            ("天线 G/T",     p["gt"],         "dB/K"),
            ("符号速率 Rs",  p["rs"],         "Mbps"),
            ("编码率",       CODE_RATE,       ""),
            ("玻尔兹曼常数", BOLTZMANN_DBW,   "dBW/Hz/K"),
        ]
        for ci_h, hdr_txt in enumerate(["参数名称", "数值", "单位"], 1):
            c2 = ws2.cell(1, ci_h)
            c2.value     = hdr_txt
            c2.font      = Font(bold=True, size=10)
            c2.fill      = hdr2_fill
            c2.alignment = ctr
            c2.border    = brd
        ws2.column_dimensions["A"].width = 22
        ws2.column_dimensions["B"].width = 16
        ws2.column_dimensions["C"].width = 14
        for ri2, (name, v, unit) in enumerate(summary, 2):
            ws2.cell(ri2, 1).value = name
            ws2.cell(ri2, 2).value = v
            ws2.cell(ri2, 3).value = unit
            for ci2 in range(1, 4):
                ws2.cell(ri2, ci2).border    = brd
                ws2.cell(ri2, ci2).alignment = ctr

        wb.save(path)

    # ── 绘图 ─────────────────────────────────────────────

    def plot_availability(self):
        p = self._collect()
        elevs = p["elevs"]
        if len(elevs) < 2:
            QMessageBox.warning(self, "参数错误", "仰角范围太小，请检查设置。")
            return

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#F8F8F6")

        bn_db = get_bn_db(p["rs"])

        for mod_idx, mod_label, color in zip(MOD_MODES, MOD_LABELS, MOD_COLORS):
            boundary = []
            for elev in elevs:
                d    = slant_range_km(p["h"], p["r"], elev)
                fspl = free_space_loss_db(p["freq"], d)
                boundary.append(max_rain_for_zero_margin(
                    p["eirp"], fspl, p["atm"], p["iono"],
                    p["gt"], bn_db, mod_idx
                ))
            ax.plot(boundary, elevs, color=color, linewidth=2, label=mod_label)
            ax.fill_betweenx(elevs, 0, boundary, color=color, alpha=0.07)

        ax.set_xlabel("雨衰 (dB)", fontsize=12)
        ax.set_ylabel("地面站仰角 (deg)", fontsize=12)
        ax.set_title("调制方式可用性（曲线左侧为可用）",
                     fontsize=13, pad=12)
        ax.set_xlim(left=0)
        ax.set_ylim(elevs[0], elevs[-1])
        ax.grid(True, color="#E0E0E0", linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_color("#D0D0D0")
        ax.tick_params(colors="#444441")
        ax.legend(loc="lower right", fontsize=10,
                  framealpha=0.95, edgecolor="#D0D0D0")

        info = (f"EIRP={p['eirp']} dBW  |  f={p['freq']} GHz  |  "
                f"G/T={p['gt']} dB/K  |  Rs={p['rs']} Mbps")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                color="#5F5E5A",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor="#D0D0D0", alpha=0.9))

        plt.tight_layout()

        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "调制方式可用性.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            plt.savefig(path, dpi=150, bbox_inches="tight")
            QMessageBox.information(self, "保存成功", f"图像已保存：\n{path}")
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            self._show_preview(buf.read())

        plt.close(fig)

    def _show_preview(self, png_bytes: bytes):
        from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QScrollArea
        from PyQt6.QtGui import QPixmap
        dlg = QDialog(self)
        dlg.setWindowTitle("链路余量图预览")
        dlg.resize(920, 600)
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        lbl = QLabel()
        px = QPixmap()
        px.loadFromData(png_bytes)
        lbl.setPixmap(px)
        scroll.setWidget(lbl)
        lay.addWidget(scroll)
        dlg.exec()

    # ── 构建辅助 ─────────────────────────────────────────

    @staticmethod
    def _section(title: str) -> QGroupBox:
        gb = QGroupBox(title)
        gb.setStyleSheet("""
            QGroupBox {
                background:#FFFFFF; border:1px solid #E8E8E5;
                border-radius:8px; margin-top:10px;
                padding:10px 12px 8px 12px;
                font-size:13px; font-weight:500; color:#444441;
            }
            QGroupBox::title {
                subcontrol-origin:margin; subcontrol-position:top left;
                left:10px; padding:0 6px;
                color:#378ADD; font-size:12px; font-weight:500;
            }
        """)
        vl = QVBoxLayout(gb)
        vl.setSpacing(3)
        vl.setContentsMargins(8, 8, 8, 8)
        return gb

    @staticmethod
    def _row(group: QGroupBox, label, sym, unit, default) -> ParamRow:
        row = ParamRow(label, sym, unit, default)
        group.layout().addWidget(row)
        return row

    @staticmethod
    def _btn_style(color: str) -> str:
        from PyQt6.QtGui import QColor
        dark = QColor(color).darker(130).name()
        return (
            f"QPushButton {{"
            f"  background:{color}; color:#FFFFFF;"
            f"  border:none; border-radius:6px;"
            f"  padding:0 20px; font-size:13px; font-weight:500;"
            f"}}"
            f"QPushButton:hover   {{ background:{dark}; }}"
            f"QPushButton:pressed {{ background:{dark}; }}"
        )