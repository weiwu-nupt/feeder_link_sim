"""
混频器建模模块 — 卫星馈电链路信号处理仿真平台
==============================================
基于物理原理的行为级混频器模型，覆盖四大功能：

  1. 变频与频谱  : 双音 → 三阶非线性 → 乘 LO → FFT
                   显示和频/差频/镜像/LO 泄漏
  2. 非线性指标  : IP3/IM3 双音扫描、1dB 压缩扫描、杂散表
  3. 噪声与链路  : NF、变频损耗、SNR、级联预算
  4. 本振相位噪声: 锚点 L(f) 积分 → RMS 相位误差/抖动/EVM

数学取向：解析公式给指标（快、可解释），时域+FFT 给频谱（真实波形）。

核心非线性模型
  y = g1·x − g3·x³           弱非线性（三阶，含增益压缩）
  g1 = 10^(Gain/20)          线性电压增益
  g3 由 IIP3 反算：IIP3 处基波与 IM3 幅度相等 → A_iip3² = (4/3)·g1/g3
理想正弦混频每边带 −6.02 dB（单边带损耗），解析功率中显式计入。
"""

import numpy as np
from scipy import signal as sp_signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QSplitter, QFileDialog,
    QMessageBox, QSizePolicy, QScrollArea, QFrame, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
)
from PyQt6.QtCore import Qt

from ui.base_dialog import ModuleDialog


# ── 中文字体 ──────────────────────────────────────────────
def _setup_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "STHeiti",
        "Source Han Sans SC", "Source Han Sans CN",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
        "Noto Sans CJK HK", "Noto Sans CJK KR",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for n in candidates:
        if n in available:
            plt.rcParams["font.sans-serif"] = [n] + plt.rcParams.get(
                "font.sans-serif", [])
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return n
    plt.rcParams["axes.unicode_minus"] = False
    return None
_setup_font()


_Z0 = 50.0
_K = 1.380649e-23
_T0 = 290.0
_MIX_LOSS_DB = 6.02   # 理想正弦混频单边带损耗


# ══════════════════════════════════════════════════════════
#  基础换算
# ══════════════════════════════════════════════════════════

def dbm_to_amp(dbm, z0=_Z0):
    return np.sqrt(2.0 * z0 * 10 ** ((dbm - 30) / 10.0))


def amp_to_dbm(amp, z0=_Z0):
    return 10 * np.log10(np.asarray(amp) ** 2 / (2.0 * z0) / 1e-3 + 1e-30)


def g3_from_iip3(g1, iip3_dbm, z0=_Z0):
    """由 IIP3 反算三阶系数 g3（正值）。A_iip3² = (4/3)·g1/g3。"""
    if not np.isfinite(iip3_dbm):
        return 0.0
    a2 = 2.0 * z0 * 10 ** ((iip3_dbm - 30) / 10.0)
    return (4.0 / 3.0) * g1 / a2


# ══════════════════════════════════════════════════════════
#  1. 变频与频谱（时域 + FFT）
# ══════════════════════════════════════════════════════════

def run_spectrum(cfg):
    """
    时域双音仿真：双音 → 三阶非线性 → 乘 LO → FFT。

    cfg: f_in, tone_sep, p_in, f_lo, gain, iip3, lo_leak(dB),
         is_down, fs, n, z0
    返回 freq(Hz), mag(dBm), 及关键频点。
    """
    fs = cfg["fs"]; N = cfg["n"]; z0 = cfg["z0"]
    f0 = cfg["f_in"]; sep = cfg["tone_sep"]; f_lo = cfg["f_lo"]
    g1 = 10 ** (cfg["gain"] / 20.0)
    g3 = g3_from_iip3(g1, cfg["iip3"], z0)

    f1 = f0 - sep / 2.0
    f2 = f0 + sep / 2.0
    a_in = dbm_to_amp(cfg["p_in"], z0)
    t = np.arange(N) / fs
    x = a_in * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))

    # 三阶非线性
    y_nl = g1 * x - g3 * x ** 3
    # 乘本振（理想正弦 LO）
    lo = np.cos(2 * np.pi * f_lo * t)
    y = y_nl * lo
    # LO 泄漏（有限隔离度）
    y = y + 10 ** (-cfg["lo_leak"] / 20.0) * np.cos(2 * np.pi * f_lo * t)

    # 加窗 FFT
    win = np.blackman(N)
    Y = np.fft.rfft(y * win)
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    # 单边功率谱（dBm），窗增益归一化
    cg = win.sum() / N
    v = np.abs(Y) / N / cg * 2.0
    mag = amp_to_dbm(v, z0)

    f_if = abs(f0 - f_lo) if cfg["is_down"] else (f0 + f_lo)
    return dict(
        freq=freq, mag=mag, f1=f1, f2=f2, f_lo=f_lo, f_in=f0,
        f_if=f_if, f_sum=f_lo + f0, f_diff=abs(f_lo - f0),
        f_image=abs(2 * f_lo - f0), sep=sep, is_down=cfg["is_down"],
    )


# ══════════════════════════════════════════════════════════
#  2. 非线性：IP3 / 压缩 / 杂散（解析）
# ══════════════════════════════════════════════════════════

def sweep_ip3(cfg, pin_dbm):
    """双音扫描，解析基波/IM3 输出，外推 OIP3。"""
    z0 = cfg["z0"]
    g1 = 10 ** (cfg["gain"] / 20.0)
    g3 = g3_from_iip3(g1, cfg["iip3"], z0)
    pin = np.asarray(pin_dbm, dtype=float)
    A = dbm_to_amp(pin, z0)
    a_fund = np.abs(g1 * A - 0.75 * g3 * A ** 3)
    a_im3 = 0.75 * g3 * A ** 3
    p_fund = amp_to_dbm(a_fund, z0) - _MIX_LOSS_DB
    p_im3 = amp_to_dbm(a_im3, z0) - _MIX_LOSS_DB
    # 小信号外推 OIP3
    oip3 = p_fund[0] + (p_fund[0] - p_im3[0]) / 2.0
    iip3 = oip3 - (cfg["gain"] - _MIX_LOSS_DB)
    return pin, p_fund, p_im3, oip3, iip3


def sweep_compression(cfg, pin_dbm):
    """单音扫描，求 1dB 压缩点。"""
    z0 = cfg["z0"]
    g1 = 10 ** (cfg["gain"] / 20.0)
    g3 = g3_from_iip3(g1, cfg["iip3"], z0)
    pin = np.asarray(pin_dbm, dtype=float)
    A = dbm_to_amp(pin, z0)
    a_out = np.abs(g1 * A - 0.75 * g3 * A ** 3)
    pout = amp_to_dbm(a_out, z0) - _MIX_LOSS_DB
    lin = pin + cfg["gain"] - _MIX_LOSS_DB
    dev = lin - pout
    cross = np.where(dev >= 1.0)[0]
    if len(cross) and cross[0] > 0:
        i = cross[0]
        d0, d1 = dev[i - 1], dev[i]
        frac = (1.0 - d0) / (d1 - d0) if d1 != d0 else 0.0
        p1_in = pin[i - 1] + frac * (pin[i] - pin[i - 1])
        p1_out = pout[i - 1] + frac * (pout[i] - pout[i - 1])
    else:
        p1_in = p1_out = np.nan
    return pin, pout, lin, p1_in, p1_out


def spur_table(f_lo, f_rf, fs, max_m=3, max_n=3):
    """杂散表：m·LO ± n·RF，落在 [0, fs/2] 内。"""
    rows = []
    for m in range(0, max_m + 1):
        for n in range(0, max_n + 1):
            if m == 0 and n == 0:
                continue
            for s in (1, -1):
                f = abs(m * f_lo + s * n * f_rf)
                if 0 < f <= fs / 2:
                    kind = "期望边带" if (m, n) == (1, 1) else (
                        "LO 泄漏" if n == 0 else (
                        "RF 馈通" if m == 0 else "杂散"))
                    rows.append((m, s * n, f, kind))
    rows.sort(key=lambda r: r[2])
    return rows


# ══════════════════════════════════════════════════════════
#  3. 噪声与链路预算（解析）
# ══════════════════════════════════════════════════════════

def link_budget(cfg):
    """
    变频损耗、NF、SNR、级联预算（解析）。
    SNR = Pin − kTB − NF（输入参考）。
    """
    z0 = cfg["z0"]; bw = cfg["bw"]
    kTB = 10 * np.log10(_K * _T0 * bw / 1e-3)
    gain = cfg["gain"]
    pout = cfg["p_in"] + gain
    oip3 = cfg["iip3"] + gain if np.isfinite(cfg["iip3"]) else float("inf")
    nf = cfg["nf"]
    snr_in = cfg["p_in"] - kTB
    snr_out = snr_in - nf
    f_if = abs(cfg["f_in"] - cfg["f_lo"]) if cfg["is_down"] else (cfg["f_in"] + cfg["f_lo"])
    return dict(
        f_if=f_if, pout=pout, gain=gain, nf=nf, oip3=oip3,
        snr_in=snr_in, snr_out=snr_out, kTB=kTB,
        conv_loss=-gain if gain < 0 else 0.0,
    )


# ══════════════════════════════════════════════════════════
#  4. 本振相位噪声
# ══════════════════════════════════════════════════════════

_DEFAULT_PN = [(1e2, -70), (1e3, -90), (1e4, -110),
               (1e5, -128), (1e6, -145), (1e7, -160)]


def parse_pn_anchors(text):
    """解析 '1k:-90, 100k:-128' 格式锚点。失败返回 None。"""
    if not text or not text.strip():
        return None
    mult = {"k": 1e3, "K": 1e3, "M": 1e6, "m": 1e6, "g": 1e9, "G": 1e9}
    out = []
    for it in text.replace("\n", ",").split(","):
        it = it.strip()
        if not it or ":" not in it:
            continue
        fs_, ls_ = it.split(":", 1)
        fs_ = fs_.strip(); ls_ = ls_.strip()
        try:
            f = float(fs_[:-1]) * mult[fs_[-1]] if fs_[-1] in mult else float(fs_)
            out.append((f, float(ls_)))
        except (ValueError, KeyError, IndexError):
            continue
    out.sort(key=lambda p: p[0])
    return out if len(out) >= 2 else None


def phase_noise(anchors, f_lo, f_start=10.0, f_stop=1e7, n=4000):
    """
    积分 SSB 相位噪声 → RMS 相位误差、抖动、EVM。
    σ_φ = √(2·∫L(f)df)；σ_t = σ_φ/(2π·f_LO)；EVM ≈ σ_φ。
    """
    if anchors is None:
        anchors = _DEFAULT_PN
    offsets = np.logspace(np.log10(f_start), np.log10(f_stop), n)
    fa = [a[0] for a in anchors]; La = [a[1] for a in anchors]
    L = np.interp(np.log10(offsets), np.log10(fa), La)
    trap = getattr(np, "trapezoid", None) or np.trapz
    sig = np.sqrt(2.0 * trap(10 ** (L / 10.0), offsets))
    return dict(
        offsets=offsets, L=L, anchors=anchors,
        rms_phase_deg=np.degrees(sig), rms_phase_rad=sig,
        jitter_s=sig / (2 * np.pi * f_lo), evm_pct=sig * 100.0,
        f_start=f_start, f_stop=f_stop,
    )


# ══════════════════════════════════════════════════════════
#  UI 样式
# ══════════════════════════════════════════════════════════

_ACCENT = "#9C4DA0"
_ES = ("QLineEdit{background:#FFF;border:1px solid #D0D0D0;border-radius:3px;"
       "padding:3px 6px;font-size:10pt;color:#111;}"
       f"QLineEdit:focus{{border:1.5px solid {_ACCENT};}}")
_LS = "font-size:10pt;color:#2C2C2A;"
_GB = ("QGroupBox{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;"
       "margin-top:8px;padding:6px 8px;}"
       "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;"
       f"color:{_ACCENT};font-size:9pt;font-weight:bold;}}")


def _group(title):
    gb = QGroupBox(title); gb.setStyleSheet(_GB)
    vl = QVBoxLayout(gb); vl.setSpacing(4); vl.setContentsMargins(6, 4, 6, 6)
    return gb


def _cb_style():
    return ("QComboBox{background:#FFF;border:1px solid #D0D0D0;border-radius:3px;"
            "padding:3px 8px;font-size:10pt;color:#111;}"
            "QComboBox QAbstractItemView{background:#FFF;color:#111;"
            "selection-background-color:#F0E3F2;font-size:10pt;}")


def _style_ax(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color="#E4E8EE", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#CCCCCC")
    ax.tick_params(labelsize=9)


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6.8, 5.2), dpi=96)
        self.fig.patch.set_facecolor("#F8F8F8")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════
#  杂散表对话框
# ══════════════════════════════════════════════════════════

class SpurDialog(QDialog):
    def __init__(self, rows, f_lo, f_rf, parent=None):
        super().__init__(parent)
        self.setWindowTitle("杂散频率表 (m·LO ± n·RF)")
        self.resize(440, 460)
        lay = QVBoxLayout(self)
        info = QLabel(f"LO = {f_lo/1e9:.3f} GHz    RF = {f_rf/1e9:.3f} GHz")
        info.setStyleSheet("font-size:10pt;color:#444;padding:4px;")
        lay.addWidget(info)
        tbl = QTableWidget(len(rows), 4)
        tbl.setHorizontalHeaderLabels(["m", "n", "频率 (MHz)", "类型"])
        tbl.setStyleSheet(
            "QTableWidget{font-size:9.5pt;color:#1A1A1A;gridline-color:#EADCEC;}"
            "QHeaderView::section{background:#F3EAF4;color:#5A2A5E;"
            "font-weight:bold;border:none;padding:4px;}")
        tbl.verticalHeader().setVisible(False)
        for i, (m, n, f, kind) in enumerate(rows):
            for j, val in enumerate([str(m), f"{n:+d}", f"{f/1e6:.2f}", kind]):
                it = QTableWidgetItem(val)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if j < 3:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                tbl.setItem(i, j, it)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(tbl)
        btn = QPushButton("导出 CSV")
        btn.setStyleSheet("font-size:9.5pt;padding:5px;")
        btn.clicked.connect(lambda: self._export(rows))
        lay.addWidget(btn)

    def _export(self, rows):
        path, _ = QFileDialog.getSaveFileName(self, "导出杂散表", "spurs.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write("m,n,frequency_MHz,type\n")
                for m, n, fr, kind in rows:
                    f.write(f"{m},{n},{fr/1e6:.4f},{kind}\n")
            QMessageBox.information(self, "导出成功", f"已保存：\n{path}")


# ══════════════════════════════════════════════════════════
#  主对话框（多视图）
# ══════════════════════════════════════════════════════════

class MixerModelDialog(ModuleDialog):
    TITLE        = "混频器建模"
    ACCENT_COLOR = "#9C4DA0"
    MIN_WIDTH    = 1080
    MIN_HEIGHT   = 700

    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1)
        sp.setStyleSheet("QSplitter::handle{background:#DDDDDD;}")

        # ═══ 左侧参数（滚动）═══
        left = QWidget(); left.setMinimumWidth(300); left.setMaximumWidth(350)
        lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 8, 0); lv.setSpacing(7)

        # ① 变频配置
        cg = _group("① 变频配置")
        cf = QFormLayout(); cf.setSpacing(5); cf.setContentsMargins(0, 0, 0, 0)
        self.cb_mode = QComboBox(); self.cb_mode.setStyleSheet(_cb_style())
        self.cb_mode.addItems(["下变频 (RF→IF)", "上变频 (IF→RF)"])
        self.cb_mode.currentIndexChanged.connect(self._update_hint)
        self.e_fin = self._edit("2.4")
        self.e_flo = self._edit("2.3")
        self.e_sep = self._edit("10")
        cf.addRow(self._lbl("变频方式:"), self.cb_mode)
        cf.addRow(self._lbl("输入频率 (GHz):"), self.e_fin)
        cf.addRow(self._lbl("本振频率 (GHz):"), self.e_flo)
        cf.addRow(self._lbl("双音间隔 (MHz):"), self.e_sep)
        cg.layout().addLayout(cf)
        self.lbl_hint = QLabel(""); self.lbl_hint.setStyleSheet("font-size:8.5pt;color:#888;")
        self.lbl_hint.setWordWrap(True)
        cg.layout().addWidget(self.lbl_hint)
        lv.addWidget(cg)

        # ② 电平与非线性
        pg = _group("② 电平与非线性")
        pf = QFormLayout(); pf.setSpacing(5); pf.setContentsMargins(0, 0, 0, 0)
        self.e_pin = self._edit("-20")
        self.e_gain = self._edit("-6")
        self.e_iip3 = self._edit("15")
        self.e_loleak = self._edit("30")
        pf.addRow(self._lbl("输入功率 (dBm):"), self.e_pin)
        pf.addRow(self._lbl("变频增益 (dB):"), self.e_gain)
        pf.addRow(self._lbl("IIP3 (dBm):"), self.e_iip3)
        pf.addRow(self._lbl("LO 隔离度 (dB):"), self.e_loleak)
        pg.layout().addLayout(pf)
        lv.addWidget(pg)

        # ③ 噪声与带宽
        ng = _group("③ 噪声与带宽")
        nf_ = QFormLayout(); nf_.setSpacing(5); nf_.setContentsMargins(0, 0, 0, 0)
        self.e_nf = self._edit("8")
        self.e_bw = self._edit("10")
        self.e_z0 = self._edit("50")
        nf_.addRow(self._lbl("噪声系数 NF (dB):"), self.e_nf)
        nf_.addRow(self._lbl("信号带宽 (MHz):"), self.e_bw)
        nf_.addRow(self._lbl("端口阻抗 (Ω):"), self.e_z0)
        ng.layout().addLayout(nf_)
        lv.addWidget(ng)

        # ④ 本振相位噪声
        pnoise = _group("④ 本振相位噪声")
        png = QFormLayout(); png.setSpacing(5); png.setContentsMargins(0, 0, 0, 0)
        self.e_pn = self._edit("", w=170); self.e_pn.setFixedWidth(170)
        self.e_pn.setPlaceholderText("留空=默认综合器")
        png.addRow(self._lbl("L(f) 锚点:"), self.e_pn)
        pnoise.layout().addLayout(png)
        hint = QLabel("格式：频偏:电平，逗号分隔\n例 1k:-90, 100k:-128, 1M:-145")
        hint.setStyleSheet("font-size:8pt;color:#999;"); hint.setWordWrap(True)
        pnoise.layout().addWidget(hint)
        lv.addWidget(pnoise)

        # ⑤ 视图与仿真
        vg = _group("⑤ 视图与仿真")
        vf = QFormLayout(); vf.setSpacing(5); vf.setContentsMargins(0, 0, 0, 0)
        self.cb_view = QComboBox(); self.cb_view.setStyleSheet(_cb_style())
        self.cb_view.addItems(["变频频谱", "IP3 / IM3 扫描",
                               "1 dB 压缩", "本振相位噪声"])
        self.cb_view.currentIndexChanged.connect(lambda: self._run())
        self.e_fs = self._edit("80")
        self.e_n = self._edit("16384")
        vf.addRow(self._lbl("显示:"), self.cb_view)
        vf.addRow(self._lbl("采样率 (GHz):"), self.e_fs)
        vf.addRow(self._lbl("采样点数:"), self.e_n)
        vg.layout().addLayout(vf)
        lv.addWidget(vg)

        self.btn_run = QPushButton("计算 / 绘图")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{_ACCENT};color:#FFF;border:none;"
            "border-radius:5px;font-size:10pt;font-weight:bold;}"
            "QPushButton:hover{background:#7C3A80;}")
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        hb = QHBoxLayout(); hb.setSpacing(6)
        self.btn_spur = QPushButton("杂散表")
        self.btn_save = QPushButton("保存图像")
        for b in (self.btn_spur, self.btn_save):
            b.setFixedHeight(30)
            b.setStyleSheet("QPushButton{background:#FFF;color:#444;border:1px solid #CCC;"
                            "border-radius:5px;font-size:9.5pt;}QPushButton:hover{background:#F5F5F5;}")
            hb.addWidget(b)
        self.btn_spur.clicked.connect(self._show_spurs)
        self.btn_save.clicked.connect(self._save)
        lv.addLayout(hb)
        lv.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        scroll.setWidget(left); scroll.setMinimumWidth(320); scroll.setMaximumWidth(372)
        sp.addWidget(scroll)

        # ═══ 右侧：指标栏 + 图 ═══
        right = QWidget(); rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0); rv.setSpacing(4)

        self.metric_box = QFrame()
        self.metric_box.setStyleSheet("QFrame{background:#FBF5FC;border:1px solid #E5D5E8;border-radius:6px;}")
        mg = QGridLayout(self.metric_box); mg.setContentsMargins(10, 6, 10, 6); mg.setSpacing(4)
        self._metrics = {}
        specs = [("变频损耗", "dB"), ("OIP3", "dBm"), ("P1dB", "dBm"),
                 ("NF", "dB"), ("SNR", "dB"), ("相位抖动", "fs")]
        for i, (name, unit) in enumerate(specs):
            lab = QLabel(name); lab.setStyleSheet("font-size:8.5pt;color:#888;")
            val = QLabel("—"); val.setStyleSheet(f"font-size:12pt;font-weight:bold;color:{_ACCENT};")
            u = QLabel(unit); u.setStyleSheet("font-size:8pt;color:#AAA;")
            mg.addWidget(lab, 0, i); mg.addWidget(val, 1, i); mg.addWidget(u, 2, i)
            self._metrics[name] = val
        rv.addWidget(self.metric_box)

        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas, stretch=1)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("font-size:9pt;color:#888;")
        rv.addWidget(self.status)
        sp.addWidget(right)
        sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        layout.addWidget(sp, stretch=1)

        self._update_hint()
        self._run()

    # ── 辅助 ──
    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet(_LS); return l

    def _edit(self, default, w=110):
        e = QLineEdit(default); e.setFixedWidth(w); e.setStyleSheet(_ES); return e

    @staticmethod
    def _fv(text, default=0.0):
        try:
            return float(str(text).strip())
        except (ValueError, AttributeError):
            return default

    def _update_hint(self):
        f_in = self._fv(self.e_fin.text(), 2.4)
        f_lo = self._fv(self.e_flo.text(), 2.3)
        if self.cb_mode.currentIndex() == 0:
            self.lbl_hint.setText(
                f"中频 IF = |{f_in}−{f_lo}| = {abs(f_in-f_lo):.3g} GHz\n"
                f"镜像 RF = {abs(2*f_lo-f_in):.3g} GHz")
        else:
            self.lbl_hint.setText(
                f"射频 RF = {f_in}+{f_lo} = {f_in+f_lo:.3g} GHz（和频）")

    def _collect_cfg(self):
        self._update_hint()
        is_down = self.cb_mode.currentIndex() == 0
        return dict(
            f_in=self._fv(self.e_fin.text(), 2.4) * 1e9,
            f_lo=self._fv(self.e_flo.text(), 2.3) * 1e9,
            tone_sep=self._fv(self.e_sep.text(), 10.0) * 1e6,
            p_in=self._fv(self.e_pin.text(), -20.0),
            gain=self._fv(self.e_gain.text(), -6.0),
            iip3=self._fv(self.e_iip3.text(), 15.0),
            lo_leak=self._fv(self.e_loleak.text(), 30.0),
            nf=self._fv(self.e_nf.text(), 8.0),
            bw=self._fv(self.e_bw.text(), 10.0) * 1e6,
            z0=self._fv(self.e_z0.text(), 50.0),
            pn_anchors=parse_pn_anchors(self.e_pn.text()),
            is_down=is_down,
            fs=self._fv(self.e_fs.text(), 80.0) * 1e9,
            n=max(2048, int(self._fv(self.e_n.text(), 16384))),
        )

    def _run(self):
        try:
            cfg = self._collect_cfg()
            self._update_metrics(cfg)
            view = self.cb_view.currentText()
            if view.startswith("变频频谱"):
                self._plot_spectrum(cfg)
            elif view.startswith("IP3"):
                self._plot_ip3(cfg)
            elif view.startswith("1 dB"):
                self._plot_comp(cfg)
            else:
                self._plot_pn(cfg)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "计算错误", str(e))

    def _update_metrics(self, cfg):
        lb = link_budget(cfg)
        _, _, _, oip3, _ = sweep_ip3(cfg, np.linspace(cfg["p_in"]-40, cfg["p_in"]-10, 30))
        _, _, _, p1i, p1o = sweep_compression(cfg, np.linspace(-40, 20, 400))
        pn = phase_noise(cfg["pn_anchors"], cfg["f_lo"])
        self._metrics["变频损耗"].setText(f"{lb['conv_loss']:.1f}")
        self._metrics["OIP3"].setText(f"{oip3:.1f}")
        self._metrics["P1dB"].setText(f"{p1o:.1f}" if np.isfinite(p1o) else "—")
        self._metrics["NF"].setText(f"{lb['nf']:.1f}")
        self._metrics["SNR"].setText(f"{lb['snr_out']:.1f}")
        self._metrics["相位抖动"].setText(f"{pn['jitter_s']*1e15:.0f}")

    # ── 视图1：变频频谱 ──
    def _plot_spectrum(self, cfg):
        r = run_spectrum(cfg)
        fig = self.canvas.fig; fig.clf(); fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111); _style_ax(ax)
        freq = r["freq"] / 1e9; mag = r["mag"]
        # 显示窗：目标 IF 附近
        ax.plot(freq, mag, color=_ACCENT, lw=0.9)
        ax.set_ylim(-120, 10)
        # 标注关键频点（错开高度避免重叠）
        def mark(f, txt, col, ytxt):
            fg = f / 1e9
            if freq[0] <= fg <= freq[-1]:
                ax.axvline(fg, color=col, lw=0.8, ls="--", alpha=0.6)
                ax.text(fg, ytxt, txt, fontsize=7.5, color=col,
                        va="top", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#FFF",
                                  ec=col, lw=0.5, alpha=0.85))
        mark(r["f_if"], "IF", "#1F4FD8", 8)
        mark(r["f_lo"], "LO泄漏", "#C0392B", 8)
        mark(r["f_sum"], "和频", "#2E7D32", 8)
        mark(r["f_image"], "镜像", "#E08A00", -18)
        ax.set_xlim(0, max(r["f_sum"], r["f_lo"]) / 1e9 * 1.1)
        ax.set_xlabel("频率 (GHz)", fontsize=10)
        ax.set_ylabel("功率 (dBm)", fontsize=10)
        mode = "下变频" if r["is_down"] else "上变频"
        ax.set_title(f"变频频谱 — {mode}（双音输入，含 IM3/杂散）", fontsize=11)
        self.canvas.draw()
        self.status.setText(
            f"IF={r['f_if']/1e6:.1f}MHz  镜像={r['f_image']/1e9:.3f}GHz  "
            f"和频={r['f_sum']/1e9:.3f}GHz  （竖虚线为关键频点）")

    # ── 视图2：IP3 / IM3 ──
    def _plot_ip3(self, cfg):
        # 扫描到 IIP3 之上，让外推交点（IP3 星）可见
        hi = (cfg["iip3"] if np.isfinite(cfg["iip3"]) else cfg["p_in"]+15) + 3
        pin = np.linspace(cfg["p_in"]-30, hi, 200)
        pin, pf, pi, oip3, iip3 = sweep_ip3(cfg, pin)
        fig = self.canvas.fig; fig.clf(); fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111); _style_ax(ax)
        ax.plot(pin, pf, color=_ACCENT, lw=1.8, label="基波输出")
        ax.plot(pin, pi, color="#E08A00", lw=1.5, ls="--", label="三阶交调 IM3")
        # 外推线
        g = cfg["gain"] - _MIX_LOSS_DB
        ax.plot(pin, pin + g, color="#C0392B", lw=0.8, ls="-.", alpha=0.7,
                label="基波外推（斜率1）")
        ax.plot(pin, 3*pin + (pi[0]-3*pin[0]), color="#888", lw=0.8, ls=":",
                alpha=0.7, label="IM3 外推（斜率3）")
        ax.plot(iip3, oip3, "*", color="#2E7D32", ms=14, zorder=6)
        ax.annotate(f"IP3\nOIP3={oip3:.1f}\nIIP3={iip3:.1f}", (iip3, oip3),
                    textcoords="offset points", xytext=(-75, -5), fontsize=8.5,
                    color="#2E7D32", bbox=dict(boxstyle="round,pad=0.3",
                    fc="#F0FAF0", ec="#2E7D32", lw=0.8))
        ax.set_xlabel("输入功率 (dBm)", fontsize=10)
        ax.set_ylabel("输出功率 (dBm)", fontsize=10)
        ax.set_title("IP3 / IM3 双音扫描", fontsize=11)
        ax.set_ylim(pi[0] - 5, oip3 + 8)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95, edgecolor="#DDD")
        self.canvas.draw()
        self.status.setText(f"OIP3={oip3:.2f}dBm  IIP3={iip3:.2f}dBm  "
                            f"（含 {_MIX_LOSS_DB}dB 变频损耗）")

    # ── 视图3：1dB 压缩 ──
    def _plot_comp(self, cfg):
        # 扫描范围覆盖到压缩区（IIP3 附近）
        hi = max(cfg["iip3"] + 5, cfg["p_in"] + 25)
        pin = np.linspace(cfg["p_in"]-25, hi, 400)
        pin, pout, lin, p1i, p1o = sweep_compression(cfg, pin)
        fig = self.canvas.fig; fig.clf(); fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111); _style_ax(ax)
        ax.plot(pin, pout, color=_ACCENT, lw=2.0, label="实际输出")
        ax.plot(pin, lin, color="#C0392B", lw=0.9, ls="-.", label="线性外推")
        if np.isfinite(p1i):
            ax.plot(p1i, p1o, "o", color="#1F4FD8", ms=8, zorder=6)
            ax.axvline(p1i, color="#AAA", lw=0.7, ls="--")
            ax.axhline(p1o, color="#AAA", lw=0.7, ls="--")
            ax.annotate(f"P1dB\n输入={p1i:.2f}\n输出={p1o:.2f}", (p1i, p1o),
                        textcoords="offset points", xytext=(10, -35), fontsize=8.5,
                        color="#1F4FD8", bbox=dict(boxstyle="round,pad=0.3",
                        fc="#F0F4FC", ec="#1F4FD8", lw=0.8))
        ax.set_xlabel("输入功率 (dBm)", fontsize=10)
        ax.set_ylabel("输出功率 (dBm)", fontsize=10)
        ax.set_title("1 dB 压缩点", fontsize=11)
        # 限制 y 轴到有效区（三次模型仅在压缩点附近有效，避免显示过驱动伪峰）
        if np.isfinite(p1o):
            ax.set_ylim(pout[0] - 2, p1o + 8)
            ax.set_xlim(pin[0], min(pin[-1], p1i + 6))
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95, edgecolor="#DDD")
        self.canvas.draw()
        s = f"输入 P1dB={p1i:.2f}dBm  输出 P1dB={p1o:.2f}dBm" if np.isfinite(p1i) else "未达压缩点"
        self.status.setText(s)

    # ── 视图4：本振相位噪声 ──
    def _plot_pn(self, cfg):
        pn = phase_noise(cfg["pn_anchors"], cfg["f_lo"])
        fig = self.canvas.fig; fig.clf(); fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111); _style_ax(ax)
        ax.semilogx(pn["offsets"], pn["L"], color=_ACCENT, lw=1.8, label="SSB 相位噪声 L(f)")
        fa = [a[0] for a in pn["anchors"]]; La = [a[1] for a in pn["anchors"]]
        ax.scatter(fa, La, s=28, facecolors="#FFF", edgecolors="#C04A3B", lw=1.3,
                   zorder=4, label="锚点")
        ax.axvspan(pn["f_start"], pn["f_stop"], color="#F0E3F2", alpha=0.4)
        ax.grid(True, which="minor", color="#F0F2F5", lw=0.4)
        ax.set_xlim(pn["offsets"][0], pn["offsets"][-1])
        ax.set_xlabel("频偏 (Hz)", fontsize=10)
        ax.set_ylabel("L(f) (dBc/Hz)", fontsize=10)
        ax.set_title(f"本振相位噪声  f_LO={cfg['f_lo']/1e9:.2f}GHz", fontsize=11)
        txt = (f"RMS 相位误差 = {pn['rms_phase_deg']:.3f}°\n"
               f"RMS 抖动 = {pn['jitter_s']*1e15:.1f} fs\n"
               f"相噪 EVM ≈ {pn['evm_pct']:.3f} %")
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=8.5, va="top",
                ha="right", color="#5A2A5E", bbox=dict(boxstyle="round,pad=0.4",
                fc="#FBF5FC", ec="#C9A8CD", lw=0.8))
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95, edgecolor="#DDD")
        self.canvas.draw()
        self.status.setText(
            f"RMS 相位={pn['rms_phase_deg']:.3f}°  抖动={pn['jitter_s']*1e15:.0f}fs  "
            f"EVM≈{pn['evm_pct']:.3f}%  （相噪直接搬移到变频输出）")

    # ── 杂散表 ──
    def _show_spurs(self):
        cfg = self._collect_cfg()
        rows = spur_table(cfg["f_lo"], cfg["f_in"], cfg["fs"])
        SpurDialog(rows, cfg["f_lo"], cfg["f_in"], parent=self).exec()

    # ── 保存 ──
    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "mixer.png",
                                              "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")