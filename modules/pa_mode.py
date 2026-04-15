"""
功放模型模块 (Power Amplifier Model)
无记忆: Saleh, Rapp
有记忆: Memory Polynomial (MP), GMP, Volterra
输出特性: AM/AM, AM/PM, 幅频, 相频, 非线性失真, 压缩点/回退, 温度影响
"""

import math
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QGroupBox, QFrame, QScrollArea, QSizePolicy,
    QCheckBox, QSplitter, QFileDialog, QMessageBox, QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ui.base_dialog import ModuleDialog


# ══════════════════════════════════════════════════════════
#  字体
# ══════════════════════════════════════════════════════════

def _setup_font():
    for name in ["Microsoft YaHei", "SimHei", "PingFang SC",
                 "Noto Sans CJK SC", "Arial Unicode MS"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return

_setup_font()


# ══════════════════════════════════════════════════════════
#  PA 数学模型
# ══════════════════════════════════════════════════════════

def saleh_model(A, alpha_a=2.1587, beta_a=1.1517,
                alpha_p=4.0033, beta_p=9.1040):
    """Saleh 无记忆模型"""
    A = np.asarray(A, dtype=float)
    F   = alpha_a * A / (1.0 + beta_a * A**2)
    Phi = alpha_p * A**2 / (1.0 + beta_p * A**2)   # rad
    g   = 20 * np.log10(np.where(A > 1e-12, F / A, alpha_a))
    return g, np.degrees(Phi), F


def rapp_model(A, v=2.0, A_sat=1.0, G_lin=1.0):
    """Rapp 无记忆模型"""
    A = np.asarray(A, dtype=float)
    F = G_lin * A / (1.0 + (A / A_sat)**(2*v))**(1.0/(2*v))
    g = 20 * np.log10(np.where(A > 1e-12, F / A, G_lin))
    return g, np.zeros_like(A), F


def mp_model_fast(A_arr, coeffs, K):
    """MP 模型快速标量扫幅（避免逐样本循环）"""
    orders = np.arange(1, coeffs.shape[1]*2, 2)   # 1,3,5,...
    F = np.zeros_like(A_arr, dtype=complex)
    for qi, q in enumerate(orders):
        if qi < coeffs.shape[1]:
            # 只取 k=0 主项做 AM/AM（记忆项对稳态幅度影响次要）
            F += coeffs[0, qi] * A_arr * (A_arr ** (q - 1))
    return F


def gmp_model_fast(A_arr, cm, cl, cg):
    """GMP 快速标量幅度扫描（主项 + 超前/滞后简化）"""
    F = mp_model_fast(A_arr, cm, cm.shape[0])
    F += mp_model_fast(A_arr, cl, cl.shape[0]) * 0.1
    F += mp_model_fast(A_arr, cg, cg.shape[0]) * 0.08
    return F


def volterra_fast(A_arr, h1_0, h3_0):
    """Volterra 快速标量：只取 k=0 核"""
    return h1_0 * A_arr + h3_0 * A_arr * A_arr**2


def mp_model_signal(x, coeffs, K):
    """MP 模型信号域（用于频响）"""
    N = len(x)
    orders = list(range(1, coeffs.shape[1]*2, 2))
    y = np.zeros(N, dtype=complex)
    for n in range(K, N):
        val = 0.0+0j
        for ki in range(min(K, coeffs.shape[0])):
            xk = x[n - ki]
            for qi, q in enumerate(orders):
                if qi < coeffs.shape[1]:
                    val += coeffs[ki, qi] * xk * (abs(xk)**(q-1))
        y[n] = val
    return y


def volterra_signal(x, h1, h3):
    """Volterra 信号域"""
    N = len(x)
    K1, K3 = len(h1), len(h3)
    y = np.zeros(N, dtype=complex)
    for n in range(max(K1, K3), N):
        v1 = sum(h1[k]*x[n-k] for k in range(K1))
        v3 = sum(h3[k]*x[n-k]*abs(x[n-k])**2 for k in range(K3))
        y[n] = v1 + v3
    return y


def default_mp_coeffs(K=3, Q=5):
    nq = (Q+1)//2
    c = np.zeros((K, nq), dtype=complex)
    c[0, 0] = 1.0 + 0j
    c[0, 1] = -0.05 + 0.01j if nq > 1 else 0
    if K > 1: c[1, 0] = 0.02 + 0.005j
    return c

def default_volterra_h1(K=5):
    h = np.zeros(K, dtype=complex)
    h[0]=1.0; h[1]=0.1 if K>1 else 0; h[2]=0.05 if K>2 else 0
    return h

def default_volterra_h3(K=3):
    h = np.zeros(K, dtype=complex)
    h[0]=-0.05+0.01j; h[1]=0.01j if K>1 else 0
    return h


# ── 通用特性计算 ──────────────────────────────────────────

def find_p1db(A_arr, gain_db_arr):
    """找 1dB 压缩点输入幅度"""
    g0 = gain_db_arr[0]   # 小信号增益
    target = g0 - 1.0
    idx = np.where(gain_db_arr <= target)[0]
    if len(idx) == 0:
        return A_arr[-1], gain_db_arr[-1]
    i = idx[0]
    if i == 0:
        return A_arr[0], gain_db_arr[0]
    # 线性插值
    x0, x1 = A_arr[i-1], A_arr[i]
    y0, y1 = gain_db_arr[i-1], gain_db_arr[i]
    t = (target - y0) / (y1 - y0 + 1e-30)
    A_p1 = x0 + t * (x1 - x0)
    return float(A_p1), float(target)


def compute_imd3(A_arr, F_arr, A0):
    """
    估算三阶互调失真 IMD3 (dBc)
    用双音展开：IMD3 ≈ 3*(输出三阶项) / 基波，简化为从 AM/AM 斜率估算
    """
    # 拟合三阶多项式 F(A) = a1*A + a3*A^3
    poly = np.polyfit(A_arr[:len(A_arr)//3], F_arr[:len(A_arr)//3], 3)
    a1 = poly[-2] if len(poly) >= 2 else 1.0
    a3 = poly[-4] if len(poly) >= 4 else 0.0
    # 双音 IMD3 = 20*log10(3*|a3|/4 * A0^2 / |a1|)
    if abs(a1) < 1e-12:
        return np.full_like(A_arr, 0.0)
    imd3 = 20 * np.log10(np.abs(3 * a3 / 4 * A_arr**2 / (a1 + 1e-30)) + 1e-30)
    return imd3


def compute_thd(A_arr, F_arr):
    """
    THD 估算：对每个输入幅度，用 AM/AM 非线性拟合计算谐波比
    简化为用增益压缩量估算：THD ≈ (F(A)/A/G0 - 1) 的 dB 表示
    """
    g0 = F_arr[0] / (A_arr[0] + 1e-12)
    ratio = F_arr / (g0 * A_arr + 1e-12)
    ratio = np.clip(ratio, 1e-6, 2.0)
    thd = 20 * np.log10(np.abs(ratio - 1.0) + 1e-6)
    return thd


def temperature_gain_shift(gain_db, temp_c, temp_ref=25.0,
                            coeff_db_per_c=-0.02):
    """
    温度对增益的影响（线性模型）
    典型 GaAs HPA: -0.02 ~ -0.03 dB/°C
    """
    delta = coeff_db_per_c * (temp_c - temp_ref)
    return gain_db + delta


def temperature_phase_shift(phase_deg, temp_c, temp_ref=25.0,
                             coeff_deg_per_c=0.05):
    """
    温度对相位的影响
    典型值: +0.05 deg/°C
    """
    delta = coeff_deg_per_c * (temp_c - temp_ref)
    return phase_deg + delta


# ══════════════════════════════════════════════════════════
#  UI 辅助
# ══════════════════════════════════════════════════════════

_INPUT_STYLE = """
    QLineEdit {
        background:#FFFFFF; border:1px solid #D3D1C7;
        border-radius:5px; padding:4px 8px;
        font-size:13px; color:#2C2C2A;
    }
    QLineEdit:focus { border:1.5px solid #BA7517; }
"""

_COMBO_STYLE = """
    QComboBox {
        background:#FFFFFF; border:1px solid #D3D1C7;
        border-radius:5px; padding:4px 8px;
        font-size:13px; color:#2C2C2A; min-width:130px;
    }
    QComboBox:focus { border:1.5px solid #BA7517; }
    QComboBox::drop-down { border:none; width:20px; }
    QComboBox QAbstractItemView {
        background:#FFFFFF; border:1px solid #D3D1C7;
        color:#2C2C2A;
        selection-background-color:#FAEEDA;
        selection-color:#2C2C2A;
        border-radius:5px; padding:3px; font-size:13px;
    }
"""

_SECTION_STYLE = """
    QGroupBox {
        background:#FFFFFF; border:1px solid #E8E8E5;
        border-radius:8px; margin-top:10px;
        padding:8px 10px 6px 10px;
        font-size:13px; font-weight:500; color:#444441;
    }
    QGroupBox::title {
        subcontrol-origin:margin; subcontrol-position:top left;
        left:10px; padding:0 6px;
        color:#BA7517; font-size:12px; font-weight:500;
    }
"""


def _section(title):
    gb = QGroupBox(title)
    gb.setStyleSheet(_SECTION_STYLE)
    vl = QVBoxLayout(gb)
    vl.setSpacing(4); vl.setContentsMargins(6, 6, 6, 6)
    return gb


def _param_row(parent_layout, label, symbol, unit, default, w=100):
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 1, 0, 1); hl.setSpacing(5)
    lbl = QLabel(label); lbl.setFixedWidth(135)
    lbl.setStyleSheet("font-size:12px; color:#2C2C2A;")
    sym = QLabel(symbol); sym.setFixedWidth(32)
    sym.setStyleSheet("font-size:11px; color:#888780; font-style:italic;")
    unt = QLabel(unit); unt.setFixedWidth(52)
    unt.setStyleSheet("font-size:11px; color:#888780;")
    edit = QLineEdit(str(default))
    edit.setFixedWidth(w); edit.setStyleSheet(_INPUT_STYLE)
    hl.addWidget(lbl); hl.addWidget(sym); hl.addWidget(unt)
    hl.addWidget(edit); hl.addStretch()
    parent_layout.addWidget(row)
    return edit


# ══════════════════════════════════════════════════════════
#  嵌入 Matplotlib 画布
# ══════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 5), dpi=96)
        self.fig.patch.set_facecolor("#FAFAF8")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════
#  功放模型对话框
# ══════════════════════════════════════════════════════════

class PAModelDialog(ModuleDialog):
    TITLE        = "功放模型"
    SUBTITLE     = "Power Amplifier Model"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH    = 1080
    MIN_HEIGHT   = 700

    _MODEL_NAMES = ["Saleh", "Rapp",
                    "Memory Polynomial (MP)", "GMP", "Volterra"]

    def build_content(self, layout: QVBoxLayout):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background:#E8E8E5;}")

        # ══ 左侧配置区 ════════════════════════════════════
        left = QWidget()
        left.setMinimumWidth(330); left.setMaximumWidth(390)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0); lv.setSpacing(8)

        # ── 模型选择 ──────────────────────────────────────
        ms = _section("模型选择")
        ml = ms.layout()
        mrow = QWidget(); mhl = QHBoxLayout(mrow)
        mhl.setContentsMargins(0,2,0,2); mhl.setSpacing(8)
        mlbl = QLabel("模型类型"); mlbl.setFixedWidth(68)
        mlbl.setStyleSheet("font-size:13px; color:#2C2C2A;")
        mhl.addWidget(mlbl)

        self.combo_model = QComboBox()
        self.combo_model.addItems(self._MODEL_NAMES)
        self.combo_model.setStyleSheet(_COMBO_STYLE)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        mhl.addWidget(self.combo_model); mhl.addStretch()
        ml.addWidget(mrow)
        lv.addWidget(ms)

        # ── 参数 Tab ──────────────────────────────────────
        self.param_tabs = QTabWidget()
        self.param_tabs.setStyleSheet("""
            QTabWidget::pane {
                background:#FFFFFF; border:1px solid #E8E8E5; border-radius:8px;
            }
            QTabBar::tab {
                padding:5px 12px; font-size:12px; color:#888780;
                border-bottom:2px solid transparent; background:transparent;
            }
            QTabBar::tab:selected { color:#BA7517; border-bottom:2px solid #BA7517; }
            QTabBar::tab:hover:!selected {
                color:#444441; background:#F1EFE8; border-radius:5px 5px 0 0;
            }
        """)
        self._build_saleh_tab()
        self._build_rapp_tab()
        self._build_mp_tab()
        self._build_gmp_tab()
        self._build_volterra_tab()
        lv.addWidget(self.param_tabs, stretch=1)

        # ── 输出特性勾选 ──────────────────────────────────
        out_s = _section("输出特性（可多选）")
        og = QGridLayout(); og.setSpacing(4)
        out_s.layout().addLayout(og)

        cb_style = "QCheckBox{font-size:12px;color:#2C2C2A;}"
        self.cb_amam   = QCheckBox("AM/AM");          self.cb_amam.setChecked(True)
        self.cb_ampm   = QCheckBox("AM/PM");          self.cb_ampm.setChecked(True)
        self.cb_gain   = QCheckBox("幅频特性")
        self.cb_phase  = QCheckBox("相频特性")
        self.cb_imd    = QCheckBox("非线性失真 IMD3") ; self.cb_imd.setChecked(True)
        self.cb_thd    = QCheckBox("THD")
        self.cb_p1db   = QCheckBox("1dB压缩点 & 回退"); self.cb_p1db.setChecked(True)
        self.cb_temp   = QCheckBox("温度影响")

        for cb in [self.cb_amam, self.cb_ampm, self.cb_gain, self.cb_phase,
                   self.cb_imd, self.cb_thd, self.cb_p1db, self.cb_temp]:
            cb.setStyleSheet(cb_style)

        og.addWidget(self.cb_amam,  0, 0); og.addWidget(self.cb_ampm,  0, 1)
        og.addWidget(self.cb_gain,  1, 0); og.addWidget(self.cb_phase,  1, 1)
        og.addWidget(self.cb_imd,   2, 0); og.addWidget(self.cb_thd,   2, 1)
        og.addWidget(self.cb_p1db,  3, 0); og.addWidget(self.cb_temp,  3, 1)
        lv.addWidget(out_s)

        # ── 输入信号 ──────────────────────────────────────
        sig_s = _section("输入信号 & 环境")
        self.e_amp_max  = _param_row(sig_s.layout(), "幅度最大值",  "A_max", "",    1.0)
        self.e_amp_pts  = _param_row(sig_s.layout(), "扫幅点数",    "N_A",   "",    200, 80)
        self.e_freq     = _param_row(sig_s.layout(), "中心频率",    "f_c",   "MHz", 1000)
        self.e_bw       = _param_row(sig_s.layout(), "带宽",        "BW",    "MHz", 200)
        self.e_temp     = _param_row(sig_s.layout(), "工作温度",    "T",     "°C",  25)
        lv.addWidget(sig_s)

        # ── 按钮 ──────────────────────────────────────────
        bhl = QHBoxLayout(); bhl.setSpacing(8)
        self.btn_run  = QPushButton("计算并绘图")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(self._amber_btn())
        self.btn_run.clicked.connect(self._run)
        bhl.addWidget(self.btn_run)

        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(34)
        self.btn_save.setStyleSheet(self._outline_btn())
        self.btn_save.clicked.connect(self._save_fig)
        bhl.addWidget(self.btn_save)
        lv.addLayout(bhl)

        splitter.addWidget(left)

        # ══ 右侧画布 ══════════════════════════════════════
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0); rv.setSpacing(4)
        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas)
        self.status_lbl = QLabel("就绪 — 配置参数后点击「计算并绘图」")
        self.status_lbl.setStyleSheet("font-size:11px; color:#888780;")
        rv.addWidget(self.status_lbl)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self._on_model_changed(0)

    # ── 参数 Tab ──────────────────────────────────────────

    def _build_saleh_tab(self):
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(3)
        self.s_aa = _param_row(vl, "alpha_a", "α_a", "", 2.1587)
        self.s_ba = _param_row(vl, "beta_a",  "β_a", "", 1.1517)
        self.s_ap = _param_row(vl, "alpha_p", "α_p", "", 4.0033)
        self.s_bp = _param_row(vl, "beta_p",  "β_p", "", 9.1040)
        vl.addStretch()
        self.param_tabs.addTab(w, "Saleh")

    def _build_rapp_tab(self):
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(3)
        self.r_v    = _param_row(vl, "平滑因子",    "v",     "",   2.0)
        self.r_asat = _param_row(vl, "饱和幅度",    "A_sat", "",   1.0)
        self.r_g    = _param_row(vl, "小信号增益",  "G",     "dB", 0.0)
        vl.addStretch()
        self.param_tabs.addTab(w, "Rapp")

    def _build_mp_tab(self):
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(3)
        self.mp_K = _param_row(vl, "记忆深度",   "K", "", 3, 60)
        self.mp_Q = _param_row(vl, "非线性阶次", "Q", "", 5, 60)
        note = QLabel("系数：线性主项 + 三阶压缩 + 一阶记忆（示例）")
        note.setStyleSheet("font-size:11px; color:#888780; padding:4px 0;")
        note.setWordWrap(True); vl.addWidget(note); vl.addStretch()
        self.param_tabs.addTab(w, "MP")

    def _build_gmp_tab(self):
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(3)
        self.gmp_K  = _param_row(vl, "记忆深度 K",    "K",   "", 3, 60)
        self.gmp_La = _param_row(vl, "超前项深度 La", "L_a", "", 2, 60)
        self.gmp_Lb = _param_row(vl, "滞后项深度 Lb", "L_b", "", 2, 60)
        self.gmp_Q  = _param_row(vl, "非线性阶次 Q",  "Q",   "", 5, 60)
        note = QLabel("超前/滞后系数 = 主项系数 × 0.1 / 0.08（示例）")
        note.setStyleSheet("font-size:11px; color:#888780; padding:4px 0;")
        note.setWordWrap(True); vl.addWidget(note); vl.addStretch()
        self.param_tabs.addTab(w, "GMP")

    def _build_volterra_tab(self):
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(3)
        self.vol_K1 = _param_row(vl, "一阶核长度", "K1", "", 5, 60)
        self.vol_K3 = _param_row(vl, "三阶核长度", "K3", "", 3, 60)
        note = QLabel("对角 Volterra（一阶 + 三阶对角核示例）")
        note.setStyleSheet("font-size:11px; color:#888780; padding:4px 0;")
        note.setWordWrap(True); vl.addWidget(note); vl.addStretch()
        self.param_tabs.addTab(w, "Volterra")

    def _on_model_changed(self, idx):
        self.param_tabs.setCurrentIndex(
            self._MODEL_NAMES.index(self.combo_model.currentText())
            if self.combo_model.currentText() in self._MODEL_NAMES else 0
        )

    # ── 数值读取 ──────────────────────────────────────────

    def _f(self, e, d=1.0):
        try: return float(e.text())
        except: return d

    def _i(self, e, d=1):
        try: return max(1, int(float(e.text())))
        except: return d

    # ══════════════════════════════════════════════════════
    #  计算主入口
    # ══════════════════════════════════════════════════════

    def _run(self):
        try:
            self._do_run()
        except Exception as ex:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "计算错误", str(ex))

    def _do_run(self):
        name   = self.combo_model.currentText()
        N_A    = self._i(self.e_amp_pts, 200)
        A_max  = self._f(self.e_amp_max, 1.0)
        f_c    = self._f(self.e_freq, 1000.0)
        bw     = self._f(self.e_bw,   200.0)
        temp   = self._f(self.e_temp, 25.0)

        A_arr = np.linspace(1e-4, A_max, N_A)

        # ── 根据模型计算 AM/AM, AM/PM (稳态扫幅) ──────────
        if name == "Saleh":
            aa = self._f(self.s_aa, 2.1587); ba = self._f(self.s_ba, 1.1517)
            ap = self._f(self.s_ap, 4.0033); bp = self._f(self.s_bp, 9.1040)
            gain_db, phase_deg, F_out = saleh_model(A_arr, aa, ba, ap, bp)
            freq_fn = lambda x: self._freq_resp_memoryless(
                x, lambda a: saleh_model(np.atleast_1d(a), aa, ba, ap, bp))

        elif name == "Rapp":
            v    = self._f(self.r_v, 2.0)
            asat = self._f(self.r_asat, 1.0)
            G_db = self._f(self.r_g, 0.0)
            G_lin = 10 ** (G_db / 20.0)
            gain_db, phase_deg, F_out = rapp_model(A_arr, v, asat, G_lin)
            freq_fn = lambda x: self._freq_resp_memoryless(
                x, lambda a: rapp_model(np.atleast_1d(a), v, asat, G_lin))

        elif name == "Memory Polynomial (MP)":
            K = self._i(self.mp_K, 3); Q = self._i(self.mp_Q, 5)
            coeffs = default_mp_coeffs(K, Q)
            F_complex = mp_model_fast(A_arr.astype(complex), coeffs, K)
            F_out  = np.abs(F_complex)
            gain_db = 20 * np.log10(np.where(A_arr > 1e-12,
                                              F_out / A_arr, 1.0))
            phase_deg = np.angle(F_complex + 1e-30, deg=True)
            freq_fn = lambda x: self._freq_resp_signal(
                x, f_c, bw, lambda s: mp_model_signal(s, coeffs, K))

        elif name == "GMP":
            K  = self._i(self.gmp_K, 3); Q = self._i(self.gmp_Q, 5)
            La = self._i(self.gmp_La, 2); Lb = self._i(self.gmp_Lb, 2)
            cm = default_mp_coeffs(K, Q)
            cl = cm * 0.1; cg = cm * 0.08
            F_complex = gmp_model_fast(A_arr.astype(complex), cm, cl, cg)
            F_out  = np.abs(F_complex)
            gain_db = 20 * np.log10(np.where(A_arr > 1e-12,
                                              F_out / A_arr, 1.0))
            phase_deg = np.angle(F_complex + 1e-30, deg=True)
            freq_fn = lambda x: self._freq_resp_signal(
                x, f_c, bw, lambda s: mp_model_signal(s, cm, K))

        elif name == "Volterra":
            K1 = self._i(self.vol_K1, 5); K3 = self._i(self.vol_K3, 3)
            h1 = default_volterra_h1(K1); h3 = default_volterra_h3(K3)
            h1_0, h3_0 = h1[0], h3[0]
            F_complex = volterra_fast(A_arr.astype(complex), h1_0, h3_0)
            F_out  = np.abs(F_complex)
            gain_db = 20 * np.log10(np.where(A_arr > 1e-12,
                                              F_out / A_arr, 1.0))
            phase_deg = np.angle(F_complex + 1e-30, deg=True)
            freq_fn = lambda x: self._freq_resp_signal(
                x, f_c, bw, lambda s: volterra_signal(s, h1, h3))
        else:
            self.status_lbl.setText("请先选择一个有效模型")
            return

        # ── 温度修正 ──────────────────────────────────────
        gain_db_t   = temperature_gain_shift(gain_db,   temp)
        phase_deg_t = temperature_phase_shift(phase_deg, temp)

        # ── 1dB 压缩点 ────────────────────────────────────
        A_p1, g_p1 = find_p1db(A_arr, gain_db_t)
        # IBO / OBO (相对 A_max)
        A_in_dB  = 20 * np.log10(A_arr / A_max + 1e-30)
        IBO = 20 * np.log10(A_p1 / A_max + 1e-12)   # dB，负值
        F_at_p1 = np.interp(A_p1, A_arr, F_out)
        F_max   = F_out[-1]
        OBO = 20 * np.log10(F_at_p1 / (F_max + 1e-12) + 1e-12)

        # ── 非线性失真 ────────────────────────────────────
        imd3_arr = compute_imd3(A_arr, F_out, A_max * 0.5)
        thd_arr  = compute_thd(A_arr, F_out)

        # ── 频响（仅有记忆模型或带宽特性） ────────────────
        freq_pts = np.linspace(f_c - bw/2, f_c + bw/2, 64)
        gf, pf = freq_fn(A_max * 0.5)

        # ══ 组装绘图任务 ══════════════════════════════════
        plots = []

        if self.cb_amam.isChecked():
            plots.append({
                "title": "AM/AM",
                "curves": [
                    {"x": A_arr, "y": F_out,     "label": f"{name} (T={temp}°C)", "color": "#BA7517"},
                    {"x": A_arr, "y": A_arr,      "label": "理想线性",              "color": "#CCCCCC",
                     "linestyle": "--"},
                ],
                "xlabel": "Input Amplitude", "ylabel": "Output Amplitude",
                "vline": (A_p1, f"P1dB={A_p1:.3f}"),
            })

        if self.cb_ampm.isChecked():
            plots.append({
                "title": "AM/PM",
                "curves": [
                    {"x": A_arr, "y": phase_deg_t, "label": f"{name} (T={temp}°C)", "color": "#E91E63"},
                ],
                "xlabel": "Input Amplitude", "ylabel": "Phase Shift (deg)",
                "vline": (A_p1, f"P1dB"),
            })

        if self.cb_gain.isChecked():
            plots.append({
                "title": "Gain vs Frequency",
                "curves": [{"x": gf[0], "y": gf[1], "label": name, "color": "#2196F3"}],
                "xlabel": "Frequency (MHz)", "ylabel": "Gain (dB)",
            })

        if self.cb_phase.isChecked():
            plots.append({
                "title": "Phase vs Frequency",
                "curves": [{"x": pf[0], "y": pf[1], "label": name, "color": "#9C27B0"}],
                "xlabel": "Frequency (MHz)", "ylabel": "Phase (deg)",
            })

        if self.cb_imd.isChecked():
            A_dBm = 20 * np.log10(A_arr + 1e-30)
            plots.append({
                "title": "IMD3 (dBc)",
                "curves": [{"x": A_dBm, "y": imd3_arr, "label": "IMD3", "color": "#FF5722"}],
                "xlabel": "Input Amplitude (dB)", "ylabel": "IMD3 (dBc)",
            })

        if self.cb_thd.isChecked():
            plots.append({
                "title": "THD (dB)",
                "curves": [{"x": A_arr, "y": thd_arr, "label": "THD", "color": "#FF9800"}],
                "xlabel": "Input Amplitude", "ylabel": "THD (dB)",
            })

        if self.cb_p1db.isChecked():
            G0   = gain_db_t[0]
            ideal_g = np.full_like(A_arr, G0)
            plots.append({
                "title": "增益压缩 & 1dB压缩点 / 回退",
                "curves": [
                    {"x": A_arr, "y": gain_db_t, "label": f"Gain (T={temp}°C)", "color": "#BA7517"},
                    {"x": A_arr, "y": ideal_g,   "label": f"G0={G0:.1f} dB",    "color": "#BBBBBB",
                     "linestyle": "--"},
                    {"x": A_arr, "y": ideal_g - 1.0, "label": "G0-1 dB",        "color": "#FF5722",
                     "linestyle": ":"},
                ],
                "xlabel": "Input Amplitude", "ylabel": "Gain (dB)",
                "vline": (A_p1, f"P1dB  IBO={IBO:.1f}dB  OBO={OBO:.1f}dB"),
                "annot": f"P1dB 输入幅度: {A_p1:.4f}\nIBO: {IBO:.2f} dB\nOBO: {OBO:.2f} dB",
            })

        if self.cb_temp.isChecked():
            temps = np.linspace(-40, 85, 6)
            tc = ["#2196F3","#4CAF50","#FF9800","#BA7517","#E91E63","#9C27B0"]
            t_curves = []
            for ti, Tc in enumerate(temps):
                gT = temperature_gain_shift(gain_db, Tc)
                t_curves.append({"x": A_arr, "y": gT,
                                  "label": f"T={Tc:.0f}°C",
                                  "color": tc[ti % len(tc)]})
            plots.append({
                "title": "温度影响（增益 vs 幅度）",
                "curves": t_curves,
                "xlabel": "Input Amplitude", "ylabel": "Gain (dB)",
            })

        if not plots:
            self.status_lbl.setText("请至少勾选一种输出特性"); return

        self._render(plots, name)
        self.status_lbl.setText(
            f"模型: {name}  T={temp}°C  "
            f"P1dB(in)={A_p1:.4f}  IBO={IBO:.1f}dB  OBO={OBO:.1f}dB"
        )

    # ── 频响辅助 ──────────────────────────────────────────

    def _freq_resp_memoryless(self, A0, model_fn):
        freqs = np.linspace(
            self._f(self.e_freq,1000) - self._f(self.e_bw,200)/2,
            self._f(self.e_freq,1000) + self._f(self.e_bw,200)/2,
            64)
        g0, ph0, _ = model_fn(A0)
        gains  = np.full(64, float(g0[0]))
        phases = np.full(64, float(ph0[0]))
        return (freqs, gains), (freqs, phases)

    def _freq_resp_signal(self, A0, f_c, bw, sig_fn):
        N  = 256
        fs = max(2.5 * bw, 10.0)
        t  = np.arange(N) / fs
        x  = A0 * np.exp(2j * np.pi * f_c * t)
        y  = sig_fn(x)
        if y is None or len(y) < N:
            freqs = np.linspace(f_c - bw/2, f_c + bw/2, 64)
            return (freqs, np.zeros(64)), (freqs, np.zeros(64))
        X = np.fft.fftshift(np.fft.fft(x, N))
        Y = np.fft.fftshift(np.fft.fft(y, N))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) + f_c
        mask  = (freqs >= f_c - bw/2) & (freqs <= f_c + bw/2)
        fp    = freqs[mask]
        eps   = 1e-15
        gd    = 20 * np.log10(np.abs(Y[mask]) / (np.abs(X[mask]) + eps) + eps)
        pd    = np.angle(Y[mask] / (X[mask] + eps), deg=True)
        if len(fp) < 2:
            fp = np.linspace(f_c - bw/2, f_c + bw/2, 64)
            gd = np.zeros(64); pd = np.zeros(64)
        return (fp, gd), (fp, pd)

    # ── 渲染 ──────────────────────────────────────────────

    def _render(self, plots, model_name):
        n    = len(plots)
        cols = min(n, 2)
        rows = math.ceil(n / cols)
        self.canvas.fig.clf()
        self.canvas.fig.patch.set_facecolor("#FAFAF8")
        axes = self.canvas.fig.subplots(rows, cols, squeeze=False)

        for i, p in enumerate(plots):
            ax = axes[i // cols][i % cols]
            ax.set_facecolor("#F8F8F6")
            for c in p["curves"]:
                ax.plot(c["x"], c["y"],
                        label=c.get("label",""),
                        color=c.get("color","#BA7517"),
                        linestyle=c.get("linestyle","-"),
                        linewidth=1.6)
            # 垂直标线
            if "vline" in p:
                xv, vl = p["vline"]
                ax.axvline(xv, color="#FF5722", linewidth=1.0,
                           linestyle="--", alpha=0.8)
                ax.text(xv, ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1]
                        else 0, f" {vl}", fontsize=7, color="#FF5722",
                        va="bottom", rotation=90)
            # 文字标注
            if "annot" in p:
                ax.text(0.03, 0.97, p["annot"], transform=ax.transAxes,
                        fontsize=7.5, va="top", color="#444441",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="#D0D0D0",
                                  alpha=0.85))
            ax.set_title(p["title"], fontsize=10, pad=5)
            ax.set_xlabel(p["xlabel"], fontsize=8)
            ax.set_ylabel(p["ylabel"], fontsize=8)
            ax.grid(True, color="#E0E0E0", linewidth=0.5)
            for sp in ax.spines.values(): sp.set_color("#D0D0D0")
            ax.tick_params(colors="#444441", labelsize=7)
            if len(p["curves"]) > 1:
                ax.legend(fontsize=7, framealpha=0.9, edgecolor="#D0D0D0",
                          loc="best")

        for i in range(n, rows * cols):
            axes[i // cols][i % cols].set_visible(False)

        self.canvas.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    # ── 保存 ──────────────────────────────────────────────

    def _save_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像",
            f"PA_{self.combo_model.currentText().replace(' ','_')}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")

    # ── 样式 ──────────────────────────────────────────────

    @staticmethod
    def _amber_btn():
        return ("QPushButton{background:#BA7517;color:#FFFFFF;border:none;"
                "border-radius:6px;padding:0 18px;font-size:13px;font-weight:500;}"
                "QPushButton:hover{background:#854F0B;}"
                "QPushButton:pressed{background:#633806;}")

    @staticmethod
    def _outline_btn():
        return ("QPushButton{background:#FFFFFF;color:#444441;"
                "border:1px solid #D3D1C7;border-radius:6px;"
                "padding:0 18px;font-size:13px;}"
                "QPushButton:hover{background:#F1EFE8;}")