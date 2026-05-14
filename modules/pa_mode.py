"""
功放模型 — Power Amplifier Model
无记忆: Saleh, Modified Rapp
有记忆: Memory Polynomial (MP), Cross-Term Memory (CTM)
参考:
  - MathWorks RF Blockset Power Amplifier block
  - Morgan et al., IEEE Trans. SP, Vol.54, Oct.2006 (eq.19, eq.23)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QGroupBox, QSplitter, QFileDialog,
    QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt

from ui.base_dialog import ModuleDialog


# ── 字体 ──────────────────────────────────────────────────
def _setup_font():
    for n in ["Microsoft YaHei","SimHei","PingFang SC","Noto Sans CJK SC"]:
        if n in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = n
            plt.rcParams["axes.unicode_minus"] = False
            return
_setup_font()


# ══════════════════════════════════════════════════════════
#  单位约定（与 MathWorks RF Blockset 一致）
#  r = sqrt(P_W) = sqrt(1e-3 * 10^(Pin_dBm/10))
#  Pout_dBm = 10·log10(r_out² / 1e-3)
# ══════════════════════════════════════════════════════════

def _dbm2r(p: float) -> float:
    return math.sqrt(1e-3 * 10 ** (p / 10.0))

def _r2dbm(r: float) -> float:
    return 10 * math.log10(r ** 2 / 1e-3 + 1e-30)

def _dbm2r_arr(p: np.ndarray) -> np.ndarray:
    return np.sqrt(1e-3 * 10 ** (p / 10.0))

def _r2dbm_arr(r: np.ndarray) -> np.ndarray:
    return 10 * np.log10(r ** 2 / 1e-3 + 1e-30)


# ══════════════════════════════════════════════════════════
#  无记忆模型
# ══════════════════════════════════════════════════════════

def saleh_amam(r, alpha_a, beta_a):
    """F(r) = alpha_a·r / (1 + beta_a·r²)"""
    return alpha_a * r / (1.0 + beta_a * r ** 2)

def saleh_ampm(r, alpha_p, beta_p):
    """Φ(r) = alpha_p·r² / (1 + beta_p·r²)  → degrees"""
    return np.degrees(alpha_p * r ** 2 / (1.0 + beta_p * r ** 2))

def rapp_amam(r, g_lin, v_sat, p):
    """F(r) = g·r / (1 + (g·r/Vsat)^{2p})^{1/(2p)}"""
    x = (g_lin * r / v_sat) ** (2.0 * p)
    return g_lin * r / (1.0 + x) ** (1.0 / (2.0 * p))

def rapp_ampm(r, A, B, q1, q2):
    """
    Φ(r) = A·r^q1 / (1 + (r/B)^q2)  → degrees
    极限(q1=q2): A·B^q2
    """
    return np.degrees(A * r ** q1 / (1.0 + (r / B) ** q2))


# ══════════════════════════════════════════════════════════
#  有记忆模型（Morgan et al., IEEE Trans. SP, Vol.54, 2006）
#
#  系数矩阵约定（与 MathWorks Power Amplifier block 一致）:
#
#  MP  (eq.19): C shape = (M, D)
#    y(n) = Σ_{m=0}^{M-1} Σ_{d=0}^{D-1} C[m,d] · x(n-m) · |x(n-m)|^d
#    M=记忆深度(Memory Depth), D=电压阶次(Voltage Order)
#    d=0: 线性; d=1: 三阶; d=2: 五阶 ...
#
#  CTM (eq.23 简化): C shape = (M, M*(D-1)+1)
#    y(n) = C .* M_CTM 所有元素之和
#    M_CTM[m, 0]      = x(n-m)                    (d=0, 线性列)
#    M_CTM[m, d*M+l'] = x(n-m) · |x(n-(M-1-l'))|^d  (d>=1, l'=0..M-1)
#    参见 fit_memory_poly_model 的 ctMemPoly case
# ══════════════════════════════════════════════════════════

def _apply_mp(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Memory Polynomial (eq.19):
    y(n) = Σ_{m=0}^{M-1} Σ_{d=0}^{D-1} C[m,d] · x(n-m) · |x(n-m)|^d
    C: (M, D) complex
    """
    M, D = C.shape
    N = len(x)
    y = np.zeros(N, dtype=complex)
    for n in range(M - 1, N):
        for m in range(M):
            xm = x[n - m]
            em = abs(xm)
            for d in range(D):
                y[n] += C[m, d] * xm * em ** d
    return y


def _apply_ctm(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Cross-Term Memory，严格按 fit_memory_poly_model ctMemPoly 列顺序:
    C shape: (M, M*(D-1)+1)

    列排列（与 MATLAB 代码一致）:
      col=0:    x(t-m)                               [线性]
      col=j+1 (j=0..M*(D-1)-1):
        l = M-1-(j%M),   d = j//M+1
        基函数: x(t-m) * |x(t-l)|^d                [交叉包络项]

    等价展开 (M=3, D=3, ncols=7):
      col=1: d=1,l=2  col=2: d=1,l=1  col=3: d=1,l=0
      col=4: d=2,l=2  col=5: d=2,l=1  col=6: d=2,l=0
    """
    M = C.shape[0]
    ncols = C.shape[1]           # M*(D-1)+1
    D = (ncols - 1) // M + 1    # Voltage Order
    N = len(x)
    y = np.zeros(N, dtype=complex)
    for n in range(M - 1, N):
        for m in range(M):
            xm = x[n - m]
            # col=0: 线性
            y[n] += C[m, 0] * xm
            # col=j+1: 交叉包络项
            for j in range(M * (D - 1)):
                l = M - 1 - (j % M)    # 包络延迟
                d = j // M + 1          # 包络幂次
                if n - l >= 0:
                    y[n] += C[m, j + 1] * xm * abs(x[n - l]) ** d
    return y


def _gen_ofdm(n_sc=64, n_sym=60, cp=0.25, seed=42):
    """生成归一化 QPSK-OFDM 复基带信号（峰值=1）"""
    rng = np.random.default_rng(seed)
    qpsk = np.exp(1j * (np.pi/4 + rng.integers(0, 4, (n_sym, n_sc)) * np.pi/2))
    cp_len = int(n_sc * cp)
    frames = []
    for sym in qpsk:
        td = np.fft.ifft(sym) * np.sqrt(n_sc)
        frames.append(np.concatenate([td[-cp_len:], td]))
    x = np.concatenate(frames)
    pk = np.max(np.abs(x))
    return x / pk if pk > 0 else x


def _extract_amam_ampm(x_in, y_out, n_bins=40, skip=0):
    """
    从输入/输出复基带信号按幅度分bin，
    提取平均 AM/AM（输入幅度 vs 输出幅度）和 AM/PM（输入幅度 vs 相位偏移）
    返回 (amp_in_bins, amp_out_mean, phase_mean_deg)
    """
    xi = x_in[skip:];  yi = y_out[skip:]
    mask = np.abs(xi) > 1e-8
    xi, yi = xi[mask], yi[mask]
    amp_in  = np.abs(xi)
    amp_out = np.abs(yi)
    phi_deg = np.angle(yi / (xi + 1e-15), deg=True)
    edges = np.linspace(amp_in.min(), amp_in.max(), n_bins + 1)
    centers, out_m, phi_m = [], [], []
    for i in range(n_bins):
        m = (amp_in >= edges[i]) & (amp_in < edges[i+1])
        if m.sum() >= 3:
            centers.append((edges[i] + edges[i+1]) / 2)
            out_m.append(np.mean(amp_out[m]))
            phi_m.append(np.mean(phi_deg[m]))
    return np.array(centers), np.array(out_m), np.array(phi_m)


def _mem_sweep(pin_arr, C, mode, in_sc_db=0, out_sc_db=0, n_sym=60):
    """
    用 OFDM 信号驱动有记忆模型，返回 (pout_dbm, phase_deg) 插值到 pin_arr。
    pin_arr 中值作为 OFDM 信号的 RMS 功率参考。
    """
    p_ref_dbm = float(np.median(pin_arr))
    p_ref_w   = 1e-3 * 10 ** (p_ref_dbm / 10.0)

    # 生成并缩放 OFDM
    x_norm = _gen_ofdm(n_sc=64, n_sym=n_sym)
    rms_x  = np.sqrt(np.mean(np.abs(x_norm) ** 2))
    x_sig  = x_norm * (np.sqrt(p_ref_w) / (rms_x + 1e-15))

    if in_sc_db != 0:
        x_sig = x_sig * 10 ** (in_sc_db / 20.0)

    # 运行模型
    if mode == 'MP':
        y_sig = _apply_mp(x_sig, C)
    elif mode == 'CTM':
        y_sig = _apply_ctm(x_sig, C)
    else:
        y_sig = _apply_mp(x_sig, C)

    if out_sc_db != 0:
        y_sig = y_sig * 10 ** (out_sc_db / 20.0)

    M = C.shape[0]
    skip = M + 2
    amp_in, amp_out, phi_deg = _extract_amam_ampm(x_sig, y_sig, n_bins=40, skip=skip)

    # 幅度 → dBm
    pin_bin  = 10 * np.log10(np.clip(amp_in  ** 2 / 1e-3, 1e-30, None))
    pout_bin = 10 * np.log10(np.clip(amp_out ** 2 / 1e-3, 1e-30, None))

    pout_interp = np.interp(pin_arr, pin_bin, pout_bin,
                            left=pout_bin[0],  right=pout_bin[-1])
    phi_interp  = np.interp(pin_arr, pin_bin, phi_deg,
                            left=phi_deg[0],   right=phi_deg[-1])
    return pout_interp, phi_interp


def _default_mp_coeffs(M=3, D=3):
    """
    MP 默认系数矩阵 (M, D)，C[m, d]
    d=0: 线性; d=1: 三阶非线性; d=2: 五阶非线性
    m=0: 主路; m=1,2: 记忆抽头
    """
    C = np.zeros((M, D), dtype=complex)
    if D >= 1: C[0, 0] =  1.0000 + 0.0000j
    if D >= 2: C[0, 1] = -0.0500 + 0.0100j
    if D >= 3: C[0, 2] = -0.0200 + 0.0050j
    if M >= 2:
        if D >= 1: C[1, 0] =  0.0200 + 0.0050j
        if D >= 2: C[1, 1] = -0.0100 + 0.0020j
    if M >= 3:
        if D >= 1: C[2, 0] =  0.0050 + 0.0010j
    return C


def _default_ctm_coeffs(M=3, D=3):
    """
    CTM 默认系数矩阵 (M, M*(D-1)+1)，按 fit_memory_poly_model 列顺序。

    col=0:    线性项 C[m,0]
    col=j+1:  l=M-1-(j%M), d=j//M+1

    默认值参考典型 TWT 特性，主对角（l=0，即 j%M=M-1）用 MP 同等系数，
    交叉项（l≠0）用主项的 0.1 倍。
    """
    ncols = M * (D - 1) + 1
    C = np.zeros((M, ncols), dtype=complex)

    # col=0: 线性项（与 MP 一致）
    C[0, 0] =  1.0000 + 0.0000j
    if M >= 2: C[1, 0] =  0.0200 + 0.0050j
    if M >= 3: C[2, 0] =  0.0050 + 0.0010j

    # 交叉项列：对每个 j，l=M-1-(j%M), d=j//M+1
    # l=0 对应 j%M=M-1，即每段最后一列 → 等同于 MP 非线性项（自身延迟）
    # l≠0 为真正的交叉项，用主项的 0.1 倍
    mp_nonlin = {
        (0, 1): -0.0500 + 0.0100j,   # m=0, d=1 (三阶)
        (0, 2): -0.0200 + 0.0050j,   # m=0, d=2 (五阶)
        (1, 1): -0.0100 + 0.0020j,   # m=1, d=1
    }
    for j in range(M * (D - 1)):
        l = M - 1 - (j % M)
        d = j // M + 1
        col = j + 1
        for m in range(M):
            base = mp_nonlin.get((m, d), 0j)
            if l == 0:
                C[m, col] = base          # 自身延迟：用 MP 系数
            elif base != 0j:
                C[m, col] = base * 0.1    # 交叉延迟：用 10% 衰减
    return C


# ══════════════════════════════════════════════════════════
#  解析工具
# ══════════════════════════════════════════════════════════

def _f(text, default=0.0):
    try:    return float(text.strip())
    except: return default

def _vec2(text, default=(1.0, 1.0)):
    import re
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else default


# ══════════════════════════════════════════════════════════
#  画布
# ══════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6.5, 6), dpi=96)
        self.fig.patch.set_facecolor("#F8F8F8")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════
#  UI 辅助
# ══════════════════════════════════════════════════════════

_ES = ("QLineEdit{background:#FFF;border:1px solid #D0D0D0;"
       "border-radius:3px;padding:3px 6px;font-size:10pt;color:#111;}"
       "QLineEdit:focus{border:1.5px solid #BA7517;}")
_LS = "font-size:10pt;color:#2C2C2A;"
_GB = ("QGroupBox{background:#FFF;border:1px solid #E0E0E0;"
       "border-radius:6px;margin-top:8px;padding:6px 8px;}"
       "QGroupBox::title{subcontrol-origin:margin;left:10px;"
       "padding:0 4px;color:#BA7517;font-size:9pt;font-weight:bold;}")

def _group(title):
    gb = QGroupBox(title); gb.setStyleSheet(_GB)
    vl = QVBoxLayout(gb); vl.setSpacing(4); vl.setContentsMargins(6,4,6,6)
    return gb

def _form_row(form, label, default, hint="", w=140):
    lbl = QLabel(label); lbl.setStyleSheet(_LS)
    container = QWidget(); hl = QHBoxLayout(container)
    hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
    edit = QLineEdit(default); edit.setFixedWidth(w); edit.setStyleSheet(_ES)
    hl.addWidget(edit)
    if hint:
        h = QLabel(hint)
        h.setStyleSheet("font-size:9pt;color:#BBBBBB;font-style:italic;")
        hl.addWidget(h)
    hl.addStretch()
    form.addRow(lbl, container)
    return edit


# ══════════════════════════════════════════════════════════
#  对话框
# ══════════════════════════════════════════════════════════

class PAModelDialog(ModuleDialog):
    TITLE        = "功放模型"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH    = 980
    MIN_HEIGHT   = 660

    _MODELS = [
        ("── 无记忆 ──",       None),
        ("Saleh",              "saleh"),
        ("Modified Rapp",      "rapp"),
        ("── 有记忆 ──",       None),
        ("Memory Polynomial",  "mp"),
        ("Cross-Term Memory",  "ctm"),
    ]

    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1)
        sp.setStyleSheet("QSplitter::handle{background:#DDDDDD;}")

        # ══ 左侧配置 ══════════════════════════════════════
        left = QWidget()
        left.setMinimumWidth(280); left.setMaximumWidth(360)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0,0,8,0); lv.setSpacing(7)

        # ── 模型选择 ──────────────────────────────────────
        mg = _group("模型选择")
        mf = QFormLayout(); mf.setSpacing(5); mf.setContentsMargins(0,0,0,0)
        ml = QLabel("模型类型"); ml.setStyleSheet(_LS)
        self.combo = QComboBox()
        self.combo.setStyleSheet(
            "QComboBox{background:#FFF;border:1px solid #D0D0D0;"
            "border-radius:3px;padding:3px 8px;font-size:10pt;color:#111;}"
            "QComboBox QAbstractItemView{background:#FFF;border:1px solid #D0D0D0;"
            "color:#111;selection-background-color:#FAEEDA;font-size:10pt;}")
        for name, _ in self._MODELS:
            self.combo.addItem(name)
        for i, (n, k) in enumerate(self._MODELS):
            if k is None:
                it = self.combo.model().item(i)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                it.setForeground(Qt.GlobalColor.gray)
        self.combo.setCurrentIndex(1)
        self.combo.currentIndexChanged.connect(self._switch)
        mf.addRow(ml, self.combo)
        mg.layout().addLayout(mf)
        lv.addWidget(mg)

        # ── 参数面板（切换显示）──────────────────────────
        self.saleh_g = self._build_saleh()
        self.rapp_g  = self._build_rapp()
        self.mp_g    = self._build_mp()
        self.ctm_g   = self._build_ctm()
        for g in (self.saleh_g, self.rapp_g, self.mp_g, self.ctm_g):
            lv.addWidget(g)

        # ── 输入功率范围 ───────────────────────────────────
        pg = _group("输入功率范围")
        pf = QFormLayout(); pf.setSpacing(5); pf.setContentsMargins(0,0,0,0)
        self.e_pmin = _form_row(pf, "Pin 起始 (dBm):", "-10", "", 80)
        self.e_pmax = _form_row(pf, "Pin 终止 (dBm):",  "50", "", 80)
        self.e_npts = _form_row(pf, "点数:",            "300", "", 65)
        pg.layout().addLayout(pf)
        lv.addWidget(pg)

        # ── 按钮 ──────────────────────────────────────────
        bhl = QHBoxLayout(); bhl.setSpacing(8); bhl.setContentsMargins(0,4,0,0)
        self.btn_run = QPushButton("计算并绘图")
        self.btn_run.setFixedHeight(32)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#BA7517;color:#FFF;border:none;"
            "border-radius:5px;font-size:10pt;font-weight:bold;}"
            "QPushButton:hover{background:#8B5A0F;}")
        self.btn_run.clicked.connect(self._run)
        bhl.addWidget(self.btn_run)
        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(32)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#FFF;color:#444;border:1px solid #CCC;"
            "border-radius:5px;font-size:10pt;}"
            "QPushButton:hover{background:#F5F5F5;}")
        self.btn_save.clicked.connect(self._save)
        bhl.addWidget(self.btn_save)
        lv.addLayout(bhl)
        lv.addStretch()
        sp.addWidget(left)

        # ══ 右侧画布 ══════════════════════════════════════
        right = QWidget(); rv = QVBoxLayout(right)
        rv.setContentsMargins(6,0,0,0); rv.setSpacing(4)
        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("font-size:9pt;color:#888;")
        rv.addWidget(self.status)
        sp.addWidget(right)
        sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        layout.addWidget(sp, stretch=1)

        self._switch(1)

    # ── 参数面板 ──────────────────────────────────────────

    def _build_saleh(self):
        g = _group("Saleh 参数")
        f = QFormLayout(); f.setSpacing(5); f.setContentsMargins(0,0,0,0)
        note = QLabel("AM/AM: F(r)=αₐ·r/(1+βₐ·r²)\n"
                       "AM/PM: Φ(r)=αₚ·r²/(1+βₚ·r²)  [rad]")
        note.setStyleSheet("font-size:8pt;color:#555;padding:2px 0;")
        note.setWordWrap(True); g.layout().addWidget(note)
        g.layout().addLayout(f)
        self.s_in   = _form_row(f, "Input scaling (dB):",           "0",                "", 70)
        self.s_amam = _form_row(f, "AM/AM 参数 [alpha beta]:",       "[ 2.1587, 1.1517 ]","", 175)
        self.s_ampm = _form_row(f, "AM/PM 参数 [alpha beta]:",       "[ 4.0033, 9.1040 ]","", 175)
        self.s_out  = _form_row(f, "Output scaling (dB):",           "0",                "", 70)
        return g

    def _build_rapp(self):
        g = _group("Modified Rapp 参数")
        f = QFormLayout(); f.setSpacing(5); f.setContentsMargins(0,0,0,0)
        note = QLabel("AM/AM: g·r/(1+(g·r/Vsat)^{2p})^{1/2p}\n"
                       "AM/PM: A·r^q1/(1+(r/B)^q2)  [rad]")
        note.setStyleSheet("font-size:8pt;color:#555;padding:2px 0;")
        note.setWordWrap(True); g.layout().addWidget(note)
        g.layout().addLayout(f)
        self.r_gain = _form_row(f, "Linear power gain (dB):",    "7",           "", 70)
        self.r_vsat = _form_row(f, "Output saturation level (V):","1",           "", 70)
        self.r_p    = _form_row(f, "Magnitude smoothness factor:","2",           "", 70)
        self.r_A    = _form_row(f, "Phase gain (rad):",           "-.45", "-0.45", 70)
        self.r_B    = _form_row(f, "Phase saturation:",           ".88",  "0.88",  70)
        self.r_q    = _form_row(f, "Phase smoothness factor:",    "[ 3.43, 3.43 ]","", 140)
        return g

    def _build_mp(self):
        g = _group("Memory Polynomial (eq.19)")
        f = QFormLayout(); f.setSpacing(5); f.setContentsMargins(0,0,0,0)
        note = QLabel(
            "y(n) = Σ_{m=0}^{M-1} Σ_{d=0}^{D-1} C[m,d]·x(n-m)·|x(n-m)|^d\n"
            "C 矩阵维度: M×D（M=记忆深度，D=电压阶次）\n"
            "d=0:线性  d=1:三阶  d=2:五阶...\n"
            "使用 OFDM 信号提取 AM/AM 和 AM/PM")
        note.setStyleSheet("font-size:8pt;color:#555;padding:2px 0;")
        note.setWordWrap(True); g.layout().addWidget(note)
        g.layout().addLayout(f)
        self.mp_M   = _form_row(f, "记忆深度 M:", "3", "", 60)
        self.mp_D   = _form_row(f, "电压阶次 D:", "3", "", 60)
        self.mp_in  = _form_row(f, "Input scaling (dB):",  "0", "", 60)
        self.mp_out = _form_row(f, "Output scaling (dB):", "0", "", 60)
        self.mp_sym = _form_row(f, "OFDM 符号数:", "60", "", 60)
        return g

    def _build_ctm(self):
        g = _group("Cross-Term Memory (eq.23)")
        f = QFormLayout(); f.setSpacing(5); f.setContentsMargins(0,0,0,0)
        note = QLabel(
            "M_CTM[m,col] = x(n-m)·|x(n-l)|^d\n"
            "C 矩阵维度: M × (M·(D-1)+1)\n"
            "包含所有延迟时刻的包络交叉项\n"
            "使用 OFDM 信号提取 AM/AM 和 AM/PM")
        note.setStyleSheet("font-size:8pt;color:#555;padding:2px 0;")
        note.setWordWrap(True); g.layout().addWidget(note)
        g.layout().addLayout(f)
        self.ctm_M   = _form_row(f, "记忆深度 M:", "3", "", 60)
        self.ctm_D   = _form_row(f, "电压阶次 D:", "3", "", 60)
        self.ctm_in  = _form_row(f, "Input scaling (dB):",  "0", "", 60)
        self.ctm_out = _form_row(f, "Output scaling (dB):", "0", "", 60)
        self.ctm_sym = _form_row(f, "OFDM 符号数:", "60", "", 60)
        return g

    # ── 切换 ──────────────────────────────────────────────

    def _switch(self, _=None):
        key = self._key()
        self.saleh_g.setVisible(key == "saleh")
        self.rapp_g.setVisible(key == "rapp")
        self.mp_g.setVisible(key == "mp")
        self.ctm_g.setVisible(key == "ctm")

    def _key(self):
        i = self.combo.currentIndex()
        return self._MODELS[i][1] if 0 <= i < len(self._MODELS) else None

    # ── 计算 ──────────────────────────────────────────────

    def _run(self):
        try:    self._do_run()
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "计算错误", str(e))

    def _do_run(self):
        key = self._key()
        if not key:
            self.status.setText("请选择有效模型"); return

        pmin = _f(self.e_pmin.text(), -10)
        pmax = _f(self.e_pmax.text(),  50)
        N    = max(10, int(_f(self.e_npts.text(), 300)))
        pins = np.linspace(pmin, pmax, N)

        if key == "saleh":
            isc    = _f(self.s_in.text(),   0)
            osc    = _f(self.s_out.text(),  0)
            aa, ba = _vec2(self.s_amam.text(), (2.1587, 1.1517))
            ap, bp = _vec2(self.s_ampm.text(), (4.0033, 9.1040))
            r      = _dbm2r_arr(pins + isc)
            pout   = _r2dbm_arr(saleh_amam(r, aa, ba)) + osc
            phi    = saleh_ampm(r, ap, bp)
            g_db   = 20 * math.log10(aa) + osc + isc
            title  = "Saleh"

        elif key == "rapp":
            G_db   = _f(self.r_gain.text(), 7)
            g_lin  = 10 ** (G_db / 20.0)
            vsat   = _f(self.r_vsat.text(), 1.0)
            p      = _f(self.r_p.text(),    2.0)
            A      = _f(self.r_A.text(),   -0.45)
            B      = _f(self.r_B.text(),    0.88)
            q1, q2 = _vec2(self.r_q.text(), (3.43, 3.43))
            r      = _dbm2r_arr(pins)
            pout   = _r2dbm_arr(rapp_amam(r, g_lin, vsat, p))
            phi    = rapp_ampm(r, A, B, q1, q2)
            g_db   = G_db
            title  = "Modified Rapp"

        elif key == "mp":
            M   = max(1, int(_f(self.mp_M.text(),   3)))
            D   = max(1, int(_f(self.mp_D.text(),   3)))
            isc = _f(self.mp_in.text(),  0)
            osc = _f(self.mp_out.text(), 0)
            n_sym = max(10, int(_f(self.mp_sym.text(), 60)))
            C   = _default_mp_coeffs(M, D)
            self.status.setText("计算中，请稍候（OFDM 信号处理）…")
            QWidget.repaint(self)
            pout, phi = _mem_sweep(pins, C, 'MP', isc, osc, n_sym=n_sym)
            g_db  = 20 * math.log10(abs(C[0, 0]) + 1e-30) + isc + osc
            title = f"Memory Polynomial  (M={M}, D={D})"

        elif key == "ctm":
            M   = max(1, int(_f(self.ctm_M.text(),   3)))
            D   = max(1, int(_f(self.ctm_D.text(),   3)))
            isc = _f(self.ctm_in.text(),  0)
            osc = _f(self.ctm_out.text(), 0)
            n_sym = max(10, int(_f(self.ctm_sym.text(), 60)))
            C   = _default_ctm_coeffs(M, D)
            self.status.setText("计算中，请稍候（OFDM 信号处理）…")
            QWidget.repaint(self)
            pout, phi = _mem_sweep(pins, C, 'CTM', isc, osc, n_sym=n_sym)
            g_db  = 20 * math.log10(abs(C[0, 0]) + 1e-30) + isc + osc
            title = f"Cross-Term Memory  (M={M}, D={D})"

        else:
            return

        sat_i    = int(np.argmax(pout))
        pin_sat  = pins[sat_i]
        pout_sat = pout[sat_i]
        lin_ref  = pins + g_db

        self._plot(pins, pout, phi, lin_ref, g_db, pin_sat, pout_sat, title)
        self.status.setText(
            f"{title}  |  G={g_db:.1f}dB  "
            f"Pout_sat={pout_sat:.1f}dBm @ Pin={pin_sat:.1f}dBm")

    # ── 绘图 ──────────────────────────────────────────────

    def _plot(self, pins, pout, phi, lin_ref, g_db,
              pin_sat, pout_sat, title):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.08,
                                        hspace=0.08, wspace=0.05)
        ax1, ax2 = fig.subplots(2, 1)
        fig.patch.set_facecolor("#F8F8F8")

        # AM/AM
        ax1.set_facecolor("#FFFFFF")
        ax1.plot(pins, pout,    color="#0055CC", lw=2,   label=title.split("  ")[0])
        ax1.plot(pins, lin_ref, color="#CC2200", lw=1.2, ls="-.", label="Linear Gain")
        ax1.plot(pin_sat, pout_sat, "o", color="#555", ms=7, zorder=5)
        ax1.axhline(pout_sat, color="#AAAAAA", lw=0.8, ls="--")
        ax1.axvline(pin_sat,  color="#AAAAAA", lw=0.8, ls="--")

        ylim = ax1.get_ylim()
        yr   = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1
        xr   = pins[-1] - pins[0]
        ax1.text(pins[0] + xr*0.03, pout_sat + yr*0.04,
                 f"Pout$_{{sat}}$ = {pout_sat:.1f}", fontsize=8.5, color="#333")
        ax1.annotate(f"← Pin$_{{sat}}$ = {pin_sat:.1f}",
                     xy=(pin_sat, pout_sat),
                     xytext=(pin_sat - xr*0.32, pout_sat - yr*0.15),
                     fontsize=8.5, color="#333",
                     arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))
        ax1.annotate(f"Pout = Pin + {g_db:.1f}",
                     xy=(pins[len(pins)//4], lin_ref[len(pins)//4]),
                     xytext=(pins[len(pins)//4] + xr*0.08,
                             lin_ref[len(pins)//4] + yr*0.06),
                     fontsize=8.5, color="#CC2200",
                     arrowprops=dict(arrowstyle="->", color="#CC2200", lw=0.8))
        ax1.set_xlabel("P$_{in}$  (dBm)", fontsize=9)
        ax1.set_ylabel("P$_{out}$  (dBm)", fontsize=9)
        ax1.set_title(f"{title} AM/AM", fontsize=10)
        ax1.legend(fontsize=8.5, framealpha=0.9, edgecolor="#DDD", loc="upper left")
        ax1.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax1.spines.values(): s.set_color("#CCCCCC")
        ax1.tick_params(labelsize=8)

        # AM/PM
        ax2.set_facecolor("#FFFFFF")
        ax2.plot(pins, phi, color="#0055CC", lw=2)
        ax2.set_xlabel("P$_{in}$  (dBm)", fontsize=9)
        ax2.set_ylabel("Phase  (degs)", fontsize=9)
        ax2.set_title(f"{title} AM/PM", fontsize=10)
        ax2.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax2.spines.values(): s.set_color("#CCCCCC")
        ax2.tick_params(labelsize=8)

        self.canvas.draw()

    def _save(self):
        key  = self._key() or "pa"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", f"PA_{key}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")