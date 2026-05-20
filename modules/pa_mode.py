"""
功放表征模块 — Power Amplifier Characterization
================================================
按 MathWorks PowerAmplifierCharacterizationExample 流程实现：

  1. 从 .mat 文件读取功放实测数据
       helperPACharSavedData<BW>MHz.mat
       results.InputWaveform / OutputWaveform / ReferencePower / MeasuredAMToAM
  2. 生成 5G-like OFDM 测试波形 (helperPACharGenerateOFDM)
  3. 由实测数据绘制 AM/AM 与 Gain vs Input Power 曲线
  4. 用记忆多项式模型仿真功放，模型可选：
       MP  — Memory Polynomial          (memPoly)
       CM  — Cross-Term Memory Polynomial (ctMemPoly)
     记忆长度 memLen 与多项式阶数 degLen 可调
  5. 求得拟合系数矩阵 fitCoefMatMem，可导出
  6. 绘制 helperPACharPlotGain：实测 vs 拟合 增益对比

参考：
  - PowerAmplifierCharacterizationExample, The MathWorks Inc.
  - helperPACharMemPolyModel.m
  - IEEE Std 802.11a-1999, Eq.28 (RMS EVM)
"""

import os
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
    QMessageBox, QSizePolicy, QPlainTextEdit,
)
from PyQt6.QtCore import Qt

from ui.base_dialog import ModuleDialog


# ── 字体 ──────────────────────────────────────────────────
def _setup_font():
    for n in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
        if n in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = n
            plt.rcParams["axes.unicode_minus"] = False
            return
_setup_font()


# ══════════════════════════════════════════════════════════
#  1. OFDM 测试波形生成  (helperPACharGenerateOFDM)
# ══════════════════════════════════════════════════════════

# bw -> (scs, fftLength, NSubcarriers, cpLength, windowLength)
_OFDM_PARAMS = {
    5e6:   (30e3,  256,  132,  18,  6),
    15e6:  (30e3, 1024,  456,  72,  6),
    40e6:  (30e3, 2048, 1272, 144,  8),
    100e6: (30e3, 4096, 3276, 288, 20),
}


def generate_ofdm(bw: float):
    """
    生成 5G-like 64-QAM OFDM 复基带波形。

    返回 (txWaveform, sampleRate, numFrames)
      txWaveform : 1-D complex ndarray（已归一化，峰值=1）
      sampleRate : 采样率 (Hz)
      numFrames  : 帧数

    说明：matplotlib/numpy 无 comm.OFDMModulator，此处用 IFFT 直接
    实现等效的加保护带 + 循环前缀 + 过采样 OFDM 调制。
    """
    if bw not in _OFDM_PARAMS:
        raise ValueError(f"不支持的带宽 {bw/1e6} MHz，可选 5/15/40/100 MHz")

    scs, fftLength, NSub, cpLength, _win = _OFDM_PARAMS[bw]
    M = 64                       # 64-QAM
    osr = 7                      # 过采样率
    numFrames = 30
    sampleRate = scs * fftLength * osr
    nGuard = fftLength - NSub    # 保护带子载波总数

    # 64-QAM 星座（Gray 映射，平均功率归一化）
    k = int(np.sqrt(M))          # 8
    lvl = np.arange(-(k - 1), k, 2)
    I, Q = np.meshgrid(lvl, lvl)
    const = (I.ravel() + 1j * Q.ravel())
    const = const / np.sqrt(np.mean(np.abs(const) ** 2))

    rng = np.random.RandomState(12345)         # 可复现
    nDataSc = fftLength - nGuard - 1           # 每帧数据子载波数

    # 保护带与 DC 置零的子载波映射
    g_lo = nGuard // 2 + 1
    g_hi = nGuard // 2
    frames = []
    for _ in range(numFrames):
        idx = rng.randint(0, M, nDataSc)
        sym = const[idx]
        grid = np.zeros(fftLength, dtype=complex)
        # 数据子载波放在 [g_lo, g_lo+half) 与 (DC) 两侧
        half = nDataSc // 2
        grid[g_lo:g_lo + half] = sym[:half]
        grid[g_lo + half + 1:g_lo + half + 1 + (nDataSc - half)] = sym[half:]
        td = np.fft.ifft(np.fft.ifftshift(grid)) * np.sqrt(fftLength)
        td = np.concatenate([td[-cpLength:], td])          # 加循环前缀
        td = _resample(td, osr)                            # 过采样
        frames.append(td)

    txWaveform = np.concatenate(frames)
    pk = np.max(np.abs(txWaveform))
    if pk > 0:
        txWaveform = txWaveform / pk
    return txWaveform, sampleRate, numFrames


def _resample(x: np.ndarray, factor: int) -> np.ndarray:
    """整数倍频域过采样（zero-padding interpolation）。"""
    n = len(x)
    X = np.fft.fft(x)
    Xup = np.zeros(n * factor, dtype=complex)
    h = n // 2
    Xup[:h] = X[:h]
    Xup[-h:] = X[-h:]
    return np.fft.ifft(Xup) * factor


# ══════════════════════════════════════════════════════════
#  读取功放实测数据  (load helperPACharSavedData<BW>MHz.mat)
# ══════════════════════════════════════════════════════════

class PAData:
    """功放实测数据容器。"""
    __slots__ = ("input_wave", "output_wave", "reference_power",
                 "measured_amam", "linear_gain", "sample_rate",
                 "oversampling_rate", "num_frames", "bw")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def load_pa_data(mat_path: str) -> PAData:
    """
    读取 MATLAB 保存的功放表征数据文件。
    对应 MATLAB: load(dataFileName, "results", "sampleRate", ...)
    """
    import scipy.io as sio
    m = sio.loadmat(mat_path)
    if "results" not in m:
        raise ValueError("文件中缺少 results 结构体，"
                          "请选择 helperPACharSavedData*MHz.mat 文件")
    r = m["results"][0, 0]

    def fld(name):
        return r[name].ravel() if name in r.dtype.names else None

    sr = m["sampleRate"].item() if "sampleRate" in m else None
    osr = int(m["overSamplingRate"].item()) if "overSamplingRate" in m else None
    nf = int(m["numFrames"].item()) if "numFrames" in m else None

    # 从文件名推断带宽
    bw = None
    base = os.path.basename(mat_path)
    import re
    mt = re.search(r"(\d+)MHz", base)
    if mt:
        bw = float(mt.group(1)) * 1e6

    return PAData(
        input_wave=np.asarray(fld("InputWaveform"), dtype=np.complex128),
        output_wave=np.asarray(fld("OutputWaveform"), dtype=np.complex128),
        reference_power=np.asarray(fld("ReferencePower"), dtype=np.float64),
        measured_amam=np.asarray(fld("MeasuredAMToAM"), dtype=np.float64),
        linear_gain=float(r["LinearGain"].item())
        if "LinearGain" in r.dtype.names else None,
        sample_rate=sr, oversampling_rate=osr, num_frames=nf, bw=bw,
    )


# ══════════════════════════════════════════════════════════
#  记忆多项式模型  (helperPACharMemPolyModel)
#  modType: 'memPoly'  → MP   (Memory Polynomial)
#           'ctMemPoly'→ CM   (Cross-Term Memory Polynomial)
# ══════════════════════════════════════════════════════════

def mp_coefficient_finder(x, y, memLen, degLen, modType):
    """
    coefficientFinder：由输入/输出信号最小二乘求解系数矩阵。

    返回 coefMat
      MP : 形状 (memLen, degLen)
      CM : 形状 (memLen, memLen*(degLen-1)+1)

    严格对应 MATLAB helperPACharMemPolyModel 'coefficientFinder' 分支。
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    xLen = len(x)

    if modType == "memPoly":
        # xrow = reshape((memLen:-1:1)' + (0:xLen:xLen*(degLen-1)), 1, [])
        a = np.arange(memLen, 0, -1).reshape(-1, 1)
        b = np.arange(0, xLen * degLen, xLen).reshape(1, -1)
        xrow = (a + b).reshape(1, -1, order="F")
        rows = np.arange(0, xLen - memLen + 1).reshape(-1, 1)
        xVecIdx = rows + xrow
        # xPow = x .* (abs(x).^(0:degLen-1))
        xPow = x.reshape(-1, 1) * (np.abs(x).reshape(-1, 1) ** np.arange(degLen))
        xVec = xPow.flatten(order="F")[xVecIdx - 1]

    elif modType == "ctMemPoly":
        L = xLen - memLen + 1
        # absPow = abs(x).^(1:degLen-1)
        absPow = np.abs(x).reshape(-1, 1) ** np.arange(1, degLen)
        a = np.arange(memLen, 0, -1).reshape(-1, 1)
        b = np.arange(0, xLen * (degLen - 1), xLen).reshape(1, -1)
        partTop1 = (a + b).reshape(1, -1, order="F")
        rows = np.arange(0, L).reshape(-1, 1)
        topVals = absPow.flatten(order="F")[(rows + partTop1) - 1]
        topPlane_2d = np.concatenate([np.ones((L, 1)), topVals], axis=1)
        ncolsTop = memLen * (degLen - 1) + 1
        topPlane = topPlane_2d.T.reshape(1, ncolsTop, L)
        sideIdx = rows + a.reshape(1, -1)
        side = x[sideIdx - 1]
        sidePlane = side.T.reshape(memLen, 1, L)
        cube = sidePlane * topPlane
        xVec = cube.reshape(memLen * ncolsTop, L, order="F").T
    else:
        raise ValueError(f"未知模型类型 {modType}")

    rhs = y[memLen - 1:xLen]
    # 用 QR + 列主元最小二乘（LAPACK gelsy），与 MATLAB 反斜杠 mldivide
    # 同属一类算法。注意：PA 数据为 7 倍过采样，线性抽头
    # x(n-m) 高度共线，最小二乘解在该子空间内非唯一——不同 LAPACK
    # 例程会给出不同但等价的系数（预测输出与增益曲线一致）。
    import scipy.linalg as _sla
    coef, *_ = _sla.lstsq(xVec, rhs, lapack_driver="gelsy")
    return coef.reshape(memLen, -1, order="F")


def mp_signal_generator(x, coefMat, modType):
    """
    signalGenerator：由输入信号与系数矩阵生成功放输出（向量化实现，
    与 MATLAB 逐点循环算法数值等价）。
    """
    x = np.asarray(x).ravel()
    memLen, numCols = coefMat.shape
    N = len(x)
    y = np.zeros(N, dtype=complex)

    # 预计算延迟副本 xd[m][n] = x[n-m]
    xd = np.zeros((memLen, N), dtype=complex)
    for m in range(memLen):
        xd[m, m:] = x[:N - m]
    ad = np.abs(xd)

    if modType == "memPoly":
        degLen = numCols
        for m in range(memLen):
            for d in range(degLen):
                y += coefMat[m, d] * xd[m] * (ad[m] ** d)

    elif modType == "ctMemPoly":
        degLen = round((numCols - 1) / memLen) + 1
        # col 0: 线性项
        for m in range(memLen):
            y += coefMat[m, 0] * xd[m]
        # col = 1 + memLen*(powIdx-1) + l
        for powIdx in range(1, degLen):
            for l in range(memLen):
                col = 1 + memLen * (powIdx - 1) + l
                env = ad[l] ** powIdx
                for m in range(memLen):
                    y += coefMat[m, col] * xd[m] * env
    else:
        raise ValueError(f"未知模型类型 {modType}")

    y[:memLen - 1] = 0
    return y


def mp_error_measure(x, y, coefMat, modType):
    """
    errorMeasure：计算时域 RMS 误差百分比 (IEEE 802.11a Eq.28)。
    """
    memLen = coefMat.shape[0]
    yp = mp_signal_generator(x, coefMat, modType)
    err = y[memLen - 1:] - yp[memLen - 1:]
    return np.sqrt(np.mean(np.abs(err) ** 2)) / \
        np.sqrt(np.mean(np.abs(y[memLen - 1:]) ** 2)) * 100


# ══════════════════════════════════════════════════════════
#  解析工具
# ══════════════════════════════════════════════════════════

def _f(text, default=0.0):
    try:    return float(str(text).strip())
    except: return default

def _i(text, default=0):
    try:    return int(float(str(text).strip()))
    except: return default


# ══════════════════════════════════════════════════════════
#  画布
# ══════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6.4), dpi=96)
        self.fig.patch.set_facecolor("#F8F8F8")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════
#  UI 样式
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
    vl = QVBoxLayout(gb); vl.setSpacing(4); vl.setContentsMargins(6, 4, 6, 6)
    return gb


def _form_row(form, label, default, hint="", w=140):
    lbl = QLabel(label); lbl.setStyleSheet(_LS)
    container = QWidget(); hl = QHBoxLayout(container)
    hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(6)
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
    TITLE        = "功放表征"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH    = 1040
    MIN_HEIGHT   = 700

    # 模型类型：显示名 -> (内部 modType, 简称)
    _MODELS = [
        ("Memory Polynomial (MP)",        "memPoly"),
        ("Cross-Term Memory (CM)",        "ctMemPoly"),
    ]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pa_data = None      # PAData
        self.tx_waveform = None  # 生成的 OFDM
        self.coef_mat = None     # fitCoefMatMem
        self.last_modType = None

    # ──────────────────────────────────────────────────────
    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1)
        sp.setStyleSheet("QSplitter::handle{background:#DDDDDD;}")

        # ══ 左侧配置 ══════════════════════════════════════
        left = QWidget()
        left.setMinimumWidth(310); left.setMaximumWidth(390)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0); lv.setSpacing(7)

        # ── ① 功放数据文件 ─────────────────────────────────
        dg = _group("① 功放实测数据")
        df = QVBoxLayout(); df.setSpacing(5)
        fl = QHBoxLayout(); fl.setSpacing(6)
        self.e_file = QLineEdit(); self.e_file.setReadOnly(True)
        self.e_file.setStyleSheet(_ES)
        self.e_file.setPlaceholderText("helperPACharSavedData100MHz.mat")
        btn_browse = QPushButton("浏览…")
        btn_browse.setFixedWidth(64); btn_browse.setFixedHeight(28)
        btn_browse.setStyleSheet(
            "QPushButton{background:#FFF;color:#444;border:1px solid #CCC;"
            "border-radius:4px;font-size:9pt;}"
            "QPushButton:hover{background:#F5F5F5;}")
        btn_browse.clicked.connect(self._browse_file)
        fl.addWidget(self.e_file); fl.addWidget(btn_browse)
        df.addLayout(fl)
        self.lbl_datainfo = QLabel("未加载数据")
        self.lbl_datainfo.setStyleSheet("font-size:8.5pt;color:#888;")
        self.lbl_datainfo.setWordWrap(True)
        df.addWidget(self.lbl_datainfo)
        dg.layout().addLayout(df)
        lv.addWidget(dg)

        # ── ② OFDM 测试波形 ───────────────────────────────
        og = _group("② OFDM 测试波形")
        of = QFormLayout(); of.setSpacing(5); of.setContentsMargins(0, 0, 0, 0)
        ol = QLabel("信道带宽"); ol.setStyleSheet(_LS)
        self.combo_bw = QComboBox()
        self.combo_bw.setStyleSheet(
            "QComboBox{background:#FFF;border:1px solid #D0D0D0;"
            "border-radius:3px;padding:3px 8px;font-size:10pt;color:#111;}"
            "QComboBox QAbstractItemView{background:#FFF;border:1px solid #D0D0D0;"
            "color:#111;selection-background-color:#FAEEDA;font-size:10pt;}")
        for s in ["5 MHz", "15 MHz", "40 MHz", "100 MHz"]:
            self.combo_bw.addItem(s)
        self.combo_bw.setCurrentIndex(3)
        of.addRow(ol, self.combo_bw)
        og.layout().addLayout(of)
        self.btn_ofdm = QPushButton("生成 OFDM 并绘制 AM/AM、Gain")
        self.btn_ofdm.setFixedHeight(30)
        self.btn_ofdm.setStyleSheet(
            "QPushButton{background:#FFF;color:#BA7517;border:1px solid #BA7517;"
            "border-radius:5px;font-size:9.5pt;font-weight:bold;}"
            "QPushButton:hover{background:#FAEEDA;}")
        self.btn_ofdm.clicked.connect(self._run_characterize)
        og.layout().addWidget(self.btn_ofdm)
        lv.addWidget(og)

        # ── ③ 拟合模型 ────────────────────────────────────
        mg = _group("③ 记忆多项式拟合模型")
        mf = QFormLayout(); mf.setSpacing(5); mf.setContentsMargins(0, 0, 0, 0)
        ml = QLabel("模型类型"); ml.setStyleSheet(_LS)
        self.combo_model = QComboBox()
        self.combo_model.setStyleSheet(self.combo_bw.styleSheet())
        for name, _ in self._MODELS:
            self.combo_model.addItem(name)
        mf.addRow(ml, self.combo_model)
        mg.layout().addLayout(mf)

        note = QLabel(
            "MP : y(n)=ΣΣ C[m,d]·x(n-m)·|x(n-m)|^d\n"
            "CM : 含所有延迟时刻的包络交叉项\n"
            "memLen=记忆长度  degLen=多项式阶数")
        note.setStyleSheet("font-size:8pt;color:#666;padding:2px 0;")
        note.setWordWrap(True)
        mg.layout().addWidget(note)

        pf = QFormLayout(); pf.setSpacing(5); pf.setContentsMargins(0, 0, 0, 0)
        self.e_memlen = _form_row(pf, "memLen 记忆长度:", "5", "", 60)
        self.e_deglen = _form_row(pf, "degLen 多项式阶数:", "5", "", 60)
        mg.layout().addLayout(pf)

        self.btn_fit = QPushButton("拟合模型并绘制 Gain 对比")
        self.btn_fit.setFixedHeight(32)
        self.btn_fit.setStyleSheet(
            "QPushButton{background:#BA7517;color:#FFF;border:none;"
            "border-radius:5px;font-size:10pt;font-weight:bold;}"
            "QPushButton:hover{background:#8B5A0F;}")
        self.btn_fit.clicked.connect(self._run_fit)
        mg.layout().addWidget(self.btn_fit)
        lv.addWidget(mg)

        # ── ④ 拟合系数 / 导出 ─────────────────────────────
        cg = _group("④ 拟合系数 fitCoefMatMem")
        self.txt_coef = QPlainTextEdit()
        self.txt_coef.setReadOnly(True)
        self.txt_coef.setFixedHeight(110)
        self.txt_coef.setStyleSheet(
            "QPlainTextEdit{background:#FFF;border:1px solid #D0D0D0;"
            "border-radius:3px;font-family:Consolas,monospace;"
            "font-size:8.5pt;color:#222;}")
        self.txt_coef.setPlainText("尚未拟合")
        cg.layout().addWidget(self.txt_coef)
        eh = QHBoxLayout(); eh.setSpacing(8)
        self.btn_export = QPushButton("导出系数与参数")
        self.btn_export.setFixedHeight(30)
        self.btn_export.setStyleSheet(
            "QPushButton{background:#FFF;color:#444;border:1px solid #CCC;"
            "border-radius:5px;font-size:9.5pt;}"
            "QPushButton:hover{background:#F5F5F5;}")
        self.btn_export.clicked.connect(self._export)
        eh.addWidget(self.btn_export)
        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setStyleSheet(self.btn_export.styleSheet())
        self.btn_save.clicked.connect(self._save)
        eh.addWidget(self.btn_save)
        cg.layout().addLayout(eh)
        lv.addWidget(cg)

        lv.addStretch()
        sp.addWidget(left)

        # ══ 右侧画布 ══════════════════════════════════════
        right = QWidget(); rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0); rv.setSpacing(4)
        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas)
        self.status = QLabel("就绪 — 请先加载功放数据文件")
        self.status.setStyleSheet("font-size:9pt;color:#888;")
        rv.addWidget(self.status)
        sp.addWidget(right)
        sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        layout.addWidget(sp, stretch=1)

    # ──────────────────────────────────────────────────────
    #  ① 文件加载
    # ──────────────────────────────────────────────────────
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择功放数据文件", "",
            "MAT 文件 (*.mat);;所有文件 (*)")
        if not path:
            return
        try:
            self.pa_data = load_pa_data(path)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            return

        d = self.pa_data
        self.e_file.setText(os.path.basename(path))
        info = (f"输入波形: {len(d.input_wave)} 点  |  "
                f"输出波形: {len(d.output_wave)} 点\n"
                f"采样率: {d.sample_rate/1e6:.1f} MHz  |  "
                f"过采样: {d.oversampling_rate}  |  帧数: {d.num_frames}\n"
                f"线性增益: {d.linear_gain:.2f} dB"
                + (f"  |  带宽: {d.bw/1e6:.0f} MHz" if d.bw else ""))
        self.lbl_datainfo.setText(info)
        self.lbl_datainfo.setStyleSheet("font-size:8.5pt;color:#3A7D44;")
        # 同步带宽下拉
        if d.bw:
            for i, s in enumerate(["5 MHz", "15 MHz", "40 MHz", "100 MHz"]):
                if abs(float(s.split()[0]) * 1e6 - d.bw) < 1:
                    self.combo_bw.setCurrentIndex(i)
        self.status.setText("数据已加载 — 可生成 OFDM 并绘制曲线")
        # 自动绘制实测特性
        self._run_characterize()

    def _bw_value(self):
        return float(self.combo_bw.currentText().split()[0]) * 1e6

    # ──────────────────────────────────────────────────────
    #  ② / ③ 实测特性绘制
    # ──────────────────────────────────────────────────────
    def _run_characterize(self):
        if self.pa_data is None:
            QMessageBox.information(self, "提示", "请先加载功放数据文件")
            return
        try:
            self.status.setText("生成 OFDM 测试波形…")
            QWidget.repaint(self)
            bw = self._bw_value()
            tx, sr, nf = generate_ofdm(bw)
            self.tx_waveform = tx
            self._plot_characterization()
            self.status.setText(
                f"OFDM 已生成: {len(tx)} 点 @ {sr/1e6:.1f} MHz, "
                f"{nf} 帧  |  已绘制实测 AM/AM 与 Gain 曲线")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "错误", str(e))

    def _plot_characterization(self):
        """绘制实测 AM/AM 与 Gain vs Input Power（上 OFDM 频谱，下 AM/AM+Gain）。"""
        d = self.pa_data
        refP = d.reference_power
        amam = d.measured_amam

        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        ax1, ax2, ax3 = fig.subplots(3, 1)
        fig.patch.set_facecolor("#F8F8F8")

        # — OFDM 输入频谱（PA Input）—
        ax1.set_facecolor("#FFFFFF")
        x = self.tx_waveform
        sr = d.sample_rate or 1.0
        win = np.hanning(len(x))
        X = np.fft.fftshift(np.fft.fft(x * win))
        psd = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-12)
        freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / sr)) / 1e6
        ax1.plot(freq, psd, color="#0055CC", lw=0.7)
        ax1.set_xlabel("频率 (MHz)", fontsize=9)
        ax1.set_ylabel("归一化功率 (dB)", fontsize=9)
        ax1.set_title("OFDM 测试波形频谱 (PA Input)", fontsize=10)
        ax1.set_ylim([-100, 5])
        ax1.grid(True, color="#E8E8E8", lw=0.5)

        # — AM/AM —
        ax2.set_facecolor("#FFFFFF")
        ax2.plot(refP, refP + amam, ".", color="#0055CC", ms=2)
        ax2.set_xlabel("输入功率 (dBm)", fontsize=9)
        ax2.set_ylabel("输出功率 (dBm)", fontsize=9)
        ax2.set_title("AM/AM 实测特性", fontsize=10)
        ax2.grid(True, color="#E8E8E8", lw=0.5)

        # — Gain vs Input Power —
        ax3.set_facecolor("#FFFFFF")
        ax3.plot(refP, amam, ".", color="#CC2200", ms=2)
        ax3.set_xlabel("输入功率 (dBm)", fontsize=9)
        ax3.set_ylabel("增益 (dB)", fontsize=9)
        ax3.set_title("Gain vs Input Power 实测特性", fontsize=10)
        ax3.grid(True, color="#E8E8E8", lw=0.5)

        for ax in (ax1, ax2, ax3):
            for s in ax.spines.values():
                s.set_color("#CCCCCC")
            ax.tick_params(labelsize=8)

        self.canvas.draw()

    # ──────────────────────────────────────────────────────
    #  ④ 模型拟合
    # ──────────────────────────────────────────────────────
    def _run_fit(self):
        if self.pa_data is None:
            QMessageBox.information(self, "提示", "请先加载功放数据文件")
            return
        try:
            self._do_fit()
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "拟合错误", str(e))

    def _do_fit(self):
        d = self.pa_data
        paInput = d.input_wave
        paOutput = d.output_wave

        modType = self._MODELS[self.combo_model.currentIndex()][1]
        memLen = max(1, _i(self.e_memlen.text(), 5))
        degLen = max(1, _i(self.e_deglen.text(), 5))

        self.status.setText("拟合记忆多项式系数…")
        QWidget.repaint(self)

        # —— coefficientFinder：用前半段数据拟合 ——
        numDataPts = len(paInput)
        halfDataPts = round(numDataPts / 2)
        coefMat = mp_coefficient_finder(
            paInput[:halfDataPts], paOutput[:halfDataPts],
            memLen, degLen, modType)
        self.coef_mat = coefMat
        self.last_modType = modType

        # —— errorMeasure：全段时域 RMS 误差 ——
        self.status.setText("计算时域 RMS 误差…")
        QWidget.repaint(self)
        rmsErr = mp_error_measure(paInput, paOutput, coefMat, modType)

        # —— 由模型生成拟合输出 paOutputFitMem ——
        paOutputFit = mp_signal_generator(paInput, coefMat, modType)

        # —— 显示系数矩阵 abs(fitCoefMatMem) ——
        absC = np.abs(coefMat)
        lines = [f"模型: {modType}   memLen={memLen}  degLen={degLen}",
                 f"系数矩阵维度: {coefMat.shape[0]} × {coefMat.shape[1]}",
                 "abs(fitCoefMatMem):"]
        for row in absC:
            lines.append("  " + "  ".join(f"{v:9.4f}" for v in row))
        lines.append(f"时域 RMS 误差: {rmsErr:.4f} %")
        lines.append("注: 数据为7倍过采样，线性抽头共线，")
        lines.append("    系数解非唯一；增益曲线与RMS误差为准。")
        self.txt_coef.setPlainText("\n".join(lines))

        # —— 绘制 Gain 拟合对比 (helperPACharPlotGain) ——
        self._plot_gain(paInput, paOutput, paOutputFit, modType, rmsErr)

        short = "MP" if modType == "memPoly" else "CM"
        self.status.setText(
            f"{short} 拟合完成  |  memLen={memLen} degLen={degLen}  |  "
            f"时域 RMS 误差 = {rmsErr:.3f}%")

    def _plot_gain(self, paInput, paOutput, paOutputFit, modType, rmsErr):
        """
        helperPACharPlotGain 等效（与 MATLAB 示例图一致）：
        单幅 "Comparison of Actual and Estimated Gain"，
        横轴输入功率 (dBm)，纵轴功率增益 (dB)。

          实测增益  Actual Gain    : 蓝色空心圆
            直接取自数据文件 results.ReferencePower / MeasuredAMToAM
          估计增益  Estimated Gain : 橙色实心点
            由记忆多项式模型输出 paOutputFitMem 计算增益

        功率换算与 MATLAB spectrumAnalyzer 一致，参考阻抗负载 100 Ω：
          P (W)   = |V|^2 / R
          P (dBm) = 10·log10(P / 1e-3)
        """
        REF_LOAD = 100.0          # 与 sa.ReferenceLoad = 100 一致
        memLen = self.coef_mat.shape[0]
        d = self.pa_data

        # ── 实测增益：直接用保存的参考功率与 AM/AM 测量 ──
        refP = d.reference_power          # 输入功率 (dBm)
        gain_act = d.measured_amam        # 实测增益 (dB)

        # ── 估计增益：由模型预测输出计算 ──
        # 丢弃前 memLen 个暂态点（与 errorMeasure 一致）
        xin  = paInput[memLen:]
        yfit = paOutputFit[memLen:]
        mask = np.abs(xin) > 1e-9
        xi, yf = xin[mask], yfit[mask]
        pin_est  = 10 * np.log10(np.abs(xi) ** 2 / (REF_LOAD * 1e-3) + 1e-30)
        gain_est = 20 * np.log10(np.abs(yf) / np.abs(xi) + 1e-30)

        short = "MP" if modType == "memPoly" else "CM"

        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        ax = fig.subplots(1, 1)
        fig.patch.set_facecolor("#F8F8F8")
        ax.set_facecolor("#FFFFFF")

        # 实测增益：蓝色空心圆
        ax.scatter(refP, gain_act, s=22, facecolors="none",
                   edgecolors="#1F77B4", linewidths=0.7,
                   label="Actual Gain")
        # 估计增益：橙色实心点
        ax.scatter(pin_est, gain_est, s=6, color="#D95319",
                   label="Estimated Gain")

        # 坐标轴限制在实测数据的有效区间（与 MATLAB 图一致）
        x_lo, x_hi = np.min(refP), np.max(refP)
        y_lo, y_hi = np.min(gain_act), np.max(gain_act)
        xpad = 0.08 * (x_hi - x_lo)
        ypad = 0.15 * (y_hi - y_lo)
        ax.set_xlim(x_lo - xpad, x_hi + xpad)
        ax.set_ylim(y_lo - ypad, y_hi + ypad)

        ax.set_xlabel("Input Power (dBm)", fontsize=10)
        ax.set_ylabel("Power Gain (dB)", fontsize=10)
        ax.set_title(
            f"Comparison of Actual and Estimated Gain  "
            f"（{short} 模型, 时域 RMS 误差 = {rmsErr:.3f}%）",
            fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD",
                  loc="upper right")
        ax.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)

        self.canvas.draw()

    # ──────────────────────────────────────────────────────
    #  导出
    # ──────────────────────────────────────────────────────
    def _export(self):
        if self.coef_mat is None:
            QMessageBox.information(self, "提示", "请先完成模型拟合")
            return
        path, sel = QFileDialog.getSaveFileName(
            self, "导出拟合系数与参数", "fitCoefMatMem.csv",
            "CSV 文件 (*.csv);;NumPy 文件 (*.npz);;文本文件 (*.txt)")
        if not path:
            return
        try:
            modType = self.last_modType
            short = "MP" if modType == "memPoly" else "CM"
            memLen = max(1, _i(self.e_memlen.text(), 5))
            degLen = max(1, _i(self.e_deglen.text(), 5))
            C = self.coef_mat

            if path.endswith(".npz"):
                np.savez(path, fitCoefMatMem=C, modType=modType,
                         memLen=memLen, degLen=degLen)
            else:
                # CSV / TXT：写参数头 + 复数系数（实部+虚部）
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# 功放表征拟合系数导出\n")
                    f.write(f"# 模型类型, {short} ({modType})\n")
                    f.write(f"# memLen, {memLen}\n")
                    f.write(f"# degLen, {degLen}\n")
                    f.write(f"# 系数矩阵维度, {C.shape[0]} x {C.shape[1]}\n")
                    f.write(f"# 数据文件, {self.e_file.text()}\n")
                    f.write("# 格式: 每个系数为 real+imagj\n")
                    for row in C:
                        f.write(",".join(f"{v.real:.8e}{v.imag:+.8e}j"
                                          for v in row) + "\n")
            QMessageBox.information(self, "导出成功", f"已导出：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "PA_characterization.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")