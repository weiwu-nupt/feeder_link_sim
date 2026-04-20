"""
误码率分析模块 — BER Analysis
调制方式: PSK (QPSK, 8PSK), APSK (8/16/32/64/128/256APSK), QAM (16/64/256QAM)
信道编码: DVB-S2/S2X  LDPC+BCH 级联码
码长: 16200 / 32400 / 64800 bit
码率: 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10

理论BER曲线 = 未编码BER（解析式） + 编码增益估算
QEF门限参考 DVB-S2 EN 302307 Table 13
"""

import math
import numpy as np
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QSplitter, QFileDialog, QMessageBox, QSizePolicy,
    QCheckBox, QScrollArea, QFrame,
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

Q = special.erfc   # Q(x) = erfc(x/sqrt(2)) / 2,  erfc(x) = 2*Q(x*sqrt(2))
def qfunc(x):
    return 0.5 * special.erfc(x / math.sqrt(2))
def qfunc_arr(x):
    return 0.5 * special.erfc(x / np.sqrt(2))


# ══════════════════════════════════════════════════════════
#  调制配置
# ══════════════════════════════════════════════════════════

MOD_GROUPS = {
    "PSK":  ["QPSK", "8PSK"],
    "APSK": ["8APSK", "16APSK", "32APSK", "64APSK", "128APSK", "256APSK"],
    "QAM":  ["16QAM", "64QAM", "256QAM"],
}

MOD_BITS = {
    "QPSK":2, "8PSK":3,
    "8APSK":3,"16APSK":4,"32APSK":5,"64APSK":6,"128APSK":7,"256APSK":8,
    "16QAM":4,"64QAM":6,"256QAM":8,
}

# ══════════════════════════════════════════════════════════
#  DVB-S2/S2X 码率定义
# ══════════════════════════════════════════════════════════

CODE_RATES_ALL = [
    ("1/4", 0.25), ("1/3", 1/3), ("2/5", 0.4),
    ("1/2", 0.5),  ("3/5", 0.6), ("2/3", 2/3),
    ("3/4", 0.75), ("4/5", 0.8), ("5/6", 5/6),
    ("8/9", 8/9),  ("9/10",0.9),
]

FECFRAME_SIZES = [16200, 32400, 64800]  # bit

# ══════════════════════════════════════════════════════════
#  DVB-S2 EN302307 Table 13 QEF 门限（Es/No dB，AWGN）
#  来源: ETSI EN 302 307-1 Table 13，BER=1e-7
#  格式: {(调制, 码率字符串): Es/No dB}
# ══════════════════════════════════════════════════════════

QEF_THRESHOLD = {
    # QPSK
    ("QPSK", "1/4"): -2.35, ("QPSK", "1/3"): -1.24, ("QPSK", "2/5"): -0.30,
    ("QPSK", "1/2"):  1.00, ("QPSK", "3/5"):  2.23, ("QPSK", "2/3"):  3.10,
    ("QPSK", "3/4"):  4.03, ("QPSK", "4/5"):  4.68, ("QPSK", "5/6"):  5.18,
    ("QPSK", "8/9"):  6.20, ("QPSK", "9/10"): 6.42,
    # 8PSK
    ("8PSK", "3/5"):  5.50, ("8PSK", "2/3"):  6.62, ("8PSK", "3/4"):  7.91,
    ("8PSK", "5/6"):  9.35, ("8PSK", "8/9"): 10.69, ("8PSK", "9/10"):10.98,
    # 16APSK
    ("16APSK","2/3"): 8.97,("16APSK","3/4"):10.21,("16APSK","4/5"):11.03,
    ("16APSK","5/6"):11.61,("16APSK","8/9"):12.89,("16APSK","9/10"):13.13,
    # 32APSK
    ("32APSK","3/4"):12.73,("32APSK","4/5"):13.64,("32APSK","5/6"):14.28,
    ("32APSK","8/9"):15.69,("32APSK","9/10"):16.05,
    # 64APSK (DVB-S2X，估算值)
    ("64APSK","2/3"):14.0,("64APSK","3/4"):15.2,("64APSK","4/5"):16.1,
    ("64APSK","5/6"):16.8,("64APSK","8/9"):18.2,("64APSK","9/10"):18.7,
    # 128APSK (DVB-S2X)
    ("128APSK","3/4"):17.5,("128APSK","5/6"):19.0,("128APSK","8/9"):20.5,
    ("128APSK","9/10"):21.2,
    # 256APSK (DVB-S2X)
    ("256APSK","3/4"):20.0,("256APSK","5/6"):21.5,("256APSK","8/9"):23.2,
    ("256APSK","9/10"):24.0,
    # QAM (AWGN，估算值)
    ("16QAM","1/2"):4.0,("16QAM","2/3"):6.5,("16QAM","3/4"):7.5,
    ("16QAM","5/6"):8.5,("16QAM","8/9"):9.5,("16QAM","9/10"):10.0,
    ("64QAM","1/2"):9.5,("64QAM","2/3"):11.5,("64QAM","3/4"):12.5,
    ("64QAM","5/6"):13.5,("64QAM","8/9"):14.5,("64QAM","9/10"):15.0,
    ("256QAM","1/2"):15.0,("256QAM","2/3"):17.0,("256QAM","3/4"):18.0,
    ("256QAM","5/6"):19.0,("256QAM","8/9"):20.0,("256QAM","9/10"):20.5,
}


# ══════════════════════════════════════════════════════════
#  未编码 BER 解析公式（在 AWGN 信道下）
#  x 轴: Es/No (dB)
# ══════════════════════════════════════════════════════════

def _esno_db_to_lin(esno_db):
    return 10 ** (esno_db / 10.0)

def ber_qpsk(esno_db):
    """QPSK uncoded BER = Q(sqrt(2*Es/No))"""
    esno = _esno_db_to_lin(esno_db)
    return qfunc_arr(np.sqrt(2 * esno))

def ber_mpsk(esno_db, M):
    """M-PSK 近似BER（Gray编码）≈ erfc(sqrt(Es/No * sin(pi/M)^2 * log2(M))) / log2(M)"""
    m = np.log2(M)
    esno = _esno_db_to_lin(esno_db)
    return (1/m) * special.erfc(np.sqrt(esno * (np.sin(np.pi/M)**2) * m))

def ber_apsk(esno_db, M):
    """
    APSK BER 近似（基于等效 AWGN SNR + 星座点间距）
    使用 APSK 的 SER 上界近似，再转换为 BER
    DVB-S2 APSK 采用同心圆结构，近似为等效 MPSK
    """
    m = np.log2(M)
    esno = _esno_db_to_lin(esno_db)
    # APSK 近似：外圆的点间距比 MPSK 大，用修正因子
    # 参考：APSK 编码增益约比等阶 QAM 好 0.5-1dB（卫星非线性信道）
    if M <= 8:
        factor = 0.85   # 8APSK 约等于 8PSK
        return (1/m) * special.erfc(np.sqrt(esno * factor * np.sin(np.pi/M)**2 * m))
    elif M == 16:
        # 16APSK(4+12): 内圆4点 + 外圆12点
        r1 = 1.0; r2 = 2.57   # 典型码率3/4的环比
        # SER 近似
        ser = (4/M) * qfunc_arr(np.sqrt(esno) * r1 * np.sqrt(2/(r1**2 + r2**2)) * np.sin(np.pi/4)) + \
              (12/M) * qfunc_arr(np.sqrt(esno) * r2 * np.sqrt(2/(r1**2 + r2**2)) * np.sin(np.pi/12))
        return np.clip(ser / m, 0, 0.5)
    else:
        # 高阶 APSK 用 Gray 编码 MQAM 近似
        return ber_mqam(esno_db, M)

def ber_mqam(esno_db, M):
    """M-QAM BER（Gray编码，矩形星座）
    BER ≈ (4/log2(M)) * (1 - 1/sqrt(M)) * Q(sqrt(3*log2(M)*Es/No / (M-1)))
    """
    m = np.log2(M)
    esno = _esno_db_to_lin(esno_db)
    return (4/m) * (1 - 1/np.sqrt(M)) * qfunc_arr(np.sqrt(3 * m * esno / (M - 1)))

def uncoded_ber(mod: str, esno_db_arr: np.ndarray) -> np.ndarray:
    """根据调制方式计算未编码 BER"""
    if mod == "QPSK":
        return ber_qpsk(esno_db_arr)
    elif mod == "8PSK":
        return ber_mpsk(esno_db_arr, 8)
    elif mod.endswith("APSK"):
        M = int(mod.replace("APSK",""))
        return ber_apsk(esno_db_arr, M)
    elif mod.endswith("QAM"):
        M = int(mod.replace("QAM",""))
        return ber_mqam(esno_db_arr, M)
    return np.full_like(esno_db_arr, 0.5)


# ══════════════════════════════════════════════════════════
#  编码 BER 曲线（基于 QEF 门限 + 瀑布曲线近似）
#
#  DVB-S2 LDPC+BCH 编码后的 BER 曲线形状：
#  - 在 QEF 门限以下：BER ≈ 1 - 缓慢改善（误差平台）
#  - 在 QEF 门限附近：陡峭瀑布（waterfall）下降
#  - 在 QEF 门限以上：BER < 1e-7（QEF）
#
#  近似方法：
#  1. 瀑布段用 Sigmoid 函数近似（斜率由码长决定）
#  2. 误差平台区域（error floor）：高码率较低，低码率较高
# ══════════════════════════════════════════════════════════

def coded_ber(mod: str, code_rate_str: str, fecframe: int,
              esno_db_arr: np.ndarray) -> np.ndarray:
    """
    近似编码后BER曲线。

    瀑布段近似：BER ≈ BER_floor + (0.5 - BER_floor) / (1 + exp(k*(Es/No - Es/No_qef)))
    k 越大曲线越陡（码长越长越陡），码长64800比16200约陡1.5倍
    """
    key = (mod, code_rate_str)
    if key not in QEF_THRESHOLD:
        return np.full_like(esno_db_arr, np.nan)

    esno_qef = QEF_THRESHOLD[key]

    # 码长决定瀑布斜率（估算值）
    slope_map = {16200: 2.5, 32400: 3.2, 64800: 4.0}
    k = slope_map.get(fecframe, 3.0)

    # 误差平台（error floor）：极高 Es/No 下的剩余 BER
    # 实际测量约 1e-8 到 1e-10，这里取 1e-8 做保守估计
    ber_floor = 1e-8

    # BER_uncoded（用于瀑布顶部水平）
    ber_top = uncoded_ber(mod, np.array([esno_qef - 3.0]))[0]
    ber_top = min(ber_top, 0.3)

    # Sigmoid 瀑布
    ber = np.zeros_like(esno_db_arr)
    for i, esno in enumerate(esno_db_arr):
        if esno < esno_qef - 5:
            # 远低于门限：近似为未编码BER（稍有改善）
            ber[i] = min(uncoded_ber(mod, np.array([esno]))[0], 0.5)
        elif esno > esno_qef + 3:
            # 远高于门限：QEF（取 error floor）
            ber[i] = ber_floor
        else:
            # 瀑布段：Sigmoid
            sigmoid = 1.0 / (1.0 + math.exp(k * (esno - esno_qef)))
            ber[i] = max(ber_floor, ber_top * sigmoid + ber_floor * (1 - sigmoid))

    return ber


# ══════════════════════════════════════════════════════════
#  有效信息 BER（信息比特误码率）
#  BER_info = BER_coded / R   （R = 码率）
# ══════════════════════════════════════════════════════════

def info_ber(mod, code_rate_str, fecframe, esno_db_arr):
    rate = dict(CODE_RATES_ALL).get(code_rate_str, 0.5)
    ber = coded_ber(mod, code_rate_str, fecframe, esno_db_arr)
    return ber  # 已是比特误码率，不需要除以码率


# ══════════════════════════════════════════════════════════
#  UI 辅助
# ══════════════════════════════════════════════════════════

_ES = ("QLineEdit{background:#FFF;border:1px solid #D0D0D0;"
       "border-radius:3px;padding:3px 6px;font-size:10pt;color:#111;}"
       "QLineEdit:focus{border:1.5px solid #1D9E75;}")
_LS = "font-size:10pt;color:#2C2C2A;"
_GB = ("QGroupBox{background:#FFF;border:1px solid #E0E0E0;"
       "border-radius:6px;margin-top:8px;padding:6px 8px;}"
       "QGroupBox::title{subcontrol-origin:margin;left:10px;"
       "padding:0 4px;color:#1D9E75;font-size:9pt;font-weight:bold;}")

def _group(t):
    gb = QGroupBox(t); gb.setStyleSheet(_GB)
    vl = QVBoxLayout(gb); vl.setSpacing(4); vl.setContentsMargins(6,4,6,6)
    return gb

_COLORS = [
    "#0055CC","#CC2200","#1D9E75","#BA7517","#7B3FA0",
    "#00838F","#C62828","#37474F","#2E7D32","#6A1B9A",
    "#E65100","#0277BD","#558B2F","#4527A0","#AD1457",
]


# ══════════════════════════════════════════════════════════
#  画布
# ══════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 5.5), dpi=96)
        self.fig.patch.set_facecolor("#F8F8F8")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════
#  对话框
# ══════════════════════════════════════════════════════════

class BERAnalysisDialog(ModuleDialog):
    TITLE        = "误码率分析"
    ACCENT_COLOR = "#1D9E75"
    MIN_WIDTH    = 1050
    MIN_HEIGHT   = 680

    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1)
        sp.setStyleSheet("QSplitter::handle{background:#DDDDDD;}")

        # ══ 左侧配置 ══════════════════════════════════════
        left = QWidget()
        left.setMinimumWidth(280); left.setMaximumWidth(350)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0,0,8,0); lv.setSpacing(7)

        # ── SNR 范围 ──────────────────────────────────────
        sg = _group("信噪比范围 (Es/No)")
        sf = QFormLayout(); sf.setSpacing(5); sf.setContentsMargins(0,0,0,0)
        sg.layout().addLayout(sf)

        self._snr_min = self._lineedit("-5")
        self._snr_max = self._lineedit("25")
        self._snr_pts = self._lineedit("200")
        sf.addRow(self._label("起始 (dB):"), self._snr_min)
        sf.addRow(self._label("终止 (dB):"), self._snr_max)
        sf.addRow(self._label("点数:"),      self._snr_pts)
        lv.addWidget(sg)

        # ── 调制方式 ──────────────────────────────────────
        mg = _group("调制方式")
        mf = QFormLayout(); mf.setSpacing(5); mf.setContentsMargins(0,0,0,0)
        mg.layout().addLayout(mf)

        self._mod_type = QComboBox()
        self._mod_type.setStyleSheet(self._cb_style())
        self._mod_type.addItems(["PSK", "APSK", "QAM"])
        self._mod_type.currentTextChanged.connect(self._on_mod_type_changed)
        mf.addRow(self._label("类型:"), self._mod_type)

        self._mod_order = QComboBox()
        self._mod_order.setStyleSheet(self._cb_style())
        mf.addRow(self._label("阶数:"), self._mod_order)

        # 未编码 BER 开关
        self._uncoded_cb = QCheckBox("显示未编码 BER")
        self._uncoded_cb.setStyleSheet("font-size:10pt;color:#444;")
        self._uncoded_cb.setChecked(True)
        mg.layout().addWidget(self._uncoded_cb)
        lv.addWidget(mg)

        # ── 信道编码 ──────────────────────────────────────
        cg = _group("信道编码（LDPC+BCH）")
        cg_note = QLabel(
            "DVB-S2/S2X 级联码\n"
            "外码: BCH   内码: LDPC\n"
            "可选多条曲线同时显示")
        cg_note.setStyleSheet("font-size:8pt;color:#666;")
        cg_note.setWordWrap(True)
        cg.layout().addWidget(cg_note)

        # 码长选择
        cf = QFormLayout(); cf.setSpacing(5); cf.setContentsMargins(0,0,0,0)
        cg.layout().addLayout(cf)
        self._fecframe = QComboBox()
        self._fecframe.setStyleSheet(self._cb_style())
        self._fecframe.addItems(["64800 (Normal)", "32400 (Medium)", "16200 (Short)"])
        cf.addRow(self._label("码长 (bit):"), self._fecframe)
        lv.addWidget(cg)

        # 码率多选
        rg = _group("码率（可多选）")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(180)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner = QWidget(); inner_lv = QVBoxLayout(inner)
        inner_lv.setSpacing(2); inner_lv.setContentsMargins(2,2,2,2)
        self._rate_cbs = {}
        for name, val in CODE_RATES_ALL:
            cb = QCheckBox(name)
            cb.setStyleSheet("font-size:10pt;color:#333;")
            # 默认勾选常用码率
            cb.setChecked(name in ("1/2","2/3","3/4","5/6","8/9"))
            self._rate_cbs[name] = cb
            inner_lv.addWidget(cb)
        inner_lv.addStretch()
        scroll.setWidget(inner)
        rg.layout().addWidget(scroll)

        # 全选/全不选
        sel_hl = QHBoxLayout(); sel_hl.setSpacing(6)
        btn_all = QPushButton("全选"); btn_all.setFixedHeight(24)
        btn_all.setStyleSheet("QPushButton{font-size:9pt;padding:0 8px;"
                               "border:1px solid #CCC;border-radius:3px;background:#F5F5F5;}"
                               "QPushButton:hover{background:#E8E8E8;}")
        btn_none = QPushButton("全不选"); btn_none.setFixedHeight(24)
        btn_none.setStyleSheet(btn_all.styleSheet())
        btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self._rate_cbs.values()])
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self._rate_cbs.values()])
        sel_hl.addWidget(btn_all); sel_hl.addWidget(btn_none); sel_hl.addStretch()
        rg.layout().addLayout(sel_hl)
        lv.addWidget(rg)

        # ── 按钮 ──────────────────────────────────────────
        bhl = QHBoxLayout(); bhl.setSpacing(8); bhl.setContentsMargins(0,4,0,0)
        self.btn_run = QPushButton("绘制 BER 曲线")
        self.btn_run.setFixedHeight(32)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#1D9E75;color:#FFF;border:none;"
            "border-radius:5px;font-size:10pt;font-weight:bold;}"
            "QPushButton:hover{background:#14705A;}")
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
        self.status = QLabel("就绪 — 选择参数后点击「绘制 BER 曲线」")
        self.status.setStyleSheet("font-size:9pt;color:#888;")
        rv.addWidget(self.status)
        sp.addWidget(right)
        sp.setStretchFactor(0,0); sp.setStretchFactor(1,1)
        layout.addWidget(sp, stretch=1)

        # 初始化调制阶数
        self._on_mod_type_changed("PSK")

    # ── 调制类型切换 ──────────────────────────────────────

    def _on_mod_type_changed(self, t):
        self._mod_order.clear()
        self._mod_order.addItems(MOD_GROUPS.get(t, []))

    # ── 计算并绘图 ────────────────────────────────────────

    def _run(self):
        try:    self._do_run()
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "错误", str(e))

    def _do_run(self):
        try:    snr_min = float(self._snr_min.text())
        except: snr_min = -5
        try:    snr_max = float(self._snr_max.text())
        except: snr_max = 25
        try:    n_pts = max(50, int(float(self._snr_pts.text())))
        except: n_pts = 200

        mod = self._mod_order.currentText()
        if not mod:
            QMessageBox.information(self, "提示", "请选择调制方式"); return

        fecframe_txt = self._fecframe.currentText()
        fecframe = int(fecframe_txt.split()[0])

        selected_rates = [name for name, cb in self._rate_cbs.items() if cb.isChecked()]
        show_uncoded   = self._uncoded_cb.isChecked()

        if not selected_rates and not show_uncoded:
            QMessageBox.information(self, "提示", "请至少选择一个码率或勾选未编码BER"); return

        esno_arr = np.linspace(snr_min, snr_max, n_pts)
        self._plot(mod, fecframe, esno_arr, selected_rates, show_uncoded)

    def _plot(self, mod, fecframe, esno_arr, selected_rates, show_uncoded):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")
        fig.patch.set_facecolor("#F8F8F8")

        color_idx = 0
        plotted = 0

        # 未编码 BER
        if show_uncoded:
            ber_u = uncoded_ber(mod, esno_arr)
            valid = ber_u > 1e-12
            if valid.any():
                ax.semilogy(esno_arr[valid], ber_u[valid],
                            color="#999999", lw=1.5, ls="--",
                            label=f"{mod} 未编码", zorder=2)
                plotted += 1

        # 编码 BER（每个选中码率一条曲线）
        for rate_str in selected_rates:
            clr = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1
            ber_c = coded_ber(mod, rate_str, fecframe, esno_arr)
            valid = ~np.isnan(ber_c) & (ber_c > 1e-12)
            if not valid.any():
                continue
            ax.semilogy(esno_arr[valid], ber_c[valid],
                        color=clr, lw=2.0,
                        label=f"{mod} R={rate_str}", zorder=3)
            # QEF 门限标注
            key = (mod, rate_str)
            if key in QEF_THRESHOLD:
                qef = QEF_THRESHOLD[key]
                if snr_arr_has(esno_arr, qef):
                    ax.axvline(qef, color=clr, lw=0.7, ls=":", alpha=0.6)
            plotted += 1

        if plotted == 0:
            self.status.setText("无可用曲线（该调制方式未定义此码率的门限）"); return

        # QEF 参考线
        ax.axhline(1e-7, color="#CC2200", lw=0.8, ls="-.",
                   alpha=0.7, label="QEF (BER=10⁻⁷)", zorder=1)
        ax.axhline(1e-4, color="#FF6600", lw=0.6, ls=":",
                   alpha=0.5, label="BER=10⁻⁴", zorder=1)

        # 轴设置
        ax.set_xlabel("Es/No  (dB)", fontsize=10)
        ax.set_ylabel("BER", fontsize=10)
        fsize = {64800:"Normal 64800", 32400:"Medium 32400", 16200:"Short 16200"}.get(fecframe,"")
        ax.set_title(f"{mod}  DVB-S2/S2X  LDPC+BCH  FECFRAME {fsize} bit",
                     fontsize=10, pad=6)
        ax.set_xlim(esno_arr[0], esno_arr[-1])
        ax.set_ylim(1e-9, 1.0)
        ax.grid(True, which='both', color="#E8E8E8", lw=0.5)
        ax.grid(True, which='major', color="#D0D0D0", lw=0.8)
        for sp in ax.spines.values(): sp.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)

        ax.legend(fontsize=8, framealpha=0.92, edgecolor="#DDDDDD",
                  loc="upper right", ncol=2 if len(selected_rates)>4 else 1)

        self.canvas.draw()
        self.status.setText(
            f"{mod}  FECFRAME={fecframe}bit  码率={len(selected_rates)}条  "
            f"Es/No={esno_arr[0]:.0f}~{esno_arr[-1]:.0f}dB")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "BER_Analysis.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")

    # ── 辅助 ──────────────────────────────────────────────
    def _label(self, t):
        l = QLabel(t); l.setStyleSheet(_LS); return l

    def _lineedit(self, default, w=80):
        from PyQt6.QtWidgets import QLineEdit
        e = QLineEdit(default); e.setFixedWidth(w); e.setStyleSheet(_ES)
        return e

    @staticmethod
    def _cb_style():
        return ("QComboBox{background:#FFF;border:1px solid #D0D0D0;"
                "border-radius:3px;padding:3px 8px;font-size:10pt;color:#111;}"
                "QComboBox QAbstractItemView{background:#FFF;font-size:10pt;"
                "color:#111;selection-background-color:#E1F5EE;}")


def snr_arr_has(arr, val):
    return arr[0] <= val <= arr[-1]