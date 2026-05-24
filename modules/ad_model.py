"""
AD/DA 转换器建模模块 — Data Converter Behavioral Model
======================================================
对馈电链路收发端的 ADC / DAC 做信号处理层面的行为建模：

  AD（接收端）  连续信号 → 抗混叠滤波 → 采样(含孔径抖动) → 量化
  DA（发射端）  数字码字 → 量化 → 零阶保持(ZOH) → sinc droop → 重建滤波

核心建模效应
  1. 均匀量化      量化噪声功率 = LSB²/12，SQNR ≈ 6.02N + 1.76 dB
  2. 孔径抖动      jitter 导致 SNR 上限 = -20·log10(2π·f·σ_t)
  3. 混叠          欠采样时频谱副本折回 Nyquist 区
  4. 零阶保持      DAC 输出频响 H(f) = sinc(f/fs)，引入通带 droop
  5. 热噪声        输入端等效热噪声，可设定 SNR 上限

指标
  SNR / SINAD / ENOB / SFDR / THD —— 均由单次加窗 FFT 提取

参考
  - W. Kester, "Analog-Digital Conversion", Analog Devices
  - IEEE Std 1241-2010  ADC 测试标准术语与方法
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QSizePolicy,
    QGridLayout,
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
#  核心算法
# ══════════════════════════════════════════════════════════

def quantize(x, n_bits, vfs, mode="mid-rise"):
    """
    理想均匀量化器。

    参数
    ----
    x      : 输入信号（V），自动限幅到满量程
    n_bits : 量化位数
    vfs    : 满量程峰峰值 V_FS（V），量程为 [-vfs/2, +vfs/2]
    mode   : 'mid-rise'（无零电平）或 'mid-tread'（有零电平）

    返回 (xq, lsb)
    """
    half = vfs / 2.0
    lsb = vfs / (2 ** n_bits)
    if mode == "mid-rise":
        xc = np.clip(x, -half, half - lsb)
        q = (np.floor(xc / lsb) + 0.5) * lsb
    else:
        xc = np.clip(x, -half, half - lsb)
        q = np.round(xc / lsb) * lsb
    return q, lsb


def spectrum(x, fs, window="hann"):
    """加窗单边功率谱。返回 (freq_Hz, psd)。"""
    n = len(x)
    if window == "hann":
        w = np.hanning(n)
    elif window == "blackman":
        w = np.blackman(n)
    elif window == "flattop":
        # 5 项 flat-top，幅度精度高
        a = [0.21557895, 0.41663158, 0.277263158,
             0.083578947, 0.006947368]
        k = np.arange(n)
        w = sum((-1) ** i * a[i] * np.cos(2 * np.pi * i * k / (n - 1))
                for i in range(5))
    else:
        w = np.ones(n)
    cg = np.sum(w) / n
    X = np.fft.rfft(x * w) / (n * cg)
    X[1:] *= 2.0
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    return freq, np.abs(X) ** 2


def extract_metrics(freq, psd, fs, f_sig, n_harm=6, leak_bins=4):
    """
    由单边功率谱提取动态指标。

    返回 dict: SNR / SINAD / THD / ENOB / SFDR （均为 dB，ENOB 为 bit）
    """
    n = len(psd)
    df = freq[1] - freq[0]
    sig_bin = int(round(f_sig / df))
    fnyq = fs / 2.0

    def band(c):
        return slice(max(0, c - leak_bins), min(n, c + leak_bins + 1))

    total = np.sum(psd[1:])                       # 排除 DC
    p_sig = np.sum(psd[band(sig_bin)])

    p_harm = 0.0
    for h in range(2, n_harm + 1):
        fh = (h * f_sig) % fs
        if fh > fnyq:
            fh = fs - fh                          # 混叠折回
        hb = int(round(fh / df))
        if 0 < hb < n and abs(hb - sig_bin) > leak_bins:
            p_harm += np.sum(psd[band(hb)])

    p_noise = max(total - p_sig - p_harm, 1e-30)
    p_nad = max(total - p_sig, 1e-30)

    snr = 10 * np.log10(p_sig / p_noise)
    sinad = 10 * np.log10(p_sig / p_nad)
    thd = 10 * np.log10(max(p_harm, 1e-30) / p_sig)
    enob = (sinad - 1.76) / 6.02

    psd2 = psd.copy()
    psd2[band(sig_bin)] = 0
    psd2[0] = 0
    sfdr = 10 * np.log10(np.max(psd[band(sig_bin)]) / max(np.max(psd2), 1e-30))

    return dict(SNR=snr, SINAD=sinad, THD=thd, ENOB=enob, SFDR=sfdr)


def run_adc_chain(cfg):
    """
    执行 ADC 链路仿真。cfg 为参数 dict，返回结果 dict。

    关键步骤：连续信号 → 孔径抖动采样 → 热噪声 → 量化 → FFT → 指标
    """
    fs = cfg["fs"]
    N = cfg["n_samples"]
    n_bits = cfg["n_bits"]
    vfs = cfg["vfs"]
    f_in = cfg["f_sig"]
    dbfs = cfg["amp_dbfs"]
    jitter = cfg["jitter"]
    thermal_snr = cfg["thermal_snr"]
    window = cfg["window"]
    coherent = cfg["coherent"]

    # 相干采样：把 f_sig 微调到最近的 M/N·fs（M 与 N 互质）
    if coherent:
        M = max(1, int(round(f_in / fs * N)))
        while np.gcd(M, N) != 1:
            M += 1
        f_sig = M / N * fs
    else:
        f_sig = f_in

    amp = (vfs / 2.0) * (10 ** (dbfs / 20.0))
    n = np.arange(N)
    t_ideal = n / fs

    # 孔径抖动：采样时刻扰动
    rng = np.random.RandomState(2024)
    if jitter > 0:
        t_samp = t_ideal + rng.normal(0.0, jitter, size=N)
    else:
        t_samp = t_ideal
    x = amp * np.sin(2 * np.pi * f_sig * t_samp)

    # 输入端等效热噪声
    if np.isfinite(thermal_snr):
        p_sig = amp ** 2 / 2.0
        p_noise = p_sig / (10 ** (thermal_snr / 10.0))
        x = x + rng.normal(0.0, np.sqrt(p_noise), size=N)

    # 量化
    xq, lsb = quantize(x, n_bits, vfs, cfg["q_mode"])

    # 频谱与指标
    freq, psd = spectrum(xq, fs, window)
    m = extract_metrics(freq, psd, fs, f_sig)

    return dict(
        t=t_ideal, x_analog=amp * np.sin(2 * np.pi * f_sig * t_ideal),
        x_quant=xq, lsb=lsb, freq=freq, psd=psd,
        f_sig=f_sig, fs=fs, n_bits=n_bits, metrics=m,
        sqnr_ideal=6.02 * n_bits + 1.76,
        jitter_snr=(-20 * np.log10(2 * np.pi * f_sig * jitter)
                    if jitter > 0 else float("inf")),
    )


def run_dac_chain(cfg):
    """
    执行 DAC 链路仿真：量化码字 → 零阶保持 → sinc droop → 频谱。
    重点演示 ZOH 的 sinc 幅度衰减。
    """
    fs = cfg["fs"]
    N = cfg["n_samples"]
    n_bits = cfg["n_bits"]
    vfs = cfg["vfs"]
    dbfs = cfg["amp_dbfs"]
    osr = cfg["osr"]                         # 过采样观察倍数

    M = max(1, int(round(cfg["f_sig"] / fs * N)))
    while np.gcd(M, N) != 1:
        M += 1
    f_sig = M / N * fs
    amp = (vfs / 2.0) * (10 ** (dbfs / 20.0))

    n = np.arange(N)
    x = amp * np.sin(2 * np.pi * f_sig * n / fs)
    xq, lsb = quantize(x, n_bits, vfs, cfg["q_mode"])

    # 零阶保持：每个样本重复 osr 次，得到阶梯波
    x_zoh = np.repeat(xq, osr)
    fs_hi = fs * osr
    freq, psd = spectrum(x_zoh, fs_hi, cfg["window"])

    # 理论 ZOH 频响 H(f) = |sinc(f/fs)|
    sinc_resp = np.abs(np.sinc(freq / fs))

    # 通带边缘（0.4·fs 处）的 droop
    f_edge = 0.4 * fs
    droop_db = 20 * np.log10(np.sinc(f_edge / fs))

    return dict(
        x_quant=xq, x_zoh=x_zoh, lsb=lsb, freq=freq, psd=psd,
        sinc_resp=sinc_resp, f_sig=f_sig, fs=fs, fs_hi=fs_hi,
        osr=osr, droop_db=droop_db, n_bits=n_bits,
    )


def snr_vs_bits(cfg, bit_range):
    """扫描量化位数，返回 (bits, sinad_list, enob_list, ideal_list)。"""
    bits, sinad_l, enob_l, ideal_l = [], [], [], []
    for nb in bit_range:
        c = dict(cfg)
        c["n_bits"] = nb
        r = run_adc_chain(c)
        bits.append(nb)
        sinad_l.append(r["metrics"]["SINAD"])
        enob_l.append(r["metrics"]["ENOB"])
        ideal_l.append(6.02 * nb + 1.76)
    return bits, sinad_l, enob_l, ideal_l


# ══════════════════════════════════════════════════════════
#  UI 样式（与 pa_mode / ber_analysis 对齐）
# ══════════════════════════════════════════════════════════

_ACCENT = "#BA7517"
_ES = ("QLineEdit{background:#FFF;border:1px solid #D0D0D0;"
       "border-radius:3px;padding:3px 6px;font-size:10pt;color:#111;}"
       f"QLineEdit:focus{{border:1.5px solid {_ACCENT};}}")
_LS = "font-size:10pt;color:#2C2C2A;"
_GB = ("QGroupBox{background:#FFF;border:1px solid #E0E0E0;"
       "border-radius:6px;margin-top:8px;padding:6px 8px;}"
       "QGroupBox::title{subcontrol-origin:margin;left:10px;"
       f"padding:0 4px;color:{_ACCENT};font-size:9pt;font-weight:bold;}}")


def _group(title):
    gb = QGroupBox(title)
    gb.setStyleSheet(_GB)
    vl = QVBoxLayout(gb)
    vl.setSpacing(4)
    vl.setContentsMargins(6, 4, 6, 6)
    return gb


def _cb_style():
    return ("QComboBox{background:#FFF;border:1px solid #D0D0D0;"
            "border-radius:3px;padding:3px 8px;font-size:10pt;color:#111;}"
            "QComboBox QAbstractItemView{background:#FFF;color:#111;"
            "selection-background-color:#FAEEDA;font-size:10pt;}")


# ══════════════════════════════════════════════════════════
#  画布
# ══════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6.2), dpi=96)
        self.fig.patch.set_facecolor("#F8F8F8")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════
#  对话框
# ══════════════════════════════════════════════════════════

class ADDAModelDialog(ModuleDialog):
    TITLE        = "AD/DA 模型"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH    = 1060
    MIN_HEIGHT   = 720

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._last_result = None

    # ──────────────────────────────────────────────────────
    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1)
        sp.setStyleSheet("QSplitter::handle{background:#DDDDDD;}")

        # ══ 左侧配置 ══════════════════════════════════════
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(360)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(7)

        # ── ① 转换器类型 ──────────────────────────────────
        tg = _group("① 转换器")
        tf = QFormLayout()
        tf.setSpacing(5)
        tf.setContentsMargins(0, 0, 0, 0)
        self.cb_type = QComboBox()
        self.cb_type.setStyleSheet(_cb_style())
        self.cb_type.addItems(["ADC（接收端 模数转换）",
                               "DAC（发射端 数模转换）"])
        self.cb_type.currentIndexChanged.connect(self._on_type_changed)
        tf.addRow(self._lbl("类型:"), self.cb_type)
        tg.layout().addLayout(tf)
        lv.addWidget(tg)

        # ── ② 信号参数 ────────────────────────────────────
        sg = _group("② 输入信号")
        sf = QFormLayout()
        sf.setSpacing(5)
        sf.setContentsMargins(0, 0, 0, 0)
        self.e_fsig = self._edit("14.6")
        self.e_dbfs = self._edit("-1.0")
        self.e_nsamp = self._edit("8192")
        sf.addRow(self._lbl("信号频率 (MHz):"), self.e_fsig)
        sf.addRow(self._lbl("幅度 (dBFS):"), self.e_dbfs)
        sf.addRow(self._lbl("采样点数 N:"), self.e_nsamp)
        self.chk_coherent = QCheckBox("相干采样（自动微调信号频率）")
        self.chk_coherent.setChecked(True)
        self.chk_coherent.setStyleSheet("font-size:9pt;color:#555;")
        sg.layout().addLayout(sf)
        sg.layout().addWidget(self.chk_coherent)
        lv.addWidget(sg)

        # ── ③ 转换器参数 ──────────────────────────────────
        cg = _group("③ 转换器参数")
        cf = QFormLayout()
        cf.setSpacing(5)
        cf.setContentsMargins(0, 0, 0, 0)
        self.e_fs = self._edit("100.0")
        self.e_bits = self._edit("12")
        self.e_vfs = self._edit("2.0")
        self.cb_qmode = QComboBox()
        self.cb_qmode.setStyleSheet(_cb_style())
        self.cb_qmode.addItems(["mid-rise", "mid-tread"])
        cf.addRow(self._lbl("采样率 fs (MHz):"), self.e_fs)
        cf.addRow(self._lbl("量化位数 N:"), self.e_bits)
        cf.addRow(self._lbl("满量程 V_FS (V):"), self.e_vfs)
        cf.addRow(self._lbl("量化方式:"), self.cb_qmode)
        cg.layout().addLayout(cf)
        lv.addWidget(cg)

        # ── ④ 非理想效应 ──────────────────────────────────
        ng = _group("④ 非理想效应")
        nf = QFormLayout()
        nf.setSpacing(5)
        nf.setContentsMargins(0, 0, 0, 0)
        self.e_jitter = self._edit("0.5")
        self.e_thermal = self._edit("90")
        self.e_osr = self._edit("8")
        nf.addRow(self._lbl("孔径抖动 σ (ps):"), self.e_jitter)
        nf.addRow(self._lbl("热噪声 SNR 上限 (dB):"), self.e_thermal)
        nf.addRow(self._lbl("ZOH 观察过采样:"), self.e_osr)
        ng.layout().addLayout(nf)
        hint = QLabel("抖动填 0、热噪声留空 = 理想转换器；\n"
                      "ZOH 过采样仅 DAC 模式生效")
        hint.setStyleSheet("font-size:8pt;color:#999;")
        hint.setWordWrap(True)
        ng.layout().addWidget(hint)
        lv.addWidget(ng)

        # ── ⑤ 视图 / 操作 ─────────────────────────────────
        vg = _group("⑤ 视图")
        vf = QFormLayout()
        vf.setSpacing(5)
        vf.setContentsMargins(0, 0, 0, 0)
        self.cb_view = QComboBox()
        self.cb_view.setStyleSheet(_cb_style())
        self.cb_view.addItems(["时域波形", "频谱 (FFT)",
                               "SNR vs 量化位数"])
        self.cb_window = QComboBox()
        self.cb_window.setStyleSheet(_cb_style())
        self.cb_window.addItems(["hann", "blackman", "flattop", "rect"])
        vf.addRow(self._lbl("显示:"), self.cb_view)
        vf.addRow(self._lbl("FFT 窗:"), self.cb_window)
        vg.layout().addLayout(vf)
        lv.addWidget(vg)

        self.btn_run = QPushButton("运行仿真")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{_ACCENT};color:#FFF;border:none;"
            "border-radius:5px;font-size:10pt;font-weight:bold;}"
            "QPushButton:hover{background:#8B5A0F;}")
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#FFF;color:#444;border:1px solid #CCC;"
            "border-radius:5px;font-size:9.5pt;}"
            "QPushButton:hover{background:#F5F5F5;}")
        self.btn_save.clicked.connect(self._save)
        lv.addWidget(self.btn_save)

        lv.addStretch()
        sp.addWidget(left)

        # ══ 右侧：画布 + 指标栏 ═══════════════════════════
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(4)

        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas, stretch=1)

        # 指标栏
        self.metric_box = QWidget()
        self.metric_box.setStyleSheet(
            "background:#FFF;border:1px solid #E0E0E0;border-radius:6px;")
        mg = QGridLayout(self.metric_box)
        mg.setContentsMargins(10, 6, 10, 6)
        mg.setSpacing(4)
        self._metric_labels = {}
        specs = [("SNR", "dB"), ("SINAD", "dB"), ("ENOB", "bit"),
                 ("SFDR", "dB"), ("THD", "dB")]
        for i, (name, unit) in enumerate(specs):
            cap = QLabel(name)
            cap.setStyleSheet("font-size:8.5pt;color:#999;")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("—")
            val.setStyleSheet(f"font-size:13pt;font-weight:bold;color:{_ACCENT};")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            uni = QLabel(unit)
            uni.setStyleSheet("font-size:8pt;color:#BBB;")
            uni.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mg.addWidget(cap, 0, i)
            mg.addWidget(val, 1, i)
            mg.addWidget(uni, 2, i)
            self._metric_labels[name] = val
        rv.addWidget(self.metric_box)

        self.status = QLabel("就绪 — 设置参数后点击「运行仿真」")
        self.status.setStyleSheet("font-size:9pt;color:#888;")
        rv.addWidget(self.status)

        sp.addWidget(right)
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        layout.addWidget(sp, stretch=1)

        self._on_type_changed(0)
        self._run()

    # ── 辅助 ──────────────────────────────────────────────
    def _lbl(self, t):
        l = QLabel(t)
        l.setStyleSheet(_LS)
        return l

    def _edit(self, default, w=110):
        e = QLineEdit(default)
        e.setFixedWidth(w)
        e.setStyleSheet(_ES)
        return e

    @staticmethod
    def _fv(text, default=0.0):
        try:
            return float(str(text).strip())
        except (ValueError, AttributeError):
            return default

    def _on_type_changed(self, idx):
        is_dac = (idx == 1)
        # DAC 模式下抖动/热噪声无关，ZOH 过采样相关
        self.e_jitter.setEnabled(not is_dac)
        self.e_thermal.setEnabled(not is_dac)
        self.e_osr.setEnabled(is_dac)
        if is_dac:
            # DAC 视图换成 ZOH 相关
            self.cb_view.clear()
            self.cb_view.addItems(["ZOH 阶梯波形", "频谱 + sinc 包络"])
            self.metric_box.setVisible(False)
        else:
            self.cb_view.clear()
            self.cb_view.addItems(["时域波形", "频谱 (FFT)",
                                   "SNR vs 量化位数"])
            self.metric_box.setVisible(True)

    # ── 收集参数 ──────────────────────────────────────────
    def _collect_cfg(self):
        thermal_txt = self.e_thermal.text().strip()
        cfg = dict(
            fs=self._fv(self.e_fs.text(), 100.0) * 1e6,
            f_sig=self._fv(self.e_fsig.text(), 14.6) * 1e6,
            n_bits=max(2, int(self._fv(self.e_bits.text(), 12))),
            vfs=self._fv(self.e_vfs.text(), 2.0),
            amp_dbfs=self._fv(self.e_dbfs.text(), -1.0),
            n_samples=max(256, int(self._fv(self.e_nsamp.text(), 8192))),
            jitter=self._fv(self.e_jitter.text(), 0.0) * 1e-12,
            thermal_snr=(self._fv(thermal_txt, float("inf"))
                         if thermal_txt else float("inf")),
            osr=max(2, int(self._fv(self.e_osr.text(), 8))),
            q_mode=self.cb_qmode.currentText(),
            window=self.cb_window.currentText(),
            coherent=self.chk_coherent.isChecked(),
        )
        return cfg

    # ── 运行 ──────────────────────────────────────────────
    def _run(self):
        try:
            self._do_run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "仿真错误", str(e))

    def _do_run(self):
        cfg = self._collect_cfg()
        is_dac = (self.cb_type.currentIndex() == 1)
        view = self.cb_view.currentText()

        if is_dac:
            r = run_dac_chain(cfg)
            self._last_result = ("dac", r)
            if view.startswith("ZOH"):
                self._plot_zoh_waveform(r)
            else:
                self._plot_dac_spectrum(r)
        else:
            if view.startswith("SNR vs"):
                bits, sinad, enob, ideal = snr_vs_bits(
                    cfg, range(4, 17))
                self._last_result = ("scan", (bits, sinad, enob, ideal))
                self._plot_snr_scan(bits, sinad, enob, ideal)
            else:
                r = run_adc_chain(cfg)
                self._last_result = ("adc", r)
                if view.startswith("时域"):
                    self._plot_adc_waveform(r)
                else:
                    self._plot_adc_spectrum(r)
                self._update_metrics(r["metrics"])

    # ── 指标栏 ────────────────────────────────────────────
    def _update_metrics(self, m):
        fmt = {"ENOB": "{:.2f}"}
        for name, lbl in self._metric_labels.items():
            lbl.setText(fmt.get(name, "{:.1f}").format(m[name]))

    # ── 绘图：ADC 时域 ────────────────────────────────────
    def _plot_adc_waveform(self, r):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        # 只画前若干周期，便于看清量化台阶
        n_show = min(len(r["t"]),
                     int(round(8 / r["f_sig"] * r["fs"])))
        t_us = r["t"][:n_show] * 1e6
        ax.plot(t_us, r["x_analog"][:n_show], color="#999999",
                lw=1.4, label="模拟输入")
        ax.step(t_us, r["x_quant"][:n_show], where="mid",
                color="#BA7517", lw=1.3, label="量化输出")
        ax.set_xlabel("时间 (μs)", fontsize=10)
        ax.set_ylabel("电压 (V)", fontsize=10)
        ax.set_title(
            f"ADC 采样量化波形  {r['n_bits']}-bit  "
            f"fs={r['fs']/1e6:.1f}MHz  LSB={r['lsb']*1e3:.3f}mV",
            fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD")
        ax.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)
        self.canvas.draw()
        self.status.setText(
            f"ADC 时域  |  f_sig={r['f_sig']/1e6:.4f}MHz  "
            f"理论 SQNR={r['sqnr_ideal']:.2f}dB")

    # ── 绘图：ADC 频谱 ────────────────────────────────────
    def _plot_adc_spectrum(self, r):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        psd_db = 10 * np.log10(r["psd"] + 1e-30)
        psd_db -= np.max(psd_db)                       # 归一化到 0 dBFS
        ax.plot(r["freq"] / 1e6, psd_db, color="#BA7517", lw=0.9)
        ax.axvline(r["f_sig"] / 1e6, color="#1F77B4", lw=0.8,
                   ls="--", alpha=0.7, label="信号")
        m = r["metrics"]
        ax.set_xlabel("频率 (MHz)", fontsize=10)
        ax.set_ylabel("幅度 (dBFS)", fontsize=10)
        ax.set_title(
            f"ADC 输出频谱  {r['n_bits']}-bit  "
            f"SINAD={m['SINAD']:.1f}dB  ENOB={m['ENOB']:.2f}bit  "
            f"SFDR={m['SFDR']:.1f}dB",
            fontsize=10)
        ax.set_xlim(0, r["fs"] / 2e6)
        ax.set_ylim(-150, 5)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD",
                  loc="upper right")
        ax.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)
        self.canvas.draw()
        self.status.setText(
            f"ADC 频谱  |  SNR={m['SNR']:.2f}dB  "
            f"THD={m['THD']:.2f}dB  "
            + (f"抖动SNR上限={r['jitter_snr']:.1f}dB"
               if np.isfinite(r["jitter_snr"]) else "无抖动"))

    # ── 绘图：SNR vs 位数 ─────────────────────────────────
    def _plot_snr_scan(self, bits, sinad, enob, ideal):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        ax.plot(bits, ideal, color="#999999", lw=1.5, ls="--",
                marker="o", ms=4, label="理论 6.02N+1.76")
        ax.plot(bits, sinad, color="#BA7517", lw=2.0,
                marker="s", ms=4, label="仿真 SINAD")
        ax.set_xlabel("量化位数 N (bit)", fontsize=10)
        ax.set_ylabel("SINAD (dB)", fontsize=10)
        ax.set_title("SINAD vs 量化位数 — 含非理想效应时偏离理论线",
                     fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD")
        ax.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)
        self.canvas.draw()
        self.status.setText(
            f"SNR 扫描  |  N={bits[0]}~{bits[-1]}bit  "
            f"max ENOB={max(enob):.2f}bit")

    # ── 绘图：DAC ZOH 阶梯波 ──────────────────────────────
    def _plot_zoh_waveform(self, r):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        n_cyc = 6
        n_samp = int(round(n_cyc / r["f_sig"] * r["fs"]))
        n_samp = min(n_samp, len(r["x_quant"]))
        n_hi = n_samp * r["osr"]

        t_samp = np.arange(n_samp) / r["fs"] * 1e6
        t_hi = np.arange(n_hi) / r["fs_hi"] * 1e6
        ax.step(t_hi, r["x_zoh"][:n_hi], where="post",
                color="#BA7517", lw=1.3, label="ZOH 阶梯输出")
        ax.plot(t_samp, r["x_quant"][:n_samp], "o",
                color="#1F77B4", ms=4, label="DAC 码字样本")
        ax.set_xlabel("时间 (μs)", fontsize=10)
        ax.set_ylabel("电压 (V)", fontsize=10)
        ax.set_title(
            f"DAC 零阶保持输出  {r['n_bits']}-bit  "
            f"fs={r['fs']/1e6:.1f}MHz",
            fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD")
        ax.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)
        self.canvas.draw()
        self.status.setText(
            f"DAC ZOH 波形  |  通带边缘(0.4fs) sinc droop = "
            f"{r['droop_db']:.3f} dB")

    # ── 绘图：DAC 频谱 + sinc 包络 ────────────────────────
    def _plot_dac_spectrum(self, r):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        fig.patch.set_facecolor("#F8F8F8")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        psd_db = 10 * np.log10(r["psd"] + 1e-30)
        ref = np.max(psd_db)
        psd_db -= ref
        sinc_db = 20 * np.log10(r["sinc_resp"] + 1e-30)

        ax.plot(r["freq"] / 1e6, psd_db, color="#BA7517", lw=0.9,
                label="ZOH 输出频谱")
        ax.plot(r["freq"] / 1e6, sinc_db, color="#1F77B4", lw=1.4,
                ls="--", label="sinc 包络 |H(f)|")
        # 标注 Nyquist 区边界与镜像
        for k in range(1, r["osr"]):
            ax.axvline(k * r["fs"] / 1e6, color="#CCC", lw=0.6, ls=":")
        ax.set_xlabel("频率 (MHz)", fontsize=10)
        ax.set_ylabel("幅度 (dB)", fontsize=10)
        ax.set_title(
            f"DAC 输出频谱与 sinc 包络  "
            f"通带边缘 droop = {r['droop_db']:.3f} dB",
            fontsize=10)
        ax.set_xlim(0, r["fs_hi"] / 2e6)
        ax.set_ylim(-120, 5)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD",
                  loc="upper right")
        ax.grid(True, color="#E8E8E8", lw=0.5)
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)
        self.canvas.draw()
        self.status.setText(
            f"DAC 频谱  |  虚线为 ZOH 的 sinc 频响，"
            f"镜像副本受其抑制")

    # ── 保存 ──────────────────────────────────────────────
    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "ADDA_model.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")