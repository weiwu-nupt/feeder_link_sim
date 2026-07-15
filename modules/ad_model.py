"""ADC 行为模型：量化输入信号与多个确定性杂散。"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from ui.base_dialog import ModuleDialog


_VFS = 2.0                         # 归一化输入范围 [-1, 1]
_INPUT_AMPLITUDE = 0.9              # 基波幅度，约 -0.915 dBFS
_ACCENT = "#BA7517"
_MAX_SPURS = 32


def _setup_font():
    available = {font.name for font in fm.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return


_setup_font()


def quantize(x, n_bits, vfs=_VFS):
    """N 位 mid-tread 量化器，返回重构输出、LSB 和过载样本数。"""
    n_bits = max(2, int(n_bits))
    lsb = vfs / 2 ** n_bits
    min_code, max_code = -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1
    code = np.round(x / lsb)
    overload_count = int(np.count_nonzero((code < min_code) | (code > max_code)))
    code = np.clip(code, min_code, max_code)
    return code * lsb, lsb, overload_count


def blackman_harris(length):
    """四项 Blackman-Harris 窗，与 MATLAB ``blackmanharris`` 用途一致。"""
    if length < 2:
        return np.ones(length)
    n = np.arange(length)
    phase = 2 * np.pi * n / (length - 1)
    return (
        0.35875 - 0.48829 * np.cos(phase) + 0.14128 * np.cos(2 * phase)
        - 0.01168 * np.cos(3 * phase)
    )


def amplitude_spectrum(x, fs):
    """计算经 Blackman-Harris 窗校正后的单边幅度谱。"""
    n_samples = len(x)
    window = blackman_harris(n_samples)
    coherent_gain = np.mean(window)
    fft = np.fft.rfft(x * window) / (n_samples * coherent_gain)
    if len(fft) > 2:
        fft[1:-1] *= 2.0
    elif len(fft) == 2:
        fft[1] *= 2.0
    return np.fft.rfftfreq(n_samples, 1.0 / fs), np.abs(fft)


def _nearest_bin(freq, target):
    return int(np.argmin(np.abs(freq - target)))


def run_adc_model(n_bits, fs, nfft, f_sig, spur_freqs, spur_dbcs):
    """运行 ``量化(基波 + 多个零相位杂散)`` 的 ADC 行为模型。

    所有频率单位均为 Hz。``spur_dbcs`` 相对于基波幅度，且与
    ``spur_freqs`` 一一对应；杂散相位固定为 0。
    """
    n_bits = max(2, int(n_bits))
    fs = float(fs)
    nfft = int(nfft)
    f_sig = float(f_sig)
    spur_freqs = np.asarray(spur_freqs, dtype=float)
    spur_dbcs = np.asarray(spur_dbcs, dtype=float)

    if fs <= 0 or nfft < 256:
        raise ValueError("采样率必须大于 0，FFT 点数不得小于 256。")
    if not 0 < f_sig < fs / 2:
        raise ValueError("基波频率必须位于 0 与 Nyquist 频率之间。")
    if len(spur_freqs) != len(spur_dbcs) or len(spur_freqs) == 0:
        raise ValueError("每根杂散都需要一组频率和 dBc 电平。")
    if np.any((spur_freqs <= 0) | (spur_freqs >= fs / 2)):
        raise ValueError("所有杂散频率必须位于 0 与 Nyquist 频率之间。")
    if np.any(spur_dbcs > 0):
        raise ValueError("杂散电平应小于或等于 0 dBc。")

    time = np.arange(nfft) / fs
    baseband = _INPUT_AMPLITUDE * np.sin(2 * np.pi * f_sig * time)
    spur_amplitudes = _INPUT_AMPLITUDE * 10 ** (spur_dbcs / 20.0)
    spurs = np.sum(
        spur_amplitudes[:, None] * np.sin(2 * np.pi * spur_freqs[:, None] * time),
        axis=0,
    )
    adc_input = baseband + spurs
    output, lsb, overload_count = quantize(adc_input, n_bits)

    freq, amplitude = amplitude_spectrum(output, fs)
    amplitude_dbfs = 20 * np.log10(amplitude + 1e-30)
    sig_bin = _nearest_bin(freq, f_sig)
    sig_dbfs = amplitude_dbfs[sig_bin]

    # 排除 DC 与基波主瓣，剩余最高谱线即为最大杂散。
    exclude_bins = 10
    valid = np.ones(len(freq), dtype=bool)
    valid[0] = False
    valid[max(0, sig_bin - exclude_bins):sig_bin + exclude_bins + 1] = False
    if not np.any(valid):
        raise ValueError("FFT 点数不足以计算 SFDR。")
    valid_indices = np.flatnonzero(valid)
    largest_spur_bin = valid_indices[np.argmax(amplitude_dbfs[valid])]
    largest_spur_dbfs = amplitude_dbfs[largest_spur_bin]
    sfdr = sig_dbfs - largest_spur_dbfs

    noise_valid = valid.copy()
    for f_spur in spur_freqs:
        spur_bin = _nearest_bin(freq, f_spur)
        noise_valid[max(0, spur_bin - 5):spur_bin + 6] = False
    if np.any(noise_valid):
        noise_floor_dbfs = 10 * np.log10(
            np.mean(amplitude[noise_valid] ** 2) + 1e-30
        )
    else:
        noise_floor_dbfs = float("-inf")

    return {
        "t": time,
        "x_input": adc_input,
        "x_quant": output,
        "lsb": lsb,
        "freq": freq,
        "amplitude": amplitude,
        "amplitude_dbfs": amplitude_dbfs,
        "fs": fs,
        "nfft": nfft,
        "n_bits": n_bits,
        "f_sig": f_sig,
        "spur_freqs": spur_freqs,
        "spur_dbcs": spur_dbcs,
        "largest_spur_bin": largest_spur_bin,
        "overload_count": overload_count,
        "metrics": {
            "signal_dbfs": sig_dbfs,
            "largest_spur_dbfs": largest_spur_dbfs,
            "largest_spur_freq": freq[largest_spur_bin],
            "SFDR": sfdr,
            "noise_floor_dbfs": noise_floor_dbfs,
            "SQNR": 6.02 * n_bits + 1.76,
        },
    }


def run_adc_chain(cfg):
    """兼容调用入口；频率以 Hz 传入。"""
    return run_adc_model(
        cfg.get("n_bits", 12), cfg.get("fs", 100e6), cfg.get("nfft", 65536),
        cfg.get("f_sig", 10e6), cfg.get("spur_freqs", [5e6, 20e6, 35e6]),
        cfg.get("spur_dbcs", [-80.0, -85.0, -90.0]),
    )


_EDIT_STYLE = (
    "QLineEdit{background:#FFF;border:1px solid #D0D0D0;border-radius:3px;"
    "padding:3px 6px;font-size:10pt;color:#111;}"
    f"QLineEdit:focus{{border:1.5px solid {_ACCENT};}}"
)
_LABEL_STYLE = "font-size:10pt;color:#2C2C2A;"
_GROUP_STYLE = (
    "QGroupBox{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;"
    "margin-top:8px;padding:6px 8px;}"
    "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;"
    f"color:{_ACCENT};font-size:9pt;font-weight:bold;}}"
)


def _group(title):
    group = QGroupBox(title)
    group.setStyleSheet(_GROUP_STYLE)
    layout = QVBoxLayout(group)
    layout.setSpacing(4)
    layout.setContentsMargins(6, 4, 6, 6)
    return group


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6.2), dpi=96)
        self.fig.patch.set_facecolor("#F8F8F8")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


class ADDAModelDialog(ModuleDialog):
    """ADC：量化位数和多杂散输入的行为模型界面。"""

    TITLE = "AD 模型"
    ACCENT_COLOR = _ACCENT
    MIN_WIDTH = 1050
    MIN_HEIGHT = 700

    def __init__(self, *args, **kwargs):
        self._last_result = None
        self._spur_edits = []
        super().__init__(*args, **kwargs)

    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background:#DDDDDD;}")

        left = QWidget()
        left.setMinimumWidth(310)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(7)

        basic_group = _group("① ADC 与基波")
        basic_form = QFormLayout()
        basic_form.setSpacing(6)
        self.e_bits = self._edit("12")
        self.e_fs = self._edit("100.0")
        self.e_nfft = self._edit("65536")
        self.e_fsig = self._edit("10.000")
        basic_form.addRow(self._label("量化位数 (bit):"), self.e_bits)
        basic_form.addRow(self._label("采样率 (MHz):"), self.e_fs)
        basic_form.addRow(self._label("FFT 点数:"), self.e_nfft)
        basic_form.addRow(self._label("基波频率 (MHz):"), self.e_fsig)
        basic_group.layout().addLayout(basic_form)
        left_layout.addWidget(basic_group)

        self.spur_group = _group("② 杂散输入")
        count_row = QHBoxLayout()
        count_row.addWidget(self._label("杂散数量:"))
        self.e_spur_count = self._edit("3", width=58)
        count_row.addWidget(self.e_spur_count)
        self.btn_update_spurs = QPushButton("更新条目")
        self.btn_update_spurs.setStyleSheet("font-size:9pt;padding:3px 8px;")
        self.btn_update_spurs.clicked.connect(self._rebuild_spur_rows)
        count_row.addWidget(self.btn_update_spurs)
        count_row.addStretch()
        self.spur_group.layout().addLayout(count_row)
        self.spur_form = QFormLayout()
        self.spur_form.setSpacing(5)
        self.spur_group.layout().addLayout(self.spur_form)
        hint = QLabel("每根杂散均为零初相位正弦；dBc 相对于基波幅度。")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:8.5pt;color:#777;")
        self.spur_group.layout().addWidget(hint)
        left_layout.addWidget(self.spur_group)

        self.btn_run = QPushButton("运行仿真")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{_ACCENT};color:#FFF;border:none;border-radius:5px;"
            "font-size:10pt;font-weight:bold;}QPushButton:hover{background:#8B5A0F;}"
        )
        self.btn_run.clicked.connect(self._run)
        left_layout.addWidget(self.btn_run)
        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#FFF;color:#444;border:1px solid #CCC;border-radius:5px;"
            "font-size:9.5pt;}QPushButton:hover{background:#F5F5F5;}"
        )
        self.btn_save.clicked.connect(self._save)
        left_layout.addWidget(self.btn_save)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(4)
        self.canvas = PlotCanvas()
        right_layout.addWidget(self.canvas, stretch=1)
        self.status = QLabel("就绪 — 设置 ADC、基波与杂散参数后点击「运行仿真」")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size:9pt;color:#666;padding:5px 8px;background:#FFF;"
                                  "border:1px solid #E0E0E0;border-radius:6px;")
        right_layout.addWidget(self.status)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self._rebuild_spur_rows()
        self._run()

    @staticmethod
    def _value(text, default):
        try:
            return float(str(text).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _edit(default, width=110):
        edit = QLineEdit(default)
        edit.setFixedWidth(width)
        edit.setStyleSheet(_EDIT_STYLE)
        return edit

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setStyleSheet(_LABEL_STYLE)
        return label

    def _spur_count(self):
        count = int(self._value(self.e_spur_count.text(), 1))
        if not 1 <= count <= _MAX_SPURS:
            raise ValueError(f"杂散数量必须在 1 到 {_MAX_SPURS} 之间。")
        return count

    def _rebuild_spur_rows(self):
        try:
            count = self._spur_count()
        except ValueError as error:
            QMessageBox.warning(self, "参数错误", str(error))
            return False
        previous = [(freq.text(), level.text()) for freq, level in self._spur_edits]
        while self.spur_form.count():
            item = self.spur_form.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        defaults = [("5.000", "-80"), ("20.000", "-85"), ("35.000", "-90")]
        self._spur_edits = []
        for index in range(count):
            freq_default, level_default = (
                previous[index] if index < len(previous)
                else defaults[index] if index < len(defaults) else ("0.400", "-90")
            )
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            freq = self._edit(freq_default, width=74)
            level = self._edit(level_default, width=62)
            row_layout.addWidget(freq)
            row_layout.addWidget(QLabel("MHz"))
            row_layout.addWidget(level)
            row_layout.addWidget(QLabel("dBc"))
            row_layout.addStretch()
            self.spur_form.addRow(self._label(f"杂散 #{index + 1}:"), row)
            self._spur_edits.append((freq, level))
        return True

    def _collect_cfg(self):
        return {
            "n_bits": max(2, int(self._value(self.e_bits.text(), 12))),
            "fs": self._value(self.e_fs.text(), 100.0) * 1e6,
            "nfft": max(256, int(self._value(self.e_nfft.text(), 65536))),
            "f_sig": self._value(self.e_fsig.text(), 10.000) * 1e6,
            "spur_freqs": [self._value(freq.text(), 0.0) * 1e6 for freq, _ in self._spur_edits],
            "spur_dbcs": [self._value(level.text(), -80.0) for _, level in self._spur_edits],
        }

    def _run(self):
        try:
            if not self._rebuild_spur_rows():
                return
            result = run_adc_model(**self._collect_cfg())
            self._last_result = result
            self._plot_spectrum(result)
        except Exception as error:
            QMessageBox.critical(self, "仿真错误", str(error))

    def _plot_spectrum(self, result):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")
        metrics = result["metrics"]
        ax.plot(result["freq"] / 1e6, result["amplitude_dbfs"], color=_ACCENT,
                lw=0.75, label="ADC 输出频谱")
        max_spur_x = metrics["largest_spur_freq"] / 1e6
        ax.plot(max_spur_x, metrics["largest_spur_dbfs"], "rx", ms=8, mew=1.8,
                label="最大杂散")
        ax.axhline(metrics["noise_floor_dbfs"], color="#777777", lw=1.0, ls="--",
                   label="平均噪声底")
        ax.set_xlim(0, result["fs"] / 2e6)
        lower_limit = min(-300.0, metrics["noise_floor_dbfs"] - 12.0,
                          metrics["largest_spur_dbfs"] - 12.0)
        ax.set_ylim(lower_limit, 5)
        ax.set_xlabel("频率 (MHz)", fontsize=10)
        ax.set_ylabel("幅度 (dBFS)", fontsize=10)
        ax.set_title(
            f"ADC 输出频谱  |  {result['n_bits']}-bit  |  "
            f"{len(result['spur_freqs'])} 根杂散  |  SFDR {result['metrics']['SFDR']:.1f} dBc",
            fontsize=10,
        )
        ax.grid(True, color="#E8E8E8", lw=0.5)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD", loc="upper right")
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
        ax.tick_params(labelsize=9)
        self.canvas.draw()
        self.status.setText(
            f"量化步长 (归一化) = {result['lsb']:.6e}\n"
            f"SFDR = {metrics['SFDR']:.2f} dBc "
            f"(最大杂散在 {self._format_frequency(metrics['largest_spur_freq'])}, "
            f"{metrics['largest_spur_dbfs']:.2f} dBFS)\n"
            f"估计量化噪声基底 = {metrics['noise_floor_dbfs']:.2f} dBFS"
        )

    @staticmethod
    def _format_frequency(frequency_hz):
        if frequency_hz >= 1e6:
            return f"{frequency_hz / 1e6:.3f} MHz"
        return f"{frequency_hz / 1e3:.1f} kHz"

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "ad_model.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")
