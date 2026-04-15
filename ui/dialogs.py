"""
各功能模块对话框（当前均为占位，后续逐步填充内容）
Module Dialogs — Stubs for future implementation
"""

from ui.base_dialog import ModuleDialog
from modules.Link_budget import LinkBudgetDialog
from modules.pa_mode import PAModelDialog


# ── 无线信道建模 ───────────────────────────────────────────

class ChannelModelDialog(ModuleDialog):
    TITLE = "无线信道建模"
    SUBTITLE = "Radio Channel Modeling"
    ACCENT_COLOR = "#1D9E75"
    MIN_WIDTH = 720
    MIN_HEIGHT = 520

    def build_content(self, layout):
        pass   # TODO: 信道模型选择、参数配置、响应曲线


# ── 器件建模：AD/DA ────────────────────────────────────────

class ADDAModelDialog(ModuleDialog):
    TITLE = "AD/DA 模型"
    SUBTITLE = "ADC / DAC Model"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH = 680
    MIN_HEIGHT = 480

    def build_content(self, layout):
        pass   # TODO: 量化精度、采样率、ENOB、SFDR


# ── 器件建模：滤波器 ───────────────────────────────────────

class FilterModelDialog(ModuleDialog):
    TITLE = "滤波器模型"
    SUBTITLE = "Filter Model"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH = 720
    MIN_HEIGHT = 520

    def build_content(self, layout):
        pass   # TODO: 滤波器类型选择、幅频/相频响应绘图


# ── 器件建模：混频器 ───────────────────────────────────────

class MixerModelDialog(ModuleDialog):
    TITLE = "混频器模型"
    SUBTITLE = "Mixer Model"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH = 680
    MIN_HEIGHT = 480

    def build_content(self, layout):
        pass   # TODO: 变频损耗、IIP3、镜像抑制配置


# ── 单链路仿真 ─────────────────────────────────────────────

class SingleLinkSimDialog(ModuleDialog):
    TITLE = "单链路仿真"
    SUBTITLE = "Single Link Simulation"
    ACCENT_COLOR = "#7F77DD"
    MIN_WIDTH = 900
    MIN_HEIGHT = 620

    def build_content(self, layout):
        pass   # TODO: 信号链路框图 + 各级参数配置 + 波形显示


# ── 多链路仿真 ─────────────────────────────────────────────

class MultiLinkSimDialog(ModuleDialog):
    TITLE = "多链路仿真"
    SUBTITLE = "Multi-Link Simulation"
    ACCENT_COLOR = "#D85A30"
    MIN_WIDTH = 1000
    MIN_HEIGHT = 680

    def build_content(self, layout):
        pass   # TODO: 多链路拓扑配置、干扰分析、频率规划