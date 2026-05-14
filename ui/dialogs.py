"""
各功能模块对话框
"""

from ui.base_dialog import ModuleDialog
from modules.Link_budget import LinkBudgetDialog
from modules.pa_mode import PAModelDialog
from modules.ber_analysis import BERAnalysisDialog
from modules.channel_model import ChannelModelDialog


# ── 器件建模：AD/DA ────────────────────────────────────────

class ADDAModelDialog(ModuleDialog):
    TITLE = "AD/DA 模型"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH = 680
    MIN_HEIGHT = 480

    def build_content(self, layout):
        pass


# ── 器件建模：滤波器 ───────────────────────────────────────

class FilterModelDialog(ModuleDialog):
    TITLE = "滤波器模型"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH = 720
    MIN_HEIGHT = 520

    def build_content(self, layout):
        pass


# ── 器件建模：混频器 ───────────────────────────────────────

class MixerModelDialog(ModuleDialog):
    TITLE = "混频器模型"
    ACCENT_COLOR = "#BA7517"
    MIN_WIDTH = 680
    MIN_HEIGHT = 480

    def build_content(self, layout):
        pass


# ── 单链路仿真 ─────────────────────────────────────────────

class SingleLinkSimDialog(ModuleDialog):
    TITLE = "单链路仿真"
    ACCENT_COLOR = "#7F77DD"
    MIN_WIDTH = 900
    MIN_HEIGHT = 620

    def build_content(self, layout):
        pass


# ── 多链路仿真 ─────────────────────────────────────────────

class MultiLinkSimDialog(ModuleDialog):
    TITLE = "多链路仿真"
    ACCENT_COLOR = "#D85A30"
    MIN_WIDTH = 1000
    MIN_HEIGHT = 680

    def build_content(self, layout):
        pass