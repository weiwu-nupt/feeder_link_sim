"""
各功能模块对话框
"""

from ui.base_dialog import ModuleDialog
from modules.Link_budget import LinkBudgetDialog
from modules.pa_model import PAModelDialog
from modules.ber_analysis import BERAnalysisDialog
from modules.channel_model import ChannelModelDialog
from modules.ad_model import ADDAModelDialog   
from modules.filter_model import FilterModelDialog
from modules.mixer_model import MixerModelDialog

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