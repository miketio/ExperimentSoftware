"""Widgets package."""

from app.widgets.camera_view import CameraViewWidget
from app.widgets.stage_control import StageControlWidget
from app.widgets.alignment_panel import AlignmentPanelWidget
from app.widgets.block_grid import BlockGridWidget
from app.widgets.waveguide_panel import WaveguidePanelWidget
from app.widgets.status_bar import CustomStatusBar

__all__ = [
    'CameraViewWidget',
    'StageControlWidget',
    'AlignmentPanelWidget',
    'BlockGridWidget',
    'WaveguidePanelWidget',
    'CustomStatusBar'
]