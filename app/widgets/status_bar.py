"""Custom Status Bar."""

from PyQt6.QtWidgets import QStatusBar, QLabel
from PyQt6.QtCore import Qt


class CustomStatusBar(QStatusBar):
    """Enhanced status bar with permanent widgets."""
    
    def __init__(self, state, signals, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        
        # Position label (permanent)
        self.pos_label = QLabel("Stage: X=0.000 Y=0.000 Z=0.000 µm")
        self.pos_label.setStyleSheet("QLabel { font-family: monospace; }")
        self.addPermanentWidget(self.pos_label)
        
        # Hardware status (permanent)
        self.hw_label = QLabel("📷 ✓ | 🔬 ✓")
        self.hw_label.setToolTip("Camera and Stage status")
        self.addPermanentWidget(self.hw_label)
        
        # Alignment status (permanent)
        self.align_label = QLabel("⚪ Not calibrated")
        self.addPermanentWidget(self.align_label)
        
        self._connect_signals()
        self._update_display()
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.stage_position_changed.connect(self._update_position)
        self.signals.global_alignment_complete.connect(lambda res: self._update_alignment())
    
    def _update_position(self, axis: str, position: float):
        """Update position display."""
        x, y, z = self.state.get_stage_position()
        self.pos_label.setText(f"Stage: X={x:.3f} Y={y:.3f} Z={z:.3f} µm")
    
    def _update_alignment(self):
        """Update alignment status."""
        if self.state.global_calibrated:
            self.align_label.setText("🟢 Calibrated")
        else:
            self.align_label.setText("⚪ Not calibrated")
    
    def _update_display(self):
        """Update all displays."""
        x, y, z = self.state.get_stage_position()
        self.pos_label.setText(f"Stage: X={x:.3f} Y={y:.3f} Z={z:.3f} µm")
        
        cam_icon = "📷 ✓" if self.state.camera_connected else "📷 ✗"
        stage_icon = "🔬 ✓" if self.state.stage_connected else "🔬 ✗"
        self.hw_label.setText(f"{cam_icon} | {stage_icon}")
        
        self._update_alignment()