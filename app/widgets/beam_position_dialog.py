# NEW FILE: app/widgets/beam_position_dialog.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QPushButton, QGroupBox
)

class BeamPositionDialog(QDialog):
    """Simple dialog to set beam position manually."""
    
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        
        self.setWindowTitle("Set Beam Position")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Info
        info = QLabel(
            "⚠️ <b>Define Beam Position</b>\n\n"
            "Enter pixel coordinates where the optical beam hits the camera.\n"
            "This is typically NOT the image center (1024, 1024)."
        )
        info.setStyleSheet("QLabel { background-color: #FFF3CD; padding: 10px; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Current position
        group = QGroupBox("Beam Position (pixels)")
        group_layout = QHBoxLayout()
        
        group_layout.addWidget(QLabel("X:"))
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 2048)
        self.x_spin.setValue(self.state.camera.beam_position_px[0])
        group_layout.addWidget(self.x_spin)
        
        group_layout.addWidget(QLabel("Y:"))
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 2048)
        self.y_spin.setValue(self.state.camera.beam_position_px[1])
        group_layout.addWidget(self.y_spin)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Offset from center
        self.offset_label = QLabel()
        self.offset_label.setStyleSheet("QLabel { font-family: monospace; }")
        self._update_offset()
        layout.addWidget(self.offset_label)
        
        self.x_spin.valueChanged.connect(self._update_offset)
        self.y_spin.valueChanged.connect(self._update_offset)
        
        # Buttons
        buttons = QHBoxLayout()
        
        btn_center = QPushButton("Reset to Center")
        btn_center.clicked.connect(self._reset_center)
        buttons.addWidget(btn_center)
        
        buttons.addStretch()
        
        btn_ok = QPushButton("✅ OK")
        btn_ok.clicked.connect(self.accept)
        buttons.addWidget(btn_ok)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_cancel)
        
        layout.addLayout(buttons)
        self.setLayout(layout)
    
    def _update_offset(self):
        """Update offset display."""
        center_x, center_y = 1024, 1024
        beam_x = self.x_spin.value()
        beam_y = self.y_spin.value()
        
        offset_x = beam_x - center_x
        offset_y = beam_y - center_y
        
        self.offset_label.setText(
            f"Offset from center: ΔX={offset_x:+d} px, ΔY={offset_y:+d} px"
        )
    
    def _reset_center(self):
        """Reset to image center."""
        self.x_spin.setValue(1024)
        self.y_spin.setValue(1024)
    
    def accept(self):
        """Save beam position on OK."""
        self.state.camera.beam_position_px = (
            self.x_spin.value(),
            self.y_spin.value()
        )
        super().accept()