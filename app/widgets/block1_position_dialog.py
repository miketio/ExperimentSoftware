# app/widgets/block1_position_dialog.py
"""Simple dialog to set Block 1 position for existing layouts."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QMessageBox, QGroupBox
)
from PyQt6.QtCore import QTimer

# app/widgets/block1_position_dialog.py - SIMPLIFIED VERSION
class Block1PositionDialog(QDialog):
    """Block 1 position dialog WITHOUT duplicate camera view."""
    
    def __init__(self, state, runtime_layout, parent=None):
        super().__init__(parent)
        self.state = state
        self.runtime_layout = runtime_layout
        
        self.setWindowTitle("Set Block 1 Position")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # Info
        info = QLabel(
            "⭐ <b>Set Block 1 Center Position</b>\n\n"
            "1. Use stage controls in main window to move to Block 1 center\n"
            "2. Watch the camera view in main window\n"
            "3. Click 'Capture Position' when centered"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Current position (larger display)
        self.current_label = QLabel("Current: Y=?.???, Z=?.???")
        self.current_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 18pt; "
            "font-weight: bold; background-color: black; color: lime; "
            "padding: 20px; border: 3px solid #2196F3; }"
        )
        layout.addWidget(self.current_label)
        
        # Capture button (large and prominent)
        btn_capture = QPushButton("📷 Capture Current Position")
        btn_capture.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 15px; font-size: 14pt; }"
        )
        btn_capture.clicked.connect(self._capture)
        layout.addWidget(btn_capture)
        
        # Manual entry section
        manual_group = QGroupBox("Captured Position (or enter manually)")
        manual_layout = QHBoxLayout()
        
        manual_layout.addWidget(QLabel("Y:"))
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-100000, 100000)
        self.y_spin.setDecimals(3)
        self.y_spin.setSuffix(" µm")
        self.y_spin.setMinimumWidth(150)
        
        # Load current value if exists
        if self.runtime_layout.has_block_1_position():
            y, z = self.runtime_layout.get_block_1_position()
            self.y_spin.setValue(y)
        
        manual_layout.addWidget(self.y_spin)
        
        manual_layout.addWidget(QLabel("Z:"))
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-100000, 100000)
        self.z_spin.setDecimals(3)
        self.z_spin.setSuffix(" µm")
        self.z_spin.setMinimumWidth(150)
        
        if self.runtime_layout.has_block_1_position():
            y, z = self.runtime_layout.get_block_1_position()
            self.z_spin.setValue(z)
        
        manual_layout.addWidget(self.z_spin)
        manual_layout.addStretch()
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        # Buttons
        buttons = QHBoxLayout()
        
        btn_ok = QPushButton("✅ OK")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("QPushButton { padding: 10px; font-size: 12pt; }")
        buttons.addWidget(btn_ok)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setStyleSheet("QPushButton { padding: 10px; font-size: 12pt; }")
        buttons.addWidget(btn_cancel)
        
        layout.addLayout(buttons)
        self.setLayout(layout)
        
        # Update timer for current position
        from PyQt6.QtCore import QTimer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_position)
        self.update_timer.start(200)
    
    def _update_position(self):
        """Update current position display."""
        y, z = self.state.stage_position['y'], self.state.stage_position['z']
        self.current_label.setText(f"Current: Y={y:.3f}, Z={z:.3f} µm")
    
    def _capture(self):
        """Capture current position."""
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        
        self.y_spin.setValue(y)
        self.z_spin.setValue(z)
        
        # Flash green to confirm capture
        self.current_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 18pt; "
            "font-weight: bold; background-color: green; color: white; "
            "padding: 20px; border: 3px solid lime; }"
        )
        
        # Reset after 500ms
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, lambda: self.current_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 18pt; "
            "font-weight: bold; background-color: black; color: lime; "
            "padding: 20px; border: 3px solid #2196F3; }"
        ))
    
    def accept(self):
        """Save position on OK."""
        self.runtime_layout.set_block_1_position(
            self.y_spin.value(),
            self.z_spin.value()
        )
        super().accept()