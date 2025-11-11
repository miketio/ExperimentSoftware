# app/widgets/block1_position_dialog_live.py - NEW FILE

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
import numpy as np


class Block1PositionDialogLive(QDialog):
    """Block 1 position dialog WITH live camera view."""
    
    def __init__(self, state, runtime_layout, camera_thread, parent=None):
        super().__init__(parent)
        self.state = state
        self.runtime_layout = runtime_layout
        self.camera_thread = camera_thread
        
        self.setWindowTitle("Set Block 1 Position")
        self.setModal(True)
        self.setMinimumSize(800, 700)
        
        layout = QVBoxLayout()
        
        # Info
        info = QLabel(
            "â­ <b>Set Block 1 Center Position</b>\n\n"
            "1. Use stage controls to move to Block 1 center\n"
            "2. Verify position in camera view below\n"
            "3. Click 'Capture Position'"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #2196F3; }")
        self.camera_label.setScaledContents(True)
        layout.addWidget(self.camera_label)
        
        # Current position
        self.current_label = QLabel("Current: Y=?.???, Z=?.???")
        self.current_label.setStyleSheet("QLabel { font-family: monospace; font-size: 14pt; font-weight: bold; }")
        layout.addWidget(self.current_label)
        
        # Capture button
        btn_capture = QPushButton("📷 Capture Position")
        btn_capture.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 12pt; }")
        btn_capture.clicked.connect(self._capture)
        layout.addWidget(btn_capture)
        
        # Manual entry
        manual = QHBoxLayout()
        manual.addWidget(QLabel("Captured Position:"))
        
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-100000, 100000)
        self.y_spin.setDecimals(3)
        self.y_spin.setSuffix(" µm")
        
        # Load current value
        if self.runtime_layout.has_block_1_position():
            y, z = self.runtime_layout.get_block_1_position()
            self.y_spin.setValue(y)
        else:
            self.y_spin.setValue(0.0)
        
        manual.addWidget(QLabel("Y:"))
        manual.addWidget(self.y_spin)
        
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-100000, 100000)
        self.z_spin.setDecimals(3)
        self.z_spin.setSuffix(" µm")
        
        if self.runtime_layout.has_block_1_position():
            y, z = self.runtime_layout.get_block_1_position()
            self.z_spin.setValue(z)
        else:
            self.z_spin.setValue(0.0)
        
        manual.addWidget(QLabel("Z:"))
        manual.addWidget(self.z_spin)
        
        layout.addLayout(manual)
        
        # Buttons
        buttons = QHBoxLayout()
        
        btn_ok = QPushButton("✅ OK")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("QPushButton { padding: 8px; font-size: 11pt; }")
        buttons.addWidget(btn_ok)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_cancel)
        
        layout.addLayout(buttons)
        self.setLayout(layout)
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_position)
        self.update_timer.start(200)
        
        # Connect to camera stream
        if camera_thread:
            camera_thread.frame_ready.connect(self._update_camera_view)
    
    def _update_position(self):
        """Update current position display."""
        y, z = self.state.stage_position['y'], self.state.stage_position['z']
        self.current_label.setText(f"Current: Y={y:.3f}, Z={z:.3f} µm")
    
    def _update_camera_view(self, frame):
        """Update camera display."""
        if frame is None or frame.size == 0:
            return
        
        h, w = frame.shape[:2]
        
        if len(frame.shape) == 3:
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap)
    
    def _capture(self):
        """Capture current position."""
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        
        self.y_spin.setValue(y)
        self.z_spin.setValue(z)
        
        self.current_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 14pt; font-weight: bold; color: green; }"
        )
    
    def accept(self):
        """Save position on OK."""
        self.runtime_layout.set_block_1_position(
            self.y_spin.value(),
            self.z_spin.value()
        )
        super().accept()