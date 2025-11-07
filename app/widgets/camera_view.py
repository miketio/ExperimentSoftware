"""
Camera View Widget

Displays live camera feed with controls for zoom, colormap, and color scaling.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QCheckBox, QGroupBox, QPushButton, QSpinBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import numpy as np


class CameraViewWidget(QWidget):
    """Camera display with live feed and controls."""
    
    def __init__(self, state, signals, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        
        self.current_frame = None
        self.current_stats = {}
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Camera display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.image_label, stretch=1)
        
        # Controls
        controls = self._create_controls()
        layout.addWidget(controls)
    
    def _create_controls(self):
        """Create camera controls."""
        group = QGroupBox("Camera Controls")
        layout = QVBoxLayout()
        
        # Row 1: Colormap and zoom
        row1 = QHBoxLayout()
        
        row1.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'jet', 'hot', 'viridis', 'plasma', 'inferno'])
        self.colormap_combo.setCurrentText(self.state.camera.colormap)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        row1.addWidget(self.colormap_combo)
        
        row1.addSpacing(20)
        
        row1.addWidget(QLabel("Zoom:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(['25%', '50%', '100%', '200%', 'Fit'])
        self.zoom_combo.setCurrentText('Fit')
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_changed)
        row1.addWidget(self.zoom_combo)
        
        row1.addStretch()
        layout.addLayout(row1)
        
        # Row 2: Auto-scale toggle
        row2 = QHBoxLayout()
        
        self.auto_scale_check = QCheckBox("Auto-scale")
        self.auto_scale_check.setChecked(self.state.camera.color_scale_auto)
        self.auto_scale_check.toggled.connect(self._on_auto_scale_toggled)
        row2.addWidget(self.auto_scale_check)
        
        self.crosshair_check = QCheckBox("Show crosshair")
        self.crosshair_check.setChecked(self.state.camera.show_crosshair)
        row2.addWidget(self.crosshair_check)
        
        row2.addStretch()
        layout.addLayout(row2)
        
        # Row 3: Manual scale controls
        row3 = QHBoxLayout()
        
        row3.addWidget(QLabel("Min:"))
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 65535)
        self.min_spin.setValue(self.state.camera.color_scale_min)
        self.min_spin.valueChanged.connect(self._on_manual_scale_changed)
        self.min_spin.setEnabled(not self.state.camera.color_scale_auto)
        row3.addWidget(self.min_spin)
        
        row3.addWidget(QLabel("Max:"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 65535)
        self.max_spin.setValue(self.state.camera.color_scale_max)
        self.max_spin.valueChanged.connect(self._on_manual_scale_changed)
        self.max_spin.setEnabled(not self.state.camera.color_scale_auto)
        row3.addWidget(self.max_spin)
        
        row3.addStretch()
        layout.addLayout(row3)
        
        # Stats display
        self.stats_label = QLabel("Min: -- | Max: -- | Mean: --")
        self.stats_label.setStyleSheet("QLabel { font-family: monospace; }")
        layout.addWidget(self.stats_label)
        
        group.setLayout(layout)
        return group
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.color_scale_changed.connect(self._request_frame_update)
    
    def _on_colormap_changed(self, colormap: str):
        """Handle colormap change."""
        self.state.camera.colormap = colormap
        self.signals.colormap_changed.emit(colormap)
    
    def _on_zoom_changed(self, zoom_text: str):
        """Handle zoom change."""
        if zoom_text == 'Fit':
            self.zoom_fit()
        else:
            zoom = float(zoom_text.strip('%')) / 100.0
            self.set_zoom(zoom)
    
    def _on_auto_scale_toggled(self, checked: bool):
        """Handle auto-scale toggle."""
        self.state.camera.color_scale_auto = checked
        self.min_spin.setEnabled(not checked)
        self.max_spin.setEnabled(not checked)
        self.signals.color_scale_changed.emit()
    
    def _on_manual_scale_changed(self):
        """Handle manual scale change."""
        self.state.camera.color_scale_min = self.min_spin.value()
        self.state.camera.color_scale_max = self.max_spin.value()
        if not self.state.camera.color_scale_auto:
            self.signals.color_scale_changed.emit()
    
    def _request_frame_update(self):
        """Request frame update from camera thread."""
        # Camera thread will automatically use new settings
        pass
    
    def update_frame(self, frame: np.ndarray):
        """
        Update displayed frame.
        
        Args:
            frame: RGB image (H, W, 3) uint8
        """
        if frame is None or frame.size == 0:
            return
        
        self.current_frame = frame
        
        # Convert to QImage
        h, w = frame.shape[:2]
        if len(frame.shape) == 3:
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # Add crosshair if enabled
        if self.state.camera.show_crosshair:
            pixmap = QPixmap.fromImage(q_img)
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            cx, cy = w // 2, h // 2
            painter.drawLine(cx - 20, cy, cx + 20, cy)
            painter.drawLine(cx, cy - 20, cx, cy + 20)
            painter.end()
        else:
            pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit label
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled)
    
    def update_stats(self, stats: dict):
        """Update image statistics display."""
        self.current_stats = stats
        self.stats_label.setText(
            f"Min: {stats.get('min', 0):.0f} | "
            f"Max: {stats.get('max', 0):.0f} | "
            f"Mean: {stats.get('mean', 0):.1f}"
        )
    
    def zoom_fit(self):
        """Zoom to fit window."""
        self.state.camera.zoom_level = -1.0  # Special value for "fit"
    
    def set_zoom(self, zoom: float):
        """Set zoom level."""
        self.state.camera.zoom_level = zoom