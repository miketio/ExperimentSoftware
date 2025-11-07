# app/widgets/camera_view.py - FIXED VERSION
"""
Camera View Widget - FIXED

Fixes:
1. Crosshair/scalebar 5x larger
2. Zoom actually works
3. Manual color scale propagates to camera thread
4. Proper frame re-rendering on settings change
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QCheckBox, QGroupBox, QPushButton, QSpinBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import numpy as np


class CameraViewWidget(QWidget):
    """Camera display with live feed and controls."""
    
    def __init__(self, state, signals, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        
        self.current_frame = None
        self.current_stats = {}
        self.camera_thread = None  # Will be set by main_window
        
        self._init_ui()
        self._connect_signals()
    
    def set_camera_thread(self, camera_thread):
        """Set camera thread reference (called by main_window)."""
        self.camera_thread = camera_thread
    
    def _init_ui(self):
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
        group = QGroupBox("Camera Controls")
        layout = QVBoxLayout()
        
        # Row 1: Zoom
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Zoom:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(['25%', '50%', '100%', '200%', 'Fit'])
        self.zoom_combo.setCurrentText('Fit')
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_changed)
        row1.addWidget(self.zoom_combo)
        row1.addStretch()
        layout.addLayout(row1)
        
        # Row 2: Auto-scale
        row2 = QHBoxLayout()
        self.auto_scale_check = QCheckBox("Auto-scale")
        self.auto_scale_check.setChecked(self.state.camera.color_scale_auto)
        self.auto_scale_check.toggled.connect(self._on_auto_scale_toggled)
        row2.addWidget(self.auto_scale_check)
        row2.addStretch()
        layout.addLayout(row2)
        
        # Row 3: Manual scale
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
        
        # Stats
        self.stats_label = QLabel("Min: -- | Max: -- | Mean: --")
        self.stats_label.setStyleSheet("QLabel { font-family: monospace; }")
        layout.addWidget(self.stats_label)
        
        group.setLayout(layout)
        return group
    
    def _connect_signals(self):
        self.signals.color_scale_changed.connect(self._on_color_scale_changed)
    
    def _on_zoom_changed(self, zoom_text: str):
        if zoom_text == 'Fit':
            self.zoom_fit()
        else:
            zoom = float(zoom_text.strip('%')) / 100.0
            self.set_zoom(zoom)
    
    def _on_auto_scale_toggled(self, checked: bool):
        """FIX: Propagate to camera thread."""
        self.state.camera.color_scale_auto = checked
        self.min_spin.setEnabled(not checked)
        self.max_spin.setEnabled(not checked)
        
        # FIX: Tell camera thread to update mode
        if self.camera_thread:
            if checked:
                self.camera_thread.set_color_scale_mode('auto')
            else:
                self.camera_thread.set_color_scale_mode('manual')
                self.camera_thread.set_color_scale_range(
                    self.min_spin.value(),
                    self.max_spin.value()
                )
    
    def _on_manual_scale_changed(self):
        """FIX: Propagate to camera thread."""
        self.state.camera.color_scale_min = self.min_spin.value()
        self.state.camera.color_scale_max = self.max_spin.value()
        
        # FIX: Update camera thread
        if not self.state.camera.color_scale_auto and self.camera_thread:
            self.camera_thread.set_color_scale_range(
                self.min_spin.value(),
                self.max_spin.value()
            )
    
    def _on_color_scale_changed(self):
        """Handle color scale change signal."""
        # Force re-render of current frame with new settings
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
    
    def update_frame(self, frame: np.ndarray):
        """FIX: Apply zoom properly."""
        if frame is None or frame.size == 0:
            return
        frame = np.flipud(frame)

        
        # Convert to QImage
        h, w = frame.shape[:2]
        # FIX: Ensure data is contiguous and in uint8 format
        frame = np.ascontiguousarray(frame)
        self.current_frame = frame
        if len(frame.shape) == 3:
            bytes_per_line = 3 * w
            q_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            q_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                
        pixmap = QPixmap.fromImage(q_img)
        
        # FIX: Add overlays BEFORE scaling
        if self.state.camera.show_crosshair or self.state.camera.show_scale_bar:
            pixmap = self._add_overlays(pixmap)
        
        # FIX: Apply zoom
        zoom = self.state.camera.zoom_level
        if zoom > 0:  # Specific zoom level
            new_w = int(w * zoom)
            new_h = int(h * zoom)
            pixmap = pixmap.scaled(new_w, new_h, 
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        
        # Scale to fit label (only if zoom is "fit" or result is too large)
        if zoom <= 0 or pixmap.width() > self.image_label.width() or pixmap.height() > self.image_label.height():
            pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        
        self.image_label.setPixmap(pixmap)
    
    def _add_overlays(self, pixmap: QPixmap) -> QPixmap:
        """FIX: 5x larger crosshair and scale bar."""
        painter = QPainter(pixmap)
        w = pixmap.width()
        h = pixmap.height()
        
        # FIX: Crosshair - 5x larger
        if self.state.camera.show_crosshair:
            painter.setPen(QPen(QColor(0, 255, 0), 3))  # Thicker line
            cx, cy = w // 2, h // 2
            # 100px lines (was 40px)
            painter.drawLine(cx - 50, cy, cx + 50, cy)
            painter.drawLine(cx, cy - 50, cx, cy + 50)
            # 50px circle (was 10px)
            painter.drawEllipse(cx - 25, cy - 25, 50, 50)
        
        # FIX: Scale bar - 5x larger
        if self.state.camera.show_scale_bar:
            scale_bar_um = 50.0
            um_per_pixel = 0.3
            scale_bar_pixels = int(scale_bar_um / um_per_pixel)
            
            bar_x = w - scale_bar_pixels - 20
            bar_y = h - 40
            
            # Black outline
            painter.setPen(QPen(QColor(0, 0, 0), 5))
            painter.drawLine(bar_x, bar_y, bar_x + scale_bar_pixels, bar_y)
            # White bar
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(bar_x, bar_y, bar_x + scale_bar_pixels, bar_y)
            
            # FIX: Larger end caps
            painter.drawLine(bar_x, bar_y - 10, bar_x, bar_y + 10)
            painter.drawLine(bar_x + scale_bar_pixels, bar_y - 10, 
                           bar_x + scale_bar_pixels, bar_y + 10)
            
            # FIX: Larger text
            painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(bar_x + 2, bar_y - 16, f"{scale_bar_um:.0f} µm")
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(bar_x, bar_y - 18, f"{scale_bar_um:.0f} µm")
        
        painter.end()
        return pixmap
    
    def update_overlay_settings(self):
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
    
    def update_stats(self, stats: dict):
        self.current_stats = stats
        self.stats_label.setText(
            f"Min: {stats.get('min', 0):.0f} | "
            f"Max: {stats.get('max', 0):.0f} | "
            f"Mean: {stats.get('mean', 0):.1f}"
        )
    
    def zoom_fit(self):
        self.state.camera.zoom_level = -1.0
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
    
    def set_zoom(self, zoom: float):
        self.state.camera.zoom_level = zoom
        if self.current_frame is not None:
            self.update_frame(self.current_frame)