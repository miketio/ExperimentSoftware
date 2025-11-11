# app/widgets/camera_view.py - MODIFIED TO ADD COLORBAR

"""Camera View Widget - NOW WITH COLORBAR!"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QCheckBox, QGroupBox, QPushButton, QSpinBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import numpy as np
import cv2


class CameraViewWidget(QWidget):
    """Camera display with live feed, overlays, and colorbar."""
    
    # Colorbar settings
    COLORBAR_WIDTH = 80  # pixels
    SHOW_COLORBAR = True  # Toggle this to enable/disable
    
    def __init__(self, state, signals, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        
        self.current_frame = None
        self.current_stats = {}
        self.camera_thread = None
        
        self._init_ui()
        self._connect_signals()
    
    def set_camera_thread(self, camera_thread):
        """Set camera thread reference."""
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
        self.zoom_combo.addItems(['100%', '200%', '400%', 'Fit'])
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
        self.state.camera.color_scale_auto = checked
        self.min_spin.setEnabled(not checked)
        self.max_spin.setEnabled(not checked)
        
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
        self.state.camera.color_scale_min = self.min_spin.value()
        self.state.camera.color_scale_max = self.max_spin.value()
        
        if not self.state.camera.color_scale_auto and self.camera_thread:
            self.camera_thread.set_color_scale_range(
                self.min_spin.value(),
                self.max_spin.value()
            )
    
    def _on_color_scale_changed(self):
        pass
    
    def update_frame(self, frame: np.ndarray):
        """Display frame with optional colorbar."""
        if frame is None or frame.size == 0:
            return
        
        frame = np.flipud(frame)
        self.current_frame = frame
        
        # Convert to contiguous array
        h, w = frame.shape[:2]
        frame = np.ascontiguousarray(frame)
        
        # Add colorbar if enabled
        if self.SHOW_COLORBAR and self.current_stats:
            frame = self._add_colorbar(frame)
            h, w = frame.shape[:2]  # Update dimensions after colorbar
        
        # Convert to QImage
        if len(frame.shape) == 3:
            bytes_per_line = 3 * w
            q_img = QImage(frame.tobytes(), w, h, bytes_per_line, 
                          QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            q_img = QImage(frame.tobytes(), w, h, bytes_per_line, 
                          QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        
        # Add overlays (crosshair, scale bar)
        if not self.state.camera.show_fourier and \
           (self.state.camera.show_crosshair or self.state.camera.show_scale_bar):
            pixmap = self._add_overlays(pixmap)
        
        # Scale to fit display
        pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(pixmap)
    
    def _add_colorbar(self, frame: np.ndarray) -> np.ndarray:
        """
        Add colorbar to the right side of the frame.
        
        Returns:
            Combined frame with colorbar
        """
        h, w = frame.shape[:2]
        
        # Get current colormap from camera thread
        if self.camera_thread:
            colormap_name = self.camera_thread.color_manager.colormap
        else:
            colormap_name = self.state.camera.colormap
        
        # Get OpenCV colormap constant
        COLORMAPS = {
            'gray': None,
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'inferno': cv2.COLORMAP_INFERNO,
            'turbo': cv2.COLORMAP_TURBO,
            'rainbow': cv2.COLORMAP_RAINBOW,
        }
        colormap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)
        
        # Create gradient (high to low, top to bottom)
        gradient = np.linspace(255, 0, h).astype(np.uint8)
        gradient = np.tile(gradient[:, None], (1, self.COLORBAR_WIDTH))
        
        # Apply colormap
        if colormap is not None:
            colorbar = cv2.applyColorMap(gradient, colormap)
        else:
            # Grayscale - convert to RGB
            colorbar = cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB)
        
        # Check if frame needs RGB conversion
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Combine frame + colorbar
        combined = np.hstack((frame, colorbar))
        
        # Add min/max labels
        vmin = self.current_stats.get('min', 0)
        vmax = self.current_stats.get('max', 65535)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 2
        
        # Max label (top)
        cv2.putText(combined, f"Max: {int(vmax)}", 
                   (w + 5, 20), font, font_scale, font_color, thickness)
        
        # Min label (bottom)
        cv2.putText(combined, f"Min: {int(vmin)}", 
                   (w + 5, h - 10), font, font_scale, font_color, thickness)
        
        return combined
    
    def _add_overlays(self, pixmap: QPixmap) -> QPixmap:
        """Add crosshair and scale bar."""
        painter = QPainter(pixmap)
        w = pixmap.width()
        h = pixmap.height()
        
        # Adjust for colorbar width if present
        if self.SHOW_COLORBAR:
            display_w = int(w * (1 - self.COLORBAR_WIDTH / (w + self.COLORBAR_WIDTH)))
        else:
            display_w = w
        
        if self.state.camera.show_crosshair:
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            cx, cy = display_w // 2, h // 2
            painter.drawLine(cx - 50, cy, cx + 50, cy)
            painter.drawLine(cx, cy - 50, cx, cy + 50)
            painter.drawEllipse(cx - 25, cy - 25, 50, 50)
        
        if self.state.camera.show_scale_bar:
            scale_bar_um = 50.0
            um_per_pixel = 0.3
            scale_bar_pixels = int(scale_bar_um / um_per_pixel)
            
            bar_x = display_w - scale_bar_pixels - 20
            bar_y = h - 40
            
            painter.setPen(QPen(QColor(0, 0, 0), 5))
            painter.drawLine(bar_x, bar_y, bar_x + scale_bar_pixels, bar_y)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(bar_x, bar_y, bar_x + scale_bar_pixels, bar_y)
            
            painter.drawLine(bar_x, bar_y - 10, bar_x, bar_y + 10)
            painter.drawLine(bar_x + scale_bar_pixels, bar_y - 10, 
                           bar_x + scale_bar_pixels, bar_y + 10)
            
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
    
    # Zoom methods (unchanged)
    def set_zoom(self, zoom: float):
        self.state.camera.zoom_level = zoom
        if not self.camera_thread or not hasattr(self.camera_thread.camera, 'set_roi'):
            return
        
        try:
            sensor_w, sensor_h = self.camera_thread.camera.get_sensor_size()
            roi_w = int(sensor_w / zoom)
            roi_h = int(sensor_h / zoom)
            
            min_roi_size = 64
            roi_w = max(min_roi_size, min(roi_w, sensor_w))
            roi_h = max(min_roi_size, min(roi_h, sensor_h))
            
            left = (sensor_w - roi_w) // 2
            top = (sensor_h - roi_h) // 2
            
            left = max(0, min(left, sensor_w - roi_w))
            top = max(0, min(top, sensor_h - roi_h))
            
            self.camera_thread.camera.set_roi(left, top, roi_w, roi_h)
            
        except Exception as e:
            print(f"[CameraView] Failed to set zoom: {e}")
            try:
                sensor_w, sensor_h = self.camera_thread.camera.get_sensor_size()
                self.camera_thread.camera.set_roi(0, 0, sensor_w, sensor_h)
            except:
                pass
    
    def zoom_fit(self):
        self.state.camera.zoom_level = 1.0
        if self.camera_thread and hasattr(self.camera_thread.camera, 'set_roi'):
            try:
                sensor_w, sensor_h = self.camera_thread.camera.get_sensor_size()
                self.camera_thread.camera.set_roi(0, 0, sensor_w, sensor_h)
            except Exception as e:
                print(f"[CameraView] Failed to reset ROI: {e}")