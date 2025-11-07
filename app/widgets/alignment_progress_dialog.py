# app/widgets/alignment_progress_dialog.py
"""
Alignment Progress Dialog

Shows real-time progress during alignment with:
- Progress bar
- Status messages
- Fiducial thumbnails
- Text log
- Cancel button
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QScrollArea, QWidget, QGridLayout
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap
import numpy as np


class AlignmentProgressDialog(QDialog):
    """
    Progress dialog for alignment operations.
    
    Features:
    - Progress bar (0-100%)
    - Status text
    - Thumbnail grid (fiducials found)
    - Text log
    - Cancel button
    """
    
    def __init__(self, title: str = "Alignment Progress", parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(800, 600)
        
        # Prevent close button (must use Cancel)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint
        )
        
        self._init_ui()
        
        # Track state
        self.is_complete = False
        self.thumbnail_count = 0
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        layout.addWidget(self.status_label)
        
        # Thumbnail grid (scrollable)
        thumb_label = QLabel("Found Fiducials:")
        thumb_label.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(thumb_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QGridLayout()
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        scroll.setWidget(self.thumbnail_container)
        
        layout.addWidget(scroll)
        
        # Text log
        log_label = QLabel("Progress Log:")
        log_label.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        layout.addWidget(self.log_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def update_progress(self, current: int, total: int, status: str):
        """
        Update progress bar and status.
        
        Args:
            current: Current step (0-based)
            total: Total steps
            status: Status message
        """
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"{current}/{total} - {percent}%")
        
        self.status_label.setText(status)
        self.append_log(status)
    
    def add_fiducial_thumbnail(
        self,
        block_id: int,
        corner: str,
        image: np.ndarray,
        error_um: float
    ):
        """
        Add fiducial thumbnail to grid.
        
        Args:
            block_id: Block ID
            corner: Corner name
            image: Image array (uint8 or uint16)
            error_um: Verification error in µm
        """
        if image is None or image.size == 0:
            return
        
        # Convert to 8-bit RGB if needed
        if image.dtype == np.uint16:
            # Use percentile scaling for better visualization
            vmin, vmax = np.percentile(image, [1, 99])
            if vmax > vmin:
                img_norm = np.clip((image - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            else:
                img_norm = (image / 256).astype(np.uint8)
        else:
            img_norm = image
        
        if len(img_norm.shape) == 2:
            # Grayscale to RGB
            h, w = img_norm.shape
            rgb = np.stack([img_norm] * 3, axis=2)
        else:
            h, w = img_norm.shape[:2]
            rgb = img_norm
        
        # Create QImage
        bytes_per_line = 3 * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to thumbnail size
        thumb_size = QSize(120, 120)
        pixmap = QPixmap.fromImage(q_img).scaled(
            thumb_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Create label with image
        thumb_widget = QWidget()
        thumb_layout = QVBoxLayout()
        thumb_layout.setContentsMargins(2, 2, 2, 2)
        
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_layout.addWidget(img_label)
        
        # Add text
        text = f"B{block_id} {corner}\n{error_um:.3f}µm"
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("QLabel { font-size: 8pt; }")
        thumb_layout.addWidget(text_label)
        
        thumb_widget.setLayout(thumb_layout)
        
        # Add to grid (4 columns)
        row = self.thumbnail_count // 4
        col = self.thumbnail_count % 4
        self.thumbnail_layout.addWidget(thumb_widget, row, col)
        
        self.thumbnail_count += 1
    
    def append_log(self, message: str):
        """Append message to log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def mark_complete(self, success: bool = True):
        """
        Mark operation as complete.
        
        Args:
            success: Whether operation succeeded
        """
        self.is_complete = True
        
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("✅ Alignment Complete!")
            self.status_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; color: green; }")
        else:
            self.status_label.setText("❌ Alignment Failed")
            self.status_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; color: red; }")
        
        # Switch buttons
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.close_button.setDefault(True)
    
    def closeEvent(self, event):
        """Prevent close if not complete."""
        if not self.is_complete:
            # Treat as cancel
            self.reject()
        else:
            event.accept()