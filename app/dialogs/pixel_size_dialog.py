# app/widgets/pixel_size_dialog.py
"""
Pixel Size Dialog - Simple view and edit

Shows current pixel size and allows editing with full precision.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QGroupBox
)
from PyQt6.QtGui import QDoubleValidator


class PixelSizeDialog(QDialog):
    """Simple dialog to view/edit pixel size (µm/pixel)."""
    
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        
        self.setWindowTitle("Camera Pixel Size")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Info
        info = QLabel(
            "<b>Camera Pixel Size Configuration</b>\n\n"
            "This value is used to convert pixel measurements to micrometers.\n"
            "It depends on your camera sensor and optical magnification."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Current value group
        group = QGroupBox("Pixel Size")
        group_layout = QVBoxLayout()
        
        # Display current value
        current_label = QLabel("Current value:")
        group_layout.addWidget(current_label)
        
        self.current_value_label = QLabel()
        self.current_value_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 14pt; "
            "font-weight: bold; color: #2196F3; padding: 5px; }"
        )
        self._update_current_display()
        group_layout.addWidget(self.current_value_label)
        
        # Edit field
        edit_label = QLabel("New value (µm/pixel):")
        group_layout.addWidget(edit_label)
        
        self.edit_field = QLineEdit()
        self.edit_field.setStyleSheet(
            "QLineEdit { font-family: monospace; font-size: 12pt; padding: 5px; }"
        )
        
        # Set validator (allow positive floats with many decimals)
        validator = QDoubleValidator(0.0, 100.0, 10)  # Up to 10 decimal places
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.edit_field.setValidator(validator)
        
        # Initialize with current value (trimmed to last non-zero digit)
        initial_text = self._format_initial_value(self.state.camera.um_per_pixel)
        self.edit_field.setText(initial_text)
        self.edit_field.selectAll()  # Select all for easy editing
        
        group_layout.addWidget(self.edit_field)
        
        # Helper text
        helper = QLabel(
            "💡 Tip: Enter value with full precision (e.g., 0.00002 or 0.3)\n"
            "The value will not be rounded."
        )
        helper.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        helper.setWordWrap(True)
        group_layout.addWidget(helper)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch()
        
        btn_apply = QPushButton("✅ Apply")
        btn_apply.clicked.connect(self._apply_value)
        btn_apply.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        buttons.addWidget(btn_apply)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_cancel)
        
        layout.addLayout(buttons)
        self.setLayout(layout)
        
        # Focus on edit field
        self.edit_field.setFocus()
    
    def _format_initial_value(self, value: float) -> str:
        """
        Format value for initial display, trimming trailing zeros but keeping precision.
        
        Examples:
            0.30000 -> "0.3"
            0.00002 -> "0.00002"
            0.123456 -> "0.123456"
            1.50000 -> "1.5"
        """
        # Convert to string with high precision
        text = f"{value:.10f}"
        
        # Remove trailing zeros after decimal point
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        
        # If we ended up with just "0", show at least one decimal
        if text == "0":
            text = "0.0"
        
        return text
    
    def _update_current_display(self):
        """Update the current value display label."""
        value = self.state.camera.um_per_pixel
        
        # Show full precision without scientific notation
        if value < 0.001:
            # Very small values - show many decimals
            text = f"{value:.10f}".rstrip('0').rstrip('.')
        else:
            # Normal values
            text = f"{value:.6f}".rstrip('0').rstrip('.')
        
        self.current_value_label.setText(f"{text} µm/pixel")
    
    def _apply_value(self):
        """Apply the new pixel size value."""
        text = self.edit_field.text().strip()
        
        if not text:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a pixel size value."
            )
            return
        
        try:
            new_value = float(text)
            
            if new_value <= 0:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Invalid Value",
                    "Pixel size must be greater than zero."
                )
                return
            
            # Apply to state
            old_value = self.state.camera.um_per_pixel
            self.state.camera.um_per_pixel = new_value
            
            print(f"[PixelSizeDialog] Pixel size updated: {old_value} → {new_value} µm/pixel")
            
            self.accept()
            
        except ValueError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid number."
            )


# Example usage in main_window.py menu:
# 
# tools_menu.addSeparator()
# 
# set_pixel_size_action = QAction("Set &Pixel Size...", self)
# set_pixel_size_action.triggered.connect(self._set_pixel_size)
# tools_menu.addAction(set_pixel_size_action)
#
# def _set_pixel_size(self):
#     """Open pixel size dialog."""
#     from app.widgets.pixel_size_dialog import PixelSizeDialog
#     
#     dialog = PixelSizeDialog(
#         state=self.state,
#         parent=self
#     )
#     
#     if dialog.exec() == QDialog.DialogCode.Accepted:
#         self.signals.status_message.emit(
#             f"Pixel size set to {self.state.camera.um_per_pixel} µm/pixel"
#         )