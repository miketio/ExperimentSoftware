from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, 
    QDialogButtonBox, QFormLayout
)
from PyQt6.QtCore import Qt

class PixelSizeDialog(QDialog):
    """
    Dialog to configure camera pixel size (µm/pixel).
    Accepts a float value, returns a float value.
    """
    
    def __init__(self, current_size: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Size Configuration")
        self.setMinimumWidth(300)
        
        # Store the initial value passed from MainWindow
        self.current_size = float(current_size)
        
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Info text
        info = QLabel(
            "Enter the calibrated pixel size.\n"
            "This value is used to convert pixels to micrometers."
        )
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)
        
        # Form layout for input
        form = QFormLayout()
        
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setRange(0.000001, 100.0)  # Wide range
        self.spin_box.setDecimals(6)             # High precision
        self.spin_box.setSingleStep(0.01)
        self.spin_box.setSuffix(" µm/px")
        
        # Set the value passed in __init__
        self.spin_box.setValue(self.current_size)
        
        form.addRow("Pixel Size:", self.spin_box)
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_value(self) -> float:
        """Get the value from the spinbox."""
        return self.spin_box.value()