# app/dialogs/layout_selection_dialog.py
"""
Layout Selection Dialog - Choose between existing layout or create new
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QFileDialog, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from pathlib import Path
import json


class LayoutSelectionDialog(QDialog):
    """Dialog for selecting layout source at startup."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layout Selection")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.selected_mode = None  # "existing" or "wizard"
        self.layout_file = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("<h2>Select Layout Source</h2>")
        layout.addWidget(title)
        
        info = QLabel(
            "Choose how to define the chip layout:\n"
            "• Load existing layout file (JSON)\n"
            "• Create new layout using wizard"
        )
        info.setStyleSheet("QLabel { color: #666; background-color: #F0F0F0; padding: 10px; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        layout.addSpacing(20)
        
        # Radio buttons
        self.button_group = QButtonGroup()
        
        # Option 1: Load existing
        self.existing_radio = QRadioButton("Load Existing Layout File")
        self.existing_radio.setChecked(True)
        self.existing_radio.toggled.connect(self._on_mode_changed)
        self.button_group.addButton(self.existing_radio)
        layout.addWidget(self.existing_radio)
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addSpacing(30)
        
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select layout file...")
        self.file_path.setReadOnly(True)
        file_layout.addWidget(self.file_path)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_file)
        file_layout.addWidget(self.btn_browse)
        
        layout.addLayout(file_layout)
        
        layout.addSpacing(10)
        
        # Option 2: Create new
        self.wizard_radio = QRadioButton("Create New Layout (Wizard)")
        self.wizard_radio.toggled.connect(self._on_mode_changed)
        self.button_group.addButton(self.wizard_radio)
        layout.addWidget(self.wizard_radio)
        
        wizard_info = QLabel(
            "The wizard will guide you through:\n"
            "  1. Block array configuration\n"
            "  2. ASCII file assignment\n"
            "  3. Block 1 position setup"
        )
        wizard_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; margin-left: 30px; }")
        wizard_info.setWordWrap(True)
        layout.addWidget(wizard_info)
        
        layout.addSpacing(30)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_button = QPushButton("Continue")
        ok_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; min-width: 100px; }"
        )
        ok_button.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        # Check for default layout
        default_layout = Path("config/mock_layout.json")
        if default_layout.exists():
            self.file_path.setText(str(default_layout))
    
    def _on_mode_changed(self):
        """Handle mode radio button change."""
        is_existing = self.existing_radio.isChecked()
        self.file_path.setEnabled(is_existing)
        self.btn_browse.setEnabled(is_existing)
    
    def _browse_file(self):
        """Browse for layout file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Layout File",
            "config",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            self.file_path.setText(filename)
    
    def _on_ok(self):
        """Validate and accept."""
        if self.existing_radio.isChecked():
            # Validate file exists
            filepath = self.file_path.text()
            if not filepath:
                QMessageBox.warning(
                    self,
                    "No File Selected",
                    "Please select a layout file or choose the wizard option."
                )
                return
            
            if not Path(filepath).exists():
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"Layout file not found:\n{filepath}"
                )
                return
            
            self.selected_mode = "existing"
            self.layout_file = filepath
        else:
            self.selected_mode = "wizard"
            self.layout_file = None
        
        self.accept()
    
    def get_selection(self):
        """
        Get selected mode and file.
        
        Returns:
            tuple: (mode, filepath) where mode is "existing" or "wizard"
        """
        return (self.selected_mode, self.layout_file)