#!/usr/bin/env python3
"""
Microscope Alignment GUI - Main Entry Point

Launch the PyQt6 application with hardware initialization.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QLabel, QPushButton, QRadioButton, QButtonGroup
from PyQt6.QtCore import Qt

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.main_window import MainWindow
from app.system_state import SystemState, HardwareMode
from app.signals import SystemSignals
from app.controllers.hardware_manager import HardwareManager


class HardwareSelectionDialog(QDialog):
    """Dialog for selecting hardware mode at startup."""
    
    def __init__(self, availability: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Selection")
        self.selected_mode = None
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>Select Hardware Mode</h2>")
        layout.addWidget(title)
        
        # Hardware availability status
        status_text = "<b>Hardware Detection:</b><br>"
        status_text += f"• Real Camera: {'✓ Available' if availability['real_camera'] else '✗ Not found'}<br>"
        status_text += f"• Real Stage: {'✓ Available' if availability['real_stage'] else '✗ Not found'}<br>"
        status_text += f"• Mock Hardware: ✓ Always available"
        
        status_label = QLabel(status_text)
        layout.addWidget(status_label)
        
        layout.addSpacing(20)
        
        # Radio buttons
        self.button_group = QButtonGroup()
        
        self.mock_radio = QRadioButton("Mock Hardware (Simulation)")
        self.mock_radio.setToolTip("Use simulated camera and stage for testing")
        self.button_group.addButton(self.mock_radio)
        layout.addWidget(self.mock_radio)
        
        self.real_radio = QRadioButton("Real Hardware (Zyla + SmarAct)")
        self.real_radio.setToolTip("Connect to actual laboratory hardware")
        self.real_radio.setEnabled(availability['real_camera'] or availability['real_stage'])
        self.button_group.addButton(self.real_radio)
        layout.addWidget(self.real_radio)
        
        # Default selection
        if availability['real_camera'] and availability['real_stage']:
            self.real_radio.setChecked(True)
        else:
            self.mock_radio.setChecked(True)
        
        layout.addSpacing(20)
        
        # Buttons
        ok_button = QPushButton("Continue")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)
        
        self.setLayout(layout)
        self.setMinimumWidth(400)
    
    def get_selected_mode(self) -> str:
        """Get selected hardware mode ('mock' or 'real')."""
        if self.mock_radio.isChecked():
            return "mock"
        elif self.real_radio.isChecked():
            return "real"
        return "mock"


def main():
    """Main application entry point."""
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Microscope Alignment")
    app.setOrganizationName("YourLab")
    
    # Set application style
    app.setStyle("Fusion")
    
    print("="*70)
    print("Microscope Alignment GUI")
    print("="*70)
    
    # Initialize hardware manager
    layout_path = project_root / "config" / "mock_layout.json"
    hw_manager = HardwareManager(layout_path=str(layout_path))
    
    # Get hardware availability
    availability = hw_manager.get_hardware_availability()
    print(f"\nHardware Detection:")
    print(f"  Real Camera: {availability['real_camera']}")
    print(f"  Real Stage: {availability['real_stage']}")
    
    # Show hardware selection dialog
    dialog = HardwareSelectionDialog(availability)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        print("\nStartup cancelled by user")
        return 0
    
    selected_mode = dialog.get_selected_mode()
    print(f"\nSelected mode: {selected_mode.upper()}")
    
    # Initialize hardware
    print("Initializing hardware...")
    if selected_mode == "mock":
        success, message = hw_manager.initialize_mock_hardware()
    else:
        success, message = hw_manager.initialize_real_hardware()
    
    if not success:
        QMessageBox.critical(
            None,
            "Hardware Initialization Failed",
            f"Failed to initialize hardware:\n\n{message}\n\nApplication will exit."
        )
        print(f"\nERROR: {message}")
        return 1
    
    print(f"✓ {message}")
    
    # Create system state and signals
    state = SystemState()
    state.hardware_mode = HardwareMode.MOCK if selected_mode == "mock" else HardwareMode.REAL
    state.camera_connected = True
    state.stage_connected = True
    
    signals = SystemSignals()
    
    # Create main window
    print("\nLaunching GUI...")
    main_window = MainWindow(
        state=state,
        signals=signals,
        camera=hw_manager.get_camera(),
        stage=hw_manager.get_stage(),
        hw_manager=hw_manager
    )
    
    main_window.show()
    
    print("✓ GUI launched successfully")
    print("\nApplication running. Close window to exit.")
    print("="*70)
    
    # Run application
    exit_code = app.exec()
    
    # Cleanup
    print("\nShutting down...")
    main_window.cleanup()
    hw_manager.shutdown()
    print("✓ Shutdown complete")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())