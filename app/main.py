#!/usr/bin/env python3
"""
Microscope Alignment GUI - Main Entry Point

Launch the PyQt6 application with hardware initialization.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog
from PyQt6.QtCore import Qt
import json

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
        from PyQt6.QtWidgets import QVBoxLayout, QLabel, QRadioButton, QButtonGroup, QPushButton
        
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
    
    # Initialize hardware manager (without layout initially)
    hw_manager = HardwareManager(layout_path=None)
    
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
    
    # Initialize hardware BEFORE layout (so we can capture stage position)
    print("Initializing hardware...")
    if selected_mode == "mock":
        # For mock mode, we need a temporary layout for camera
        temp_layout_path = "config/mock_layout.json"
        if Path(temp_layout_path).exists():
            hw_manager.layout_path = temp_layout_path
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
    
    print(f"✅ {message}")
    
    # Create system state and signals
    state = SystemState()
    state.hardware_mode = HardwareMode.MOCK if selected_mode == "mock" else HardwareMode.REAL
    state.camera_connected = True
    state.stage_connected = True
    
    # Get um_per_pixel from camera (standardized)
    camera = hw_manager.get_camera()
    if hasattr(camera, 'um_per_pixel'):
        state.camera.um_per_pixel = camera.um_per_pixel
        print(f"✅ Camera pixel size: {camera.um_per_pixel:.3f} µm/pixel")
    
    signals = SystemSignals()
    
    # ========================================================================
    # LAYOUT SELECTION - NEW DIALOG
    # ========================================================================
    
    from app.dialogs.layout_selection_dialog import LayoutSelectionDialog
    
    layout_dialog = LayoutSelectionDialog()
    if layout_dialog.exec() != QDialog.DialogCode.Accepted:
        print("\nLayout selection cancelled by user")
        hw_manager.shutdown()
        return 0
    
    layout_mode, layout_file = layout_dialog.get_selection()
    print(f"\nLayout mode: {layout_mode}")
    
    runtime_layout = None
    RUNTIME_FILE = "config/runtime_state.json"
    
    if layout_mode == "existing":
        # Load from existing file
        print(f"Loading layout from: {layout_file}")
        
        try:
            from config.layout_models import RuntimeLayout
            runtime_layout = RuntimeLayout.from_json_file(layout_file)
            runtime_layout.metadata['source_file'] = layout_file
            print(f"✅ Loaded design: {runtime_layout.design_name}")
            
            # Load runtime state if exists
            if Path(RUNTIME_FILE).exists():
                print(f"\nFound runtime state: {RUNTIME_FILE}")
                try:
                    with open(RUNTIME_FILE, 'r') as f:
                        runtime_data = json.load(f)
                    
                    # Apply Block 1 position
                    if 'block_1_stage_position_um' in runtime_data:
                        pos = runtime_data['block_1_stage_position_um']
                        runtime_layout.set_block_1_position(pos[0], pos[1])
                        print(f"  ✅ Block 1 position: Y={pos[0]:.3f}, Z={pos[1]:.3f} µm")
                    
                    # Apply calibration
                    if 'measured_calibration' in runtime_data:
                        runtime_layout._load_measured_calibration(runtime_data['measured_calibration'])
                        print(f"  ✅ Loaded calibration data")
                
                except Exception as e:
                    print(f"  ⚠️ Failed to load runtime state: {e}")
            
            # Default Block 1 position if not set
            if not runtime_layout.has_block_1_position():
                print("\n⚠️ Block 1 position not set - defaulting to (0, 0)")
                print("   Use menu: Setup > Set Block 1 Position")
                runtime_layout.set_block_1_position(0.0, 0.0)
        
        except Exception as e:
            QMessageBox.critical(
                None,
                "Layout Load Failed",
                f"Failed to load layout:\n\n{e}\n\nApplication will exit."
            )
            print(f"ERROR: {e}")
            hw_manager.shutdown()
            return 1
    
    elif layout_mode == "wizard":
        # Launch wizard
        print("Launching Layout Wizard...")
        
        from app.widgets.layout_wizard import LayoutWizard
        
        wizard = LayoutWizard(state)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            print("\nWizard cancelled by user")
            hw_manager.shutdown()
            return 0
        
        runtime_layout = wizard.get_runtime_layout()
        
        if runtime_layout is None:
            QMessageBox.critical(
                None,
                "Layout Creation Failed",
                "Failed to create layout from wizard.\n\nApplication will exit."
            )
            hw_manager.shutdown()
            return 1
        
        print(f"✅ Created layout: {runtime_layout.design_name}")
    
    else:
        print(f"ERROR: Unknown layout mode: {layout_mode}")
        hw_manager.shutdown()
        return 1
    
    # ========================================================================
    # LAUNCH MAIN WINDOW
    # ========================================================================
    
    print("\nLaunching GUI...")
    main_window = MainWindow(
        state=state,
        signals=signals,
        camera=hw_manager.get_camera(),
        stage=hw_manager.get_stage(),
        hw_manager=hw_manager,
        runtime_layout=runtime_layout
    )
    
    # Store runtime file path for saving
    main_window.runtime_file_path = RUNTIME_FILE
    main_window.show()
    
    print("✅ GUI launched successfully")
    print("\nApplication running. Close window to exit.")
    print("="*70)
    
    # Run application
    exit_code = app.exec()
    
    # Cleanup
    print("\nShutting down...")
    main_window.cleanup()
    hw_manager.shutdown()
    print("✅ Shutdown complete")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())