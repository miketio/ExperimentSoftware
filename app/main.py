#!/usr/bin/env python3
"""
Microscope Alignment GUI - Main Entry Point

Three operation modes:
  1. Full Experiment  – camera + stages + alignment
  2. Mock / Testing   – simulated camera + stage
  3. Stage Control    – real stages only, no camera, no alignment
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QRadioButton, QButtonGroup, QPushButton
)
from PyQt6.QtCore import Qt
import json

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.system_state import SystemState, HardwareMode
from app.signals import SystemSignals
from app.controllers.hardware_manager import HardwareManager


# ---------------------------------------------------------------------------
# Startup dialog
# ---------------------------------------------------------------------------

class ModeSelectionDialog(QDialog):
    """Pick one of three operation modes at launch."""

    def __init__(self, availability: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Selection")
        self.setMinimumWidth(400)
        self._build_ui(availability)

    def _build_ui(self, av: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("<h2>Select Operation Mode</h2>")
        layout.addWidget(title)

        # Hardware detection status
        status_text = "<b>Hardware Detection:</b><br>"
        status_text += f"• Real Camera: {'✓ Available' if av['real_camera'] else '✗ Not found'}<br>"
        status_text += f"• Real Stage: {'✓ Available' if av['real_stage'] else '✗ Not found'}<br>"
        status_text += f"• Mock Hardware: ✓ Always available"

        status_label = QLabel(status_text)
        layout.addWidget(status_label)

        layout.addSpacing(10)

        # Radio buttons
        self._group = QButtonGroup(self)

        self._full_radio = QRadioButton("Full Experiment Mode (Camera + Stage + Alignment)")
        self._full_radio.setEnabled(av["real_camera"] or av["real_stage"])
        self._group.addButton(self._full_radio, 0)
        layout.addWidget(self._full_radio)

        self._mock_radio = QRadioButton("Mock / Testing Mode (Simulated Hardware)")
        self._group.addButton(self._mock_radio, 1)
        layout.addWidget(self._mock_radio)

        self._stage_radio = QRadioButton("Stage Control Mode (Stages Only, No Camera)")
        self._stage_radio.setEnabled(av["real_stage"])
        self._group.addButton(self._stage_radio, 2)
        layout.addWidget(self._stage_radio)

        # Smart default
        if av["real_camera"] and av["real_stage"]:
            self._full_radio.setChecked(True)
        elif av["real_stage"]:
            self._stage_radio.setChecked(True)
        else:
            self._mock_radio.setChecked(True)

        layout.addSpacing(10)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        ok_button = QPushButton("Continue")
        ok_button.setDefault(True)
        ok_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; min-width: 100px; }"
        )
        ok_button.clicked.connect(self.accept)
        btn_row.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        btn_row.addWidget(cancel_button)

        layout.addLayout(btn_row)

    def selected_mode(self) -> int:
        """Return 0=full, 1=mock, 2=stage_only."""
        return self._group.checkedId()


# ---------------------------------------------------------------------------
# Stage-only launch path
# ---------------------------------------------------------------------------

def _launch_stage_only(hw_manager: HardwareManager):
    """Initialize hardware and open StageOnlyWindow."""
    success, msg = hw_manager.initialize_stage_only_hardware()
    if not success:
        QMessageBox.critical(None, "Stage Init Failed", f"{msg}\n\nApplication will exit.")
        return None

    print(f"✅ {msg}")

    from app.system_state import SystemState, HardwareMode
    from app.signals import SystemSignals
    from app.stage_only_window import StageOnlyWindow

    state = SystemState()
    state.hardware_mode = HardwareMode.REAL
    state.stage_connected = True

    signals = SystemSignals()

    window = StageOnlyWindow(
        state=state,
        signals=signals,
        stage=hw_manager.get_stage(),
        filter_stage=hw_manager.get_filter_stage(),
    )
    window.show()
    return window


# ---------------------------------------------------------------------------
# Original full / mock launch path
# ---------------------------------------------------------------------------

def _launch_full_or_mock(app, hw_manager: HardwareManager, selected_mode: int):
    """Run the original camera+stage launch flow. Returns exit code."""
    from PyQt6.QtWidgets import QDialog
    from app.main_window import MainWindow

    # Hardware init
    if selected_mode == 1:  # mock
        temp_layout_path = "config/mock_layout.json"
        if Path(temp_layout_path).exists():
            hw_manager.layout_path = temp_layout_path
        success, message = hw_manager.initialize_mock_hardware()
    else:  # real
        success, message = hw_manager.initialize_real_hardware()

    if not success:
        QMessageBox.critical(
            None, "Hardware Initialization Failed",
            f"Failed to initialize hardware:\n\n{message}\n\nApplication will exit."
        )
        return 1

    print(f"✅ {message}")

    state = SystemState()
    state.hardware_mode = HardwareMode.MOCK if selected_mode == 1 else HardwareMode.REAL
    state.camera_connected = True
    state.stage_connected = True

    camera = hw_manager.get_camera()
    if hasattr(camera, "um_per_pixel"):
        state.camera.um_per_pixel = camera.um_per_pixel
        print(f"✅ Camera pixel size: {camera.um_per_pixel:.3f} µm/pixel")

    signals = SystemSignals()

    # Layout selection
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
        print(f"Loading layout from: {layout_file}")
        try:
            from config.layout_models import RuntimeLayout
            runtime_layout = RuntimeLayout.from_json_file(layout_file)
            runtime_layout.metadata["source_file"] = layout_file
            print(f"✅ Loaded design: {runtime_layout.design_name}")

            if Path(RUNTIME_FILE).exists():
                try:
                    with open(RUNTIME_FILE) as f:
                        runtime_data = json.load(f)
                    if "block_1_stage_position_um" in runtime_data:
                        pos = runtime_data["block_1_stage_position_um"]
                        runtime_layout.set_block_1_position(pos[0], pos[1])
                        print(f"  ✅ Block 1 position: Y={pos[0]:.3f}, Z={pos[1]:.3f} µm")
                    if "measured_calibration" in runtime_data:
                        runtime_layout._load_measured_calibration(
                            runtime_data["measured_calibration"]
                        )
                        print("  ✅ Loaded calibration data")
                except Exception as e:
                    print(f"  ⚠️ Failed to load runtime state: {e}")

            if not runtime_layout.has_block_1_position():
                print("\n⚠️ Block 1 position not set — defaulting to (0, 0)")
                runtime_layout.set_block_1_position(0.0, 0.0)

        except Exception as e:
            QMessageBox.critical(
                None, "Layout Load Failed",
                f"Failed to load layout:\n\n{e}\n\nApplication will exit."
            )
            hw_manager.shutdown()
            return 1

    elif layout_mode == "wizard":
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
                None, "Layout Creation Failed",
                "Failed to create layout from wizard.\n\nApplication will exit."
            )
            hw_manager.shutdown()
            return 1

        print(f"✅ Created layout: {runtime_layout.design_name}")

    else:
        print(f"ERROR: Unknown layout mode: {layout_mode}")
        hw_manager.shutdown()
        return 1

    # Launch main window
    print("\nLaunching GUI...")
    main_window = MainWindow(
        state=state,
        signals=signals,
        camera=hw_manager.get_camera(),
        stage=hw_manager.get_stage(),
        hw_manager=hw_manager,
        runtime_layout=runtime_layout,
    )
    main_window.runtime_file_path = RUNTIME_FILE
    main_window.show()

    print("✅ GUI launched successfully")
    print("\nApplication running. Close window to exit.")
    print("=" * 70)

    exit_code = app.exec()

    print("\nShutting down...")
    main_window.cleanup()
    hw_manager.shutdown()
    print("✅ Shutdown complete")
    return exit_code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Microscope Control")
    app.setOrganizationName("YourLab")
    app.setStyle("Fusion")

    print("=" * 70)
    print("Microscope Control — Startup")
    print("=" * 70)

    hw_manager = HardwareManager(layout_path=None)
    availability = hw_manager.get_hardware_availability()
    print(f"\nHardware Detection:")
    print(f"  Real Camera: {availability['real_camera']}")
    print(f"  Real Stage:  {availability['real_stage']}")

    dialog = ModeSelectionDialog(availability)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        print("\nStartup cancelled by user")
        return 0

    mode = dialog.selected_mode()
    mode_names = {0: "FULL EXPERIMENT", 1: "MOCK / TESTING", 2: "STAGE CONTROL"}
    print(f"\nSelected mode: {mode_names[mode]}")

    if mode == 2:
        # Stage-only path — no layout dialog, no camera
        window = _launch_stage_only(hw_manager)
        if window is None:
            return 1
        exit_code = app.exec()
        print("\nShutting down...")
        hw_manager.shutdown()
        print("✅ Shutdown complete")
        return exit_code
    else:
        return _launch_full_or_mock(app, hw_manager, mode)


if __name__ == "__main__":
    sys.exit(main())