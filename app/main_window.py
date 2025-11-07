"""
Main Application Window

Central window containing all UI panels and managing application lifecycle.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QMenuBar, QStatusBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from pathlib import Path

from app.system_state import SystemState
from app.signals import SystemSignals
from app.controllers.camera_stream import CameraStreamThread

# Import widgets (will create these next)
from app.widgets.camera_view import CameraViewWidget
from app.widgets.stage_control import StageControlWidget
from app.widgets.alignment_panel import AlignmentPanelWidget
from app.widgets.block_grid import BlockGridWidget
from app.widgets.waveguide_panel import WaveguidePanelWidget
from app.widgets.status_bar import CustomStatusBar


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Layout:
    ┌────────────────────────────────────────────────┐
    │ Menu Bar                                       │
    ├──────────────────┬─────────────────────────────┤
    │                  │  Stage Control              │
    │  Camera View     │  Alignment Panel            │
    │                  │                             │
    ├──────────────────┴─────────────────────────────┤
    │  Block Grid                                    │
    ├────────────────────────────────────────────────┤
    │  Waveguide Panel                               │
    ├────────────────────────────────────────────────┤
    │  Status Bar                                    │
    └────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        state: SystemState,
        signals: SystemSignals,
        camera,
        stage,
        hw_manager,
        parent=None
    ):
        super().__init__(parent)
        
        self.state = state
        self.signals = signals
        self.camera = camera
        self.stage = stage
        self.hw_manager = hw_manager
        
        # Camera stream thread
        self.camera_thread = None
        
        # Position update timer
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._update_stage_position)
        
        # Setup UI
        self._init_ui()
        self._create_menu_bar()
        self._connect_signals()
        self._start_camera_stream()
        self._start_position_updates()
        
        # Window properties
        self.setWindowTitle(f"Microscope Alignment - {state.hardware_mode.value.upper()} Mode")
        self.resize(1600, 1000)
    
    def _init_ui(self):
        """Initialize UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top section: Camera + Controls (horizontal split)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Camera view
        self.camera_view = CameraViewWidget(self.state, self.signals)
        top_splitter.addWidget(self.camera_view)
        
        # Right: Control panels (tabs)
        control_tabs = QTabWidget()
        
        # Stage control tab
        self.stage_control = StageControlWidget(
            self.state, self.signals, self.stage
        )
        control_tabs.addTab(self.stage_control, "Stage Control")
        
        # Alignment panel tab
        self.alignment_panel = AlignmentPanelWidget(self.state, self.signals)
        control_tabs.addTab(self.alignment_panel, "Alignment")
        
        top_splitter.addWidget(control_tabs)
        top_splitter.setStretchFactor(0, 3)  # Camera gets more space
        top_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(top_splitter, stretch=3)
        
        # Middle section: Block grid
        self.block_grid = BlockGridWidget(self.state, self.signals)
        main_layout.addWidget(self.block_grid, stretch=1)
        
        # Bottom section: Waveguide panel
        self.waveguide_panel = WaveguidePanelWidget(
            self.state, self.signals, self.stage
        )
        main_layout.addWidget(self.waveguide_panel, stretch=2)
        
        # Status bar
        self.status_bar = CustomStatusBar(self.state, self.signals)
        self.setStatusBar(self.status_bar)
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_state_action = QAction("&Open State...", self)
        open_state_action.setShortcut("Ctrl+O")
        open_state_action.triggered.connect(self._open_state)
        file_menu.addAction(open_state_action)
        
        save_state_action = QAction("&Save State...", self)
        save_state_action.setShortcut("Ctrl+S")
        save_state_action.triggered.connect(self._save_state)
        file_menu.addAction(save_state_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Calibration menu
        calib_menu = menubar.addMenu("&Calibration")
        
        run_global_action = QAction("Run &Global Alignment", self)
        run_global_action.triggered.connect(self._run_global_alignment)
        calib_menu.addAction(run_global_action)
        
        calib_block_action = QAction("Calibrate Selected &Block", self)
        calib_block_action.triggered.connect(self._calibrate_selected_block)
        calib_menu.addAction(calib_block_action)
        
        calib_menu.addSeparator()
        
        reset_calib_action = QAction("&Reset All Calibrations", self)
        reset_calib_action.triggered.connect(self._reset_calibrations)
        calib_menu.addAction(reset_calib_action)
        
        # Navigation menu
        nav_menu = menubar.addMenu("&Navigation")
        
        goto_target_action = QAction("Go to &Target Waveguide", self)
        goto_target_action.setShortcut("Ctrl+G")
        goto_target_action.triggered.connect(self._goto_target)
        nav_menu.addAction(goto_target_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_fit_action = QAction("Zoom to &Fit", self)
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.camera_view.zoom_fit)
        view_menu.addAction(zoom_fit_action)
        
        zoom_100_action = QAction("Zoom &100%", self)
        zoom_100_action.setShortcut("Ctrl+1")
        zoom_100_action.triggered.connect(lambda: self.camera_view.set_zoom(1.0))
        view_menu.addAction(zoom_100_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        autofocus_action = QAction("Run &Autofocus", self)
        autofocus_action.setShortcut("Ctrl+F")
        autofocus_action.triggered.connect(self._run_autofocus)
        tools_menu.addAction(autofocus_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self):
        """Connect signals between components."""
        # Status messages
        self.signals.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 3000)
        )
        
        # Errors
        self.signals.error_occurred.connect(self._show_error)
        
        # Block selection
        self.signals.block_selected.connect(self._on_block_selected)
    
    def _start_camera_stream(self):
        """Start camera streaming thread."""
        if self.camera is None:
            print("[MainWindow] No camera available")
            return
        
        self.camera_thread = CameraStreamThread(
            camera=self.camera,
            target_fps=20
        )
        
        # Connect camera thread signals
        self.camera_thread.frame_ready.connect(self.camera_view.update_frame)
        self.camera_thread.stats_updated.connect(self.camera_view.update_stats)
        self.camera_thread.error_occurred.connect(
            lambda msg: self.signals.error_occurred.emit("Camera Error", msg)
        )
        
        # Start thread
        self.camera_thread.start()
        print("[MainWindow] Camera stream started")
    
    def _start_position_updates(self):
        """Start periodic stage position updates."""
        if self.stage is None:
            return
        
        self.position_timer.start(100)  # Update every 100ms
    
    def _update_stage_position(self):
        """Update stage position from hardware."""
        if self.stage is None:
            return
        
        try:
            x = self.stage.get_pos('x')
            y = self.stage.get_pos('y')
            z = self.stage.get_pos('z')
            
            self.state.update_stage_position('x', x)
            self.state.update_stage_position('y', y)
            self.state.update_stage_position('z', z)
            
            # Emit signals
            self.signals.stage_position_changed.emit('x', x)
            self.signals.stage_position_changed.emit('y', y)
            self.signals.stage_position_changed.emit('z', z)
        except Exception as e:
            # Silently ignore position read errors (too noisy)
            pass
    
    # ========================================================================
    # Menu Actions
    # ========================================================================
    
    def _open_state(self):
        """Open saved state file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open State",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                self.state.load_from_file(filename)
                self.signals.state_loaded.emit(filename)
                self.signals.status_message.emit(f"Loaded state from {Path(filename).name}")
            except Exception as e:
                self.signals.error_occurred.emit(
                    "Failed to Load State",
                    str(e)
                )
    
    def _save_state(self):
        """Save current state to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save State",
            "alignment_state.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                self.state.save_to_file(filename)
                self.signals.state_saved.emit(filename)
                self.signals.status_message.emit(f"Saved state to {Path(filename).name}")
            except Exception as e:
                self.signals.error_occurred.emit(
                    "Failed to Save State",
                    str(e)
                )
    
    def _run_global_alignment(self):
        """Trigger global alignment (stub)."""
        self.signals.status_message.emit("Global alignment not yet implemented")
        # TODO: Implement alignment controller
    
    def _calibrate_selected_block(self):
        """Calibrate currently selected block (stub)."""
        if self.state.navigation.current_block is None:
            self.signals.error_occurred.emit(
                "No Block Selected",
                "Please select a block first"
            )
            return
        
        self.signals.status_message.emit("Block calibration not yet implemented")
        # TODO: Implement block calibration
    
    def _reset_calibrations(self):
        """Reset all calibrations."""
        reply = QMessageBox.question(
            self,
            "Reset Calibrations",
            "This will reset all alignment calibrations. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.state.global_calibrated = False
            self.state.global_calibration_params = None
            for block in self.state.blocks.values():
                block.status = self.state.blocks[1].status.NOT_CALIBRATED
                block.calibration_error = None
            self.signals.state_reset.emit()
            self.signals.status_message.emit("All calibrations reset")
    
    def _goto_target(self):
        """Navigate to target waveguide."""
        self.waveguide_panel.navigate_to_target()
    
    def _run_autofocus(self):
        """Run autofocus (stub)."""
        self.signals.status_message.emit("Autofocus not yet implemented")
        # TODO: Implement autofocus controller
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Microscope Alignment",
            "<h3>Microscope Alignment System</h3>"
            "<p>Version 1.0.0</p>"
            "<p>A PyQt6 application for automated microscope alignment and navigation.</p>"
            "<p><b>Hardware Mode:</b> " + self.state.hardware_mode.value.upper() + "</p>"
        )
    
    # ========================================================================
    # Signal Handlers
    # ========================================================================
    
    def _on_block_selected(self, block_id: int):
        """Handle block selection."""
        self.state.set_current_block(block_id)
        self.waveguide_panel.refresh_waveguide_list()
    
    def _show_error(self, title: str, message: str):
        """Show error message box."""
        QMessageBox.critical(self, title, message)
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.signals.application_closing.emit()
            self.cleanup()
            event.accept()
        else:
            event.ignore()
    
    def cleanup(self):
        """Cleanup resources before exit."""
        print("[MainWindow] Cleaning up...")
        
        # Stop camera thread
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        
        # Stop position timer
        self.position_timer.stop()
        
        print("[MainWindow] Cleanup complete")