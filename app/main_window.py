"""
Main Application Window

Central window containing all UI panels and managing application lifecycle.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QMenuBar, QStatusBar, QMessageBox, QFileDialog, QDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from pathlib import Path

from app.system_state import SystemState
from app.signals import SystemSignals
from app.controllers.camera_stream import CameraStreamThread

# Import widgets (will create these next)
from app.widgets.camera_view import CameraViewWidget
from app.widgets.stage_control import StageControlWidget
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
        runtime_layout,
        parent=None
    ):
        super().__init__(parent)
        
        self.state = state
        self.signals = signals
        self.camera = camera
        self.stage = stage
        self.hw_manager = hw_manager
        self.runtime_layout = runtime_layout
        
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

        # Create alignment controller FIRST (needed by alignment_panel)
        from app.controllers.alignment_controller import AlignmentController
        self.alignment_controller = AlignmentController(
            state=self.state,
            signals=self.signals,
            camera=self.camera,
            stage=self.stage,
            runtime_layout=self.runtime_layout  # Added this parameter
        )
        # Add to imports at top of file
        from app.controllers.autofocus_controller import AutofocusController
        self.autofocus_controller = AutofocusController(
            camera=self.camera,  
            stage=self.stage,
            signals=self.signals
        )
        # Create navigation controller
        from app.controllers.navigation_controller import NavigationController
        self.navigation_controller = NavigationController(
            state=self.state,
            signals=self.signals,
            stage=self.stage,
            alignment_system=self.alignment_controller.alignment_system,  # Pass HierarchicalAlignment
            autofocus_controller=self.autofocus_controller
        )

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

        # Automated alignment tab (NEW)
        from app.widgets.automated_alignment_panel import AutomatedAlignmentPanel
        self.automated_alignment = AutomatedAlignmentPanel(
            self.state, self.signals, self.alignment_controller
        )
        control_tabs.addTab(self.automated_alignment, "Automated Alignment")

        # Manual alignment tab (NEW)
        from app.widgets.manual_calibration_panel import ManualCalibrationPanel
        self.manual_alignment = ManualCalibrationPanel(
            self.state, self.signals, self.runtime_layout, 
            self.alignment_controller
        )
        control_tabs.addTab(self.manual_alignment, "Manual Calibration")

        # Setup panel tab (NEW)
        from app.widgets.setup_panel import SetupPanelWidget
        self.setup_panel = SetupPanelWidget(
            self.state, self.signals, self.runtime_layout, self.autofocus_controller
        )
        control_tabs.addTab(self.setup_panel, "Setup")

        top_splitter.addWidget(control_tabs)
        top_splitter.setStretchFactor(0, 3)  # Camera gets more space
        top_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(top_splitter, stretch=3)
        
        # Middle section: Block grid
        self.block_grid = BlockGridWidget(self.state, self.signals)
        main_layout.addWidget(self.block_grid, stretch=1)
        
        # Bottom section: Waveguide panel
        self.waveguide_panel = WaveguidePanelWidget(
            self.state, self.signals, self.stage, self.navigation_controller
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
        
        # NEW: Set Block 1 Position (with camera view)
        # set_block1_action = QAction("Set &Block 1 Position...", self)
        # set_block1_action.setShortcut("Ctrl+B")
        # set_block1_action.triggered.connect(self._set_block1_position)
        # calib_menu.addAction(set_block1_action)
        
        calib_menu.addSeparator()
        
        run_global_action = QAction("Run &Global Alignment", self)
        run_global_action.triggered.connect(self._run_global_alignment)
        calib_menu.addAction(run_global_action)
        
        calib_block_action = QAction("Calibrate Selected &Block", self)
        calib_block_action.triggered.connect(self._calibrate_selected_block)
        calib_menu.addAction(calib_block_action)
        
        calib_menu.addSeparator()
        
        # NEW: Manual fiducial capture
        manual_fiducial_action = QAction("&Manual Fiducial Capture...", self)
        manual_fiducial_action.setShortcut("Ctrl+M")
        manual_fiducial_action.triggered.connect(self._manual_fiducial_capture)
        calib_menu.addAction(manual_fiducial_action)

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
        
        view_menu = menubar.addMenu("&View")
        
        # Colormap submenu
        colormap_menu = view_menu.addMenu("&Colormap")
        
        # Create action group for colormap (mutually exclusive)
        colormap_group = QActionGroup(self)
        colormap_group.setExclusive(True)
        
        colormaps = ['gray', 'jet', 'hot', 'viridis', 'plasma', 'inferno', 'turbo', 'rainbow']
        for cmap in colormaps:
            action = QAction(cmap.capitalize(), self)
            action.setCheckable(True)
            action.setData(cmap)
            action.triggered.connect(lambda checked, c=cmap: self._set_colormap(c))
            colormap_group.addAction(action)
            colormap_menu.addAction(action)
            
            # Set gray as default checked
            if cmap == self.state.camera.colormap:
                action.setChecked(True)
        
        # Invert colors option
        view_menu.addSeparator()
        self.invert_action = QAction("&Invert Colors", self)
        self.invert_action.setCheckable(True)
        self.invert_action.setChecked(False)
        self.invert_action.triggered.connect(self._toggle_invert)
        view_menu.addAction(self.invert_action)
        

        view_menu.addSeparator()

        # Fourier transform option
        self.fourier_action = QAction("Show &Fourier Transform", self)
        self.fourier_action.setCheckable(True)
        self.fourier_action.setChecked(False)
        self.fourier_action.setShortcut("Ctrl+F")
        self.fourier_action.triggered.connect(self._toggle_fourier)
        view_menu.addAction(self.fourier_action)


        view_menu.addSeparator()
        
        # Zoom controls
        zoom_fit_action = QAction("Zoom to &Fit", self)
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.camera_view.zoom_fit)
        view_menu.addAction(zoom_fit_action)
        
        zoom_100_action = QAction("Zoom &100%", self)
        zoom_100_action.setShortcut("Ctrl+1")
        zoom_100_action.triggered.connect(lambda: self.camera_view.set_zoom(1.0))
        view_menu.addAction(zoom_100_action)
        
        zoom_200_action = QAction("Zoom &200%", self)
        zoom_200_action.setShortcut("Ctrl+2")
        zoom_200_action.triggered.connect(lambda: self.camera_view.set_zoom(2.0))
        view_menu.addAction(zoom_200_action)
        
        zoom_50_action = QAction("Zoom &400%", self)
        zoom_50_action.triggered.connect(lambda: self.camera_view.set_zoom(4.0))
        view_menu.addAction(zoom_50_action)
        
        view_menu.addSeparator()
        
        # Display options
        self.crosshair_action = QAction("Show &Crosshair", self)
        self.crosshair_action.setCheckable(True)
        self.crosshair_action.setChecked(self.state.camera.show_crosshair)
        self.crosshair_action.triggered.connect(self._toggle_crosshair)
        view_menu.addAction(self.crosshair_action)
        
        self.scalebar_action = QAction("Show &Scale Bar", self)
        self.scalebar_action.setCheckable(True)
        self.scalebar_action.setChecked(self.state.camera.show_scale_bar)
        self.scalebar_action.triggered.connect(self._toggle_scalebar)
        view_menu.addAction(self.scalebar_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # autofocus_action = QAction("Run &Autofocus", self)
        # autofocus_action.setShortcut("Ctrl+Shift+F")
        # autofocus_action.triggered.connect(self._run_autofocus)
        # tools_menu.addAction(autofocus_action)
        tools_menu.addSeparator()

        set_beam_action = QAction("Set &Beam Position...", self)
        set_beam_action.setShortcut("Ctrl+Shift+B")
        set_beam_action.triggered.connect(self._set_beam_position)
        tools_menu.addAction(set_beam_action)

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
        
        # Update both alignment tabs when calibration completes
        self.signals.global_alignment_complete.connect(
            self.automated_alignment._update_global_status
        )
        self.signals.global_alignment_complete.connect(
            self.manual_alignment._update_calibration_status
        )

        self.signals.block_alignment_complete.connect(
            lambda bid: self.automated_alignment._update_block_status(bid)
        )
        self.signals.block_alignment_complete.connect(
            self.manual_alignment._update_calibration_status
        )

    def _start_camera_stream(self):
        """Start camera streaming thread."""
        if self.camera is None:
            print("[MainWindow] No camera available")
            return
        
        self.camera_thread = CameraStreamThread(camera=self.camera, target_fps=20)
        
        # Connect signals
        self.camera_thread.frame_ready.connect(self.camera_view.update_frame)
        self.camera_thread.stats_updated.connect(self.camera_view.update_stats)
        self.camera_thread.error_occurred.connect(
            lambda msg: self.signals.error_occurred.emit("Camera Error", msg)
        )
        
        # FIX: Give camera_view reference to thread for control
        self.camera_view.set_camera_thread(self.camera_thread)
        
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
    # View Menu Actions
    # ========================================================================
    
    def _set_colormap(self, colormap: str):
        """Set colormap from menu."""
        self.state.camera.colormap = colormap
        if self.camera_thread:
            self.camera_thread.color_manager.set_colormap(colormap)
        self.signals.colormap_changed.emit(colormap)
        self.signals.status_message.emit(f"Colormap: {colormap.capitalize()}")
    
    def _toggle_invert(self, checked: bool):
        """Toggle color inversion - FIXED."""
        if self.camera_thread is None:
            return
        
        # Get color manager and modify instance variable
        color_mgr = self.camera_thread.color_manager
        
        # Store original colormap if not already stored
        if not hasattr(self, '_original_colormap'):
            self._original_colormap = self.state.camera.colormap
        
        if checked:
            # Create inverted version by using thread's mutex
            from PyQt6.QtCore import QMutexLocker
            with QMutexLocker(self.camera_thread.mutex):
                # Invert by setting flag that will be checked during rendering
                if not hasattr(color_mgr, 'invert_enabled'):
                    color_mgr.invert_enabled = False
                color_mgr.invert_enabled = True
            
            self.signals.status_message.emit("Colors inverted")
        else:
            from PyQt6.QtCore import QMutexLocker
            with QMutexLocker(self.camera_thread.mutex):
                if hasattr(color_mgr, 'invert_enabled'):
                    color_mgr.invert_enabled = False
            
            self.signals.status_message.emit("Colors normal")
        
        # Force frame update
        self.signals.color_scale_changed.emit()
    
    def _toggle_crosshair(self, checked: bool):
        """Toggle crosshair display."""
        self.state.camera.show_crosshair = checked
        self.camera_view.update_overlay_settings()
    
    def _toggle_scalebar(self, checked: bool):
        """Toggle scale bar display."""
        self.state.camera.show_scale_bar = checked
        self.camera_view.update_overlay_settings()
    
    def _on_colormap_changed_signal(self, colormap: str):
        """Handle colormap change signal (to sync menu with other controls)."""
        # This ensures menu stays in sync if colormap is changed elsewhere
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
    
    
    # Add the handler method:
    def _manual_fiducial_capture(self):
        """Open manual fiducial capture dialog."""
        from app.widgets.manual_fiducial_dialog import ManualFiducialDialog
        
        dialog = ManualFiducialDialog(
            state=self.state,
            runtime_layout=self.runtime_layout,
            alignment_controller=self.alignment_controller,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save state
            self.runtime_layout.save_to_json(self.runtime_file_path)
            self.signals.status_message.emit("Manual calibration saved")

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

            # Reset RuntimeLayout (ADDED)
            if hasattr(self, 'runtime_layout'):
                self.runtime_layout.clear_measured_calibration()
            
            # Reset HierarchicalAlignment (ADDED)
            if hasattr(self, 'alignment_controller'):
                self.alignment_controller.alignment_system.reset_calibration()
            
            # Emit state reset signal
            self.signals.state_reset.emit()
            
            # Force UI updates (ADDED)
            self.alignment_panel._update_global_status()
            self.alignment_panel._update_block_status(self.state.navigation.current_block)
            self.block_grid._update_all_buttons()
            
            # Refresh waveguide panel if block selected
            if self.state.navigation.current_block is not None:
                self.waveguide_panel.refresh_waveguide_list()
            
            self.signals.status_message.emit("All calibrations reset")
            
            print("[MainWindow] Calibrations reset complete")
    
    def _goto_target(self):
        """Navigate to target waveguide."""
        self.waveguide_panel.navigate_to_target()
    
    def _run_autofocus(self):
        """Run autofocus dialog."""
        from app.widgets.autofocus_dialog import AutofocusDialog
        
        dialog = AutofocusDialog(
            autofocus_controller=self.autofocus_controller,
            state=self.state,
            parent=self
        )
        dialog.exec()
    
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
    

    def _set_block1_position(self):
        """Open Block 1 position dialog - SIMPLIFIED."""
        from app.widgets.block1_position_dialog import Block1PositionDialog  # New simplified version
        
        dialog = Block1PositionDialog(
            state=self.state,
            runtime_layout=self.runtime_layout,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save immediately
            self.runtime_layout.save_to_json(self.runtime_file_path)
            self.signals.status_message.emit("Block 1 position updated")

    def _save_state(self):
        """Save current state to runtime file."""
        if not hasattr(self, 'runtime_file_path'):
            print("ERROR: No runtime file path set")
            return
        
        try:
            self.runtime_layout.save_to_json(self.runtime_file_path)
            self.signals.state_saved.emit(self.runtime_file_path)
            self.signals.status_message.emit(f"Saved state to {Path(self.runtime_file_path).name}")
        except Exception as e:
            self.signals.error_occurred.emit("Failed to Save State", str(e))
    
    def _toggle_fourier(self, checked: bool):
        """Toggle Fourier transform display - NOW FAST!"""
        self.state.camera.show_fourier = checked
        
        # Tell camera thread to enable/disable FFT
        if self.camera_thread:
            self.camera_thread.set_fourier_mode(checked)
        
        if checked:
            self.signals.status_message.emit("Fourier transform enabled (frequency domain)")
            # Suggest good colormap for Fourier
            if self.state.camera.colormap == 'gray':
                self.signals.status_message.emit("Tip: Try 'jet' or 'hot' colormap for better frequency visualization")
        else:
            self.signals.status_message.emit("Real space image")
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
    
    def _set_beam_position(self):
        """Open beam position dialog."""
        from app.widgets.beam_position_dialog import BeamPositionDialog
        
        dialog = BeamPositionDialog(
            state=self.state,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update overlay
            self.camera_view.update_overlay_settings()
            self.signals.status_message.emit(
                f"Beam position set to ({self.state.camera.beam_position_px[0]}, "
                f"{self.state.camera.beam_position_px[1]}) px"
            )