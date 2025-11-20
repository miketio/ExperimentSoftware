"""
Main Application Window - WITH RESPONSIVE LANDSCAPE/PORTRAIT LAYOUTS

Automatically detects screen orientation and applies appropriate layout:
- Portrait (1080×1920): Camera top, controls below, block grid, waveguide table
- Landscape (1920×1080): Camera + table LEFT, controls + grid RIGHT
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

# Import widgets
from app.widgets.camera_view import CameraViewWidget
from app.widgets.stage_control import StageControlWidget
from app.widgets.block_grid import BlockGridWidget
from app.widgets.waveguide_panel import WaveguidePanelWidget
from app.widgets.status_bar import CustomStatusBar


class MainWindow(QMainWindow):
    """
    Main application window with responsive layout.
    
    Supports two modes:
    1. Portrait: Traditional stacked layout
    2. Landscape: Side-by-side layout optimized for wide screens
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
        
        # Layout mode
        self.layout_mode = "auto"  # "auto", "portrait", "landscape"
        
        # Setup UI
        self._detect_and_apply_layout()
        self._create_menu_bar()

        self._connect_signals()
        self._start_camera_stream()
        self._start_position_updates()
        
        # Window properties
        self.setWindowTitle(f"Microscope Alignment - {state.hardware_mode.value.upper()} Mode")
        self.resize(1600, 1000)
    
    def _detect_and_apply_layout(self):
        """Detect screen aspect ratio and apply appropriate layout."""
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        
        aspect_ratio = screen.width() / screen.height()
        
        if self.layout_mode == "auto":
            if aspect_ratio > 1.3:  # Landscape (wide)
                self._init_landscape_layout()
                print(f"[MainWindow] Auto-detected LANDSCAPE mode (aspect: {aspect_ratio:.2f})")
            else:  # Portrait (tall)
                self._init_portrait_layout()
                print(f"[MainWindow] Auto-detected PORTRAIT mode (aspect: {aspect_ratio:.2f})")
        elif self.layout_mode == "landscape":
            self._init_landscape_layout()
        else:
            self._init_portrait_layout()
    
    def _init_portrait_layout(self):
        """Initialize PORTRAIT layout (current working layout)."""
        # Create alignment controller FIRST
        from app.controllers.alignment_controller import AlignmentController
        self.alignment_controller = AlignmentController(
            state=self.state,
            signals=self.signals,
            camera=self.camera,
            stage=self.stage,
            runtime_layout=self.runtime_layout
        )
        
        from app.controllers.autofocus_controller import AutofocusController
        self.autofocus_controller = AutofocusController(
            camera=self.camera,  
            stage=self.stage,
            signals=self.signals
        )
        
        from app.controllers.navigation_controller import NavigationController
        self.navigation_controller = NavigationController(
            state=self.state,
            signals=self.signals,
            stage=self.stage,
            alignment_system=self.alignment_controller.alignment_system,
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
        
        # Right: Control panels (tabs) - SHORTENED NAMES
        control_tabs = QTabWidget()
        
        self.stage_control = StageControlWidget(
            self.state, self.signals, self.stage
        )
        control_tabs.addTab(self.stage_control, "Stage")  # SHORTENED
        
        from app.widgets.automated_alignment_panel import AutomatedAlignmentPanel
        self.automated_alignment = AutomatedAlignmentPanel(
            self.state, self.signals, self.alignment_controller
        )
        control_tabs.addTab(self.automated_alignment, "Auto Align")  # SHORTENED
        
        from app.widgets.manual_calibration_panel import ManualCalibrationPanel
        self.manual_alignment = ManualCalibrationPanel(
            self.state, self.signals, self.runtime_layout, 
            self.alignment_controller
        )
        control_tabs.addTab(self.manual_alignment, "Manual Calib")  # SHORTENED
        
        from app.widgets.setup_panel import SetupPanelWidget
        self.setup_panel = SetupPanelWidget(
            self.state, self.signals, self.runtime_layout, self.autofocus_controller
        )
        control_tabs.addTab(self.setup_panel, "Setup")
        
        top_splitter.addWidget(control_tabs)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(top_splitter, stretch=3)
        
        # Middle section: Block grid
        self.block_grid = BlockGridWidget(self.state, self.signals, self.runtime_layout)
        # Limit maximum height so it doesn't get too huge on tall screens
        self.block_grid.setMaximumHeight(350)

        main_layout.addWidget(self.block_grid, stretch=2)
        
        # Bottom section: Waveguide panel
        self.waveguide_panel = WaveguidePanelWidget(
            self.state, self.signals, self.stage, self.navigation_controller
        )
        main_layout.addWidget(self.waveguide_panel, stretch=2)
        
        # Status bar
        self.status_bar = CustomStatusBar(self.state, self.signals)
        self.setStatusBar(self.status_bar)
    
    def _init_landscape_layout(self):
        """Initialize LANDSCAPE layout (wide screen optimized)."""
        # Create controllers
        from app.controllers.alignment_controller import AlignmentController
        self.alignment_controller = AlignmentController(
            state=self.state,
            signals=self.signals,
            camera=self.camera,
            stage=self.stage,
            runtime_layout=self.runtime_layout
        )
        
        from app.controllers.autofocus_controller import AutofocusController
        self.autofocus_controller = AutofocusController(
            camera=self.camera,  
            stage=self.stage,
            signals=self.signals
        )
        
        from app.controllers.navigation_controller import NavigationController
        self.navigation_controller = NavigationController(
            state=self.state,
            signals=self.signals,
            stage=self.stage,
            alignment_system=self.alignment_controller.alignment_system,
            autofocus_controller=self.autofocus_controller
        )
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # HORIZONTAL SPLIT: Left (Visual) | Right (Controls)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ========================================
        # LEFT ZONE: Camera + Waveguide Table
        # ========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Camera view (larger)
        self.camera_view = CameraViewWidget(self.state, self.signals)
        left_layout.addWidget(self.camera_view, stretch=3)
        
        # Waveguide table (wide format)
        self.waveguide_panel = WaveguidePanelWidget(
            self.state, self.signals, self.stage, self.navigation_controller
        )
        left_layout.addWidget(self.waveguide_panel, stretch=2)
        
        main_splitter.addWidget(left_widget)
        
        # ========================================
        # RIGHT ZONE: Control Tabs + Block Grid
        # ========================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # Control tabs (SHORTENED NAMES)
        control_tabs = QTabWidget()
        
        self.stage_control = StageControlWidget(
            self.state, self.signals, self.stage
        )
        control_tabs.addTab(self.stage_control, "Stage")  # SHORTENED
        
        from app.widgets.automated_alignment_panel import AutomatedAlignmentPanel
        self.automated_alignment = AutomatedAlignmentPanel(
            self.state, self.signals, self.alignment_controller
        )
        control_tabs.addTab(self.automated_alignment, "Auto Align")  # SHORTENED
        
        from app.widgets.manual_calibration_panel import ManualCalibrationPanel
        self.manual_alignment = ManualCalibrationPanel(
            self.state, self.signals, self.runtime_layout, 
            self.alignment_controller
        )
        control_tabs.addTab(self.manual_alignment, "Manual Cal")  # SHORTENED
        
        from app.widgets.setup_panel import SetupPanelWidget
        self.setup_panel = SetupPanelWidget(
            self.state, self.signals, self.runtime_layout, self.autofocus_controller
        )
        control_tabs.addTab(self.setup_panel, "Setup")
        
        right_layout.addWidget(control_tabs, stretch=2)
        
        # Block grid
        self.block_grid = BlockGridWidget(self.state, self.signals, self.runtime_layout)
        right_layout.addWidget(self.block_grid, stretch=1)
        
        main_splitter.addWidget(right_widget)
        
        # Set splitter sizes (60% left, 40% right)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = CustomStatusBar(self.state, self.signals)
        self.setStatusBar(self.status_bar)
    
    def _create_menu_bar(self):
        """Create SIMPLIFIED menu bar."""
        menubar = self.menuBar()
        
        # ========================================
        # FILE MENU
        # ========================================
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
        
        # ========================================
        # CALIBRATION MENU (SIMPLIFIED)
        # ========================================
        calib_menu = menubar.addMenu("&Calibration")
        
        reset_calib_action = QAction("&Reset All Calibrations", self)
        reset_calib_action.triggered.connect(self._reset_calibrations)
        calib_menu.addAction(reset_calib_action)
        
        # ========================================
        # VIEW MENU
        # ========================================
        view_menu = menubar.addMenu("&View")
        
        # Layout mode submenu
        layout_menu = view_menu.addMenu("Layout Mode")
        
        layout_group = QActionGroup(self)
        layout_group.setExclusive(True)
        
        auto_layout = QAction("Auto-detect", self)
        auto_layout.setCheckable(True)
        auto_layout.setChecked(True)
        auto_layout.triggered.connect(lambda: self._switch_layout("auto"))
        layout_group.addAction(auto_layout)
        layout_menu.addAction(auto_layout)
        
        portrait_layout = QAction("Portrait (1080×1920)", self)
        portrait_layout.setCheckable(True)
        portrait_layout.triggered.connect(lambda: self._switch_layout("portrait"))
        layout_group.addAction(portrait_layout)
        layout_menu.addAction(portrait_layout)
        
        landscape_layout = QAction("Landscape (1920×1080)", self)
        landscape_layout.setCheckable(True)
        landscape_layout.triggered.connect(lambda: self._switch_layout("landscape"))
        layout_group.addAction(landscape_layout)
        layout_menu.addAction(landscape_layout)
        
        view_menu.addSeparator()
        
        # Colormap submenu
        colormap_menu = view_menu.addMenu("&Colormap")
        
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
            
            if cmap == self.state.camera.colormap:
                action.setChecked(True)
        
        # Invert colors
        view_menu.addSeparator()
        self.invert_action = QAction("&Invert Colors", self)
        self.invert_action.setCheckable(True)
        self.invert_action.setChecked(False)
        self.invert_action.triggered.connect(self._toggle_invert)
        view_menu.addAction(self.invert_action)
        
        view_menu.addSeparator()
        
        # Fourier transform
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
        
        zoom_400_action = QAction("Zoom &400%", self)
        zoom_400_action.triggered.connect(lambda: self.camera_view.set_zoom(4.0))
        view_menu.addAction(zoom_400_action)
        
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
        
        # ========================================
        # CAMERA MENU (NEW)
        # ========================================
        camera_menu = menubar.addMenu("&Camera")

        # Pixel size configuration
        pixel_size_action = QAction("Set Pixel Size (µm/pixel)...", self)
        pixel_size_action.setShortcut("Ctrl+Shift+P")
        pixel_size_action.setToolTip("Configure camera pixel size calibration")
        pixel_size_action.triggered.connect(self._set_pixel_size)
        camera_menu.addAction(pixel_size_action)

        camera_menu.addSeparator()

        # Camera info
        camera_info_action = QAction("Camera Information...", self)
        camera_info_action.triggered.connect(self._show_camera_info)
        camera_menu.addAction(camera_info_action)

        # ========================================
        # TOOLS MENU
        # ========================================
        tools_menu = menubar.addMenu("&Tools")
        
        set_beam_action = QAction("Set &Beam Position...", self)
        set_beam_action.setShortcut("Ctrl+Shift+B")
        set_beam_action.triggered.connect(self._set_beam_position)
        tools_menu.addAction(set_beam_action)
        
        # ========================================
        # HELP MENU
        # ========================================
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _switch_layout(self, mode: str):
        """Switch layout mode and rebuild UI."""
        if mode == self.layout_mode:
            return
        
        print(f"[MainWindow] Switching layout mode: {self.layout_mode} → {mode}")
        
        # Stop camera stream
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        # Stop position timer
        self.position_timer.stop()
        
        # Set new mode
        self.layout_mode = mode
        
        # Rebuild UI
        self._detect_and_apply_layout()
        
        # Reconnect signals
        self._connect_signals()
        
        # Restart camera
        self._start_camera_stream()
        self._start_position_updates()
        
        self.signals.status_message.emit(f"Layout switched to {mode} mode")
    
    def _connect_signals(self):
        """Connect signals between components."""
        self.signals.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 3000)
        )
        
        self.signals.error_occurred.connect(self._show_error)
        self.signals.block_selected.connect(self._on_block_selected)
        
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
        
        self.camera_thread.frame_ready.connect(self.camera_view.update_frame)
        self.camera_thread.stats_updated.connect(self.camera_view.update_stats)
        self.camera_thread.error_occurred.connect(
            lambda msg: self.signals.error_occurred.emit("Camera Error", msg)
        )
        
        self.camera_view.set_camera_thread(self.camera_thread)
        
        self.camera_thread.start()
        print("[MainWindow] Camera stream started")
    
    def _start_position_updates(self):
        """Start periodic stage position updates."""
        if self.stage is None:
            return
        
        self.position_timer.start(100)
    
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
            
            self.signals.stage_position_changed.emit('x', x)
            self.signals.stage_position_changed.emit('y', y)
            self.signals.stage_position_changed.emit('z', z)
        except Exception:
            pass
    
    # ========================================
    # Menu Actions
    # ========================================
    
    def _set_colormap(self, colormap: str):
        """Set colormap from menu."""
        self.state.camera.colormap = colormap
        if self.camera_thread:
            self.camera_thread.color_manager.set_colormap(colormap)
        self.signals.colormap_changed.emit(colormap)
        self.signals.status_message.emit(f"Colormap: {colormap.capitalize()}")
    
    def _toggle_invert(self, checked: bool):
        """Toggle color inversion."""
        if self.camera_thread is None:
            return
        
        from PyQt6.QtCore import QMutexLocker
        with QMutexLocker(self.camera_thread.mutex):
            color_mgr = self.camera_thread.color_manager
            if not hasattr(color_mgr, 'invert_enabled'):
                color_mgr.invert_enabled = False
            color_mgr.invert_enabled = checked
        
        self.signals.status_message.emit("Colors inverted" if checked else "Colors normal")
        self.signals.color_scale_changed.emit()
    
    def _toggle_crosshair(self, checked: bool):
        """Toggle crosshair display."""
        self.state.camera.show_crosshair = checked
        self.camera_view.update_overlay_settings()
    
    def _toggle_scalebar(self, checked: bool):
        """Toggle scale bar display."""
        self.state.camera.show_scale_bar = checked
        self.camera_view.update_overlay_settings()
    
    def _toggle_fourier(self, checked: bool):
        """Toggle Fourier transform display."""
        self.state.camera.show_fourier = checked
        
        if self.camera_thread:
            self.camera_thread.set_fourier_mode(checked)
        
        if checked:
            self.signals.status_message.emit("Fourier transform enabled")
        else:
            self.signals.status_message.emit("Real space image")
    
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
            
            if hasattr(self, 'runtime_layout'):
                self.runtime_layout.clear_measured_calibration()
            
            if hasattr(self, 'alignment_controller'):
                self.alignment_controller.alignment_system.reset_calibration()
            
            self.signals.state_reset.emit()
            
            self.automated_alignment._update_global_status()
            self.automated_alignment._update_block_status(self.state.navigation.current_block)
            self.block_grid._update_all_buttons()
            
            if self.state.navigation.current_block is not None:
                self.waveguide_panel.refresh_waveguide_list()
            
            self.signals.status_message.emit("All calibrations reset")
    
    def _set_beam_position(self):
        """Open beam position dialog."""
        from app.widgets.beam_position_dialog import BeamPositionDialog
        
        dialog = BeamPositionDialog(
            state=self.state,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.camera_view.update_overlay_settings()
            self.signals.status_message.emit(
                f"Beam position set to ({self.state.camera.beam_position_px[0]}, "
                f"{self.state.camera.beam_position_px[1]}) px"
            )
    
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
    
    def _on_block_selected(self, block_id: int):
        """Handle block selection."""
        self.state.set_current_block(block_id)
        self.waveguide_panel.refresh_waveguide_list()
    
    def _show_error(self, title: str, message: str):
        """Show error message box."""
        QMessageBox.critical(self, title, message)
    
    # ========================================
    # Cleanup
    # ========================================
    
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
        
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.position_timer.stop()


    def _set_pixel_size(self):
        """Open pixel size configuration dialog."""
        from app.dialogs.pixel_size_dialog import PixelSizeDialog
        
        # Get current value from camera
        current_value = self.camera.um_per_pixel if hasattr(self.camera, 'um_per_pixel') else 0.3
        
        dialog = PixelSizeDialog(current_value, parent=self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_value = dialog.get_value()
            
            # Update camera
            if hasattr(self.camera, 'um_per_pixel'):
                self.camera.um_per_pixel = new_value
            
            # Update system state (SINGLE SOURCE OF TRUTH)
            self.state.camera.um_per_pixel = new_value
            
            # Emit signal for any listeners
            self.signals.status_message.emit(
                f"Camera pixel size set to {new_value:.6f} µm/pixel"
            )
            
            # Update camera view overlays
            self.camera_view.update_overlay_settings()
            
            print(f"[MainWindow] Pixel size updated: {new_value:.6f} µm/pixel")

    def _show_camera_info(self):
        """Show camera information dialog."""
        from PyQt6.QtWidgets import QMessageBox
        
        info_text = "<h3>Camera Information</h3>"
        
        if self.camera is not None:
            info_text += f"<p><b>Type:</b> {type(self.camera).__name__}</p>"
            
            if hasattr(self.camera, 'um_per_pixel'):
                info_text += f"<p><b>Pixel Size:</b> {self.camera.um_per_pixel:.6f} µm/pixel</p>"
            
            if hasattr(self.camera, 'get_sensor_size'):
                try:
                    w, h = self.camera.get_sensor_size()
                    info_text += f"<p><b>Sensor Size:</b> {w} × {h} pixels</p>"
                    
                    if hasattr(self.camera, 'um_per_pixel'):
                        fov_w = w * self.camera.um_per_pixel
                        fov_h = h * self.camera.um_per_pixel
                        info_text += f"<p><b>Field of View:</b> {fov_w:.1f} × {fov_h:.1f} µm</p>"
                except:
                    pass
            
            if hasattr(self.camera, 'roi') and self.camera.roi is not None:
                roi = self.camera.roi
                info_text += f"<p><b>Current ROI:</b> ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})</p>"
            
            info_text += f"<p><b>Connected:</b> {'Yes' if self.state.camera_connected else 'No'}</p>"
        else:
            info_text += "<p>No camera connected</p>"
        
        QMessageBox.information(self, "Camera Information", info_text)


    # ========================================
    # STANDARDIZE um_per_pixel ACCESS
    # ========================================

    # CREATE a new method to get um_per_pixel consistently:

    def get_um_per_pixel(self) -> float:
        """
        Get camera pixel size (µm/pixel).
        
        SINGLE SOURCE OF TRUTH: Always get from state.camera.um_per_pixel
        
        Returns:
            float: Pixel size in micrometers per pixel
        """
        return self.state.camera.um_per_pixel