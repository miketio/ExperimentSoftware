"""
Main Application Window - WITH K-FILTER + HCU SLIT STAGE SUPPORT

Layout adapts to screen orientation (portrait / landscape).
Tabs:  Stage | Auto Align | Manual Cal | Setup | K-Filter | Stage 2 | Bookmarks
                                                              ^^^^^^^^
       Stage 2 = HCU slit-stage control (position presets, jog, saved positions)
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QMenuBar, QStatusBar, QMessageBox, QFileDialog, QDialog,
    QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from pathlib import Path

from app.system_state import SystemState
from app.signals import SystemSignals
from app.controllers.camera_stream import CameraStreamThread

from app.widgets.camera_view import CameraViewWidget
from app.widgets.stage_control import StageControlWidget
from app.widgets.block_grid import BlockGridWidget
from app.widgets.waveguide_panel import WaveguidePanelWidget
from app.widgets.status_bar import CustomStatusBar


class MainWindow(QMainWindow):
    """
    Main application window with responsive layout.
    Supports portrait (stacked) and landscape (side-by-side) modes.
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

        self.state          = state
        self.signals        = signals
        self.camera         = camera
        self.stage          = stage
        self.hw_manager     = hw_manager
        self.runtime_layout = runtime_layout

        self.camera_thread  = None
        self.layout_mode    = "auto"

        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._update_stage_position)

        self._detect_and_apply_layout()
        self._create_menu_bar()
        self._connect_signals()
        self._start_camera_stream()
        self._start_position_updates()

        self.setWindowTitle(
            f"Microscope Alignment — {state.hardware_mode.value.upper()} Mode"
        )
        self.resize(1600, 1000)

    # ------------------------------------------------------------------
    # Layout detection
    # ------------------------------------------------------------------

    def _detect_and_apply_layout(self):
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        ratio  = screen.width() / screen.height()

        if self.layout_mode == "auto":
            if ratio > 1.3:
                self._init_landscape_layout()
            else:
                self._init_portrait_layout()
        elif self.layout_mode == "landscape":
            self._init_landscape_layout()
        else:
            self._init_portrait_layout()

    # ------------------------------------------------------------------
    # Controller factory  (shared between both layout methods)
    # ------------------------------------------------------------------

    def _build_controllers(self):
        """Create all controllers.  Called once per layout init."""
        from app.controllers.alignment_controller import AlignmentController
        self.alignment_controller = AlignmentController(
            state=self.state, signals=self.signals,
            camera=self.camera, stage=self.stage,
            runtime_layout=self.runtime_layout
        )

        from app.controllers.autofocus_controller import AutofocusController
        self.autofocus_controller = AutofocusController(
            camera=self.camera, stage=self.stage, signals=self.signals
        )

        from app.controllers.navigation_controller import NavigationController
        self.navigation_controller = NavigationController(
            state=self.state, signals=self.signals,
            stage=self.stage,
            alignment_system=self.alignment_controller.alignment_system,
            autofocus_controller=self.autofocus_controller
        )

        # ── 1D filter stage ───────────────────────────────────────────
        filter_stage = self.hw_manager.get_filter_stage()
        if filter_stage is None:
            from hardware_control.setup_motor.mock_filter_stage import MockFilterStage
            filter_stage = MockFilterStage()
            print("[MainWindow] 1D filter stage: using MOCK")

        from app.controllers.filter_controller import FilterController
        self.filter_controller = FilterController(
            state=self.state, signals=self.signals,
            filter_stage=filter_stage,
            camera=self.camera, stage=self.stage
        )

        # ── HCU slit stage ────────────────────────────────────────────
        hcu_stage = self.hw_manager.get_hcu_stage()   # may be None
        from app.controllers.hcu_controller import HCUController
        self.hcu_controller = HCUController(
            state=self.state, signals=self.signals,
            hcu_stage=hcu_stage
        )
        if hcu_stage is None:
            print("[MainWindow] HCU stage: not connected — Stage 2 controls disabled")
        else:
            print("[MainWindow] HCU stage: connected ✅")

    # ------------------------------------------------------------------
    # Shared tab builder
    # ------------------------------------------------------------------

    def _wrap_in_scroll_area(self, widget: QWidget) -> QScrollArea:
        """Wrap large panels so content remains accessible in compact layouts."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

    def _build_control_tabs(self) -> QTabWidget:
        """Create the right-side control tab widget (used by both layouts)."""
        tabs = QTabWidget()

        # Stage
        self.stage_control = StageControlWidget(self.state, self.signals, self.stage)
        tabs.addTab(self.stage_control, "Stage")

        # Auto Align
        from app.widgets.automated_alignment_panel import AutomatedAlignmentPanel
        self.automated_alignment = AutomatedAlignmentPanel(
            self.state, self.signals, self.alignment_controller
        )
        tabs.addTab(self.automated_alignment, "Auto Align")

        # Manual Calib
        from app.widgets.manual_calibration_panel import ManualCalibrationPanel
        self.manual_alignment = ManualCalibrationPanel(
            self.state, self.signals, self.runtime_layout,
            self.alignment_controller
        )
        tabs.addTab(self.manual_alignment, "Manual Cal")

        # Setup
        from app.widgets.setup_panel import SetupPanelWidget
        self.setup_panel = SetupPanelWidget(
            self.state, self.signals, self.runtime_layout,
            self.autofocus_controller
        )
        tabs.addTab(self.setup_panel, "Setup")

        # K-Filter  (pass hcu_controller so Move Away/In buttons work)
        from app.widgets.filter_panel import FilterPanelWidget
        self.filter_panel = FilterPanelWidget(
            self.state, self.signals,
            self.filter_controller,
            hcu_controller=self.hcu_controller   # ← wired here
        )
        tabs.addTab(self._wrap_in_scroll_area(self.filter_panel), "K-Filter")

        # Stage 2  — HCU slit stage control
        from app.widgets.hcu_stage_panel import HCUStagePanelWidget
        self.hcu_panel = HCUStagePanelWidget(
            self.state, self.signals, self.hcu_controller
        )
        tabs.addTab(self._wrap_in_scroll_area(self.hcu_panel), "Stage 2")

        # Bookmarks
        from app.widgets.saved_positions_panel import SavedPositionsPanel
        self.saved_positions_panel = SavedPositionsPanel(
            self.state, self.signals, self.stage
        )
        tabs.addTab(self.saved_positions_panel, "Bookmarks")

        return tabs

    # ------------------------------------------------------------------
    # Portrait layout
    # ------------------------------------------------------------------

    def _init_portrait_layout(self):
        self._build_controllers()

        central     = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setCentralWidget(central)

        # Top: Camera + controls
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.camera_view = CameraViewWidget(self.state, self.signals)
        top_splitter.addWidget(self.camera_view)

        top_splitter.addWidget(self._build_control_tabs())
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(top_splitter, stretch=3)

        # Middle: Block grid
        self.block_grid = BlockGridWidget(
            self.state, self.signals, self.runtime_layout
        )
        self.block_grid.setMaximumHeight(350)
        main_layout.addWidget(self.block_grid, stretch=2)

        # Bottom: Waveguide panel
        self.waveguide_panel = WaveguidePanelWidget(
            self.state, self.signals, self.stage, self.navigation_controller
        )
        main_layout.addWidget(self.waveguide_panel, stretch=2)

        self.status_bar = CustomStatusBar(self.state, self.signals)
        self.setStatusBar(self.status_bar)

    # ------------------------------------------------------------------
    # Landscape layout
    # ------------------------------------------------------------------

    def _init_landscape_layout(self):
        self._build_controllers()

        central     = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setCentralWidget(central)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Camera + Waveguide table ────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)

        self.camera_view = CameraViewWidget(self.state, self.signals)
        left_layout.addWidget(self.camera_view, stretch=3)

        self.waveguide_panel = WaveguidePanelWidget(
            self.state, self.signals, self.stage, self.navigation_controller
        )
        left_layout.addWidget(self.waveguide_panel, stretch=2)
        main_splitter.addWidget(left_widget)

        # ── Right: Control tabs + Block grid ──────────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)

        right_tabs = self._build_control_tabs()
        right_tabs.setMinimumHeight(420)
        right_splitter.addWidget(right_tabs)

        self.block_grid = BlockGridWidget(
            self.state, self.signals, self.runtime_layout
        )
        self.block_grid.setMinimumHeight(220)
        right_splitter.addWidget(self.block_grid)

        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        main_splitter.addWidget(right_splitter)

        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        main_layout.addWidget(main_splitter)

        self.status_bar = CustomStatusBar(self.state, self.signals)
        self.setStatusBar(self.status_bar)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        act = QAction("&Open State…", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._open_state)
        file_menu.addAction(act)

        act = QAction("&Save State…", self)
        act.setShortcut("Ctrl+S")
        act.triggered.connect(self._save_state)
        file_menu.addAction(act)

        file_menu.addSeparator()
        act = QAction("E&xit", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # Calibration
        calib_menu = menubar.addMenu("&Calibration")
        act = QAction("&Reset All Calibrations", self)
        act.triggered.connect(self._reset_calibrations)
        calib_menu.addAction(act)

        # View
        view_menu = menubar.addMenu("&View")

        layout_menu  = view_menu.addMenu("Layout Mode")
        layout_group = QActionGroup(self)
        layout_group.setExclusive(True)

        for label, mode in [
            ("Auto-detect",       "auto"),
            ("Portrait (1080×1920)", "portrait"),
            ("Landscape (1920×1080)", "landscape"),
        ]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(mode == "auto")
            a.triggered.connect(lambda _, m=mode: self._switch_layout(m))
            layout_group.addAction(a)
            layout_menu.addAction(a)

        view_menu.addSeparator()

        colormap_menu  = view_menu.addMenu("&Colormap")
        colormap_group = QActionGroup(self)
        colormap_group.setExclusive(True)
        for cmap in ['gray', 'jet', 'hot', 'viridis', 'plasma', 'inferno', 'turbo', 'rainbow']:
            a = QAction(cmap.capitalize(), self)
            a.setCheckable(True)
            a.setChecked(cmap == self.state.camera.colormap)
            a.triggered.connect(lambda checked, c=cmap: self._set_colormap(c))
            colormap_group.addAction(a)
            colormap_menu.addAction(a)

        view_menu.addSeparator()

        self.invert_action = QAction("&Invert Colors", self)
        self.invert_action.setCheckable(True)
        self.invert_action.triggered.connect(self._toggle_invert)
        view_menu.addAction(self.invert_action)

        view_menu.addSeparator()

        self.fourier_action = QAction("Show &Fourier Transform", self)
        self.fourier_action.setCheckable(True)
        self.fourier_action.setShortcut("Ctrl+F")
        self.fourier_action.triggered.connect(self._toggle_fourier)
        view_menu.addAction(self.fourier_action)

        view_menu.addSeparator()

        for label, shortcut, fn in [
            ("Zoom to &Fit",  "Ctrl+0", self.camera_view.zoom_fit),
            ("Zoom &100%",    "Ctrl+1", lambda: self.camera_view.set_zoom(1.0)),
            ("Zoom &200%",    "Ctrl+2", lambda: self.camera_view.set_zoom(2.0)),
            ("Zoom &400%",    "",       lambda: self.camera_view.set_zoom(4.0)),
        ]:
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(fn)
            view_menu.addAction(a)

        view_menu.addSeparator()

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

        # Camera
        camera_menu = menubar.addMenu("&Camera")
        act = QAction("Set Pixel Size (µm/pixel)…", self)
        act.setShortcut("Ctrl+Shift+P")
        act.triggered.connect(self._set_pixel_size)
        camera_menu.addAction(act)
        camera_menu.addSeparator()
        act = QAction("Camera Information…", self)
        act.triggered.connect(self._show_camera_info)
        camera_menu.addAction(act)

        # Tools
        tools_menu = menubar.addMenu("&Tools")
        act = QAction("Set &Beam Position…", self)
        act.setShortcut("Ctrl+Shift+B")
        act.triggered.connect(self._set_beam_position)
        tools_menu.addAction(act)

        # Help
        help_menu = menubar.addMenu("&Help")
        act = QAction("&About", self)
        act.triggered.connect(self._show_about)
        help_menu.addAction(act)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
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
            lambda bid, res: self.automated_alignment._update_block_status(bid)
        )
        self.signals.block_alignment_complete.connect(
            lambda bid, res: self.manual_alignment._update_calibration_status()
        )

        # Camera stream control
        self.signals.request_stop_camera_stream.connect(self._stop_camera_stream)
        self.signals.request_start_camera_stream.connect(self._start_camera_stream_safe)

    # ------------------------------------------------------------------
    # Camera stream
    # ------------------------------------------------------------------

    def _stop_camera_stream(self):
        if self.camera_thread is not None:
            print("[MainWindow] Stopping camera stream")
            self.camera_thread.stop()
            self.camera_thread = None
            self.signals.camera_stream_stopped.emit()

    def _start_camera_stream(self):
        if self.camera is None:
            return
        if hasattr(self.camera, 'acquisition_running') and self.camera.acquisition_running:
            try:
                self.camera.stop_streaming()
            except Exception as e:
                print(f"[MainWindow] Warning: {e}")
        try:
            self.camera.start_streaming()
        except Exception as e:
            print(f"[MainWindow] Failed to start streaming: {e}")
            return

        self.camera_thread = CameraStreamThread(camera=self.camera, target_fps=20)
        self.camera_thread.frame_ready.connect(self.camera_view.update_frame)
        self.camera_thread.stats_updated.connect(self.camera_view.update_stats)
        self.camera_thread.error_occurred.connect(self._handle_camera_error)
        self.camera_view.set_camera_thread(self.camera_thread)

        # Give filter_controller access to camera thread for config preservation
        if hasattr(self, 'filter_controller'):
            self.filter_controller.set_camera_thread(self.camera_thread)

        self.camera_thread.start()
        print("[MainWindow] Camera stream started")

    def _start_camera_stream_safe(self):
        if self.camera_thread is None:
            self._start_camera_stream()
            self.signals.camera_stream_started.emit()

    def _handle_camera_error(self, msg: str):
        if not hasattr(self, '_last_camera_error') or self._last_camera_error != msg:
            self._last_camera_error = msg
            print(f"[MainWindow] Camera error: {msg}")
            if "AT_ERR_NOTWRITABLE" not in msg and "cannot join thread" not in msg:
                self.signals.error_occurred.emit("Camera Error", msg)

    # ------------------------------------------------------------------
    # Stage position polling
    # ------------------------------------------------------------------

    def _start_position_updates(self):
        if self.stage is None:
            return
        self.position_timer.start(100)

    def _update_stage_position(self):
        if self.stage is None:
            return
        try:
            for axis in ('x', 'y', 'z'):
                pos = self.stage.get_pos(axis)
                self.state.update_stage_position(axis, pos)
                self.signals.stage_position_changed.emit(axis, pos)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _switch_layout(self, mode: str):
        if mode == self.layout_mode:
            return
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.position_timer.stop()
        self.layout_mode = mode
        self._detect_and_apply_layout()
        self._connect_signals()
        self._start_camera_stream()
        self._start_position_updates()
        self.signals.status_message.emit(f"Layout: {mode}")

    def _set_colormap(self, colormap: str):
        self.state.camera.colormap = colormap
        if self.camera_thread:
            self.camera_thread.color_manager.set_colormap(colormap)
        self.signals.colormap_changed.emit(colormap)
        self.signals.status_message.emit(f"Colormap: {colormap.capitalize()}")

    def _toggle_invert(self, checked: bool):
        if self.camera_thread is None:
            return
        from PyQt6.QtCore import QMutexLocker
        with QMutexLocker(self.camera_thread.mutex):
            cm = self.camera_thread.color_manager
            if not hasattr(cm, 'invert_enabled'):
                cm.invert_enabled = False
            cm.invert_enabled = checked
        self.signals.status_message.emit("Colors inverted" if checked else "Colors normal")

    def _toggle_crosshair(self, checked: bool):
        self.state.camera.show_crosshair = checked
        self.camera_view.update_overlay_settings()

    def _toggle_scalebar(self, checked: bool):
        self.state.camera.show_scale_bar = checked
        self.camera_view.update_overlay_settings()

    def _toggle_fourier(self, checked: bool):
        self.state.camera.show_fourier = checked
        if self.camera_thread:
            self.camera_thread.set_fourier_mode(checked)
        msg = "Fourier transform enabled" if checked else "Real space image"
        self.signals.status_message.emit(msg)

    def _open_state(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open State", "", "JSON Files (*.json);;All Files (*)"
        )
        if filename:
            try:
                self.state.load_from_file(filename)
                self.signals.state_loaded.emit(filename)
                self.signals.status_message.emit(f"Loaded state from {Path(filename).name}")
            except Exception as e:
                self.signals.error_occurred.emit("Failed to Load State", str(e))

    def _save_state(self):
        if not hasattr(self, 'runtime_file_path'):
            return
        try:
            self.runtime_layout.save_to_json(self.runtime_file_path)
            self.signals.state_saved.emit(self.runtime_file_path)
            self.signals.status_message.emit(
                f"Saved state to {Path(self.runtime_file_path).name}"
            )
        except Exception as e:
            self.signals.error_occurred.emit("Failed to Save State", str(e))

    def _reset_calibrations(self):
        reply = QMessageBox.question(
            self, "Reset Calibrations",
            "This will reset all alignment calibrations. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.state.global_calibrated          = False
        self.state.global_calibration_params  = None
        for block in self.state.blocks.values():
            block.status            = block.status.NOT_CALIBRATED
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
        from app.widgets.beam_position_dialog import BeamPositionDialog
        dialog = BeamPositionDialog(state=self.state, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.camera_view.update_overlay_settings()
            bx, by = self.state.camera.beam_position_px
            self.signals.status_message.emit(f"Beam position set to ({bx}, {by}) px")

    def _set_pixel_size(self):
        from app.dialogs.pixel_size_dialog import PixelSizeDialog
        current = self.camera.um_per_pixel if hasattr(self.camera, 'um_per_pixel') else 0.3
        dialog  = PixelSizeDialog(current, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            v = dialog.get_value()
            if hasattr(self.camera, 'um_per_pixel'):
                self.camera.um_per_pixel = v
            self.state.camera.um_per_pixel = v
            self.signals.status_message.emit(f"Pixel size: {v:.6f} µm/pixel")
            self.camera_view.update_overlay_settings()

    def _show_camera_info(self):
        info = "<h3>Camera Information</h3>"
        if self.camera is not None:
            info += f"<p><b>Type:</b> {type(self.camera).__name__}</p>"
            if hasattr(self.camera, 'um_per_pixel'):
                info += f"<p><b>Pixel Size:</b> {self.camera.um_per_pixel:.6f} µm/pixel</p>"
            if hasattr(self.camera, 'get_sensor_size'):
                try:
                    w, h = self.camera.get_sensor_size()
                    info += f"<p><b>Sensor Size:</b> {w} × {h} pixels</p>"
                    if hasattr(self.camera, 'um_per_pixel'):
                        info += (f"<p><b>FOV:</b> {w*self.camera.um_per_pixel:.1f} × "
                                 f"{h*self.camera.um_per_pixel:.1f} µm</p>")
                except Exception:
                    pass
        QMessageBox.information(self, "Camera Information", info)

    def _show_about(self):
        QMessageBox.about(
            self, "About Microscope Alignment",
            "<h3>Microscope Alignment System</h3>"
            "<p>Version 1.1.0</p>"
            "<p>PyQt6 application for automated microscope alignment, "
            "navigation, and k-space slit control.</p>"
            f"<p><b>Hardware Mode:</b> {self.state.hardware_mode.value.upper()}</p>"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_block_selected(self, block_id: int):
        self.state.set_current_block(block_id)
        self.waveguide_panel.refresh_waveguide_list()

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Confirm Exit", "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.signals.application_closing.emit()
            self.cleanup()
            event.accept()
        else:
            event.ignore()

    def cleanup(self):
        print("[MainWindow] Cleaning up …")

        if hasattr(self, 'hcu_controller') and self.hcu_controller is not None:
            self.hcu_controller.shutdown()

        if hasattr(self, 'navigation_controller') and self.navigation_controller is not None:
            self.navigation_controller.shutdown()

        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        self.position_timer.stop()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_um_per_pixel(self) -> float:
        return self.state.camera.um_per_pixel