# app/widgets/setup_panel.py
"""
Setup Panel Widget

Contains initial setup tasks:
- Block 1 Position
- Autofocus

Non-modal, always accessible.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QComboBox, QSpinBox, QMessageBox
)
from PyQt6.QtCore import QTimer


class SetupPanelWidget(QWidget):
    """Setup tasks: Block 1 position and autofocus."""
    
    def __init__(self, state, signals, runtime_layout, autofocus_controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.runtime_layout = runtime_layout
        self.autofocus = autofocus_controller
        
        self._init_ui()
        
        # Position update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_position)
        self.timer.start(200)
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # ========================================
        # BLOCK 1 POSITION
        # ========================================
        block1_group = QGroupBox("Block 1 Position")
        block1_layout = QVBoxLayout()
        
        info = QLabel(
            "⭐ Move stage to Block 1 center and capture position.\n"
            "This defines the coordinate system origin."
        )
        info.setStyleSheet("QLabel { color: #666; font-size: 9pt; background-color: #E3F2FD; padding: 8px; }")
        info.setWordWrap(True)
        block1_layout.addWidget(info)
        
        # Current position
        self.block1_current = QLabel("Current: Y=?.???, Z=?.???")
        self.block1_current.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 13pt; "
            "background-color: #2C2C2C; color: lime; padding: 12px; "
            "font-weight: bold; }"
        )
        block1_layout.addWidget(self.block1_current)
        
        # Capture button
        btn_capture = QPushButton("📷 Capture Block 1 Position")
        btn_capture.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 11pt; }"
        )
        btn_capture.clicked.connect(self._capture_block1)
        block1_layout.addWidget(btn_capture)
        
        # Manual entry
        manual = QHBoxLayout()
        manual.addWidget(QLabel("Y:"))
        self.block1_y = QDoubleSpinBox()
        self.block1_y.setRange(-100000, 100000)
        self.block1_y.setDecimals(3)
        self.block1_y.setSuffix(" µm")
        
        if self.runtime_layout.has_block_1_position():
            y, z = self.runtime_layout.get_block_1_position()
            self.block1_y.setValue(y)
        
        manual.addWidget(self.block1_y)
        
        manual.addWidget(QLabel("Z:"))
        self.block1_z = QDoubleSpinBox()
        self.block1_z.setRange(-100000, 100000)
        self.block1_z.setDecimals(3)
        self.block1_z.setSuffix(" µm")
        
        if self.runtime_layout.has_block_1_position():
            y, z = self.runtime_layout.get_block_1_position()
            self.block1_z.setValue(z)
        
        manual.addWidget(self.block1_z)
        block1_layout.addLayout(manual)
        
        # Apply button
        btn_apply = QPushButton("✅ Apply")
        btn_apply.clicked.connect(self._apply_block1)
        block1_layout.addWidget(btn_apply)
        
        block1_group.setLayout(block1_layout)
        layout.addWidget(block1_group)
        
        # ========================================
        # AUTOFOCUS
        # ========================================
        af_group = QGroupBox("Autofocus")
        af_layout = QVBoxLayout()
        
        # Axis
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis:"))
        self.af_axis = QComboBox()
        self.af_axis.addItems(['X (Focus)', 'Y', 'Z'])
        axis_row.addWidget(self.af_axis)
        axis_row.addStretch()
        af_layout.addLayout(axis_row)
        
        # Range
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range:"))
        self.af_range = QDoubleSpinBox()
        self.af_range.setRange(1.0, 100.0)
        self.af_range.setValue(self.state.autofocus_config.get('range_um', 10.0))
        self.af_range.setSuffix(" µm")
        range_row.addWidget(self.af_range)
        range_row.addStretch()
        af_layout.addLayout(range_row)
        
        # Step
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self.af_step = QDoubleSpinBox()
        self.af_step.setRange(0.1, 5.0)
        self.af_step.setValue(self.state.autofocus_config.get('step_um', 0.5))
        self.af_step.setSuffix(" µm")
        self.af_step.setDecimals(2)
        step_row.addWidget(self.af_step)
        step_row.addStretch()
        af_layout.addLayout(step_row)
        
        # Status
        self.af_status = QLabel("Ready")
        self.af_status.setStyleSheet("QLabel { font-style: italic; color: #666; }")
        af_layout.addWidget(self.af_status)
        
        # Buttons
        btn_row = QHBoxLayout()
        
        self.btn_af_start = QPushButton("▶ Run Autofocus")
        self.btn_af_start.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self.btn_af_start.clicked.connect(self._run_autofocus)
        btn_row.addWidget(self.btn_af_start)
        
        self.btn_af_cancel = QPushButton("Cancel")
        self.btn_af_cancel.clicked.connect(self._cancel_autofocus)
        self.btn_af_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_af_cancel)
        
        af_layout.addLayout(btn_row)
        
        af_group.setLayout(af_layout)
        layout.addWidget(af_group)
        
        layout.addStretch()
        
        # Connect autofocus signals
        if self.autofocus:
            self.autofocus.signals.autofocus_started.connect(self._af_started)
            self.autofocus.signals.autofocus_progress.connect(self._af_progress)
            self.autofocus.signals.autofocus_complete.connect(self._af_complete)
            self.autofocus.signals.autofocus_failed.connect(self._af_failed)
    
    def _update_position(self):
        """Update current position display."""
        y, z = self.state.stage_position['y'], self.state.stage_position['z']
        self.block1_current.setText(f"Current: Y={y:.3f}, Z={z:.3f} µm")
    
    def _capture_block1(self):
        """Capture Block 1 position."""
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        
        self.block1_y.setValue(y)
        self.block1_z.setValue(z)
        
        # Flash green
        self.block1_current.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 13pt; "
            "background-color: green; color: white; padding: 12px; "
            "font-weight: bold; }"
        )
        QTimer.singleShot(500, lambda: self.block1_current.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 13pt; "
            "background-color: #2C2C2C; color: lime; padding: 12px; "
            "font-weight: bold; }"
        ))
    
    def _apply_block1(self):
        """Apply Block 1 position."""
        self.runtime_layout.set_block_1_position(
            self.block1_y.value(),
            self.block1_z.value()
        )
        
        # Save immediately
        from app.main_window import MainWindow
        main = self.window()
        if isinstance(main, MainWindow) and hasattr(main, 'runtime_file_path'):
            self.runtime_layout.save_to_json(main.runtime_file_path)
        
        self.signals.status_message.emit("Block 1 position updated")
    
    def _run_autofocus(self):
        """Run autofocus."""
        axis = self.af_axis.currentText()[0].lower()
        
        self.autofocus.run_autofocus(
            axis=axis,
            scan_range_um=self.af_range.value(),
            step_um=self.af_step.value(),
            enable_plot=False
        )
    
    def _cancel_autofocus(self):
        """Cancel autofocus."""
        self.autofocus.cancel()
    
    def _af_started(self, axis: str):
        """Handle autofocus start."""
        self.af_status.setText(f"Scanning {axis.upper()}-axis...")
        self.btn_af_start.setEnabled(False)
        self.btn_af_cancel.setEnabled(True)
    
    def _af_progress(self, pos: float, metric: float, progress: float):
        """Handle progress."""
        self.af_status.setText(f"Position: {pos:.3f} µm | Focus: {metric:.1f}")
    
    def _af_complete(self, pos: float, metric: float):
        """Handle completion."""
        self.af_status.setText(f"✅ Best: {pos:.3f} µm (metric: {metric:.1f})")
        self.btn_af_start.setEnabled(True)
        self.btn_af_cancel.setEnabled(False)
    
    def _af_failed(self, error: str):
        """Handle failure."""
        self.af_status.setText(f"❌ Failed: {error}")
        self.btn_af_start.setEnabled(True)
        self.btn_af_cancel.setEnabled(False)