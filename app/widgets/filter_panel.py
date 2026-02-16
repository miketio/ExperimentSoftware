"""
Filter Stage Control Panel - UPDATED for ±15mm range

CHANGES:
- Extended all range controls to ±15000 µm
- Updated quick position buttons
- Larger default step size for full range
- FIXED: Manual position input now works correctly (was going to 0)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QLineEdit, QFileDialog
)
from PyQt6.QtCore import QTimer


class FilterPanelWidget(QWidget):
    """Control panel for filter stage - UPDATED for ±15mm range."""
    
    def __init__(self, state, signals, filter_controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.filter = filter_controller
        
        self._init_ui()
        
        # Position update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_position)
        self.timer.start(500)  # Update every 0.5s
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # ========================================
        # STATUS
        # ========================================
        status_group = QGroupBox("Filter Stage Status")
        status_layout = QVBoxLayout()
        
        self.position_label = QLabel("Position: -- µm")
        self.position_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 14pt; "
            "background-color: #2C2C2C; color: lime; padding: 10px; }"
        )
        status_layout.addWidget(self.position_label)
        
        # Exposure time display
        self.exposure_label = QLabel("Camera Exposure: -- ms")
        self.exposure_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 10pt; "
            "background-color: #1E1E1E; color: cyan; padding: 5px; }"
        )
        status_layout.addWidget(self.exposure_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # ========================================
        # MANUAL CONTROL
        # ========================================
        manual_group = QGroupBox("Manual Position Control")
        manual_layout = QVBoxLayout()
        
        # Go to position - ✅ UPDATED RANGE to ±15000
        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Target Position:"))
        
        self.goto_spin = QDoubleSpinBox()
        self.goto_spin.setRange(-15000, 15000)  # ✅ Changed from (-8000, 100)
        self.goto_spin.setValue(0)
        self.goto_spin.setSuffix(" µm")
        self.goto_spin.setDecimals(3)
        goto_row.addWidget(self.goto_spin)
        
        self.btn_goto = QPushButton("Go To")
        self.btn_goto.clicked.connect(self._goto_position)
        self.btn_goto.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 8px; }"
        )
        goto_row.addWidget(self.btn_goto)
        
        manual_layout.addLayout(goto_row)
        
        # Quick positions - ✅ UPDATED with new range
        quick_row1 = QHBoxLayout()
        quick_row1.addWidget(QLabel("Negative:"))
        
        for pos_um in [-15000, -10000, -5000, -1000]:
            btn = QPushButton(f"{pos_um}µm")
            # ✅ FIXED: Call controller directly, don't use _goto_position with argument
            btn.clicked.connect(lambda checked, p=pos_um: self.filter.move_to_position(p))
            quick_row1.addWidget(btn)
        
        manual_layout.addLayout(quick_row1)
        
        quick_row2 = QHBoxLayout()
        quick_row2.addWidget(QLabel("Positive:"))
        
        for pos_um in [0, 1000, 5000, 10000, 15000]:
            btn = QPushButton(f"{pos_um}µm")
            # ✅ FIXED: Call controller directly, don't use _goto_position with argument
            btn.clicked.connect(lambda checked, p=pos_um: self.filter.move_to_position(p))
            quick_row2.addWidget(btn)
        
        manual_layout.addLayout(quick_row2)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        # ========================================
        # SWEEP CONFIGURATION
        # ========================================
        sweep_group = QGroupBox("Sweep Configuration")
        sweep_layout = QVBoxLayout()
        
        # Range - ✅ UPDATED RANGE to ±15000
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start:"))
        self.sweep_start = QDoubleSpinBox()
        self.sweep_start.setRange(-15000, 15000)  # ✅ Changed from (-8000, 100)
        self.sweep_start.setValue(-15000)         # ✅ Changed default
        self.sweep_start.setSuffix(" µm")
        self.sweep_start.setDecimals(3)
        range_row.addWidget(self.sweep_start)
        
        range_row.addWidget(QLabel("End:"))
        self.sweep_end = QDoubleSpinBox()
        self.sweep_end.setRange(-15000, 15000)   # ✅ Changed from (-8000, 100)
        self.sweep_end.setValue(15000)           # ✅ Changed default
        self.sweep_end.setSuffix(" µm")
        self.sweep_end.setDecimals(3)
        range_row.addWidget(self.sweep_end)
        
        sweep_layout.addLayout(range_row)
        
        # Step size - ✅ UPDATED for larger range
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step Size:"))
        self.sweep_step = QDoubleSpinBox()
        self.sweep_step.setRange(0.001, 1000)     # ✅ Increased max from 100
        self.sweep_step.setValue(100.0)           # ✅ Larger default for full range
        self.sweep_step.setSuffix(" µm")
        self.sweep_step.setDecimals(3)
        step_row.addWidget(self.sweep_step)
        
        # Calculate number of positions
        self.num_positions_label = QLabel("Positions: --")
        step_row.addWidget(self.num_positions_label)
        step_row.addStretch()
        
        self.sweep_start.valueChanged.connect(self._update_position_count)
        self.sweep_end.valueChanged.connect(self._update_position_count)
        self.sweep_step.valueChanged.connect(self._update_position_count)
        
        sweep_layout.addLayout(step_row)
        
        # Settle time
        settle_row = QHBoxLayout()
        settle_row.addWidget(QLabel("Settle Time:"))
        self.settle_spin = QDoubleSpinBox()
        self.settle_spin.setRange(0.1, 5.0)
        self.settle_spin.setValue(0.5)
        self.settle_spin.setSuffix(" s")
        self.settle_spin.setDecimals(2)
        settle_row.addWidget(self.settle_spin)
        settle_row.addStretch()
        sweep_layout.addLayout(settle_row)
        
        # Output directory
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Auto-generate timestamp folder")
        output_row.addWidget(self.output_edit)
        
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output)
        output_row.addWidget(btn_browse)
        
        sweep_layout.addLayout(output_row)
        
        sweep_group.setLayout(sweep_layout)
        layout.addWidget(sweep_group)
        
        # ========================================
        # SWEEP CONTROLS
        # ========================================
        control_row = QHBoxLayout()
        
        self.btn_run_sweep = QPushButton("▶️ Run Sweep")
        self.btn_run_sweep.clicked.connect(self._run_sweep)
        self.btn_run_sweep.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 12px; font-size: 12pt; }"
        )
        control_row.addWidget(self.btn_run_sweep)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.filter.cancel_sweep)
        self.btn_cancel.setEnabled(False)
        control_row.addWidget(self.btn_cancel)
        
        layout.addLayout(control_row)
        
        # Initial position count
        self._update_position_count()
        
        layout.addStretch()
    
    def _update_position(self):
        """Update current position display."""
        if self.filter.filter_stage is None:
            self.position_label.setText("Filter Stage: Not Connected")
            self.exposure_label.setText("Camera Exposure: N/A")
            return
        
        try:
            pos_nm = self.filter.filter_stage.get_position()
            pos_um = pos_nm / 1000.0
            pos_mm = pos_um / 1000.0
            # Show both µm and mm for large positions
            self.position_label.setText(f"Position: {pos_um:.3f} µm ({pos_mm:.3f} mm)")
        except Exception as e:
            self.position_label.setText(f"Position: Error - {e}")
        
        # Update exposure time display
        try:
            if self.filter.camera and hasattr(self.filter.camera, 'get_exposure_time'):
                exp_s = self.filter.camera.get_exposure_time()
                exp_ms = exp_s * 1000
                self.exposure_label.setText(f"Camera Exposure: {exp_ms:.2f} ms")
            else:
                self.exposure_label.setText("Camera Exposure: N/A")
        except Exception as e:
            self.exposure_label.setText(f"Camera Exposure: Error")
    
    def _goto_position(self, checked=False):
        """Move to position (button callback).
        
        ✅ FIXED: Ignore 'checked' argument from button clicked signal.
        Always read from the spin box widget.
        """
        # Get value from spin box (ignore 'checked' arg from button signal)
        pos_um = self.goto_spin.value()
        self.filter.move_to_position(pos_um)
    
    def _update_position_count(self):
        """Update calculated number of positions."""
        start = self.sweep_start.value()
        end = self.sweep_end.value()
        step = self.sweep_step.value()
        
        if step > 0 and end >= start:
            num = int((end - start) / step) + 1
            distance_mm = (end - start) / 1000
            self.num_positions_label.setText(f"Positions: {num} (over {distance_mm:.1f} mm)")
        else:
            self.num_positions_label.setText("Positions: Invalid")
    
    def _browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "results"
        )
        if directory:
            self.output_edit.setText(directory)
    
    def _run_sweep(self):
        """Run sweep."""
        start = self.sweep_start.value()
        end = self.sweep_end.value()
        step = self.sweep_step.value()
        settle = self.settle_spin.value()
        
        output = self.output_edit.text() or None
        
        success = self.filter.run_sweep(
            start_um=start,
            end_um=end,
            step_um=step,
            output_dir=output,
            settle_time_s=settle
        )
        
        if success:
            self.btn_run_sweep.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            
            # Re-enable after completion
            self.signals.busy_ended.connect(self._on_sweep_ended)
    
    def _on_sweep_ended(self):
        """Handle sweep end."""
        self.btn_run_sweep.setEnabled(True)
        self.btn_cancel.setEnabled(False)