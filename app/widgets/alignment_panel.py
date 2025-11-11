# app/widgets/alignment_panel.py
"""Alignment Control Panel - Updated with controller integration."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QSpinBox, QCheckBox, QComboBox, QMessageBox
from PyQt6.QtCore import QTimer, Qt  # ← Add this line


class AlignmentPanelWidget(QWidget):
    """Alignment controls and status display."""
    
    def __init__(self, state, signals, alignment_controller, parent=None):
        """
        Initialize alignment panel.
        
        Args:
            state: SystemState instance
            signals: SystemSignals instance
            alignment_controller: AlignmentController instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.alignment_controller = alignment_controller
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Global alignment
        global_group = QGroupBox("Global Alignment")
        global_layout = QVBoxLayout()
        
        self.global_status = QLabel("⚪ Not Calibrated")
        self.global_status.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        global_layout.addWidget(self.global_status)
        
        self.global_info = QLabel("Rotation: --\nTranslation: --\nError: --")
        self.global_info.setStyleSheet("QLabel { font-family: monospace; }")
        global_layout.addWidget(self.global_info)
        
        # Global alignment settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Search radius:"))
        self.global_radius_spin = QSpinBox()
        self.global_radius_spin.setRange(20, 500)
        self.global_radius_spin.setValue(100)
        self.global_radius_spin.setSuffix(" µm")
        settings_layout.addWidget(self.global_radius_spin)
        
        settings_layout.addWidget(QLabel("Step:"))
        self.global_step_spin = QSpinBox()
        self.global_step_spin.setRange(5, 100)
        self.global_step_spin.setValue(20)
        self.global_step_spin.setSuffix(" µm")
        settings_layout.addWidget(self.global_step_spin)
        settings_layout.addStretch()
        global_layout.addLayout(settings_layout)
        
        self.btn_run_global = QPushButton("Run Global Alignment")
        self.btn_run_global.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.btn_run_global.clicked.connect(self._run_global)
        global_layout.addWidget(self.btn_run_global)
        
        global_group.setLayout(global_layout)
        layout.addWidget(global_group)
        
        # Block alignment
        block_group = QGroupBox("Block Alignment")
        block_layout = QVBoxLayout()
        
        self.block_status = QLabel("No block selected")
        self.block_status.setStyleSheet("QLabel { font-size: 11pt; }")
        block_layout.addWidget(self.block_status)
        
        # Block alignment settings
        block_settings = QHBoxLayout()
        block_settings.addWidget(QLabel("Search radius:"))
        self.block_radius_spin = QSpinBox()
        self.block_radius_spin.setRange(10, 200)
        self.block_radius_spin.setValue(60)
        self.block_radius_spin.setSuffix(" µm")
        block_settings.addWidget(self.block_radius_spin)
        
        block_settings.addWidget(QLabel("Step:"))
        self.block_step_spin = QSpinBox()
        self.block_step_spin.setRange(5, 50)
        self.block_step_spin.setValue(15)
        self.block_step_spin.setSuffix(" µm")
        block_settings.addWidget(self.block_step_spin)
        block_settings.addStretch()
        block_layout.addLayout(block_settings)
        
        self.btn_calibrate_block = QPushButton("Calibrate Selected Block")
        self.btn_calibrate_block.clicked.connect(self._calibrate_block)
        self.btn_calibrate_block.setEnabled(False)
        block_layout.addWidget(self.btn_calibrate_block)
        
        self.btn_calibrate_all = QPushButton("Calibrate All Blocks")
        self.btn_calibrate_all.clicked.connect(self._calibrate_all)
        self.btn_calibrate_all.setEnabled(False)  # Requires global calibration
        block_layout.addWidget(self.btn_calibrate_all)
        
        block_group.setLayout(block_layout)
        layout.addWidget(block_group)
        
        layout.addStretch()
        # ADD TO EXISTING alignment_panel.py - after Block Alignment section
        # Manual Fiducial Capture
        manual_group = QGroupBox("Manual Fiducial Capture")
        manual_layout = QVBoxLayout()

        info = QLabel(
            "Manually capture fiducial positions by moving stage\n"
            "and clicking Capture. Requires ≥2 fiducials."
        )
        info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        info.setWordWrap(True)
        manual_layout.addWidget(info)

        # Selection
        selection = QHBoxLayout()
        selection.addWidget(QLabel("Block:"))
        self.manual_block_combo = QComboBox()
        self.manual_block_combo.addItems([str(i) for i in range(1, 21)])
        selection.addWidget(self.manual_block_combo)

        selection.addWidget(QLabel("Corner:"))
        self.manual_corner_combo = QComboBox()
        self.manual_corner_combo.addItems(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        selection.addWidget(self.manual_corner_combo)
        selection.addStretch()
        manual_layout.addLayout(selection)

        # Current position (live update)
        self.manual_pos_label = QLabel("Current: Y=?.???, Z=?.???")
        self.manual_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 11pt; "
            "background-color: #2C2C2C; color: lime; padding: 8px; }"
        )
        manual_layout.addWidget(self.manual_pos_label)

        # Capture button
        self.btn_manual_capture = QPushButton("📷 Capture Fiducial")
        self.btn_manual_capture.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self.btn_manual_capture.clicked.connect(self._manual_capture)
        manual_layout.addWidget(self.btn_manual_capture)

        # Captured list
        self.manual_list = QLabel("Captured: 0 fiducials")
        self.manual_list.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        manual_layout.addWidget(self.manual_list)

        btn_row = QHBoxLayout()
        self.btn_manual_clear = QPushButton("Clear")
        self.btn_manual_clear.clicked.connect(self._manual_clear)
        btn_row.addWidget(self.btn_manual_clear)

        self.btn_manual_apply = QPushButton("✅ Apply Calibration")
        self.btn_manual_apply.clicked.connect(self._manual_apply)
        self.btn_manual_apply.setEnabled(False)
        btn_row.addWidget(self.btn_manual_apply)
        manual_layout.addLayout(btn_row)

        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

        # Storage for captured fiducials
        self.manual_fiducials = []

        # Position update timer
        self.manual_timer = QTimer()
        self.manual_timer.timeout.connect(self._update_manual_position)
        self.manual_timer.start(200)
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.block_selected.connect(self._update_block_status)
        self.signals.global_alignment_complete.connect(self._update_global_status)
        self.signals.block_alignment_complete.connect(lambda bid, res: self._update_block_status(bid))
    
    def _run_global(self):
        """Run global alignment."""
        search_radius = self.global_radius_spin.value()
        step = self.global_step_spin.value()
        
        print(f"[AlignmentPanel] Starting global alignment (radius={search_radius}µm, step={step}µm)")
        
        # Disable button during alignment
        self.btn_run_global.setEnabled(False)
        

        # Start alignment (will show progress dialog)
        self.alignment_controller.start_global_alignment(
            corner_pairs=[
            (1, 'top_left'),      # Block 1
            (20, 'bottom_right')  # Block 20
        ],  # Corner blocks
            search_radius_um=search_radius,
            step_um=step
        )
        
        # Re-enable button
        self.btn_run_global.setEnabled(True)
    
    def _calibrate_block(self):
        """Calibrate selected block."""
        if self.state.navigation.current_block is None:
            return
        
        block_id = self.state.navigation.current_block
        search_radius = self.block_radius_spin.value()
        step = self.block_step_spin.value()
        
        print(f"[AlignmentPanel] Calibrating block {block_id} (radius={search_radius}µm, step={step}µm)")
        
        # Disable button
        self.btn_calibrate_block.setEnabled(False)
        
        # Start alignment
        self.alignment_controller.start_block_alignment(
            block_id=block_id,
            corners=['top_left', 'bottom_right'],
            search_radius_um=search_radius,
            step_um=step
        )
        
        # Re-enable
        self.btn_calibrate_block.setEnabled(True)
    
    def _calibrate_all(self):
        """Calibrate all blocks."""
        search_radius = self.block_radius_spin.value()
        step = self.block_step_spin.value()
        
        all_blocks = list(range(1, 21))  # Blocks 1-20
        
        print(f"[AlignmentPanel] Starting batch alignment for {len(all_blocks)} blocks")
        
        # Disable buttons
        self.btn_calibrate_all.setEnabled(False)
        
        # Start batch alignment
        self.alignment_controller.start_batch_alignment(
            block_ids=all_blocks,
            search_radius_um=search_radius,
            step_um=step
        )
        
        # Re-enable
        self.btn_calibrate_all.setEnabled(True)
    
    def _update_global_status(self, results=None):
        """Update global status display."""
        if self.state.global_calibrated:
            self.global_status.setText("🟢 Calibrated")
            
            if self.state.global_calibration_params:
                params = self.state.global_calibration_params
                self.global_info.setText(
                    f"Rotation: {params.get('rotation_deg', 0):.3f}°\n"
                    f"Translation: ({params.get('translation_um', (0,0))[0]:.2f}, {params.get('translation_um', (0,0))[1]:.2f}) µm\n"
                    f"Error: {params.get('mean_error_um', 0):.3f} µm"
                )
            
            # Enable batch calibration after global is done
            self.btn_calibrate_all.setEnabled(True)
        else:
            self.global_status.setText("⚪ Not Calibrated")
            self.global_info.setText("Rotation: --\nTranslation: --\nError: --")
            self.btn_calibrate_all.setEnabled(False)
    
    def _update_block_status(self, block_id: int):
        """Update block status display."""
        if block_id is None:
            self.block_status.setText("No block selected")
            self.btn_calibrate_block.setEnabled(False)
            return
        
        state = self.state.get_block_state(block_id)
        status_text = f"Block {block_id}: {state.status.name}"
        if state.calibration_error is not None:
            status_text += f" (Error: {state.calibration_error:.3f} µm)"
        
        self.block_status.setText(status_text)
        
        # Enable block calibration only if global is done
        self.btn_calibrate_block.setEnabled(self.state.global_calibrated)

    def _update_manual_position(self):
        """Update current position display."""
        y, z = self.state.stage_position['y'], self.state.stage_position['z']
        self.manual_pos_label.setText(f"Current: Y={y:.3f}, Z={z:.3f} µm")

    def _manual_capture(self):
        """Capture current fiducial position."""
        block_id = int(self.manual_block_combo.currentText())
        corner = self.manual_corner_combo.currentText()
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        
        # Check for duplicates
        for fid in self.manual_fiducials:
            if fid['block_id'] == block_id and fid['corner'] == corner:
                self.manual_fiducials.remove(fid)
                break
        
        # Add
        self.manual_fiducials.append({
            'block_id': block_id,
            'corner': corner,
            'stage_Y': y,
            'stage_Z': z,
            'confidence': 1.0,
            'verification_error_um': 0.0
        })
        
        # Update UI
        self.manual_list.setText(f"Captured: {len(self.manual_fiducials)} fiducials")
        self.btn_manual_apply.setEnabled(len(self.manual_fiducials) >= 2)
        
        # Flash green
        self.manual_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 11pt; "
            "background-color: green; color: white; padding: 8px; }"
        )
        QTimer.singleShot(500, lambda: self.manual_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 11pt; "
            "background-color: #2C2C2C; color: lime; padding: 8px; }"
        ))
        
        print(f"[AlignmentPanel] Captured Block {block_id} {corner}: ({y:.3f}, {z:.3f}) µm")

    def _manual_clear(self):
        """Clear captured fiducials."""
        self.manual_fiducials.clear()
        self.manual_list.setText("Captured: 0 fiducials")
        self.btn_manual_apply.setEnabled(False)

    def _manual_apply(self):
        """Apply manual calibration."""
        if len(self.manual_fiducials) < 2:
            QMessageBox.warning(
                self,
                "Not Enough Data",
                "Need at least 2 fiducials for calibration."
            )
            return
        
        # Determine type (global or block)
        unique_blocks = set(f['block_id'] for f in self.manual_fiducials)
        
        try:
            if len(unique_blocks) > 1:
                # Global calibration
                result = self.alignment_controller.alignment_system.calibrate_global(
                    self.manual_fiducials
                )
                
                # Update runtime layout
                from app.main_window import MainWindow
                main = self.window()
                if isinstance(main, MainWindow):
                    main.runtime_layout.set_global_calibration(
                        rotation=result['rotation_deg'],
                        translation=result['translation_um'],
                        calibration_error=result['mean_error_um'],
                        num_points=len(self.manual_fiducials)
                    )
                
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Global calibration successful!\n\n"
                    f"Rotation: {result['rotation_deg']:.6f}°\n"
                    f"Error: {result['mean_error_um']:.6f} µm"
                )
                
            else:
                # Block calibration
                block_id = list(unique_blocks)[0]
                result = self.alignment_controller.alignment_system.calibrate_block(
                    block_id,
                    self.manual_fiducials
                )
                
                from app.main_window import MainWindow
                main = self.window()
                if isinstance(main, MainWindow):
                    main.runtime_layout.set_block_calibration(
                        block_id=block_id,
                        rotation=result['rotation_deg'],
                        translation=result['origin_stage_um'],
                        calibration_error=result['mean_error_um'],
                        num_points=len(self.manual_fiducials)
                    )
                
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Block {block_id} calibration successful!\n\n"
                    f"Error: {result['mean_error_um']:.6f} µm"
                )
            
            # Clear after success
            self._manual_clear()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Calibration Failed",
                f"Failed to apply calibration:\n\n{e}"
            )
            import traceback
            traceback.print_exc()