# app/widgets/manual_calibration_panel.py - WITH AUTO-REFRESH
"""
Manual Calibration Panel - NOW SHOWS AUTOMATED FIDUCIALS TOO

Changes:
1. Refresh fiducial list when automated alignment completes
2. Shows both manual and automated captures
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QRadioButton, QButtonGroup, QTextEdit,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor


class ManualCalibrationPanel(QWidget):
    """Manual fiducial capture and calibration."""
    
    def __init__(self, state, signals, runtime_layout, alignment_controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.runtime_layout = runtime_layout
        self.alignment_controller = alignment_controller
        
        self._init_ui()
        self._connect_signals()
        
        # Load existing fiducials
        self._load_fiducials_from_runtime()
        
        # Position update timer
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._update_current_position)
        self.position_timer.start(200)
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # ========================================
        # CAPTURED FIDUCIALS TABLE
        # ========================================
        fid_group = QGroupBox("📍 Captured Fiducials (Manual + Automated)")
        fid_layout = QVBoxLayout()
        
        info = QLabel(
            "💡 Click any row to navigate to that position\n"
            "🤖 Automated captures appear here automatically"
        )
        info.setStyleSheet("QLabel { color: #2196F3; font-style: italic; }")
        fid_layout.addWidget(info)
        
        # Refresh button
        btn_refresh = QPushButton("🔄 Refresh from Runtime")
        btn_refresh.setToolTip("Reload fiducials from RuntimeLayout")
        btn_refresh.clicked.connect(self._load_fiducials_from_runtime)
        fid_layout.addWidget(btn_refresh)
        
        self.fiducial_table = QTableWidget()
        self.fiducial_table.setColumnCount(4)
        self.fiducial_table.setHorizontalHeaderLabels([
            'Block', 'Corner', 'Position (Y, Z) µm', 'Action'
        ])
        self.fiducial_table.setAlternatingRowColors(True)
        self.fiducial_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.fiducial_table.cellClicked.connect(self._on_fiducial_row_clicked)
        
        # Set column widths
        header = self.fiducial_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.fiducial_table.setColumnWidth(3, 60)
        
        fid_layout.addWidget(self.fiducial_table)
        
        fid_group.setLayout(fid_layout)
        layout.addWidget(fid_group)
        
        # ========================================
        # CAPTURE NEW FIDUCIAL
        # ========================================
        capture_group = QGroupBox("Capture New Fiducial")
        capture_layout = QVBoxLayout()
        
        # Selection row
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Block:"))
        self.block_combo = QComboBox()
        self.block_combo.addItems([str(i) for i in range(1, 21)])
        select_row.addWidget(self.block_combo)
        
        select_row.addWidget(QLabel("Corner:"))
        self.corner_combo = QComboBox()
        self.corner_combo.addItems([
            'top_left', 'top_right', 'bottom_left', 'bottom_right'
        ])
        select_row.addWidget(self.corner_combo)
        select_row.addStretch()
        capture_layout.addLayout(select_row)
        
        # Current position
        self.current_pos_label = QLabel("Current Stage: Y=?.???, Z=?.??? µm")
        self.current_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 13pt; "
            "background-color: #2C2C2C; color: lime; padding: 10px; "
            "font-weight: bold; }"
        )
        capture_layout.addWidget(self.current_pos_label)
        
        # Buttons
        btn_row = QHBoxLayout()
        
        self.btn_capture = QPushButton("📷 Capture Position")
        self.btn_capture.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 11pt; }"
        )
        self.btn_capture.clicked.connect(self._capture_fiducial)
        btn_row.addWidget(self.btn_capture)
        
        btn_clear_all = QPushButton("Clear All")
        btn_clear_all.clicked.connect(self._clear_all_fiducials)
        btn_row.addWidget(btn_clear_all)
        
        btn_row.addStretch()
        capture_layout.addLayout(btn_row)
        
        capture_group.setLayout(capture_layout)
        layout.addWidget(capture_group)
        
        # ========================================
        # RUN CALIBRATION
        # ========================================
        calib_group = QGroupBox("Run Calibration")
        calib_layout = QVBoxLayout()
        
        # Calibration type selection
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Calibration Type:"))
        
        self.calib_button_group = QButtonGroup()
        
        self.radio_global = QRadioButton("Global")
        self.radio_global.setChecked(True)
        self.radio_global.toggled.connect(self._update_calibration_status)
        self.calib_button_group.addButton(self.radio_global)
        type_row.addWidget(self.radio_global)
        
        self.radio_block = QRadioButton("Block")
        self.radio_block.toggled.connect(self._update_calibration_status)
        self.calib_button_group.addButton(self.radio_block)
        type_row.addWidget(self.radio_block)
        
        self.block_calib_combo = QComboBox()
        self.block_calib_combo.addItems([str(i) for i in range(1, 21)])
        self.block_calib_combo.setEnabled(False)
        self.block_calib_combo.currentTextChanged.connect(
            self._update_calibration_status
        )
        self.radio_block.toggled.connect(
            lambda checked: self.block_calib_combo.setEnabled(checked)
        )
        type_row.addWidget(self.block_calib_combo)
        
        type_row.addStretch()
        calib_layout.addLayout(type_row)
        
        # Status check display
        calib_layout.addWidget(QLabel("Status Check:"))
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 10pt; "
            "background-color: #F5F5F5; }"
        )
        calib_layout.addWidget(self.status_text)
        
        # Run button
        self.btn_run_calibration = QPushButton("▶️ Run Calibration")
        self.btn_run_calibration.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt; }"
        )
        self.btn_run_calibration.clicked.connect(self._run_calibration)
        self.btn_run_calibration.setEnabled(False)
        calib_layout.addWidget(self.btn_run_calibration)
        
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)
        
        # Initial status update
        self._update_calibration_status()
    
    def _connect_signals(self):
        """Connect signals - WITH AUTOMATED ALIGNMENT REFRESH."""
        # ✅ REFRESH when automated alignment completes
        self.signals.global_alignment_complete.connect(
            lambda res: self._on_automated_calibration_complete()
        )
        self.signals.block_alignment_complete.connect(
            lambda bid, res: self._on_automated_calibration_complete()
        )
        
        self.signals.block_selected.connect(self._on_block_selected)
    
    def _on_automated_calibration_complete(self):
        """Refresh fiducial list after automated calibration."""
        print("[ManualCalibrationPanel] Automated calibration complete - refreshing fiducials")
        self._load_fiducials_from_runtime()
        self._update_calibration_status()
    
    def _on_block_selected(self, block_id: int):
        """Handle block selection from grid."""
        if block_id is not None:
            self.block_calib_combo.setCurrentText(str(block_id))
    
    # ========================================
    # Fiducial Management
    # ========================================
    
    def _load_fiducials_from_runtime(self):
        """Load existing fiducials from RuntimeLayout."""
        self.fiducial_table.setRowCount(0)
        
        fiducials = self.runtime_layout.get_all_captured_fiducials()
        
        print(f"[ManualCalibrationPanel] Loading {len(fiducials)} fiducials")
        
        for fid in fiducials:
            self._add_fiducial_to_table(
                fid['block_id'],
                fid['corner'],
                fid['Y'],
                fid['Z']
            )
        
        self._update_calibration_status()
    
    def _add_fiducial_to_table(self, block_id: int, corner: str, Y: float, Z: float):
        """Add fiducial to table."""
        row = self.fiducial_table.rowCount()
        self.fiducial_table.insertRow(row)
        
        # Block
        item = QTableWidgetItem(str(block_id))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setData(Qt.ItemDataRole.UserRole, (block_id, corner, Y, Z))
        self.fiducial_table.setItem(row, 0, item)
        
        # Corner
        item = QTableWidgetItem(corner)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fiducial_table.setItem(row, 1, item)
        
        # Position
        item = QTableWidgetItem(f"({Y:.3f}, {Z:.3f})")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fiducial_table.setItem(row, 2, item)
        
        # Delete button
        btn_delete = QPushButton("🗑")
        btn_delete.setToolTip("Delete this fiducial")
        btn_delete.clicked.connect(lambda checked, r=row: self._delete_fiducial(r))
        self.fiducial_table.setCellWidget(row, 3, btn_delete)
    
    def _on_fiducial_row_clicked(self, row: int, col: int):
        """Handle row click - navigate with confirmation."""
        if col == 3:  # Delete button column
            return
        
        item = self.fiducial_table.item(row, 0)
        if not item:
            return
            
        block_id, corner, Y, Z = item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Navigate to Fiducial",
            f"Move stage to Block {block_id} {corner}?\n\n"
            f"Target position:\n"
            f"Y = {Y:.3f} µm\n"
            f"Z = {Z:.3f} µm",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Ok:
            main_window = self.window()
            if hasattr(main_window, 'stage'):
                stage = main_window.stage
                stage.move_abs('y', Y)
                stage.move_abs('z', Z)
                self.signals.status_message.emit(
                    f"✅ Moved to Block {block_id} {corner}"
                )
    
    def _delete_fiducial(self, row: int):
        """Delete fiducial from runtime and table."""
        item = self.fiducial_table.item(row, 0)
        if not item:
            return
            
        block_id, corner, Y, Z = item.data(Qt.ItemDataRole.UserRole)
        
        # Remove from runtime_layout
        self.runtime_layout.remove_captured_fiducial(block_id, corner)
        
        # Save immediately
        main_window = self.window()
        if hasattr(main_window, 'runtime_file_path'):
            self.runtime_layout.save_to_json(main_window.runtime_file_path)
        
        # Remove from table
        self.fiducial_table.removeRow(row)
        
        # Update status
        self._update_calibration_status()
        
        self.signals.status_message.emit(
            f"Deleted Block {block_id} {corner}"
        )
    
    def _capture_fiducial(self):
        """Capture current position as fiducial."""
        block_id = int(self.block_combo.currentText())
        corner = self.corner_combo.currentText()
        Y = self.state.stage_position['y']
        Z = self.state.stage_position['z']
        
        if abs(Y) < 0.001 and abs(Z) < 0.001:
            reply = QMessageBox.question(
                self,
                "Suspicious Position",
                "Current stage position is at origin (0, 0).\n\n"
                "This may indicate stage hasn't moved yet.\n"
                "Capture anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Check for duplicate
        if self.runtime_layout.has_captured_fiducial(block_id, corner):
            reply = QMessageBox.question(
                self,
                "Overwrite Fiducial",
                f"Block {block_id} {corner} already captured.\n\n"
                f"Replace with new position?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            
            # Remove old entry
            for row in range(self.fiducial_table.rowCount()):
                item = self.fiducial_table.item(row, 0)
                if item:
                    b, c, _, _ = item.data(Qt.ItemDataRole.UserRole)
                    if b == block_id and c == corner:
                        self.fiducial_table.removeRow(row)
                        break
        
        # Add to runtime_layout
        self.runtime_layout.add_captured_fiducial(block_id, corner, Y, Z)
        
        # Save immediately
        main_window = self.window()
        if hasattr(main_window, 'runtime_file_path'):
            self.runtime_layout.save_to_json(main_window.runtime_file_path)
        
        # Add to table
        self._add_fiducial_to_table(block_id, corner, Y, Z)
        
        # Update status
        self._update_calibration_status()
        
        # Visual feedback
        original_style = self.current_pos_label.styleSheet()
        self.current_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 13pt; "
            "background-color: #4CAF50; color: white; padding: 10px; "
            "font-weight: bold; }"
        )
        QTimer.singleShot(500, lambda: self.current_pos_label.setStyleSheet(
            original_style
        ))
        
        self.signals.status_message.emit(
            f"📷 Captured Block {block_id} {corner} at ({Y:.3f}, {Z:.3f})"
        )
    
    def _clear_all_fiducials(self):
        """Clear all captured fiducials."""
        reply = QMessageBox.warning(
            self,
            "Clear All Fiducials",
            "Delete all manually AND automatically captured positions?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.runtime_layout.clear_all_captured_fiducials()
            
            # Save
            main_window = self.window()
            if hasattr(main_window, 'runtime_file_path'):
                self.runtime_layout.save_to_json(main_window.runtime_file_path)
            
            # Clear table
            self.fiducial_table.setRowCount(0)
            
            # Update status
            self._update_calibration_status()
            
            self.signals.status_message.emit("All fiducials cleared")
    
    def _update_current_position(self):
        """Update current position display."""
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        self.current_pos_label.setText(
            f"Current Stage: Y={y:.3f}, Z={z:.3f} µm"
        )
    
    # [Rest of calibration methods unchanged - check_global_requirements, etc.]
    # ... (keeping the same logic as before)
    
    def _update_calibration_status(self):
        """Update calibration status text and button."""
        if self.radio_global.isChecked():
            self._check_global_requirements()
        else:
            self._check_block_requirements()
    
    def _check_global_requirements(self):
        """Check if global calibration can run."""
        fiducials = self.runtime_layout.get_all_captured_fiducials()
        
        block1_fids = [f for f in fiducials if f['block_id'] == 1]
        block20_fids = [f for f in fiducials if f['block_id'] == 20]
        
        can_calibrate = len(block1_fids) >= 2 and len(block20_fids) >= 2
        
        status_lines = []
        
        if can_calibrate:
            status_lines.append("✅ Ready to calibrate!\n")
        else:
            status_lines.append("⚠️ Need more fiducials\n")
        
        status_lines.append("Global calibration requirements:")
        
        if len(block1_fids) >= 2:
            corners = ', '.join(f['corner'] for f in block1_fids)
            status_lines.append(f"  ✅ Block 1: {len(block1_fids)} fiducials ({corners})")
        else:
            status_lines.append(f"  ⚠️ Block 1: {len(block1_fids)}/2 fiducials (need {2-len(block1_fids)} more)")
        
        if len(block20_fids) >= 2:
            corners = ', '.join(f['corner'] for f in block20_fids)
            status_lines.append(f"  ✅ Block 20: {len(block20_fids)} fiducials ({corners})")
        else:
            status_lines.append(f"  ⚠️ Block 20: {len(block20_fids)}/2 fiducials (need {2-len(block20_fids)} more)")
        
        self.status_text.setText('\n'.join(status_lines))
        self.btn_run_calibration.setEnabled(can_calibrate)
        
        if can_calibrate:
            self.btn_run_calibration.setText("▶️ Run Global Calibration")
    
    def _check_block_requirements(self):
        """Check if block calibration can run."""
        block_id = int(self.block_calib_combo.currentText())
        fiducials = self.runtime_layout.get_all_captured_fiducials()
        
        block_fids = [f for f in fiducials if f['block_id'] == block_id]
        global_done = self.state.global_calibrated
        
        can_calibrate = global_done and len(block_fids) >= 2
        
        status_lines = []
        
        if can_calibrate:
            status_lines.append(f"✅ Ready to calibrate Block {block_id}!\n")
        else:
            status_lines.append("⚠️ Cannot calibrate block yet\n")
        
        status_lines.append(f"Block {block_id} calibration requirements:")
        
        if global_done:
            status_lines.append("  ✅ Global calibration: Complete")
        else:
            status_lines.append("  ❌ Global calibration: NOT DONE")
        
        if len(block_fids) >= 2:
            corners = ', '.join(f['corner'] for f in block_fids)
            status_lines.append(f"  ✅ Block {block_id}: {len(block_fids)} fiducials ({corners})")
        else:
            status_lines.append(f"  ⚠️ Block {block_id}: {len(block_fids)}/2 fiducials (need {2-len(block_fids)} more)")
        
        self.status_text.setText('\n'.join(status_lines))
        self.btn_run_calibration.setEnabled(can_calibrate)
        
        if can_calibrate:
            self.btn_run_calibration.setText(f"▶️ Run Block {block_id} Calibration")
    
    def _run_calibration(self):
        """Execute calibration based on selected type."""
        if self.radio_global.isChecked():
            self._run_global_calibration()
        else:
            block_id = int(self.block_calib_combo.currentText())
            self._run_block_calibration(block_id)
    
    def _run_global_calibration(self):
        """Run manual global calibration."""
        fiducials = self.runtime_layout.get_all_captured_fiducials()
        
        measurements = [
            {
                'block_id': f['block_id'],
                'corner': f['corner'],
                'stage_Y': f['Y'],
                'stage_Z': f['Z'],
                'confidence': 1.0,
                'verification_error_um': 0.0
            }
            for f in fiducials
        ]
        
        try:
            result = self.alignment_controller.alignment_system.calibrate_global(
                measurements
            )
            
            self.runtime_layout.set_global_calibration(
                rotation=result['rotation_deg'],
                translation=result['translation_um'],
                calibration_error=result['mean_error_um'],
                num_points=len(measurements)
            )
            
            self.state.global_calibrated = True
            self.state.global_calibration_params = result
            
            from app.system_state import AlignmentStatus
            for bid in range(1, 21):
                self.state.set_block_status(bid, AlignmentStatus.GLOBAL_ONLY)
            
            main_window = self.window()
            if hasattr(main_window, 'runtime_file_path'):
                self.runtime_layout.save_to_json(main_window.runtime_file_path)
            
            self.signals.global_alignment_complete.emit(result)
            self._update_calibration_status()
            
            trans = result['translation_um']
            QMessageBox.information(
                self,
                "Global Calibration Complete",
                f"✅ Success!\n\n"
                f"Rotation: {result['rotation_deg']:.6f}°\n"
                f"Translation: ({trans[0]:.2f}, {trans[1]:.2f}) µm\n"
                f"Error: {result['mean_error_um']:.6f} µm"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Calibration Failed",
                f"Failed to run global calibration:\n\n{e}"
            )
            import traceback
            traceback.print_exc()
    
    def _run_block_calibration(self, block_id: int):
        """Run manual block calibration."""
        fiducials = self.runtime_layout.get_all_captured_fiducials()
        block_fids = [f for f in fiducials if f['block_id'] == block_id]
        
        measurements = [
            {
                'corner': f['corner'],
                'stage_Y': f['Y'],
                'stage_Z': f['Z']
            }
            for f in block_fids
        ]
        
        try:
            result = self.alignment_controller.alignment_system.calibrate_block(
                block_id, measurements
            )
            
            self.runtime_layout.set_block_calibration(
                block_id=block_id,
                rotation=result['rotation_deg'],
                translation=result['origin_stage_um'],
                calibration_error=result['mean_error_um'],
                num_points=len(measurements)
            )
            
            from app.system_state import AlignmentStatus
            self.state.set_block_status(
                block_id,
                AlignmentStatus.BLOCK_CALIBRATED,
                error=result['mean_error_um'],
                fiducials_found=len(measurements)
            )
            
            main_window = self.window()
            if hasattr(main_window, 'runtime_file_path'):
                self.runtime_layout.save_to_json(main_window.runtime_file_path)
            
            self.signals.block_alignment_complete.emit(block_id, result)
            self._update_calibration_status()
            
            QMessageBox.information(
                self,
                "Block Calibration Complete",
                f"✅ Block {block_id} calibrated!\n\n"
                f"Error: {result['mean_error_um']:.6f} µm"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Calibration Failed",
                f"Failed to calibrate Block {block_id}:\n\n{e}"
            )
            import traceback
            traceback.print_exc()