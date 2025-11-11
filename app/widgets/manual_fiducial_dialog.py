# app/widgets/manual_fiducial_dialog.py
"""
Manual Fiducial Capture Dialog

Allows user to manually capture fiducial positions by:
1. Selecting block + corner
2. Moving stage to fiducial location
3. Capturing position
4. Saving to calibration
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QMessageBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import QTimer, Qt


class ManualFiducialDialog(QDialog):
    """Dialog for manually capturing fiducial positions."""
    
    def __init__(self, state, runtime_layout, alignment_controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.runtime_layout = runtime_layout
        self.alignment_controller = alignment_controller
        
        self.setWindowTitle("Manual Fiducial Capture")
        self.setModal(True)
        self.setMinimumWidth(600)
        
        self.captured_fiducials = []
        
        self._init_ui()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_position)
        self.timer.start(200)
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        
        # Info
        info = QLabel(
            "⭐ <b>Manual Fiducial Calibration</b>\n\n"
            "1. Select block and corner from dropdowns\n"
            "2. Use stage controls to position camera on fiducial\n"
            "3. Click 'Capture' to save position\n"
            "4. Repeat for at least 2 fiducials\n"
            "5. Click 'Run Calibration' to apply"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Selection group
        selection_group = QGroupBox("Fiducial Selection")
        selection_layout = QHBoxLayout()
        
        selection_layout.addWidget(QLabel("Block:"))
        self.block_combo = QComboBox()
        self.block_combo.addItems([str(i) for i in range(1, 21)])
        selection_layout.addWidget(self.block_combo)
        
        selection_layout.addWidget(QLabel("Corner:"))
        self.corner_combo = QComboBox()
        self.corner_combo.addItems(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        selection_layout.addWidget(self.corner_combo)
        
        selection_layout.addStretch()
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Current position
        pos_group = QGroupBox("Current Stage Position")
        pos_layout = QVBoxLayout()
        
        self.pos_label = QLabel("Y = ?.???, Z = ?.???")
        self.pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 16pt; "
            "font-weight: bold; background-color: black; color: lime; "
            "padding: 15px; }"
        )
        pos_layout.addWidget(self.pos_label)
        
        btn_capture = QPushButton("📷 Capture Fiducial Position")
        btn_capture.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 12px; font-size: 12pt; }"
        )
        btn_capture.clicked.connect(self._capture_fiducial)
        pos_layout.addWidget(btn_capture)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Captured list
        list_group = QGroupBox("Captured Fiducials")
        list_layout = QVBoxLayout()
        
        self.fiducial_list = QListWidget()
        list_layout.addWidget(self.fiducial_list)
        
        btn_layout = QHBoxLayout()
        
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected)
        btn_layout.addWidget(btn_remove)
        
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_all)
        btn_layout.addWidget(btn_clear)
        
        btn_layout.addStretch()
        list_layout.addLayout(btn_layout)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Action buttons
        buttons = QHBoxLayout()
        
        self.btn_calibrate = QPushButton("✅ Run Calibration")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self._run_calibration)
        self.btn_calibrate.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt; }"
        )
        buttons.addWidget(self.btn_calibrate)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        buttons.addWidget(btn_close)
        
        layout.addLayout(buttons)
        self.setLayout(layout)
    
    def _update_position(self):
        """Update current position display."""
        y, z = self.state.stage_position['y'], self.state.stage_position['z']
        self.pos_label.setText(f"Y = {y:.3f} µm, Z = {z:.3f} µm")
    
    def _capture_fiducial(self):
        """Capture current fiducial position."""
        block_id = int(self.block_combo.currentText())
        corner = self.corner_combo.currentText()
        
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        
        # Check if already captured
        for fid in self.captured_fiducials:
            if fid['block_id'] == block_id and fid['corner'] == corner:
                reply = QMessageBox.question(
                    self,
                    "Already Captured",
                    f"Block {block_id} {corner} already captured.\n\nOverwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                else:
                    self.captured_fiducials.remove(fid)
                    # Remove from list widget
                    for i in range(self.fiducial_list.count()):
                        item = self.fiducial_list.item(i)
                        if item.data(Qt.ItemDataRole.UserRole) == (block_id, corner):
                            self.fiducial_list.takeItem(i)
                            break
        
        # Add to captured list
        fiducial = {
            'block_id': block_id,
            'corner': corner,
            'stage_Y': y,
            'stage_Z': z,
            'confidence': 1.0,  # Manual capture
            'verification_error_um': 0.0
        }
        self.captured_fiducials.append(fiducial)
        
        # Add to list widget
        item = QListWidgetItem(
            f"Block {block_id} {corner}: Y={y:.3f}µm, Z={z:.3f}µm"
        )
        item.setData(Qt.ItemDataRole.UserRole, (block_id, corner))
        self.fiducial_list.addItem(item)
        
        # Flash green
        self.pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 16pt; "
            "font-weight: bold; background-color: green; color: white; "
            "padding: 15px; }"
        )
        QTimer.singleShot(500, lambda: self.pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 16pt; "
            "font-weight: bold; background-color: black; color: lime; "
            "padding: 15px; }"
        ))
        
        # Enable calibration if we have enough
        self.btn_calibrate.setEnabled(len(self.captured_fiducials) >= 2)
        
        print(f"[ManualFiducial] Captured Block {block_id} {corner}: ({y:.3f}, {z:.3f}) µm")
    
    def _remove_selected(self):
        """Remove selected fiducial from list."""
        current = self.fiducial_list.currentItem()
        if current:
            block_id, corner = current.data(Qt.ItemDataRole.UserRole)
            
            # Remove from captured list
            self.captured_fiducials = [
                f for f in self.captured_fiducials
                if not (f['block_id'] == block_id and f['corner'] == corner)
            ]
            
            # Remove from widget
            self.fiducial_list.takeItem(self.fiducial_list.row(current))
            
            # Update button
            self.btn_calibrate.setEnabled(len(self.captured_fiducials) >= 2)
    
    def _clear_all(self):
        """Clear all captured fiducials."""
        reply = QMessageBox.question(
            self,
            "Clear All",
            "Remove all captured fiducials?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.captured_fiducials.clear()
            self.fiducial_list.clear()
            self.btn_calibrate.setEnabled(False)
    
    def _run_calibration(self):
        """Run calibration with captured fiducials."""
        if len(self.captured_fiducials) < 2:
            QMessageBox.warning(
                self,
                "Not Enough Fiducials",
                "At least 2 fiducials required for calibration."
            )
            return
        
        # Determine if this is global or block calibration
        unique_blocks = set(f['block_id'] for f in self.captured_fiducials)
        
        if len(unique_blocks) > 1:
            # Global calibration (multiple blocks)
            print("[ManualFiducial] Running global calibration")
            
            try:
                result = self.alignment_controller.alignment_system.calibrate_global(
                    self.captured_fiducials
                )
                
                # Update runtime layout
                self.runtime_layout.set_global_calibration(
                    rotation=result['rotation_deg'],
                    translation=result['translation_um'],
                    calibration_error=result['mean_error_um'],
                    num_points=len(self.captured_fiducials)
                )
                
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Global calibration successful!\n\n"
                    f"Rotation: {result['rotation_deg']:.6f}°\n"
                    f"Translation: {result['translation_um']}\n"
                    f"Error: {result['mean_error_um']:.6f} µm"
                )
                
                self.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Calibration Failed",
                    f"Global calibration failed:\n\n{e}"
                )
        else:
            # Block calibration (single block)
            block_id = list(unique_blocks)[0]
            print(f"[ManualFiducial] Running block {block_id} calibration")
            
            try:
                result = self.alignment_controller.alignment_system.calibrate_block(
                    block_id,
                    self.captured_fiducials
                )
                
                # Update runtime layout
                self.runtime_layout.set_block_calibration(
                    block_id=block_id,
                    rotation=result['rotation_deg'],
                    translation=result['origin_stage_um'],
                    calibration_error=result['mean_error_um'],
                    num_points=len(self.captured_fiducials)
                )
                
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Block {block_id} calibration successful!\n\n"
                    f"Error: {result['mean_error_um']:.6f} µm"
                )
                
                self.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Calibration Failed",
                    f"Block {block_id} calibration failed:\n\n{e}"
                )