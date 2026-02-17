"""
Filter Stage Control Panel - WITH MULTI-POSITION SWEEP

NEW FEATURES:
- Multi-position sweep from saved positions
- Position selection with checkboxes
- Automatic subfolder organization by position name
- Progress tracking for position + sweep
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QLineEdit, QFileDialog,
    QListWidget, QListWidgetItem, QCheckBox
)

from PyQt6.QtCore import QTimer, Qt
import json
from pathlib import Path
import os
from datetime import datetime


class FilterPanelWidget(QWidget):
    """Control panel for filter stage with multi-position sweep."""
    
    POSITIONS_FILE = "config/saved_positions.json"
    
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

        # ✅ NEW: Connect sweep completion signal
        self.signals.busy_ended.connect(self._on_sweep_finished)

        # Load saved positions
        self._refresh_saved_positions()

    def _on_sweep_finished(self):
        """✅ NEW: Re-enable buttons when sweep finishes."""
        print("[FilterPanel] Sweep finished - re-enabling buttons")
        self.btn_run_sweep.setEnabled(True)
        self.btn_run_multi.setEnabled(True)
        self.btn_cancel.setEnabled(False)
    
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
        
        # Go to position
        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Target Position:"))
        
        self.goto_spin = QDoubleSpinBox()
        self.goto_spin.setRange(-15000, 15000)
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
        
        # Quick positions
        quick_row1 = QHBoxLayout()
        quick_row1.addWidget(QLabel("Negative:"))
        
        for pos_um in [-15000, -10000, -5000, -1000]:
            btn = QPushButton(f"{pos_um}µm")
            btn.clicked.connect(lambda checked, p=pos_um: self.filter.move_to_position(p))
            quick_row1.addWidget(btn)
        
        manual_layout.addLayout(quick_row1)
        
        quick_row2 = QHBoxLayout()
        quick_row2.addWidget(QLabel("Positive:"))
        
        for pos_um in [0, 1000, 5000, 10000, 15000]:
            btn = QPushButton(f"{pos_um}µm")
            btn.clicked.connect(lambda checked, p=pos_um: self.filter.move_to_position(p))
            quick_row2.addWidget(btn)
        
        manual_layout.addLayout(quick_row2)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        # ========================================
        # ✅ NEW: MULTI-POSITION SWEEP
        # ========================================
        multi_group = QGroupBox("Multi-Position Sweep")
        multi_layout = QVBoxLayout()
        
        # Info
        info = QLabel(
            "Run filter sweeps at multiple saved XYZ positions.\n"
            "Each position will get its own subfolder."
        )
        info.setStyleSheet("QLabel { color: #888; font-style: italic; font-size: 9pt; }")
        info.setWordWrap(True)
        multi_layout.addWidget(info)
        
        # Position list with checkboxes
        positions_label = QLabel("Select Positions:")
        positions_label.setStyleSheet("QLabel { font-weight: bold; margin-top: 8px; }")
        multi_layout.addWidget(positions_label)
        
        self.position_list = QListWidget()
        self.position_list.setMaximumHeight(150)
        self.position_list.setStyleSheet(
            "QListWidget { "
            "  background-color: #E0E0E0; "  # Light gray background
            "} "
            "QListWidget::item:selected { "
            "  background-color: #B0B0B0; "  # Darker gray when selected
            "}"
        )
        multi_layout.addWidget(self.position_list)
        
        # Selection controls
        select_row = QHBoxLayout()
        
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all_positions)
        select_row.addWidget(btn_select_all)
        
        btn_select_none = QPushButton("Clear All")
        btn_select_none.clicked.connect(self._select_no_positions)
        select_row.addWidget(btn_select_none)
        
        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self._refresh_saved_positions)
        select_row.addWidget(btn_refresh)
        
        select_row.addStretch()
        multi_layout.addLayout(select_row)
        
        # Selected count
        self.selected_count_label = QLabel("Selected: 0 positions")
        self.selected_count_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        multi_layout.addWidget(self.selected_count_label)
        
        multi_group.setLayout(multi_layout)
        layout.addWidget(multi_group)
        
        # ========================================
        # SWEEP CONFIGURATION
        # ========================================
        sweep_group = QGroupBox("Sweep Configuration")
        sweep_layout = QVBoxLayout()
        
        # Range
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start:"))
        self.sweep_start = QDoubleSpinBox()
        self.sweep_start.setRange(-15000, 15000)
        self.sweep_start.setValue(-15000)
        self.sweep_start.setSuffix(" µm")
        self.sweep_start.setDecimals(3)
        range_row.addWidget(self.sweep_start)
        
        range_row.addWidget(QLabel("End:"))
        self.sweep_end = QDoubleSpinBox()
        self.sweep_end.setRange(-15000, 15000)
        self.sweep_end.setValue(15000)
        self.sweep_end.setSuffix(" µm")
        self.sweep_end.setDecimals(3)
        range_row.addWidget(self.sweep_end)
        
        sweep_layout.addLayout(range_row)
        
        # Step size
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step Size:"))
        self.sweep_step = QDoubleSpinBox()
        self.sweep_step.setRange(0.001, 1000)
        self.sweep_step.setValue(100.0)
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
        self.settle_spin.setRange(0.0, 30.0)  # ✅ Changed from (0.1, 5.0)
        self.settle_spin.setValue(0.5)
        self.settle_spin.setSuffix(" s")
        self.settle_spin.setDecimals(2)
        self.settle_spin.setToolTip(
            "Wait time AFTER filter movement, BEFORE image capture.\n"
            "Allows mechanical vibrations to dampen.\n\n"
            "Sequence:\n"
            "1. Move filter stage → 2. Wait (settle time) → 3. Capture image (exposure time)\n\n"
            "Image is NOT taken during movement - this prevents motion blur."
        )
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
        
        # Single position sweep
        self.btn_run_sweep = QPushButton("▶️ Run Sweep (Current)")
        self.btn_run_sweep.clicked.connect(self._run_single_sweep)
        self.btn_run_sweep.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 12px; font-size: 11pt; }"
        )
        control_row.addWidget(self.btn_run_sweep)
        
        # ✅ NEW: Multi-position sweep button
        self.btn_run_multi = QPushButton("▶️▶️ Run Multi-Position Sweep")
        self.btn_run_multi.clicked.connect(self._run_multi_sweep)
        self.btn_run_multi.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 12px; font-size: 11pt; }"
        )
        control_row.addWidget(self.btn_run_multi)
        
        layout.addLayout(control_row)
        
        # Cancel button (separate row)
        cancel_row = QHBoxLayout()
        self.btn_cancel = QPushButton("⏹ Cancel")
        self.btn_cancel.clicked.connect(self.filter.cancel_sweep)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 8px; }"
        )
        cancel_row.addWidget(self.btn_cancel)
        layout.addLayout(cancel_row)
        
        # Initial position count
        self._update_position_count()
        
        layout.addStretch()
    
    def _refresh_saved_positions(self):
        """Load and display saved positions."""
        self.position_list.clear()
        
        path = Path(self.POSITIONS_FILE)
        if not path.exists():
            self.selected_count_label.setText("No saved positions found")
            return
        
        try:
            with open(path, 'r') as f:
                positions = json.load(f)
            
            for name in sorted(positions.keys()):
                pos = positions[name]
                
                # Create list item with checkbox
                item = QListWidgetItem()
                checkbox = QCheckBox(
                    f"{name}  →  X={pos['x']:.3f}, Y={pos['y']:.3f}, Z={pos['z']:.3f} µm"
                )
                # Black text on light gray background
                checkbox.setStyleSheet("QCheckBox { color: black; }")
                checkbox.stateChanged.connect(self._update_selected_count)
                
                self.position_list.addItem(item)
                self.position_list.setItemWidget(item, checkbox)
                
                # Store position data
                item.setData(Qt.ItemDataRole.UserRole, {
                    'name': name,
                    'x': pos['x'],
                    'y': pos['y'],
                    'z': pos['z']
                })
            
            self._update_selected_count()
            
        except Exception as e:
            self.selected_count_label.setText(f"Error loading positions: {e}")
    
    def _select_all_positions(self):
        """Select all positions."""
        for i in range(self.position_list.count()):
            item = self.position_list.item(i)
            widget = self.position_list.itemWidget(item)
            if isinstance(widget, QCheckBox):
                widget.setChecked(True)
    
    def _select_no_positions(self):
        """Deselect all positions."""
        for i in range(self.position_list.count()):
            item = self.position_list.item(i)
            widget = self.position_list.itemWidget(item)
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
    
    def _update_selected_count(self):
        """Update selected position count."""
        selected = self._get_selected_positions()
        count = len(selected)
        
        if count == 0:
            self.selected_count_label.setText("Selected: 0 positions")
            self.selected_count_label.setStyleSheet("QLabel { color: #888; }")
        else:
            self.selected_count_label.setText(f"Selected: {count} position{'s' if count != 1 else ''}")
            self.selected_count_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
    
    def _get_selected_positions(self) -> list:
        """Get list of selected positions."""
        selected = []
        
        for i in range(self.position_list.count()):
            item = self.position_list.item(i)
            widget = self.position_list.itemWidget(item)
            
            if isinstance(widget, QCheckBox) and widget.isChecked():
                pos_data = item.data(Qt.ItemDataRole.UserRole)
                selected.append(pos_data)
        
        return selected
    
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
        """Move to position (button callback)."""
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
    
    def _get_desktop_path(self):
        """Get user's Desktop folder path."""
        home = Path.home()
        # Try Desktop (works on Windows, macOS, most Linux)
        desktop = home / "Desktop"
        if desktop.exists():
            return str(desktop)
        # Try Russian localization
        desktop = home / "Рабочий стол"
        if desktop.exists():
            return str(desktop)
        # Try German localization
        desktop = home / "Schreibtisch"
        if desktop.exists():
            return str(desktop)
        # Fallback to home directory
        return str(home)

    def _browse_output(self):
        """Browse for output directory - default to Desktop."""
        from PyQt6.QtWidgets import QFileDialog
        # Get Desktop path
        desktop = self._get_desktop_path()
        # If output field is empty, start at Desktop
        current = self.output_edit.text()
        if not current:
            start_dir = desktop
        else:
            start_dir = current
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            start_dir,  # Start at Desktop
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_edit.setText(directory)
    
    def _run_single_sweep(self):
        """Start single-position sweep."""
        start = self.sweep_start.value()
        end = self.sweep_end.value()
        step = self.sweep_step.value()
        settle = self.settle_spin.value()

        # Get output directory
        output = self.output_edit.text().strip()
        if not output:
            # ✅ Auto-generate on Desktop
            desktop = self._get_desktop_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(Path(desktop) / f"filter_sweep_{timestamp}")
            print(f"[FilterPanel] Auto-generated output: {output}")

        success = self.filter.run_sweep(
            start_um=start,
            end_um=end,
            step_um=step,
            output_dir=output,
            settle_time_s=settle
        )

        if success:
            self._set_sweep_running(True)
    
    def _run_multi_sweep(self):
        """Start multi-position sweep."""
        selected = self._get_selected_positions()

        if len(selected) == 0:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Positions Selected",
                "Please select at least one position for multi-position sweep."
            )
            return

        # Get sweep parameters
        start = self.sweep_start.value()
        end = self.sweep_end.value()
        step = self.sweep_step.value()
        settle = self.settle_spin.value()

        # Get output directory
        output = self.output_edit.text().strip()
        if not output:
            # ✅ Auto-generate on Desktop
            desktop = self._get_desktop_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(Path(desktop) / f"multi_sweep_{timestamp}")
            print(f"[FilterPanel] Auto-generated output: {output}")

        # Confirm
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Multi-Position Sweep",
            f"Run filter sweep at {len(selected)} positions?\n\n"
            f"Filter range: {start:.1f} to {end:.1f} µm (step {step:.1f} µm)\n"
            f"Each position will get its own subfolder.\n\n"
            f"This may take a while. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Start multi-position sweep
        success = self.filter.run_multi_position_sweep(
            positions=selected,
            start_um=start,
            end_um=end,
            step_um=step,
            output_dir=output,
            settle_time_s=settle
        )

        if success:
            self._set_sweep_running(True)
    
    def _set_sweep_running(self, running: bool):
        """Update UI for running/stopped state."""
        self.btn_run_sweep.setEnabled(not running)
        self.btn_run_multi.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        # ❌ REMOVED: Don't connect signal here anymore!
        # The signal is now connected in __init__