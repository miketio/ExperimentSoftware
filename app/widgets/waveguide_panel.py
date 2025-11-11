# app/widgets/waveguide_panel.py
"""Waveguide Navigation Panel - COMPLETE IMPLEMENTATION"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QSpinBox, QComboBox,
    QCheckBox, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class WaveguidePanelWidget(QWidget):
    """Waveguide table and navigation controls - COMPLETE."""
    
    def __init__(self, state, signals, stage, navigation_controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.stage = stage
        self.navigation = navigation_controller
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        group = QGroupBox("Waveguide Navigation")
        group_layout = QVBoxLayout()
        
        # Controls row
        controls = QHBoxLayout()
        
        # Target waveguide
        controls.addWidget(QLabel("Target WG:"))
        self.wg_spin = QSpinBox()
        self.wg_spin.setRange(1, 50)
        self.wg_spin.setValue(self.state.navigation.target_waveguide)
        self.wg_spin.valueChanged.connect(self._on_target_changed)
        controls.addWidget(self.wg_spin)
        
        # Side
        controls.addWidget(QLabel("Side:"))
        self.side_combo = QComboBox()
        self.side_combo.addItems(['left', 'center', 'right'])
        self.side_combo.setCurrentText(self.state.navigation.target_grating_side)
        self.side_combo.currentTextChanged.connect(self._on_side_changed)
        controls.addWidget(self.side_combo)
        
        # Autofocus option
        self.autofocus_check = QCheckBox("Autofocus")
        self.autofocus_check.setChecked(self.state.autofocus_config.get('auto_after_nav', True))
        self.autofocus_check.toggled.connect(self._on_autofocus_toggled)
        controls.addWidget(self.autofocus_check)
        
        # Go to target button
        self.btn_goto_target = QPushButton("🎯 Go to Target")
        self.btn_goto_target.clicked.connect(self.navigate_to_target)
        self.btn_goto_target.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; font-size: 11pt; }"
        )
        self.btn_goto_target.setEnabled(False)  # Disabled until block selected
        controls.addWidget(self.btn_goto_target)
        
        self.btn_goto_beam = QPushButton("🎯 Move to Beam")
        self.btn_goto_beam.setToolTip("Navigate to target, then offset to beam position")
        self.btn_goto_beam.clicked.connect(self.navigate_to_beam)
        self.btn_goto_beam.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 8px; font-size: 11pt; }"
        )
        self.btn_goto_beam.setEnabled(False)
        controls.addWidget(self.btn_goto_beam)

        # Cancel button
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.navigation.cancel_navigation)
        self.btn_cancel.setEnabled(False)
        controls.addWidget(self.btn_cancel)
        
        controls.addStretch()
        group_layout.addLayout(controls)
        
        # Info label
        self.info_label = QLabel("Select a block to view waveguides")
        self.info_label.setStyleSheet("QLabel { font-style: italic; color: #666; }")
        group_layout.addWidget(self.info_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'WG#', 'V-Center (µm)', 'Left', 'Center', 'Right', 'Status'
        ])
        self.table.setRowCount(50)
        self.table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 80)
        
        # Populate table (will be updated when block is selected)
        self._populate_table()
        
        group_layout.addWidget(self.table)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.block_selected.connect(self._on_block_selected)
        self.signals.navigation_started.connect(self._on_navigation_started)
        self.signals.navigation_complete.connect(self._on_navigation_complete)
        self.signals.navigation_failed.connect(self._on_navigation_failed)
    
    def _on_target_changed(self, wg_num: int):
        """Handle target waveguide change."""
        self.state.set_target_waveguide(wg_num)
        self.signals.target_waveguide_changed.emit(wg_num)
    
    def _on_side_changed(self, side: str):
        """Handle target side change."""
        self.state.navigation.target_grating_side = side
        self.signals.target_grating_changed.emit(side)
    
    def _on_autofocus_toggled(self, checked: bool):
        """Handle autofocus toggle."""
        self.state.autofocus_config['auto_after_nav'] = checked
    
    def _populate_table(self):
        """Populate table with waveguide data."""
        for i in range(50):
            wg_num = i + 1
            
            # WG number
            item = QTableWidgetItem(str(wg_num))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, item)
            
            # V-center (will update when block selected)
            item = QTableWidgetItem("--")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 1, item)
            
            # Go buttons for each side
            for col, side in enumerate(['left', 'center', 'right'], start=2):
                btn = QPushButton("Go")
                btn.setToolTip(f"Navigate to WG{wg_num} {side} grating")
                btn.clicked.connect(
                    lambda checked, w=wg_num, s=side: self._goto_waveguide(w, s)
                )
                btn.setEnabled(False)  # Disabled until block selected
                self.table.setCellWidget(i, col, btn)
            
            # Status
            item = QTableWidgetItem("")
            self.table.setItem(i, 5, item)
    
    # In waveguide_panel.py - UPDATE _on_block_selected method

    def _on_block_selected(self, block_id: int):
        """Handle block selection - show only available gratings."""
        print(f"[WaveguidePanel] Block {block_id} selected")
        
        runtime_layout = self.navigation.alignment.layout
        is_calibrated = runtime_layout.is_block_calibrated(block_id)
        
        if is_calibrated:
            self.info_label.setText(f"✅ Block {block_id} selected (calibrated)")
            self.info_label.setStyleSheet("QLabel { font-style: normal; color: green; font-weight: bold; }")
        else:
            self.info_label.setText(f"⚠️ Block {block_id} selected (NOT calibrated - using global only)")
            self.info_label.setStyleSheet("QLabel { font-style: italic; color: orange; font-weight: bold; }")
        
        self.btn_goto_target.setEnabled(True)
        self.btn_goto_beam.setEnabled(True)  # NEW
        try:
            block = runtime_layout.get_block(block_id)
            available_wgs = block.list_waveguides()
            
            print(f"[WaveguidePanel] Block {block_id} has {len(available_wgs)} waveguides")
            
            for i in range(50):
                wg_num = i + 1
                
                if wg_num in available_wgs:
                    wg = block.get_waveguide(wg_num)
                    
                    # Update V-center
                    item = self.table.item(i, 1)
                    item.setText(f"{wg.v_center:.2f}")
                    
                    # Check which gratings exist for this waveguide
                    grating_sides = ['left', 'center', 'right']
                    for col, side in enumerate(grating_sides, start=2):
                        btn = self.table.cellWidget(i, col)
                        if btn:
                            # Check if this grating exists
                            has_grating = False
                            
                            if side == 'center':
                                # Center always exists (waveguide center)
                                has_grating = True
                            else:
                                # Check if left/right grating exists
                                try:
                                    grating = block.get_grating(wg_num, side)
                                    has_grating = (grating is not None)
                                except Exception:
                                    has_grating = False
                            
                            if has_grating:
                                btn.setEnabled(True)
                                if is_calibrated:
                                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
                                else:
                                    btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
                            else:
                                # Gray out - no grating here
                                btn.setEnabled(False)
                                btn.setStyleSheet("QPushButton { background-color: #CCCCCC; color: #666; }")
                    
                    # Status
                    status_item = self.table.item(i, 5)
                    if is_calibrated:
                        status_item.setText("Ready (calibrated)")
                        status_item.setForeground(QColor(0, 128, 0))
                    else:
                        status_item.setText("⚠️ Global only")
                        status_item.setForeground(QColor(255, 140, 0))
                else:
                    # Waveguide doesn't exist
                    item = self.table.item(i, 1)
                    item.setText("--")
                    
                    for col in range(2, 5):
                        btn = self.table.cellWidget(i, col)
                        if btn:
                            btn.setEnabled(False)
                            btn.setStyleSheet("QPushButton { background-color: #EEEEEE; color: #999; }")
                    
                    status_item = self.table.item(i, 5)
                    status_item.setText("Not available")
                    status_item.setForeground(QColor(128, 128, 128))
            
            print(f"[WaveguidePanel] Table updated for Block {block_id}")

        except Exception as e:
            self.info_label.setText(f"❌ Error loading block data: {e}")
            self.info_label.setStyleSheet("QLabel { font-style: italic; color: red; font-weight: bold; }")
            print(f"[WaveguidePanel] Error loading block {block_id}: {e}")
            import traceback
            traceback.print_exc()

    def _goto_waveguide(self, wg_num: int, side: str):
        """Navigate to specific waveguide/grating."""
        if self.state.navigation.current_block is None:
            self.signals.error_occurred.emit(
                "No Block Selected",
                "Please select a block first"
            )
            return
        
        block_id = self.state.navigation.current_block
        autofocus = self.autofocus_check.isChecked()
        
        print(f"[WaveguidePanel] Navigating to Block {block_id} WG{wg_num} {side}")
        
        # Start navigation
        success = self.navigation.navigate_to_grating(
            block_id=block_id,
            waveguide=wg_num,
            side=side,
            autofocus=autofocus
        )
        
        if not success:
            # Error already emitted by navigation controller
            return
    
    def navigate_to_target(self):
        """Navigate to target waveguide/side."""
        wg = self.wg_spin.value()
        side = self.side_combo.currentText()
        print(f"[WaveguidePanel] Navigate to target: WG{wg} {side}")
        self._goto_waveguide(wg, side)
    
    def _on_navigation_started(self, block_id: int, waveguide: int, side: str):
        """Handle navigation start."""
        print(f"[WaveguidePanel] Navigation started: Block {block_id} WG{waveguide} {side}")
        
        self.btn_goto_target.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        
        # Disable all Go buttons during navigation
        for i in range(50):
            for col in range(2, 5):
                btn = self.table.cellWidget(i, col)
                if btn:
                    btn.setEnabled(False)
        
        # Update status in table
        row = waveguide - 1
        if 0 <= row < 50:
            status_item = self.table.item(row, 5)
            status_item.setText(f"🚀 Navigating to {side}...")
            status_item.setForeground(QColor(0, 100, 200))  # Blue
    
    def _on_navigation_complete(self):
        """Handle navigation completion."""
        print("[WaveguidePanel] Navigation complete")
        
        self.btn_goto_target.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        
        # Re-enable Go buttons
        if self.state.navigation.current_block is not None:
            self._on_block_selected(self.state.navigation.current_block)
    
    def _on_navigation_failed(self, error: str):
        """Handle navigation failure."""
        print(f"[WaveguidePanel] Navigation failed: {error}")
        
        self.btn_goto_target.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        
        # Re-enable Go buttons
        if self.state.navigation.current_block is not None:
            self._on_block_selected(self.state.navigation.current_block)
    
    def refresh_waveguide_list(self):
        """Refresh waveguide positions for selected block."""
        if self.state.navigation.current_block is not None:
            self._on_block_selected(self.state.navigation.current_block)

    # Add navigation method:
    def navigate_to_beam(self):
        """Navigate to target, then apply beam offset."""
        if self.state.navigation.current_block is None:
            self.signals.error_occurred.emit(
                "No Block Selected",
                "Please select a block first"
            )
            return
        
        wg = self.wg_spin.value()
        side = self.side_combo.currentText()
        
        # Get predicted grating position (centered)
        block_id = self.state.navigation.current_block
        autofocus = self.autofocus_check.isChecked()
        
        # Calculate beam offset in stage coordinates
        center_x, center_y = 1024, 1024  # Image center
        beam_x, beam_y = self.state.camera.beam_position_px
        
        offset_px_x = beam_x - center_x
        offset_px_y = beam_y - center_y
        
        # Convert pixel offset to stage offset (µm)
        um_per_pixel = self.state.camera.um_per_pixel
        offset_stage_y = -1*offset_px_x * um_per_pixel  # X pixels → Y stage
        offset_stage_z = offset_px_y * um_per_pixel  # Y pixels → Z stage
        
        print(f"[WaveguidePanel] Navigate to beam: WG{wg} {side}")
        print(f"  Beam offset: ({offset_px_x}, {offset_px_y}) px")
        print(f"  Stage offset: ({offset_stage_y:.3f}, {offset_stage_z:.3f}) µm")
        
        # Start navigation WITH beam offset
        success = self.navigation.navigate_to_grating_with_beam_offset(
            block_id=block_id,
            waveguide=wg,
            side=side,
            beam_offset_um=(offset_stage_y, offset_stage_z),
            autofocus=autofocus
        )
        
        if not success:
            return