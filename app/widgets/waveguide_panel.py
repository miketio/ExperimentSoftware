"""Waveguide Navigation Panel."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt


class WaveguidePanelWidget(QWidget):
    """Waveguide table and navigation controls."""
    
    def __init__(self, state, signals, stage, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.stage = stage
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        group = QGroupBox("Waveguide Navigation")
        group_layout = QVBoxLayout()
        
        # Controls
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Target WG:"))
        self.wg_spin = QSpinBox()
        self.wg_spin.setRange(1, 50)
        self.wg_spin.setValue(self.state.navigation.target_waveguide)
        self.wg_spin.valueChanged.connect(self._on_target_changed)
        controls.addWidget(self.wg_spin)
        
        controls.addWidget(QLabel("Side:"))
        self.side_combo = QComboBox()
        self.side_combo.addItems(['left', 'center', 'right'])
        self.side_combo.setCurrentText(self.state.navigation.target_grating_side)
        controls.addWidget(self.side_combo)
        
        self.btn_goto_target = QPushButton("Go to Target")
        self.btn_goto_target.clicked.connect(self.navigate_to_target)
        self.btn_goto_target.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        controls.addWidget(self.btn_goto_target)
        
        controls.addStretch()
        group_layout.addLayout(controls)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['WG#', 'Position (µm)', 'Left', 'Center', 'Right'])
        self.table.setRowCount(50)
        self.table.setAlternatingRowColors(True)
        
        # Populate table (stub)
        for i in range(50):
            wg_num = i + 1
            self.table.setItem(i, 0, QTableWidgetItem(str(wg_num)))
            self.table.setItem(i, 1, QTableWidgetItem(f"(?, ?)"))  # Will update with actual positions
            
            # Go buttons
            for col, side in enumerate(['left', 'center', 'right'], start=2):
                btn = QPushButton("Go")
                btn.clicked.connect(lambda checked, w=wg_num, s=side: self._goto_waveguide(w, s))
                self.table.setCellWidget(i, col, btn)
        
        group_layout.addWidget(self.table)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.block_selected.connect(lambda bid: self.refresh_waveguide_list())
    
    def _on_target_changed(self, wg_num: int):
        """Handle target waveguide change."""
        self.state.set_target_waveguide(wg_num)
        self.signals.target_waveguide_changed.emit(wg_num)
    
    def _goto_waveguide(self, wg_num: int, side: str):
        """Navigate to specific waveguide/grating."""
        if self.state.navigation.current_block is None:
            self.signals.error_occurred.emit("No Block Selected", "Please select a block first")
            return
        
        block_id = self.state.navigation.current_block
        
        self.signals.navigation_started.emit(block_id, wg_num, side)
        self.signals.status_message.emit(f"Navigation to Block {block_id} WG{wg_num} {side} not yet implemented")
        
        # TODO: Implement actual navigation using AlignmentSystem
    
    def navigate_to_target(self):
        """Navigate to target waveguide."""
        wg = self.wg_spin.value()
        side = self.side_combo.currentText()
        self._goto_waveguide(wg, side)
    
    def refresh_waveguide_list(self):
        """Refresh waveguide positions for selected block."""
        if self.state.navigation.current_block is None:
            return
        
        # TODO: Load actual positions from layout/alignment system
        # For now, just update with placeholder
        self.signals.status_message.emit("Waveguide list refreshed")