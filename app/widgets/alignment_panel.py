"""Alignment Control Panel."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox
from PyQt6.QtCore import Qt


class AlignmentPanelWidget(QWidget):
    """Alignment controls and status display."""
    
    def __init__(self, state, signals, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        
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
        
        self.btn_run_global = QPushButton("Run Global Alignment")
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
        
        self.btn_calibrate_block = QPushButton("Calibrate Selected Block")
        self.btn_calibrate_block.clicked.connect(self._calibrate_block)
        self.btn_calibrate_block.setEnabled(False)
        block_layout.addWidget(self.btn_calibrate_block)
        
        self.btn_calibrate_all = QPushButton("Calibrate All Blocks")
        self.btn_calibrate_all.clicked.connect(self._calibrate_all)
        block_layout.addWidget(self.btn_calibrate_all)
        
        block_group.setLayout(block_layout)
        layout.addWidget(block_group)
        
        layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.block_selected.connect(self._update_block_status)
        self.signals.global_alignment_complete.connect(self._update_global_status)
        self.signals.block_alignment_complete.connect(lambda bid, res: self._update_block_status(bid))
    
    def _run_global(self):
        """Run global alignment."""
        self.signals.status_message.emit("Global alignment not yet implemented")
        # TODO: Implement
    
    def _calibrate_block(self):
        """Calibrate selected block."""
        if self.state.navigation.current_block is None:
            return
        
        block_id = self.state.navigation.current_block
        self.signals.status_message.emit(f"Block {block_id} calibration not yet implemented")
        # TODO: Implement
    
    def _calibrate_all(self):
        """Calibrate all blocks."""
        self.signals.status_message.emit("Batch calibration not yet implemented")
        # TODO: Implement
    
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
        else:
            self.global_status.setText("⚪ Not Calibrated")
            self.global_info.setText("Rotation: --\nTranslation: --\nError: --")
    
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
        self.btn_calibrate_block.setEnabled(True)