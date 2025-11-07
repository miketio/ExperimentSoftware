# app/widgets/alignment_panel.py
"""Alignment Control Panel - Updated with controller integration."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QSpinBox, QCheckBox
from PyQt6.QtCore import Qt


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