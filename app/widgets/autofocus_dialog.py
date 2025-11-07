"""
Autofocus Dialog

Dialog for configuring and running autofocus scans with live progress.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QProgressBar, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np


class AutofocusDialog(QDialog):
    """
    Dialog for running autofocus with live progress and plotting.
    
    Features:
    - Axis selection (X, Y, Z)
    - Scan range configuration
    - Step size configuration
    - Live progress bar
    - Live plot of focus metric vs position
    - Cancel button
    """
    
    def __init__(self, autofocus_controller, state, parent=None):
        """
        Initialize autofocus dialog.
        
        Args:
            autofocus_controller: AutofocusController instance
            state: SystemState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.autofocus = autofocus_controller
        self.state = state
        
        self.setWindowTitle("Autofocus")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._init_ui()
        self._connect_signals()
        
        # Plot data
        self.plot_positions = []
        self.plot_metrics = []
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Configuration
        config_group = QGroupBox("Autofocus Configuration")
        config_layout = QVBoxLayout()
        
        # Axis selection
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Axis:"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(['X (Focus)', 'Y', 'Z'])
        self.axis_combo.setCurrentIndex(0)
        axis_layout.addWidget(self.axis_combo)
        axis_layout.addStretch()
        config_layout.addLayout(axis_layout)
        
        # Scan range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Scan Range:"))
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1.0, 100.0)
        self.range_spin.setValue(self.state.autofocus_config.get('range_um', 10.0))
        self.range_spin.setSuffix(" µm")
        self.range_spin.setDecimals(1)
        range_layout.addWidget(self.range_spin)
        range_layout.addStretch()
        config_layout.addLayout(range_layout)
        
        # Step size
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 5.0)
        self.step_spin.setValue(self.state.autofocus_config.get('step_um', 0.5))
        self.step_spin.setSuffix(" µm")
        self.step_spin.setDecimals(2)
        step_layout.addWidget(self.step_spin)
        step_layout.addStretch()
        config_layout.addLayout(step_layout)
        
        # Enable plot
        self.plot_check = QCheckBox("Show live plot")
        self.plot_check.setChecked(True)
        config_layout.addWidget(self.plot_check)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to start")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; }")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.position_label = QLabel("Position: --")
        self.position_label.setStyleSheet("QLabel { font-family: monospace; }")
        progress_layout.addWidget(self.position_label)
        
        self.metric_label = QLabel("Focus Metric: --")
        self.metric_label.setStyleSheet("QLabel { font-family: monospace; }")
        progress_layout.addWidget(self.metric_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Plot (optional - requires pyqtgraph)
        try:
            plot_group = QGroupBox("Focus Metric Plot")
            plot_layout = QVBoxLayout()
            
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel('left', 'Focus Metric')
            self.plot_widget.setLabel('bottom', 'Position', units='µm')
            self.plot_widget.showGrid(x=True, y=True)
            self.plot_widget.setMinimumHeight(200)
            
            plot_layout.addWidget(self.plot_widget)
            plot_group.setLayout(plot_layout)
            layout.addWidget(plot_group)
            
            self.has_plot = True
        except:
            # pyqtgraph not available
            self.has_plot = False
            self.plot_widget = None
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start Autofocus")
        self.btn_start.clicked.connect(self._start_autofocus)
        self.btn_start.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.btn_start)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_autofocus)
        self.btn_cancel.setEnabled(False)
        button_layout.addWidget(self.btn_cancel)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        button_layout.addWidget(self.btn_close)
        
        layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """Connect signals."""
        signals = self.autofocus.signals
        
        signals.autofocus_started.connect(self._on_started)
        signals.autofocus_progress.connect(self._on_progress)
        signals.autofocus_complete.connect(self._on_complete)
        signals.autofocus_failed.connect(self._on_failed)
        signals.autofocus_cancelled.connect(self._on_cancelled)
        
        # Plot data (if worker emits it)
        if hasattr(self.autofocus.worker, 'plot_data'):
            try:
                self.autofocus.worker.plot_data.connect(self._update_plot)
            except:
                pass
    
    def _start_autofocus(self):
        """Start autofocus scan."""
        # Get parameters
        axis_text = self.axis_combo.currentText()
        axis = axis_text[0].lower()  # Extract 'x', 'y', or 'z'
        scan_range = self.range_spin.value()
        step_size = self.step_spin.value()
        enable_plot = self.plot_check.isChecked()
        
        # Update config
        self.state.autofocus_config['range_um'] = scan_range
        self.state.autofocus_config['step_um'] = step_size
        
        # Clear previous plot
        if self.has_plot and self.plot_widget:
            self.plot_widget.clear()
        self.plot_positions = []
        self.plot_metrics = []
        
        # Start autofocus
        success = self.autofocus.run_autofocus(
            axis=axis,
            scan_range_um=scan_range,
            step_um=step_size,
            enable_plot=enable_plot
        )
        
        if success:
            self.btn_start.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.btn_close.setEnabled(False)
    
    def _cancel_autofocus(self):
        """Cancel running autofocus."""
        self.autofocus.cancel()
    
    def _on_started(self, axis: str):
        """Handle autofocus start."""
        self.status_label.setText(f"Scanning {axis.upper()}-axis...")
        self.progress_bar.setValue(0)
    
    def _on_progress(self, position: float, metric: float, progress: float):
        """Handle progress update."""
        self.progress_bar.setValue(int(progress))
        self.position_label.setText(f"Position: {position:.3f} µm")
        self.metric_label.setText(f"Focus Metric: {metric:.2f}")
        
        # Update plot
        if self.has_plot and self.plot_widget:
            self.plot_positions.append(position)
            self.plot_metrics.append(metric)
            self._update_plot_display()
    
    def _on_complete(self, best_position: float, best_metric: float):
        """Handle autofocus completion."""
        self.status_label.setText(f"✅ Complete! Best focus at {best_position:.3f} µm")
        self.progress_bar.setValue(100)
        
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        
        # Mark best position on plot
        if self.has_plot and self.plot_widget and len(self.plot_positions) > 0:
            self.plot_widget.plot([best_position], [best_metric], 
                                 pen=None, symbol='o', symbolSize=15,
                                 symbolBrush='r', name='Best Focus')
    
    def _on_failed(self, error: str):
        """Handle autofocus failure."""
        self.status_label.setText(f"❌ Failed: {error}")
        
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
    
    def _on_cancelled(self):
        """Handle autofocus cancellation."""
        self.status_label.setText("⚠️ Cancelled by user")
        
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
    
    def _update_plot(self, positions, metrics):
        """Update plot with new data from worker."""
        self.plot_positions = positions
        self.plot_metrics = metrics
        self._update_plot_display()
    
    def _update_plot_display(self):
        """Update plot display."""
        if not self.has_plot or not self.plot_widget:
            return
        
        if len(self.plot_positions) > 0:
            self.plot_widget.plot(
                self.plot_positions,
                self.plot_metrics,
                pen='b',
                clear=True
            )


# Simplified version without pyqtgraph dependency
class AutofocusDialogSimple(QDialog):
    """
    Simplified autofocus dialog without live plotting.
    
    Use this if pyqtgraph is not available.
    """
    
    def __init__(self, autofocus_controller, state, parent=None):
        super().__init__(parent)
        self.autofocus = autofocus_controller
        self.state = state
        
        self.setWindowTitle("Autofocus")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Configuration
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        # Axis
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Axis:"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(['X (Focus)', 'Y', 'Z'])
        axis_layout.addWidget(self.axis_combo)
        axis_layout.addStretch()
        config_layout.addLayout(axis_layout)
        
        # Range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range:"))
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1.0, 100.0)
        self.range_spin.setValue(10.0)
        self.range_spin.setSuffix(" µm")
        range_layout.addWidget(self.range_spin)
        range_layout.addStretch()
        config_layout.addLayout(range_layout)
        
        # Step
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step:"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 5.0)
        self.step_spin.setValue(0.5)
        self.step_spin.setSuffix(" µm")
        self.step_spin.setDecimals(2)
        step_layout.addWidget(self.step_spin)
        step_layout.addStretch()
        config_layout.addLayout(step_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("QLabel { font-family: monospace; }")
        progress_layout.addWidget(self.info_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._start)
        btn_layout.addWidget(self.btn_start)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.autofocus.cancel)
        self.btn_cancel.setEnabled(False)
        btn_layout.addWidget(self.btn_cancel)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)
    
    def _connect_signals(self):
        """Connect signals."""
        self.autofocus.signals.autofocus_started.connect(
            lambda axis: self._update_state(True)
        )
        self.autofocus.signals.autofocus_progress.connect(self._on_progress)
        self.autofocus.signals.autofocus_complete.connect(self._on_complete)
        self.autofocus.signals.autofocus_failed.connect(
            lambda msg: self._update_state(False, f"Failed: {msg}")
        )
    
    def _start(self):
        """Start autofocus."""
        axis = self.axis_combo.currentText()[0].lower()
        self.autofocus.run_autofocus(
            axis=axis,
            scan_range_um=self.range_spin.value(),
            step_um=self.step_spin.value()
        )
    
    def _update_state(self, running: bool, status: str = ""):
        """Update UI state."""
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        if status:
            self.status_label.setText(status)
    
    def _on_progress(self, pos: float, metric: float, progress: float):
        """Handle progress."""
        self.progress_bar.setValue(int(progress))
        self.info_label.setText(f"Position: {pos:.3f} µm | Metric: {metric:.1f}")
    
    def _on_complete(self, pos: float, metric: float):
        """Handle completion."""
        self._update_state(False, f"✅ Best: {pos:.3f} µm (metric: {metric:.1f})")