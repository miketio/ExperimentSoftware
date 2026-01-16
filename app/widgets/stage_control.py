"""Stage Control Widget - Manual stage positioning.

COORDINATE SYSTEM:
- X axis: LEFT/RIGHT (horizontal movement)
- Y axis: FOCUS (in/out of focus)
- Z axis: UP/DOWN (vertical movement)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class StageMoveWorker(QThread):
    """Worker thread for stage movements."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, stage, axis, position, is_relative=False):
        super().__init__()
        self.stage = stage
        self.axis = axis
        self.position = position
        self.is_relative = is_relative
    
    def run(self):
        try:
            if self.is_relative:
                self.stage.move_rel(self.axis, self.position)
            else:
                self.stage.move_abs(self.axis, self.position)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class GoToWorker(QThread):
    """Worker thread for Go To positioning (sequential Y→X→Z)."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, stage, x, y, z):
        super().__init__()
        self.stage = stage
        self.x = x
        self.y = y
        self.z = z
    
    def run(self):
        try:
            # Move sequentially to avoid collision (Y focus first, then X/Z)
            self.stage.move_abs('y', self.y)
            self.stage.move_abs('x', self.x)
            self.stage.move_abs('z', self.z)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class StageControlWidget(QWidget):
    """Stage jog and positioning controls.
    
    NEW COORDINATE SYSTEM:
    - X = LEFT/RIGHT
    - Y = FOCUS
    - Z = UP/DOWN
    """
    
    def __init__(self, state, signals, stage, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.stage = stage
        self.move_worker = None
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Position display
        pos_group = QGroupBox("Current Position (µm)")
        pos_layout = QGridLayout()
        
        self.pos_labels = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            pos_layout.addWidget(QLabel(f"{axis}:"), i, 0)
            label = QLabel("0.000")
            label.setStyleSheet("QLabel { font-family: monospace; font-size: 14pt; }")
            self.pos_labels[axis.lower()] = label
            pos_layout.addWidget(label, i, 1)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Jog controls
        jog_group = QGroupBox("Jog Controls")
        jog_layout = QVBoxLayout()
        
        # Step size - UPDATED with new values
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size (µm):"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(['0.1', '0.5', '1', '5', '10', '20', '50', '100', '500'])
        self.step_combo.setCurrentText('10')
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        step_layout.addWidget(self.step_combo)
        step_layout.addStretch()
        jog_layout.addLayout(step_layout)
        
        # Arrow buttons - X/Z axes (horizontal/vertical)
        arrows = QWidget()
        arrows_layout = QGridLayout()
        arrows_layout.setSpacing(5)
        
        # X axis (horizontal: left/right) - CHANGED from Y
        self.btn_x_left = QPushButton("← X-")
        self.btn_x_left.clicked.connect(lambda: self._jog('x', -1))
        arrows_layout.addWidget(self.btn_x_left, 1, 0)

        self.btn_x_right = QPushButton("X+ →")
        self.btn_x_right.clicked.connect(lambda: self._jog('x', 1))
        arrows_layout.addWidget(self.btn_x_right, 1, 2)

        # Z axis (vertical: up/down) - UNCHANGED
        self.btn_z_up = QPushButton("↑ Z+")
        self.btn_z_up.clicked.connect(lambda: self._jog('z', 1))
        arrows_layout.addWidget(self.btn_z_up, 0, 1)

        self.btn_z_down = QPushButton("↓ Z-")
        self.btn_z_down.clicked.connect(lambda: self._jog('z', -1))
        arrows_layout.addWidget(self.btn_z_down, 2, 1)
        
        # Center (home)
        self.btn_center = QPushButton("⊙")
        self.btn_center.setToolTip("Current position")
        arrows_layout.addWidget(self.btn_center, 1, 1)
        
        arrows.setLayout(arrows_layout)
        jog_layout.addWidget(arrows)
        
        # Y axis (focus) - CHANGED from X
        y_layout = QHBoxLayout()
        self.btn_y_down = QPushButton("Y- (Out of focus)")
        self.btn_y_down.clicked.connect(lambda: self._jog('y', -1))
        y_layout.addWidget(self.btn_y_down)
        
        self.btn_y_up = QPushButton("Y+ (Into focus)")
        self.btn_y_up.clicked.connect(lambda: self._jog('y', 1))
        y_layout.addWidget(self.btn_y_up)
        jog_layout.addLayout(y_layout)
        
        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)
        
        # Go To
        goto_group = QGroupBox("Go To Position")
        goto_layout = QGridLayout()
        
        self.goto_inputs = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            goto_layout.addWidget(QLabel(f"{axis}:"), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-100000, 100000)
            spin.setDecimals(3)
            spin.setSuffix(" µm")
            self.goto_inputs[axis.lower()] = spin
            goto_layout.addWidget(spin, i, 1)
        
        self.btn_goto = QPushButton("Go To")
        self.btn_goto.clicked.connect(self._go_to_position)
        goto_layout.addWidget(self.btn_goto, 3, 0, 1, 2)
        
        goto_group.setLayout(goto_layout)
        layout.addWidget(goto_group)
        
        layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.stage_position_changed.connect(self._update_position_display)
    
    def _on_step_changed(self, step_text: str):
        """Handle step size change."""
        self.state.set_jog_step(float(step_text))
    
    def _jog(self, axis: str, direction: int):
        """Jog stage in direction."""
        if self.stage is None:
            return
        
        step = self.state.get_jog_step() * direction
        
        # Create worker thread
        self.move_worker = StageMoveWorker(self.stage, axis, step, is_relative=True)
        self.move_worker.finished.connect(lambda: self.signals.stage_move_complete.emit())
        self.move_worker.error.connect(lambda e: self.signals.stage_error.emit(e))
        self.move_worker.start()
        
    def _go_to_position(self):
        """Move to specified position (sequential Y→X→Z)."""
        if self.stage is None:
            return
        
        # Get target positions
        target_x = self.goto_inputs['x'].value()
        target_y = self.goto_inputs['y'].value()
        target_z = self.goto_inputs['z'].value()
        
        self.signals.status_message.emit(
            f"Moving to X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}µm..."
        )
        
        # Create worker for sequential movement
        self.move_worker = GoToWorker(
            self.stage, target_x, target_y, target_z
        )
        self.move_worker.finished.connect(
            lambda: self.signals.stage_move_complete.emit()
        )
        self.move_worker.error.connect(
            lambda e: self.signals.stage_error.emit(e)
        )
        self.move_worker.finished.connect(
            lambda: self.signals.status_message.emit("Go To complete")
        )
        
        # Disable button during movement
        self.btn_goto.setEnabled(False)
        self.move_worker.finished.connect(
            lambda: self.btn_goto.setEnabled(True)
        )
        
        self.move_worker.start()
    
    def _update_position_display(self, axis: str, position: float):
        """Update position display."""
        if axis in self.pos_labels:
            self.pos_labels[axis].setText(f"{position:.3f}")