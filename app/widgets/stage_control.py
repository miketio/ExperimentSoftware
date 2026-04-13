"""Stage Control Widget - Manual stage positioning.

COORDINATE SYSTEM:
- X axis: LEFT/RIGHT (horizontal movement)
- Y axis: FOCUS (in/out of focus)
- Z axis: UP/DOWN (vertical movement)

SAFETY:
- Step sizes >= 10 µm are blocked. User must reduce step size first.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox,
    QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

LARGE_STEP_THRESHOLD_UM = 10.0   # µm — warn above this value


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

    COORDINATE SYSTEM:
    - X = LEFT/RIGHT
    - Y = FOCUS (up = into focus)
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

    # ------------------------------------------------------------------
    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Current position display ───────────────────────────────────
        pos_group = QGroupBox("Current Position (µm)")
        pos_layout = QGridLayout()

        self.pos_labels = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            pos_layout.addWidget(QLabel(f"{axis}:"), i, 0)
            label = QLabel("0.000")
            label.setStyleSheet(
                "QLabel { font-family: monospace; font-size: 14pt; }"
            )
            self.pos_labels[axis.lower()] = label
            pos_layout.addWidget(label, i, 1)

        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        # ── Jog controls ───────────────────────────────────────────────
        jog_group = QGroupBox("Jog Controls")
        jog_layout = QVBoxLayout()

        # Step size selector
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size (µm):"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(['0.1', '0.5', '1', '5', '10', '20', '50', '100', '500'])
        self.step_combo.setCurrentText('1')
        self.state.set_jog_step(1.0)          # sync state immediately
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        step_layout.addWidget(self.step_combo)
        step_layout.addStretch()
        jog_layout.addLayout(step_layout)

        # Arrow grid
        #
        #        [↑ Z+]     [↑ Y+]
        #  [← X-]  [⊙]  [X+ →]
        #        [↓ Z-]     [↓ Y-]
        #
        arrows = QWidget()
        g = QGridLayout()
        g.setSpacing(6)

        # Z axis (column 1)
        btn_z_up = QPushButton("↑ Z+")
        btn_z_up.setFixedSize(80, 40)
        btn_z_up.clicked.connect(lambda: self._jog('z', 1))
        g.addWidget(btn_z_up, 0, 1)

        btn_x_left = QPushButton("← X-")
        btn_x_left.setFixedSize(80, 40)
        btn_x_left.clicked.connect(lambda: self._jog('x', -1))
        g.addWidget(btn_x_left, 1, 0)

        btn_center = QPushButton("⊙")
        btn_center.setFixedSize(80, 40)
        btn_center.setToolTip("Current position")
        g.addWidget(btn_center, 1, 1)

        btn_x_right = QPushButton("X+ →")
        btn_x_right.setFixedSize(80, 40)
        btn_x_right.clicked.connect(lambda: self._jog('x', 1))
        g.addWidget(btn_x_right, 1, 2)

        btn_z_down = QPushButton("↓ Z-")
        btn_z_down.setFixedSize(80, 40)
        btn_z_down.clicked.connect(lambda: self._jog('z', -1))
        g.addWidget(btn_z_down, 2, 1)

        # Y axis (focus) — column 3, up/down style
        y_label = QLabel("Focus (Y)")
        y_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        y_label.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        g.addWidget(y_label, 0, 3, alignment=Qt.AlignmentFlag.AlignBottom)

        btn_y_up = QPushButton("↑ Y+\n(into focus)")
        btn_y_up.setFixedSize(90, 50)
        btn_y_up.setStyleSheet(
            "QPushButton { background-color: #E3F2FD; font-size: 9pt; }"
            "QPushButton:pressed { background-color: #90CAF9; }"
        )
        btn_y_up.clicked.connect(lambda: self._jog('y', 1))
        g.addWidget(btn_y_up, 1, 3)

        btn_y_down = QPushButton("↓ Y-\n(out of focus)")
        btn_y_down.setFixedSize(90, 50)
        btn_y_down.setStyleSheet(
            "QPushButton { background-color: #FFF3E0; font-size: 9pt; }"
            "QPushButton:pressed { background-color: #FFCC80; }"
        )
        btn_y_down.clicked.connect(lambda: self._jog('y', -1))
        g.addWidget(btn_y_down, 2, 3)

        arrows.setLayout(g)
        jog_layout.addWidget(arrows)

        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)

        # ── Go To ──────────────────────────────────────────────────────
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

    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.signals.stage_position_changed.connect(self._update_position_display)

    def _on_step_changed(self, step_text: str):
        self.state.set_jog_step(float(step_text))

    # ------------------------------------------------------------------
    def _confirm_large_step(self, step_um: float) -> bool:
        """
        Block any step >= LARGE_STEP_THRESHOLD_UM and tell the user to reduce it.
        Always returns False when the threshold is exceeded — the move never happens.
        """
        if abs(step_um) < LARGE_STEP_THRESHOLD_UM:
            return True

        QMessageBox.information(
            self,
            "⚠️  Step Too Large — Move Blocked",
            f"<b>Step size {abs(step_um):.1f} µm is not allowed.</b><br><br>"
            f"Moving more than <b>{LARGE_STEP_THRESHOLD_UM:.0f} µm</b> in a single step "
            f"risks crashing the objective into the sample.<br><br>"
            f"Please set the step size to <b>{LARGE_STEP_THRESHOLD_UM:.0f} µm or less</b> "
            f"before moving.",
        )
        return False   # always block

    # ------------------------------------------------------------------
    def _jog(self, axis: str, direction: int):
        """Jog stage — with large-step safety check."""
        if self.stage is None:
            return

        step = self.state.get_jog_step() * direction

        if not self._confirm_large_step(step):
            self.signals.status_message.emit(
                f"⚠️  Move cancelled — reduce step size below "
                f"{LARGE_STEP_THRESHOLD_UM:.0f} µm first."
            )
            return

        self.move_worker = StageMoveWorker(self.stage, axis, step, is_relative=True)
        self.move_worker.finished.connect(lambda: self.signals.stage_move_complete.emit())
        self.move_worker.error.connect(lambda e: self.signals.stage_error.emit(e))
        self.move_worker.start()

    def _go_to_position(self):
        """Move to specified position (sequential Y→X→Z)."""
        if self.stage is None:
            return

        target_x = self.goto_inputs['x'].value()
        target_y = self.goto_inputs['y'].value()
        target_z = self.goto_inputs['z'].value()

        self.signals.status_message.emit(
            f"Moving to X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f} µm…"
        )

        self.move_worker = GoToWorker(self.stage, target_x, target_y, target_z)
        self.move_worker.finished.connect(lambda: self.signals.stage_move_complete.emit())
        self.move_worker.error.connect(lambda e: self.signals.stage_error.emit(e))
        self.move_worker.finished.connect(
            lambda: self.signals.status_message.emit("Go To complete")
        )

        self.btn_goto.setEnabled(False)
        self.move_worker.finished.connect(lambda: self.btn_goto.setEnabled(True))

        self.move_worker.start()

    def _update_position_display(self, axis: str, position: float):
        if axis in self.pos_labels:
            self.pos_labels[axis].setText(f"{position:.3f}")