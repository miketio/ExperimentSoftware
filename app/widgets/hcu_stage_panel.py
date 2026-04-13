# app/widgets/hcu_stage_panel.py
"""
HCU Stage Panel  ("Stage 2" tab)

Controls for the 3-axis HCU stage that positions the k-space slit.
Features:
  - Live position readout  (X / Y / Z, µm)
  - Jog controls with safety threshold
  - Open-position preset  (stage away → full image)
  - Slit-position preset  (stage in  → filtered image)
  - Named custom positions
  - All presets saved/loaded from config/hcu_positions.json automatically
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QGridLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QComboBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

# Step sizes (µm) above which the jog is blocked
LARGE_STEP_THRESHOLD_UM = 10.0


class HCUStagePanelWidget(QWidget):
    """
    Full control panel for the HCU 3-axis slit stage.

    Parameters
    ----------
    state          : SystemState
    signals        : SystemSignals
    hcu_controller : HCUController  (may wrap a None stage — buttons disabled gracefully)
    """

    def __init__(self, state, signals, hcu_controller, parent=None):
        super().__init__(parent)
        self.state      = state
        self.signals    = signals
        self.hcu        = hcu_controller

        self._init_ui()
        self._connect_signals()

        # Live position timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_position)
        self._timer.start(300)          # 300 ms polling

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        connected = self.hcu.is_connected

        if not connected:
            warn = QLabel(
                "⚠️  HCU stage not connected.\n"
                "Connect the SCU/HCU hardware and restart the application."
            )
            warn.setStyleSheet(
                "QLabel { background: #FFF3CD; color: #856404; "
                "padding: 10px; font-weight: bold; }"
            )
            warn.setWordWrap(True)
            layout.addWidget(warn)

        # ── Live position ──────────────────────────────────────────────
        pos_group = QGroupBox("Current Position (µm)")
        pos_grid  = QGridLayout()
        self._pos_labels: dict = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):
            pos_grid.addWidget(QLabel(f"{axis}:"), i, 0)
            lbl = QLabel("0.000")
            lbl.setStyleSheet(
                "QLabel { font-family: monospace; font-size: 14pt; "
                "font-weight: bold; color: #00FF00; "
                "background: #1E1E1E; padding: 4px 10px; }"
            )
            self._pos_labels[axis.lower()] = lbl
            pos_grid.addWidget(lbl, i, 1)

        pos_group.setLayout(pos_grid)
        layout.addWidget(pos_group)

        # ── Jog ───────────────────────────────────────────────────────
        jog_group  = QGroupBox("Jog Controls")
        jog_layout = QVBoxLayout()

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (µm):"))
        self._step_combo = QComboBox()
        self._step_combo.addItems(["0.1", "0.5", "1", "2", "5", "10"])
        self._step_combo.setCurrentText("1")
        step_row.addWidget(self._step_combo)
        step_row.addStretch()
        jog_layout.addLayout(step_row)

        # Arrow grid  (X left/right  |  Z up/down  |  Y focus)
        g = QGridLayout()
        g.setSpacing(6)

        def _btn(label: str, w: int = 80, h: int = 40) -> QPushButton:
            b = QPushButton(label)
            b.setFixedSize(w, h)
            b.setEnabled(connected)
            return b

        bzu = _btn("↑ Z+");  bzu.clicked.connect(lambda: self._jog('z',  1))
        bzd = _btn("↓ Z−");  bzd.clicked.connect(lambda: self._jog('z', -1))
        bxl = _btn("← X−");  bxl.clicked.connect(lambda: self._jog('x', -1))
        bxr = _btn("X+ →");  bxr.clicked.connect(lambda: self._jog('x',  1))
        byu = _btn("↑ Y+\n(focus)", w=100, h=50)
        byu.setStyleSheet("QPushButton { background: #E3F2FD; font-size: 9pt; }")
        byu.clicked.connect(lambda: self._jog('y',  1))
        byd = _btn("↓ Y−\n(focus)", w=100, h=50)
        byd.setStyleSheet("QPushButton { background: #FFF3E0; font-size: 9pt; }")
        byd.clicked.connect(lambda: self._jog('y', -1))

        center = QPushButton("⊙")
        center.setFixedSize(80, 40)
        center.setEnabled(False)

        g.addWidget(bzu,    0, 1)
        g.addWidget(bxl,    1, 0)
        g.addWidget(center, 1, 1)
        g.addWidget(bxr,    1, 2)
        g.addWidget(bzd,    2, 1)

        sep = QLabel("│")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("QLabel { color: #AAA; font-size: 20pt; }")
        g.addWidget(sep, 0, 3, 3, 1)

        ylabel = QLabel("Focus (Y)")
        ylabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ylabel.setStyleSheet("QLabel { font-size: 9pt; color: #888; }")
        g.addWidget(ylabel, 0, 4, alignment=Qt.AlignmentFlag.AlignBottom)
        g.addWidget(byu,    1, 4)
        g.addWidget(byd,    2, 4)

        jog_layout.addLayout(g)
        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)

        # ── Go To ─────────────────────────────────────────────────────
        goto_group = QGroupBox("Go To Position")
        goto_grid  = QGridLayout()
        self._goto_spins: dict = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):
            goto_grid.addWidget(QLabel(f"{axis}:"), i, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-15_000, 15_000)
            sp.setDecimals(3)
            sp.setSuffix(" µm")
            sp.setEnabled(connected)
            self._goto_spins[axis.lower()] = sp
            goto_grid.addWidget(sp, i, 1)

        btn_goto = QPushButton("▶ Go To")
        btn_goto.setEnabled(connected)
        btn_goto.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        btn_goto.clicked.connect(self._go_to)
        goto_grid.addWidget(btn_goto, 3, 0, 1, 2)
        goto_group.setLayout(goto_grid)
        layout.addWidget(goto_group)

        # ── Preset positions ──────────────────────────────────────────
        preset_group  = QGroupBox("Preset Positions")
        preset_layout = QVBoxLayout()

        # -- Open preset --
        open_box    = QGroupBox("📷  Open Position  (full image visible)")
        open_layout = QVBoxLayout()

        self._open_label = QLabel("Not set")
        self._open_label.setStyleSheet(
            "QLabel { font-family: monospace; background: #1E1E1E; "
            "color: #00FF00; padding: 6px; }"
        )
        open_layout.addWidget(self._open_label)

        open_btns = QHBoxLayout()
        btn_save_open = QPushButton("💾 Save Current → Open")
        btn_save_open.setEnabled(connected)
        btn_save_open.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_save_open.clicked.connect(lambda: self._save_preset('open'))
        open_btns.addWidget(btn_save_open)

        self._btn_goto_open = QPushButton("🎯 Move to Open")
        self._btn_goto_open.setEnabled(connected)
        self._btn_goto_open.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; font-weight: bold; padding: 6px; }"
        )
        self._btn_goto_open.clicked.connect(self.hcu.move_away)
        open_btns.addWidget(self._btn_goto_open)
        open_layout.addLayout(open_btns)
        open_box.setLayout(open_layout)
        preset_layout.addWidget(open_box)

        # -- Slit preset --
        slit_box    = QGroupBox("🔬  Slit Position  (filter / k-space slit active)")
        slit_layout = QVBoxLayout()

        self._slit_label = QLabel("Not set")
        self._slit_label.setStyleSheet(
            "QLabel { font-family: monospace; background: #1E1E1E; "
            "color: cyan; padding: 6px; }"
        )
        slit_layout.addWidget(self._slit_label)

        slit_btns = QHBoxLayout()
        btn_save_slit = QPushButton("💾 Save Current → Slit")
        btn_save_slit.setEnabled(connected)
        btn_save_slit.setStyleSheet(
            "QPushButton { background: #FF9800; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_save_slit.clicked.connect(lambda: self._save_preset('slit'))
        slit_btns.addWidget(btn_save_slit)

        self._btn_goto_slit = QPushButton("🎯 Move to Slit")
        self._btn_goto_slit.setEnabled(connected)
        self._btn_goto_slit.setStyleSheet(
            "QPushButton { background: #9C27B0; color: white; font-weight: bold; padding: 6px; }"
        )
        self._btn_goto_slit.clicked.connect(self.hcu.move_in)
        slit_btns.addWidget(self._btn_goto_slit)
        slit_layout.addLayout(slit_btns)
        slit_box.setLayout(slit_layout)
        preset_layout.addWidget(slit_box)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # ── Custom positions ──────────────────────────────────────────
        custom_group  = QGroupBox("Custom Saved Positions")
        custom_layout = QVBoxLayout()

        # Save row
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Name:"))
        self._custom_name = QLineEdit()
        self._custom_name.setPlaceholderText("e.g. 'reference', 'sample edge'")
        save_row.addWidget(self._custom_name)
        btn_save_custom = QPushButton("💾 Save")
        btn_save_custom.setEnabled(connected)
        btn_save_custom.setStyleSheet(
            "QPushButton { background: #607D8B; color: white; padding: 6px; }"
        )
        btn_save_custom.clicked.connect(self._save_custom)
        save_row.addWidget(btn_save_custom)
        custom_layout.addLayout(save_row)

        # Table
        self._custom_table = QTableWidget()
        self._custom_table.setColumnCount(3)
        self._custom_table.setHorizontalHeaderLabels(['Name', 'Position (X, Y, Z) µm', 'Go'])
        self._custom_table.setMaximumHeight(180)
        self._custom_table.setAlternatingRowColors(True)
        hdr = self._custom_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._custom_table.setColumnWidth(2, 60)
        custom_layout.addWidget(self._custom_table)

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        layout.addStretch()

        # Initial preset label update
        self._refresh_preset_labels()
        self._refresh_custom_table()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        pass   # no external signals needed beyond the timer

    # ------------------------------------------------------------------
    # Jog
    # ------------------------------------------------------------------

    def _jog(self, axis: str, sign: int):
        step_um = float(self._step_combo.currentText()) * sign
        if abs(step_um) >= LARGE_STEP_THRESHOLD_UM:
            QMessageBox.information(
                self,
                "Step Too Large — Move Blocked",
                f"Step {abs(step_um):.1f} µm is above the safety limit "
                f"({LARGE_STEP_THRESHOLD_UM:.0f} µm).\n"
                "Please reduce the step size first."
            )
            return

        pos_nm = {ax: self.hcu.hcu_stage.get_pos(ax) for ax in ('x', 'y', 'z')}
        pos_nm[axis] = int(pos_nm[axis] + step_um * 1000)
        self.hcu.move_to_nm(**pos_nm)

    # ------------------------------------------------------------------
    # Go To
    # ------------------------------------------------------------------

    def _go_to(self):
        x_nm = int(self._goto_spins['x'].value() * 1000)
        y_nm = int(self._goto_spins['y'].value() * 1000)
        z_nm = int(self._goto_spins['z'].value() * 1000)
        self.hcu.move_to_nm(x_nm, y_nm, z_nm)

    # ------------------------------------------------------------------
    # Preset management
    # ------------------------------------------------------------------

    def _save_preset(self, name: str):
        if self.hcu.save_current_as(name):
            self._refresh_preset_labels()

    def _save_custom(self):
        name = self._custom_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Enter a name for this position.")
            return
        if self.hcu.save_current_as(name):
            self._custom_name.clear()
            self._refresh_custom_table()

    def _refresh_preset_labels(self):
        o = self.hcu.open_position_um
        s = self.hcu.slit_position_um
        self._open_label.setText(
            f"X={o['x']:.1f}  Y={o['y']:.1f}  Z={o['z']:.1f} µm"
        )
        self._slit_label.setText(
            f"X={s['x']:.1f}  Y={s['y']:.1f}  Z={s['z']:.1f} µm"
        )

    def _refresh_custom_table(self):
        self._custom_table.setRowCount(0)
        for name in self.hcu.list_custom_presets():
            pos = self.hcu.get_preset_um(name)
            if pos is None:
                continue
            row = self._custom_table.rowCount()
            self._custom_table.insertRow(row)

            self._custom_table.setItem(row, 0, QTableWidgetItem(name))

            pos_text = f"({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})"
            pos_item = QTableWidgetItem(pos_text)
            pos_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._custom_table.setItem(row, 1, pos_item)

            btn_go = QPushButton("🎯")
            btn_go.setEnabled(self.hcu.is_connected)
            btn_go.clicked.connect(lambda _, n=name: self.hcu._move_to_preset(n))
            self._custom_table.setCellWidget(row, 2, btn_go)

    # ------------------------------------------------------------------
    # Position refresh
    # ------------------------------------------------------------------

    def _refresh_position(self):
        pos = self.hcu.get_current_um()
        for axis, lbl in self._pos_labels.items():
            lbl.setText(f"{pos[axis]:.3f}")