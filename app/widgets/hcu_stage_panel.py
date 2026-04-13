# app/widgets/hcu_stage_panel.py
"""
HCU Stage Panel  ("Stage 2" tab)

FIXES:
- Units changed to mm throughout (HCU DLL unit chain gives mm, not µm)
- _jog: uses correct unit conversion (step_mm * 1000 → internal units)
- _go_to: spinboxes in mm, converts to internal units before calling move_to_nm
- move_to_nm called with correct keyword args (x_nm=, y_nm=, z_nm=)
- Safety threshold raised to 20 mm (HCU is slit stage, not sample stage)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QGridLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QComboBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

# HCU is a slit stage — much larger safe travel range than the XYZ sample stage
LARGE_STEP_THRESHOLD_MM = 20.0   # mm


class HCUStagePanelWidget(QWidget):
    """
    Full control panel for the HCU 3-axis slit stage.

    All displayed values and spinboxes are in MILLIMETRES.
    Internally, get_pos() returns units where value/1000 == mm,
    and move_to_nm() expects those same internal units.
    """

    def __init__(self, state, signals, hcu_controller, parent=None):
        super().__init__(parent)
        self.state  = state
        self.signals = signals
        self.hcu    = hcu_controller

        self._init_ui()
        self._connect_signals()

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_position)
        self._timer.start(300)

    # ------------------------------------------------------------------
    # Helpers: unit conversion
    # ------------------------------------------------------------------

    def _get_pos_mm(self) -> dict:
        """Return {axis: float_mm} for all axes using get_current_um() which actually gives mm."""
        return self.hcu.get_current_um()   # named "um" but returns mm — see controller note

    def _mm_to_internal(self, mm: float) -> int:
        """Convert mm to internal units expected by move_to_nm / get_pos."""
        return int(mm * 1000)

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
        pos_group = QGroupBox("Current Position (mm)")
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

            unit_lbl = QLabel("mm")
            unit_lbl.setStyleSheet("QLabel { color: #AAA; }")
            pos_grid.addWidget(unit_lbl, i, 2)

        pos_group.setLayout(pos_grid)
        layout.addWidget(pos_group)

        # ── Jog ───────────────────────────────────────────────────────
        jog_group  = QGroupBox("Jog Controls")
        jog_layout = QVBoxLayout()

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (mm):"))
        self._step_combo = QComboBox()
        self._step_combo.addItems(["0.01", "0.05", "0.1", "0.5", "1", "2", "5"])
        self._step_combo.setCurrentText("0.1")
        step_row.addWidget(self._step_combo)
        step_row.addStretch()
        jog_layout.addLayout(step_row)

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
        goto_group = QGroupBox("Go To Position (mm)")
        goto_grid  = QGridLayout()
        self._goto_spins: dict = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):
            goto_grid.addWidget(QLabel(f"{axis}:"), i, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-25.0, 25.0)   # ±25 mm
            sp.setDecimals(3)
            sp.setSuffix(" mm")
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

        self._custom_table = QTableWidget()
        self._custom_table.setColumnCount(4)
        self._custom_table.setHorizontalHeaderLabels(['Name', 'Position (X, Y, Z) mm', 'Go', 'Delete'])
        self._custom_table.setMaximumHeight(180)
        self._custom_table.setAlternatingRowColors(True)
        hdr = self._custom_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._custom_table.setColumnWidth(2, 60)
        self._custom_table.setColumnWidth(3, 75)
        custom_layout.addWidget(self._custom_table)

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        layout.addStretch()

        self._refresh_preset_labels()
        self._refresh_custom_table()

    # ------------------------------------------------------------------
    def _connect_signals(self):
        pass

    # ------------------------------------------------------------------
    # Jog
    # ------------------------------------------------------------------

    def _jog(self, axis: str, sign: int):
        """
        Jog by step_mm millimetres on the given axis.

        Unit chain:
          display (mm)  →  internal units (mm * 1000)  →  move_to_nm  →  DLL
        """
        if not self.hcu.is_connected:
            return

        step_mm = float(self._step_combo.currentText()) * sign

        if abs(step_mm) >= LARGE_STEP_THRESHOLD_MM:
            QMessageBox.information(
                self,
                "Step Too Large — Move Blocked",
                f"Step {abs(step_mm):.2f} mm exceeds the safety limit "
                f"({LARGE_STEP_THRESHOLD_MM:.0f} mm).\n"
                "Please reduce the step size first."
            )
            return

        # Read current position in internal units (same units get_pos returns)
        try:
            pos = {ax: self.hcu.hcu_stage.get_pos(ax) for ax in ('x', 'y', 'z')}
        except Exception as e:
            self.signals.error_occurred.emit("HCU Jog Error", str(e))
            return

        # Add step in internal units: 1 mm = 1000 internal units
        pos[axis] = int(pos[axis] + step_mm * 1000)

        self.hcu.move_to_nm(x_nm=pos['x'], y_nm=pos['y'], z_nm=pos['z'])

    # ------------------------------------------------------------------
    # Go To
    # ------------------------------------------------------------------

    def _go_to(self):
        """Move to position entered in the mm spinboxes."""
        x_mm = self._goto_spins['x'].value()
        y_mm = self._goto_spins['y'].value()
        z_mm = self._goto_spins['z'].value()

        # Convert mm → internal units
        self.hcu.move_to_nm(
            x_nm=self._mm_to_internal(x_mm),
            y_nm=self._mm_to_internal(y_mm),
            z_nm=self._mm_to_internal(z_mm),
        )

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
        """Show preset positions in mm."""
        o = self.hcu.open_position_um   # actually mm — see hcu_controller note
        s = self.hcu.slit_position_um   # actually mm
        self._open_label.setText(
            f"X={o['x']:.2f}  Y={o['y']:.2f}  Z={o['z']:.2f} mm"
        )
        self._slit_label.setText(
            f"X={s['x']:.2f}  Y={s['y']:.2f}  Z={s['z']:.2f} mm"
        )

    def _refresh_custom_table(self):
        self._custom_table.setRowCount(0)
        for name in self.hcu.list_custom_presets():
            pos = self.hcu.get_preset_um(name)   # actually mm
            if pos is None:
                continue
            row = self._custom_table.rowCount()
            self._custom_table.insertRow(row)

            self._custom_table.setItem(row, 0, QTableWidgetItem(name))

            pos_text = f"({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) mm"
            pos_item = QTableWidgetItem(pos_text)
            pos_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._custom_table.setItem(row, 1, pos_item)

            btn_go = QPushButton("🎯")
            btn_go.setEnabled(self.hcu.is_connected)
            btn_go.clicked.connect(lambda _, n=name: self.hcu._move_to_preset(n))
            self._custom_table.setCellWidget(row, 2, btn_go)

            btn_del = QPushButton("🗑")
            btn_del.setEnabled(self.hcu.is_connected)
            btn_del.clicked.connect(lambda _, n=name: self._delete_custom(n))
            self._custom_table.setCellWidget(row, 3, btn_del)

    def _delete_custom(self, name: str):
        reply = QMessageBox.question(
            self,
            "Delete Custom Position",
            f"Delete custom position '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.hcu.delete_custom_preset(name)
        self.signals.status_message.emit(f"Deleted custom position '{name}'")
        self._refresh_custom_table()

    # ------------------------------------------------------------------
    # Position refresh
    # ------------------------------------------------------------------

    def _refresh_position(self):
        """Update live position labels (values are in mm)."""
        pos_mm = self._get_pos_mm()
        for axis, lbl in self._pos_labels.items():
            lbl.setText(f"{pos_mm[axis]:.3f}")