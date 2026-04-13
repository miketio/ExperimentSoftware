"""
Stage-Only Control Window

Lightweight window for controlling XYZ + filter stages without camera.
Provides:
  - XYZ stage jog / Go-To  (with large-step safety warning)
  - Saved position bookmarks
  - Filter stage manual position control
  - Live position display

No camera, no alignment, no imaging — just stage control.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QStatusBar, QLabel, QMessageBox, QGroupBox,
    QPushButton, QDoubleSpinBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QGridLayout, QComboBox,
    QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor
from pathlib import Path
import json
from datetime import datetime


LARGE_STEP_THRESHOLD_UM = 10.0   # µm — warn above this value


# ---------------------------------------------------------------------------
# Minimal inline stage control (no SystemState dependency on camera fields)
# ---------------------------------------------------------------------------

class StageJogWidget(QWidget):
    """XYZ jog + Go-To panel, camera-free."""

    def __init__(self, stage, state, signals, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.state = state
        self.signals = signals
        self._init_ui()
        self.signals.stage_position_changed.connect(self._on_pos_changed)

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        # ── Live position ──────────────────────────────────────────────
        pos_group = QGroupBox("Current Position (µm)")
        pos_grid = QGridLayout()
        self._pos_labels = {}
        for i, axis in enumerate(("X", "Y", "Z")):
            pos_grid.addWidget(QLabel(f"{axis}:"), i, 0)
            lbl = QLabel("0.000")
            lbl.setStyleSheet(
                "QLabel { font-family: monospace; font-size: 14pt; "
                "font-weight: bold; color: #00FF00; "
                "background: #1E1E1E; padding: 4px 8px; }"
            )
            self._pos_labels[axis.lower()] = lbl
            pos_grid.addWidget(lbl, i, 1)
        pos_group.setLayout(pos_grid)
        root.addWidget(pos_group)

        # ── Step size ─────────────────────────────────────────────────
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (µm):"))
        self._step_combo = QComboBox()
        self._step_combo.addItems(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50"])
        self._step_combo.setCurrentText("1")
        self.state.set_jog_step(1.0)          # sync state immediately
        self._step_combo.currentTextChanged.connect(
            lambda v: self.state.set_jog_step(float(v))
        )
        step_row.addWidget(self._step_combo)
        step_row.addStretch()
        root.addLayout(step_row)

        # ── Arrow grid ────────────────────────────────────────────────
        #
        #        [↑ Z+]      [↑ Y+]
        #  [← X-]  [⊙]  [X+ →]
        #        [↓ Z-]      [↓ Y-]
        #
        arrows_group = QGroupBox(
            "Jog  —  X: left/right  |  Z: up/down  |  Y: focus in/out"
        )
        arrows_layout = QVBoxLayout()

        xz = QWidget()
        xz_grid = QGridLayout()
        xz_grid.setSpacing(8)

        def _btn(label, w=80, h=45):
            b = QPushButton(label)
            b.setFixedSize(w, h)
            return b

        # Z column (col 1)
        b_z_up = _btn("↑ Z+")
        b_z_up.clicked.connect(lambda: self._jog("z", 1))
        xz_grid.addWidget(b_z_up, 0, 1)

        b_x_left = _btn("← X-")
        b_x_left.clicked.connect(lambda: self._jog("x", -1))
        xz_grid.addWidget(b_x_left, 1, 0)

        center = _btn("⊙")
        center.setEnabled(False)
        center.setToolTip("Current position")
        xz_grid.addWidget(center, 1, 1)

        b_x_right = _btn("X+ →")
        b_x_right.clicked.connect(lambda: self._jog("x", 1))
        xz_grid.addWidget(b_x_right, 1, 2)

        b_z_down = _btn("↓ Z-")
        b_z_down.clicked.connect(lambda: self._jog("z", -1))
        xz_grid.addWidget(b_z_down, 2, 1)

        # Separator label
        sep = QLabel("│")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("QLabel { color: #AAA; font-size: 20pt; }")
        xz_grid.addWidget(sep, 0, 3, 3, 1)

        # Y column (col 4) — focus up/down
        y_title = QLabel("Focus (Y)")
        y_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        y_title.setStyleSheet("QLabel { font-size: 9pt; color: #888; }")
        xz_grid.addWidget(y_title, 0, 4, alignment=Qt.AlignmentFlag.AlignBottom)

        b_y_up = _btn("↑ Y+\n(into focus)", w=100, h=50)
        b_y_up.setStyleSheet(
            "QPushButton { background-color: #E3F2FD; font-size: 9pt; }"
            "QPushButton:pressed { background-color: #90CAF9; }"
        )
        b_y_up.clicked.connect(lambda: self._jog("y", 1))
        xz_grid.addWidget(b_y_up, 1, 4)

        b_y_down = _btn("↓ Y-\n(out of focus)", w=100, h=50)
        b_y_down.setStyleSheet(
            "QPushButton { background-color: #FFF3E0; font-size: 9pt; }"
            "QPushButton:pressed { background-color: #FFCC80; }"
        )
        b_y_down.clicked.connect(lambda: self._jog("y", -1))
        xz_grid.addWidget(b_y_down, 2, 4)

        xz.setLayout(xz_grid)
        arrows_layout.addWidget(xz)

        arrows_group.setLayout(arrows_layout)
        root.addWidget(arrows_group)

        # ── Go To ─────────────────────────────────────────────────────
        goto_group = QGroupBox("Go To Position")
        goto_grid = QGridLayout()
        self._goto = {}
        for i, axis in enumerate(("X", "Y", "Z")):
            goto_grid.addWidget(QLabel(f"{axis}:"), i, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-100_000, 100_000)
            sp.setDecimals(3)
            sp.setSuffix(" µm")
            self._goto[axis.lower()] = sp
            goto_grid.addWidget(sp, i, 1)

        btn_go = QPushButton("▶  Go To")
        btn_go.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        btn_go.clicked.connect(self._go_to)
        goto_grid.addWidget(btn_go, 3, 0, 1, 2)
        goto_group.setLayout(goto_grid)
        root.addWidget(goto_group)

        root.addStretch()

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

    def _jog(self, axis: str, sign: int):
        """Move stage by one jog step — with large-step safety guard."""
        step = self.state.get_jog_step() * sign

        if not self._confirm_large_step(step):
            self.signals.status_message.emit(
                f"⚠️  Move cancelled — reduce step size below "
                f"{LARGE_STEP_THRESHOLD_UM:.0f} µm first."
            )
            return

        try:
            self.stage.move_rel(axis, step)
        except Exception as e:
            self.signals.error_occurred.emit("Jog Error", str(e))

    def _go_to(self):
        try:
            self.stage.move_abs("y", self._goto["y"].value())
            self.stage.move_abs("x", self._goto["x"].value())
            self.stage.move_abs("z", self._goto["z"].value())
            self.signals.status_message.emit("Go To complete")
        except Exception as e:
            self.signals.error_occurred.emit("Go To Error", str(e))

    def _on_pos_changed(self, axis: str, position: float):
        if axis in self._pos_labels:
            self._pos_labels[axis].setText(f"{position:.3f}")


# ---------------------------------------------------------------------------

class FilterManualWidget(QWidget):
    """Manual filter stage control — no sweep, no camera."""

    def __init__(self, filter_stage, signals, parent=None):
        super().__init__(parent)
        self.filter_stage = filter_stage
        self.signals = signals
        self._init_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_pos)
        self._timer.start(500)

    def _init_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        # Live position
        self._pos_lbl = QLabel("Position: -- µm")
        self._pos_lbl.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 14pt; "
            "background: #1E1E1E; color: cyan; padding: 10px; font-weight: bold; }"
        )
        root.addWidget(self._pos_lbl)

        # Go To
        goto_group = QGroupBox("Go To Position")
        goto_layout = QHBoxLayout()
        goto_layout.addWidget(QLabel("Target:"))
        self._target_spin = QDoubleSpinBox()
        self._target_spin.setRange(-15_000, 15_000)
        self._target_spin.setValue(0)
        self._target_spin.setSuffix(" µm")
        self._target_spin.setDecimals(3)
        goto_layout.addWidget(self._target_spin)
        btn_go = QPushButton("Move")
        btn_go.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; padding: 8px; }"
        )
        btn_go.clicked.connect(self._move)
        goto_layout.addWidget(btn_go)
        goto_group.setLayout(goto_layout)
        root.addWidget(goto_group)

        # Quick positions
        quick_group = QGroupBox("Quick Positions")
        quick_layout = QVBoxLayout()
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        for pos in [-15000, -10000, -5000, -1000]:
            b = QPushButton(f"{pos} µm")
            b.clicked.connect(lambda _, p=pos: self._move_to(p))
            row1.addWidget(b)
        for pos in [0, 1000, 5000, 10000, 15000]:
            b = QPushButton(f"{pos} µm")
            b.clicked.connect(lambda _, p=pos: self._move_to(p))
            row2.addWidget(b)
        quick_layout.addLayout(row1)
        quick_layout.addLayout(row2)
        quick_group.setLayout(quick_layout)
        root.addWidget(quick_group)

        root.addStretch()

    def _move(self):
        self._move_to(self._target_spin.value())

    def _move_to(self, pos_um: float):
        if self.filter_stage is None:
            self.signals.status_message.emit("No filter stage connected")
            return
        try:
            self.filter_stage.move_abs(int(pos_um * 1000))
            self._refresh_pos()
            self.signals.status_message.emit(f"Filter → {pos_um:.3f} µm")
        except Exception as e:
            self.signals.error_occurred.emit("Filter Move Error", str(e))

    def _refresh_pos(self):
        if self.filter_stage is None:
            self._pos_lbl.setText("Filter Stage: Not connected")
            return
        try:
            pos_nm = self.filter_stage.get_position()
            pos_um = pos_nm / 1000.0
            self._pos_lbl.setText(f"Position: {pos_um:.3f} µm")
        except Exception:
            self._pos_lbl.setText("Position: error")


# ---------------------------------------------------------------------------

class SavedPositionsWidget(QWidget):
    """Bookmark stage positions — same file as main app so positions are shared."""

    POSITIONS_FILE = "config/saved_positions.json"

    def __init__(self, stage, state, signals, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.state = state
        self.signals = signals
        self._positions: dict = self._load()
        self._init_ui()
        self._populate()

        self._pos_timer = QTimer()
        self._pos_timer.timeout.connect(self._update_cur_pos)
        self._pos_timer.start(200)

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        # Current position display + save
        cur_group = QGroupBox("Current Position")
        cur_layout = QVBoxLayout()

        self._cur_lbl = QLabel("X=?.???, Y=?.???, Z=?.??? µm")
        self._cur_lbl.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 13pt; font-weight: bold; "
            "background: #1E1E1E; color: #00FF00; padding: 10px; }"
        )
        cur_layout.addWidget(self._cur_lbl)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Name:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. 'Sample edge', 'Reference mark'")
        save_row.addWidget(self._name_input)
        btn_save = QPushButton("💾 Save")
        btn_save.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        btn_save.clicked.connect(self._save_current)
        save_row.addWidget(btn_save)
        cur_layout.addLayout(save_row)
        cur_group.setLayout(cur_layout)
        root.addWidget(cur_group)

        # Table
        tbl_group = QGroupBox("Saved Positions")
        tbl_layout = QVBoxLayout()

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Name", "Position (X, Y, Z) µm", "Saved", "Actions"])
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 120)
        tbl_layout.addWidget(self._table)

        bulk_row = QHBoxLayout()
        btn_csv = QPushButton("📤 Export CSV")
        btn_csv.clicked.connect(self._export_csv)
        bulk_row.addWidget(btn_csv)
        btn_clear = QPushButton("🗑 Clear All")
        btn_clear.clicked.connect(self._clear_all)
        bulk_row.addWidget(btn_clear)
        bulk_row.addStretch()
        tbl_layout.addLayout(bulk_row)

        tbl_group.setLayout(tbl_layout)
        root.addWidget(tbl_group)

    # ------------------------------------------------------------------
    def _update_cur_pos(self):
        x = self.state.stage_position["x"]
        y = self.state.stage_position["y"]
        z = self.state.stage_position["z"]
        self._cur_lbl.setText(f"X={x:.3f},  Y={y:.3f},  Z={z:.3f} µm")

    def _save_current(self):
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Enter a name for this position.")
            return
        if name in self._positions:
            if QMessageBox.question(
                self, "Overwrite?", f"'{name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.No:
                return
        self._positions[name] = {
            "x": self.state.stage_position["x"],
            "y": self.state.stage_position["y"],
            "z": self.state.stage_position["z"],
            "timestamp": datetime.now().isoformat()
        }
        self._persist()
        self._populate()
        self._name_input.clear()
        self.signals.status_message.emit(f"💾 Saved: {name}")

    def _populate(self):
        self._table.setRowCount(0)
        for name, pos in sorted(self._positions.items()):
            row = self._table.rowCount()
            self._table.insertRow(row)

            self._table.setItem(row, 0, QTableWidgetItem(name))

            pos_item = QTableWidgetItem(f"({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            pos_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 1, pos_item)

            ts = pos.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            ts_item = QTableWidgetItem(ts)
            ts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 2, ts_item)

            actions = QWidget()
            a_layout = QHBoxLayout()
            a_layout.setContentsMargins(2, 2, 2, 2)
            a_layout.setSpacing(2)
            btn_go = QPushButton("🎯 Go")
            btn_go.setStyleSheet(
                "QPushButton { background: #2196F3; color: white; font-weight: bold; padding: 4px; }"
            )
            btn_go.clicked.connect(lambda _, n=name: self._navigate(n))
            a_layout.addWidget(btn_go)
            btn_del = QPushButton("🗑")
            btn_del.clicked.connect(lambda _, n=name: self._delete(n))
            a_layout.addWidget(btn_del)
            actions.setLayout(a_layout)
            self._table.setCellWidget(row, 3, actions)

    def _navigate(self, name: str):
        pos = self._positions.get(name)
        if not pos:
            return
        if QMessageBox.question(
            self, "Navigate",
            f"Move stage to '{name}'?\n\n"
            f"X={pos['x']:.3f}  Y={pos['y']:.3f}  Z={pos['z']:.3f} µm",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        ) != QMessageBox.StandardButton.Ok:
            return
        try:
            self.stage.move_abs("y", pos["y"])
            self.stage.move_abs("x", pos["x"])
            self.stage.move_abs("z", pos["z"])
            self.signals.status_message.emit(f"✅ Moved to: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Move Failed", str(e))

    def _delete(self, name: str):
        if QMessageBox.question(
            self, "Delete", f"Delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            del self._positions[name]
            self._persist()
            self._populate()
            self.signals.status_message.emit(f"🗑 Deleted: {name}")

    def _clear_all(self):
        if not self._positions:
            return
        if QMessageBox.warning(
            self, "Clear All",
            f"Delete all {len(self._positions)} saved positions?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self._positions.clear()
            self._persist()
            self._populate()

    def _export_csv(self):
        if not self._positions:
            QMessageBox.information(self, "No Positions", "Nothing to export.")
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Export Positions",
            f"stage_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        if fname:
            import csv
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Name", "X_um", "Y_um", "Z_um", "Timestamp"])
                for name, pos in sorted(self._positions.items()):
                    w.writerow([name, pos["x"], pos["y"], pos["z"], pos.get("timestamp", "")])
            QMessageBox.information(self, "Exported", f"Saved to:\n{fname}")

    def _load(self) -> dict:
        p = Path(self.POSITIONS_FILE)
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _persist(self):
        p = Path(self.POSITIONS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._positions, f, indent=2)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class StageOnlyWindow(QMainWindow):
    """
    Lightweight stage-control window.
    No camera, no alignment — just move stages and bookmark positions.
    """

    def __init__(self, state, signals, stage, filter_stage, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.stage = stage
        self.filter_stage = filter_stage

        self._build_ui()
        self._connect_signals()
        self._start_position_updates()

        self.setWindowTitle("Stage Control")
        self.resize(1000, 700)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = self._make_splitter()
        main_layout.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._pos_status = QLabel("Stage: X=0.000  Y=0.000  Z=0.000 µm")
        self._pos_status.setStyleSheet("QLabel { font-family: monospace; }")
        self._status_bar.addPermanentWidget(self._pos_status)

        self._filter_status = QLabel("🔬 Filter: --")
        self._status_bar.addPermanentWidget(self._filter_status)

        mode_label = QLabel("⚙️  Stage-Only Mode")
        mode_label.setStyleSheet("QLabel { color: #FF9800; font-weight: bold; }")
        self._status_bar.addPermanentWidget(mode_label)

    def _make_splitter(self):
        from PyQt6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left pane — jog + filter
        left_tabs = QTabWidget()

        self._jog_widget = StageJogWidget(self.stage, self.state, self.signals)
        left_tabs.addTab(self._jog_widget, "XYZ Stage")

        self._filter_widget = FilterManualWidget(self.filter_stage, self.signals)
        left_tabs.addTab(self._filter_widget, "Filter Stage")

        splitter.addWidget(left_tabs)

        # Right pane — saved positions
        self._bookmarks_widget = SavedPositionsWidget(
            self.stage, self.state, self.signals
        )
        splitter.addWidget(self._bookmarks_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        return splitter

    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.signals.status_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self.signals.error_occurred.connect(self._show_error)
        self.signals.stage_position_changed.connect(self._update_pos_status)

    def _show_error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    def _update_pos_status(self, axis: str, position: float):
        x = self.state.stage_position["x"]
        y = self.state.stage_position["y"]
        z = self.state.stage_position["z"]
        self._pos_status.setText(f"Stage: X={x:.3f}  Y={y:.3f}  Z={z:.3f} µm")

        if self.filter_stage:
            try:
                pos_um = self.filter_stage.get_position() / 1000.0
                self._filter_status.setText(f"🔬 Filter: {pos_um:.3f} µm")
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _start_position_updates(self):
        self._pos_timer = QTimer()
        self._pos_timer.timeout.connect(self._poll_stage)
        self._pos_timer.start(100)

    def _poll_stage(self):
        if self.stage is None:
            return
        try:
            for axis in ("x", "y", "z"):
                pos = self.stage.get_pos(axis)
                self.state.update_stage_position(axis, pos)
                self.signals.stage_position_changed.emit(axis, pos)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Exit", "Exit Stage Control?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._pos_timer.stop()
            event.accept()
        else:
            event.ignore()