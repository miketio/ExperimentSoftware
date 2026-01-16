# app/widgets/saved_positions_panel.py
"""
Saved Positions Panel

Save and navigate to bookmarked stage positions.
Stores positions in a separate JSON file for persistence.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QGroupBox,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
import json
from pathlib import Path
from datetime import datetime


class SavedPositionsPanel(QWidget):
    """Panel for managing saved stage positions."""
    
    POSITIONS_FILE = "config/saved_positions.json"
    
    def __init__(self, state, signals, stage, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.stage = stage
        
        # Load saved positions
        self.saved_positions = self._load_positions()
        
        self._init_ui()
        self._populate_table()
        
        # Position update timer
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._update_current_position)
        self.position_timer.start(200)
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header = QLabel("📍 <b>Saved Positions</b>")
        header.setStyleSheet("QLabel { font-size: 14pt; }")
        layout.addWidget(header)
        
        info = QLabel(
            "Save interesting positions during exploration and quickly return to them.\n"
            "Positions persist across sessions."
        )
        info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Current position display
        current_group = QGroupBox("Current Position")
        current_layout = QVBoxLayout()
        
        self.current_pos_label = QLabel("X=?.???, Y=?.???, Z=?.??? µm")
        self.current_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 14pt; "
            "background-color: #2C2C2C; color: lime; padding: 12px; "
            "font-weight: bold; }"
        )
        current_layout.addWidget(self.current_pos_label)
        
        # Save controls
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Name:"))
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., 'Block 5 Entrance', 'Alignment Reference'")
        save_row.addWidget(self.name_input)
        
        self.btn_save = QPushButton("💾 Save Current Position")
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self.btn_save.clicked.connect(self._save_current_position)
        save_row.addWidget(self.btn_save)
        
        current_layout.addLayout(save_row)
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # Saved positions table
        table_group = QGroupBox(f"Saved Positions ({len(self.saved_positions)})")
        table_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            'Name', 'Position (X, Y, Z) µm', 'Saved', 'Actions'
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        
        # Column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 120)
        
        table_layout.addWidget(self.table)
        
        # Bulk actions
        bulk_row = QHBoxLayout()
        
        btn_export = QPushButton("📤 Export to CSV")
        btn_export.clicked.connect(self._export_positions)
        bulk_row.addWidget(btn_export)
        
        btn_clear_all = QPushButton("🗑 Clear All")
        btn_clear_all.clicked.connect(self._clear_all_positions)
        bulk_row.addWidget(btn_clear_all)
        
        bulk_row.addStretch()
        table_layout.addLayout(bulk_row)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
    
    def _update_current_position(self):
        """Update current position display."""
        x = self.state.stage_position['x']
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        self.current_pos_label.setText(
            f"X={x:.3f}, Y={y:.3f}, Z={z:.3f} µm"
        )
    
    def _save_current_position(self):
        """Save current position with user-provided name."""
        name = self.name_input.text().strip()
        
        if not name:
            QMessageBox.warning(
                self,
                "Name Required",
                "Please enter a name for this position."
            )
            return
        
        # Check for duplicate name
        if name in self.saved_positions:
            reply = QMessageBox.question(
                self,
                "Overwrite Position",
                f"Position '{name}' already exists.\n\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Get current position
        x = self.state.stage_position['x']
        y = self.state.stage_position['y']
        z = self.state.stage_position['z']
        
        # Save
        self.saved_positions[name] = {
            'x': x,
            'y': y,
            'z': z,
            'timestamp': datetime.now().isoformat()
        }
        
        # Persist to file
        self._save_positions()
        
        # Update table
        self._populate_table()
        
        # Clear input
        self.name_input.clear()
        
        # Visual feedback
        original_style = self.current_pos_label.styleSheet()
        self.current_pos_label.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 14pt; "
            "background-color: #4CAF50; color: white; padding: 12px; "
            "font-weight: bold; }"
        )
        QTimer.singleShot(500, lambda: self.current_pos_label.setStyleSheet(
            original_style
        ))
        
        self.signals.status_message.emit(f"💾 Saved position: {name}")
    
    def _populate_table(self):
        """Populate table with saved positions."""
        self.table.setRowCount(0)
        
        for name, pos in sorted(self.saved_positions.items()):
            self._add_position_to_table(name, pos)
        
        # Update group title
        parent = self.table.parent()
        if isinstance(parent, QGroupBox):
            parent.setTitle(f"Saved Positions ({len(self.saved_positions)})")
    
    def _add_position_to_table(self, name: str, pos: dict):
        """Add position to table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Name
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, name)
        self.table.setItem(row, 0, name_item)
        
        # Position
        pos_text = f"({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})"
        pos_item = QTableWidgetItem(pos_text)
        pos_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 1, pos_item)
        
        # Timestamp
        timestamp = pos.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_text = timestamp
        else:
            time_text = 'Unknown'
        
        time_item = QTableWidgetItem(time_text)
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 2, time_item)
        
        # Action buttons
        actions_widget = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(2)
        
        btn_go = QPushButton("🎯 Go")
        btn_go.setToolTip(f"Navigate to {name}")
        btn_go.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 4px; }"
        )
        btn_go.clicked.connect(lambda checked, n=name: self._navigate_to_position(n))
        actions_layout.addWidget(btn_go)
        
        btn_delete = QPushButton("🗑")
        btn_delete.setToolTip(f"Delete {name}")
        btn_delete.clicked.connect(lambda checked, n=name: self._delete_position(n))
        actions_layout.addWidget(btn_delete)
        
        actions_widget.setLayout(actions_layout)
        self.table.setCellWidget(row, 3, actions_widget)
    
    def _navigate_to_position(self, name: str):
        """Navigate to saved position."""
        if name not in self.saved_positions:
            return
        
        pos = self.saved_positions[name]
        
        reply = QMessageBox.question(
            self,
            "Navigate to Position",
            f"Move stage to '{name}'?\n\n"
            f"Target position:\n"
            f"X = {pos['x']:.3f} µm\n"
            f"Y = {pos['y']:.3f} µm\n"
            f"Z = {pos['z']:.3f} µm",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Ok:
            if self.stage is None:
                QMessageBox.warning(self, "No Stage", "Stage not connected")
                return
            
            try:
                # Move sequentially (focus first, then X/Z)
                self.stage.move_abs('y', pos['y'])
                self.stage.move_abs('x', pos['x'])
                self.stage.move_abs('z', pos['z'])
                
                self.signals.status_message.emit(f"✅ Moved to: {name}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Movement Failed",
                    f"Failed to move to position:\n\n{e}"
                )
    
    def _delete_position(self, name: str):
        """Delete saved position."""
        reply = QMessageBox.question(
            self,
            "Delete Position",
            f"Delete saved position '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.saved_positions[name]
            self._save_positions()
            self._populate_table()
            self.signals.status_message.emit(f"🗑 Deleted: {name}")
    
    def _clear_all_positions(self):
        """Clear all saved positions."""
        if not self.saved_positions:
            QMessageBox.information(self, "No Positions", "No saved positions to clear.")
            return
        
        reply = QMessageBox.warning(
            self,
            "Clear All Positions",
            f"Delete all {len(self.saved_positions)} saved positions?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.saved_positions.clear()
            self._save_positions()
            self._populate_table()
            self.signals.status_message.emit("🗑 All positions cleared")
    
    def _export_positions(self):
        """Export positions to CSV."""
        if not self.saved_positions:
            QMessageBox.information(self, "No Positions", "No positions to export.")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Positions",
            f"saved_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                import csv
                
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Name', 'X_um', 'Y_um', 'Z_um', 'Timestamp'])
                    
                    for name, pos in sorted(self.saved_positions.items()):
                        writer.writerow([
                            name,
                            pos['x'],
                            pos['y'],
                            pos['z'],
                            pos.get('timestamp', '')
                        ])
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported {len(self.saved_positions)} positions to:\n{filename}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export positions:\n\n{e}"
                )
    
    # ========================================
    # File I/O
    # ========================================
    
    def _load_positions(self) -> dict:
        """Load saved positions from JSON file."""
        path = Path(self.POSITIONS_FILE)
        
        if not path.exists():
            print(f"[SavedPositions] No saved positions file found")
            return {}
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"[SavedPositions] Loaded {len(data)} positions")
            return data
        except Exception as e:
            print(f"[SavedPositions] Failed to load positions: {e}")
            return {}
    
    def _save_positions(self):
        """Save positions to JSON file."""
        path = Path(self.POSITIONS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                json.dump(self.saved_positions, f, indent=2)
            print(f"[SavedPositions] Saved {len(self.saved_positions)} positions")
        except Exception as e:
            print(f"[SavedPositions] Failed to save positions: {e}")
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save positions:\n\n{e}"
            )