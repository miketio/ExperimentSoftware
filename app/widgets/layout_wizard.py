# app/widgets/layout_wizard.py - UPDATED (No Block 1 Position Step)
"""
Layout Wizard - Multi-step layout creation

Step 1: Block array parameters
Step 2: ASCII file assignment

Block 1 position is now set separately in the main window.
"""

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QListWidget, QGridLayout,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsTextItem, QListWidgetItem, QDialog
)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QBrush, QColor, QPen, QFont, QDrag
from pathlib import Path
from typing import Optional
import json


class ArrayParametersPage(QWizardPage):
    """Step 1: Define block array parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Block Array Configuration")
        self.setSubTitle("Define the physical arrangement of blocks")
        
        layout = QVBoxLayout()
        
        # Parameters
        params = QGridLayout()
        
        params.addWidget(QLabel("Blocks per row:"), 0, 0)
        self.blocks_per_row = QSpinBox()
        self.blocks_per_row.setRange(1, 10)
        self.blocks_per_row.setValue(5)
        self.registerField("blocks_per_row", self.blocks_per_row)
        params.addWidget(self.blocks_per_row, 0, 1)
        
        params.addWidget(QLabel("Number of rows:"), 1, 0)
        self.num_rows = QSpinBox()
        self.num_rows.setRange(1, 10)
        self.num_rows.setValue(4)
        self.registerField("num_rows", self.num_rows)
        params.addWidget(self.num_rows, 1, 1)
        
        params.addWidget(QLabel("Block size:"), 2, 0)
        self.block_size = QDoubleSpinBox()
        self.block_size.setRange(50, 1000)
        self.block_size.setValue(200)
        self.block_size.setSuffix(" µm")
        self.registerField("block_size", self.block_size, "value")
        params.addWidget(self.block_size, 2, 1)
        
        params.addWidget(QLabel("Block spacing:"), 3, 0)
        self.block_spacing = QDoubleSpinBox()
        self.block_spacing.setRange(100, 2000)
        self.block_spacing.setValue(300)
        self.block_spacing.setSuffix(" µm")
        self.registerField("block_spacing", self.block_spacing, "value")
        params.addWidget(self.block_spacing, 3, 1)
        
        layout.addLayout(params)
        
        # Info
        info = QLabel(
            "This defines the physical array of blocks on your sample.\n"
            "Next, you'll assign ASCII design files to each block."
        )
        info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        layout.addStretch()
        self.setLayout(layout)


class AsciiAssignmentPage(QWizardPage):
    """Step 2: Assign ASCII files to blocks."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("ASCII File Assignment")
        self.setSubTitle("Click on blocks to assign ASCII files")
        
        self.ascii_files = {}  # {filename: parsed_data}
        self.block_assignments = {}  # {block_id: filename}
        
        layout = QHBoxLayout()
        
        # Left: ASCII file list
        left_panel = QVBoxLayout()
        
        left_panel.addWidget(QLabel("<b>Available ASCII Files:</b>"))
        
        self.ascii_list = QListWidget()
        self.ascii_list.itemClicked.connect(self._on_ascii_selected)
        left_panel.addWidget(self.ascii_list)
        
        btn_layout = QHBoxLayout()
        
        btn_add = QPushButton("Add File...")
        btn_add.clicked.connect(self._add_ascii_file)
        btn_layout.addWidget(btn_add)
        
        btn_add_folder = QPushButton("Add from Folder...")
        btn_add_folder.clicked.connect(self._add_from_folder)
        btn_layout.addWidget(btn_add_folder)
        
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_ascii_file)
        btn_layout.addWidget(btn_remove)
        
        left_panel.addLayout(btn_layout)
        
        # Right: Block grid
        right_panel = QVBoxLayout()
        
        right_panel.addWidget(QLabel("<b>Block Grid:</b>"))
        
        info = QLabel(
            "1. Select an ASCII file from the list\n"
            "2. Click on blocks to assign the selected file\n"
            "3. Each block gets ONE ASCII file"
        )
        info.setStyleSheet("QLabel { color: #666; font-size: 9pt; background-color: #F0F0F0; padding: 5px; }")
        info.setWordWrap(True)
        right_panel.addWidget(info)
        
        self.selected_ascii_label = QLabel("Selected: None")
        self.selected_ascii_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        right_panel.addWidget(self.selected_ascii_label)
        
        self.block_grid = BlockGridWidget(self)
        right_panel.addWidget(self.block_grid)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)
        
        self.setLayout(layout)
        
        self.selected_ascii_file = None
    
    def initializePage(self):
        """Called when page is shown."""
        # Get array parameters
        rows = self.field("num_rows")
        cols = self.field("blocks_per_row")
        
        # Update block grid
        self.block_grid.setup_grid(rows, cols)
        
        # Auto-scan for ASCII files
        ascii_folder = Path("config/ascii")
        if ascii_folder.exists():
            for ascii_file in ascii_folder.glob("*.ASC"):
                self._load_ascii_file(str(ascii_file))
    
    def _on_ascii_selected(self, item):
        """Handle ASCII file selection."""
        self.selected_ascii_file = item.data(Qt.ItemDataRole.UserRole)
        filename = Path(self.selected_ascii_file).name
        self.selected_ascii_label.setText(f"Selected: {filename}")
        
        # Highlight in list
        self.ascii_list.clearSelection()
        item.setSelected(True)
    
    def assign_to_block(self, block_id: int):
        """Assign currently selected ASCII to block."""
        if self.selected_ascii_file is None:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select an ASCII file from the list first."
            )
            return
        
        # Assign
        self.block_assignments[block_id] = self.selected_ascii_file
        
        # Update visual
        self.block_grid.update_block_assignment(block_id, self.selected_ascii_file)
        
        print(f"Assigned {Path(self.selected_ascii_file).name} to Block {block_id}")
    
    def _add_ascii_file(self):
        """Browse and add single ASCII file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASCII File",
            "config/ascii",
            "ASCII Files (*.ASC *.asc);;All Files (*)"
        )
        
        if filename:
            self._load_ascii_file(filename)
    
    def _add_from_folder(self):
        """Add all ASCII files from folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ASCII Folder",
            "config/ascii"
        )
        
        if folder:
            for ascii_file in Path(folder).glob("*.ASC"):
                self._load_ascii_file(str(ascii_file))
    
    def _load_ascii_file(self, filename: str):
        """Load and parse ASCII file."""
        if filename in self.ascii_files:
            return
        
        try:
            from alignment_system.ascii_parser import ASCIIParser
            
            parser = ASCIIParser(filename)
            parsed = parser.parse()
            
            self.ascii_files[filename] = parsed
            
            # Add to list
            name = Path(filename).name
            num_wg = len(parsed.get('waveguides', []))
            item = QListWidgetItem(f"{name} ({num_wg} WGs)")
            item.setData(Qt.ItemDataRole.UserRole, filename)
            self.ascii_list.addItem(item)
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Parse Failed",
                f"Failed to parse {Path(filename).name}:\n{e}"
            )
    
    def _remove_ascii_file(self):
        """Remove selected ASCII file."""
        current = self.ascii_list.currentItem()
        if current:
            filename = current.data(Qt.ItemDataRole.UserRole)
            
            # Check if assigned to any blocks
            assigned_blocks = [bid for bid, f in self.block_assignments.items() if f == filename]
            if assigned_blocks:
                reply = QMessageBox.question(
                    self,
                    "File In Use",
                    f"This file is assigned to {len(assigned_blocks)} block(s).\n\nRemove anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                
                # Clear assignments
                for bid in assigned_blocks:
                    del self.block_assignments[bid]
                    self.block_grid.update_block_assignment(bid, None)
            
            del self.ascii_files[filename]
            self.ascii_list.takeItem(self.ascii_list.row(current))
            
            if self.selected_ascii_file == filename:
                self.selected_ascii_file = None
                self.selected_ascii_label.setText("Selected: None")
    
    def validatePage(self):
        """Check if all blocks have assignments."""
        total_blocks = self.field("blocks_per_row") * self.field("num_rows")
        
        if len(self.block_assignments) < total_blocks:
            reply = QMessageBox.question(
                self,
                "Incomplete Assignments",
                f"Only {len(self.block_assignments)}/{total_blocks} blocks assigned.\n\n"
                "Unassigned blocks will use a default template.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes
        
        return True


class BlockGridWidget(QGraphicsView):
    """Click-based assignment grid."""
    
    def __init__(self, parent_page):
        super().__init__()
        self.parent_page = parent_page
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        self.block_items = {}
        self.block_labels = {}
    
    def setup_grid(self, rows: int, cols: int):
        """Create block grid."""
        self.scene.clear()
        self.block_items.clear()
        self.block_labels.clear()
        
        block_size = 60
        spacing = 10
        
        block_id = 1
        for row in range(rows):
            for col in range(cols):
                x = col * (block_size + spacing)
                y = row * (block_size + spacing)
                
                # Block rectangle
                rect = QGraphicsRectItem(x, y, block_size, block_size)
                rect.setBrush(QBrush(QColor(220, 220, 220)))
                rect.setPen(QPen(QColor(100, 100, 100), 2))
                rect.setData(0, block_id)
                self.scene.addItem(rect)
                self.block_items[block_id] = rect
                
                # Block number
                text = QGraphicsTextItem(str(block_id))
                text.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                text.setPos(x + block_size/2 - 10, y + 5)
                self.scene.addItem(text)
                
                # Assignment label
                label = QGraphicsTextItem("")
                label.setFont(QFont("Arial", 7))
                label.setPos(x + 5, y + block_size - 20)
                label.setDefaultTextColor(QColor(0, 100, 200))
                self.scene.addItem(label)
                self.block_labels[block_id] = label
                
                block_id += 1
        
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def mousePressEvent(self, event):
        """Handle click on block."""
        pos = self.mapToScene(event.pos())
        items = self.scene.items(pos)
        
        for item in items:
            if isinstance(item, QGraphicsRectItem):
                block_id = item.data(0)
                if block_id:
                    self.parent_page.assign_to_block(block_id)
                    return
        
        super().mousePressEvent(event)
    
    def update_block_assignment(self, block_id: int, filename: Optional[str]):
        """Update block visual after assignment."""
        if block_id not in self.block_items:
            return
        
        rect = self.block_items[block_id]
        label = self.block_labels[block_id]
        
        if filename:
            # Assigned
            name = Path(filename).stem
            label.setPlainText(name[:10])
            rect.setBrush(QBrush(QColor(200, 255, 200)))  # Green
        else:
            # Unassigned
            label.setPlainText("")
            rect.setBrush(QBrush(QColor(220, 220, 220)))  # Gray


class LayoutWizard(QWizard):
    """Main wizard for layout creation (NO Block 1 position step)."""
    
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.runtime_layout = None
        
        self.setWindowTitle("Layout Creation Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(800, 600)
        
        # Add pages (only 2 steps now)
        self.addPage(ArrayParametersPage())
        self.ascii_page = AsciiAssignmentPage()
        self.addPage(self.ascii_page)
        
        # Connect finish
        self.finished.connect(self._on_finish)
    
    def _on_finish(self, result):
        """Generate RuntimeLayout on completion."""
        if result != QDialog.DialogCode.Accepted:
            return
        
        try:
            # Get parameters
            blocks_per_row = self.field("blocks_per_row")
            num_rows = self.field("num_rows")
            block_size = self.field("block_size")
            block_spacing = self.field("block_spacing")
            print(f"DEBUG: Blocks/Row: {blocks_per_row}, Rows: {num_rows}, Size: {block_size}, Spacing: {block_spacing}")
            # Generate RuntimeLayout from ASCII assignments
            ascii_files = list(self.ascii_page.ascii_files.keys())
            if not ascii_files:
                raise ValueError("No ASCII files assigned")
            
            # Use layout generator
            from config.layout_config_generator import generate_layout_config_v3
            from config.layout_models import RuntimeLayout
            
            # Generate with first ASCII file (temporary)
            _ = generate_layout_config_v3(
                ascii_file=ascii_files[0],
                output_file="config/runtime_layout.json",
                blocks_per_row=blocks_per_row,
                num_rows=num_rows,
                block_size=block_size,
                block_spacing=block_spacing,
                simulated_rotation=0.0,
                simulated_translation=(0.0, 0.0)
            )
            
            # Load as RuntimeLayout
            self.runtime_layout = RuntimeLayout.from_json_file("config/runtime_layout.json")
            
            # DO NOT set Block 1 position here - user will do it in main window
            
            # Save (without Block 1 position)
            self.runtime_layout.save_to_json("config/runtime_layout.json", include_design=True)
            
            print(f"✅ Layout created: {blocks_per_row}x{num_rows} blocks")
            print(f"⚠️  Remember to set Block 1 position in the main window!")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Generation Failed",
                f"Failed to generate layout:\n\n{e}"
            )
            import traceback
            traceback.print_exc()
    
    def get_runtime_layout(self):
        """Get generated RuntimeLayout."""
        return self.runtime_layout