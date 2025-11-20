"""Block Selection Grid - Dynamic grid of blocks."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QPushButton, QGroupBox, QScrollArea
from PyQt6.QtCore import Qt
from app.system_state import AlignmentStatus


class BlockButton(QPushButton):
    """Colored button representing a block."""
    
    COLORS = {
        AlignmentStatus.NOT_CALIBRATED: "#CCCCCC",  # Gray
        AlignmentStatus.GLOBAL_ONLY: "#FFD700",     # Yellow
        AlignmentStatus.BLOCK_CALIBRATED: "#90EE90", # Green
        AlignmentStatus.FAILED: "#FF6B6B"           # Red
    }
    
    def __init__(self, block_id, parent=None):
        super().__init__(str(block_id), parent)
        self.block_id = block_id
        self.status = AlignmentStatus.NOT_CALIBRATED
        self.is_selected = False
        self.setFixedSize(60, 50)
        self._update_style()
    
    def set_status(self, status: AlignmentStatus):
        """Update block status."""
        self.status = status
        self._update_style()
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.is_selected = selected
        self._update_style()
    
    def _update_style(self):
        """Update button style based on state."""
        bg_color = self.COLORS[self.status]
        border_color = "#0066CC" if self.is_selected else "#666666"
        border_width = 3 if self.is_selected else 1
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: {border_width}px solid {border_color};
                border-radius: 5px;
                font-weight: bold;
                font-size: 12pt;
            }}
            QPushButton:hover {{
                border-color: #0099FF;
            }}
        """)


class BlockGridWidget(QWidget):
    """Dynamic grid for block selection based on layout."""
    
    def __init__(self, state, signals, layout_model, parent=None):
        # layout_model can be RuntimeLayout or CameraLayout
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.layout_model = layout_model
        self.buttons = {}
        
        self._init_ui()
        self._connect_signals()
        self._update_all_buttons()
    
    def _init_ui(self):
        """Initialize UI."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        group = QGroupBox("Block Selection")
        group_layout = QVBoxLayout()
        
        # Scroll Area in case grid is huge
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setSpacing(5)
        
        # Dynamic Grid Generation
        cols = self.layout_model.block_layout.blocks_per_row
        rows = self.layout_model.block_layout.num_rows
        
        # Ensure we respect 1-based ID Logic from generator
        for r in range(rows):
            for c in range(cols):
                # Calculate ID: row * width + col + 1
                block_id = r * cols + c + 1
                
                # Only add button if block exists in layout (safety check)
                if block_id in self.layout_model.blocks:
                    btn = BlockButton(block_id)
                    btn.clicked.connect(lambda checked, bid=block_id: self._on_block_clicked(bid))
                    self.buttons[block_id] = btn
                    grid.addWidget(btn, r, c)
        
        # Add grid container to scroll area
        scroll.setWidget(grid_container)
        group_layout.addWidget(scroll)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("⬜ Not calib"))
        legend_layout.addWidget(QLabel("🟨 Global"))
        legend_layout.addWidget(QLabel("🟩 Block calib"))
        legend_layout.addWidget(QLabel("🔴 Failed"))
        legend_layout.addStretch()
        group_layout.addLayout(legend_layout)
        
        # Selection info
        self.info_label = QLabel("No block selected")
        group_layout.addWidget(self.info_label)
        
        # === CHANGED: Add stretch here to push everything up ===
        group_layout.addStretch()
        
        group.setLayout(group_layout)
        main_layout.addWidget(group)
    
    def _connect_signals(self):
        """Connect signals."""
        self.signals.block_selected.connect(self._update_selection)
        self.signals.block_alignment_complete.connect(lambda bid, res: self._update_button(bid))
        self.signals.global_alignment_complete.connect(lambda res: self._update_all_buttons())
    
    def _on_block_clicked(self, block_id: int):
        """Handle block button click."""
        self.signals.block_selected.emit(block_id)
    
    def _update_selection(self, block_id: int):
        """Update visual selection."""
        for bid, btn in self.buttons.items():
            btn.set_selected(bid == block_id)
        
        if self.state:
            block_state = self.state.get_block_state(block_id)
            self.info_label.setText(
                f"Block {block_id} selected | "
                f"Status: {block_state.status.name}"
            )
    
    def _update_button(self, block_id: int):
        """Update single button status."""
        if block_id in self.buttons and self.state:
            state = self.state.get_block_state(block_id)
            self.buttons[block_id].set_status(state.status)
    
    def _update_all_buttons(self):
        """Update all button statuses."""
        for block_id in self.buttons:
            self._update_button(block_id)