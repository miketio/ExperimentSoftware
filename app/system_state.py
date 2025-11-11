"""
System State Management

Centralized state for the entire application. All coordinates in micrometers (µm).
Thread-safe for reading, updates should happen in main thread via signals.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path


class AlignmentStatus(Enum):
    """Block alignment status."""
    NOT_CALIBRATED = 0
    GLOBAL_ONLY = 1
    BLOCK_CALIBRATED = 2
    FAILED = 3


class HardwareMode(Enum):
    """Hardware operation mode."""
    MOCK = "mock"
    REAL = "real"
    DISCONNECTED = "disconnected"


@dataclass
class BlockState:
    """State for a single block."""
    block_id: int
    status: AlignmentStatus = AlignmentStatus.NOT_CALIBRATED
    calibration_error: Optional[float] = None  # µm
    fiducials_found: int = 0
    last_visited: Optional[float] = None  # timestamp
    last_waveguide: int = 25  # last visited waveguide in this block


@dataclass
class NavigationState:
    """Navigation and target state."""
    current_block: Optional[int] = None
    target_waveguide: int = 25
    target_grating_side: str = "left"  # "left", "center", "right"
    position_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # History format: (x, y, z, timestamp)
    
    def add_to_history(self, x: float, y: float, z: float):
        """Add position to history (auto-timestamps)."""
        self.position_history.append((x, y, z, time.time()))
        if len(self.position_history) > 100:
            self.position_history.pop(0)


@dataclass
class CameraState:
    """Camera display state."""
    color_scale_min: int = 0
    color_scale_max: int = 4095
    color_scale_auto: bool = True
    colormap: str = "gray"  # "gray", "jet", "hot", "viridis"
    zoom_level: float = 1.0
    show_crosshair: bool = True
    show_scale_bar: bool = True
    show_fourier: bool = False  # NEW: Fourier transform mode
    beam_position_px: Tuple[int, int] = (512, 512)  # Beam location in pixels
    show_beam_indicator: bool = True  # Show beam crosshair
    um_per_pixel: float = 0  # Micrometers per pixel
    
class SystemState:
    """
    Centralized application state.
    
    All positions in micrometers (µm).
    Thread-safe for reading, modifications via signals preferred.
    """
    
    def __init__(self):
        # Hardware state
        self.hardware_mode: HardwareMode = HardwareMode.MOCK
        self.camera_connected: bool = False
        self.stage_connected: bool = False
        self.stage_position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Alignment state
        self.global_calibrated: bool = False
        self.global_calibration_params: Optional[Dict] = None
        self.blocks: Dict[int, BlockState] = {
            i: BlockState(block_id=i) for i in range(1, 21)
        }
        
        # Navigation
        self.navigation = NavigationState()
        
        # Camera
        self.camera = CameraState()
        
        # Configuration
        self.alignment_config = {
            'search_radius_um': 50.0,
            'step_size_um': 10.0,
            'confidence_threshold': 0.5,
        }
        
        self.autofocus_config = {
            'range_um': 10.0,
            'step_um': 0.5,
            'auto_after_nav': True,
        }
        
        self.stage_config = {
            'jog_step_um': 10.0,  # current jog step size
            'safe_move_threshold_um': 1000.0,  # ask confirmation above this
        }
    
    # ========================================================================
    # Stage Position Management
    # ========================================================================
    
    def update_stage_position(self, axis: str, position: float):
        """Update stage position (µm)."""
        self.stage_position[axis] = float(position)
    
    def get_stage_position(self) -> Tuple[float, float, float]:
        """Get current stage position (x, y, z) in µm."""
        return (
            self.stage_position['x'],
            self.stage_position['y'],
            self.stage_position['z']
        )
    
    # ========================================================================
    # Block Management
    # ========================================================================
    
    def set_block_status(
        self, 
        block_id: int, 
        status: AlignmentStatus, 
        error: Optional[float] = None,
        fiducials_found: int = 0
    ):
        """Update block calibration status."""
        if block_id not in self.blocks:
            raise KeyError(f"Unknown block {block_id}")
        
        block = self.blocks[block_id]
        block.status = status
        block.calibration_error = error
        block.fiducials_found = fiducials_found
        block.last_visited = time.time()
    
    def get_block_state(self, block_id: int) -> BlockState:
        """Get block state."""
        if block_id not in self.blocks:
            raise KeyError(f"Unknown block {block_id}")
        return self.blocks[block_id]
    
    def get_calibrated_blocks(self) -> List[int]:
        """Get list of block IDs that are calibrated."""
        return [
            bid for bid, block in self.blocks.items()
            if block.status == AlignmentStatus.BLOCK_CALIBRATED
        ]
    
    # ========================================================================
    # Navigation
    # ========================================================================
    
    def set_current_block(self, block_id: Optional[int]):
        """Set currently selected block."""
        self.navigation.current_block = block_id
        if block_id is not None:
            # Restore last waveguide for this block
            self.navigation.target_waveguide = self.blocks[block_id].last_waveguide
    
    def set_target_waveguide(self, waveguide: int):
        """Set target waveguide number."""
        self.navigation.target_waveguide = waveguide
        # Remember for current block
        if self.navigation.current_block is not None:
            self.blocks[self.navigation.current_block].last_waveguide = waveguide
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_jog_step(self, step_um: float):
        """Set stage jog step size (µm)."""
        self.stage_config['jog_step_um'] = float(step_um)
    
    def get_jog_step(self) -> float:
        """Get current jog step (µm)."""
        return self.stage_config['jog_step_um']
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save_to_file(self, filename: str):
        """Save state to JSON file."""
        payload = {
            'timestamp': time.time(),
            'hardware_mode': self.hardware_mode.value,
            'stage_position': self.stage_position,
            'global_calibrated': self.global_calibrated,
            'global_params': self.global_calibration_params,
            'blocks': {
                str(bid): {
                    'status': b.status.name,
                    'error': b.calibration_error,
                    'fiducials': b.fiducials_found,
                    'last_visited': b.last_visited,
                    'last_waveguide': b.last_waveguide,
                } for bid, b in self.blocks.items()
            },
            'navigation': {
                'current_block': self.navigation.current_block,
                'target_waveguide': self.navigation.target_waveguide,
                'target_grating_side': self.navigation.target_grating_side,
                'history': self.navigation.position_history[-50:],  # last 50 positions
            },
            'camera': {
                'color_scale_min': self.camera.color_scale_min,
                'color_scale_max': self.camera.color_scale_max,
                'color_scale_auto': self.camera.color_scale_auto,
                'colormap': self.camera.colormap,
                'zoom_level': self.camera.zoom_level,
            },
            'config': {
                'alignment': self.alignment_config,
                'autofocus': self.autofocus_config,
                'stage': self.stage_config,
            }
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load state from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Hardware (don't restore mode - user selects at startup)
        if 'stage_position' in data:
            self.stage_position.update(data['stage_position'])
        
        # Alignment
        self.global_calibrated = data.get('global_calibrated', False)
        self.global_calibration_params = data.get('global_params')
        
        # Blocks
        for bid_str, bd in data.get('blocks', {}).items():
            bid = int(bid_str)
            if bid in self.blocks:
                try:
                    status = AlignmentStatus[bd.get('status', 'NOT_CALIBRATED')]
                except KeyError:
                    status = AlignmentStatus.NOT_CALIBRATED
                
                self.set_block_status(
                    bid, 
                    status, 
                    bd.get('error'),
                    bd.get('fiducials', 0)
                )
                self.blocks[bid].last_visited = bd.get('last_visited')
                self.blocks[bid].last_waveguide = bd.get('last_waveguide', 25)
        
        # Navigation
        nav_data = data.get('navigation', {})
        self.navigation.current_block = nav_data.get('current_block')
        self.navigation.target_waveguide = nav_data.get('target_waveguide', 25)
        self.navigation.target_grating_side = nav_data.get('target_grating_side', 'left')
        self.navigation.position_history = nav_data.get('history', [])
        
        # Camera
        cam_data = data.get('camera', {})
        self.camera.color_scale_min = cam_data.get('color_scale_min', 0)
        self.camera.color_scale_max = cam_data.get('color_scale_max', 4095)
        self.camera.color_scale_auto = cam_data.get('color_scale_auto', True)
        self.camera.colormap = cam_data.get('colormap', 'gray')
        self.camera.zoom_level = cam_data.get('zoom_level', 1.0)
        
        # Config
        config_data = data.get('config', {})
        if 'alignment' in config_data:
            self.alignment_config.update(config_data['alignment'])
        if 'autofocus' in config_data:
            self.autofocus_config.update(config_data['autofocus'])
        if 'stage' in config_data:
            self.stage_config.update(config_data['stage'])
    
    # ========================================================================
    # Status Queries
    # ========================================================================
    
    def get_alignment_summary(self) -> Dict:
        """Get alignment status summary."""
        calibrated_count = len(self.get_calibrated_blocks())
        return {
            'global_calibrated': self.global_calibrated,
            'blocks_calibrated': calibrated_count,
            'total_blocks': len(self.blocks),
            'calibration_percentage': (calibrated_count / len(self.blocks)) * 100
        }
    
    def is_ready_for_navigation(self) -> bool:
        """Check if system is ready for navigation."""
        return (
            self.camera_connected and 
            self.stage_connected and 
            self.global_calibrated and
            self.navigation.current_block is not None
        )
    
    def __repr__(self) -> str:
        return (
            f"SystemState("
            f"hardware={self.hardware_mode.value}, "
            f"stage={self.get_stage_position()}, "
            f"global_cal={self.global_calibrated}, "
            f"blocks_cal={len(self.get_calibrated_blocks())}/20)"
        )