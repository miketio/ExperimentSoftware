"""
layout_models.py - Type-safe layout models with camera/runtime separation.

Architecture:
- CameraLayout: Full access to design + ground truth (for rendering)
- RuntimeLayout: Design only + measurement accumulation (for analysis)

Both load from the same JSON file, but expose different data.
Ground truth includes all fabrication errors (not stored per-block).

Example:
    # Camera gets everything
    >>> camera_layout = CameraLayout.from_json_file("config/mock_layout.json")
    >>> camera = MockCamera(camera_layout, stage_ref=stage)
    
    # Runtime starts empty, fills during measurement
    >>> runtime = RuntimeLayout.from_json_file("config/mock_layout.json")
    >>> alignment = HierarchicalAlignment(runtime)
    >>> 
    >>> # After Stage 1 calibration
    >>> runtime.set_global_calibration(rotation=2.94, translation=(50.2, 29.8))
    >>> 
    >>> # Save results
    >>> runtime.save_to_json("results/calibration_2024-01-15.json")
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from enum import Enum
from datetime import datetime


# ============================================================================
# SHARED PRIMITIVES
# ============================================================================

@dataclass(frozen=True)
class Point2D:
    """Immutable 2D point in micrometers."""
    u: float
    v: float
    
    def __iter__(self):
        yield self.u
        yield self.v
    
    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.u
        elif idx == 1:
            return self.v
        raise IndexError(f"Point2D index out of range: {idx}")
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.u, self.v)
    
    def to_list(self) -> List[float]:
        return [self.u, self.v]
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'Point2D':
        if len(coords) != 2:
            raise ValueError(f"Expected 2 coordinates, got {len(coords)}")
        return cls(u=float(coords[0]), v=float(coords[1]))
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> 'Point2D':
        return cls(u=float(coords[0]), v=float(coords[1]))


class CornerType(str, Enum):
    """Valid corner identifiers."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class GratingSide(str, Enum):
    """Valid grating coupler sides."""
    LEFT = "left"
    RIGHT = "right"


# ============================================================================
# DESIGN PRIMITIVES (immutable, no fabrication errors)
# ============================================================================

@dataclass(frozen=True)
class Waveguide:
    """Waveguide as designed."""
    number: int
    v_center: float
    width: float
    u_start: float
    u_end: float
    
    @property
    def length(self) -> float:
        return abs(self.u_end - self.u_start)
    
    @property
    def center_position(self) -> Point2D:
        u_center = (self.u_start + self.u_end) / 2.0
        return Point2D(u=u_center, v=self.v_center)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Waveguide':
        return cls(
            number=int(data['number']),
            v_center=float(data['v_center']),
            width=float(data['width']),
            u_start=float(data['u_start']),
            u_end=float(data['u_end'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'number': self.number,
            'v_center': self.v_center,
            'width': self.width,
            'u_start': self.u_start,
            'u_end': self.u_end
        }


@dataclass(frozen=True)
class Grating:
    """Grating coupler as designed."""
    position: Point2D
    side: GratingSide
    waveguide: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grating':
        return cls(
            position=Point2D.from_list(data['position']),
            side=GratingSide(data['side']),
            waveguide=int(data['waveguide'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position.to_list(),
            'side': self.side.value,
            'waveguide': self.waveguide
        }


@dataclass(frozen=True)
class Block:
    """
    Block as designed (pure CAD data, no fabrication errors).
    
    ⚠️ CHANGE: Local coordinates are relative to BLOCK CENTER.
    ⚠️ OLD: Fiducials/gratings were relative to bottom-left corner.
    ⚠️ NEW: Fiducials/gratings are relative to center (0, 0).
    
    Fabrication errors are NOT stored here - they're in GroundTruth.
    This prevents RuntimeLayout from accidentally accessing them.
    """
    id: int
    row: int
    col: int
    size: int 
    design_position: Point2D
    fiducials: Dict[CornerType, Point2D]
    waveguides: Dict[str, Waveguide]
    gratings: Dict[str, Grating]
    
    @classmethod
    def from_dict(cls, block_id: int, data: Dict[str, Any]) -> 'Block':
        fiducials = {
            CornerType(corner): Point2D.from_list(pos)
            for corner, pos in data['fiducials'].items()
        }
        
        waveguides = {
            wg_id: Waveguide.from_dict(wg_data)
            for wg_id, wg_data in data.get('waveguides', {}).items()
        }
        
        gratings = {
            gr_id: Grating.from_dict(gr_data)
            for gr_id, gr_data in data.get('gratings', {}).items()
        }
        
        return cls(
            id=block_id,
            row=int(data['row']),
            col=int(data['col']),
            size=int(data['size']),
            design_position=Point2D.from_list(data['design_position']),
            fiducials=fiducials,
            waveguides=waveguides,
            gratings=gratings
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'row': self.row,
            'col': self.col,
            'size': self.size,
            'design_position': self.design_position.to_list(),
            'fiducials': {
                corner.value: pos.to_list()
                for corner, pos in self.fiducials.items()
            },
            'waveguides': {
                wg_id: wg.to_dict()
                for wg_id, wg in self.waveguides.items()
            },
            'gratings': {
                gr_id: gr.to_dict()
                for gr_id, gr in self.gratings.items()
            }
        }
    
    def get_fiducial(self, corner: str) -> Point2D:
        """Get fiducial position (local coordinates)."""
        corner_type = CornerType(corner)
        if corner_type not in self.fiducials:
            raise KeyError(f"Fiducial '{corner}' not found in block {self.id}")
        return self.fiducials[corner_type]
    
    def get_waveguide(self, number: int) -> Waveguide:
        """Get waveguide by number."""
        wg_id = f"wg{number}"
        if wg_id not in self.waveguides:
            raise KeyError(f"Waveguide {number} not found in block {self.id}")
        return self.waveguides[wg_id]
    
    def get_grating(self, waveguide: int, side: str) -> Grating:
        """Get grating coupler."""
        gr_id = f"wg{waveguide}_{side}"
        if gr_id not in self.gratings:
            raise KeyError(f"Grating {gr_id} not found in block {self.id}")
        return self.gratings[gr_id]
    
    def list_waveguides(self) -> List[int]:
        """Get sorted list of waveguide numbers."""
        return sorted([wg.number for wg in self.waveguides.values()])


@dataclass(frozen=True)
class BlockLayoutParams:
    """Array layout parameters."""
    block_size: float
    block_spacing: float
    blocks_per_row: int
    num_rows: int
    total_blocks: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockLayoutParams':
        return cls(
            block_size=float(data['block_size']),
            block_spacing=float(data['block_spacing']),
            blocks_per_row=int(data['blocks_per_row']),
            num_rows=int(data['num_rows']),
            total_blocks=int(data['total_blocks'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# GROUND TRUTH (for camera/simulation only)
# ============================================================================

@dataclass(frozen=True)
class BlockFabricationError:
    """Per-block fabrication error (rotation + translation)."""
    rotation_deg: float
    translation_um: Point2D
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockFabricationError':
        return cls(
            rotation_deg=float(data['rotation_deg']),
            translation_um=Point2D.from_list(data['translation_um'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rotation_deg': self.rotation_deg,
            'translation_um': self.translation_um.to_list()
        }


@dataclass(frozen=True)
class GenerationParams:
    """Parameters used to generate fabrication errors."""
    block_rotation_std_deg: float
    block_translation_std_um: float
    random_seed: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationParams':
        return cls(
            block_rotation_std_deg=float(data['block_rotation_std_deg']),
            block_translation_std_um=float(data['block_translation_std_um']),
            random_seed=int(data['random_seed'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruth:
    """
    Ground truth for simulation/testing.
    
    Contains:
    - Global sample transformation
    - Per-block fabrication errors
    - Generation parameters
    
    Camera has access to this. RuntimeLayout does NOT.
    """
    rotation_deg: float
    translation_um: Point2D
    block_fabrication_errors: Dict[int, BlockFabricationError]
    generation_params: Optional[GenerationParams] = None
    description: str = "Simulation ground truth"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruth':
        # Parse block errors
        block_errors = {}
        if 'block_fabrication_errors' in data:
            block_errors = {
                int(block_id): BlockFabricationError.from_dict(err_data)
                for block_id, err_data in data['block_fabrication_errors'].items()
            }
        
        # Parse generation params
        gen_params = None
        if 'generation_params' in data:
            gen_params = GenerationParams.from_dict(data['generation_params'])
        
        return cls(
            rotation_deg=float(data['rotation_deg']),
            translation_um=Point2D.from_list(data['translation_um']),
            block_fabrication_errors=block_errors,
            generation_params=gen_params,
            description=data.get('description', 'Simulation ground truth')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'rotation_deg': self.rotation_deg,
            'translation_um': self.translation_um.to_list(),
            'description': self.description
        }
        
        if self.block_fabrication_errors:
            result['block_fabrication_errors'] = {
                str(block_id): err.to_dict()
                for block_id, err in self.block_fabrication_errors.items()
            }
        
        if self.generation_params:
            result['generation_params'] = self.generation_params.to_dict()
        
        return result
    
    def get_block_error(self, block_id: int) -> BlockFabricationError:
        """Get fabrication error for specific block."""
        if block_id not in self.block_fabrication_errors:
            # Return zero error if not specified
            return BlockFabricationError(
                rotation_deg=0.0,
                translation_um=Point2D(0.0, 0.0)
            )
        return self.block_fabrication_errors[block_id]


# ============================================================================
# CAMERA LAYOUT (full access to ground truth)
# ============================================================================

@dataclass
class CameraLayout:
    """
    Layout for camera - has full access to design + ground truth.
    
    Use this for:
    - MockCamera rendering
    - Visualization
    - Validation (comparing measured vs. truth)
    
    Example:
        >>> camera_layout = CameraLayout.from_json_file("config/mock_layout.json")
        >>> camera = MockCamera(camera_layout, stage_ref=stage)
        >>> 
        >>> # Camera can access ground truth
        >>> rotation = camera_layout.ground_truth.rotation_deg
        >>> block_error = camera_layout.ground_truth.get_block_error(10)
    """
    design_name: str
    version: str
    coordinate_system: Dict[str, str]
    block_layout: BlockLayoutParams
    blocks: Dict[int, Block]
    ground_truth: GroundTruth
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'CameraLayout':
        """Load camera layout from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Layout file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraLayout':
        """Parse from JSON dictionary."""
        # Parse blocks (without fabrication_error field)
        blocks_data = data.get('blocks', {})
        blocks = {}
        for block_id_str, block_data in blocks_data.items():
            block_id = int(block_id_str)
            # Remove fabrication_error if present (it's in ground truth now)
            clean_block_data = {
                k: v for k, v in block_data.items()
                if k != 'fabrication_error'
            }
            blocks[block_id] = Block.from_dict(block_id, clean_block_data)
        
        # Parse ground truth (required for camera)
        if 'simulation_ground_truth' not in data:
            raise ValueError("CameraLayout requires 'simulation_ground_truth' in JSON")
        
        ground_truth = GroundTruth.from_dict(data['simulation_ground_truth'])
        
        # Metadata
        metadata = {
            k: v for k, v in data.items()
            if k not in ['design_name', 'version', 'coordinate_system',
                        'block_layout', 'blocks', 'simulation_ground_truth']
        }
        
        return cls(
            design_name=str(data['design_name']),
            version=str(data.get('version', '2.1')),
            coordinate_system=dict(data.get('coordinate_system', {})),
            block_layout=BlockLayoutParams.from_dict(data['block_layout']),
            blocks=blocks,
            ground_truth=ground_truth,
            metadata=metadata
        )
    
    def get_block(self, block_id: int) -> Block:
        """Get block by ID."""
        if block_id not in self.blocks:
            raise KeyError(
                f"Block {block_id} not found. Available: {sorted(self.blocks.keys())}"
            )
        return self.blocks[block_id]
    
    def list_blocks(self) -> List[int]:
        """Get sorted list of block IDs."""
        return sorted(self.blocks.keys())
    
    def __repr__(self) -> str:
        return (
            f"CameraLayout(name='{self.design_name}', "
            f"blocks={len(self.blocks)}, has_ground_truth=True)"
        )


# ============================================================================
# RUNTIME LAYOUT (measurement accumulation, no ground truth access)
# ============================================================================

@dataclass
class MeasuredTransform:
    """Measured rotation and translation."""
    rotation_deg: float
    translation_um: Point2D
    timestamp: datetime = field(default_factory=datetime.now)
    calibration_error_um: Optional[float] = None
    num_points: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rotation_deg': self.rotation_deg,
            'translation_um': self.translation_um.to_list(),
            'timestamp': self.timestamp.isoformat(),
            'calibration_error_um': self.calibration_error_um,
            'num_points': self.num_points
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeasuredTransform':
        return cls(
            rotation_deg=float(data['rotation_deg']),
            translation_um=Point2D.from_list(data['translation_um']),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            calibration_error_um=data.get('calibration_error_um'),
            num_points=data.get('num_points')
        )


@dataclass
class RuntimeLayout:
    """
    Layout for analysis - starts with design only, accumulates measurements.
    
    Does NOT have access to ground truth - must discover transformations
    through measurement.
    
    Use this for:
    - HierarchicalAlignment
    - CoordinateTransform
    - Analysis workflows
    
    Example:
        >>> runtime = RuntimeLayout.from_json_file("config/mock_layout.json")
        >>> alignment = HierarchicalAlignment(runtime)
        >>> 
        >>> # After Stage 1 calibration
        >>> runtime.set_global_calibration(rotation=2.94, translation=(50.2, 29.8))
        >>> 
        >>> # After Stage 2 calibration
        >>> runtime.set_block_calibration(10, rotation=0.17, translation=(0.58, -0.42))
        >>> 
        >>> # Save results
        >>> runtime.save_to_json("results/calibration.json")
    """
    design_name: str
    version: str
    coordinate_system: Dict[str, str]
    block_layout: BlockLayoutParams
    blocks: Dict[int, Block]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Measured values (filled during experiment)
    measured_global_transform: Optional[MeasuredTransform] = None
    measured_block_transforms: Dict[int, MeasuredTransform] = field(default_factory=dict)
    
    # Measurement log
    measurement_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # NEW: Manual Block 1 stage position (for initial alignment)
    block_1_stage_position_um: Optional[Tuple[float, float]] = None

    captured_fiducials: List[Dict[str, Any]] = field(default_factory=list)  # List of {block_id, corner, Y, Z}
        
    def set_block_1_position(self, y_um: float, z_um: float):
        """
        Set the stage position of Block 1 center.
        
        This is used to initialize the first search when no calibration exists.
        
        Args:
            y_um: Y stage position in micrometers
            z_um: Z stage position in micrometers
        """
        self.block_1_stage_position_um = (float(y_um), float(z_um))
        print(f"[RuntimeLayout] Block 1 position set to: Y={y_um:.3f}, Z={z_um:.3f} µm")
    
    def has_block_1_position(self) -> bool:
        """Check if Block 1 position has been set."""
        return self.block_1_stage_position_um is not None
    
    def get_block_1_position(self) -> Tuple[float, float]:
        """
        Get Block 1 stage position.
        
        Returns:
            (Y, Z) in micrometers
        
        Raises:
            RuntimeError: If position not set
        """
        if not self.has_block_1_position():
            raise RuntimeError(
                "Block 1 position not set. Use Layout Wizard to set initial position."
            )
        return self.block_1_stage_position_um


    def add_captured_fiducial(self, block_id: int, corner: str, Y: float, Z: float):
        """Add a manually captured fiducial position."""
        # Remove existing if duplicate
        self.captured_fiducials = [
            f for f in self.captured_fiducials 
            if not (f['block_id'] == block_id and f['corner'] == corner)
        ]
        
        self.captured_fiducials.append({
            'block_id': block_id,
            'corner': corner,
            'Y': Y,
            'Z': Z
        })
    
    def remove_captured_fiducial(self, block_id: int, corner: str):
        """Remove a captured fiducial."""
        self.captured_fiducials = [
            f for f in self.captured_fiducials 
            if not (f['block_id'] == block_id and f['corner'] == corner)
        ]
    
    def has_captured_fiducial(self, block_id: int, corner: str) -> bool:
        """Check if fiducial exists."""
        return any(
            f['block_id'] == block_id and f['corner'] == corner
            for f in self.captured_fiducials
        )
    
    def get_all_captured_fiducials(self) -> list:
        """Get all captured fiducials."""
        return self.captured_fiducials.copy()
    
    def clear_all_captured_fiducials(self):
        """Clear all captured fiducials."""
        self.captured_fiducials = []
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'RuntimeLayout':
        """
        Load runtime layout from JSON file.
        
        Loads ONLY design data, ignores simulation_ground_truth.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Layout file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    # UPDATE from_dict method - ADD THIS SECTION
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeLayout':
        """Parse from JSON dictionary."""
        # Parse blocks
        blocks_data = data.get('blocks', {})
        blocks = {}
        for block_id_str, block_data in blocks_data.items():
            block_id = int(block_id_str)
            clean_block_data = {
                k: v for k, v in block_data.items()
                if k != 'fabrication_error'
            }
            blocks[block_id] = Block.from_dict(block_id, clean_block_data)
        
        # Metadata
        metadata = {
            k: v for k, v in data.items()
            if k not in ['design_name', 'version', 'coordinate_system',
                        'block_layout', 'blocks', 'simulation_ground_truth',
                        'measured_calibration', 'block_1_stage_position_um']
        }
        
        runtime = cls(
            design_name=str(data['design_name']),
            version=str(data.get('version', '2.1')),
            coordinate_system=dict(data.get('coordinate_system', {})),
            block_layout=BlockLayoutParams.from_dict(data['block_layout']),
            blocks=blocks,
            metadata=metadata
        )
        
        # NEW: Load Block 1 position if present
        if 'block_1_stage_position_um' in data:
            pos = data['block_1_stage_position_um']
            runtime.block_1_stage_position_um = (float(pos[0]), float(pos[1]))
            print(f"[RuntimeLayout] Loaded Block 1 position: Y={pos[0]:.3f}, Z={pos[1]:.3f} µm")
        
        # Load measured calibration if present
        if 'measured_calibration' in data:
            runtime._load_measured_calibration(data['measured_calibration'])
        
        return runtime
    
    def _load_measured_calibration(self, data: Dict[str, Any]):
        """Load previously saved calibration results."""
        if 'global_transform' in data:
            self.measured_global_transform = MeasuredTransform.from_dict(
                data['global_transform']
            )
        
        if 'block_transforms' in data:
            self.measured_block_transforms = {
                int(block_id): MeasuredTransform.from_dict(trans_data)
                for block_id, trans_data in data['block_transforms'].items()
            }
        
        if 'measurement_log' in data:
            self.measurement_log = data['measurement_log']
    
    # ========================================================================
    # MEASUREMENT SETTERS
    # ========================================================================
    
    def set_global_calibration(self,
                              rotation: float,
                              translation: Tuple[float, float],
                              calibration_error: Optional[float] = None,
                              num_points: Optional[int] = None):
        """
        Store discovered global sample transformation (Stage 1 result).
        
        Args:
            rotation: Measured rotation angle in degrees
            translation: Measured (Y, Z) translation in µm
            calibration_error: Mean calibration error in µm
            num_points: Number of fiducials used
        """
        self.measured_global_transform = MeasuredTransform(
            rotation_deg=rotation,
            translation_um=Point2D.from_tuple(translation),
            calibration_error_um=calibration_error,
            num_points=num_points
        )
        
        self.measurement_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'global_calibration',
            'rotation_deg': rotation,
            'translation_um': list(translation),
            'calibration_error_um': calibration_error
        })
    
    def set_block_calibration(self,
                             block_id: int,
                             rotation: float,
                             translation: Tuple[float, float],
                             calibration_error: Optional[float] = None,
                             num_points: Optional[int] = None):
        """
        Store discovered block-specific transformation (Stage 2 result).
        
        Args:
            block_id: Block identifier
            rotation: Measured additional rotation in degrees
            translation: Measured additional (Y, Z) translation in µm
            calibration_error: Mean calibration error in µm
            num_points: Number of fiducials used
        """
        self.measured_block_transforms[block_id] = MeasuredTransform(
            rotation_deg=rotation,
            translation_um=Point2D.from_tuple(translation),
            calibration_error_um=calibration_error,
            num_points=num_points
        )
        
        self.measurement_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'block_calibration',
            'block_id': block_id,
            'rotation_deg': rotation,
            'translation_um': list(translation),
            'calibration_error_um': calibration_error
        })
    
    # ========================================================================
    # MEASUREMENT GETTERS
    # ========================================================================
    
    def is_globally_calibrated(self) -> bool:
        """Check if global calibration has been performed."""
        return self.measured_global_transform is not None
    
    def is_block_calibrated(self, block_id: int) -> bool:
        """Check if specific block has been calibrated."""
        return block_id in self.measured_block_transforms
    
    def get_global_transform(self) -> MeasuredTransform:
        """
        Get global transformation (raises if not calibrated).
        
        Raises:
            RuntimeError: If global calibration not yet performed
        """
        if not self.is_globally_calibrated():
            raise RuntimeError(
                "Global calibration not yet performed. "
                "Call set_global_calibration() first."
            )
        return self.measured_global_transform
    
    def get_block_transform(self, block_id: int) -> MeasuredTransform:
        """
        Get block transformation (raises if not calibrated).
        
        Raises:
            RuntimeError: If block not yet calibrated
        """
        if not self.is_block_calibrated(block_id):
            raise RuntimeError(
                f"Block {block_id} not yet calibrated. "
                f"Call set_block_calibration({block_id}, ...) first."
            )
        return self.measured_block_transforms[block_id]
    
    def get_calibrated_blocks(self) -> List[int]:
        """Get list of calibrated block IDs."""
        return sorted(self.measured_block_transforms.keys())
    
    # ========================================================================
    # BLOCK ACCESS (same as CameraLayout)
    # ========================================================================
    
    def get_block(self, block_id: int) -> Block:
        """Get block by ID."""
        if block_id not in self.blocks:
            raise KeyError(
                f"Block {block_id} not found. Available: {sorted(self.blocks.keys())}"
            )
        return self.blocks[block_id]
    
    def list_blocks(self) -> List[int]:
        """Get sorted list of block IDs."""
        return sorted(self.blocks.keys())
    
    def get_corner_blocks(self) -> List[int]:
        """Get corner block IDs for Stage 1 calibration."""
        blocks_per_row = self.block_layout.blocks_per_row
        num_rows = self.block_layout.num_rows
        
        corner_ids = []
        for block in self.blocks.values():
            is_corner = (
                (block.row == 0 or block.row == num_rows - 1) and
                (block.col == 0 or block.col == blocks_per_row - 1)
            )
            if is_corner:
                corner_ids.append(block.id)
        
        return sorted(corner_ids)
    
    # ========================================================================
    # SERIALIZATION
    # ========================================================================
        # UPDATE to_dict method - ADD THIS SECTION
    def to_dict(self, include_design: bool = True) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            'design_name': self.design_name,
            'version': self.version,
            'saved_timestamp': datetime.now().isoformat()
        }
        
        # NEW: Add Block 1 position if set
        if self.block_1_stage_position_um is not None:
            result['block_1_stage_position_um'] = list(self.block_1_stage_position_um)
        
        if include_design:
            result.update({
                'coordinate_system': self.coordinate_system,
                'block_layout': self.block_layout.to_dict(),
                'blocks': {
                    str(block_id): block.to_dict()
                    for block_id, block in self.blocks.items()
                }
            })
            result.update(self.metadata)
        
        # Measured calibration
        measured_cal = {}
        
        if self.measured_global_transform:
            measured_cal['global_transform'] = self.measured_global_transform.to_dict()
        
        if self.measured_block_transforms:
            measured_cal['block_transforms'] = {
                str(block_id): trans.to_dict()
                for block_id, trans in self.measured_block_transforms.items()
            }
        
        if self.measurement_log:
            measured_cal['measurement_log'] = self.measurement_log
        
        if measured_cal:
            result['measured_calibration'] = measured_cal
        
        return result
    
    # config/layout_models.py - UPDATE save_to_json method

    def save_to_json(self, filepath: str, include_design: bool = True, indent: int = 2):
        """
        Save runtime layout to JSON file.
        
        CRITICAL: When saving RuntimeLayout, we save ONLY measurements,
        NOT the full design (to avoid overwriting source).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # RuntimeLayout should ALWAYS save measurements only
        # Design stays in the source file
        data = {
            'design_name': self.design_name,
            'version': self.version,
            'saved_timestamp': datetime.now().isoformat(),
            'source_design_file': self.metadata.get('source_file', 'unknown')
        }
        
        # Block 1 position
        if self.block_1_stage_position_um is not None:
            data['block_1_stage_position_um'] = list(self.block_1_stage_position_um)
        
        # Measured calibration
        measured_cal = {}
        
        if self.measured_global_transform:
            measured_cal['global_transform'] = self.measured_global_transform.to_dict()
        
        if self.measured_block_transforms:
            measured_cal['block_transforms'] = {
                str(block_id): trans.to_dict()
                for block_id, trans in self.measured_block_transforms.items()
            }
        
        if self.measurement_log:
            measured_cal['measurement_log'] = self.measurement_log
        
        if measured_cal:
            data['measured_calibration'] = measured_cal
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        print(f"[RuntimeLayout] Saved runtime state to: {filepath}")
    
    def __repr__(self) -> str:
        calibrated = "yes" if self.is_globally_calibrated() else "no"
        num_calibrated_blocks = len(self.measured_block_transforms)
        return (
            f"RuntimeLayout(name='{self.design_name}', "
            f"blocks={len(self.blocks)}, "
            f"globally_calibrated={calibrated}, "
            f"calibrated_blocks={num_calibrated_blocks})"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_layout(filepath: str, for_camera: bool = False):
    """
    Convenience function to load the appropriate layout type.
    
    Args:
        filepath: Path to JSON layout file
        for_camera: If True, load CameraLayout (with ground truth).
                   If False, load RuntimeLayout (design only).
    
    Returns:
        CameraLayout or RuntimeLayout instance
    
    Example:
        >>> # For simulation/camera
        >>> camera_layout = load_layout("config/mock_layout.json", for_camera=True)
        >>> 
        >>> # For real measurements
        >>> runtime_layout = load_layout("config/mock_layout.json", for_camera=False)
    """
    if for_camera:
        return CameraLayout.from_json_file(filepath)
    else:
        return RuntimeLayout.from_json_file(filepath)


def create_empty_runtime_layout(
    design_name: str,
    block_size: float,
    block_spacing: float,
    blocks_per_row: int,
    num_rows: int
) -> RuntimeLayout:
    """
    Create a minimal RuntimeLayout programmatically (without JSON file).
    
    Useful for testing or creating layouts on-the-fly.
    
    Args:
        design_name: Name of the design
        block_size: Size of each block in µm
        block_spacing: Spacing between blocks in µm
        blocks_per_row: Number of blocks per row
        num_rows: Number of rows
    
    Returns:
        Empty RuntimeLayout with no blocks defined
    
    Example:
        >>> runtime = create_empty_runtime_layout(
        ...     design_name="test_chip",
        ...     block_size=5000.0,
        ...     block_spacing=1000.0,
        ...     blocks_per_row=4,
        ...     num_rows=3
        ... )
    """
    total_blocks = blocks_per_row * num_rows
    
    block_layout = BlockLayoutParams(
        block_size=block_size,
        block_spacing=block_spacing,
        blocks_per_row=blocks_per_row,
        num_rows=num_rows,
        total_blocks=total_blocks
    )
    
    return RuntimeLayout(
        design_name=design_name,
        version="2.1",
        coordinate_system={
            "origin": "stage_home",
            "axes": "Y_horizontal_Z_vertical"
        },
        block_layout=block_layout,
        blocks={},
        metadata={}
    )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_layout_consistency(layout) -> List[str]:
    """
    Validate layout for common issues.
    
    Args:
        layout: CameraLayout or RuntimeLayout instance
    
    Returns:
        List of warning/error messages (empty if all OK)
    
    Example:
        >>> layout = RuntimeLayout.from_json_file("config/layout.json")
        >>> issues = validate_layout_consistency(layout)
        >>> if issues:
        ...     for issue in issues:
        ...         print(f"WARNING: {issue}")
    """
    issues = []
    
    # Check block count matches layout params
    expected_blocks = layout.block_layout.total_blocks
    actual_blocks = len(layout.blocks)
    if actual_blocks != expected_blocks:
        issues.append(
            f"Block count mismatch: expected {expected_blocks}, "
            f"got {actual_blocks}"
        )
    
    # Check block positions are within expected grid
    for block in layout.blocks.values():
        if block.row >= layout.block_layout.num_rows:
            issues.append(
                f"Block {block.id} row {block.row} exceeds "
                f"num_rows {layout.block_layout.num_rows}"
            )
        if block.col >= layout.block_layout.blocks_per_row:
            issues.append(
                f"Block {block.id} col {block.col} exceeds "
                f"blocks_per_row {layout.block_layout.blocks_per_row}"
            )
    
    # Check for duplicate block positions
    positions = {}
    for block in layout.blocks.values():
        key = (block.row, block.col)
        if key in positions:
            issues.append(
                f"Duplicate position ({block.row}, {block.col}): "
                f"blocks {positions[key]} and {block.id}"
            )
        positions[key] = block.id
    
    # Check fiducials exist for all blocks
    for block in layout.blocks.values():
        if len(block.fiducials) < 4:
            issues.append(
                f"Block {block.id} has only {len(block.fiducials)} fiducials "
                f"(expected 4)"
            )
    
    # For CameraLayout, validate ground truth
    if isinstance(layout, CameraLayout):
        # Check all blocks have fabrication errors defined
        for block_id in layout.blocks.keys():
            if block_id not in layout.ground_truth.block_fabrication_errors:
                # This is OK - zero error assumed
                pass
    
    return issues


def compare_measured_vs_ground_truth(
    runtime: RuntimeLayout,
    camera: CameraLayout
) -> Dict[str, Any]:
    """
    Compare measured calibration results against ground truth.
    
    Useful for validating alignment algorithms in simulation.
    
    Args:
        runtime: RuntimeLayout with measured calibration
        camera: CameraLayout with ground truth
    
    Returns:
        Dictionary with comparison statistics
    
    Raises:
        ValueError: If designs don't match or runtime not calibrated
    
    Example:
        >>> runtime = RuntimeLayout.from_json_file("config/layout.json")
        >>> camera = CameraLayout.from_json_file("config/layout.json")
        >>> # ... run calibration on runtime ...
        >>> comparison = compare_measured_vs_ground_truth(runtime, camera)
        >>> print(f"Global rotation error: {comparison['global_rotation_error_deg']:.3f}°")
    """
    # Validate inputs
    if runtime.design_name != camera.design_name:
        raise ValueError(
            f"Design mismatch: runtime='{runtime.design_name}' vs "
            f"camera='{camera.design_name}'"
        )
    
    if not runtime.is_globally_calibrated():
        raise ValueError("Runtime layout not globally calibrated")
    
    # Compare global transform
    measured_global = runtime.get_global_transform()
    truth_global = camera.ground_truth
    
    rotation_error = abs(measured_global.rotation_deg - truth_global.rotation_deg)
    translation_error = (
        (measured_global.translation_um.u - truth_global.translation_um.u) ** 2 +
        (measured_global.translation_um.v - truth_global.translation_um.v) ** 2
    ) ** 0.5
    
    result = {
        'global_rotation_error_deg': rotation_error,
        'global_translation_error_um': translation_error,
        'measured_rotation_deg': measured_global.rotation_deg,
        'truth_rotation_deg': truth_global.rotation_deg,
        'measured_translation_um': measured_global.translation_um.to_list(),
        'truth_translation_um': truth_global.translation_um.to_list(),
        'block_comparisons': {}
    }
    
    # Compare per-block transforms
    for block_id in runtime.get_calibrated_blocks():
        if block_id not in camera.blocks:
            continue
        
        measured_block = runtime.get_block_transform(block_id)
        truth_block = camera.ground_truth.get_block_error(block_id)
        
        block_rot_error = abs(measured_block.rotation_deg - truth_block.rotation_deg)
        block_trans_error = (
            (measured_block.translation_um.u - truth_block.translation_um.u) ** 2 +
            (measured_block.translation_um.v - truth_block.translation_um.v) ** 2
        ) ** 0.5
        
        result['block_comparisons'][block_id] = {
            'rotation_error_deg': block_rot_error,
            'translation_error_um': block_trans_error,
            'measured_rotation_deg': measured_block.rotation_deg,
            'truth_rotation_deg': truth_block.rotation_deg,
            'measured_translation_um': measured_block.translation_um.to_list(),
            'truth_translation_um': truth_block.translation_um.to_list()
        }
    
    # Compute summary statistics
    if result['block_comparisons']:
        block_rot_errors = [
            comp['rotation_error_deg']
            for comp in result['block_comparisons'].values()
        ]
        block_trans_errors = [
            comp['translation_error_um']
            for comp in result['block_comparisons'].values()
        ]
        
        result['summary'] = {
            'num_blocks_compared': len(result['block_comparisons']),
            'mean_block_rotation_error_deg': sum(block_rot_errors) / len(block_rot_errors),
            'max_block_rotation_error_deg': max(block_rot_errors),
            'mean_block_translation_error_um': sum(block_trans_errors) / len(block_trans_errors),
            'max_block_translation_error_um': max(block_trans_errors)
        }
    
    return result