# coordinate_utils.py
"""
Coordinate transformation utilities for alignment system.

Coordinate Flow:
1. Local block coords (u, v in µm) - relative to block bottom-left
2. Global design coords (u, v in µm) - unrotated, origin at Block 1 bottom-left
3. Rotated coords (Y, Z in µm) - after rotation around origin
4. Stage coords (Y, Z in nm) - after rotation + translation
"""
import numpy as np
from typing import Tuple, Dict, Optional


class CoordinateConverter:
    """Handles all coordinate transformations for the alignment system."""
    
    def __init__(self, layout: Dict):
        """
        Initialize converter with layout configuration.
        
        Args:
            layout: Layout dict from layout_config_generator_v2
        """
        self.layout = layout
        self.blocks = layout['blocks']
        
        # Transformation parameters (will be calibrated from fiducials)
        self.rotation_deg = 0.0
        self.translation_nm = np.array([0.0, 0.0])
        self.is_calibrated = False
    
    def set_transformation(self, rotation_deg: float, translation_nm: Tuple[float, float]):
        """
        Set rotation and translation for coordinate transformation.
        
        Args:
            rotation_deg: Rotation angle in degrees
            translation_nm: (Y, Z) translation in nanometers
        """
        self.rotation_deg = rotation_deg
        self.translation_nm = np.array(translation_nm, dtype=float)
        self.is_calibrated = True
    
    def block_local_to_stage(
        self,
        block_id: int,
        u_local: float,
        v_local: float
    ) -> Tuple[int, int]:
        """
        Convert local block coordinates to stage coordinates.
        
        Args:
            block_id: Block ID (1-20)
            u_local: Local u coordinate in µm (relative to block bottom-left)
            v_local: Local v coordinate in µm (relative to block bottom-left)
        
        Returns:
            (Y, Z): Stage coordinates in nanometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transformation not calibrated. Call set_transformation() first.")
        
        # Step 1: Get block's design position (CENTER of block)
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        
        # Step 2: Get block size to convert center → bottom-left
        block_size = self.layout['block_layout']['block_size']  # 200 µm
        block_u_bottom_left = block_u_center - block_size / 2
        block_v_bottom_left = block_v_center - block_size / 2
        
        # Step 3: Convert to global design coordinates (unrotated)
        global_u = block_u_bottom_left + u_local  # µm
        global_v = block_v_bottom_left + v_local  # µm
        
        # Step 4: Convert to stage coordinates (rotate + translate)
        Y_nm, Z_nm = self.design_to_stage(global_u, global_v)
        
        return (int(round(Y_nm)), int(round(Z_nm)))
    
    def design_to_stage(self, u_um: float, v_um: float) -> Tuple[float, float]:
        """
        Convert global design coordinates to stage coordinates.
        
        Args:
            u_um: Global u coordinate in µm (unrotated)
            v_um: Global v coordinate in µm (unrotated)
        
        Returns:
            (Y, Z): Stage coordinates in nm
        """
        if not self.is_calibrated:
            raise RuntimeError("Transformation not calibrated.")
        
        # Convert to nm
        u_nm = u_um * 1000.0
        v_nm = v_um * 1000.0
        
        # Apply rotation
        angle_rad = np.radians(self.rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        Y_rotated = cos_a * u_nm - sin_a * v_nm
        Z_rotated = sin_a * u_nm + cos_a * v_nm
        
        # Apply translation
        Y_stage = Y_rotated + self.translation_nm[0]
        Z_stage = Z_rotated + self.translation_nm[1]
        
        return (Y_stage, Z_stage)
    
    def stage_to_design(self, Y_nm: float, Z_nm: float) -> Tuple[float, float]:
        """
        Convert stage coordinates to global design coordinates.
        
        Args:
            Y_nm: Stage Y coordinate in nm
            Z_nm: Stage Z coordinate in nm
        
        Returns:
            (u, v): Global design coordinates in µm (unrotated)
        """
        if not self.is_calibrated:
            raise RuntimeError("Transformation not calibrated.")
        
        # Remove translation
        Y_rotated = Y_nm - self.translation_nm[0]
        Z_rotated = Z_nm - self.translation_nm[1]
        
        # Inverse rotation
        angle_rad = np.radians(self.rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        u_nm = cos_a * Y_rotated + sin_a * Z_rotated
        v_nm = -sin_a * Y_rotated + cos_a * Z_rotated
        
        # Convert to µm
        u_um = u_nm / 1000.0
        v_um = v_nm / 1000.0
        
        return (u_um, v_um)
    
    def stage_to_block_local(
        self,
        Y_nm: float,
        Z_nm: float,
        block_id: int
    ) -> Tuple[float, float]:
        """
        Convert stage coordinates to local block coordinates.
        
        Args:
            Y_nm: Stage Y coordinate in nm
            Z_nm: Stage Z coordinate in nm
            block_id: Block ID
        
        Returns:
            (u_local, v_local): Local coordinates in µm (relative to block bottom-left)
        """
        # Convert to global design coords
        u_global, v_global = self.stage_to_design(Y_nm, Z_nm)
        
        # Get block's center position and size
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        block_size = self.layout['block_layout']['block_size']
        
        # Convert center → bottom-left
        block_u_bottom_left = block_u_center - block_size / 2
        block_v_bottom_left = block_v_center - block_size / 2
        
        # Subtract block bottom-left position to get local coords
        u_local = u_global - block_u_bottom_left
        v_local = v_global - block_v_bottom_left
        
        return (u_local, v_local)
    
    def get_fiducial_stage_position(
        self,
        block_id: int,
        corner: str
    ) -> Tuple[int, int]:
        """
        Get stage position of a fiducial marker.
        
        Args:
            block_id: Block ID
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Stage coordinates in nm
        """
        block = self.blocks[block_id]
        local_pos = block['fiducials'][corner]
        
        return self.block_local_to_stage(block_id, local_pos[0], local_pos[1])
    
    def get_grating_stage_position(
        self,
        block_id: int,
        waveguide: int,
        side: str
    ) -> Tuple[int, int]:
        """
        Get stage position of a grating coupler.
        
        Args:
            block_id: Block ID
            waveguide: Waveguide number
            side: 'left' or 'right'
        
        Returns:
            (Y, Z): Stage coordinates in nm
        """
        block = self.blocks[block_id]
        grating_id = f"wg{waveguide}_{side}"
        
        if grating_id not in block['gratings']:
            raise ValueError(f"Grating {grating_id} not found in block {block_id}")
        
        grating = block['gratings'][grating_id]
        local_pos = grating['position']
        
        return self.block_local_to_stage(block_id, local_pos[0], local_pos[1])


# Test/example usage
if __name__ == "__main__":
    from config.layout_config_generator_v2 import load_layout_config_v2
    
    print("Coordinate Converter Test")
    print("="*70)
    
    # Load layout
    layout = load_layout_config_v2("config/mock_layout.json")
    converter = CoordinateConverter(layout)
    
    # Set ground truth transformation (from simulation)
    ground_truth = layout['simulation_ground_truth']
    converter.set_transformation(
        ground_truth['rotation_deg'],
        tuple(ground_truth['translation_nm'])
    )
    
    print(f"Transformation set:")
    print(f"  Rotation: {converter.rotation_deg}°")
    print(f"  Translation: {converter.translation_nm} nm")
    
    # Test: Block 1 top-left fiducial
    print(f"\n{'='*70}")
    print("Test 1: Block 1 Top-Left Fiducial")
    print("="*70)
    
    block1 = layout['blocks'][1]
    tl_local = block1['fiducials']['top_left']
    print(f"  Local coords: {tl_local} µm")
    
    tl_stage = converter.get_fiducial_stage_position(1, 'top_left')
    print(f"  Stage coords: ({tl_stage[0]}, {tl_stage[1]}) nm")
    print(f"  Stage coords: ({tl_stage[0]/1000:.3f}, {tl_stage[1]/1000:.3f}) µm")
    
    # Test: Block 10 WG25 left grating
    print(f"\n{'='*70}")
    print("Test 2: Block 10 WG25 Left Grating")
    print("="*70)
    
    block10 = layout['blocks'][10]
    wg25_local = block10['gratings']['wg25_left']['position']
    print(f"  Local coords: {wg25_local} µm")
    print(f"  Block 10 design position: {block10['design_position']} µm")
    
    wg25_stage = converter.get_grating_stage_position(10, 25, 'left')
    print(f"  Stage coords: ({wg25_stage[0]}, {wg25_stage[1]}) nm")
    print(f"  Stage coords: ({wg25_stage[0]/1000:.3f}, {wg25_stage[1]/1000:.3f}) µm")
    
    # Test: Round-trip conversion
    print(f"\n{'='*70}")
    print("Test 3: Round-Trip Conversion")
    print("="*70)
    
    # Start with a point
    test_u, test_v = 350.0, 250.0  # µm in design space
    print(f"  Original design coords: ({test_u}, {test_v}) µm")
    
    # Convert to stage
    Y, Z = converter.design_to_stage(test_u, test_v)
    print(f"  Stage coords: ({Y:.1f}, {Z:.1f}) nm")
    
    # Convert back
    u_back, v_back = converter.stage_to_design(Y, Z)
    print(f"  Back to design: ({u_back:.6f}, {v_back:.6f}) µm")
    
    error = np.sqrt((u_back - test_u)**2 + (v_back - test_v)**2)
    print(f"  Round-trip error: {error:.9f} µm")
    
    print(f"\n✅ All coordinate conversion tests complete!")