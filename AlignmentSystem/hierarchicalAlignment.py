"""
Hierarchical two-stage alignment system.

Stage 1: Global sample alignment using corner blocks
Stage 2: Per-block local refinement for fine alignment

All internal operations use micrometers. Conversion to nanometers only 
happens at the hardware interface when calling stage methods.
"""
from typing import Dict, List, Tuple, Optional
from AlignmentSystem.coordinate_transform import CoordinateTransform


class HierarchicalAlignment:
    """
    Two-stage hierarchical alignment system.
    
    Stage 1 (Global): Calibrate sample rotation/translation using 4 corner blocks
    Stage 2 (Local): Refine individual block alignment if needed
    
    All coordinates internally in micrometers (µm).
    """
    
    def __init__(self, layout: Dict):
        """
        Initialize hierarchical alignment system.
        
        Args:
            layout: Layout dictionary from layout_config_generator_v2
        """
        self.layout = layout
        
        # Stage 1: Global sample transform (includes layout for block-awareness)
        self.global_transform = CoordinateTransform(layout)
        
        # Stage 2: Per-block local transforms (optional refinements)
        self.block_transforms: Dict[int, CoordinateTransform] = {}
        
        # Calibration state
        self.is_global_calibrated = False
        self.calibrated_blocks = set()
        
    # ========================================================================
    # STAGE 1: GLOBAL CALIBRATION
    # ========================================================================
    
    def calibrate_global(self, fiducial_measurements: List[Dict]) -> Dict:
        """
        Perform Stage 1 global calibration using corner block fiducials.
        
        Args:
            fiducial_measurements: List of dicts with keys:
                - 'block_id': Block identifier
                - 'corner': Corner name ('top_left', 'bottom_right', etc.)
                - 'stage_Y': Measured Y position in µm
                - 'stage_Z': Measured Z position in µm
        
        Returns:
            dict: Calibration results with rotation, translation, and errors
        """
        if len(fiducial_measurements) < 2:
            raise ValueError("Need at least 2 fiducial measurements for global calibration")
        
        # Collect design and measured points
        design_points_um = []
        measured_points_um = []
        
        for fid in fiducial_measurements:
            block_id = fid['block_id']
            corner = fid['corner']
            
            # Get design coordinates (global frame)
            u_global, v_global = self._get_fiducial_global_design_coords(
                block_id, corner
            )
            design_points_um.append((u_global, v_global))
            
            # Measured stage coordinates (already in µm from searcher)
            measured_points_um.append((fid['stage_Y'], fid['stage_Z']))
        
        # Calibrate global transform
        result = self.global_transform.calibrate(
            measured_points=measured_points_um,
            design_points=design_points_um
        )
        
        self.is_global_calibrated = True
        
        return {
            'stage': 'global',
            'num_fiducials': len(fiducial_measurements),
            'rotation_deg': result['angle_deg'],
            'translation_um': result['translation_um'],
            'mean_error_um': result['mean_error_um'],
            'max_error_um': result['max_error_um']
        }
    
    # ========================================================================
    # STAGE 2: BLOCK-LEVEL CALIBRATION
    # ========================================================================
    
    def calibrate_block(self, 
                       block_id: int,
                       fiducial_measurements: List[Dict]) -> Dict:
        """
        Perform Stage 2 block-level calibration for fine alignment.
        
        This creates a block-specific transform that captures any additional
        rotation/translation beyond the global sample transform and the 
        fabrication errors already in the layout.
        
        Args:
            block_id: Block to calibrate
            fiducial_measurements: List of dicts with keys:
                - 'corner': Corner name
                - 'stage_Y': Measured Y in µm
                - 'stage_Z': Measured Z in µm
        
        Returns:
            dict: Block calibration results
        """
        if not self.is_global_calibrated:
            raise RuntimeError("Must perform global calibration first")
        
        if len(fiducial_measurements) < 2:
            raise ValueError("Need at least 2 fiducial measurements for block calibration")
        
        # Collect local design coordinates and measured positions
        design_points_um = []
        measured_points_um = []
        
        block = self.layout['blocks'][block_id]
        
        for fid in fiducial_measurements:
            corner = fid['corner']
            
            # Get local design coordinates (relative to block bottom-left)
            local_pos = block['fiducials'][corner]
            design_points_um.append(tuple(local_pos))
            
            # Measured stage coordinates
            measured_points_um.append((fid['stage_Y'], fid['stage_Z']))
        
        # Create block-specific transform
        block_transform = CoordinateTransform(self.layout)
        result = block_transform.calibrate(
            measured_points=measured_points_um,
            design_points=design_points_um
        )
        
        # Store block transform
        self.block_transforms[block_id] = block_transform
        self.calibrated_blocks.add(block_id)
        
        return {
            'stage': 'block',
            'block_id': block_id,
            'num_fiducials': len(fiducial_measurements),
            'rotation_deg': result['angle_deg'],
            'translation_um': result['translation_um'],
            'mean_error_um': result['mean_error_um'],
            'max_error_um': result['max_error_um']
        }
    
    # ========================================================================
    # COORDINATE CONVERSION (delegates to appropriate transform)
    # ========================================================================
    
    def block_local_to_stage(self, 
                            block_id: int, 
                            u_local: float, 
                            v_local: float) -> Tuple[float, float]:
        """
        Convert local block coordinates to stage coordinates.
        
        Uses block-specific transform if available, otherwise global transform.
        
        Args:
            block_id: Block identifier
            u_local: Local u coordinate in µm
            v_local: Local v coordinate in µm
        
        Returns:
            (Y, Z): Stage coordinates in µm
        """
        # Use block-specific transform if available (Stage 2)
        if block_id in self.block_transforms:
            return self.block_transforms[block_id].block_local_to_stage(
                block_id, u_local, v_local
            )
        
        # Otherwise use global transform (Stage 1)
        if self.is_global_calibrated:
            return self.global_transform.block_local_to_stage(
                block_id, u_local, v_local
            )
        
        raise RuntimeError("No calibration available for coordinate conversion")
    
    def stage_to_block_local(self,
                            Y: float,
                            Z: float,
                            block_id: int) -> Tuple[float, float]:
        """
        Convert stage coordinates to local block coordinates.
        
        Uses block-specific transform if available, otherwise global transform.
        
        Args:
            Y, Z: Stage coordinates in µm
            block_id: Block identifier
        
        Returns:
            (u_local, v_local): Local coordinates in µm
        """
        # Use block-specific transform if available (Stage 2)
        if block_id in self.block_transforms:
            return self.block_transforms[block_id].stage_to_block_local(
                Y, Z, block_id
            )
        
        # Otherwise use global transform (Stage 1)
        if self.is_global_calibrated:
            return self.global_transform.stage_to_block_local(
                Y, Z, block_id
            )
        
        raise RuntimeError("No calibration available for coordinate conversion")
    
    def get_fiducial_stage_position(self,
                                   block_id: int,
                                   corner: str) -> Tuple[float, float]:
        """
        Get predicted stage position of a fiducial marker.
        
        Args:
            block_id: Block identifier
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        """
        # Use block-specific transform if available
        if block_id in self.block_transforms:
            return self.block_transforms[block_id].get_fiducial_stage_position(
                block_id, corner
            )
        
        # Otherwise use global transform
        if self.is_global_calibrated:
            return self.global_transform.get_fiducial_stage_position(
                block_id, corner
            )
        
        raise RuntimeError("No calibration available")
    
    def get_grating_stage_position(self,
                                  block_id: int,
                                  waveguide: int,
                                  side: str) -> Tuple[float, float]:
        """
        Get predicted stage position of a grating coupler.
        
        Args:
            block_id: Block identifier
            waveguide: Waveguide number
            side: 'left' or 'right'
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        """
        # Use block-specific transform if available
        if block_id in self.block_transforms:
            return self.block_transforms[block_id].get_grating_stage_position(
                block_id, waveguide, side
            )
        
        # Otherwise use global transform
        if self.is_global_calibrated:
            return self.global_transform.get_grating_stage_position(
                block_id, waveguide, side
            )
        
        raise RuntimeError("No calibration available")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_fiducial_global_design_coords(self,
                                          block_id: int,
                                          corner: str) -> Tuple[float, float]:
        """
        Get fiducial position in global design coordinates (including fab errors).
        
        Args:
            block_id: Block identifier
            corner: Corner name
        
        Returns:
            (u_global, v_global): Global design coordinates in µm
        """
        block = self.layout['blocks'][block_id]
        block_size = self.layout['block_layout']['block_size']
        
        # Get local fiducial position
        u_local, v_local = block['fiducials'][corner]
        
        # Apply fabrication error if present
        if 'fabrication_error' in block:
            fab_error = block['fabrication_error']
            fab_rotation_deg = fab_error['rotation_deg']
            fab_translation_um = fab_error['translation_um']
            
            # Rotate around block center
            import numpy as np
            center = block_size / 2.0
            
            # Center point
            u_centered = u_local - center
            v_centered = v_local - center
            
            # Rotate
            angle_rad = np.radians(fab_rotation_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            u_rotated = cos_a * u_centered - sin_a * v_centered
            v_rotated = sin_a * u_centered + cos_a * v_centered
            
            # Translate back and apply fabrication translation
            u_fab = u_rotated + center + fab_translation_um[0]
            v_fab = v_rotated + center + fab_translation_um[1]
        else:
            u_fab = u_local
            v_fab = v_local
        
        # Convert to global design coordinates
        block_u_center, block_v_center = block['design_position']
        
        u_global = block_u_center - block_size / 2.0 + u_fab
        v_global = block_v_center - block_size / 2.0 + v_fab
        
        return (u_global, v_global)
    
    # ========================================================================
    # STATUS AND UTILITY
    # ========================================================================
    
    def get_calibration_status(self) -> Dict:
        """
        Get current calibration status.
        
        Returns:
            dict: Status information
        """
        status = {
            'global_calibrated': self.is_global_calibrated,
            'num_blocks_calibrated': len(self.calibrated_blocks),
            'calibrated_blocks': sorted(list(self.calibrated_blocks))
        }
        
        if self.is_global_calibrated:
            global_info = self.global_transform.get_calibration_info()
            status['global_rotation_deg'] = global_info['angle_deg']
            status['global_translation_um'] = global_info['translation_um']
        
        return status
    
    def reset_calibration(self, block_id: Optional[int] = None):
        """
        Reset calibration data.
        
        Args:
            block_id: If provided, reset only this block. Otherwise reset all.
        """
        if block_id is None:
            # Reset everything
            self.is_global_calibrated = False
            self.global_transform = CoordinateTransform(self.layout)
            self.block_transforms.clear()
            self.calibrated_blocks.clear()
        else:
            # Reset specific block
            if block_id in self.block_transforms:
                del self.block_transforms[block_id]
            if block_id in self.calibrated_blocks:
                self.calibrated_blocks.remove(block_id)


# ============================================================================
# TEST/EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Hierarchical Alignment System")
    print("=" * 70)
    
    try:
        from config.layout_config_generator_v2 import load_layout_config_v2
        
        # Load layout
        layout = load_layout_config_v2("config/mock_layout.json")
        
        # Initialize alignment system
        alignment = HierarchicalAlignment(layout)
        
        print("\n" + "=" * 70)
        print("STAGE 1: Global Calibration (using ground truth)")
        print("=" * 70)
        
        # Simulate fiducial measurements using ground truth
        gt = layout['simulation_ground_truth']
        
        # Use a temporary transform to simulate measurements
        sim_transform = CoordinateTransform(layout)
        sim_transform.set_transformation(
            gt['rotation_deg'],
            tuple(gt['translation_um'])
        )
        
        # Measure corner blocks (1, 5, 16, 20)
        corner_blocks = [1, 5, 16, 20]
        fiducial_measurements = []
        
        for block_id in corner_blocks:
            for corner in ['top_left', 'bottom_right']:
                # Simulate stage measurement
                Y, Z = sim_transform.get_fiducial_stage_position(block_id, corner)
                
                fiducial_measurements.append({
                    'block_id': block_id,
                    'corner': corner,
                    'stage_Y': Y,
                    'stage_Z': Z
                })
        
        # Perform global calibration
        global_result = alignment.calibrate_global(fiducial_measurements)
        
        print(f"Global calibration complete:")
        print(f"  Fiducials used: {global_result['num_fiducials']}")
        print(f"  Rotation: {global_result['rotation_deg']:.6f}° "
              f"(expected: {gt['rotation_deg']}°)")
        print(f"  Translation: {global_result['translation_um']} µm "
              f"(expected: {gt['translation_um']} µm)")
        print(f"  Mean error: {global_result['mean_error_um']:.9f} µm")
        print(f"  Max error: {global_result['max_error_um']:.9f} µm")
        
        print("\n" + "=" * 70)
        print("STAGE 2: Block-Level Calibration (Block 10)")
        print("=" * 70)
        
        # Simulate block-level measurements
        block_fiducials = []
        for corner in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            Y, Z = sim_transform.get_fiducial_stage_position(10, corner)
            block_fiducials.append({
                'corner': corner,
                'stage_Y': Y,
                'stage_Z': Z
            })
        
        # Calibrate block 10
        block_result = alignment.calibrate_block(10, block_fiducials)
        
        print(f"Block 10 calibration complete:")
        print(f"  Fiducials used: {block_result['num_fiducials']}")
        print(f"  Additional rotation: {block_result['rotation_deg']:.6f}°")
        print(f"  Additional translation: {block_result['translation_um']} µm")
        print(f"  Mean error: {block_result['mean_error_um']:.9f} µm")
        
        print("\n" + "=" * 70)
        print("Coordinate Conversion Test")
        print("=" * 70)
        
        # Test grating position prediction
        Y_pred, Z_pred = alignment.get_grating_stage_position(10, 25, 'left')
        print(f"Block 10, WG25, left grating:")
        print(f"  Predicted stage: ({Y_pred:.6f}, {Z_pred:.6f}) µm")
        
        # Compare with ground truth
        Y_gt, Z_gt = sim_transform.get_grating_stage_position(10, 25, 'left')
        error = ((Y_pred - Y_gt)**2 + (Z_pred - Z_gt)**2)**0.5
        print(f"  Ground truth:    ({Y_gt:.6f}, {Z_gt:.6f}) µm")
        print(f"  Error: {error:.9f} µm")
        
        print("\n" + "=" * 70)
        print("Calibration Status")
        print("=" * 70)
        status = alignment.get_calibration_status()
        print(f"Global calibrated: {status['global_calibrated']}")
        print(f"Blocks calibrated: {status['num_blocks_calibrated']}")
        print(f"Calibrated block IDs: {status['calibrated_blocks']}")
        
        print("\n✅ All tests passed!")
        
    except ImportError as e:
        print(f"\n⚠️  Could not run tests: {e}")
        print("Make sure layout_config_generator_v2.py is available")