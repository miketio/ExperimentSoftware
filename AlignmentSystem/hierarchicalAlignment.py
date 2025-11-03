#!/usr/bin/env python3
# hierarchical_alignment.py
"""
HierarchicalAlignment: Two-stage coordinate transformation system.

Stage 1: Global sample rotation + translation
  - Uses fiducials from multiple blocks (e.g., corner blocks 1, 5, 16, 20)
  - Determines overall sample orientation on stage
  
Stage 2: Per-block local rotation + translation
  - Uses fiducials within a single block
  - Corrects for block-specific fabrication errors
  
Together, these provide accurate coordinate transformation from design to stage coordinates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from AlignmentSystem.coordinate_transform import CoordinateTransform


class HierarchicalAlignment:
    """
    Two-stage coordinate transformation system.
    
    Handles both global sample alignment and per-block local corrections.
    """
    
    def __init__(self, layout: Dict):
        """
        Initialize hierarchical alignment system.
        
        Args:
            layout: Layout configuration dict (from layout_config_generator_v2)
        """
        self.layout = layout
        self.block_size = layout['block_layout']['block_size']  # µm
        
        # Stage 1: Global sample transform
        self.global_transform = CoordinateTransform()
        self.is_global_calibrated = False
        self.global_calibration_data = {
            'design_points': [],
            'measured_points': [],
            'residuals': [],
            'rotation_deg': None,
            'translation_nm': None,
            'mean_error_nm': None,
            'max_error_nm': None
        }
        
        # Stage 2: Per-block transforms
        self.block_transforms = {}  # Dict[block_id, CoordinateTransform]
        self.calibrated_blocks = set()
        self.block_calibration_data = {}  # Dict[block_id, calibration_data]
    
    # =========================================================================
    # STAGE 1: GLOBAL SAMPLE ALIGNMENT
    # =========================================================================
    
    def calibrate_global(self, measured_fiducials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calibrate global sample rotation and translation.
        
        Uses fiducials from multiple blocks to determine overall sample orientation.
        
        Args:
            measured_fiducials: List of dicts with keys:
                - block_id: int
                - corner: str (e.g., 'top_left')
                - stage_Y: float (nm)
                - stage_Z: float (nm)
        
        Returns:
            dict with calibration results:
                - method: str
                - rotation_deg: float
                - translation_nm: tuple (Y, Z)
                - mean_error_nm: float
                - max_error_nm: float
                - residuals: list of residual magnitudes
        """
        print(f"\n{'='*70}")
        print("GLOBAL CALIBRATION (Stage 1)")
        print('='*70)
        print(f"Using {len(measured_fiducials)} fiducials from multiple blocks")
        
        # Extract design and measured points
        design_points_nm = []
        measured_points_nm = []
        
        for fid in measured_fiducials:
            block_id = fid['block_id']
            corner = fid['corner']
            
            # Get fiducial position in global design coordinates
            u_global, v_global = self._get_fiducial_global_design_coords(block_id, corner)
            design_points_nm.append((u_global * 1000.0, v_global * 1000.0))  # µm to nm
            
            # Measured stage position
            measured_points_nm.append((fid['stage_Y'], fid['stage_Z']))
            
            print(f"  Block {block_id:2d} {corner:12s}: design=({u_global:7.1f}, {v_global:7.1f}) µm, "
                  f"measured=({fid['stage_Y']:8.0f}, {fid['stage_Z']:8.0f}) nm")
        
        # Perform calibration
        result = self.global_transform.calibrate(
            measured_points=measured_points_nm,
            design_points=design_points_nm
        )
        
        # Store calibration data
        self.is_global_calibrated = True
        self.global_calibration_data = {
            'design_points': design_points_nm,
            'measured_points': measured_points_nm,
            'residuals': result.get('residuals', []),
            'rotation_deg': result['angle_deg'],
            'translation_nm': result['translation_nm'],
            'mean_error_nm': result.get('mean_error_nm', 0.0),
            'max_error_nm': result.get('max_error_nm', 0.0),
            'method': result['method']
        }
        
        print(f"\n✅ GLOBAL CALIBRATION COMPLETE")
        print(f"   Method: {result['method']}")
        print(f"   Rotation: {result['angle_deg']:.4f}°")
        print(f"   Translation: ({result['translation_nm'][0]:.1f}, {result['translation_nm'][1]:.1f}) nm")
        print(f"   Mean error: {result.get('mean_error_nm', 0.0):.3f} nm")
        print(f"   Max error: {result.get('max_error_nm', 0.0):.3f} nm")
        
        return result
    
    def get_global_calibration_result(self) -> Dict[str, Any]:
        """Return global calibration data."""
        return self.global_calibration_data.copy()
    
    # =========================================================================
    # STAGE 2: PER-BLOCK LOCAL ALIGNMENT
    # =========================================================================
    
    def calibrate_block(self, block_id: int, measured_fiducials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calibrate block-specific rotation and translation.
        
        Uses fiducials within a single block to correct for fabrication errors.
        
        Args:
            block_id: Which block to calibrate
            measured_fiducials: List of dicts with keys:
                - corner: str
                - stage_Y: float (nm)
                - stage_Z: float (nm)
        
        Returns:
            dict with calibration results
        """
        if not self.is_global_calibrated:
            raise RuntimeError("Must calibrate global transform before calibrating individual blocks")
        
        print(f"\n{'='*70}")
        print(f"BLOCK {block_id} CALIBRATION (Stage 2)")
        print('='*70)
        print(f"Using {len(measured_fiducials)} fiducials within Block {block_id}")
        
        # Extract design and measured points
        design_points_nm = []
        measured_points_nm = []
        
        for fid in measured_fiducials:
            corner = fid['corner']
            
            # Get fiducial position in global design coordinates
            u_global, v_global = self._get_fiducial_global_design_coords(block_id, corner)
            design_points_nm.append((u_global * 1000.0, v_global * 1000.0))
            
            # Measured stage position
            measured_points_nm.append((fid['stage_Y'], fid['stage_Z']))
            
            print(f"  {corner:12s}: design=({u_global:7.1f}, {v_global:7.1f}) µm, "
                  f"measured=({fid['stage_Y']:8.0f}, {fid['stage_Z']:8.0f}) nm")
        
        # Create block-specific transform
        block_transform = CoordinateTransform()
        result = block_transform.calibrate(
            measured_points=measured_points_nm,
            design_points=design_points_nm
        )
        
        # Store transform and calibration data
        self.block_transforms[block_id] = block_transform
        self.calibrated_blocks.add(block_id)
        
        self.block_calibration_data[block_id] = {
            'design_points': design_points_nm,
            'measured_points': measured_points_nm,
            'residuals': result.get('residuals', []),
            'rotation_deg': result['angle_deg'],
            'translation_nm': result['translation_nm'],
            'mean_error_nm': result.get('mean_error_nm', 0.0),
            'max_error_nm': result.get('max_error_nm', 0.0),
            'method': result['method']
        }
        
        print(f"\n✅ BLOCK {block_id} CALIBRATION COMPLETE")
        print(f"   Method: {result['method']}")
        print(f"   Rotation: {result['angle_deg']:.4f}°")
        print(f"   Translation: ({result['translation_nm'][0]:.1f}, {result['translation_nm'][1]:.1f}) nm")
        print(f"   Mean error: {result.get('mean_error_nm', 0.0):.3f} nm")
        print(f"   Max error: {result.get('max_error_nm', 0.0):.3f} nm")
        
        return result
    
    def get_block_calibration_result(self, block_id: int) -> Optional[Dict[str, Any]]:
        """Return block calibration data."""
        return self.block_calibration_data.get(block_id, None)
    
    def is_block_calibrated(self, block_id: int) -> bool:
        """Check if block has been calibrated."""
        return block_id in self.calibrated_blocks
    
    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================
    
    def block_local_to_stage(self, block_id: int, u_local: float, v_local: float) -> Tuple[float, float]:
        """
        Convert block-local coordinates to stage coordinates.
        
        Uses 2-stage transform:
        1. Convert local to global design coords (perfect grid)
        2. Apply block-specific transform if available
        3. Apply global transform
        
        Args:
            block_id: Block identifier
            u_local: Local u coordinate (µm, relative to block bottom-left)
            v_local: Local v coordinate (µm, relative to block bottom-left)
        
        Returns:
            (Y_stage, Z_stage) in nm
        """
        # Step 1: Local to global design (perfect grid assumption)
        u_global, v_global = self._local_to_global_design(block_id, u_local, v_local)
        
        # Step 2: Apply block-specific transform if available
        if block_id in self.block_transforms:
            Y_stage, Z_stage = self.block_transforms[block_id].design_to_stage(
                u_global, v_global
            )
        else:
            # No block calibration - use global only
            if self.is_global_calibrated:
                Y_stage, Z_stage = self.global_transform.design_to_stage(
                    u_global, v_global
                )
            else:
                # No calibration at all - return design coords
                Y_stage, Z_stage = u_global, v_global
        
        return Y_stage, Z_stage
    
    def predict_block_center(self, block_id: int) -> Tuple[float, float]:
        """
        Use global calibration to predict where block center should be.
        Useful for Stage 2 search initialization.
        
        Args:
            block_id: Block identifier
        
        Returns:
            (Y_stage, Z_stage) predicted center position in nm
        """
        if not self.is_global_calibrated:
            # Fallback to design position
            block_center = self.layout['blocks'][block_id]['design_position']
            return block_center[0], block_center[1]
        
        # Use global transform to predict block center
        block_center = self.layout['blocks'][block_id]['design_position']
        Y_stage, Z_stage = self.global_transform.design_to_stage(
            block_center[0],
            block_center[1]
        )
        
        return Y_stage, Z_stage
    
    def predict_fiducial_position(self, block_id: int, corner: str) -> Tuple[float, float]:
        """
        Predict stage position of a fiducial using available calibrations.
        
        Uses block-specific calibration if available, else falls back to global.
        
        Args:
            block_id: Block identifier
            corner: Corner name (e.g., 'top_left')
        
        Returns:
            (Y_stage, Z_stage) predicted position in nm
        """
        # Get fiducial local coordinates
        fiducial_local = self.layout['blocks'][block_id]['fiducials'][corner]
        
        # Use block_local_to_stage (automatically chooses best available transform)
        return self.block_local_to_stage(block_id, fiducial_local[0], fiducial_local[1])
    
    # =========================================================================
    # VALIDATION (for simulation/testing)
    # =========================================================================
    
    def validate_global_calibration(self, ground_truth_rotation: float, 
                                    ground_truth_translation: Tuple[float, float]) -> Dict[str, float]:
        """
        Compare global calibration to ground truth (simulation only).
        
        Args:
            ground_truth_rotation: True rotation in degrees
            ground_truth_translation: True translation (Y, Z) in nm
        
        Returns:
            dict with error metrics
        """
        if not self.is_global_calibrated:
            raise RuntimeError("Global calibration not yet performed")
        
        cal_data = self.global_calibration_data
        
        rotation_error = abs(cal_data['rotation_deg'] - ground_truth_rotation)
        translation_error = np.hypot(
            cal_data['translation_nm'][0] - ground_truth_translation[0],
            cal_data['translation_nm'][1] - ground_truth_translation[1]
        )
        
        return {
            'rotation_error_deg': rotation_error,
            'translation_error_nm': translation_error,
            'mean_residual_nm': cal_data['mean_error_nm'],
            'max_residual_nm': cal_data['max_error_nm']
        }
    
    def validate_block_calibration(self, block_id: int, 
                                   ground_truth_rotation: float,
                                   ground_truth_translation: Tuple[float, float]) -> Dict[str, float]:
        """
        Compare block calibration to ground truth (simulation only).
        
        Args:
            block_id: Block identifier
            ground_truth_rotation: True block rotation in degrees
            ground_truth_translation: True block translation (Y, Z) in nm
        
        Returns:
            dict with error metrics
        """
        if block_id not in self.calibrated_blocks:
            raise RuntimeError(f"Block {block_id} not yet calibrated")
        
        cal_data = self.block_calibration_data[block_id]
        
        rotation_error = abs(cal_data['rotation_deg'] - ground_truth_rotation)
        translation_error = np.hypot(
            cal_data['translation_nm'][0] - ground_truth_translation[0],
            cal_data['translation_nm'][1] - ground_truth_translation[1]
        )
        
        return {
            'rotation_error_deg': rotation_error,
            'translation_error_nm': translation_error,
            'mean_residual_nm': cal_data['mean_error_nm'],
            'max_residual_nm': cal_data['max_error_nm']
        }
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _local_to_global_design(self, block_id: int, u_local: float, v_local: float) -> Tuple[float, float]:
        """
        Convert block-local coordinates to global design coordinates.
        
        Assumes perfect grid (no block-specific errors).
        
        Args:
            block_id: Block identifier
            u_local: Local u (µm, relative to block bottom-left)
            v_local: Local v (µm, relative to block bottom-left)
        
        Returns:
            (u_global, v_global) in µm
        """
        block_center = self.layout['blocks'][block_id]['design_position']
        
        # Local coords are relative to block bottom-left
        # Block bottom-left = center - block_size/2
        u_global = block_center[0] + (u_local - self.block_size / 2.0)
        v_global = block_center[1] + (v_local - self.block_size / 2.0)
        
        return u_global, v_global
    
    def _get_fiducial_global_design_coords(self, block_id: int, corner: str) -> Tuple[float, float]:
        """
        Get fiducial position in global design coordinates.
        
        Args:
            block_id: Block identifier
            corner: Corner name (e.g., 'top_left')
        
        Returns:
            (u_global, v_global) in µm
        """
        fiducial_local = self.layout['blocks'][block_id]['fiducials'][corner]
        return self._local_to_global_design(block_id, fiducial_local[0], fiducial_local[1])
    
    def get_waveguide_position(self, block_id: int, waveguide_number: int, 
                              position: str = 'center') -> Tuple[float, float]:
        """
        Get stage position of a waveguide.
        
        Args:
            block_id: Block identifier
            waveguide_number: Waveguide number (e.g., 25)
            position: 'center', 'left_grating', or 'right_grating'
        
        Returns:
            (Y_stage, Z_stage) in nm
        """
        waveguides = self.layout['blocks'][block_id]['waveguides']
        wg_key = f"wg{waveguide_number}"
        
        if wg_key not in waveguides:
            raise ValueError(f"Waveguide {waveguide_number} not found in Block {block_id}")
        
        wg = waveguides[wg_key]
        
        if position == 'center':
            # Center of waveguide
            u_local = (wg['u_start'] + wg['u_end']) / 2.0
            v_local = wg['v_center']
        elif position == 'left_grating':
            # Left grating position
            grating_key = f"wg{waveguide_number}_left"
            if grating_key not in self.layout['blocks'][block_id]['gratings']:
                raise ValueError(f"Left grating for WG{waveguide_number} not found")
            grating = self.layout['blocks'][block_id]['gratings'][grating_key]
            u_local, v_local = grating['position']
        elif position == 'right_grating':
            # Right grating position
            grating_key = f"wg{waveguide_number}_right"
            if grating_key not in self.layout['blocks'][block_id]['gratings']:
                raise ValueError(f"Right grating for WG{waveguide_number} not found")
            grating = self.layout['blocks'][block_id]['gratings'][grating_key]
            u_local, v_local = grating['position']
        else:
            raise ValueError(f"Invalid position: {position}")
        
        return self.block_local_to_stage(block_id, u_local, v_local)
    
    # =========================================================================
    # STATUS / INFO
    # =========================================================================
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get overall calibration status.
        
        Returns:
            dict with status information
        """
        return {
            'global_calibrated': self.is_global_calibrated,
            'num_blocks_calibrated': len(self.calibrated_blocks),
            'calibrated_blocks': list(self.calibrated_blocks),
            'total_blocks': len(self.layout['blocks'])
        }
    
    def print_status(self):
        """Print calibration status."""
        status = self.get_calibration_status()
        
        print(f"\n{'='*70}")
        print("HIERARCHICAL ALIGNMENT STATUS")
        print('='*70)
        print(f"Global calibration: {'✅ YES' if status['global_calibrated'] else '❌ NO'}")
        
        if status['global_calibrated']:
            cal = self.global_calibration_data
            print(f"  Rotation: {cal['rotation_deg']:.4f}°")
            print(f"  Translation: ({cal['translation_nm'][0]:.1f}, {cal['translation_nm'][1]:.1f}) nm")
            print(f"  Mean error: {cal['mean_error_nm']:.3f} nm")
        
        print(f"\nBlock calibrations: {status['num_blocks_calibrated']}/{status['total_blocks']}")
        
        if status['num_blocks_calibrated'] > 0:
            print(f"  Calibrated blocks: {sorted(status['calibrated_blocks'])}")
        
        print('='*70)

if __name__ == "__main__":
    """
    Demonstration of hierarchical alignment system.
    
    Shows both Stage 1 (global) and Stage 2 (per-block) calibration workflow.
    """
    import sys
    from pathlib import Path
    
    # Import required modules
    try:
        from config.layout_config_generator_v2 import load_layout_config_v2
        from HardwareControl.CameraControl.mock_camera import MockCamera
        from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
        from AlignmentSystem.cv_tools import VisionTools
        from AlignmentSystem.coordinate_transform import CoordinateTransform
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("HIERARCHICAL ALIGNMENT SYSTEM - DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows:")
    print("  Stage 1: Global calibration using 4 corner blocks")
    print("  Stage 2: Block-specific calibration for accurate positioning")
    print()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n" + "="*70)
    print("SETUP: Initialize Mock Hardware")
    print("="*70)
    
    layout_config = "config/mock_layout.json"
    layout = load_layout_config_v2(layout_config)
    print(f"✅ Layout loaded: {layout['design_name']}")
    
    # Create mock hardware
    stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    camera = MockCamera(layout_config, stage_ref=stage)
    camera.connect()
    stage.set_camera_observer(camera)
    camera.set_exposure_time(0.02)
    vt = VisionTools()
    
    print(f"✅ Mock hardware initialized")
    print(f"   Camera FOV: {camera.sensor_width * camera.nm_per_pixel / 1000:.1f} µm")
    print(f"   Resolution: {camera.nm_per_pixel} nm/pixel")
    
    # Create hierarchical alignment system
    alignment = HierarchicalAlignment(layout)
    print(f"✅ Hierarchical alignment system created")
    
    # Display initial status
    alignment.print_status()
    
    # =========================================================================
    # STAGE 1: GLOBAL CALIBRATION
    # =========================================================================
    input("\nPress Enter to start Stage 1: Global Calibration...")
    
    print("\n" + "="*70)
    print("STAGE 1: GLOBAL CALIBRATION")
    print("="*70)
    print("Using corner blocks: 1, 5, 16, 20")
    print("Finding one fiducial per block to determine overall sample orientation")
    
    
    # Ground truth converter (simulation only)
    gt_converter = CoordinateTransform(layout)
    gt = layout['simulation_ground_truth']
    gt_converter.set_transformation(gt['rotation_deg'], tuple(gt['translation_nm']))
    
    global_fiducials = []
    corner_blocks = [1, 5, 16, 20]
    corners_to_use = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    for block_id, corner in zip(corner_blocks, corners_to_use):
        print(f"\n{'─'*70}")
        print(f"Finding Block {block_id} {corner}")
        print('─'*70)
        
        # Get ground truth position (simulated measurement)
        gt_pos = gt_converter.get_fiducial_stage_position(block_id, corner)
        
        # Add small measurement noise (realistic)
        noise_Y = np.random.normal(0, 50)  # 50nm std dev
        noise_Z = np.random.normal(0, 50)
        measured_Y = gt_pos[0] + noise_Y
        measured_Z = gt_pos[1] + noise_Z
        
        print(f"  Ground truth: ({gt_pos[0]:.0f}, {gt_pos[1]:.0f}) nm")
        print(f"  Measured: ({measured_Y:.0f}, {measured_Z:.0f}) nm")
        print(f"  Noise: ({noise_Y:.1f}, {noise_Z:.1f}) nm")
        
        global_fiducials.append({
            'block_id': block_id,
            'corner': corner,
            'stage_Y': measured_Y,
            'stage_Z': measured_Z
        })
    
    # Calibrate global transformation
    global_result = alignment.calibrate_global(global_fiducials)
    
    # Validate against ground truth
    print(f"\n{'─'*70}")
    print("Validating Global Calibration Against Ground Truth")
    print('─'*70)
    
    validation = alignment.validate_global_calibration(
        gt['rotation_deg'],
        tuple(gt['translation_nm'])
    )
    
    print(f"Ground truth: rotation={gt['rotation_deg']}°, translation={gt['translation_nm']} nm")
    print(f"Calibrated: rotation={global_result['angle_deg']:.4f}°, "
          f"translation=({global_result['translation_nm'][0]:.1f}, {global_result['translation_nm'][1]:.1f}) nm")
    print(f"\nValidation errors:")
    print(f"  Rotation error: {validation['rotation_error_deg']:.4f}°")
    print(f"  Translation error: {validation['translation_error_nm']:.1f} nm")
    print(f"  Mean residual: {validation['mean_residual_nm']:.3f} nm")
    print(f"  Max residual: {validation['max_residual_nm']:.3f} nm")
    
    if validation['rotation_error_deg'] < 0.01 and validation['translation_error_nm'] < 500:
        print(f"\n✅ STAGE 1 PASSED - Global calibration accurate!")
    else:
        print(f"\n⚠️ STAGE 1 WARNING - Calibration errors larger than expected")
    
    alignment.print_status()
    
    # =========================================================================
    # STAGE 2: BLOCK-SPECIFIC CALIBRATION
    # =========================================================================
    input("\nPress Enter to start Stage 2: Block Calibration...")
    
    print("\n" + "="*70)
    print("STAGE 2: BLOCK-SPECIFIC CALIBRATION")
    print("="*70)
    print("Calibrating Block 10 using multiple fiducials within the block")
    print("This corrects for block-specific fabrication errors")
    
    target_block = 10
    
    # Predict block center using global calibration
    pred_Y, pred_Z = alignment.predict_block_center(target_block)
    block_center_design = layout['blocks'][target_block]['design_position']
    
    print(f"\n{'─'*70}")
    print(f"Block {target_block} Prediction (using Stage 1 calibration)")
    print('─'*70)
    print(f"  Design center: ({block_center_design[0]:.1f}, {block_center_design[1]:.1f}) µm")
    print(f"  Predicted stage: ({pred_Y:.0f}, {pred_Z:.0f}) nm")
    
    # Simulate finding multiple fiducials in block 10
    block_corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    block10_fiducials = []
    
    print(f"\n{'─'*70}")
    print(f"Finding fiducials in Block {target_block}")
    print('─'*70)
    
    for corner in block_corners:
        # Get ground truth position
        gt_pos = gt_converter.get_fiducial_stage_position(target_block, corner)
        
        # Add measurement noise
        noise_Y = np.random.normal(0, 30)  # Better accuracy in Stage 2
        noise_Z = np.random.normal(0, 30)
        measured_Y = gt_pos[0] + noise_Y
        measured_Z = gt_pos[1] + noise_Z
        
        # Predict using Stage 1 calibration
        pred_Y_fid, pred_Z_fid = alignment.predict_fiducial_position(target_block, corner)
        prediction_error = np.hypot(pred_Y_fid - gt_pos[0], pred_Z_fid - gt_pos[1])
        
        print(f"  {corner:12s}: predicted=({pred_Y_fid:.0f}, {pred_Z_fid:.0f}) nm, "
              f"measured=({measured_Y:.0f}, {measured_Z:.0f}) nm, "
              f"pred_error={prediction_error:.0f} nm")
        
        block10_fiducials.append({
            'corner': corner,
            'stage_Y': measured_Y,
            'stage_Z': measured_Z
        })
    
    # Calibrate block 10
    block_result = alignment.calibrate_block(target_block, block10_fiducials)
    
    # Validate (in simulation, we know ground truth is same as global)
    print(f"\n{'─'*70}")
    print(f"Block {target_block} Calibration Complete")
    print('─'*70)
    print("Note: In simulation, blocks have no fabrication errors,")
    print("so block calibration should match global calibration.")
    
    validation_block = alignment.validate_block_calibration(
        target_block,
        gt['rotation_deg'],
        tuple(gt['translation_nm'])
    )
    
    print(f"\nBlock {target_block} validation:")
    print(f"  Rotation error: {validation_block['rotation_error_deg']:.4f}°")
    print(f"  Translation error: {validation_block['translation_error_nm']:.1f} nm")
    print(f"  Mean residual: {validation_block['mean_residual_nm']:.3f} nm")
    
    if validation_block['mean_residual_nm'] < 100:
        print(f"\n✅ STAGE 2 PASSED - Block calibration accurate!")
    else:
        print(f"\n⚠️ STAGE 2 WARNING - Block calibration residuals larger than expected")
    
    alignment.print_status()
    
    # =========================================================================
    # DEMONSTRATION: COORDINATE CONVERSION
    # =========================================================================
    input("\nPress Enter to demonstrate coordinate conversion...")
    
    print("\n" + "="*70)
    print("COORDINATE CONVERSION DEMONSTRATION")
    print("="*70)
    
    # Example: Get stage position of waveguide 25 left grating in Block 10
    print(f"\nTarget: Block {target_block}, Waveguide 25, Left Grating")
    
    try:
        Y_stage, Z_stage = alignment.get_waveguide_position(target_block, 25, 'left_grating')
        
        print(f"\nStage position (using Block {target_block} calibration):")
        print(f"  Y = {Y_stage:.0f} nm")
        print(f"  Z = {Z_stage:.0f} nm")
        
        # Get ground truth for comparison
        grating_key = f"wg25_left"
        grating_local = layout['blocks'][target_block]['gratings'][grating_key]['position']
        gt_pos = gt_converter.get_stage_position(target_block, grating_local[0], grating_local[1])
        
        error = np.hypot(Y_stage - gt_pos[0], Z_stage - gt_pos[1])
        
        print(f"\nGround truth position:")
        print(f"  Y = {gt_pos[0]:.0f} nm")
        print(f"  Z = {gt_pos[1]:.0f} nm")
        print(f"\nPositioning error: {error:.1f} nm ({error/1000:.3f} µm)")
        
        if error < 200:
            print(f"✅ Excellent accuracy! Ready for alignment.")
        elif error < 1000:
            print(f"✓ Good accuracy. Within acceptable range.")
        else:
            print(f"⚠️ Large positioning error. Check calibration.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nHierarchical alignment workflow:")
    print("  1. ✅ Stage 1: Global calibration (4 corner blocks)")
    print("  2. ✅ Stage 2: Block-specific calibration (Block 10)")
    print("  3. ✅ Coordinate conversion (design → stage)")
    print("\nThe system is now ready for waveguide alignment!")
    print("\nIn a real workflow:")
    print("  - Stage 1 is done once per sample")
    print("  - Stage 2 is done for each block you want to measure")
    print("  - Coordinate conversion guides the stage to targets")
    print("="*70)