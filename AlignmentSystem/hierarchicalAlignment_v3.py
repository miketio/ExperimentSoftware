"""
Hierarchical two-stage alignment system.

Stage 1: Global sample alignment using corner blocks
Stage 2: Per-block local refinement for fine alignment

Architecture:
- Works with RuntimeLayout (typed, measurement accumulation)
- Uses CoordinateTransformV3 for all coordinate conversions
- Stores all measurements back into RuntimeLayout
- No ground truth access (real measurement scenario)

All internal operations use micrometers. Conversion to nanometers only 
happens at the hardware interface when calling stage methods.
"""
from typing import Dict, List, Tuple, Optional
from config.layout_models import RuntimeLayout, Block
import math
from AlignmentSystem.coordinate_transform_v3 import CoordinateTransformV3


class HierarchicalAlignment:
    """
    Two-stage hierarchical alignment system.
    
    Stage 1 (Global): Calibrate sample rotation/translation using 4 corner blocks
    Stage 2 (Local): Refine individual block alignment if needed
    
    All coordinates internally in micrometers (µm).
    Stores all calibration results in RuntimeLayout.
    """
    
    def __init__(self, runtime_layout: RuntimeLayout):
        """
        Initialize hierarchical alignment system.
        
        Args:
            runtime_layout: RuntimeLayout instance (design-only, fills during measurement)
        """
        self.layout = runtime_layout
        
        # Single coordinate transform - reads calibration state from RuntimeLayout
        # For now, pass None since we don't have CameraLayout at runtime
        # TODO: CoordinateTransformV3 needs RuntimeLayout support
        self.transform = CoordinateTransformV3(layout=runtime_layout)

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
        
        print(f"\n{'='*70}")
        print(f"STAGE 1: Global Sample Calibration")
        print(f"{'='*70}")
        print(f"Using {len(fiducial_measurements)} fiducial measurements")
        
        # Collect design and measured points
        design_points_um = []
        measured_points_um = []
        
        for fid in fiducial_measurements:
            block_id = fid['block_id']
            corner = fid['corner']
            
            # Get design coordinates from block (in global design frame)
            block = self.layout.get_block(block_id)
            fiducial_local = block.get_fiducial(corner)
            
            # Convert local fiducial position to global design coordinates
            # Block design_position is the center, fiducials are relative to bottom-left
            block_size = self.layout.block_layout.block_size
            u_center, v_center = block.design_position.u, block.design_position.v
            u_bl = u_center - (block_size / 2.0)
            v_bl = v_center - (block_size / 2.0)
            
            u_global = u_bl + fiducial_local.u
            v_global = v_bl + fiducial_local.v
            
            design_points_um.append((u_global, v_global))
            
            # Measured stage coordinates (already in µm from searcher)
            measured_points_um.append((fid['stage_Y'], fid['stage_Z']))
            
            print(f"  Block {block_id} {corner}:")
            print(f"    Design: ({u_global:.2f}, {v_global:.2f}) µm")
            print(f"    Measured: ({fid['stage_Y']:.2f}, {fid['stage_Z']:.2f}) µm")
        
        # Calibrate transform
        result = self.transform.calibrate(
            measured_points=measured_points_um,
            design_points=design_points_um
        )
        
        print(f"\n✅ Global calibration complete:")
        print(f"  Rotation: {result['angle_deg']:.6f}°")
        print(f"  Translation: ({result['translation_um'][0]:.2f}, {result['translation_um'][1]:.2f}) µm")
        print(f"  Mean error: {result['mean_error_um']:.6f} µm")
        print(f"  Max error: {result['max_error_um']:.6f} µm")
        
        # Store in RuntimeLayout
        self.layout.set_global_calibration(
            rotation=result['angle_deg'],
            translation=result['translation_um'],
            calibration_error=result['mean_error_um'],
            num_points=result['num_points']
        )
        self.transform.sync_with_runtime()
        return {
            'stage': 'global',
            'num_fiducials': len(fiducial_measurements),
            'rotation_deg': result['angle_deg'],
            'translation_um': result['translation_um'],
            'mean_error_um': result['mean_error_um'],
            'max_error_um': result['max_error_um'],
            'method': result['method']
        }
    
    # ========================================================================
    # STAGE 2: BLOCK-LEVEL CALIBRATION
    # ========================================================================
        
    def calibrate_block(self, 
                    block_id: int,
                    fiducial_measurements: List[Dict]) -> Dict:
        """
        Perform Stage 2 block-level calibration for fine alignment.
        
        Uses precise block fiducial measurements to directly determine the block's
        position and orientation in stage coordinates. This creates a high-precision
        local coordinate frame for the block, independent of the rough global calibration.
        
        Args:
            block_id: Block to calibrate
            fiducial_measurements: List of dicts with keys:
                - 'corner': Corner name
                - 'stage_Y': Measured Y in µm (global stage frame)
                - 'stage_Z': Measured Z in µm (global stage frame)
        
        Returns:
            dict: Block calibration results
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError("Must perform global calibration first")
        
        if len(fiducial_measurements) < 2:
            raise ValueError("Need at least 2 fiducial measurements for block calibration")
        
        print(f"\n{'='*70}")
        print(f"STAGE 2: Block-Level Calibration (Block {block_id})")
        print(f"{'='*70}")
        print(f"Using {len(fiducial_measurements)} fiducial measurements")
        print(f"Strategy: Direct block pose determination from precise fiducial positions")
        
        # Get block design data
        block = self.layout.get_block(block_id)
        
        # Collect design and measured positions
        # Design: block-local coordinates
        # Measured: stage coordinates (precise measurements)
        design_points_local = []
        measured_points_stage = []
        
        for fid in fiducial_measurements:
            corner = fid['corner']
            
            # Get fiducial design position in block-local frame
            fiducial_design = block.get_fiducial(corner)
            design_points_local.append((fiducial_design.u, fiducial_design.v))
            
            # Get measured stage position (precise)
            measured_points_stage.append((fid['stage_Y'], fid['stage_Z']))
            
            print(f"  {corner}:")
            print(f"    Design (block-local): ({fiducial_design.u:>7.2f}, {fiducial_design.v:>7.2f}) µm")
            print(f"    Measured (stage):     ({fid['stage_Y']:>7.2f}, {fid['stage_Z']:>7.2f}) µm")
        
        # Create a precise block-to-stage transform
        # This directly calibrates: block_local → stage
        # Using precise fiducial measurements (not going through rough global transform)
        block_transform = CoordinateTransformV3(layout=None)
        result = block_transform.calibrate(
            measured_points=measured_points_stage,  # Precise stage measurements
            design_points=design_points_local        # Block-local design positions
        )
        
        print(f"\n✅ Block {block_id} calibration complete:")
        print(f"  Block rotation (in stage frame): {result['angle_deg']:.6f}°")
        print(f"  Block translation (in stage frame): ({result['translation_um'][0]:.2f}, {result['translation_um'][1]:.2f}) µm")
        print(f"  Mean residual error: {result['mean_error_um']:.6f} µm")
        print(f"  Max residual error: {result['max_error_um']:.6f} µm")
        
        # Now we need to figure out what this means in terms of "correction"
        # The block transform gives us: block_local → stage directly
        # But we need to store it as a correction relative to what the global transform predicts
        
        # Get block center in design coordinates
        block_center = block.design_position
        
        # What does the global transform predict for block center?
        self.transform.sync_with_runtime()
        predicted_center_Y, predicted_center_Z = self.transform.design_to_stage(
            block_center.u, block_center.v
        )
        
        # What does the block transform say about where (0, 0) in block-local should be?
        # For block-local (0, 0), which is at bottom-left corner:
        block_size = self.layout.block_layout.block_size
        
        # Block center in block-local coordinates
        if block_size:
            center_local_u = block_size / 2.0
            center_local_v = block_size / 2.0
        else:
            # If block_size not available, estimate from fiducials
            center_local_u = sum(p[0] for p in design_points_local) / len(design_points_local)
            center_local_v = sum(p[1] for p in design_points_local) / len(design_points_local)
        
        # Apply block transform to get actual center position in stage
        cos_a = math.cos(math.radians(result['angle_deg']))
        sin_a = math.sin(math.radians(result['angle_deg']))
        
        actual_center_Y = (result['translation_um'][0] + 
                        cos_a * center_local_u - sin_a * center_local_v)
        actual_center_Z = (result['translation_um'][1] + 
                        sin_a * center_local_u + cos_a * center_local_v)
        
        # The correction is the difference
        correction_translation = (
            actual_center_Y - predicted_center_Y,
            actual_center_Z - predicted_center_Z
        )
        
        # For rotation, the correction is the difference between:
        # - What block transform found
        # - What global transform has
        global_rotation = self.transform.global_transform.rotation_deg
        correction_rotation = result['angle_deg'] - global_rotation
        
        print(f"\n📊 Block correction analysis:")
        print(f"  Global transform predicts center at: ({predicted_center_Y:.2f}, {predicted_center_Z:.2f}) µm")
        print(f"  Block calibration finds center at:   ({actual_center_Y:.2f}, {actual_center_Z:.2f}) µm")
        print(f"  Correction needed: ({correction_translation[0]:.3f}, {correction_translation[1]:.3f}) µm")
        print(f"  Global rotation: {global_rotation:.6f}°")
        print(f"  Block rotation:  {result['angle_deg']:.6f}°")
        print(f"  Rotation correction: {correction_rotation:.6f}°")
        
        # Store the correction in RuntimeLayout
        self.layout.set_block_calibration(
            block_id=block_id,
            rotation=correction_rotation,
            translation=correction_translation,
            calibration_error=result['mean_error_um'],
            num_points=result['num_points']
        )
        
        # Sync transform to pick up new block calibration
        self.transform.sync_with_runtime()
        
        return {
            'stage': 'block',
            'block_id': block_id,
            'num_fiducials': len(fiducial_measurements),
            'rotation_deg': correction_rotation,
            'translation_um': correction_translation,
            'mean_error_um': result['mean_error_um'],
            'max_error_um': result['max_error_um'],
            'method': result['method'],
            # Additional info for debugging
            'absolute_rotation_deg': result['angle_deg'],
            'absolute_translation_um': result['translation_um'],
            'predicted_center_stage': (predicted_center_Y, predicted_center_Z),
            'actual_center_stage': (actual_center_Y, actual_center_Z)
        }
    
    # ========================================================================
    # PREDICTION METHODS (delegate to CoordinateTransformV3)
    # ========================================================================
    
    def get_fiducial_stage_position(self,
                                 block_id: int,
                                 corner: str) -> Tuple[float, float]:
        """
        Predict stage position of a fiducial marker.
        
        Uses global calibration + block calibration (if available).
        
        Args:
            block_id: Block identifier
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        
        Raises:
            RuntimeError: If not globally calibrated
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError(
                "Cannot predict positions - global calibration not performed. "
                "Call calibrate_global() first."
            )
        
        # Get fiducial local coordinates
        block = self.layout.get_block(block_id)
        fiducial = block.get_fiducial(corner)
        
        # Update transform with latest RuntimeLayout calibration
        self.transform.sync_with_runtime()
        
        # Delegate to CoordinateTransformV3
        # TODO: This needs block-aware logic in CoordinateTransformV3
        # For now, use block_local_to_stage if available
        try:
            return self.transform.block_local_to_stage(
                block_id, fiducial.u, fiducial.v
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not predict fiducial position: {e}. "
                f"Make sure CoordinateTransformV3 is properly configured with RuntimeLayout."
            )
    
    def get_grating_stage_position(self,
                                block_id: int,
                                waveguide: int,
                                side: str) -> Tuple[float, float]:
        """
        Predict stage position of a grating coupler.
        
        Args:
            block_id: Block identifier
            waveguide: Waveguide number
            side: 'left' or 'right'
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        
        Raises:
            RuntimeError: If not globally calibrated
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError(
                "Cannot predict positions - global calibration not performed. "
                "Call calibrate_global() first."
            )
        
        # Get grating local coordinates
        block = self.layout.get_block(block_id)
        grating = block.get_grating(waveguide, side)
        
        # Update transform with latest RuntimeLayout calibration
        self.transform.sync_with_runtime()
        
        # Delegate to CoordinateTransformV3
        try:
            return self.transform.block_local_to_stage(
                block_id, grating.position.u, grating.position.v
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not predict grating position: {e}. "
                f"Make sure CoordinateTransformV3 is properly configured with RuntimeLayout."
            )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_corner_blocks(self) -> List[int]:
        """
        Get corner block IDs for Stage 1 calibration.
        
        Returns:
            List of block IDs at array corners
        """
        return self.layout.get_corner_blocks()
    
    def get_calibration_status(self) -> Dict:
        """
        Get current calibration status from RuntimeLayout.
        
        Returns:
            dict: Status information including calibrated blocks
        """
        status = {
            'global_calibrated': self.layout.is_globally_calibrated(),
            'num_blocks_calibrated': len(self.layout.get_calibrated_blocks()),
            'calibrated_blocks': self.layout.get_calibrated_blocks()
        }
        
        if self.layout.is_globally_calibrated():
            global_trans = self.layout.get_global_transform()
            status['global_rotation_deg'] = global_trans.rotation_deg
            status['global_translation_um'] = global_trans.translation_um.to_tuple()
            status['global_calibration_error_um'] = global_trans.calibration_error_um
        
        # Add block-level calibration info
        if status['num_blocks_calibrated'] > 0:
            status['block_calibrations'] = {}
            for block_id in self.layout.get_calibrated_blocks():
                block_trans = self.layout.get_block_transform(block_id)
                status['block_calibrations'][block_id] = {
                    'rotation_deg': block_trans.rotation_deg,
                    'translation_um': block_trans.translation_um.to_tuple(),
                    'calibration_error_um': block_trans.calibration_error_um
                }
        
        return status
    
    def save_calibration(self, filepath: str):
        """
        Save calibration results to JSON file.
        
        Args:
            filepath: Output file path
        """
        self.layout.save_to_json(filepath, include_design=True)
        print(f"\n💾 Calibration saved to: {filepath}")
    
    def reset_calibration(self, block_id: Optional[int] = None):
        """
        Reset calibration data in RuntimeLayout.
        
        Args:
            block_id: If provided, reset only this block. Otherwise reset all.
        """
        if block_id is None:
            # Reset everything by creating new RuntimeLayout
            # (This is a placeholder - RuntimeLayout should have a reset method)
            print("⚠️  Warning: Full reset not implemented. Create new RuntimeLayout instance.")
        else:
            # Reset specific block
            # (RuntimeLayout should have a method to clear block calibration)
            print(f"⚠️  Warning: Block {block_id} reset not implemented in RuntimeLayout.")


# ============================================================================
# TEST/EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Hierarchical Alignment System v3")
    print("=" * 70)
    
    try:
        from config.layout_models import RuntimeLayout, CameraLayout
        
        # Load RuntimeLayout (design-only, no ground truth)
        runtime = RuntimeLayout.from_json_file("config/mock_layout.json")
        print(f"Loaded RuntimeLayout: {runtime}")
        
        # Also load CameraLayout for simulation (has ground truth)
        camera = CameraLayout.from_json_file("config/mock_layout.json")
        print(f"Loaded CameraLayout: {camera}")
        
        # Initialize alignment system with RuntimeLayout
        alignment = HierarchicalAlignment(runtime)
        
        print("\n" + "=" * 70)
        print("SIMULATION: Using ground truth to generate measurements")
        print("=" * 70)
        
        # Create a simulator transform using CameraLayout (has ground truth)
        from AlignmentSystem.coordinate_transform_v3 import CoordinateTransformV3
        sim_transform = CoordinateTransformV3(camera)
        sim_transform.use_ground_truth()
        
        # Get corner blocks for Stage 1
        corner_blocks = runtime.get_corner_blocks()
        print(f"Corner blocks for Stage 1: {corner_blocks}")
        
        # Simulate fiducial measurements for Stage 1
        fiducial_measurements = []
        for block_id in corner_blocks[:4]:  # Use first 4 corners
            for corner in ['top_left', 'bottom_right']:
                # Simulate measurement using ground truth
                Y, Z = sim_transform.get_fiducial_stage_position(block_id, corner)
                
                fiducial_measurements.append({
                    'block_id': block_id,
                    'corner': corner,
                    'stage_Y': Y,
                    'stage_Z': Z
                })
        
        # Perform Stage 1 calibration
        global_result = alignment.calibrate_global(fiducial_measurements)
        
        print("\n" + "=" * 70)
        print("Stage 1 Results vs Ground Truth")
        print("=" * 70)
        print(f"Measured rotation: {global_result['rotation_deg']:.6f}°")
        print(f"True rotation:     {camera.ground_truth.rotation_deg:.6f}°")
        print(f"Rotation error:    {abs(global_result['rotation_deg'] - camera.ground_truth.rotation_deg):.9f}°")
        
        # Check calibration status
        status = alignment.get_calibration_status()
        print("\n" + "=" * 70)
        print("Calibration Status")
        print("=" * 70)
        print(f"Global calibrated: {status['global_calibrated']}")
        print(f"Rotation: {status['global_rotation_deg']:.6f}°")
        print(f"Translation: {status['global_translation_um']} µm")
        
        # Save results
        alignment.save_calibration("results/test_calibration.json")
        
        print("\n✅ Basic test passed!")
        print("\nNOTE: Full integration requires CoordinateTransformV3 to support RuntimeLayout")
        print("      and handle block-level calibrations properly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()