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
from alignment_system.coordinate_transform_v3 import CoordinateTransformV3


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
        Perform Stage 2 block-level calibration using DIRECT transform.
        
        Creates a precise block_local → stage transform from measured fiducials.
        This bypasses the rough global transform for maximum accuracy.
        
        Args:
            block_id: Block to calibrate
            fiducial_measurements: List of dicts with keys:
                - 'corner': Corner name ('top_left', 'bottom_right')
                - 'stage_Y': Measured Y in µm
                - 'stage_Z': Measured Z in µm
        
        Returns:
            dict: Block calibration results with direct transform parameters
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError("Must perform global calibration first")
        
        if len(fiducial_measurements) < 2:
            raise ValueError("Need at least 2 fiducial measurements for block calibration")
        
        print(f"\n{'='*70}")
        print(f"STAGE 2: Block-Level Calibration (Block {block_id}) - DIRECT METHOD")
        print(f"{'='*70}")
        print(f"Using {len(fiducial_measurements)} fiducial measurements")
        
        # Get block and extract measured positions
        block = self.layout.get_block(block_id)
        
        # For 2-point calibration (top_left and bottom_right)
        meas_tl = next((m for m in fiducial_measurements if m['corner'] == 'top_left'), None)
        meas_br = next((m for m in fiducial_measurements if m['corner'] == 'bottom_right'), None)
        
        if not (meas_tl and meas_br):
            raise ValueError("Need 'top_left' and 'bottom_right' measurements for block calibration")
        
        # Measured stage positions
        meas_tl_Y, meas_tl_Z = meas_tl['stage_Y'], meas_tl['stage_Z']
        meas_br_Y, meas_br_Z = meas_br['stage_Y'], meas_br['stage_Z']
        
        print(f"  top_left measured:     ({meas_tl_Y:.3f}, {meas_tl_Z:.3f}) µm")
        print(f"  bottom_right measured: ({meas_br_Y:.3f}, {meas_br_Z:.3f}) µm")
        
        # Design positions (block-local)
        fid_tl_design = block.get_fiducial('top_left')
        fid_br_design = block.get_fiducial('bottom_right')
        design_tl_u, design_tl_v = fid_tl_design.u, fid_tl_design.v
        design_br_u, design_br_v = fid_br_design.u, fid_br_design.v
        
        print(f"  top_left design:       ({design_tl_u:.3f}, {design_tl_v:.3f}) µm (local)")
        print(f"  bottom_right design:   ({design_br_u:.3f}, {design_br_v:.3f}) µm (local)")
        
        # Calculate block rotation from vector angles
        vec_stage_Y = meas_br_Y - meas_tl_Y
        vec_stage_Z = meas_br_Z - meas_tl_Z
        vec_design_u = design_br_u - design_tl_u
        vec_design_v = design_br_v - design_tl_v
        
        angle_stage_deg = math.degrees(math.atan2(vec_stage_Z, vec_stage_Y))
        angle_design_deg = math.degrees(math.atan2(vec_design_v, vec_design_u))
        block_rotation_deg = angle_stage_deg - angle_design_deg
        block_rotation_rad = math.radians(block_rotation_deg)
        
        cos_theta = math.cos(block_rotation_rad)
        sin_theta = math.sin(block_rotation_rad)
        
        print(f"\n  Calculated rotation: {block_rotation_deg:.6f}°")
        
        # Calculate block origin (bottom-left in stage coords)
        # We know: stage = origin + R * local
        # So: origin = stage - R * local
        block_origin_Y = meas_tl_Y - (cos_theta * design_tl_u - sin_theta * design_tl_v)
        block_origin_Z = meas_tl_Z - (sin_theta * design_tl_u + cos_theta * design_tl_v)
        
        print(f"  Calculated origin: ({block_origin_Y:.3f}, {block_origin_Z:.3f}) µm")
        
        # Verify accuracy by back-transforming
        verify_tl_Y = block_origin_Y + (cos_theta * design_tl_u - sin_theta * design_tl_v)
        verify_tl_Z = block_origin_Z + (sin_theta * design_tl_u + cos_theta * design_tl_v)
        verify_br_Y = block_origin_Y + (cos_theta * design_br_u - sin_theta * design_br_v)
        verify_br_Z = block_origin_Z + (sin_theta * design_br_u + cos_theta * design_br_v)
        
        error_tl = math.hypot(verify_tl_Y - meas_tl_Y, verify_tl_Z - meas_tl_Z)
        error_br = math.hypot(verify_br_Y - meas_br_Y, verify_br_Z - meas_br_Z)
        mean_error = (error_tl + error_br) / 2.0
        max_error = max(error_tl, error_br)
        
        print(f"\n✅ Verification (back-transform):")
        print(f"  top_left error:     {error_tl:.6f} µm")
        print(f"  bottom_right error: {error_br:.6f} µm")
        print(f"  Mean error:         {mean_error:.6f} µm")
        
        # Store direct transform parameters in RuntimeLayout
        # We store: rotation, origin (translation), for direct block_local → stage
        self.layout.set_block_calibration(
            block_id=block_id,
            rotation=block_rotation_deg,
            translation=(block_origin_Y, block_origin_Z),
            calibration_error=mean_error,
            num_points=2
        )
        
        return {
            'stage': 'block',
            'block_id': block_id,
            'num_fiducials': 2,
            'rotation_deg': block_rotation_deg,
            'origin_stage_um': (block_origin_Y, block_origin_Z),
            'mean_error_um': mean_error,
            'max_error_um': max_error,
            'method': 'direct_2point_transform'
        }
    
    # ========================================================================
    # PREDICTION METHODS (delegate to CoordinateTransformV3)
    # ========================================================================
        
    def get_grating_stage_position(self,
                                block_id: int,
                                waveguide: int,
                                side: str) -> Tuple[float, float]:
        """
        Predict stage position of a grating coupler.
        
        Uses direct block transform if available, otherwise falls back to global.
        
        Args:
            block_id: Block identifier
            waveguide: Waveguide number
            side: 'left' or 'right'
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError(
                "Cannot predict positions - global calibration not performed. "
                "Call calibrate_global() first."
            )
        
        # Get grating local coordinates
        block = self.layout.get_block(block_id)
        grating = block.get_grating(waveguide, side)
        grating_u = grating.position.u
        grating_v = grating.position.v
        
        # Use DIRECT block transform if available
        if self.layout.is_block_calibrated(block_id):
            block_transform = self.layout.get_block_transform(block_id)
            
            # Get stored direct transform parameters
            # rotation_deg is the block rotation
            # translation_um is the block origin
            rotation_rad = math.radians(block_transform.rotation_deg)
            origin_Y = block_transform.translation_um.u
            origin_Z = block_transform.translation_um.v
            
            cos_theta = math.cos(rotation_rad)
            sin_theta = math.sin(rotation_rad)
            
            # Apply direct transform: stage = origin + R * local
            stage_Y = origin_Y + (cos_theta * grating_u - sin_theta * grating_v)
            stage_Z = origin_Z + (sin_theta * grating_u + cos_theta * grating_v)
            
            return (stage_Y, stage_Z)
        
        else:
            # Fallback to global transform (rough prediction)
            print(f"⚠️  Warning: Block {block_id} not calibrated, using rough global transform")
            self.transform.sync_with_runtime()
            return self.transform.block_local_to_stage(block_id, grating_u, grating_v)
        
    def get_fiducial_stage_position(self,
                                block_id: int,
                                corner: str) -> Tuple[float, float]:
        """
        Predict stage position of a fiducial marker.
        
        Uses direct block transform if available, otherwise falls back to global.
        
        Args:
            block_id: Block identifier
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError(
                "Cannot predict positions - global calibration not performed. "
                "Call calibrate_global() first."
            )
        
        # Get fiducial local coordinates
        block = self.layout.get_block(block_id)
        fiducial = block.get_fiducial(corner)
        fid_u = fiducial.u
        fid_v = fiducial.v
        
        # Use DIRECT block transform if available
        if self.layout.is_block_calibrated(block_id):
            block_transform = self.layout.get_block_transform(block_id)
            
            rotation_rad = math.radians(block_transform.rotation_deg)
            origin_Y = block_transform.translation_um.u
            origin_Z = block_transform.translation_um.v
            
            cos_theta = math.cos(rotation_rad)
            sin_theta = math.sin(rotation_rad)
            
            stage_Y = origin_Y + (cos_theta * fid_u - sin_theta * fid_v)
            stage_Z = origin_Z + (sin_theta * fid_u + cos_theta * fid_v)
            
            return (stage_Y, stage_Z)
        
        else:
            # Fallback to global transform
            print(f"⚠️  Warning: Block {block_id} not calibrated, using rough global transform")
            self.transform.sync_with_runtime()
            return self.transform.block_local_to_stage(block_id, fid_u, fid_v)
        

    def get_fiducial_stage_position(self,
                                block_id: int,
                                corner: str) -> Tuple[float, float]:
        """
        Predict stage position of a fiducial marker.
        
        Uses direct block transform if available, otherwise falls back to global.
        
        Args:
            block_id: Block identifier
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Predicted stage coordinates in µm
        """
        if not self.layout.is_globally_calibrated():
            raise RuntimeError(
                "Cannot predict positions - global calibration not performed. "
                "Call calibrate_global() first."
            )
        
        # Get fiducial local coordinates
        block = self.layout.get_block(block_id)
        fiducial = block.get_fiducial(corner)
        fid_u = fiducial.u
        fid_v = fiducial.v
        
        # Use DIRECT block transform if available
        if self.layout.is_block_calibrated(block_id):
            block_transform = self.layout.get_block_transform(block_id)
            
            rotation_rad = math.radians(block_transform.rotation_deg)
            origin_Y = block_transform.translation_um.u
            origin_Z = block_transform.translation_um.v
            
            cos_theta = math.cos(rotation_rad)
            sin_theta = math.sin(rotation_rad)
            
            stage_Y = origin_Y + (cos_theta * fid_u - sin_theta * fid_v)
            stage_Z = origin_Z + (sin_theta * fid_u + cos_theta * fid_v)
            
            return (stage_Y, stage_Z)
        
        else:
            # Fallback to global transform
            print(f"⚠️  Warning: Block {block_id} not calibrated, using rough global transform")
            self.transform.sync_with_runtime()
            return self.transform.block_local_to_stage(block_id, fid_u, fid_v)
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
        from alignment_system.coordinate_transform_v3 import CoordinateTransformV3
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