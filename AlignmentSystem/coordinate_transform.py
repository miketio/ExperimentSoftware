"""
Unified 2D coordinate transformation for sample alignment.
Handles rotation, translation, calibration, and layout-aware conversions.

Coordinate Flow:
1. Local block coords (u, v in µm) - relative to block bottom-left
2. Global design coords (u, v in µm) - unrotated, origin at Block 1 bottom-left
3. Stage coords (Y, Z in µm) - after rotation + translation

Transformation: 
    [Y]   [cos(θ)  -sin(θ)] [u]   [Y₀]
    [Z] = [sin(θ)   cos(θ)] [v] + [Z₀]
    
All internal operations use micrometers. Conversion to nanometers only 
happens at the interface with the stage hardware.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict


class CoordinateTransform:
    """
    Unified coordinate transformation with calibration and layout support.
    
    Design coordinates:
        - u: along waveguide length (typically 10-190 µm within blocks)
        - v: across waveguides (typically ~95-140 µm within blocks)
        - Units: micrometers
    
    Stage coordinates:
        - Y: corresponds to u direction
        - Z: corresponds to v direction  
        - Units: micrometers (internally), nanometers (at hardware interface)
    """
    
    def __init__(self, layout: Optional[Dict] = None):
        """
        Initialize coordinate transform.
        
        Args:
            layout: Optional layout dict from layout_config_generator_v2
                    Required for block-aware coordinate conversions
        """
        self.rotation_matrix = np.eye(2)
        self.translation = np.array([0.0, 0.0])
        self.is_calibrated = False
        self.angle_deg = 0.0
        
        # Layout support
        self.layout = layout
        self.blocks = layout['blocks'] if layout else None
        
    # ========================================================================
    # CALIBRATION METHODS
    # ========================================================================
    
    def calibrate(self, 
                  measured_points: List[Tuple[float, float]],
                  design_points: List[Tuple[float, float]]) -> Dict:
        """
        Calibrate transformation using measured fiducial positions.
        
        Args:
            measured_points: List of (Y, Z) in MICROMETERS from stage
            design_points: List of (u, v) in MICROMETERS from design
        
        Returns:
            dict with calibration results
        """
        if len(measured_points) != len(design_points):
            raise ValueError("Number of measured and design points must match")
        
        if len(measured_points) < 2:
            raise ValueError("Need at least 2 points for calibration")
        
        measured = np.array(measured_points, dtype=float)  # µm
        design = np.array(design_points, dtype=float)  # µm
        
        if len(measured) == 2:
            # Two-point calibration (exact solution)
            result = self._calibrate_two_points(measured, design)
        else:
            # Least-squares for 3+ points
            result = self._calibrate_least_squares(measured, design)
        
        self.is_calibrated = True
        return result
    
    def _calibrate_two_points(self, measured: np.ndarray, design: np.ndarray) -> Dict:
        """
        Calibrate using exactly 2 points.
        
        Args:
            measured: 2x2 array of (Y, Z) in µm
            design: 2x2 array of (u, v) in µm
        
        Returns:
            Calibration results dict
        """
        # Vector from point 0 to point 1 in both coordinate systems
        v_design = design[1] - design[0]
        v_measured = measured[1] - measured[0]
        
        # Calculate rotation angle
        angle_design = np.arctan2(v_design[1], v_design[0])
        angle_measured = np.arctan2(v_measured[1], v_measured[0])
        angle_diff = angle_measured - angle_design
        
        self.angle_deg = np.degrees(angle_diff)
        
        # Build rotation matrix
        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)
        self.rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Calculate translation using first point
        rotated_design = self.rotation_matrix @ design[0]
        self.translation = measured[0] - rotated_design
        
        # Calculate residual error
        errors = []
        for i in range(2):
            predicted = self.rotation_matrix @ design[i] + self.translation
            error = np.linalg.norm(predicted - measured[i])
            errors.append(error)
        
        return {
            'method': 'two_point',
            'angle_deg': self.angle_deg,
            'translation_um': self.translation.tolist(),
            'mean_error_um': float(np.mean(errors)),
            'max_error_um': float(np.max(errors)),
            'num_points': 2
        }
    
    def _calibrate_least_squares(self, measured: np.ndarray, design: np.ndarray) -> Dict:
        """
        Calibrate using least-squares fit for 3+ points.
        
        Args:
            measured: Nx2 array of (Y, Z) in µm
            design: Nx2 array of (u, v) in µm
        
        Returns:
            Calibration results dict
        """
        n = len(measured)
        
        # Center the point clouds
        design_center = np.mean(design, axis=0)
        measured_center = np.mean(measured, axis=0)
        
        design_centered = design - design_center
        measured_centered = measured - measured_center
        
        # Compute covariance matrix
        H = design_centered.T @ measured_centered
        
        # SVD for optimal rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        self.rotation_matrix = R
        
        # Extract angle
        self.angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        
        # Calculate translation
        rotated_center = self.rotation_matrix @ design_center
        self.translation = measured_center - rotated_center
        
        # Calculate residual errors
        errors = []
        for i in range(n):
            predicted = self.rotation_matrix @ design[i] + self.translation
            error = np.linalg.norm(predicted - measured[i])
            errors.append(error)
        
        return {
            'method': 'least_squares',
            'angle_deg': self.angle_deg,
            'translation_um': self.translation.tolist(),
            'mean_error_um': float(np.mean(errors)),
            'max_error_um': float(np.max(errors)),
            'std_error_um': float(np.std(errors)),
            'num_points': n
        }
    
    def set_transformation(self, rotation_deg: float, translation_um: Tuple[float, float]):
        """
        Manually set rotation and translation.
        
        Args:
            rotation_deg: Rotation angle in degrees
            translation_um: (Y₀, Z₀) translation in MICROMETERS
        """
        self.angle_deg = rotation_deg
        angle_rad = np.radians(rotation_deg)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        self.rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        self.translation = np.array(translation_um, dtype=float)
        self.is_calibrated = True
    
    # ========================================================================
    # CORE TRANSFORMATION METHODS
    # ========================================================================
    
    def design_to_stage(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert global design coordinates (µm) to stage coordinates (µm).
        
        Args:
            u, v: Global design coordinates in micrometers
        
        Returns:
            (Y, Z): Stage coordinates in micrometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        design_point = np.array([u, v])
        stage_point = self.rotation_matrix @ design_point + self.translation
        
        return (float(stage_point[0]), float(stage_point[1]))
    
    def stage_to_design(self, Y: float, Z: float) -> Tuple[float, float]:
        """
        Convert stage coordinates (µm) to global design coordinates (µm).
        
        Args:
            Y, Z: Stage coordinates in micrometers
        
        Returns:
            (u, v): Global design coordinates in micrometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        stage_point = np.array([float(Y), float(Z)])
        
        # Inverse transformation: R^T @ (stage - t)
        # Using transpose instead of inverse for numerical stability
        design_point = self.rotation_matrix.T @ (stage_point - self.translation)
        
        u = design_point[0]
        v = design_point[1]
        
        return (u, v)
    
    # ========================================================================
    # LAYOUT-AWARE METHODS (require layout in __init__)
    # ========================================================================
    
    def _validate_block_id(self, block_id: int):
        """Validate that block_id exists in layout."""
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found in layout")
    
    def _apply_block_fabrication_error(
        self,
        block_id: int,
        u_local: float,
        v_local: float
        ) -> Tuple[float, float]:
        """
        Apply block fabrication error (rotation + translation) to local coordinates.
        
        Manufacturing errors rotate around the BLOCK CENTER, not the corner.
        
        This transforms from ideal local coords to actual fabricated local coords.
        
        Args:
            block_id: Block identifier
            u_local: Ideal local u coordinate (µm)
            v_local: Ideal local v coordinate (µm)
        
        Returns:
            (u_fab, v_fab): Local coordinates with fabrication error applied (µm)
        """
        if self.layout is None:
            return (u_local, v_local)
        
        block = self.blocks[block_id]
        
        # Check if fabrication error exists
        if 'fabrication_error' not in block:
            return (u_local, v_local)
        
        fab_error = block['fabrication_error']
        fab_rotation_deg = fab_error['rotation_deg']
        fab_translation = np.array(fab_error['translation_um'])
        
        # Build fabrication rotation matrix
        angle_rad = np.radians(fab_rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        fab_rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Get block center in local coordinates
        block_size = self.layout['block_layout']['block_size']
        center = np.array([block_size / 2.0, block_size / 2.0])
        
        # Apply rotation around center
        local_point = np.array([u_local, v_local])
        centered = local_point - center
        rotated = fab_rotation_matrix @ centered
        
        # Apply translation and shift back
        fabricated = rotated + center + fab_translation
        
        return (float(fabricated[0]), float(fabricated[1]))
    
    def _remove_block_fabrication_error(
        self,
        block_id: int,
        u_fab: float,
        v_fab: float
        ) -> Tuple[float, float]:
        """
        Remove block fabrication error (inverse operation).
        
        This transforms from actual fabricated local coords to ideal local coords.
        
        Args:
            block_id: Block identifier
            u_fab: Fabricated local u coordinate (µm)
            v_fab: Fabricated local v coordinate (µm)
        
        Returns:
            (u_local, v_local): Ideal local coordinates (µm)
        """
        if self.layout is None:
            return (u_fab, v_fab)
        
        block = self.blocks[block_id]
        
        # Check if fabrication error exists
        if 'fabrication_error' not in block:
            return (u_fab, v_fab)
        
        fab_error = block['fabrication_error']
        fab_rotation_deg = fab_error['rotation_deg']
        fab_translation = np.array(fab_error['translation_um'])
        
        # Build fabrication rotation matrix
        angle_rad = np.radians(fab_rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        fab_rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Get block center in local coordinates
        block_size = self.layout['block_layout']['block_size']
        center = np.array([block_size / 2.0, block_size / 2.0])
        
        # Inverse transformation: R^T @ (fabricated - center - t) + center
        fabricated_point = np.array([u_fab, v_fab])
        shifted = fabricated_point - center - fab_translation
        rotated_back = fab_rotation_matrix.T @ shifted
        local_point = rotated_back + center
        
        return (float(local_point[0]), float(local_point[1]))
    
    def block_local_to_stage(
        self,
        block_id: int,
        u_local: float,
        v_local: float
    ) -> Tuple[float, float]:
        """
        Convert local block coordinates to stage coordinates.
        
        Flow: local (ideal) → local (fabricated) → global design → stage
        
        Args:
            block_id: Block ID (1-20)
            u_local: Ideal local u coordinate in µm
            v_local: Ideal local v coordinate in µm
        
        Returns:
            (Y, Z): Stage coordinates in micrometers
        """
        self._validate_block_id(block_id)
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        # Step 1: Apply block fabrication error
        u_fab, v_fab = self._apply_block_fabrication_error(block_id, u_local, v_local)
        
        # Step 2: Convert to global design coordinates
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        block_size = self.layout['block_layout']['block_size']
        
        u_global = block_u_center - block_size / 2.0 + u_fab
        v_global = block_v_center - block_size / 2.0 + v_fab
        
        # Step 3: Apply global sample transform
        return self.design_to_stage(u_global, v_global)
    
    def stage_to_block_local(
        self,
        Y: float,
        Z: float,
        block_id: int
    ) -> Tuple[float, float]:
        """
        Convert stage coordinates to local block coordinates.
        
        Flow: stage → global design → local (fabricated) → local (ideal)
        
        Args:
            Y, Z: Stage coordinates in micrometers
            block_id: Block ID
        
        Returns:
            (u_local, v_local): Ideal local coordinates in µm (relative to block bottom-left)
        """
        self._validate_block_id(block_id)
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        # Step 1: Convert stage to global design coords
        u_global, v_global = self.stage_to_design(Y, Z)
        
        # Step 2: Convert global design to local fabricated coords
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        block_size = self.layout['block_layout']['block_size']
        
        # Block bottom-left in global coords
        block_u_bottom_left = block_u_center - block_size / 2.0
        block_v_bottom_left = block_v_center - block_size / 2.0
        
        # Subtract to get fabricated local coords
        u_fab = u_global - block_u_bottom_left
        v_fab = v_global - block_v_bottom_left
        
        # Step 3: Remove block fabrication error to get ideal local coords
        u_local, v_local = self._remove_block_fabrication_error(block_id, u_fab, v_fab)
        
        return (u_local, v_local)
    
    def get_fiducial_stage_position(
        self,
        block_id: int,
        corner: str
    ) -> Tuple[float, float]:
        """
        Get stage position of a fiducial marker.
        
        Args:
            block_id: Block ID
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Stage coordinates in µm
        """
        self._validate_block_id(block_id)
        
        block = self.blocks[block_id]
        if 'fiducials' not in block or corner not in block['fiducials']:
            raise ValueError(f"Fiducial '{corner}' not found in block {block_id}")
        
        local_pos = block['fiducials'][corner]
        
        return self.block_local_to_stage(block_id, local_pos[0], local_pos[1])
    
    def get_grating_stage_position(
        self,
        block_id: int,
        waveguide: int,
        side: str
    ) -> Tuple[float, float]:
        """
        Get stage position of a grating coupler.
        
        Args:
            block_id: Block ID
            waveguide: Waveguide number
            side: 'left' or 'right'
        
        Returns:
            (Y, Z): Stage coordinates in µm
        """
        self._validate_block_id(block_id)
        
        block = self.blocks[block_id]
        grating_id = f"wg{waveguide}_{side}"
        
        if 'gratings' not in block or grating_id not in block['gratings']:
            raise ValueError(f"Grating {grating_id} not found in block {block_id}")
        
        grating = block['gratings'][grating_id]
        local_pos = grating['position']
        
        return self.block_local_to_stage(block_id, local_pos[0], local_pos[1])
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_calibration_info(self) -> Dict:
        """
        Get current calibration information.
        
        Returns:
            dict with calibration state and parameters
        """
        return {
            'is_calibrated': self.is_calibrated,
            'angle_deg': self.angle_deg,
            'translation_um': self.translation.tolist(),
            'rotation_matrix': self.rotation_matrix.tolist(),
            'has_layout': self.layout is not None
        }


# ============================================================================
# TEST/EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Unified Coordinate Transform Module")
    print("=" * 70)
    
    # ========================================================================
    # Test 1: Basic calibration without layout
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Basic Two-Point Calibration")
    print("=" * 70)
    
    # Simulated scenario with known rotation and translation
    angle_sim = np.radians(3.0)
    cos_a = np.cos(angle_sim)
    sin_a = np.sin(angle_sim)
    
    def sim_measure(u, v):
        """Simulate stage measurement with rotation and offset."""
        Y = cos_a * u - sin_a * v + 50.0  # +50 µm offset
        Z = sin_a * u + cos_a * v + 30.0  # +30 µm offset
        return (Y, Z)
    
    # Two fiducial markers in µm
    design_fid1 = (5.0, 5.0)
    design_fid2 = (1395.0, 605.0)
    measured_fid1 = sim_measure(*design_fid1)
    measured_fid2 = sim_measure(*design_fid2)
    
    print(f"Design fid 1: {design_fid1} µm → Measured: {measured_fid1} µm")
    print(f"Design fid 2: {design_fid2} µm → Measured: {measured_fid2} µm")
    
    # Calibrate
    transform = CoordinateTransform()
    result = transform.calibrate(
        measured_points=[measured_fid1, measured_fid2],
        design_points=[design_fid1, design_fid2]
    )
    
    print(f"\nCalibration results:")
    print(f"  Method: {result['method']}")
    print(f"  Angle: {result['angle_deg']:.3f}° (expected: 3.0°)")
    print(f"  Translation: {result['translation_um']} µm")
    print(f"  Mean error: {result['mean_error_um']:.6f} µm")
    
    # Test round-trip
    test_point = (100.0, 200.0)
    stage_pos = transform.design_to_stage(*test_point)
    back = transform.stage_to_design(*stage_pos)
    error = np.sqrt((back[0] - test_point[0])**2 + (back[1] - test_point[1])**2)
    print(f"\nRound-trip test: {test_point} µm → {stage_pos} µm → {back} µm")
    print(f"  Error: {error:.9f} µm")
    
    # ========================================================================
    # Test 2: Layout-aware operations
    # ========================================================================
    try:
        from config.layout_config_generator_v2 import load_layout_config_v2
        
        print("\n" + "=" * 70)
        print("TEST 2: Layout-Aware Operations")
        print("=" * 70)
        
        layout = load_layout_config_v2("config/mock_layout.json")
        transform_layout = CoordinateTransform(layout)
        
        # Use ground truth transformation
        gt = layout['simulation_ground_truth']
        transform_layout.set_transformation(
            gt['rotation_deg'],
            tuple(gt['translation_um'])
        )
        
        print(f"Set transformation from ground truth:")
        print(f"  Rotation: {transform_layout.angle_deg}°")
        print(f"  Translation: {transform_layout.translation} µm")
        
        # Test block-aware conversion
        print(f"\nBlock 1 top-left fiducial:")
        tl_stage = transform_layout.get_fiducial_stage_position(1, 'top_left')
        print(f"  Stage position: {tl_stage} µm")
        
        print(f"\nBlock 10 WG25 left grating:")
        wg25_stage = transform_layout.get_grating_stage_position(10, 25, 'left')
        print(f"  Stage position: {wg25_stage} µm")
        
        # Test inverse (round-trip)
        print(f"\n  Testing inverse transformation...")
        u_local, v_local = transform_layout.stage_to_block_local(*wg25_stage, 10)
        print(f"  Stage → Local: ({u_local:.6f}, {v_local:.6f}) µm")
        
        # Get original local position for comparison
        grating = layout['blocks'][10]['gratings']['wg25_left']
        original_local = grating['position']
        print(f"  Original local: {original_local} µm")
        error = np.sqrt((u_local - original_local[0])**2 + (v_local - original_local[1])**2)
        print(f"  Round-trip error: {error:.9f} µm")
        
        print(f"\n✅ All tests complete!")
        
    except ImportError:
        print("\n⚠️  Skipping layout tests (layout_config_generator_v2 not available)")