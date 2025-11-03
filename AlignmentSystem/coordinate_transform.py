"""
Unified 2D coordinate transformation for sample alignment.
Handles rotation, translation, calibration, and layout-aware conversions.

Coordinate Flow:
1. Local block coords (u, v in µm) - relative to block bottom-left
2. Global design coords (u, v in µm) - unrotated, origin at Block 1 bottom-left
3. Stage coords (Y, Z in nm) - after rotation + translation

Transformation: 
    [Y]   [cos(θ)  -sin(θ)] [u*1000]   [Y₀]
    [Z] = [sin(θ)   cos(θ)] [v*1000] + [Z₀]
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
        - Units: nanometers
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
        self.scale_factor = 1000.0  # µm to nm conversion
        
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
            measured_points: List of (Y, Z) in nanometers from stage
            design_points: List of (u, v) in micrometers from design
        
        Returns:
            dict with calibration results including:
                - method: 'two_point' or 'least_squares'
                - angle_deg: rotation angle
                - translation_nm: [Y₀, Z₀] translation
                - mean_error_nm: mean residual error
                - max_error_nm: maximum residual error
                - num_points: number of calibration points
        """
        if len(measured_points) != len(design_points):
            raise ValueError("Number of measured and design points must match")
        
        if len(measured_points) < 2:
            raise ValueError("Need at least 2 points for calibration")
        
        # Convert to numpy arrays
        measured = np.array(measured_points, dtype=float)  # nm
        design = np.array(design_points, dtype=float)  # µm
        
        # Convert design to nm
        design_nm = design * self.scale_factor
        
        if len(measured) == 2:
            # Two-point calibration (exact solution)
            result = self._calibrate_two_points(measured, design_nm)
        else:
            # Least-squares for 3+ points
            result = self._calibrate_least_squares(measured, design_nm)
        
        self.is_calibrated = True
        return result
    
    def _calibrate_two_points(self, measured: np.ndarray, design: np.ndarray) -> Dict:
        """
        Calibrate using exactly 2 points.
        
        Args:
            measured: 2x2 array of (Y, Z) in nm
            design: 2x2 array of (u, v) in nm
        
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
            'translation_nm': self.translation.tolist(),
            'mean_error_nm': float(np.mean(errors)),
            'max_error_nm': float(np.max(errors)),
            'num_points': 2
        }
    
    def _calibrate_least_squares(self, measured: np.ndarray, design: np.ndarray) -> Dict:
        """
        Calibrate using least-squares fit for 3+ points.
        
        Args:
            measured: Nx2 array of (Y, Z) in nm
            design: Nx2 array of (u, v) in nm
        
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
            'translation_nm': self.translation.tolist(),
            'mean_error_nm': float(np.mean(errors)),
            'max_error_nm': float(np.max(errors)),
            'std_error_nm': float(np.std(errors)),
            'num_points': n
        }
    
    def set_transformation(self, rotation_deg: float, translation_nm: Tuple[float, float]):
        """
        Manually set rotation and translation (alternative to calibration).
        
        Args:
            rotation_deg: Rotation angle in degrees
            translation_nm: (Y₀, Z₀) translation in nanometers
        """
        self.angle_deg = rotation_deg
        angle_rad = np.radians(rotation_deg)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        self.rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        self.translation = np.array(translation_nm, dtype=float)
        self.is_calibrated = True
    
    # ========================================================================
    # CORE TRANSFORMATION METHODS
    # ========================================================================
    
    def design_to_stage(self, u: float, v: float) -> Tuple[int, int]:
        """
        Convert global design coordinates (µm) to stage coordinates (nm).
        
        Args:
            u, v: Global design coordinates in micrometers
        
        Returns:
            (Y, Z): Stage coordinates in nanometers (rounded to integers)
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        # Convert to nm
        design_point = np.array([u * self.scale_factor, v * self.scale_factor])
        
        # Apply transformation: R @ design + t
        stage_point = self.rotation_matrix @ design_point + self.translation
        
        return (int(round(stage_point[0])), int(round(stage_point[1])))
    
    def stage_to_design(self, Y: float, Z: float) -> Tuple[float, float]:
        """
        Convert stage coordinates (nm) to global design coordinates (µm).
        
        Args:
            Y, Z: Stage coordinates in nanometers
        
        Returns:
            (u, v): Global design coordinates in micrometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        stage_point = np.array([float(Y), float(Z)])
        
        # Inverse transformation: R⁻¹ @ (stage - t)
        design_point_nm = np.linalg.inv(self.rotation_matrix) @ (stage_point - self.translation)
        
        # Convert to µm
        u = design_point_nm[0] / self.scale_factor
        v = design_point_nm[1] / self.scale_factor
        
        return (u, v)
    
    # ========================================================================
    # LAYOUT-AWARE METHODS (require layout in __init__)
    # ========================================================================
    
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
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated.")
        
        # Get block's design position (center of block)
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        
        # Get block size to convert center → bottom-left
        block_size = self.layout['block_layout']['block_size']  # 200 µm
        block_u_bottom_left = block_u_center - block_size / 2
        block_v_bottom_left = block_v_center - block_size / 2
        
        # Convert to global design coordinates
        global_u = block_u_bottom_left + u_local
        global_v = block_v_bottom_left + v_local
        
        # Convert to stage coordinates
        return self.design_to_stage(global_u, global_v)
    
    def stage_to_block_local(
        self,
        Y: float,
        Z: float,
        block_id: int
    ) -> Tuple[float, float]:
        """
        Convert stage coordinates to local block coordinates.
        
        Args:
            Y, Z: Stage coordinates in nanometers
            block_id: Block ID
        
        Returns:
            (u_local, v_local): Local coordinates in µm (relative to block bottom-left)
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        
        # Convert to global design coords
        u_global, v_global = self.stage_to_design(Y, Z)
        
        # Get block's center position and size
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        block_size = self.layout['block_layout']['block_size']
        
        # Convert center → bottom-left
        block_u_bottom_left = block_u_center - block_size / 2
        block_v_bottom_left = block_v_center - block_size / 2
        
        # Subtract block bottom-left to get local coords
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
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        
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
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
            ValueError: If grating not found in block
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        
        block = self.blocks[block_id]
        grating_id = f"wg{waveguide}_{side}"
        
        if grating_id not in block['gratings']:
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
            'translation_nm': self.translation.tolist(),
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
        u_nm = u * 1000
        v_nm = v * 1000
        Y = cos_a * u_nm - sin_a * v_nm + 50000  # +50 µm offset
        Z = sin_a * u_nm + cos_a * v_nm + 30000  # +30 µm offset
        return (Y, Z)
    
    # Two fiducial markers
    design_fid1 = (5.0, 5.0)
    design_fid2 = (1395.0, 605.0)
    measured_fid1 = sim_measure(*design_fid1)
    measured_fid2 = sim_measure(*design_fid2)
    
    print(f"Design fid 1: {design_fid1} µm → Measured: {measured_fid1} nm")
    print(f"Design fid 2: {design_fid2} µm → Measured: {measured_fid2} nm")
    
    # Calibrate
    transform = CoordinateTransform()
    result = transform.calibrate(
        measured_points=[measured_fid1, measured_fid2],
        design_points=[design_fid1, design_fid2]
    )
    
    print(f"\nCalibration results:")
    print(f"  Method: {result['method']}")
    print(f"  Angle: {result['angle_deg']:.3f}° (expected: 3.0°)")
    print(f"  Translation: {result['translation_nm']} nm")
    print(f"  Mean error: {result['mean_error_nm']:.3f} nm")
    
    # Test round-trip
    test_point = (100.0, 200.0)
    stage_pos = transform.design_to_stage(*test_point)
    back = transform.stage_to_design(*stage_pos)
    error = np.sqrt((back[0] - test_point[0])**2 + (back[1] - test_point[1])**2)
    print(f"\nRound-trip test: {test_point} µm → {stage_pos} nm → {back} µm")
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
            tuple(gt['translation_nm'])
        )
        
        print(f"Set transformation from ground truth:")
        print(f"  Rotation: {transform_layout.angle_deg}°")
        print(f"  Translation: {transform_layout.translation} nm")
        
        # Test block-aware conversion
        print(f"\nBlock 1 top-left fiducial:")
        tl_stage = transform_layout.get_fiducial_stage_position(1, 'top_left')
        print(f"  Stage position: {tl_stage} nm")
        
        print(f"\nBlock 10 WG25 left grating:")
        wg25_stage = transform_layout.get_grating_stage_position(10, 25, 'left')
        print(f"  Stage position: {wg25_stage} nm")
        
        # Test inverse
        u_local, v_local = transform_layout.stage_to_block_local(*wg25_stage, 10)
        print(f"  Inverse (local): ({u_local:.3f}, {v_local:.3f}) µm")
        
        print(f"\n✅ All tests complete!")
        
    except ImportError:
        print("\n⚠️  Skipping layout tests (layout_config_generator_v2 not available)")
    
    """
Unified 2D coordinate transformation for sample alignment.
Handles rotation, translation, calibration, and layout-aware conversions.

Coordinate Flow:
1. Local block coords (u, v in µm) - relative to block bottom-left
2. Global design coords (u, v in µm) - unrotated, origin at Block 1 bottom-left
3. Stage coords (Y, Z in nm) - after rotation + translation

Transformation: 
    [Y]   [cos(θ)  -sin(θ)] [u*1000]   [Y₀]
    [Z] = [sin(θ)   cos(θ)] [v*1000] + [Z₀]
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
        - Units: nanometers
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
        self.scale_factor = 1000.0  # µm to nm conversion
        
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
            measured_points: List of (Y, Z) in nanometers from stage
            design_points: List of (u, v) in micrometers from design
        
        Returns:
            dict with calibration results including:
                - method: 'two_point' or 'least_squares'
                - angle_deg: rotation angle
                - translation_nm: [Y₀, Z₀] translation
                - mean_error_nm: mean residual error
                - max_error_nm: maximum residual error
                - num_points: number of calibration points
        """
        if len(measured_points) != len(design_points):
            raise ValueError("Number of measured and design points must match")
        
        if len(measured_points) < 2:
            raise ValueError("Need at least 2 points for calibration")
        
        # Convert to numpy arrays
        measured = np.array(measured_points, dtype=float)  # nm
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
            measured: 2x2 array of (Y, Z) in nm
            design: 2x2 array of (u, v) in nm
        
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
            'translation_nm': self.translation.tolist(),
            'mean_error_nm': float(np.mean(errors)),
            'max_error_nm': float(np.max(errors)),
            'num_points': 2
        }
    
    def _calibrate_least_squares(self, measured: np.ndarray, design: np.ndarray) -> Dict:
        """
        Calibrate using least-squares fit for 3+ points.
        
        Args:
            measured: Nx2 array of (Y, Z) in nm
            design: Nx2 array of (u, v) in nm
        
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
            'translation_nm': self.translation.tolist(),
            'mean_error_nm': float(np.mean(errors)),
            'max_error_nm': float(np.max(errors)),
            'std_error_nm': float(np.std(errors)),
            'num_points': n
        }
    
    def set_transformation(self, rotation_deg: float, translation_nm: Tuple[float, float]):
        """
        Manually set rotation and translation (alternative to calibration).
        
        Args:
            rotation_deg: Rotation angle in degrees
            translation_nm: (Y₀, Z₀) translation in nanometers
        """
        self.angle_deg = rotation_deg
        angle_rad = np.radians(rotation_deg)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        self.rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        self.translation = np.array(translation_nm, dtype=float)
        self.is_calibrated = True
    
    # ========================================================================
    # CORE TRANSFORMATION METHODS
    # ========================================================================
    
    def design_to_stage(self, u: float, v: float) -> Tuple[int, int]:
        """
        Convert global design coordinates (µm) to stage coordinates (nm).
        
        Args:
            u, v: Global design coordinates in micrometers
        
        Returns:
            (Y, Z): Stage coordinates in nanometers (rounded to integers)
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        # Convert to nm
        design_point = np.array([u * self.scale_factor, v * self.scale_factor])
        
        # Apply transformation: R @ design + t
        stage_point = self.rotation_matrix @ design_point + self.translation
        
        return (int(round(stage_point[0])), int(round(stage_point[1])))
    
    def stage_to_design(self, Y: float, Z: float) -> Tuple[float, float]:
        """
        Convert stage coordinates (nm) to global design coordinates (µm).
        
        Args:
            Y, Z: Stage coordinates in nanometers
        
        Returns:
            (u, v): Global design coordinates in micrometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() or set_transformation() first.")
        
        stage_point = np.array([float(Y), float(Z)])
        
        # Inverse transformation: R⁻¹ @ (stage - t)
        design_point_nm = np.linalg.inv(self.rotation_matrix) @ (stage_point - self.translation)
        
        # Convert to µm
        u = design_point_nm[0] / self.scale_factor
        v = design_point_nm[1] / self.scale_factor
        
        return (u, v)
    
    # ========================================================================
    # LAYOUT-AWARE METHODS (require layout in __init__)
    # ========================================================================
    
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
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated.")
        
        # Get block's design position (center of block)
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        
        # Get block size to convert center → bottom-left
        block_size = self.layout['block_layout']['block_size']  # 200 µm
        block_u_bottom_left = block_u_center - block_size / 2
        block_v_bottom_left = block_v_center - block_size / 2
        
        # Convert to global design coordinates
        global_u = block_u_bottom_left + u_local
        global_v = block_v_bottom_left + v_local
        
        # Convert to stage coordinates
        return self.design_to_stage(global_u, global_v)
    
    def stage_to_block_local(
        self,
        Y: float,
        Z: float,
        block_id: int
    ) -> Tuple[float, float]:
        """
        Convert stage coordinates to local block coordinates.
        
        Args:
            Y, Z: Stage coordinates in nanometers
            block_id: Block ID
        
        Returns:
            (u_local, v_local): Local coordinates in µm (relative to block bottom-left)
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        
        # Convert to global design coords
        u_global, v_global = self.stage_to_design(Y, Z)
        
        # Get block's center position and size
        block = self.blocks[block_id]
        block_u_center, block_v_center = block['design_position']
        block_size = self.layout['block_layout']['block_size']
        
        # Convert center → bottom-left
        block_u_bottom_left = block_u_center - block_size / 2
        block_v_bottom_left = block_v_center - block_size / 2
        
        # Subtract block bottom-left to get local coords
        u_local = u_global - block_u_bottom_left
        v_local = v_global - block_v_bottom_left
        
        return (u_local, v_local)
    
    def get_stage_position(
            self,
            block_id: int,
            u_local: float,
            v_local: float
        ) -> Tuple[int, int]:
            """
            Get stage position from block-local coordinates.
            Alias for block_local_to_stage for convenience.
            
            Args:
                block_id: Block ID
                u_local: Local u coordinate in µm (relative to block bottom-left)
                v_local: Local v coordinate in µm (relative to block bottom-left)
            
            Returns:
                (Y, Z): Stage coordinates in nm
            """
            return self.block_local_to_stage(block_id, u_local, v_local)
    
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
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        
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
        
        Raises:
            RuntimeError: If layout not provided or transform not calibrated
            ValueError: If grating not found in block
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Pass layout dict to __init__()")
        
        block = self.blocks[block_id]
        grating_id = f"wg{waveguide}_{side}"
        
        if grating_id not in block['gratings']:
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
            'translation_nm': self.translation.tolist(),
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
        u_nm = u * 1000
        v_nm = v * 1000
        Y = cos_a * u_nm - sin_a * v_nm + 50000  # +50 µm offset
        Z = sin_a * u_nm + cos_a * v_nm + 30000  # +30 µm offset
        return (Y, Z)
    
    # Two fiducial markers
    design_fid1 = (5.0, 5.0)
    design_fid2 = (1395.0, 605.0)
    measured_fid1 = sim_measure(*design_fid1)
    measured_fid2 = sim_measure(*design_fid2)
    
    print(f"Design fid 1: {design_fid1} µm → Measured: {measured_fid1} nm")
    print(f"Design fid 2: {design_fid2} µm → Measured: {measured_fid2} nm")
    
    # Calibrate
    transform = CoordinateTransform()
    result = transform.calibrate(
        measured_points=[measured_fid1, measured_fid2],
        design_points=[design_fid1, design_fid2]
    )
    
    print(f"\nCalibration results:")
    print(f"  Method: {result['method']}")
    print(f"  Angle: {result['angle_deg']:.3f}° (expected: 3.0°)")
    print(f"  Translation: {result['translation_nm']} nm")
    print(f"  Mean error: {result['mean_error_nm']:.3f} nm")
    
    # Test round-trip
    test_point = (100.0, 200.0)
    stage_pos = transform.design_to_stage(*test_point)
    back = transform.stage_to_design(*stage_pos)
    error = np.sqrt((back[0] - test_point[0])**2 + (back[1] - test_point[1])**2)
    print(f"\nRound-trip test: {test_point} µm → {stage_pos} nm → {back} µm")
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
            tuple(gt['translation_nm'])
        )
        
        print(f"Set transformation from ground truth:")
        print(f"  Rotation: {transform_layout.angle_deg}°")
        print(f"  Translation: {transform_layout.translation} nm")
        
        # Test block-aware conversion
        print(f"\nBlock 1 top-left fiducial:")
        tl_stage = transform_layout.get_fiducial_stage_position(1, 'top_left')
        print(f"  Stage position: {tl_stage} nm")
        
        print(f"\nBlock 10 WG25 left grating:")
        wg25_stage = transform_layout.get_grating_stage_position(10, 25, 'left')
        print(f"  Stage position: {wg25_stage} nm")
        
        # Test inverse
        u_local, v_local = transform_layout.stage_to_block_local(*wg25_stage, 10)
        print(f"  Inverse (local): ({u_local:.3f}, {v_local:.3f}) µm")
        
        print(f"\n✅ All tests complete!")
        
        # ====================================================================
        # Test 3: Visual Test with Camera Simulation
        # ====================================================================
        print("\n" + "=" * 70)
        print("TEST 3: Visual Test with Simulated Camera")
        print("=" * 70)
        
        try:
            import matplotlib.pyplot as plt
            from Testing.test_alignment_cv_calibration import AlignmentTester
            from config.layout_config_generator_v2 import plot_layout_v2
            
            # Plot layout
            plot_layout_v2(layout, "config/mock_layout.png")
            print("Layout plot saved to: config/mock_layout.png")
            
            # Set up alignment tester with camera
            tester = AlignmentTester()
            tester.setup()
            
            # Acquire simulated image
            img = tester.camera.acquire_single_image()
            print(f"Acquired simulated camera image")
            print(f"  Image shape: {img.shape}, dtype={img.dtype}")
            
            # Display image
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray', origin='lower')
            plt.title("Simulated Fiducial Markers (Top-Left should look like 'Γ')")
            plt.xlabel("Y (pixels)")
            plt.ylabel("Z (pixels)")
            plt.tight_layout()
            plt.savefig("config/simulated_fiducials.png", dpi=150)
            print("Camera image saved to: config/simulated_fiducials.png")
            plt.show()
            
            print("\n✅ Visual test complete! Check the image windows.")
            
        except ImportError as e:
            print(f"\n⚠️  Skipping visual test: {e}")
        
        print(f"\n✅ All tests complete!")
        
    except ImportError:
        print("\n⚠️  Skipping layout tests (layout_config_generator_v2 not available)")