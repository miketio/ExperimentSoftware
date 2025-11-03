# coordinate_transform.py
"""
2D coordinate transformation for sample alignment.
Handles rotation and translation from design coordinates to stage coordinates.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict


class CoordinateTransform:
    """
    2D coordinate transformation: design (u, v) ↔ stage (Y, Z).
    
    Design coordinates:
        - u: along waveguide length (10-190 µm)
        - v: across waveguides (~95-140 µm)
        - Units: micrometers
    
    Stage coordinates:
        - Y: corresponds to u direction
        - Z: corresponds to v direction
        - Units: nanometers
    
    Transformation: 
        [Y]   [cos(θ)  -sin(θ)] [u]   [Y₀]
        [Z] = [sin(θ)   cos(θ)] [v] + [Z₀]
    """
    
    def __init__(self):
        self.rotation_matrix = np.eye(2)
        self.translation = np.array([0.0, 0.0])
        self.is_calibrated = False
        self.angle_deg = 0.0
        self.scale_factor = 1000.0  # µm to nm conversion
        
    def calibrate(self, 
                  measured_points: List[Tuple[float, float]],
                  design_points: List[Tuple[float, float]]) -> Dict:
        """
        Calibrate transformation using measured fiducial positions.
        
        Args:
            measured_points: List of (Y, Z) in nanometers from stage
            design_points: List of (u, v) in micrometers from design
        
        Returns:
            dict with calibration results
        """
        if len(measured_points) != len(design_points):
            raise ValueError("Number of measured and design points must match")
        
        if len(measured_points) < 2:
            raise ValueError("Need at least 2 points for calibration")
        
        # Convert to numpy arrays
        measured = np.array(measured_points, dtype=float)  # nm
        design = np.array(design_points, dtype=float)  # µm
        
        # Convert design to nm
        design_nm = design      # already in nm
        measured_nm = measured  # already in nm
        
        if len(measured) == 2:
            # Two-point calibration (exact solution)
            result = self._calibrate_two_points(measured_nm, design_nm)
        else:
            # Least-squares for 3+ points
            result = self._calibrate_least_squares(measured_nm, design_nm)
        
        self.is_calibrated = True
        return result
    
    def _calibrate_two_points(self, measured: np.ndarray, design: np.ndarray) -> Dict:
        """
        Calibrate using exactly 2 points.
        
        Args:
            measured: 2x2 array of (Y, Z) in nm
            design: 2x2 array of (u, v) in nm
        
        Returns:
            Calibration results
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
            error = np.sqrt(np.linalg.norm(np.array(predicted) - measured[i]))
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
            Calibration results
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
            error = np.sqrt(np.linalg.norm(np.array(predicted) - measured[i]))
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
    
    def design_to_stage(self, u: float, v: float) -> Tuple[int, int]:
        """
        Convert design coordinates (µm) to stage coordinates (nm).
        
        Args:
            u, v: Design coordinates in micrometers
        
        Returns:
            (Y, Z): Stage coordinates in nanometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() first.")
        
        # Convert to nm
        design_point = np.array([u * self.scale_factor, v * self.scale_factor])
        
        # Apply transformation
        stage_point = self.rotation_matrix @ design_point + self.translation
        
        return (int(round(stage_point[0])), int(round(stage_point[1])))
    
    def stage_to_design(self, Y: int, Z: int) -> Tuple[float, float]:
        """
        Convert stage coordinates (nm) to design coordinates (µm).
        
        Args:
            Y, Z: Stage coordinates in nanometers
        
        Returns:
            (u, v): Design coordinates in micrometers
        """
        if not self.is_calibrated:
            raise RuntimeError("Transform not calibrated. Call calibrate() first.")
        
        stage_point = np.array([float(Y), float(Z)])
        
        # Inverse transformation
        design_point_nm = np.linalg.inv(self.rotation_matrix) @ (stage_point - self.translation)
        
        # Convert to µm
        u = design_point_nm[0] / self.scale_factor
        v = design_point_nm[1] / self.scale_factor
        
        return (u, v)
    
    def get_calibration_info(self) -> Dict:
        """Get current calibration information."""
        return {
            'is_calibrated': self.is_calibrated,
            'angle_deg': self.angle_deg,
            'translation_nm': self.translation.tolist(),
            'rotation_matrix': self.rotation_matrix.tolist()
        }


# Test/example usage
if __name__ == "__main__":
    print("Coordinate Transform Module")
    print("============================")
    
    # Example: Two fiducial markers
    # Design positions (µm)
    design_fid1 = (5.0, 5.0)      # Top-left of block 1
    design_fid2 = (1395.0, 605.0) # Bottom-right of block 20
    
    # Simulated measured positions (nm) with ~3° rotation and offset
    angle_sim = np.radians(3.0)
    cos_a = np.cos(angle_sim)
    sin_a = np.sin(angle_sim)
    
    # Apply simulated transform to get "measured" positions
    def sim_measure(u, v):
        """Simulate stage measurement with rotation and offset."""
        u_nm = u * 1000
        v_nm = v * 1000
        Y = cos_a * u_nm - sin_a * v_nm + 50000  # +50 µm offset
        Z = sin_a * u_nm + cos_a * v_nm + 30000  # +30 µm offset
        return (Y, Z)
    
    measured_fid1 = sim_measure(*design_fid1)
    measured_fid2 = sim_measure(*design_fid2)
    
    print(f"\nSimulated scenario:")
    print(f"Design fid 1: {design_fid1} µm")
    print(f"Measured fid 1: ({measured_fid1[0]:.0f}, {measured_fid1[1]:.0f}) nm")
    print(f"Design fid 2: {design_fid2} µm")
    print(f"Measured fid 2: ({measured_fid2[0]:.0f}, {measured_fid2[1]:.0f}) nm")
    
    # Calibrate transform
    transform = CoordinateTransform()
    result = transform.calibrate(
        measured_points=[measured_fid1, measured_fid2],
        design_points=[design_fid1, design_fid2]
    )
    
    print(f"\nCalibration results:")
    print(f"  Method: {result['method']}")
    print(f"  Angle: {result['angle_deg']:.3f}°")
    print(f"  Translation: ({result['translation_nm'][0]:.0f}, {result['translation_nm'][1]:.0f}) nm")
    print(f"  Mean error: {result['mean_error_nm']:.3f} nm")
    print(f"  Max error: {result['max_error_nm']:.3f} nm")
    
    # Test transformation
    print(f"\nTest transformations:")
    
    # Waveguide 25 left grating position
    wg25_design = (12.0, 117.6)  # µm (approximate from ASCII)
    wg25_stage = transform.design_to_stage(*wg25_design)
    print(f"WG25 left grating: {wg25_design} µm → ({wg25_stage[0]}, {wg25_stage[1]}) nm")
    
    # Verify inverse transform
    wg25_back = transform.stage_to_design(*wg25_stage)
    print(f"  Inverse check: ({wg25_stage[0]}, {wg25_stage[1]}) nm → ({wg25_back[0]:.3f}, {wg25_back[1]:.3f}) µm")
    print(f"  Error: {abs(wg25_back[0] - wg25_design[0]):.6f} µm")