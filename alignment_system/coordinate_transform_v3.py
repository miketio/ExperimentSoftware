"""
coordinate_transform_v4.py

Unified 2D coordinate transformation supporting both CameraLayout and RuntimeLayout.

New in v4:
- Works with both CameraLayout (simulation) and RuntimeLayout (real experiments)
- Reads calibration from RuntimeLayout.measured_global_transform
- Applies block-level calibrations from RuntimeLayout.measured_block_transforms
- Handles incomplete calibration states gracefully
- Auto-syncs with RuntimeLayout when calibration updates

Features:
- Global transform: rotation + translation (sample-level)
- Per-block corrections: additional rotation + translation (fabrication + measurement errors)
- Layout-aware conversions with block ID awareness
- All internal units: micrometers (µm)

Usage:
    # Simulation with ground truth
    >>> camera_layout = CameraLayout.from_json_file("config/mock_layout.json")
    >>> ct = CoordinateTransformV4(camera_layout)
    >>> ct.use_ground_truth()  # Use simulation ground truth
    
    # Real experiment with progressive calibration
    >>> runtime_layout = RuntimeLayout.from_json_file("config/mock_layout.json")
    >>> ct = CoordinateTransformV4(runtime_layout)
    >>> # Initially not calibrated - will raise errors on predictions
    >>> # After Stage 1 calibration:
    >>> ct.sync_with_runtime()  # Read calibration from RuntimeLayout
    >>> Y, Z = ct.block_local_to_stage(block_id, u, v)  # Now works
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass
import math

from config.layout_models import (
    CameraLayout, RuntimeLayout, 
    GroundTruth, BlockFabricationError,
    MeasuredTransform
)


@dataclass
class TransformState:
    """Current transformation state (rotation + translation)."""
    rotation_deg: float
    translation_um: Tuple[float, float]
    is_set: bool = False
    source: str = "unset"  # "ground_truth", "runtime_global", "manual"


@dataclass
class BlockCorrection:
    """Per-block correction (additional rotation + translation)."""
    rotation_deg: float
    translation_um: Tuple[float, float]
    source: str = "unset"  # "fabrication_error", "runtime_calibration", "manual"


class CoordinateTransformV3:
    """
    Coordinate transform supporting both CameraLayout and RuntimeLayout.
    
    Handles:
    1. Global transform (sample-level rotation + translation)
    2. Per-block corrections (fabrication + calibration)
    3. Progressive calibration (works before/during/after calibration)
    
    All coordinates in micrometers (µm).
    """

    def __init__(self, layout = None):
        """
        Initialize transform with optional layout.
        
        Args:
            layout: CameraLayout (simulation) or RuntimeLayout (real experiment)
        """
        self.layout = layout
        self.layout_type = self._detect_layout_type(layout)
        
        # Block size (needed for per-block transforms)
        self.block_size: Optional[float] = None
        
        # Global transform state
        self.global_transform = TransformState(
            rotation_deg=0.0,
            translation_um=(0.0, 0.0),
            is_set=False,
            source="unset"
        )
        
        # Per-block corrections: {block_id: BlockCorrection}
        self.block_corrections: Dict[int, BlockCorrection] = {}
        
        # Cached transform matrices (updated when global_transform changes)
        self._rotation_matrix = np.eye(2, dtype=float)
        self._translation_vector = np.zeros(2, dtype=float)
        self._matrix_valid = False
        
        # Load layout data if provided
        if layout is not None:
            self._load_layout(layout)
    
    # ========================================================================
    # LAYOUT LOADING AND TYPE DETECTION
    # ========================================================================
    
    def _detect_layout_type(self, layout) -> str:
        """Detect whether layout is CameraLayout or RuntimeLayout."""
        if layout is None:
            return "none"
        
        # Check for ground_truth attribute (CameraLayout has it, RuntimeLayout doesn't)
        if hasattr(layout, "ground_truth"):
            return "camera"
        elif hasattr(layout, "measured_global_transform"):
            return "runtime"
        else:
            return "unknown"
    
    def _load_layout(self, layout):
        """Load layout data (block_size, and prepare for calibration reading)."""
        self.layout = layout
        self.layout_type = self._detect_layout_type(layout)
        
        # Get block size from layout
        if hasattr(layout, "block_layout"):
            self.block_size = float(layout.block_layout.block_size)
        
        # If CameraLayout, optionally load ground truth
        if self.layout_type == "camera":
            # Don't auto-load ground truth - user must call use_ground_truth()
            # This keeps behavior explicit
            pass
        
        # If RuntimeLayout, check if already calibrated
        elif self.layout_type == "runtime":
            # Try to sync with any existing calibration
            if layout.is_globally_calibrated():
                self.sync_with_runtime()
    
    # ========================================================================
    # CALIBRATION SOURCES
    # ========================================================================
    
    def calibrate(self,
                  measured_points: List[Tuple[float, float]],
                  design_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Calibrate global transform from measured (stage Y,Z) <-> design (u,v) correspondences.

        measured_points: list of (Y, Z) in µm (stage)
        design_points:   list of (u, v) in µm (global design coordinates)

        Returns:
            dict summarizing calibration results
        """
        if len(measured_points) != len(design_points):
            raise ValueError("measured_points and design_points must have same length")
        n = len(measured_points)
        measured = np.asarray(measured_points, dtype=float)
        design = np.asarray(design_points, dtype=float)

        if n < 2:
            raise ValueError("Need at least 2 points for calibration")

        if n == 2:
            res = self._calibrate_two_points(measured, design)
        else:
            res = self._calibrate_least_squares(measured, design)

        # store
        self.is_calibrated = True

            # Store the calibration result
        self.set_global_transform(
            rotation_deg=res['angle_deg'],
            translation_um=res['translation_um'],
            source='calibrated'
        )
        return res

    def _calibrate_two_points(self, measured: np.ndarray, design: np.ndarray) -> Dict[str, Any]:
        """
        Two-point calibration: compute rotation that maps design vector -> measured vector,
        then compute translation from one point.
        """
        v_design = design[1] - design[0]
        v_meas = measured[1] - measured[0]

        angle_design = math.atan2(v_design[1], v_design[0])
        angle_meas = math.atan2(v_meas[1], v_meas[0])
        angle_diff = angle_meas - angle_design

        self.angle_deg = float(np.degrees(angle_diff))
        cos_a = math.cos(angle_diff)
        sin_a = math.sin(angle_diff)
        self.rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)

        # translation = measured_point0 - R * design_point0
        rotated_design0 = self.rotation_matrix @ design[0]
        self.translation = measured[0] - rotated_design0

        # residuals
        errs = []
        for i in range(2):
            pred = (self.rotation_matrix @ design[i]) + self.translation
            errs.append(float(np.linalg.norm(pred - measured[i])))

        result = {
            "method": "two_point",
            "angle_deg": float(self.angle_deg),
            "translation_um": (float(self.translation[0]), float(self.translation[1])),
            "mean_error_um": float(np.mean(errs)),
            "max_error_um": float(np.max(errs)),
            "num_points": 2
        }
        return result

    def _calibrate_least_squares(self, measured: np.ndarray, design: np.ndarray) -> Dict[str, Any]:
        """
        Kabsch-like solution: find rotation + translation minimizing |R*design + t - measured|.
        """
        n = measured.shape[0]
        design_center = np.mean(design, axis=0)
        meas_center = np.mean(measured, axis=0)

        design_c = design - design_center
        meas_c = measured - meas_center

        H = design_c.T @ meas_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # fix possible reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        self.rotation_matrix = R
        self.angle_deg = float(np.degrees(math.atan2(R[1, 0], R[0, 0])))

        rotated_center = self.rotation_matrix @ design_center
        self.translation = meas_center - rotated_center

        # residuals
        errs = []
        for i in range(n):
            pred = (self.rotation_matrix @ design[i]) + self.translation
            errs.append(float(np.linalg.norm(pred - measured[i])))

        result = {
            "method": "least_squares",
            "angle_deg": float(self.angle_deg),
            "translation_um": (float(self.translation[0]), float(self.translation[1])),
            "mean_error_um": float(np.mean(errs)),
            "max_error_um": float(np.max(errs)),
            "std_error_um": float(np.std(errs)),
            "num_points": n
        }
        return result
    
    def use_ground_truth(self):
        """
        Use ground truth from CameraLayout (simulation mode).
        
        Only works if layout is CameraLayout.
        
        Raises:
            RuntimeError: If not using CameraLayout or no ground truth available
        """
        if self.layout_type != "camera":
            raise RuntimeError(
                "use_ground_truth() requires CameraLayout. "
                f"Current layout type: {self.layout_type}"
            )
        
        if not hasattr(self.layout, "ground_truth"):
            raise RuntimeError("CameraLayout has no ground_truth attribute")
        
        gt = self.layout.ground_truth
        
        # Set global transform from ground truth
        self.set_global_transform(
            rotation_deg=gt.rotation_deg,
            translation_um=(gt.translation_um.u, gt.translation_um.v),
            source="ground_truth"
        )
        
        # Load per-block fabrication errors
        self.block_corrections.clear()
        if hasattr(gt, "block_fabrication_errors") and gt.block_fabrication_errors:
            for block_id, err in gt.block_fabrication_errors.items():
                self.block_corrections[int(block_id)] = BlockCorrection(
                    rotation_deg=float(err.rotation_deg),
                    translation_um=(float(err.translation_um.u), float(err.translation_um.v)),
                    source="fabrication_error"
                )
        
        print(f"✓ Loaded ground truth: rot={gt.rotation_deg:.6f}°, "
              f"trans={gt.translation_um.to_tuple()}, "
              f"block_errors={len(self.block_corrections)}")
    
    def sync_with_runtime(self):
        """
        Sync with RuntimeLayout calibration data.
        
        Reads:
        - Global transform from measured_global_transform
        - Block corrections from measured_block_transforms
        
        Only works if layout is RuntimeLayout.
        
        Raises:
            RuntimeError: If not using RuntimeLayout
        """
        if self.layout_type != "runtime":
            raise RuntimeError(
                "sync_with_runtime() requires RuntimeLayout. "
                f"Current layout type: {self.layout_type}"
            )
        
        # Read global calibration
        if self.layout.is_globally_calibrated():
            global_trans = self.layout.get_global_transform()
            self.set_global_transform(
                rotation_deg=global_trans.rotation_deg,
                translation_um=(global_trans.translation_um.u, global_trans.translation_um.v),
                source="runtime_global"
            )
        else:
            # Clear global transform if not calibrated
            self.global_transform.is_set = False
            self.global_transform.source = "unset"
            self._matrix_valid = False
        
        # Read block-level calibrations
        # Note: These are ADDITIONAL corrections beyond global + fabrication
        calibrated_blocks = self.layout.get_calibrated_blocks()
        for block_id in calibrated_blocks:
            block_trans = self.layout.get_block_transform(block_id)
            
            # Store as block correction
            # In RuntimeLayout, these represent the correction needed
            # beyond the global transform (includes fabrication + measurement errors)
            self.block_corrections[block_id] = BlockCorrection(
                rotation_deg=float(block_trans.rotation_deg),
                translation_um=(float(block_trans.translation_um.u), float(block_trans.translation_um.v)),
                source="runtime_calibration"
            )
        
        if self.global_transform.is_set:
            print(f"✓ Synced with RuntimeLayout: "
                  f"global_calibrated=True, "
                  f"block_calibrations={len(calibrated_blocks)}")
        else:
            print(f"✓ Synced with RuntimeLayout: global_calibrated=False")
    
    def set_global_transform(self, 
                           rotation_deg: float, 
                           translation_um: Tuple[float, float],
                           source: str = "manual"):
        """
        Manually set global transformation.
        
        Args:
            rotation_deg: Rotation angle in degrees
            translation_um: (Y, Z) translation in µm
            source: Source identifier ("manual", "ground_truth", "runtime_global")
        """
        self.global_transform.rotation_deg = float(rotation_deg)
        self.global_transform.translation_um = (float(translation_um[0]), float(translation_um[1]))
        self.global_transform.is_set = True
        self.global_transform.source = source
        
        # Invalidate cached matrices
        self._matrix_valid = False
        self._update_matrices()
    
    def set_block_correction(self,
                           block_id: int,
                           rotation_deg: float,
                           translation_um: Tuple[float, float],
                           source: str = "manual"):
        """
        Manually set per-block correction.
        
        Args:
            block_id: Block identifier
            rotation_deg: Additional rotation in degrees
            translation_um: Additional (u, v) translation in µm
            source: Source identifier
        """
        self.block_corrections[int(block_id)] = BlockCorrection(
            rotation_deg=float(rotation_deg),
            translation_um=(float(translation_um[0]), float(translation_um[1])),
            source=source
        )
    
    def _update_matrices(self):
        """Update cached rotation matrix and translation vector."""
        if not self.global_transform.is_set:
            self._rotation_matrix = np.eye(2, dtype=float)
            self._translation_vector = np.zeros(2, dtype=float)
            self._matrix_valid = False
            return
        
        angle_rad = np.radians(self.global_transform.rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        self._rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=float)
        
        self._translation_vector = np.array([
            self.global_transform.translation_um[0],
            self.global_transform.translation_um[1]
        ], dtype=float)
        
        self._matrix_valid = True
    
    # ========================================================================
    # CORE TRANSFORMATIONS (Global design <-> Stage)
    # ========================================================================
    
    def design_to_stage(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert global design coordinates (u, v) -> stage (Y, Z).
        
        Applies only global rotation + translation (no block corrections).
        
        Args:
            u, v: Design coordinates in µm
        
        Returns:
            (Y, Z): Stage coordinates in µm
        
        Raises:
            RuntimeError: If global transform not set
        """
        if not self.global_transform.is_set:
            raise RuntimeError(
                "Global transform not set. Call use_ground_truth(), "
                "sync_with_runtime(), or set_global_transform() first."
            )
        
        if not self._matrix_valid:
            self._update_matrices()
        
        pt = np.array([float(u), float(v)], dtype=float)
        out = (self._rotation_matrix @ pt) + self._translation_vector
        return float(out[0]), float(out[1])
    
    def stage_to_design(self, Y: float, Z: float) -> Tuple[float, float]:
        """
        Convert stage coordinates (Y, Z) -> global design (u, v).
        
        Inverse of design_to_stage().
        
        Args:
            Y, Z: Stage coordinates in µm
        
        Returns:
            (u, v): Design coordinates in µm
        
        Raises:
            RuntimeError: If global transform not set
        """
        if not self.global_transform.is_set:
            raise RuntimeError(
                "Global transform not set. Call use_ground_truth(), "
                "sync_with_runtime(), or set_global_transform() first."
            )
        
        if not self._matrix_valid:
            self._update_matrices()
        
        pt = np.array([float(Y), float(Z)], dtype=float)
        # Inverse: R^T @ (pt - t)
        local = self._rotation_matrix.T @ (pt - self._translation_vector)
        return float(local[0]), float(local[1])
    
    # ========================================================================
    # BLOCK-AWARE TRANSFORMATIONS
    # ========================================================================
    
    def _get_block_correction(self, block_id: int) -> BlockCorrection:
        """
        Get block correction (returns zero correction if not defined).
        
        Args:
            block_id: Block identifier
        
        Returns:
            BlockCorrection (may be zero if not defined)
        """
        if int(block_id) in self.block_corrections:
            return self.block_corrections[int(block_id)]
        else:
            # Return zero correction
            return BlockCorrection(
                rotation_deg=0.0,
                translation_um=(0.0, 0.0),
                source="none"
            )
    
    def _apply_block_correction(self, 
                               block_id: int, 
                               u_local: float, 
                               v_local: float) -> Tuple[float, float]:
        """
        Apply per-block correction to local coordinates.
        
        Rotation is around block center (block_size/2, block_size/2).
        
        Args:
            block_id: Block identifier
            u_local, v_local: Local coordinates (relative to block bottom-left)
        
        Returns:
            (u_corrected, v_corrected): Corrected local coordinates
        """
        correction = self._get_block_correction(block_id)
        
        # Early exit if no correction
        if abs(correction.rotation_deg) < 1e-12 and \
           abs(correction.translation_um[0]) < 1e-12 and \
           abs(correction.translation_um[1]) < 1e-12:
            return float(u_local), float(v_local)
        
        if self.block_size is None:
            raise RuntimeError(
                "block_size unknown. Initialize with a layout that has block_layout.block_size"
            )
        
        # Rotate around block center
        center = np.array([self.block_size / 2.0, self.block_size / 2.0], dtype=float)
        pt = np.array([float(u_local), float(v_local)], dtype=float)
        
        # Apply rotation
        angle_rad = np.radians(correction.rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
        
        rotated = (R @ (pt - center)) + center
        
        # Apply translation
        corrected = rotated + np.array([
            correction.translation_um[0],
            correction.translation_um[1]
        ], dtype=float)
        
        return float(corrected[0]), float(corrected[1])
    
    def _remove_block_correction(self,
                                block_id: int,
                                u_corrected: float,
                                v_corrected: float) -> Tuple[float, float]:
        """
        Remove per-block correction (inverse of _apply_block_correction).
        
        Args:
            block_id: Block identifier
            u_corrected, v_corrected: Corrected local coordinates
        
        Returns:
            (u_local, v_local): Ideal local coordinates
        """
        correction = self._get_block_correction(block_id)
        
        # Early exit if no correction
        if abs(correction.rotation_deg) < 1e-12 and \
           abs(correction.translation_um[0]) < 1e-12 and \
           abs(correction.translation_um[1]) < 1e-12:
            return float(u_corrected), float(v_corrected)
        
        if self.block_size is None:
            raise RuntimeError(
                "block_size unknown. Initialize with a layout that has block_layout.block_size"
            )
        
        # Remove translation first
        center = np.array([self.block_size / 2.0, self.block_size / 2.0], dtype=float)
        pt = np.array([float(u_corrected), float(v_corrected)], dtype=float)
        
        translated_back = pt - np.array([
            correction.translation_um[0],
            correction.translation_um[1]
        ], dtype=float)
        
        # Remove rotation (inverse: R^T)
        angle_rad = np.radians(correction.rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
        
        local = (R.T @ (translated_back - center)) + center
        
        return float(local[0]), float(local[1])
    
    def block_local_to_stage(self, 
                            block_id: int, 
                            u_local: float, 
                            v_local: float) -> Tuple[float, float]:
        """
        Convert block-local coordinates -> stage coordinates.
        
        Flow:
        1. Apply per-block correction (fabrication + calibration)
        2. Convert to global design coordinates
        3. Apply global transform to get stage coordinates
        
        Args:
            block_id: Block identifier
            u_local, v_local: Local coordinates (relative to block bottom-left)
        
        Returns:
            (Y, Z): Stage coordinates in µm
        
        Raises:
            RuntimeError: If global transform not set or layout not available
        """
        if self.layout is None:
            raise RuntimeError(
                "Layout required for block-aware transforms. "
                "Initialize with CameraLayout or RuntimeLayout."
            )
        
        if not self.global_transform.is_set:
            raise RuntimeError(
                "Global transform not set. Call use_ground_truth(), "
                "sync_with_runtime(), or set_global_transform() first."
            )
        
        # Get block design position
        if int(block_id) not in self.layout.blocks:
            raise ValueError(f"Block {block_id} not found in layout")
        
        block = self.layout.blocks[int(block_id)]
        
        # 1. Apply per-block correction
        u_corrected, v_corrected = self._apply_block_correction(block_id, u_local, v_local)
        
        # 2. Convert to global design coordinates
        # Block design_position is the center
        u_center, v_center = block.design_position.u, block.design_position.v
        u_bl = u_center - (self.block_size / 2.0)
        v_bl = v_center - (self.block_size / 2.0)
        
        u_global = u_bl + u_corrected
        v_global = v_bl + v_corrected
        
        # 3. Apply global transform
        return self.design_to_stage(u_global, v_global)
    
    def stage_to_block_local(self,
                            Y: float,
                            Z: float,
                            block_id: int) -> Tuple[float, float]:
        """
        Convert stage coordinates -> block-local coordinates.
        
        Inverse of block_local_to_stage().
        
        Args:
            Y, Z: Stage coordinates in µm
            block_id: Block identifier
        
        Returns:
            (u_local, v_local): Local coordinates in µm
        
        Raises:
            RuntimeError: If global transform not set or layout not available
        """
        if self.layout is None:
            raise RuntimeError(
                "Layout required for block-aware transforms. "
                "Initialize with CameraLayout or RuntimeLayout."
            )
        
        if not self.global_transform.is_set:
            raise RuntimeError(
                "Global transform not set. Call use_ground_truth(), "
                "sync_with_runtime(), or set_global_transform() first."
            )
        
        # Get block design position
        if int(block_id) not in self.layout.blocks:
            raise ValueError(f"Block {block_id} not found in layout")
        
        block = self.layout.blocks[int(block_id)]
        
        # 1. Remove global transform
        u_global, v_global = self.stage_to_design(Y, Z)
        
        # 2. Convert to block-local (corrected) coordinates
        u_center, v_center = block.design_position.u, block.design_position.v
        u_bl = u_center - (self.block_size / 2.0)
        v_bl = v_center - (self.block_size / 2.0)
        
        u_corrected = u_global - u_bl
        v_corrected = v_global - v_bl
        
        # 3. Remove per-block correction
        u_local, v_local = self._remove_block_correction(block_id, u_corrected, v_corrected)
        
        return u_local, v_local
    
    # ========================================================================
    # CONVENIENCE METHODS (Fiducials, Gratings)
    # ========================================================================
    
    def get_fiducial_stage_position(self, 
                                   block_id: int, 
                                   corner: str) -> Tuple[float, float]:
        """
        Get stage position of a fiducial marker.
        
        Args:
            block_id: Block identifier
            corner: Corner name ('top_left', 'bottom_right', etc.)
        
        Returns:
            (Y, Z): Stage coordinates in µm
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Initialize with CameraLayout or RuntimeLayout.")
        
        block = self.layout.blocks[int(block_id)]
        fiducial = block.get_fiducial(corner)
        
        return self.block_local_to_stage(block_id, fiducial.u, fiducial.v)
    
    def get_grating_stage_position(self,
                                  block_id: int,
                                  waveguide: int,
                                  side: str) -> Tuple[float, float]:
        """
        Get stage position of a grating coupler.
        
        Args:
            block_id: Block identifier
            waveguide: Waveguide number
            side: 'left' or 'right'
        
        Returns:
            (Y, Z): Stage coordinates in µm
        """
        if self.layout is None:
            raise RuntimeError("Layout required. Initialize with CameraLayout or RuntimeLayout.")
        
        block = self.layout.blocks[int(block_id)]
        grating = block.get_grating(waveguide, side)
        
        return self.block_local_to_stage(block_id, grating.position.u, grating.position.v)
    
    # ========================================================================
    # STATUS AND INTROSPECTION
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current transformation status.
        
        Returns:
            dict: Status information
        """
        return {
            'layout_type': self.layout_type,
            'has_layout': self.layout is not None,
            'block_size': self.block_size,
            'global_transform': {
                'is_set': self.global_transform.is_set,
                'rotation_deg': self.global_transform.rotation_deg if self.global_transform.is_set else None,
                'translation_um': self.global_transform.translation_um if self.global_transform.is_set else None,
                'source': self.global_transform.source
            },
            'block_corrections': {
                'count': len(self.block_corrections),
                'blocks': list(self.block_corrections.keys()),
                'sources': {bid: corr.source for bid, corr in self.block_corrections.items()}
            }
        }
    
    def is_ready(self) -> bool:
        """Check if transform is ready for coordinate conversions."""
        return self.global_transform.is_set
    
    def is_block_corrected(self, block_id: int) -> bool:
        """Check if specific block has correction defined."""
        correction = self._get_block_correction(block_id)
        return correction.source != "none"
    
    def __repr__(self) -> str:
        if not self.global_transform.is_set:
            status = "not_calibrated"
        else:
            status = f"calibrated({self.global_transform.source})"
        
        return (
            f"CoordinateTransformV4("
            f"layout={self.layout_type}, "
            f"status={status}, "
            f"block_corrections={len(self.block_corrections)})"
        )


# ============================================================================
# TEST / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    print("CoordinateTransformV4 - Dual Layout Support")
    print("=" * 70)
    
    layout_path = "config/mock_layout.json"
    if not Path(layout_path).exists():
        print(f"Error: {layout_path} not found")
        raise SystemExit(1)
    
    # ========================================================================
    # TEST 1: CameraLayout with ground truth (simulation)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: CameraLayout (Simulation)")
    print("="*70)
    
    from config.layout_models import CameraLayout, RuntimeLayout
    
    camera_layout = CameraLayout.from_json_file(layout_path)
    print(f"Loaded: {camera_layout}")
    
    ct_camera = CoordinateTransformV3(camera_layout)
    print(f"Transform: {ct_camera}")
    print(f"Status: {ct_camera.get_status()}")
    
    # Use ground truth
    ct_camera.use_ground_truth()
    print(f"\nAfter use_ground_truth():")
    print(f"  Status: {ct_camera.get_status()['global_transform']}")
    
    # Test conversions
    test_block = 1
    block = camera_layout.blocks[test_block]
    fiducial = block.get_fiducial('top_left')
    
    print(f"\nBlock {test_block} top_left fiducial:")
    print(f"  Local: ({fiducial.u:.2f}, {fiducial.v:.2f}) µm")
    
    Y, Z = ct_camera.block_local_to_stage(test_block, fiducial.u, fiducial.v)
    print(f"  Stage: ({Y:.2f}, {Z:.2f}) µm")
    
    u_back, v_back = ct_camera.stage_to_block_local(Y, Z, test_block)
    print(f"  Round-trip: ({u_back:.6f}, {v_back:.6f}) µm")
    print(f"  Error: {math.hypot(u_back - fiducial.u, v_back - fiducial.v):.9f} µm")
    
    # ========================================================================
    # TEST 2: RuntimeLayout with progressive calibration
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: RuntimeLayout (Progressive Calibration)")
    print("="*70)
    
    runtime_layout = RuntimeLayout.from_json_file(layout_path)
    print(f"Loaded: {runtime_layout}")
    
    ct_runtime = CoordinateTransformV3(runtime_layout)
    print(f"Transform: {ct_runtime}")
    
    # Initially not calibrated
    print(f"\nBefore calibration:")
    print(f"  is_ready: {ct_runtime.is_ready()}")
    
    try:
        Y, Z = ct_runtime.block_local_to_stage(1, 100, 100)
        print(f"  ERROR: Should have raised RuntimeError!")
    except RuntimeError as e:
        print(f"  ✓ Correctly raised: {e}")
    
    # Simulate Stage 1 calibration
    print(f"\nSimulating Stage 1 calibration...")
    runtime_layout.set_global_calibration(
        rotation=camera_layout.ground_truth.rotation_deg,
        translation=(camera_layout.ground_truth.translation_um.u,
                    camera_layout.ground_truth.translation_um.v),
        calibration_error=0.05,
        num_points=8
    )
    
    # Sync transform with RuntimeLayout
    ct_runtime.sync_with_runtime()
    print(f"After sync_with_runtime():")
    print(f"  is_ready: {ct_runtime.is_ready()}")
    print(f"  Status: {ct_runtime.get_status()['global_transform']}")
    
    # Now conversions should work
    Y, Z = ct_runtime.block_local_to_stage(test_block, fiducial.u, fiducial.v)
    print(f"\nBlock {test_block} top_left fiducial:")
    print(f"  Stage: ({Y:.2f}, {Z:.2f}) µm")
    
    # Simulate Stage 2 calibration for block 1
    print(f"\nSimulating Stage 2 calibration for Block {test_block}...")
    block_error = camera_layout.ground_truth.get_block_error(test_block)
    runtime_layout.set_block_calibration(
        block_id=test_block,
        rotation=block_error.rotation_deg,
        translation=(block_error.translation_um.u, block_error.translation_um.v),
        calibration_error=0.02,
        num_points=4
    )
    
    # Sync again
    ct_runtime.sync_with_runtime()
    print(f"After block calibration:")
    print(f"  Block corrections: {ct_runtime.get_status()['block_corrections']}")
    
    # Test with block correction
    Y2, Z2 = ct_runtime.block_local_to_stage(test_block, fiducial.u, fiducial.v)
    print(f"\nWith block correction:")
    print(f"  Stage: ({Y2:.2f}, {Z2:.2f}) µm")
    print(f"  Difference from camera: ({Y2-Y:.6f}, {Z2-Z:.6f}) µm")
    
    # ========================================================================
    # TEST 3: Compare RuntimeLayout vs CameraLayout results
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: RuntimeLayout vs CameraLayout Comparison")
    print("="*70)
    
    # Test several points across block
    test_points = [
        (50, 50), (150, 50), (50, 150), (150, 150),  # corners
        (100, 100)  # center
    ]
    
    print(f"\nTesting {len(test_points)} points in Block {test_block}:")
    print(f"{'Local (u,v)':<20} {'Camera Stage':<25} {'Runtime Stage':<25} {'Difference':<15}")
    print("-" * 85)
    
    max_diff = 0.0
    for u, v in test_points:
        Y_cam, Z_cam = ct_camera.block_local_to_stage(test_block, u, v)
        Y_run, Z_run = ct_runtime.block_local_to_stage(test_block, u, v)
        
        diff = math.hypot(Y_cam - Y_run, Z_cam - Z_run)
        max_diff = max(max_diff, diff)
        
        print(f"({u:>3}, {v:>3}) µm      "
              f"({Y_cam:>8.2f}, {Z_cam:>8.2f})     "
              f"({Y_run:>8.2f}, {Z_run:>8.2f})     "
              f"{diff:>8.6f} µm")
    
    print(f"\nMax difference: {max_diff:.9f} µm")
    
    # ========================================================================
    # TEST 4: Manual transform setting
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 4: Manual Transform Setting")
    print("="*70)
    
    ct_manual = CoordinateTransformV3()  # No layout
    print(f"Empty transform: {ct_manual}")
    print(f"is_ready: {ct_manual.is_ready()}")
    
    # Set manual transform
    ct_manual.set_global_transform(
        rotation_deg=5.0,
        translation_um=(100.0, -50.0),
        source="manual"
    )
    print(f"\nAfter manual setting:")
    print(f"  is_ready: {ct_manual.is_ready()}")
    print(f"  Status: {ct_manual.get_status()['global_transform']}")
    
    # Test basic design<->stage conversion (no layout needed)
    u_test, v_test = 1000.0, 2000.0
    Y_test, Z_test = ct_manual.design_to_stage(u_test, v_test)
    u_back, v_back = ct_manual.stage_to_design(Y_test, Z_test)
    
    print(f"\nBasic conversion test:")
    print(f"  Design: ({u_test}, {v_test}) µm")
    print(f"  Stage: ({Y_test:.2f}, {Z_test:.2f}) µm")
    print(f"  Back to design: ({u_back:.6f}, {v_back:.6f}) µm")
    print(f"  Round-trip error: {math.hypot(u_back - u_test, v_back - v_test):.9f} µm")
    
    # ========================================================================
    # TEST 5: Error handling
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 5: Error Handling")
    print("="*70)
    
    # Test uncalibrated RuntimeLayout
    runtime_fresh = RuntimeLayout.from_json_file(layout_path)
    ct_fresh = CoordinateTransformV3(runtime_fresh)
    
    print(f"\nFresh RuntimeLayout (no calibration):")
    print(f"  is_ready: {ct_fresh.is_ready()}")
    
    try:
        Y, Z = ct_fresh.design_to_stage(100, 100)
        print(f"  ERROR: Should have raised RuntimeError!")
    except RuntimeError as e:
        print(f"  ✓ Correctly raised: {str(e)[:60]}...")
    
    # Test block operations without layout
    ct_no_layout = CoordinateTransformV3()
    ct_no_layout.set_global_transform(1.0, (0, 0))
    
    print(f"\nTransform without layout:")
    try:
        Y, Z = ct_no_layout.block_local_to_stage(1, 100, 100)
        print(f"  ERROR: Should have raised RuntimeError!")
    except RuntimeError as e:
        print(f"  ✓ Correctly raised: {str(e)[:60]}...")
    
    # Test invalid block ID
    try:
        Y, Z = ct_runtime.block_local_to_stage(999, 100, 100)
        print(f"  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {str(e)[:60]}...")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY: CoordinateTransformV4 Features")
    print("="*70)
    print("""
✓ Dual Layout Support:
  - CameraLayout (simulation with ground truth)
  - RuntimeLayout (real experiments with progressive calibration)

✓ Progressive Calibration:
  - Handles uncalibrated state gracefully
  - Syncs with RuntimeLayout as calibration progresses
  - Stage 1: Global sample transform
  - Stage 2: Per-block corrections

✓ Coordinate Transformations:
  - design_to_stage() / stage_to_design() for global conversions
  - block_local_to_stage() / stage_to_block_local() for block-aware conversions
  - Convenience methods for fiducials and gratings

✓ Flexible Calibration Sources:
  - use_ground_truth() - load from CameraLayout
  - sync_with_runtime() - load from RuntimeLayout measurements
  - set_global_transform() - manual setting
  - set_block_correction() - manual block corrections

✓ Robust Error Handling:
  - Clear error messages for missing calibration
  - Validates layout availability for block operations
  - Safe handling of incomplete calibration states

Usage Patterns:

# Simulation (with ground truth):
>>> camera_layout = CameraLayout.from_json_file("layout.json")
>>> ct = CoordinateTransformV4(camera_layout)
>>> ct.use_ground_truth()
>>> Y, Z = ct.get_fiducial_stage_position(block_id, 'top_left')

# Real experiment (progressive calibration):
>>> runtime_layout = RuntimeLayout.from_json_file("layout.json")
>>> ct = CoordinateTransformV4(runtime_layout)
>>> # After Stage 1 calibration in HierarchicalAlignment:
>>> ct.sync_with_runtime()
>>> Y, Z = ct.block_local_to_stage(block_id, u, v)
>>> # After Stage 2 for specific block:
>>> ct.sync_with_runtime()  # Picks up block corrections
>>> Y, Z = ct.block_local_to_stage(block_id, u, v)  # Now includes block correction
""")