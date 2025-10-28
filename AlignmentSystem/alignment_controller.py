# alignment_controller.py
"""
Main alignment controller that orchestrates the complete alignment workflow.
"""
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.coordinate_transform import CoordinateTransform
from AlignmentSystem.alignment_state import AlignmentState, AlignmentStatus, get_alignment_state
from AlignmentSystem.ascii_parser import ASCIIParser, find_waveguide_grating


class AlignmentController:
    """
    Main controller for automated sample alignment and grating coupling optimization.
    
    Workflow:
        1. Find fiducial markers (2 opposite corners)
        2. Calibrate coordinate transform
        3. Navigate to target grating
        4. Optimize coupling (2D grid search)
    """
    
    def __init__(self, camera_app, stage_app, layout_config: Dict):
        """
        Initialize alignment controller.
        
        Args:
            camera_app: AndorCameraApp instance
            stage_app: XYZStageApp instance
            layout_config: Dictionary with block/grating positions
        """
        self.camera = camera_app
        self.stage = stage_app
        self.layout = layout_config
        
        self.vision = VisionTools()
        self.transform = CoordinateTransform()
        self.state = get_alignment_state()
        
        # Optimization parameters
        self.grid_search_range = 5000  # ±5 µm in nm
        self.grid_search_step = 500    # 500 nm steps
        
        print("[ALIGNMENT] Controller initialized")
    
    # ========================================
    # High-Level Workflows
    # ========================================
    
    def calibrate_sample(self, block1_id: int = 1, block2_id: int = 20) -> Dict:
        """
        Calibrate sample using two fiducial markers from opposite corners.
        
        Args:
            block1_id: First block (default: 1, top-left)
            block2_id: Second block (default: 20, bottom-right)
        
        Returns:
            dict with calibration results
        """
        print(f"\n[ALIGNMENT] ========================================")
        print(f"[ALIGNMENT] Starting Sample Calibration")
        print(f"[ALIGNMENT] Using blocks {block1_id} and {block2_id}")
        print(f"[ALIGNMENT] ========================================")
        
        self.state.set_status(AlignmentStatus.CALIBRATING)
        
        try:
            # Get block positions from layout
            block1 = self.layout['blocks'][block1_id]
            block2 = self.layout['blocks'][block2_id]
            
            # Find fiducials
            print(f"\n[ALIGNMENT] Finding fiducial 1 (block {block1_id}, top-left)...")
            fid1_design = block1['fiducials']['top_left']
            fid1_result = self._find_fiducial_with_navigation(fid1_design, 'top_left')
            
            if not fid1_result['found']:
                raise RuntimeError(f"Failed to find fiducial 1: {fid1_result.get('error')}")
            
            print(f"\n[ALIGNMENT] Finding fiducial 2 (block {block2_id}, bottom-right)...")
            fid2_design = block2['fiducials']['bottom_right']
            fid2_result = self._find_fiducial_with_navigation(fid2_design, 'bottom_right')
            
            if not fid2_result['found']:
                raise RuntimeError(f"Failed to find fiducial 2: {fid2_result.get('error')}")
            
            # Perform calibration
            print(f"\n[ALIGNMENT] Computing coordinate transform...")
            measured_points = [
                (fid1_result['stage_Y'], fid1_result['stage_Z']),
                (fid2_result['stage_Z'], fid2_result['stage_Z'])
            ]
            design_points = [fid1_design, fid2_design]
            
            calib_result = self.transform.calibrate(measured_points, design_points)
            
            # Update state
            self.state.set_calibration(calib_result)
            
            print(f"\n[ALIGNMENT] ========================================")
            print(f"[ALIGNMENT] Calibration Complete!")
            print(f"[ALIGNMENT] Rotation: {calib_result['angle_deg']:.3f}°")
            print(f"[ALIGNMENT] Mean error: {calib_result['mean_error_nm']:.1f} nm")
            print(f"[ALIGNMENT] ========================================")
            
            return {
                'success': True,
                'calibration': calib_result,
                'fiducial1': fid1_result,
                'fiducial2': fid2_result
            }
            
        except Exception as e:
            self.state.set_status(AlignmentStatus.ERROR, str(e))
            print(f"\n[ALIGNMENT] ❌ Calibration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def align_to_grating(self, block_id: int, waveguide_number: int, side: str = 'left') -> Dict:
        """
        Complete alignment workflow for a specific grating coupler.
        
        Args:
            block_id: Block number (1-20)
            waveguide_number: Waveguide number within block (1-50)
            side: 'left' or 'right'
        
        Returns:
            dict with alignment results
        """
        print(f"\n[ALIGNMENT] ========================================")
        print(f"[ALIGNMENT] Aligning to Grating")
        print(f"[ALIGNMENT] Block: {block_id}, WG: {waveguide_number}, Side: {side}")
        print(f"[ALIGNMENT] ========================================")
        
        # Check calibration
        if not self.state.is_calibrated:
            print(f"[ALIGNMENT] ⚠️  Sample not calibrated. Running calibration first...")
            calib_result = self.calibrate_sample()
            if not calib_result['success']:
                return {
                    'success': False,
                    'error': 'Calibration failed',
                    'calibration_result': calib_result
                }
        
        try:
            # Get grating position from layout
            grating_key = f"wg{waveguide_number}_{side}"
            block = self.layout['blocks'][block_id]
            
            if grating_key not in block['gratings']:
                raise ValueError(f"Grating {grating_key} not found in block {block_id}")
            
            design_coords = block['gratings'][grating_key]
            
            # Convert to stage coordinates
            stage_coords = self.transform.design_to_stage(*design_coords)
            
            # Update state
            self.state.set_target(block_id, waveguide_number, side, design_coords, stage_coords)
            
            # Navigate to rough position
            print(f"\n[ALIGNMENT] Navigating to rough position...")
            print(f"  Design: ({design_coords[0]:.1f}, {design_coords[1]:.1f}) µm")
            print(f"  Stage: ({stage_coords[0]}, {stage_coords[1]}) nm")
            
            self.state.set_status(AlignmentStatus.NAVIGATING)
            self.stage.move_abs('y', stage_coords[0])
            self.stage.move_abs('z', stage_coords[1])
            time.sleep(0.5)
            
            # Run autofocus on X axis
            print(f"\n[ALIGNMENT] Running autofocus...")
            # Note: Using existing autofocus from your system
            # You may need to integrate with your AutofocusController here
            
            # Optimize coupling
            print(f"\n[ALIGNMENT] Optimizing coupling...")
            opt_result = self.optimize_coupling()
            
            if opt_result['success']:
                print(f"\n[ALIGNMENT] ========================================")
                print(f"[ALIGNMENT] Alignment Complete!")
                print(f"[ALIGNMENT] Best position: ({opt_result['best_position'][0]}, {opt_result['best_position'][1]}) nm")
                print(f"[ALIGNMENT] Best intensity: {opt_result['best_intensity']:.1f}")
                print(f"[ALIGNMENT] ========================================")
            
            return opt_result
            
        except Exception as e:
            self.state.set_status(AlignmentStatus.ERROR, str(e))
            print(f"\n[ALIGNMENT] ❌ Alignment failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # ========================================
    # Component Functions
    # ========================================
    
    def _find_fiducial_with_navigation(self, design_pos: Tuple[float, float], 
                                      corner: str) -> Dict:
        """
        Navigate to approximate fiducial position and find it with vision.
        
        Args:
            design_pos: (u, v) design coordinates in µm
            corner: Corner name for state tracking
        
        Returns:
            dict with fiducial finding results
        """
        self.state.set_status(AlignmentStatus.FINDING_FIDUCIAL)
        
        try:
            # If not calibrated, use rough estimate
            if not self.state.is_calibrated:
                # Assume 1:1 mapping with offset (µm to nm + rough offset)
                rough_Y = int(design_pos[0] * 1000)
                rough_Z = int(design_pos[1] * 1000)
            else:
                rough_Y, rough_Z = self.transform.design_to_stage(*design_pos)
            
            print(f"  Navigating to approximate position: ({rough_Y}, {rough_Z}) nm")
            self.stage.move_abs('y', rough_Y)
            self.stage.move_abs('z', rough_Z)
            time.sleep(0.5)
            
            # Capture image
            print(f"  Capturing image...")
            image = self.camera.acquire_image()
            self.vision.save_image(image)
            
            # Try to find fiducial
            print(f"  Searching for fiducial...")
            # Assume camera FOV is ~300µm, image center is at stage position
            img_center = (image.shape[1] // 2, image.shape[0] // 2)
            
            result = self.vision.find_fiducial_auto(
                image,
                expected_position=img_center,
                search_radius=100  # pixels
            )
            
            if result:
                print(f"  ✅ Fiducial found!")
                print(f"     Pixel position: {result['position']}")
                print(f"     Confidence: {result['confidence']:.3f}")
                print(f"     Method: {result['method']}")
                
                # Convert pixel offset to stage offset (approximate)
                # Assume 1 pixel ≈ 0.65 µm (typical for scientific cameras)
                pixel_to_nm = 650  # nm per pixel (adjust for your camera)
                
                pixel_offset_x = result['position'][0] - img_center[0]
                pixel_offset_y = result['position'][1] - img_center[1]
                
                stage_offset_Y = int(pixel_offset_x * pixel_to_nm)
                stage_offset_Z = int(pixel_offset_y * pixel_to_nm)
                
                final_Y = rough_Y + stage_offset_Y
                final_Z = rough_Z + stage_offset_Z
                
                # Store in state
                self.state.add_fiducial(corner, final_Y, final_Z, result['confidence'])
                
                return {
                    'found': True,
                    'stage_Y': final_Y,
                    'stage_Z': final_Z,
                    'pixel_position': result['position'],
                    'confidence': result['confidence'],
                    'method': result['method']
                }
            else:
                print(f"  ❌ Fiducial not found")
                return {
                    'found': False,
                    'error': 'Fiducial not detected in image'
                }
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                'found': False,
                'error': str(e)
            }
    
    def optimize_coupling(self) -> Dict:
        """
        Optimize coupling using 2D grid search around current position.
        
        Returns:
            dict with optimization results
        """
        self.state.start_optimization()
        
        try:
            # Get current position
            current_Y = self.stage.get_pos('y')
            current_Z = self.stage.get_pos('z')
            
            print(f"  Current position: Y={current_Y} nm, Z={current_Z} nm")
            print(f"  Search range: ±{self.grid_search_range} nm")
            print(f"  Step size: {self.grid_search_step} nm")
            
            # Generate grid
            Y_start = current_Y - self.grid_search_range
            Y_end = current_Y + self.grid_search_range
            Z_start = current_Z - self.grid_search_range
            Z_end = current_Z + self.grid_search_range
            
            Y_positions = range(Y_start, Y_end + 1, self.grid_search_step)
            Z_positions = range(Z_start, Z_end + 1, self.grid_search_step)
            
            total_points = len(Y_positions) * len(Z_positions)
            print(f"  Grid: {len(Y_positions)} × {len(Z_positions)} = {total_points} points")
            
            # Search grid
            best_intensity = 0
            best_position = (current_Y, current_Z)
            results = []
            
            point_count = 0
            for Y in Y_positions:
                for Z in Z_positions:
                    point_count += 1
                    
                    # Move stage
                    self.stage.move_abs('y', Y)
                    self.stage.move_abs('z', Z)
                    time.sleep(0.05)  # Short settling time
                    
                    # Measure intensity
                    image = self.camera.acquire_image()
                    metrics = self.vision.measure_intensity(image)
                    intensity = metrics['sum']
                    
                    results.append({
                        'Y': Y,
                        'Z': Z,
                        'intensity': intensity
                    })
                    
                    # Update best
                    if intensity > best_intensity:
                        best_intensity = intensity
                        best_position = (Y, Z)
                    
                    # Update progress
                    progress = point_count / total_points
                    self.state.update_optimization_progress(progress)
                    
                    if point_count % 10 == 0:
                        print(f"  Progress: {progress * 100:.1f}% | Best: {best_intensity:.1f}")
            
            # Move to best position
            print(f"\n  Moving to best position: ({best_position[0]}, {best_position[1]}) nm")
            self.stage.move_abs('y', best_position[0])
            self.stage.move_abs('z', best_position[1])
            time.sleep(0.2)
            
            # Store result
            result = {
                'success': True,
                'best_position': best_position,
                'best_intensity': best_intensity,
                'grid_size': total_points,
                'search_range_nm': self.grid_search_range,
                'step_size_nm': self.grid_search_step,
                'all_results': results
            }
            
            self.state.finish_optimization(result)
            
            return result
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e)
            }
            self.state.finish_optimization(result)
            return result


# Test/example usage
if __name__ == "__main__":
    print("Alignment Controller Module")
    print("============================")
    print("\nThis module requires hardware connections.")
    print("Run integration tests with real hardware using test scripts.")