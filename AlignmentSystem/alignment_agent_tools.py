# alignment_agent_tools.py
"""
Agent tools for alignment system.
These tools are added to the existing AgentFramework tools.
"""
from typing import Dict, Any, Optional
from AlignmentSystem.alignment_controller import AlignmentController
from AlignmentSystem.alignment_state import get_alignment_state
from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.layout_config_generator import load_layout_config


class AlignmentTools:
    """Collection of alignment tools for the AI agent."""
    
    def __init__(self, alignment_controller: AlignmentController):
        self.controller = alignment_controller
        self.vision = alignment_controller.vision
        self.state = get_alignment_state()
    
    # ========================================
    # Tier 1: Atomic Tools
    # ========================================
    
    def capture_and_analyze_image(self) -> Dict[str, Any]:
        """
        Capture camera image, save to tempImage.npy, and return metadata.
        
        Returns:
            dict: Image metadata with intensity metrics
        """
        try:
            image = self.controller.camera.acquire_image()
            self.vision.save_image(image)
            
            metrics = self.vision.measure_intensity(image)
            
            return {
                "success": True,
                "image_saved": "tempImage.npy",
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "intensity_metrics": metrics,
                "message": f"Image captured: {image.shape}, mean={metrics['mean']:.1f}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to capture image: {e}"
            }
    
    def find_fiducial(self, corner: str, block_id: int = 1) -> Dict[str, Any]:
        """
        Find a fiducial marker at specified corner of a block.
        
        Args:
            corner: 'top_left', 'top_right', 'bottom_left', or 'bottom_right'
            block_id: Block number (1-20)
        
        Returns:
            dict: Fiducial position and confidence
        """
        try:
            block = self.controller.layout['blocks'][block_id]
            design_pos = block['fiducials'][corner]
            
            result = self.controller._find_fiducial_with_navigation(design_pos, corner)
            
            if result['found']:
                return {
                    "success": True,
                    "found": True,
                    "corner": corner,
                    "block_id": block_id,
                    "stage_Y": result['stage_Y'],
                    "stage_Z": result['stage_Z'],
                    "confidence": result['confidence'],
                    "method": result['method'],
                    "message": f"Fiducial found at Y={result['stage_Y']}nm, Z={result['stage_Z']}nm (confidence={result['confidence']:.3f})"
                }
            else:
                return {
                    "success": False,
                    "found": False,
                    "error": result.get('error', 'Unknown error'),
                    "message": f"Fiducial not found: {result.get('error')}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to find fiducial: {e}"
            }
    
    def measure_current_intensity(self, roi: Optional[str] = None) -> Dict[str, Any]:
        """
        Measure intensity at current stage position.
        
        Args:
            roi: Optional ROI as "x,y,width,height" string
        
        Returns:
            dict: Intensity metrics
        """
        try:
            image = self.controller.camera.acquire_image()
            
            # Parse ROI if provided
            roi_tuple = None
            if roi:
                parts = roi.split(',')
                if len(parts) == 4:
                    roi_tuple = tuple(map(int, parts))
            
            metrics = self.vision.measure_intensity(image, roi_tuple)
            
            return {
                "success": True,
                "metrics": metrics,
                "roi": roi,
                "message": f"Intensity: mean={metrics['mean']:.1f}, sum={metrics['sum']:.1f}, max={metrics['max']}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to measure intensity: {e}"
            }
    
    def navigate_to_design_coords(self, u: float, v: float) -> Dict[str, Any]:
        """
        Navigate to design coordinates using calibrated transform.
        
        Args:
            u, v: Design coordinates in micrometers
        
        Returns:
            dict: Navigation result
        """
        try:
            if not self.state.is_calibrated:
                return {
                    "success": False,
                    "error": "Sample not calibrated",
                    "message": "Please run calibrate_sample() first"
                }
            
            Y, Z = self.controller.transform.design_to_stage(u, v)
            
            self.controller.stage.move_abs('y', Y)
            self.controller.stage.move_abs('z', Z)
            
            return {
                "success": True,
                "design_coords": {"u": u, "v": v},
                "stage_coords": {"Y": Y, "Z": Z},
                "message": f"Navigated to ({u}, {v}) µm → stage ({Y}, {Z}) nm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Navigation failed: {e}"
            }
    
    # ========================================
    # Tier 2: Composite Workflows
    # ========================================
    
    def calibrate_sample(self, block1_id: int = 1, block2_id: int = 20) -> Dict[str, Any]:
        """
        Calibrate sample using two fiducial markers.
        
        Args:
            block1_id: First block (default: 1)
            block2_id: Second block (default: 20)
        
        Returns:
            dict: Calibration results
        """
        result = self.controller.calibrate_sample(block1_id, block2_id)
        
        if result['success']:
            calib = result['calibration']
            return {
                "success": True,
                "angle_deg": calib['angle_deg'],
                "mean_error_nm": calib['mean_error_nm'],
                "max_error_nm": calib['max_error_nm'],
                "num_points": calib['num_points'],
                "message": f"✅ Sample calibrated! Rotation: {calib['angle_deg']:.3f}°, Error: {calib['mean_error_nm']:.1f} nm"
            }
        else:
            return {
                "success": False,
                "error": result['error'],
                "message": f"❌ Calibration failed: {result['error']}"
            }
    
    def align_to_grating(self, block_id: int, waveguide_number: int, 
                        side: str = 'left') -> Dict[str, Any]:
        """
        Complete alignment workflow for a grating coupler.
        Includes navigation and optimization.
        
        Args:
            block_id: Block number (1-20)
            waveguide_number: Waveguide number (1-50)
            side: 'left' or 'right'
        
        Returns:
            dict: Alignment results
        """
        result = self.controller.align_to_grating(block_id, waveguide_number, side)
        
        if result['success']:
            return {
                "success": True,
                "block_id": block_id,
                "waveguide_number": waveguide_number,
                "side": side,
                "best_position": result['best_position'],
                "best_intensity": result['best_intensity'],
                "grid_size": result['grid_size'],
                "message": f"✅ Aligned! Best position: ({result['best_position'][0]}, {result['best_position'][1]}) nm, Intensity: {result['best_intensity']:.1f}"
            }
        else:
            return {
                "success": False,
                "error": result['error'],
                "message": f"❌ Alignment failed: {result['error']}"
            }
    
    def get_alignment_state(self) -> Dict[str, Any]:
        """
        Get current alignment system state.
        
        Returns:
            dict: Complete state information
        """
        state_dict = self.state.get_state_dict()
        
        return {
            "success": True,
            "state": state_dict,
            "message": f"Status: {state_dict['status']}, Calibrated: {state_dict['calibration']['is_calibrated']}"
        }
    
    # ========================================
    # Tier 3: Automation
    # ========================================
    
    def scan_all_center_gratings(self) -> Dict[str, Any]:
        """
        Scan waveguide 25 left gratings in all 20 blocks.
        
        Returns:
            dict: Results for all blocks
        """
        try:
            # Ensure calibrated
            if not self.state.is_calibrated:
                calib_result = self.calibrate_sample()
                if not calib_result['success']:
                    return {
                        "success": False,
                        "error": "Calibration failed",
                        "message": "Cannot scan without calibration"
                    }
            
            results = []
            target_wg = self.controller.layout['target_waveguide']
            
            for block_id in range(1, 21):
                print(f"\n[SCAN] Processing block {block_id}/20...")
                
                result = self.controller.align_to_grating(block_id, target_wg, 'left')
                
                results.append({
                    "block_id": block_id,
                    "success": result['success'],
                    "best_intensity": result.get('best_intensity'),
                    "best_position": result.get('best_position')
                })
            
            # Summary
            successful = sum(1 for r in results if r['success'])
            
            return {
                "success": True,
                "total_blocks": 20,
                "successful": successful,
                "failed": 20 - successful,
                "results": results,
                "message": f"✅ Scan complete: {successful}/20 blocks aligned successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Scan failed: {e}"
            }


def create_alignment_tool_definitions():
    """
    Create tool definitions for agent framework.
    These are added to the existing tool definitions.
    """
    return [
        {
            "name": "capture_and_analyze_image",
            "description": "Capture camera image, save to tempImage.npy, and return intensity metrics",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "find_fiducial",
            "description": "Find a fiducial marker at specified corner of a block",
            "parameters": {
                "type": "object",
                "properties": {
                    "corner": {
                        "type": "string",
                        "enum": ["top_left", "top_right", "bottom_left", "bottom_right"],
                        "description": "Which corner marker to find"
                    },
                    "block_id": {
                        "type": "integer",
                        "description": "Block number (1-20)",
                        "default": 1
                    }
                },
                "required": ["corner"]
            }
        },
        {
            "name": "measure_current_intensity",
            "description": "Measure intensity at current stage position",
            "parameters": {
                "type": "object",
                "properties": {
                    "roi": {
                        "type": "string",
                        "description": "Optional ROI as 'x,y,width,height' string"
                    }
                },
                "required": []
            }
        },
        {
            "name": "navigate_to_design_coords",
            "description": "Navigate to design coordinates using calibrated transform",
            "parameters": {
                "type": "object",
                "properties": {
                    "u": {
                        "type": "number",
                        "description": "U coordinate in micrometers"
                    },
                    "v": {
                        "type": "number",
                        "description": "V coordinate in micrometers"
                    }
                },
                "required": ["u", "v"]
            }
        },
        {
            "name": "calibrate_sample",
            "description": "Calibrate sample using two fiducial markers from opposite corners",
            "parameters": {
                "type": "object",
                "properties": {
                    "block1_id": {
                        "type": "integer",
                        "description": "First block ID (default: 1)",
                        "default": 1
                    },
                    "block2_id": {
                        "type": "integer",
                        "description": "Second block ID (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        },
        {
            "name": "align_to_grating",
            "description": "Complete alignment workflow: navigate to grating and optimize coupling",
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {
                        "type": "integer",
                        "description": "Block number (1-20)"
                    },
                    "waveguide_number": {
                        "type": "integer",
                        "description": "Waveguide number (1-50)"
                    },
                    "side": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which side grating (default: left)",
                        "default": "left"
                    }
                },
                "required": ["block_id", "waveguide_number"]
            }
        },
        {
            "name": "get_alignment_state",
            "description": "Get current alignment system state and calibration status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "scan_all_center_gratings",
            "description": "Automatically align to waveguide 25 left grating in all 20 blocks",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]