# agent_tools.py
"""
Tool functions that the AI agent can call to control the experiment.
These wrap the REST API client with additional safety checks and formatting.
"""
from typing import Optional, Dict, Any
from Testing.test_api_client import ExperimentAPIClient
from AgentFramework.agentConfig import MAX_SAFE_MOVE, MAX_SAFE_AUTOFOCUS_RANGE, VALID_AXES


class ExperimentTools:
    """Collection of tools the agent can use to control the experiment."""
    
    def __init__(self, api_client: ExperimentAPIClient):
        self.api = api_client
        
    # ========================================
    # Position Tools
    # ========================================
    
    def get_current_position(self) -> Dict[str, Any]:
        """
        Get the current position of all axes.
        
        Returns:
            dict: {"x": int, "y": int, "z": int} positions in nanometers
        """
        try:
            result = self.api.get_status()
            return {
                "success": True,
                "x": result['x'],
                "y": result['y'],
                "z": result['z'],
                "message": f"X={result['x']}nm, Y={result['y']}nm, Z={result['z']}nm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get position: {e}"
            }
    
    def get_axis_position(self, axis: str) -> Dict[str, Any]:
        """
        Get the current position of a specific axis.
        
        Args:
            axis: 'x', 'y', or 'z'
            
        Returns:
            dict: {"axis": str, "position": int, "success": bool}
        """
        axis = axis.lower()
        if axis not in VALID_AXES:
            return {
                "success": False,
                "error": f"Invalid axis '{axis}'. Must be x, y, or z.",
                "message": f"Invalid axis '{axis}'"
            }
        
        try:
            result = self.api.get_position(axis)
            return {
                "success": True,
                "axis": result['axis'],
                "position": result['position'],
                "message": f"{axis.upper()}={result['position']}nm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get {axis} position: {e}"
            }
    
    # ========================================
    # Movement Tools
    # ========================================
    
    def move_axis_absolute(
        self, 
        axis: str, 
        position: int,
        skip_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        Move an axis to an absolute position.
        
        Args:
            axis: 'x', 'y', or 'z'
            position: Target position in nanometers
            skip_confirmation: Skip safety check (used after user confirms)
            
        Returns:
            dict: {"success": bool, "axis": str, "position": int, "message": str}
        """
        axis = axis.lower()
        if axis not in VALID_AXES:
            return {
                "success": False,
                "error": f"Invalid axis '{axis}'",
                "message": f"Invalid axis '{axis}'. Must be x, y, or z."
            }
        
        # Safety check: get current position and calculate distance
        if not skip_confirmation:
            try:
                current = self.api.get_position(axis)
                distance = abs(position - current['position'])
                
                if distance > MAX_SAFE_MOVE:
                    return {
                        "success": False,
                        "needs_confirmation": True,
                        "distance": distance,
                        "message": (
                            f"⚠️  Large movement detected: {distance}nm ({distance/1000:.1f}µm)\n"
                            f"Current {axis.upper()}: {current['position']}nm → Target: {position}nm\n"
                            f"Please confirm this movement."
                        )
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to check current position: {e}"
                }
        
        # Execute movement
        try:
            result = self.api.move_absolute(axis, position)
            return {
                "success": True,
                "axis": result['axis'],
                "position": result['position'],
                "message": f"Moved {axis.upper()} to {result['position']}nm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to move {axis}: {e}"
            }
    
    def move_axis_relative(
        self,
        axis: str,
        shift: int,
        skip_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        Move an axis by a relative amount.
        
        Args:
            axis: 'x', 'y', or 'z'
            shift: Shift in nanometers (can be negative)
            skip_confirmation: Skip safety check
            
        Returns:
            dict: {"success": bool, "axis": str, "position": int, "message": str}
        """
        axis = axis.lower()
        if axis not in VALID_AXES:
            return {
                "success": False,
                "error": f"Invalid axis '{axis}'",
                "message": f"Invalid axis '{axis}'. Must be x, y, or z."
            }
        
        # Safety check
        if not skip_confirmation and abs(shift) > MAX_SAFE_MOVE:
            return {
                "success": False,
                "needs_confirmation": True,
                "distance": abs(shift),
                "message": (
                    f"⚠️  Large relative movement: {shift}nm ({shift/1000:.1f}µm)\n"
                    f"Please confirm this movement."
                )
            }
        
        # Execute movement
        try:
            result = self.api.move_relative(axis, shift)
            return {
                "success": True,
                "axis": result['axis'],
                "position": result['position'],
                "message": f"Moved {axis.upper()} by {shift:+d}nm → now at {result['position']}nm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to move {axis}: {e}"
            }
    
    # ========================================
    # Autofocus Tools
    # ========================================
    
    def run_autofocus(
        self,
        axis: str = "x",
        range: Optional[int] = None,
        step: Optional[int] = None,
        enable_plot: bool = False
    ) -> Dict[str, Any]:
        """
        Run autofocus scan on specified axis.
        
        Args:
            axis: 'x', 'y', or 'z' (default: 'x')
            range: Scan range in nanometers (default: 10000)
            step: Step size in nanometers (default: 500)
            enable_plot: Show live plot during scan
            
        Returns:
            dict: Autofocus results with best position and metric
        """
        axis = axis.lower()
        if axis not in VALID_AXES:
            return {
                "success": False,
                "error": f"Invalid axis '{axis}'",
                "message": f"Invalid axis '{axis}'. Must be x, y, or z."
            }
        
        # Safety check for range
        if range and range > MAX_SAFE_AUTOFOCUS_RANGE:
            return {
                "success": False,
                "needs_confirmation": True,
                "range": range,
                "message": (
                    f"⚠️  Large autofocus range: {range}nm ({range/1000:.1f}µm)\n"
                    f"Please confirm this scan range."
                )
            }
        
        try:
            result = self.api.run_autofocus(
                axis=axis,
                range=range,
                step=step,
                enable_plot=enable_plot
            )
            return {
                "success": True,
                "axis": result['axis'],
                "best_position": result['best_position'],
                "best_metric": result['best_metric'],
                "scan_complete": result['scan_complete'],
                "message": (
                    f"Autofocus complete on {axis.upper()}-axis!\n"
                    f"Best focus: {result['best_position']}nm (metric: {result['best_metric']:.2f})"
                )
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Autofocus failed: {e}"
            }
    
    def get_autofocus_results(self) -> Dict[str, Any]:
        """
        Get results from the last autofocus scan.
        
        Returns:
            dict: Last autofocus results
        """
        try:
            result = self.api.get_autofocus_results()
            return {
                "success": True,
                "axis": result['axis'],
                "best_position": result['best_position'],
                "best_metric": result['best_metric'],
                "message": (
                    f"Last autofocus ({result['axis'].upper()}-axis):\n"
                    f"Best position: {result['best_position']}nm\n"
                    f"Focus metric: {result['best_metric']:.2f}"
                )
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"No autofocus results available or error: {e}"
            }
    
    # ========================================
    # Camera Tools
    # ========================================
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and current settings.
        
        Returns:
            dict: Camera model, exposure time, sensor size
        """
        try:
            result = self.api.get_camera_info()
            return {
                "success": True,
                "model": result['model'],
                "serial": result['serial'],
                "exposure_time": result['exposure_time'],
                "sensor_width": result['sensor_width'],
                "sensor_height": result['sensor_height'],
                "message": (
                    f"Camera: {result['model']} (S/N: {result['serial']})\n"
                    f"Exposure: {result['exposure_time']}s\n"
                    f"Sensor: {result['sensor_width']}x{result['sensor_height']} pixels"
                )
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get camera info: {e}"
            }
    
    # ========================================
    # System Tools
    # ========================================
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check if the experiment control system is healthy.
        
        Returns:
            dict: System health status
        """
        try:
            result = self.api.health_check()
            threads = result['threads_running']
            
            status_msg = f"System Status: {result['status']}\n"
            status_msg += f"Camera: {'✅ Connected' if result['camera_connected'] else '❌ Disconnected'}\n"
            status_msg += f"Stage: {'✅ Connected' if result['stage_connected'] else '❌ Disconnected'}\n"
            status_msg += f"Threads: Camera={'✅' if threads['camera'] else '❌'}, "
            status_msg += f"Stage={'✅' if threads['stage'] else '❌'}, "
            status_msg += f"Input={'✅' if threads['input'] else '❌'}"
            
            return {
                "success": True,
                "status": result['status'],
                "camera_connected": result['camera_connected'],
                "stage_connected": result['stage_connected'],
                "threads_running": threads,
                "message": status_msg
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to check system health: {e}"
            }


# ========================================
# Tool Function Definitions for Agent
# ========================================

def create_tool_definitions():
    """
    Create tool definitions for the agent framework.
    These describe the tools available to the agent.
    """
    return [
        {
            "name": "get_current_position",
            "description": "Get the current position of all X, Y, Z axes in nanometers",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "move_axis_absolute",
            "description": "Move an axis to an absolute position in nanometers",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["x", "y", "z"],
                        "description": "The axis to move (x, y, or z)"
                    },
                    "position": {
                        "type": "integer",
                        "description": "Target position in nanometers"
                    }
                },
                "required": ["axis", "position"]
            }
        },
        {
            "name": "move_axis_relative",
            "description": "Move an axis by a relative amount in nanometers (can be negative)",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["x", "y", "z"],
                        "description": "The axis to move"
                    },
                    "shift": {
                        "type": "integer",
                        "description": "Amount to shift in nanometers (positive or negative)"
                    }
                },
                "required": ["axis", "shift"]
            }
        },
        {
            "name": "run_autofocus",
            "description": "Run an autofocus scan to find the optimal focus position",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["x", "y", "z"],
                        "description": "Axis to scan for focus (default: x)"
                    },
                    "range": {
                        "type": "integer",
                        "description": "Scan range in nanometers (default: 10000)"
                    },
                    "step": {
                        "type": "integer",
                        "description": "Step size in nanometers (default: 500)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_autofocus_results",
            "description": "Get the results from the last autofocus scan",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_camera_info",
            "description": "Get information about the camera (model, exposure, sensor size)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "check_system_health",
            "description": "Check if the experiment control system is running and healthy",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]