# integrate_alignment.py
"""
Integration of alignment system with existing experiment control.
Extends agentTools.py with alignment capabilities.
"""
from typing import Dict, Any
from AlignmentSystem.alignment_controller import AlignmentController
from AlignmentSystem.alignment_agent_tools import AlignmentTools, create_alignment_tool_definitions
from AlignmentSystem.layout_config_generator import load_layout_config


class ExtendedExperimentTools:
    """
    Extended version of ExperimentTools with alignment capabilities.
    Combines existing stage/camera tools with new alignment tools.
    """
    
    def __init__(self, api_client, camera_app, stage_app, layout_config_path: str):
        """
        Initialize extended tools.
        
        Args:
            api_client: ExperimentAPIClient instance
            camera_app: AndorCameraApp instance
            stage_app: XYZStageApp instance
            layout_config_path: Path to layout JSON file
        """
        # Import existing tools
        from AgentFramework.agentTools import ExperimentTools
        self.basic_tools = ExperimentTools(api_client)
        
        # Load layout configuration
        self.layout = load_layout_config(layout_config_path)
        
        # Initialize alignment system
        self.alignment_controller = AlignmentController(
            camera_app,
            stage_app,
            self.layout
        )
        
        self.alignment_tools = AlignmentTools(self.alignment_controller)
        
        print("[INTEGRATION] Extended tools initialized with alignment capabilities")
    
    # ========================================
    # Delegate to basic tools (existing functionality)
    # ========================================
    
    def get_current_position(self) -> Dict[str, Any]:
        return self.basic_tools.get_current_position()
    
    def get_axis_position(self, axis: str) -> Dict[str, Any]:
        return self.basic_tools.get_axis_position(axis)
    
    def move_axis_absolute(self, axis: str, position: int, 
                          skip_confirmation: bool = False) -> Dict[str, Any]:
        return self.basic_tools.move_axis_absolute(axis, position, skip_confirmation)
    
    def move_axis_relative(self, axis: str, shift: int,
                          skip_confirmation: bool = False) -> Dict[str, Any]:
        return self.basic_tools.move_axis_relative(axis, shift, skip_confirmation)
    
    def run_autofocus(self, axis: str = "x", range: int = None,
                     step: int = None, enable_plot: bool = False) -> Dict[str, Any]:
        return self.basic_tools.run_autofocus(axis, range, step, enable_plot)
    
    def get_autofocus_results(self) -> Dict[str, Any]:
        return self.basic_tools.get_autofocus_results()
    
    def get_camera_info(self) -> Dict[str, Any]:
        return self.basic_tools.get_camera_info()
    
    def check_system_health(self) -> Dict[str, Any]:
        return self.basic_tools.check_system_health()
    
    # ========================================
    # New alignment tools
    # ========================================
    
    def capture_and_analyze_image(self) -> Dict[str, Any]:
        return self.alignment_tools.capture_and_analyze_image()
    
    def find_fiducial(self, corner: str, block_id: int = 1) -> Dict[str, Any]:
        return self.alignment_tools.find_fiducial(corner, block_id)
    
    def measure_current_intensity(self, roi: str = None) -> Dict[str, Any]:
        return self.alignment_tools.measure_current_intensity(roi)
    
    def navigate_to_design_coords(self, u: float, v: float) -> Dict[str, Any]:
        return self.alignment_tools.navigate_to_design_coords(u, v)
    
    def calibrate_sample(self, block1_id: int = 1, block2_id: int = 20) -> Dict[str, Any]:
        return self.alignment_tools.calibrate_sample(block1_id, block2_id)
    
    def align_to_grating(self, block_id: int, waveguide_number: int,
                        side: str = 'left') -> Dict[str, Any]:
        return self.alignment_tools.align_to_grating(block_id, waveguide_number, side)
    
    def get_alignment_state(self) -> Dict[str, Any]:
        return self.alignment_tools.get_alignment_state()
    
    def scan_all_center_gratings(self) -> Dict[str, Any]:
        return self.alignment_tools.scan_all_center_gratings()


def create_extended_tool_definitions():
    """
    Create complete tool definitions including both basic and alignment tools.
    """
    # Import existing tool definitions
    from AgentFramework.agentTools import create_tool_definitions
    
    basic_tools = create_tool_definitions()
    alignment_tools = create_alignment_tool_definitions()
    
    # Combine
    return basic_tools + alignment_tools


# ========================================
# Usage Example
# ========================================

def initialize_extended_agent(camera_app, stage_app, api_client, layout_config_path):
    """
    Initialize agent with extended tools including alignment.
    
    Returns:
        Extended agent ready to use
    """
    from AgentFramework.agentController import ExperimentAgent
    from AgentFramework.agentConfig import (
        LITELLM_BASE_URL,
        MODEL_NAME,
        API_KEY,
        AGENT_SYSTEM_PROMPT
    )
    from openai import OpenAI
    
    # Create extended tools
    extended_tools = ExtendedExperimentTools(
        api_client,
        camera_app,
        stage_app,
        layout_config_path
    )
    
    # Initialize OpenAI client
    client = OpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=API_KEY
    )
    
    # Enhanced system prompt
    enhanced_prompt = AGENT_SYSTEM_PROMPT + """

## Alignment Capabilities

You now have advanced alignment tools for automated grating coupler alignment:

**New Tools Available:**
- capture_and_analyze_image() - Capture and analyze camera images
- find_fiducial(corner, block_id) - Find fiducial markers
- calibrate_sample() - Calibrate sample coordinate system
- navigate_to_design_coords(u, v) - Navigate using design coordinates
- align_to_grating(block_id, waveguide_number, side) - Full alignment workflow
- get_alignment_state() - Check calibration and alignment status
- scan_all_center_gratings() - Scan all 20 blocks automatically

**Typical Workflow:**
1. calibrate_sample() - Find fiducials and calibrate (do this first!)
2. align_to_grating(10, 25, 'left') - Align to specific grating
3. measure_current_intensity() - Check coupling efficiency
4. scan_all_center_gratings() - Automated scan of all blocks

Always check get_alignment_state() to verify calibration before alignment.
"""
    
    # Create agent instance with extended tools
    # Note: You'll need to modify ExperimentAgent to accept tool instance
    # For now, return components
    
    return {
        'client': client,
        'tools': extended_tools,
        'tool_definitions': create_extended_tool_definitions(),
        'system_prompt': enhanced_prompt
    }


if __name__ == "__main__":
    print("Alignment System Integration")
    print("=============================")
    print("\nThis module integrates alignment capabilities with existing system.")
    print("Use this in your main application to enable automated alignment.")
    print("\nExample usage:")
    print("""
    from integrate_alignment import initialize_extended_agent
    
    # Initialize with your existing components
    agent_components = initialize_extended_agent(
        camera_app=my_camera_app,
        stage_app=my_stage_app,
        api_client=my_api_client,
        layout_config_path='config/sample_layout.json'
    )
    
    # Now agent has both basic and alignment tools
    """)