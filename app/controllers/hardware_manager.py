"""
Hardware Manager

Handles hardware initialization, detection, and mode switching (mock/real).
"""

from typing import Tuple, Optional
from pathlib import Path


class HardwareManager:
    """
    Manages hardware initialization and mode selection.
    
    Provides:
    - Mock/real hardware detection
    - Initialization with proper adapters
    - Hardware status queries
    """
    
    def __init__(self, layout_path: str = "config/mock_layout.json"):
        """
        Initialize hardware manager.
        
        Args:
            layout_path: Path to layout configuration (for mock camera)
        """
        self.layout_path = layout_path
        
        self.camera = None
        self.stage = None
        self.stage_adapter = None  # Micrometer adapter
        
        self.mode = "disconnected"  # "mock", "real", "disconnected"
        
        # Hardware availability
        self.real_camera_available = False
        self.real_stage_available = False
        
        self._detect_real_hardware()
    
    def _detect_real_hardware(self):
        """Detect if real hardware is available."""
        # Try importing real hardware modules
        try:
            from hardware_control.camera_control.zyla_camera import ZylaCamera
            self.real_camera_available = True
        except Exception:
            self.real_camera_available = False
        
        try:
            from hardware_control.setup_motor.smartact_stage import SmarActXYZStage
            self.real_stage_available = True
        except Exception:
            self.real_stage_available = False
    
    def get_hardware_availability(self) -> dict:
        """Get hardware detection results."""
        return {
            'real_camera': self.real_camera_available,
            'real_stage': self.real_stage_available,
            'mock_always': True  # Mock hardware always available
        }
    
    def initialize_mock_hardware(self, layout_source: str = "config/mock_layout.json") -> Tuple[bool, str]:
        """
        Initialize mock hardware.
        
        Args:
            layout_source: Path to DESIGN file (with ground truth)
        """
        try:
            from hardware_control.setup_motor.mock_stage import MockXYZStage
            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            from hardware_control.camera_control.mock_camera import MockCamera
            
            # Stage
            stage_nm = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
            self.stage = stage_nm
            self.stage_adapter = StageAdapterUM(stage_nm)
            
            # Camera needs DESIGN file (with ground truth)
            if not Path(layout_source).exists():
                return False, f"Design file not found: {layout_source}"
            
            self.camera = MockCamera(
                layout_config_path=layout_source,  # ALWAYS use design source
                stage_ref=self.stage_adapter
            )
            self.camera.connect()
            
            stage_nm.set_camera_observer(self.camera)
            
            self.mode = "mock"
            return True, "Mock hardware initialized successfully"
        
        except Exception as e:
            self.mode = "disconnected"
            return False, f"Failed to initialize mock hardware: {e}"
    
    def initialize_real_hardware(self) -> Tuple[bool, str]:
        """
        Initialize real hardware.
        
        Returns:
            (success, message)
        """
        try:
            from hardware_control.camera_control.zyla_camera import ZylaCamera
            from hardware_control.setup_motor.smartact_stage import SmarActXYZStage
            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            
            # Initialize stage
            stage_nm = SmarActXYZStage()
            self.stage = stage_nm
            self.stage_adapter = StageAdapterUM(stage_nm)
            
            # Initialize camera
            self.camera = ZylaCamera()
            self.camera.connect()
            
            self.mode = "real"
            return True, "Real hardware connected successfully"
        
        except Exception as e:
            self.mode = "disconnected"
            return False, f"Failed to connect to real hardware: {e}"
    
    def shutdown(self):
        """Shutdown all hardware."""
        try:
            if self.camera is not None:
                self.camera.disconnect()
                self.camera = None
        except Exception as e:
            print(f"Error disconnecting camera: {e}")
        
        try:
            if self.stage is not None:
                self.stage.close()
                self.stage = None
                self.stage_adapter = None
        except Exception as e:
            print(f"Error closing stage: {e}")
        
        self.mode = "disconnected"
    
    def get_camera(self):
        """Get camera instance (or None)."""
        return self.camera
    
    def get_stage(self):
        """Get stage adapter instance (µm interface, or None)."""
        return self.stage_adapter
    
    def get_mode(self) -> str:
        """Get current hardware mode."""
        return self.mode
    
    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        return self.camera is not None and self.stage_adapter is not None