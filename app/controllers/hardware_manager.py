# app/controllers/hardware_manager.py
"""
✅ FIXED: Hardware Manager with proper cleanup and error recovery
"""

from typing import Tuple, Optional
from pathlib import Path


class HardwareManager:
    """
    Manages hardware initialization and cleanup.
    
    ✅ Key improvements:
    - Defensive disconnect in shutdown()
    - Better error messages for common issues
    - Cleanup on initialization failure
    """
    
    def __init__(self, layout_path: str = "config/mock_layout.json"):
        self.layout_path = layout_path
        
        self.camera = None
        self.stage = None
        self.stage_adapter = None
        self.filter_stage = None
        self.mcs_manager = None
        
        self.mode = "disconnected"
        
        self.real_camera_available = False
        self.real_stage_available = False
        
        self._detect_real_hardware()
    
    def _detect_real_hardware(self):
        """Detect available hardware drivers."""
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
        return {
            'real_camera': self.real_camera_available,
            'real_stage': self.real_stage_available,
            'mock_always': True
        }
    
    def initialize_mock_hardware(self, layout_source: str = "config/mock_layout.json") -> Tuple[bool, str]:
        """Initialize mock hardware (unchanged)."""
        try:
            from hardware_control.setup_motor.mock_stage import MockXYZStage
            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            from hardware_control.camera_control.mock_camera import MockCamera
            
            stage_nm = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
            self.stage = stage_nm
            self.stage_adapter = StageAdapterUM(stage_nm)
            
            if not Path(layout_source).exists():
                return False, f"Design file not found: {layout_source}"
            
            self.camera = MockCamera(
                layout_config_path=layout_source,
                stage_ref=self.stage_adapter
            )
            self.camera.connect()
            
            stage_nm.set_camera_observer(self.camera)
            
            self.filter_stage = None
            
            self.mode = "mock"
            return True, "Mock hardware initialized successfully"
        
        except Exception as e:
            self.mode = "disconnected"
            return False, f"Failed to initialize mock hardware: {e}"
    
    def initialize_real_hardware(self) -> Tuple[bool, str]:
        """
        ✅ FIXED: Initialize real hardware with proper error handling.
        """
        print("[HardwareManager] Initializing real hardware...")
        
        # Step 1: Initialize MCS stages
        try:
            from hardware_control.setup_motor.multi_mcs_manager import MultiMCSManager
            
            self.mcs_manager = MultiMCSManager()
            devices = self.mcs_manager.discover_devices()
            
            if len(devices) == 0:
                return False, "No MCS devices found"
            
            print(f"[HardwareManager] Found {len(devices)} MCS device(s)")
            
            if not self.mcs_manager.auto_assign_roles():
                return False, "Failed to auto-assign device roles"
            
            if not self.mcs_manager.validate_assignments():
                print("[HardwareManager] ⚠️  Device assignment validation failed, proceeding...")
            
            # Initialize XYZ stage
            stage_nm = self.mcs_manager.get_xyz_stage()
            self.stage = stage_nm
            
            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            self.stage_adapter = StageAdapterUM(stage_nm)
            print("[HardwareManager] ✅ XYZ stage initialized")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to initialize MCS stages: {e}"
        
        # Step 2: Initialize camera (FIXED ERROR HANDLING)
        try:
            from hardware_control.camera_control.zyla_camera import ZylaCamera
            
            print("[HardwareManager] Connecting to camera...")
            self.camera = ZylaCamera()
            self.camera.connect()
            print("[HardwareManager] ✅ Camera initialized")
            
        except RuntimeError as e:
            # ✅ HANDLE DEVICE-IN-USE ERROR SPECIFICALLY
            error_msg = str(e)
            
            if "already in use" in error_msg.lower() or "deviceinuse" in error_msg.lower():
                # Clean up stage before returning
                self._emergency_cleanup()
                
                return False, (
                    "❌ Camera is in use by another process\n\n"
                    "SOLUTIONS:\n"
                    "1️⃣  Close any camera software (Solis, AndorView, etc.)\n"
                    "2️⃣  Kill orphaned Python processes in Task Manager\n"
                    "3️⃣  Restart the camera (power cycle)\n"
                    "4️⃣  Restart this application\n\n"
                    f"Technical details: {error_msg}"
                )
            else:
                self._emergency_cleanup()
                return False, f"Camera initialization failed: {error_msg}"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._emergency_cleanup()
            return False, f"Camera initialization failed: {e}"
        
        # Step 3: Initialize filter stage (optional)
        try:
            self.filter_stage = self.mcs_manager.get_filter_stage()
            print("[HardwareManager] ✅ Filter stage initialized")
        except RuntimeError as e:
            print(f"[HardwareManager] ℹ️  No filter stage available: {e}")
            self.filter_stage = None
        except Exception as e:
            print(f"[HardwareManager] ⚠️  Filter stage initialization failed: {e}")
            self.filter_stage = None
        
        self.mode = "real"
        return True, "Real hardware connected successfully"
    
    def _emergency_cleanup(self):
        """
        ✅ NEW: Emergency cleanup when initialization fails partway through.
        """
        print("[HardwareManager] 🚨 Emergency cleanup...")
        
        try:
            if self.camera is not None:
                self.camera.disconnect()
                self.camera = None
        except Exception as e:
            print(f"[HardwareManager]   Camera cleanup error: {e}")
        
        try:
            if self.stage is not None:
                self.stage.close()
                self.stage = None
                self.stage_adapter = None
        except Exception as e:
            print(f"[HardwareManager]   Stage cleanup error: {e}")
        
        try:
            if self.mcs_manager is not None:
                self.mcs_manager.close_all()
                self.mcs_manager = None
        except Exception as e:
            print(f"[HardwareManager]   MCS cleanup error: {e}")
    
    def shutdown(self):
        """
        ✅ FIXED: Shutdown with defensive error handling.
        """
        print("[HardwareManager] Shutting down hardware...")
        
        # Camera (FIXED: More defensive)
        if self.camera is not None:
            try:
                print("[HardwareManager]   Disconnecting camera...")
                self.camera.disconnect()
                self.camera = None
                print("[HardwareManager]   ✅ Camera closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Camera disconnect error: {e}")
                self.camera = None  # Force cleanup
        
        # Filter stage
        if self.filter_stage is not None:
            try:
                print("[HardwareManager]   Closing filter stage...")
                self.filter_stage.close()
                self.filter_stage = None
                print("[HardwareManager]   ✅ Filter stage closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Filter stage error: {e}")
                self.filter_stage = None
        
        # Stage
        if self.stage is not None:
            try:
                print("[HardwareManager]   Closing stage...")
                self.stage.close()
                self.stage = None
                self.stage_adapter = None
                print("[HardwareManager]   ✅ Stage closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Stage error: {e}")
                self.stage = None
                self.stage_adapter = None
        
        # MCS manager
        if self.mcs_manager is not None:
            try:
                print("[HardwareManager]   Closing MCS manager...")
                self.mcs_manager.close_all()
                self.mcs_manager = None
                print("[HardwareManager]   ✅ MCS manager closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  MCS error: {e}")
                self.mcs_manager = None
        
        self.mode = "disconnected"
        print("[HardwareManager] ✅ All hardware shutdown complete")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Getters (unchanged)
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_camera(self):
        return self.camera
    
    def get_stage(self):
        return self.stage_adapter
    
    def get_filter_stage(self):
        return self.filter_stage
    
    def get_mode(self) -> str:
        return self.mode
    
    def is_connected(self) -> bool:
        return self.camera is not None and self.stage_adapter is not None