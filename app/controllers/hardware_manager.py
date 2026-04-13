# app/controllers/hardware_manager.py
"""
Hardware Manager with proper cleanup and error recovery.
Supports: MCS XYZ stage, MCS 1D filter stage, HCU 3D stage.
"""

from typing import Tuple, Optional
from pathlib import Path
import time


class HardwareManager:
    """
    Manages hardware initialization and cleanup.

    Devices managed:
        camera         — Andor Zyla (or MockCamera)
        stage_adapter  — SmarActXYZStage wrapped in StageAdapterUM (MCS, nm→µm)
        filter_stage   — FilterStage  (MCS 1-channel, nm)
        hcu_stage      — HCU3CStage   (SCU 3-channel, nm)  ← NEW
    """

    def __init__(self, layout_path: str = "config/mock_layout.json"):
        self.layout_path = layout_path

        self.camera        = None
        self.stage         = None     # raw nm stage
        self.stage_adapter = None     # µm adapter
        self.filter_stage  = None
        self.hcu_stage     = None     # ← NEW
        self.mcs_manager   = None

        self.mode = "disconnected"

        self.real_camera_available = False
        self.real_stage_available  = False
        self.real_hcu_available    = False   # ← NEW

        self._detect_real_hardware()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_real_hardware(self):
        """Detect available hardware drivers AND actual devices."""

        # Camera: check driver import
        try:
            from hardware_control.camera_control.zyla_camera import ZylaCamera
            self.real_camera_available = True
        except Exception:
            self.real_camera_available = False

        # MCS stages
        try:
            from hardware_control.setup_motor.multi_mcs_manager import MultiMCSManager
            manager  = MultiMCSManager()
            devices  = manager.discover_devices()
            if devices:
                self.real_stage_available = True
                print(f"[HardwareManager] MCS detection: {len(devices)} device(s) found")
            else:
                self.real_stage_available = False
                print("[HardwareManager] MCS detection: No devices found")
            try:
                manager.close_all()
            except Exception:
                pass
        except Exception as e:
            self.real_stage_available = False
            print(f"[HardwareManager] MCS detection failed: {e}")

        # HCU stage (separate SCU driver)
        try:
            from hardware_control.setup_motor.hcu3c_stage import list_hcu_devices
            devices = list_hcu_devices()
            if devices:
                self.real_hcu_available = True
                print(f"[HardwareManager] HCU detection: {len(devices)} device(s) found")
            else:
                self.real_hcu_available = False
                print("[HardwareManager] HCU detection: No devices found")
        except Exception as e:
            self.real_hcu_available = False
            print(f"[HardwareManager] HCU detection failed (driver missing?): {e}")

    def get_hardware_availability(self) -> dict:
        return {
            'real_camera': self.real_camera_available,
            'real_stage':  self.real_stage_available,
            'real_hcu':    self.real_hcu_available,
            'mock_always': True
        }

    # ------------------------------------------------------------------
    # Mock hardware
    # ------------------------------------------------------------------

    def initialize_mock_hardware(
        self, layout_source: str = "config/mock_layout.json"
    ) -> Tuple[bool, str]:
        """Initialize mock hardware (no HCU mock — buttons disabled in UI)."""
        try:
            from hardware_control.setup_motor.mock_stage import MockXYZStage
            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            from hardware_control.camera_control.mock_camera import MockCamera

            stage_nm = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
            self.stage         = stage_nm
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
            self.hcu_stage    = None   # no HCU mock

            self.mode = "mock"
            return True, "Mock hardware initialized successfully"

        except Exception as e:
            self.mode = "disconnected"
            return False, f"Failed to initialize mock hardware: {e}"

    # ------------------------------------------------------------------
    # Real hardware
    # ------------------------------------------------------------------

    def initialize_real_hardware(self) -> Tuple[bool, str]:
        """Initialize real hardware (MCS stages + camera + optional HCU)."""
        print("[HardwareManager] Initializing real hardware...")

        # Step 1: MCS stages
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

            stage_nm = self.mcs_manager.get_xyz_stage()
            self.stage = stage_nm

            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            self.stage_adapter = StageAdapterUM(stage_nm)
            print("[HardwareManager] ✅ XYZ stage initialized")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to initialize MCS stages: {e}"

        # Step 2: Camera
        try:
            from hardware_control.camera_control.zyla_camera import ZylaCamera

            print("[HardwareManager] Connecting to camera...")
            self.camera = ZylaCamera()
            self.camera.connect()

            for mode in (
                "16-bit (low noise & high well capacity)",
                "High dynamic range (16-bit)",
            ):
                try:
                    self.camera.set_bit_depth_mode(mode)
                    print(f"[HardwareManager] ✅ Camera bit-depth mode: {mode}")
                    break
                except Exception:
                    pass
            else:
                print("[HardwareManager] ⚠️  Could not set 16-bit mode")

            print("[HardwareManager] ✅ Camera initialized")

        except RuntimeError as e:
            error_msg = str(e)
            self._emergency_cleanup()
            if "already in use" in error_msg.lower() or "deviceinuse" in error_msg.lower():
                return False, (
                    "❌ Camera is in use by another process\n\n"
                    "SOLUTIONS:\n"
                    "1️⃣  Close any camera software (Solis, AndorView, etc.)\n"
                    "2️⃣  Kill orphaned Python processes in Task Manager\n"
                    "3️⃣  Restart the camera (power cycle)\n"
                    "4️⃣  Restart this application\n\n"
                    f"Technical details: {error_msg}"
                )
            return False, f"Camera initialization failed: {error_msg}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._emergency_cleanup()
            return False, f"Camera initialization failed: {e}"

        # Step 3: Filter stage (optional MCS 1D)
        try:
            self.filter_stage = self.mcs_manager.get_filter_stage()
            print("[HardwareManager] ✅ Filter stage initialized")
        except RuntimeError as e:
            print(f"[HardwareManager] ℹ️  No MCS filter stage: {e}")
            self.filter_stage = None
        except Exception as e:
            print(f"[HardwareManager] ⚠️  Filter stage init failed: {e}")
            self.filter_stage = None

        # Step 4: HCU stage (optional, separate SCU controller)
        self._init_hcu_stage()

        self.mode = "real"
        return True, "Real hardware connected successfully"

    def _init_hcu_stage(self):
        """
        Try to initialise the HCU 3-axis stage (SCU driver, separate from MCS).
        Non-fatal — sets self.hcu_stage to None on failure.
        """
        try:
            from hardware_control.setup_motor.hcu3c_stage import HCU3CStage, list_hcu_devices

            devices = list_hcu_devices()
            if not devices:
                print("[HardwareManager] ℹ️  HCU stage: no devices found")
                self.hcu_stage = None
                return

            print(f"[HardwareManager] HCU: {len(devices)} device(s) found — "
                  f"using device 0 (id={devices[0]['device_id']})")

            self.hcu_stage = HCU3CStage(device_idx=0, verbose=False)
            print("[HardwareManager] ✅ HCU stage initialized")

        except Exception as e:
            print(f"[HardwareManager] ⚠️  HCU stage init failed: {e}")
            self.hcu_stage = None

    # ------------------------------------------------------------------
    # Stage-only mode (no camera)
    # ------------------------------------------------------------------

    def initialize_stage_only_hardware(self) -> Tuple[bool, str]:
        """Initialize real XYZ + filter stages + HCU, skip camera."""
        print("[HardwareManager] Initializing stage-only hardware (no camera)...")

        try:
            from hardware_control.setup_motor.multi_mcs_manager import MultiMCSManager
            self.mcs_manager = MultiMCSManager()
            devices = self.mcs_manager.discover_devices()
            if not devices:
                return False, "No MCS devices found"
            print(f"[HardwareManager] Found {len(devices)} MCS device(s)")
            if not self.mcs_manager.auto_assign_roles():
                return False, "Failed to auto-assign device roles"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"MCS discovery failed: {e}"

        try:
            stage_nm = self.mcs_manager.get_xyz_stage()
            self.stage = stage_nm
            from hardware_control.setup_motor.stage_adapter import StageAdapterUM
            self.stage_adapter = StageAdapterUM(stage_nm)
            print("[HardwareManager] ✅ XYZ stage ready")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"XYZ stage init failed: {e}"

        try:
            self.filter_stage = self.mcs_manager.get_filter_stage()
            print("[HardwareManager] ✅ Filter stage ready")
        except RuntimeError as e:
            print(f"[HardwareManager] ℹ️  No filter stage: {e}")
            self.filter_stage = None
        except Exception as e:
            print(f"[HardwareManager] ⚠️  Filter stage failed: {e}")
            self.filter_stage = None

        # HCU optional
        self._init_hcu_stage()

        self.mode = "stage_only"
        return True, "Stage-only hardware initialized"

    # ------------------------------------------------------------------
    # Emergency cleanup
    # ------------------------------------------------------------------

    def _emergency_cleanup(self):
        print("[HardwareManager] 🚨 Emergency cleanup...")
        for attr, label in [('camera', 'Camera'), ('stage', 'Stage'), ('hcu_stage', 'HCU')]:
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    if hasattr(obj, 'disconnect'):
                        obj.disconnect()
                    elif hasattr(obj, 'close'):
                        obj.close()
                except Exception as e:
                    print(f"[HardwareManager]   {label} cleanup error: {e}")
                setattr(self, attr, None)
        if self.mcs_manager is not None:
            try:
                self.mcs_manager.close_all()
            except Exception as e:
                print(f"[HardwareManager]   MCS cleanup error: {e}")
            self.mcs_manager   = None
            self.stage_adapter = None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Shutdown all hardware — homes filter stage first."""
        print("[HardwareManager] Shutting down hardware...")

        # Home 1D filter stage
        if self.filter_stage is not None:
            try:
                print("[HardwareManager]   Homing 1D filter stage to 0 …")
                self.filter_stage.move_abs(0)
                time.sleep(0.8)
                pos = self.filter_stage.get_position()
                print(f"[HardwareManager]   ✅ Filter stage homed ({pos} nm)")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Failed to home filter stage: {e}")

        # Close camera
        if self.camera is not None:
            try:
                print("[HardwareManager]   Disconnecting camera …")
                self.camera.disconnect()
                self.camera = None
                print("[HardwareManager]   ✅ Camera closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Camera disconnect error: {e}")
                self.camera = None

        # Close 1D filter stage
        if self.filter_stage is not None:
            try:
                self.filter_stage.close()
                self.filter_stage = None
                print("[HardwareManager]   ✅ Filter stage closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Filter stage error: {e}")
                self.filter_stage = None

        # Close HCU stage
        if self.hcu_stage is not None:
            try:
                print("[HardwareManager]   Closing HCU stage …")
                self.hcu_stage.close()
                self.hcu_stage = None
                print("[HardwareManager]   ✅ HCU stage closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  HCU stage error: {e}")
                self.hcu_stage = None

        # Close MCS XYZ stage
        if self.stage is not None:
            try:
                self.stage.close()
                self.stage         = None
                self.stage_adapter = None
                print("[HardwareManager]   ✅ XYZ stage closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  Stage error: {e}")
                self.stage = self.stage_adapter = None

        if self.mcs_manager is not None:
            try:
                self.mcs_manager.close_all()
                self.mcs_manager = None
                print("[HardwareManager]   ✅ MCS manager closed")
            except Exception as e:
                print(f"[HardwareManager]   ⚠️  MCS error: {e}")
                self.mcs_manager = None

        self.mode = "disconnected"
        print("[HardwareManager] ✅ All hardware shutdown complete")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_camera(self):
        return self.camera

    def get_stage(self):
        return self.stage_adapter

    def get_filter_stage(self):
        return self.filter_stage

    def get_hcu_stage(self):          # ← NEW
        return self.hcu_stage

    def get_mode(self) -> str:
        return self.mode

    def is_connected(self) -> bool:
        return self.camera is not None and self.stage_adapter is not None