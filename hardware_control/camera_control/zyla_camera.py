# hardware_control/camera_control/zyla_camera.py
"""
✅ ROBUST: Safe parameter changes during acquisition
"""

import numpy as np
from typing import Optional, Tuple
from hardware_control.camera_control.andor_camera_base import AndorCameraBase
import pylablib.devices.Andor as Andor
import time


class ZylaCamera(AndorCameraBase):
    """
    Andor Zyla camera with ROBUST parameter changes.
    
    Key features:
    - Safe exposure changes (stop acquisition if needed)
    - Safe ROI changes (stop acquisition if needed)
    - Proper state tracking
    - Defensive error handling
    """

    def __init__(self):
        super().__init__()
        self._cam = None
        self._streaming = False
        self._current_roi = None
        self.acquisition_running = False
        self._changing_params = False  # ✅ NEW: Flag for parameter changes

    # ═══════════════════════════════════════════════════════════════════════
    # Connection Management
    # ═══════════════════════════════════════════════════════════════════════

    def connect(self) -> None:
        """Connect to camera with proper error handling."""
        if self._cam is not None:
            raise RuntimeError("Camera already connected.")
        
        print("[ZylaCamera] Connecting to Andor SDK3 camera...")
        
        try:
            self._cam = Andor.AndorSDK3Camera()
            print("[ZylaCamera] ✅ Successfully connected.")
            
        except Exception as e:
            error_msg = str(e)
            
            if "AT_ERR_DEVICEINUSE" in error_msg or "38" in error_msg:
                raise RuntimeError(
                    "Camera is already in use by another process.\n\n"
                    "Solutions:\n"
                    "1. Close any other camera software (Solis, previous Python sessions)\n"
                    "2. Restart the camera hardware\n"
                    "3. Restart this application\n"
                    "4. Check Task Manager for orphaned Python processes"
                ) from e
            else:
                raise RuntimeError(f"Failed to connect to camera: {error_msg}") from e

    def disconnect(self) -> None:
        """Properly disconnect camera with full cleanup."""
        if self._cam is None:
            print("[ZylaCamera] Already disconnected")
            return
        
        print("[ZylaCamera] Disconnecting camera...")
        
        try:
            # Stop streaming
            if self._streaming:
                print("[ZylaCamera]   Stopping stream...")
                self.stop_streaming()
            
            # Ensure acquisition stopped
            if self.acquisition_running:
                print("[ZylaCamera]   Force stopping acquisition...")
                try:
                    self._cam.stop_acquisition()
                    self.acquisition_running = False
                except Exception as e:
                    print(f"[ZylaCamera]   Warning: {e}")
            
            time.sleep(0.1)
            
            # Close camera
            print("[ZylaCamera]   Closing camera handle...")
            self._cam.close()
            
            # Clear state
            self._cam = None
            self._streaming = False
            self.acquisition_running = False
            
            print("[ZylaCamera] ✅ Camera disconnected successfully")
            
        except Exception as e:
            print(f"[ZylaCamera] ⚠️  Error during disconnect: {e}")
            self._cam = None
            self._streaming = False
            self.acquisition_running = False
            print("[ZylaCamera] ⚠️  Camera state cleared despite errors")

    # ═══════════════════════════════════════════════════════════════════════
    # Camera Info
    # ═══════════════════════════════════════════════════════════════════════

    def get_camera_info(self) -> dict:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        info = self._cam.get_device_info()
        return {
            "model": info.camera_model,
            "serial": info.serial_number
        }

    def get_sensor_size(self) -> Tuple[int, int]:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        w, h = self._cam.get_detector_size()
        return (w, h)

    # ═══════════════════════════════════════════════════════════════════════
    # ✅ SAFE PARAMETER CHANGES
    # ═══════════════════════════════════════════════════════════════════════

    def set_exposure_time(self, seconds: float) -> None:
        """
        ✅ SAFE: Set exposure time with acquisition management.
        
        Automatically stops/restarts acquisition if needed.
        """
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        if seconds <= 0:
            raise ValueError("Exposure time must be positive.")
        
        # ✅ Check if we need to stop acquisition
        was_acquiring = self.acquisition_running
        
        if was_acquiring:
            print(f"[ZylaCamera] Stopping acquisition to change exposure...")
            try:
                self._cam.stop_acquisition()
                self.acquisition_running = False
                time.sleep(0.05)  # Brief settle time
            except Exception as e:
                print(f"[ZylaCamera] Warning during acquisition stop: {e}")
        
        # Set exposure
        try:
            self._cam.set_attribute_value("ExposureTime", seconds)
            print(f"[ZylaCamera] ✅ Exposure set to {seconds:.4f}s")
        except Exception as e:
            print(f"[ZylaCamera] ❌ Failed to set exposure: {e}")
            raise
        
        # ✅ Restart acquisition if it was running
        if was_acquiring:
            print(f"[ZylaCamera] Restarting acquisition...")
            try:
                self._cam.start_acquisition()
                self.acquisition_running = True
            except Exception as e:
                print(f"[ZylaCamera] ❌ Failed to restart acquisition: {e}")
                # Don't raise - leave acquisition stopped

    def get_exposure_time(self) -> float:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        return float(self._cam.get_attribute_value("ExposureTime"))

    def set_bit_depth_mode(self, mode: str) -> None:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        if "SimplePreAmpGainControl" not in self._cam.get_all_attributes():
            raise RuntimeError("Camera does not support SimplePreAmpGainControl.")
        options = self._cam.get_attribute("SimplePreAmpGainControl").values
        if mode not in options:
            raise ValueError(f"Mode '{mode}' not in available options: {options}")
        self._cam.set_attribute_value("SimplePreAmpGainControl", mode)

    def set_roi(self, x: Optional[int] = None, y: Optional[int] = None,
                width: Optional[int] = None, height: Optional[int] = None) -> None:
        """
        ✅ SAFE: Set ROI with acquisition management.
        
        Automatically stops/restarts acquisition if needed.
        """
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        
        # ✅ Check if we need to stop acquisition
        was_acquiring = self.acquisition_running
        
        if was_acquiring:
            print(f"[ZylaCamera] Stopping acquisition to change ROI...")
            try:
                self._cam.stop_acquisition()
                self.acquisition_running = False
                time.sleep(0.05)
            except Exception as e:
                print(f"[ZylaCamera] Warning during acquisition stop: {e}")
        
        # Set ROI
        try:
            if any(v is None for v in (x, y, width, height)):
                # Reset to full sensor
                self._cam.set_roi()
                self._current_roi = None
                print("[ZylaCamera] ✅ ROI reset to full sensor")
            else:
                sensor_w, sensor_h = self.get_sensor_size()
                
                if not (0 <= x < sensor_w and 0 <= y < sensor_h):
                    raise ValueError("ROI origin out of sensor bounds.")
                if not (0 < width <= sensor_w - x and 0 < height <= sensor_h - y):
                    raise ValueError("ROI size exceeds sensor bounds.")
                
                # Andor SDK3 uses 1-based indexing
                self._cam.set_roi(hstart=x + 1, hend=x + width, vstart=y + 1, vend=y + height)
                self._current_roi = (x, y, width, height)
                print(f"[ZylaCamera] ✅ ROI set to ({x}, {y}, {width}, {height})")
        
        except Exception as e:
            print(f"[ZylaCamera] ❌ Failed to set ROI: {e}")
            raise
        
        # ✅ Restart acquisition if it was running
        if was_acquiring:
            print(f"[ZylaCamera] Restarting acquisition...")
            try:
                self._cam.start_acquisition()
                self.acquisition_running = True
            except Exception as e:
                print(f"[ZylaCamera] ❌ Failed to restart acquisition: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Acquisition
    # ═══════════════════════════════════════════════════════════════════════

    def acquire_single_image(self) -> np.ndarray:
        """Acquire single frame (snap mode)."""
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        
        # Snap mode doesn't conflict with streaming
        image = self._cam.snap()
        image = np.asarray(image)
        return self._apply_software_gain(image)

    def start_streaming(self) -> None:
        """Start streaming with proper state tracking."""
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        
        if self._streaming:
            print("[ZylaCamera] Already streaming")
            return
        
        print("[ZylaCamera] Starting acquisition...")
        
        try:
            # Setup and start
            self._cam.setup_acquisition(mode="sequence")
            self._cam.start_acquisition()
            
            self._streaming = True
            self.acquisition_running = True
            
            print("[ZylaCamera] ✅ Streaming started")
        
        except Exception as e:
            print(f"[ZylaCamera] ❌ Failed to start streaming: {e}")
            self._streaming = False
            self.acquisition_running = False
            raise

    def stop_streaming(self) -> None:
        """Stop streaming with proper cleanup."""
        if self._cam is None or not self._streaming:
            return
        
        print("[ZylaCamera] Stopping streaming...")
        
        try:
            self._cam.stop_acquisition()
            self.acquisition_running = False
            
            # Optional: Clear buffers
            # self._cam.clear_acquisition()
            
        except Exception as e:
            print(f"[ZylaCamera] Warning during stream stop: {e}")
        
        finally:
            self._streaming = False
            print("[ZylaCamera] ✅ Streaming stopped")

    def read_next_image(self) -> Optional[np.ndarray]:
        """
        ✅ ROBUST: Read next frame with error handling.
        
        Returns None if no frame available or error occurs.
        """
        if self._cam is None or not self._streaming:
            return None
        
        try:
            # ✅ Use read_newest_image (non-blocking)
            image = self._cam.read_newest_image()
            
            if image is None:
                return None
            
            image = np.asarray(image)
            return self._apply_software_gain(image)
            
        except Exception as e:
            # ✅ Don't spam console with frame read errors
            # These are normal during parameter changes
            if "not iterable" not in str(e) and "cannot join thread" not in str(e):
                print(f"[ZylaCamera] Frame read error: {e}")
            return None

    def _apply_software_gain(self, image: np.ndarray) -> np.ndarray:
        """Apply software gain."""
        if self._software_gain == 1.0:
            return image
        
        img_f = image.astype(np.float32) * self._software_gain
        max_val = (2**16 - 1) * 0.9
        img_f = np.clip(img_f, 0, max_val)
        return img_f.astype(np.uint16)

    # ═══════════════════════════════════════════════════════════════════════
    # ✅ NEW: Safe parameter change helpers
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current ROI (left, top, width, height) or None for full sensor."""
        return self._current_roi

    def is_streaming(self) -> bool:
        """Check if camera is currently streaming."""
        return self._streaming


# ═══════════════════════════════════════════════════════════════════════
# Test Code
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Testing ZylaCamera...")

    try:
        with ZylaCamera() as cam:
            info = cam.get_camera_info()
            print(f"✅ Camera Model: {info['model']}, Serial: {info['serial']}")

            w, h = cam.get_sensor_size()
            print(f"✅ Sensor size: {w} x {h} pixels")

            try:
                cam.set_bit_depth_mode("High dynamic range (16-bit)")
                print("✅ Set to HDR mode")
            except (ValueError, RuntimeError) as e:
                try:
                    cam.set_bit_depth_mode("16-bit (low noise & high well capacity)")
                    print("✅ Set to alternative HDR mode")
                except Exception:
                    print("⚠️  Could not set HDR mode:", e)

            # Test safe exposure change
            print("\n🧪 Testing safe exposure change during streaming...")
            cam.start_streaming()
            time.sleep(0.2)
            
            print("Changing exposure while streaming...")
            cam.set_exposure_time(0.01)  # Should handle gracefully
            time.sleep(0.2)
            
            print("Changing ROI while streaming...")
            cam.set_roi(512, 512, 1024, 1024)  # Should handle gracefully
            time.sleep(0.2)
            
            cam.stop_streaming()
            
            print("✅ Safe parameter change test passed")
            print("🧪 ZylaCamera test completed successfully.")
            
    except Exception as e:
        print("❌ ZylaCamera test failed:", e)
        import traceback
        traceback.print_exc()