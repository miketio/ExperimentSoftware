# ZylaCamera.py
import numpy as np
from typing import Optional, Tuple
from hardware_control.camera_control.andor_camera_base import AndorCameraBase
import pylablib.devices.Andor as Andor
import time
from typing import Optional
import cv2

class ZylaCamera(AndorCameraBase):
    """
    Concrete implementation of AndorCameraBase for Andor Zyla (and other SDK3) cameras
    using the pylablib library.
    """

    def __init__(self):
        super().__init__()
        self._cam = None
        self._streaming = False
        self._current_roi = None  # (x, y, width, height)

    # ───────────────────────
    # Connection Management
    # ───────────────────────

    def connect(self) -> None:
        if self._cam is not None:
            raise RuntimeError("Camera already connected.")
        print("Connecting to Andor SDK3 camera...")
        self._cam = Andor.AndorSDK3Camera()
        print("Successfully connected.")

    def disconnect(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass  # Ignore errors on close
            self._cam = None
            self._streaming = False

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
        # Full sensor size: width x height
        w, h = self._cam.get_detector_size()
        return (w, h)

    # ───────────────────────
    # Configuration
    # ───────────────────────

    def set_exposure_time(self, seconds: float) -> None:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        if seconds <= 0:
            raise ValueError("Exposure time must be positive.")
        self._cam.set_attribute_value("ExposureTime", seconds)

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
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        if any(v is None for v in (x, y, width, height)):
             self._cam.set_roi()
        else:
            sensor_w, sensor_h = self.get_sensor_size()
            print(f"Sensor sizes{sensor_h}, {sensor_w}")
            if not (0 <= x < sensor_w and 0 <= y < sensor_h):
                raise ValueError("ROI origin out of sensor bounds.")
            if not (0 < x <= sensor_w - width and 0 < y<= sensor_h - height):
                raise ValueError("ROI size exceeds sensor bounds.")
            # Andor SDK3 uses 1-based indexing
            self._cam.set_roi(hstart=x + 1, hend=x + width, vstart=y + 1, vend=y + height)
            self._current_roi = (x, y, width, height)

    # ───────────────────────
    # Acquisition
    # ───────────────────────

    def acquire_single_image(self) -> np.ndarray:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        image = self._cam.snap()
        image = np.asarray(image)
        return self._apply_software_gain(image)

    def start_streaming(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera not connected.")
        if self._streaming:
            return
        # Do NOT call setup_acquisition — just start continuous acquisition
        self._cam.setup_acquisition(mode="sequence")
        self._cam.start_acquisition()
        self._streaming = True

    def stop_streaming(self) -> None:
        if self._cam is None or not self._streaming:
            return
        try:
            self._cam.stop_acquisition()
            # Optional: clear buffers if needed
            # self._cam.clear_acquisition()
        except Exception:
            pass
        self._streaming = False

    def read_next_image(self) -> Optional[np.ndarray]:
        """
        For live streaming, we actually read the OLDEST unread frame
        to avoid dropping frames and ensure smooth playback.
        Despite the method name 'latest', this matches common live-view behavior.
        """
        if self._cam is None or not self._streaming:
            return None
        image = self._cam.read_newest_image()  # 100 ms timeout
        if image is None:
            return None
        image = np.asarray(image)
        return self._apply_software_gain(image)

    # ───────────────────────
    # Internal Helpers
    # ───────────────────────

    def _apply_software_gain(self, image: np.ndarray) -> np.ndarray:
        if self._software_gain == 1.0:
            return image
        img_f = image.astype(np.float32) * self._software_gain
        max_val = (2**16 - 1) * 0.9
        img_f = np.clip(img_f, 0, max_val)
        return img_f.astype(np.uint16)


# ───────────────────────────────────────
# Test / Demo Code
# ───────────────────────────────────────

if __name__ == "__main__":
    print("🧪 Testing ZylaCamera...")

    try:
        with ZylaCamera() as cam:
            # 1. Camera info
            info = cam.get_camera_info()
            print(f"✅ Camera Model: {info['model']}, Serial: {info['serial']}")

            # 2. Sensor size
            w, h = cam.get_sensor_size()
            print(f"✅ Sensor size: {w} x {h} pixels")

            # 3. Set HDR mode
            try:
                cam.set_bit_depth_mode("High dynamic range (16-bit)")
                print("✅ Set to HDR mode")
            except (ValueError, RuntimeError) as e:
                # Try alternative label
                try:
                    cam.set_bit_depth_mode("16-bit (low noise & high well capacity)")
                    print("✅ Set to alternative HDR mode")
                except Exception:
                    print("⚠️  Could not set HDR mode:", e)

            # 4. Set exposure
            cam.set_exposure_time(0.02)  # 20 ms
            actual_exp = cam.get_exposure_time()
            print(f"✅ Exposure set to: {actual_exp:.4f} s")

            # 5. Single image
            print("📸 Acquiring single image...")
            img = cam.acquire_single_image()
            print(f"✅ Single image shape: {img.shape}, dtype: {img.dtype}, min={img.min()}, max={img.max()}")

            # 6. Test software gain
            cam.set_software_gain(2.0)
            img_gained = cam.acquire_single_image()
            print(f"✅ With gain=2.0 → max={img_gained.max()} (original max was {img.max()})")

            # 7. Live streaming test

            cam.start_streaming()
            time.sleep(0.1) 
            for i in range(5):
                frame = cam.read_next_image()
                print(f"Frame {i+1}: {'OK' if frame is not None else 'None'}")
                time.sleep(0.05)
            cam.stop_streaming()
            print("✅ Live streaming test completed.")
            print("🧪 ZylaCamera test completed successfully.")
    except Exception as e:
        print("❌ ZylaCamera test failed:", e)