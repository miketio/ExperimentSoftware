# app/controllers/camera_stream.py
"""
Camera Stream Controller - WITH FOURIER TRANSFORM SUPPORT + DEBUGGING

Manages camera acquisition in a separate thread with color scaling and FFT.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import time
import numpy as np
import cv2
from typing import Optional
try:
    import scipy.fft as fft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, Fourier transform disabled")


class ColorScaleManager:
    """
    Manages color scaling and colormap application for camera images.
    NOW HANDLES FOURIER TRANSFORM TOO!
    """
    
    COLORMAPS = {
        'gray': None,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'turbo': cv2.COLORMAP_TURBO,
        'rainbow': cv2.COLORMAP_RAINBOW,
    }
    
    def __init__(self):
        self.mode = 'auto'
        self.min_val = 0
        self.max_val = 4095
        self.auto_percentile = (1, 99)
        self.colormap = 'gray'
        self.last_stats = {'min': 0, 'max': 4095, 'mean': 0}
        self.fourier_mode = False
    
    def set_fourier_mode(self, enabled: bool):
        """Enable/disable Fourier transform mode."""
        self.fourier_mode = enabled
    
    def set_mode(self, mode: str):
        """Set scaling mode ('auto' or 'manual')."""
        self.mode = mode
    
    def set_manual_range(self, min_val: int, max_val: int):
        """Set manual scaling range."""
        self.min_val = min_val
        self.max_val = max_val
        self.mode = 'manual'
    
    def set_auto_percentile(self, low: float, high: float):
        """Set percentiles for auto-scaling."""
        self.auto_percentile = (low, high)
    
    def set_colormap(self, colormap: str):
        """Set colormap name."""
        if colormap not in self.COLORMAPS:
            raise ValueError(f"Unknown colormap: {colormap}")
        self.colormap = colormap
    
    def apply_fourier_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply 2D Fourier transform to image."""
        if not HAS_SCIPY:
            return image
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.dtype == np.uint16:
            gray = (image / 256).astype(np.uint8)
        else:
            gray = image
        
        f_transform = fft.fft2(gray)
        f_shifted = fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        magnitude_log = np.log1p(magnitude)
        
        return magnitude_log.astype(np.float32)
    
    def apply_scaling(self, image: np.ndarray) -> np.ndarray:
        """Apply color scaling to image."""
        if image is None or image.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        if self.fourier_mode and HAS_SCIPY:
            image = self.apply_fourier_transform(image)
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
            else:
                image = np.zeros_like(image, dtype=np.uint16)
        
        if self.mode == 'auto':
            try:
                vmin = np.percentile(image, self.auto_percentile[0])
                vmax = np.percentile(image, self.auto_percentile[1])
            except Exception:
                vmin, vmax = image.min(), image.max()
        else:
            vmin, vmax = self.min_val, self.max_val
        
        self.last_stats = {
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'scale_min': float(vmin),
            'scale_max': float(vmax)
        }
        
        if vmax > vmin:
            scaled = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            scaled = np.zeros_like(image, dtype=float)
        
        img_8bit = (scaled * 255).astype(np.uint8)
        
        if hasattr(self, 'invert_enabled') and self.invert_enabled:
            img_8bit = 255 - img_8bit
        
        cmap = self.COLORMAPS.get(self.colormap)
        if cmap is not None:
            img_colored = cv2.applyColorMap(img_8bit, cmap)
        else:
            img_colored = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
        
        return img_colored
    
    def get_stats(self) -> dict:
        """Get last computed image statistics."""
        return self.last_stats.copy()


class CameraStreamThread(QThread):
    """Camera streaming thread with diagnostics."""
    
    frame_ready = pyqtSignal(object)
    frame_dropped = pyqtSignal()
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera, target_fps: int = 20, parent=None):
        super().__init__(parent)
        
        self.camera = camera
        self.target_fps = target_fps
        self.stop_flag = False
        
        self.color_manager = ColorScaleManager()
        
        self.mutex = QMutex()
        
        self.frame_count = 0
        self.dropped_frames = 0
        self.none_count = 0  # ✅ NEW: Track None returns
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
    
    def set_target_fps(self, fps: int):
        """Set target frame rate."""
        with QMutexLocker(self.mutex):
            self.target_fps = max(1, min(fps, 60))
    
    def set_colormap(self, colormap: str):
        """Set colormap (thread-safe)."""
        with QMutexLocker(self.mutex):
            self.color_manager.set_colormap(colormap)
    
    def set_color_scale_mode(self, mode: str):
        """Set color scale mode ('auto' or 'manual')."""
        with QMutexLocker(self.mutex):
            self.color_manager.set_mode(mode)
    
    def set_color_scale_range(self, min_val: int, max_val: int):
        """Set manual color scale range."""
        with QMutexLocker(self.mutex):
            self.color_manager.set_manual_range(min_val, max_val)
    
    def set_fourier_mode(self, enabled: bool):
        """Enable/disable Fourier transform (thread-safe)."""
        with QMutexLocker(self.mutex):
            self.color_manager.set_fourier_mode(enabled)
    
    def run(self):
        """Main thread loop."""
        print(f"[CameraStream] Started (target {self.target_fps} fps)")
        
        if hasattr(self.camera, 'is_streaming'):
            print(f"[CameraStream] Camera streaming status: {self.camera.is_streaming()}")
        
        consecutive_nones = 0
        max_consecutive_nones = 1000

        try:
            while not self.stop_flag:
                # Get the current exposure to know how long to sleep
                try:
                    exposure_s = self.camera.get_exposure_time()
                except Exception:
                    exposure_s = 1.0 / self.target_fps

                # Target interval — never faster than exposure time
                target_interval = max(exposure_s * 1.05, 1.0 / self.target_fps)

                # Try to get a frame
                try:
                    frame = self.camera.read_next_image()
                except Exception as e:
                    self.error_occurred.emit(f"Camera acquisition error: {e}")
                    print(f"[CameraStream] Error reading frame: {e}")
                    time.sleep(0.1)
                    continue

                if frame is None:
                    consecutive_nones += 1
                    self.none_count += 1

                    if consecutive_nones == 10:
                        print(f"[CameraStream] ⚠️ Getting many None frames (exposure might be long)")
                    elif consecutive_nones == 50:
                        print(f"[CameraStream] ⚠️ Still no frames after 50 attempts")
                        print(f"[CameraStream]    Current exposure: {exposure_s:.3f}s")
                    elif consecutive_nones >= max_consecutive_nones:
                        # Don't kill the thread -- attempt a self-healing acquisition reset.
                        # Recovers from camera buffer getting stuck (e.g. after a sweep).
                        print(f"[CameraStream] ⚠️ No frames after {max_consecutive_nones} attempts -- resetting acquisition")
                        try:
                            if hasattr(self.camera, 'stop_streaming'):
                                self.camera.stop_streaming()
                            time.sleep(0.2)
                            if hasattr(self.camera, 'start_streaming'):
                                self.camera.start_streaming()
                            print("[CameraStream] ✅ Acquisition reset -- resuming")
                        except Exception as reset_err:
                            print(f"[CameraStream] ❌ Reset failed: {reset_err}")
                        consecutive_nones = 0
                        time.sleep(0.5)
                        continue

                    # Sleep close to one full exposure before polling again —
                    # the camera delivers at most 1 frame per exposure period.
                    # Clamp between 5ms (very short exposures) and 500ms (very long).
                    time.sleep(max(0.005, min(exposure_s * 0.9, 0.5)))
                    continue

                # Got a frame
                if consecutive_nones > 0:
                    print(f"[CameraStream] ✅ Got frame after {consecutive_nones} None returns")
                consecutive_nones = 0

                # Process and emit
                with QMutexLocker(self.mutex):
                    try:
                        processed = self.color_manager.apply_scaling(frame)
                        stats = self.color_manager.get_stats()
                    except Exception as e:
                        self.error_occurred.emit(f"Frame processing error: {e}")
                        continue

                self.frame_ready.emit(processed)
                self.stats_updated.emit(stats)
                self.frame_count += 1

                # Report FPS every 5 seconds
                now = time.time()
                if now - self.last_fps_time >= 5.0:
                    self.actual_fps = self.frame_count / (now - self.last_fps_time)
                    print(f"[CameraStream] FPS: {self.actual_fps:.1f}, None returns: {self.none_count}")
                    self.frame_count = 0
                    self.none_count = 0
                    self.last_fps_time = now

                # Sleep for one full exposure interval before polling again —
                # the camera won't have a new frame until then anyway
                time.sleep(target_interval)

        except Exception as e:
            self.error_occurred.emit(f"Camera stream fatal error: {e}")
            print(f"[CameraStream] Fatal error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"[CameraStream] Stopped (total frames: {self.frame_count})")
    
    def stop(self):
        """Stop the camera stream."""
        self.stop_flag = True
        self.wait(2000)
    
    def get_actual_fps(self) -> float:
        """Get actual achieved frame rate."""
        return self.actual_fps