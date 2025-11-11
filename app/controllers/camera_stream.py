
# ============================================================================
# FILE 1: app/controllers/camera_stream.py - MODIFIED (OPTIMIZED)
# ============================================================================

"""
Camera Stream Controller - WITH FOURIER TRANSFORM SUPPORT

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
        self.fourier_mode = False  # NEW
    
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
        """
        Apply 2D Fourier transform to image.
        Returns magnitude spectrum ready for color scaling.
        """
        if not HAS_SCIPY:
            return image  # Fallback
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.dtype == np.uint16:
            gray = (image / 256).astype(np.uint8)
        else:
            gray = image
        
        # 2D FFT
        f_transform = fft.fft2(gray)
        f_shifted = fft.fftshift(f_transform)
        
        # Magnitude spectrum (log scale)
        magnitude = np.abs(f_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # Return as float32 for scaling
        return magnitude_log.astype(np.float32)
    
    def apply_scaling(self, image: np.ndarray) -> np.ndarray:
        """Apply color scaling to image - HANDLES BOTH REAL AND FOURIER."""
        if image is None or image.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Step 1: Apply Fourier transform if enabled
        if self.fourier_mode and HAS_SCIPY:
            image = self.apply_fourier_transform(image)
        
        # Step 2: Normalize to uint16 range if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Already processed (Fourier), scale to 0-65535
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
            else:
                image = np.zeros_like(image, dtype=np.uint16)
        
        # Step 3: Calculate scaling range
        if self.mode == 'auto':
            try:
                vmin = np.percentile(image, self.auto_percentile[0])
                vmax = np.percentile(image, self.auto_percentile[1])
            except Exception:
                vmin, vmax = image.min(), image.max()
        else:
            vmin, vmax = self.min_val, self.max_val
        
        # Update stats
        self.last_stats = {
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'scale_min': float(vmin),
            'scale_max': float(vmax)
        }
        
        # Step 4: Scale to 0-255
        if vmax > vmin:
            scaled = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            scaled = np.zeros_like(image, dtype=float)
        
        img_8bit = (scaled * 255).astype(np.uint8)
        
        # Step 5: Check for inversion
        if hasattr(self, 'invert_enabled') and self.invert_enabled:
            img_8bit = 255 - img_8bit
        
        # Step 6: Apply colormap
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
    """
    Camera streaming thread with Fourier transform support.
    """
    
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
        interval = 1.0 / self.target_fps
        last_emit = 0
        
        print(f"[CameraStream] Started (target {self.target_fps} fps)")
        
        try:
            while not self.stop_flag:
                loop_start = time.time()
                
                # Acquire frame
                try:
                    frame = self.camera.acquire_single_image()
                except Exception as e:
                    self.error_occurred.emit(f"Camera acquisition error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Check if enough time has passed
                now = time.time()
                if now - last_emit < interval:
                    self.dropped_frames += 1
                    self.frame_dropped.emit()
                    time.sleep(interval * 0.5)
                    continue
                
                # Process frame (FFT happens here in worker thread!)
                with QMutexLocker(self.mutex):
                    try:
                        processed = self.color_manager.apply_scaling(frame)
                        stats = self.color_manager.get_stats()
                    except Exception as e:
                        self.error_occurred.emit(f"Frame processing error: {e}")
                        continue
                
                # Emit processed frame
                self.frame_ready.emit(processed)
                self.stats_updated.emit(stats)
                last_emit = now
                self.frame_count += 1
                
                # Update FPS
                if now - self.last_fps_time >= 1.0:
                    self.actual_fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.dropped_frames = 0
                    self.last_fps_time = now
                
                # Avoid busy loop
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep((interval - elapsed) * 0.5)
        
        except Exception as e:
            self.error_occurred.emit(f"Camera stream fatal error: {e}")
        
        finally:
            print(f"[CameraStream] Stopped")
    
    def stop(self):
        """Stop the camera stream."""
        self.stop_flag = True
        self.wait(2000)
    
    def get_actual_fps(self) -> float:
        """Get actual achieved frame rate."""
        return self.actual_fps