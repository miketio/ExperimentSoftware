"""
Camera Stream Controller

Manages camera acquisition in a separate thread with color scaling.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import time
import numpy as np
import cv2
from typing import Optional


class ColorScaleManager:
    """
    Manages color scaling and colormap application for camera images.
    
    Handles:
    - Auto-scaling (percentile-based)
    - Manual min/max scaling
    - Colormap application (gray, jet, hot, viridis, etc.)
    """
    
    # Available colormaps
    COLORMAPS = {
        'gray': None,  # No colormap = grayscale
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'turbo': cv2.COLORMAP_TURBO,
        'rainbow': cv2.COLORMAP_RAINBOW,
    }
    
    def __init__(self):
        self.mode = 'auto'  # 'auto' or 'manual'
        self.min_val = 0
        self.max_val = 4095
        self.auto_percentile = (1, 99)  # percentiles for auto-scaling
        self.colormap = 'gray'
        self.last_stats = {'min': 0, 'max': 4095, 'mean': 0}
    
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
    
    def apply_scaling(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color scaling to image.
        
        Args:
            image: Raw image (uint16 or uint8)
        
        Returns:
            8-bit RGB image (H, W, 3) for display
        """
        if image is None or image.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Calculate scaling range
        if self.mode == 'auto':
            try:
                vmin = np.percentile(image, self.auto_percentile[0])
                vmax = np.percentile(image, self.auto_percentile[1])
            except Exception:
                vmin, vmax = image.min(), image.max()
        else:
            vmin, vmax = self.min_val, self.max_val
        
        # Update stats (for display)
        self.last_stats = {
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'scale_min': float(vmin),
            'scale_max': float(vmax)
        }
        
        # Scale to 0-255
        if vmax > vmin:
            scaled = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            scaled = np.zeros_like(image, dtype=float)
        
        img_8bit = (scaled * 255).astype(np.uint8)
        
        # Apply colormap
        cmap = self.COLORMAPS.get(self.colormap)
        if cmap is not None:
            # Apply OpenCV colormap
            img_colored = cv2.applyColorMap(img_8bit, cmap)
        else:
            # Grayscale - convert to RGB for consistency
            img_colored = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
        
        return img_colored
    
    def get_stats(self) -> dict:
        """Get last computed image statistics."""
        return self.last_stats.copy()


class CameraStreamThread(QThread):
    """
    Camera streaming thread.
    
    Continuously acquires frames from camera and emits processed images.
    Handles color scaling in this thread to avoid blocking UI.
    """
    
    frame_ready = pyqtSignal(object)  # numpy array (8-bit RGB)
    frame_dropped = pyqtSignal()
    stats_updated = pyqtSignal(dict)  # image statistics
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera, target_fps: int = 20, parent=None):
        """
        Initialize camera stream.
        
        Args:
            camera: Camera object (MockCamera or ZylaCamera)
            target_fps: Target frames per second
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self.camera = camera
        self.target_fps = target_fps
        self.stop_flag = False
        
        self.color_manager = ColorScaleManager()
        
        # Thread safety
        self.mutex = QMutex()
        
        # Performance tracking
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
                
                # Check if enough time has passed since last emit
                now = time.time()
                if now - last_emit < interval:
                    # Skip frame to maintain target FPS
                    self.dropped_frames += 1
                    self.frame_dropped.emit()
                    time.sleep(interval * 0.5)  # Small sleep
                    continue
                
                # Process frame (with thread-safe color manager access)
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
                
                # Update FPS calculation every second
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
        self.wait(2000)  # Wait up to 2 seconds
    
    def get_actual_fps(self) -> float:
        """Get actual achieved frame rate."""
        return self.actual_fps