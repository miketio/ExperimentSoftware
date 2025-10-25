# andorCameraBase.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class AndorCameraBase(ABC):
    """
    Abstract base class for Andor SDK3-compatible scientific cameras.

    Provides a standardized interface for:
    - Connection management
    - Configuration (exposure, gain, ROI)
    - Image acquisition (single and streaming)
    
    Implementations may support hardware ROI, but software cropping
    and digital gain are handled optionally in subclasses.
    """

    def __init__(self):
        self._software_gain: float = 1.0

    # ───────────────────────
    # Connection Management
    # ───────────────────────

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the camera hardware."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Safely close connection and deallocate resources."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_info(self) -> dict:
        """
        Return basic camera info.
        Example: {'model': 'Zyla 4.2', 'serial': '12345'}
        """
        raise NotImplementedError

    @abstractmethod
    def get_sensor_size(self) -> Tuple[int, int]:
        """Return full sensor dimensions as (width, height) in pixels."""
        raise NotImplementedError

    # ───────────────────────
    # Configuration
    # ───────────────────────

    @abstractmethod
    def set_exposure_time(self, seconds: float) -> None:
        """Set exposure time in seconds (e.g., 0.02 for 20 ms)."""
        raise NotImplementedError

    @abstractmethod
    def get_exposure_time(self) -> float:
        """Get current exposure time in seconds."""
        raise NotImplementedError

    @abstractmethod
    def set_bit_depth_mode(self, mode: str) -> None:
        """
        Set gain/bit-depth mode (e.g., '16-bit (low noise & high well capacity)').
        Must be one of the values reported by the camera.
        """
        raise NotImplementedError

    def set_software_gain(self, factor: float) -> None:
        """
        Optional: Set digital (software) gain applied during image retrieval.
        Default implementation stores the factor; subclasses may apply it in `acquire_single_image` or `read_latest_image`.
        """
        if factor <= 0:
            raise ValueError("Software gain must be positive.")
        self._software_gain = float(factor)

    def get_software_gain(self) -> float:
        """Return current software gain factor."""
        return self._software_gain

    @abstractmethod
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set hardware region of interest (ROI).
        Parameters:
            x, y: top-left corner (pixels, origin at top-left)
            width, height: ROI dimensions (pixels)
        Note: Not all cameras support arbitrary ROI; implementations should validate.
        """
        raise NotImplementedError
    
    # @abstractmethod
    # def get_roi(self) -> dict:
    #     """
    #     Get current ROI settings.
        
    #     Returns:
    #         dict: {'left': int, 'top': int, 'width': int, 'height': int}
    #     """
    #     raise NotImplementedError

    # ───────────────────────
    # Acquisition
    # ───────────────────────

    @abstractmethod
    def acquire_single_image(self) -> np.ndarray:
        """
        Acquire and return a single image as a NumPy array (dtype typically uint16).
        Should apply current exposure, bit depth, and ROI settings.
        Software gain (if used) may be applied here or in a subclass override.
        """
        raise NotImplementedError

    @abstractmethod
    def start_streaming(self) -> None:
        """Start continuous acquisition buffer."""
        raise NotImplementedError

    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop streaming and flush internal buffers."""
        raise NotImplementedError

    @abstractmethod
    def read_next_image(self) -> Optional[np.ndarray]:
        """
        Non-blocking read of the most recent available image.
        Returns None if no new frame is ready.
        Like `acquire_single_image`, may apply software gain in subclasses.
        """
        raise NotImplementedError

    # ───────────────────────
    # Context Manager
    # ───────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop_streaming()
        except Exception:
            pass  # Don't mask original exceptions
        self.disconnect()