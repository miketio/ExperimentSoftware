# andor_camera_app.py
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional
from andorCameraBase import AndorCameraBase


class AndorCameraApp:
    """
    High-level application logic for Andor cameras.
    Wraps AndorCameraBase with logging, saving, and live view capabilities.
    """

    # ───────────────────────────────────────
    # 🎨 Rendering Configuration (Class-Level)
    # ───────────────────────────────────────

    # Colormap settings
    COLORMAP = cv2.COLORMAP_JET          # Use JET
    INVERT_INTENSITY = True              # True → red = low, blue = high

    # Font & label settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = (255, 255, 255)         # White
    FONT_THICKNESS = 2                   # Bold but same size

    # Colorbar settings
    COLORBAR_WIDTH = 100                 # Pixels
    COLORBAR_INTERPOLATION = cv2.INTER_NEAREST

    # Live view display limits (for auto-scaling)
    MAX_DISP_WIDTH = 1550
    MAX_DISP_HEIGHT = 800

    # Window title
    DEFAULT_WINDOW_TITLE = "Andor Live View (Red=Low, Blue=High)"

    # ───────────────────────────────────────
    # Initialization
    # ───────────────────────────────────────

    def __init__(self, camera: AndorCameraBase):
        self.camera = camera
        print("[APP] Andor camera application initialized")

    # ───────────────────────
    # Thin Wrappers (with logging)
    # ───────────────────────

    def set_exposure(self, seconds: float):
        print(f"[APP] Setting exposure to {seconds:.4f} s")
        self.camera.set_exposure_time(seconds)

    def set_gain_mode(self, mode: str):
        print(f"[APP] Setting gain mode to: '{mode}'")
        self.camera.set_bit_depth_mode(mode)

    def set_software_gain(self, factor: float):
        print(f"[APP] Setting software gain to: {factor}x")
        self.camera.set_software_gain(factor)


    def set_roi(self, left: Optional[int] = None, top: Optional[int] = None,
            width: Optional[int] = None, height: Optional[int] = None) -> None:
        print(f"[APP] Setting ROI to left={left}, top={top}, width={width}, height={height}")
        self.camera.set_roi(left, top, width, height)
        
        
    def acquire_image(self) -> np.ndarray:
        print("[APP] Acquiring single image...")
        img = self.camera.acquire_single_image()
        print(f"    → Shape: {img.shape}, dtype: {img.dtype}, min={img.min()}, max={img.max()}")
        return img

    # ───────────────────────
    # High-Level Operations
    # ───────────────────────

    def take_and_save_image(
        self,
        base_name: str = "snapshot",
        exposure_seconds: Optional[float] = None,
        gain_mode: Optional[str] = None,
        software_gain: Optional[float] = None,
    ) -> Optional[str]:
        if exposure_seconds is not None:
            self.set_exposure(exposure_seconds)
        if gain_mode is not None:
            self.set_gain_mode(gain_mode)
        if software_gain is not None:
            self.set_software_gain(software_gain)

        img = self.acquire_image()
        time.sleep(0.1)
        fname = self._save_image(img, base_name)
        if fname:
            print(f"[APP] Image saved to: {fname}")
        else:
            print("[APP] ❌ Failed to save image")
        return fname

    def start_live_view(
        self,
        max_disp_width: Optional[int] = None,
        max_disp_height: Optional[int] = None,
        window_title: Optional[str] = None,
    ):
        print("[APP] Starting live view... Press 'q' in the window or Ctrl+C to quit.")
        self.camera.start_streaming()

        # Use instance or class defaults
        w_max = max_disp_width if max_disp_width is not None else self.MAX_DISP_WIDTH
        h_max = max_disp_height if max_disp_height is not None else self.MAX_DISP_HEIGHT
        title = window_title if window_title is not None else self.DEFAULT_WINDOW_TITLE

        try:
            while True:
                image_data = self.camera.read_next_image()
                if image_data is not None:
                    # Render frame using shared method
                    combined = self._render_frame(image_data, max_width=w_max, max_height=h_max)
                    cv2.imshow(title, combined)
                    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)  # Keep on top
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[APP] Live view interrupted by user (Ctrl+C).")
        finally:
            cv2.destroyAllWindows()
            self.camera.stop_streaming()
            print("[APP] Live view stopped.")

    # ───────────────────────
    # 🔑 Core Rendering Logic (Shared)
    # ───────────────────────

    def _render_frame(self, img: np.ndarray, max_width: Optional[int] = None, max_height: Optional[int] = None) -> np.ndarray:
        """
        Render a single frame exactly like the live stream:
        - Apply colormap
        - Add colorbar
        - Add min/max labels
        - Optionally scale to fit max_width/max_height
        """
        vmin, vmax = img.min(), img.max()

        # Normalize to 8-bit
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply colormap (with optional inversion)
        if self.INVERT_INTENSITY:
            colored = cv2.applyColorMap(255 - img_norm, self.COLORMAP)
        else:
            colored = cv2.applyColorMap(img_norm, self.COLORMAP)

        h_orig, w_orig = colored.shape[:2]

        # Scale if limits provided
        if max_width is not None and max_height is not None:
            scale_w = max_width / w_orig
            scale_h = max_height / h_orig
            scale = min(scale_w, scale_h, 1.0)
            if scale < 1.0:
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                colored_disp = cv2.resize(colored, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                colored_disp = colored.copy()
        else:
            colored_disp = colored.copy()

        h_disp, w_disp = colored_disp.shape[:2]

        # Build colorbar
        colorbar = np.arange(255, -1, -1, dtype=np.uint8)[:, None, None]
        colorbar = np.tile(colorbar, (1, self.COLORBAR_WIDTH, 1))
        colorbar = cv2.applyColorMap(colorbar.astype(np.uint8), self.COLORMAP)
        colorbar_resized = cv2.resize(colorbar, (self.COLORBAR_WIDTH, h_disp), interpolation=self.COLORBAR_INTERPOLATION)

        # Combine
        combined = np.hstack((colored_disp, colorbar_resized))

        # Add labels
        cv2.putText(
            combined,
            f"Max: {vmax}",
            (w_disp + 5, 20),
            self.FONT,
            self.FONT_SCALE,
            self.FONT_COLOR,
            self.FONT_THICKNESS
        )
        cv2.putText(
            combined,
            f"Min: {vmin}",
            (w_disp + 5, h_disp - 10),
            self.FONT,
            self.FONT_SCALE,
            self.FONT_COLOR,
            self.FONT_THICKNESS
        )

        return combined

    # ───────────────────────
    # Helpers
    # ───────────────────────

    def _save_image(self, img: np.ndarray, base_name: str = "snapshot") -> Optional[str]:
        try:
            Path(base_name).parent.mkdir(parents=True, exist_ok=True)
            saved_files = []

            # Save raw
            if img.dtype == np.uint16:
                raw_fname = f"{base_name}_raw_uint16.png"
                if cv2.imwrite(raw_fname, img):
                    saved_files.append(raw_fname)
                    print(f"[APP] Saved raw 16-bit image: {raw_fname}")
            else:
                raw_fname = f"{base_name}_raw.png"
                if cv2.imwrite(raw_fname, img):
                    saved_files.append(raw_fname)
                    print(f"[APP] Saved raw image: {raw_fname}")

            # Render full-res image (no scaling)
            rendered = self._render_frame(img)  # No max_width/max_height → full res

            rendered_fname = f"{base_name}_rendered.png"
            if cv2.imwrite(rendered_fname, rendered):
                print(f"[APP] Saved rendered image: {rendered_fname}")
                return rendered_fname
            else:
                print("[APP] Warning: Failed to save rendered image.")
                return None

        except Exception as ex:
            print(f"[APP] Exception while saving image: {ex}")
            return None

    def close(self):
        self.camera.close()