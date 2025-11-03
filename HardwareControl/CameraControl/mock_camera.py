# mock_camera.py
"""
Mock camera implementation that renders synthetic images based on stage position.
Subclass of AndorCameraBase for drop-in replacement.
"""
import numpy as np
import cv2
from typing import Optional, Tuple
from HardwareControl.CameraControl.andorCameraBase import AndorCameraBase
from AlignmentSystem.coordinate_transform import CoordinateTransform
from config.layout_config_generator_v2 import load_layout_config_v2


class MockCamera(AndorCameraBase):
    """
    Mock camera that renders synthetic microscopy images.
    
    Uses layout configuration and stage position to render what the
    camera would see, including fiducials, waveguides, and gratings.
    """
    
    def __init__(
        self,
        layout_config_path: str = "config/mock_layout.json",
        stage_ref=None
    ):
        """
        Initialize mock camera.
        
        Args:
            layout_config_path: Path to layout JSON file
            stage_ref: Reference to MockXYZStage instance
        """
        super().__init__()
        
        # Load layout and setup coordinate converter
        self.layout = load_layout_config_v2(layout_config_path)
        self.converter = CoordinateTransform(self.layout)
        
        # Use ground truth transformation from layout (simulation only!)
        ground_truth = self.layout['simulation_ground_truth']
        self.converter.set_transformation(
            ground_truth['rotation_deg'],
            tuple(ground_truth['translation_nm'])
        )
        
        # Stage reference
        self.stage = stage_ref
        
        # Camera parameters
        self.sensor_width = 2048
        self.sensor_height = 2048
        self.nm_per_pixel = 300  # 1 µm per pixel
        
        # Current settings
        self.exposure_time = 0.02  # seconds
        self.bit_depth_mode = "16-bit (low noise & high well capacity)"
        self.roi = None  # (left, top, width, height) or None for full sensor
        
        # Streaming state
        self.is_streaming = False
        self._last_frame = None
        
        print(f"[MockCamera] Initialized")
        print(f"  Sensor: {self.sensor_width}x{self.sensor_height} pixels")
        print(f"  Resolution: {self.nm_per_pixel} nm/pixel")
        print(f"  FOV: {self.sensor_width * self.nm_per_pixel / 1000:.1f} x "
              f"{self.sensor_height * self.nm_per_pixel / 1000:.1f} µm")
    
    # =====================================================================
    # Connection Management
    # =====================================================================
    
    def connect(self) -> None:
        """Mock connection - always succeeds."""
        print("[MockCamera] Connected (mock)")
    
    def disconnect(self) -> None:
        """Mock disconnection."""
        self.is_streaming = False
        print("[MockCamera] Disconnected (mock)")
    
    def get_camera_info(self) -> dict:
        """Return mock camera info."""
        return {
            'model': 'MockZyla',
            'serial': 'MOCK-12345',
            'sensor_width': self.sensor_width,
            'sensor_height': self.sensor_height
        }
    
    def get_sensor_size(self) -> Tuple[int, int]:
        """Return sensor dimensions."""
        return (self.sensor_width, self.sensor_height)
    
    # =====================================================================
    # Configuration
    # =====================================================================
    
    def set_exposure_time(self, seconds: float) -> None:
        """Set exposure time."""
        self.exposure_time = seconds
        print(f"[MockCamera] Exposure set to {seconds:.4f}s")
    
    def get_exposure_time(self) -> float:
        """Get current exposure time."""
        return self.exposure_time
    
    def set_bit_depth_mode(self, mode: str) -> None:
        """Set bit depth mode."""
        self.bit_depth_mode = mode
        print(f"[MockCamera] Bit depth mode: {mode}")
    
    def set_roi(self, left: int, top: int, width: int, height: int) -> None:
        """Set region of interest."""
        self.roi = (left, top, width, height)
        print(f"[MockCamera] ROI set: left={left}, top={top}, width={width}, height={height}")
    
    # =====================================================================
    # Image Acquisition
    # =====================================================================
    
    def acquire_single_image(self) -> np.ndarray:
        """Acquire a single image based on current stage position."""
        if self.stage is None:
            raise RuntimeError("MockCamera: No stage reference set")
        
        # Get current stage position
        Y_nm = self.stage.get_pos('y')
        Z_nm = self.stage.get_pos('z')
        X_nm = self.stage.get_pos('x')
        
        # Render image
        img = self._render_image(Y_nm, Z_nm, X_nm)
        
        # Apply ROI if set
        if self.roi is not None:
            left, top, width, height = self.roi
            img = img[top:top+height, left:left+width].copy()
        
        # Apply software gain if set
        if self._software_gain != 1.0:
            img = np.clip(img * self._software_gain, 0, 65535).astype(np.uint16)
        
        return img
    
    def start_streaming(self) -> None:
        """Start continuous acquisition."""
        self.is_streaming = True
        print("[MockCamera] Streaming started")
    
    def stop_streaming(self) -> None:
        """Stop streaming."""
        self.is_streaming = False
        print("[MockCamera] Streaming stopped")
    
    def read_next_image(self) -> Optional[np.ndarray]:
        """Read next frame in streaming mode."""
        if not self.is_streaming:
            return None
        
        # In mock mode, just acquire a new image
        return self.acquire_single_image()
    
    # =====================================================================
    # Image Rendering (The Core Logic!)
    # =====================================================================
    
    def _render_image(self, Y_center_nm: int, Z_center_nm: int, X_nm: int) -> np.ndarray:
        """
        Render synthetic image based on camera position.
        
        Args:
            Y_center_nm: Camera center Y position in nm
            Z_center_nm: Camera center Z position in nm
            X_nm: X position (affects focus)
        
        Returns:
            np.ndarray: Rendered image (uint16)
        """
        # Create more realistic background noise
        rng = np.random.default_rng()
        # Base level + structured noise (simulates uneven illumination, substrate texture)
        base_level = rng.integers(300, 600, size=(self.sensor_height, self.sensor_width), dtype=np.uint16)
        # Add Gaussian noise (simulates camera sensor noise)
        sensor_noise = rng.normal(0, 50, size=(self.sensor_height, self.sensor_width))
        img = np.clip(base_level + sensor_noise, 0, 65535).astype(np.uint16)

        # Calculate FOV bounds in stage coordinates
        half_fov_Y = (self.sensor_width * self.nm_per_pixel) / 2
        half_fov_Z = (self.sensor_height * self.nm_per_pixel) / 2
        
        Y_min = Y_center_nm - half_fov_Y
        Y_max = Y_center_nm + half_fov_Y
        Z_min = Z_center_nm - half_fov_Z
        Z_max = Z_center_nm + half_fov_Z
        
        # Render all visible elements
        for block_id, block in self.layout['blocks'].items():
            # Render fiducials
            for corner, local_pos in block['fiducials'].items():
                self._render_fiducial(
                    img, block_id, local_pos, corner,
                    Y_center_nm, Z_center_nm, Y_min, Y_max, Z_min, Z_max
                )
            
            # Render waveguides
            for wg_id, wg in block['waveguides'].items():
                self._render_waveguide(
                    img, block_id, wg,
                    Y_center_nm, Z_center_nm, Y_min, Y_max, Z_min, Z_max
                )
            
            # Render gratings
            for grating_id, grating in block['gratings'].items():
                self._render_grating(
                    img, block_id, grating,
                    Y_center_nm, Z_center_nm, Y_min, Y_max, Z_min, Z_max
                )
        
        # Apply focus blur based on X position
        img = self._apply_focus_blur(img, X_nm)
        
        # Apply exposure scaling (brighter with longer exposure)
        exposure_scale = self.exposure_time / 0.02  # Normalized to 20ms
        img = np.clip(img * exposure_scale, 0, 65535).astype(np.uint16)
        
        return img
    
    def _stage_to_pixel(self, Y_nm: float, Z_nm: float, 
                       Y_center_nm: float, Z_center_nm: float) -> Tuple[int, int]:
        """Convert stage coordinates to pixel coordinates."""
        # Relative to camera center
        dY = Y_nm - Y_center_nm
        dZ = Z_nm - Z_center_nm
        
        # Convert to pixels (centered)
        px = int(round(self.sensor_width / 2 + dY / self.nm_per_pixel))
        py = int(round(self.sensor_height / 2 + dZ / self.nm_per_pixel))

            # ADD DEBUG:
        # Uncomment when debugging specific fiducials
        # print(f"  _stage_to_pixel: stage=({Y_nm:.0f},{Z_nm:.0f}), center=({Y_center_nm:.0f},{Z_center_nm:.0f})")
        # print(f"    delta=({dY:.0f},{dZ:.0f})nm -> pixel=({px},{py})")
        
        return (px, py)
    
    def _render_fiducial(self, img, block_id, local_pos, corner,
                        Y_center_nm, Z_center_nm, Y_min, Y_max, Z_min, Z_max):
        """Render L-shaped fiducial marker — rotated with block by transforming arm endpoints."""
        # Convert fiducial center to stage coordinates (already rotated + translated)
        Y_nm, Z_nm = self.converter.block_local_to_stage(block_id, local_pos[0], local_pos[1])

        # Check if center is in FOV (early out)
        if not (Y_min <= Y_nm <= Y_max and Z_min <= Z_nm <= Z_max):
            return

        # Parameters in *pixels* (keeps behaviour similar to original)
        size_um = 40
        thickness_px = 8
        brightness = 3000

        # Convert arm length from pixels -> µm (block_local units)
        size_px = size_um * 1000 / self.nm_per_pixel  # µm to pixels
        # Corner -> which directions arms go in local (u, v) coordinates (µm)
        # We define arms along local u / v axes so rotation is applied by converter
        # Mapping chosen to match original axis-aligned behaviour before rotation:
        #  - bottom_left: horizontal +u, vertical -v
        #  - top_left:    horizontal +u, vertical +v
        #  - bottom_right:horizontal -u, vertical -v
        #  - top_right:   horizontal -u, vertical +v
        if corner == 'bottom_left':
            horiz_delta = ( size_um,  0.0)
            vert_delta  = ( 0.0, +size_um)
        elif corner == 'top_left':
            horiz_delta = ( size_um,  0.0)
            vert_delta  = ( 0.0, -size_um)
        elif corner == 'bottom_right':
            horiz_delta = (-size_um,  0.0)
            vert_delta  = ( 0.0, +size_um)
        elif corner == 'top_right':
            horiz_delta = (-size_um,  0.0)
            vert_delta  = ( 0.0, -size_um)

        # Center in local coords is local_pos (u_local, v_local) in µm
        u0, v0 = local_pos[0], local_pos[1]

        # Horizontal arm endpoints in local coords (µm)
        u_h_end = u0 + horiz_delta[0]
        v_h_end = v0 + horiz_delta[1]

        # Vertical arm endpoints in local coords (µm)
        u_v_end = u0 + vert_delta[0]
        v_v_end = v0 + vert_delta[1]

        # Transform both start (center) and ends to stage coords (nm)
        start_stage = self.converter.block_local_to_stage(block_id, u0, v0)
        h_end_stage = self.converter.block_local_to_stage(block_id, u_h_end, v_h_end)
        v_end_stage = self.converter.block_local_to_stage(block_id, u_v_end, v_v_end)

        # Convert stage coords -> pixel coords
        px0, py0 = self._stage_to_pixel(start_stage[0], start_stage[1], Y_center_nm, Z_center_nm)
        pxh, pyh = self._stage_to_pixel(h_end_stage[0], h_end_stage[1], Y_center_nm, Z_center_nm)
        pxv, pyv = self._stage_to_pixel(v_end_stage[0], v_end_stage[1], Y_center_nm, Z_center_nm)

        # Draw thick lines for arms (cv2 expects (x,y) tuples)
        # Clip endpoints to image bounds before drawing to avoid exceptions
        h1 = max(0, min(img.shape[1]-1, px0))
        v1 = max(0, min(img.shape[0]-1, py0))
        h2 = max(0, min(img.shape[1]-1, pxh))
        v2 = max(0, min(img.shape[0]-1, pyh))
        h3 = max(0, min(img.shape[1]-1, pxv))
        v3 = max(0, min(img.shape[0]-1, pyv))

        # Ensure integer coordinates
        p0 = (int(h1), int(v1))
        ph = (int(h2), int(v2))
        pv = (int(h3), int(v3))

        # Draw horizontal and vertical arms (they'll be rotated because endpoints were transformed)
        cv2.line(img, p0, ph, int(brightness), thickness=int(thickness_px))
        cv2.line(img, p0, pv, int(brightness), thickness=int(thickness_px))

        # Optionally, draw a small filled circle at center so intersection looks crisp
        cv2.circle(img, p0, max(1, thickness_px//2), int(brightness), -1)
    
    def _render_waveguide(self, img, block_id, wg,
                         Y_center_nm, Z_center_nm, Y_min, Y_max, Z_min, Z_max):
        """Render waveguide as bright rectangle."""
        # Get waveguide corners in local coords
        u_start = wg['u_start']
        u_end = wg['u_end']
        v_center = wg['v_center']
        width = wg['width']
        
        v_bottom = v_center - width / 2
        v_top = v_center + width / 2
        
        # Convert corners to stage coords
        corners_stage = [
            self.converter.block_local_to_stage(block_id, u_start, v_bottom),
            self.converter.block_local_to_stage(block_id, u_end, v_bottom),
            self.converter.block_local_to_stage(block_id, u_end, v_top),
            self.converter.block_local_to_stage(block_id, u_start, v_top)
        ]
        
        # Check if any corner is in FOV
        if not any(Y_min <= Y <= Y_max and Z_min <= Z <= Z_max for Y, Z in corners_stage):
            return
        
        # Convert to pixels
        pixels = [self._stage_to_pixel(Y, Z, Y_center_nm, Z_center_nm) for Y, Z in corners_stage]
        
        # Draw filled polygon
        self._fill_polygon(img, pixels, brightness=3000)
    
    def _render_grating(self, img, block_id, grating,
                       Y_center_nm, Z_center_nm, Y_min, Y_max, Z_min, Z_max):
        """Render grating coupler as small bright spot."""
        local_pos = grating['position']
        
        # Convert to stage coordinates
        Y_nm, Z_nm = self.converter.block_local_to_stage(block_id, local_pos[0], local_pos[1])
        
        # Check if in FOV
        if not (Y_min <= Y_nm <= Y_max and Z_min <= Z_nm <= Z_max):
            return
        
        # Convert to pixel coords
        px, py = self._stage_to_pixel(Y_nm, Z_nm, Y_center_nm, Z_center_nm)
        
        # Draw bright spot
        radius = 8
        self._draw_circle(img, px, py, radius, brightness=2000)
    
    # =====================================================================
    # Drawing Primitives
    # =====================================================================
    
    def _fill_polygon(self, img, pixels, brightness=3000):
        """Fill polygon using OpenCV."""
        pts = np.array(pixels, dtype=np.int32)
        cv2.fillPoly(img, [pts], brightness)
    
    def _draw_circle(self, img, px, py, radius, brightness=2000):
        """Draw filled circle."""
        cv2.circle(img, (px, py), radius, brightness, -1)
    
    def _apply_focus_blur(self, img: np.ndarray, X_nm: int) -> np.ndarray:
        """
        Apply gaussian blur based on distance from optimal focus (X=0).
        
        Args:
            img: Input image
            X_nm: X position in nanometers
        
        Returns:
            Blurred image
        """
        # Blur increases with distance from X=0
        blur_coefficient = 0.1  # sigma per 1000nm of defocus
        sigma = abs(X_nm) * blur_coefficient / 1000.0
        
        if sigma > 0:  # Only blur if significant
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        
        return img
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from config.layout_config_generator_v2 import load_layout_config_v2, plot_layout_v2

    print("[TEST] Running MockCamera standalone test...")

    # # --- Load layout and ground-truth transformation ---
    layout = load_layout_config_v2("config/mock_layout.json")
    converter = CoordinateTransform(layout)
    plot_layout_v2(layout, "config/mock_layout.png")
    gt = layout["simulation_ground_truth"]
    converter.set_transformation(gt["rotation_deg"], tuple(gt["translation_nm"]))

    # --- Create dummy stage (centered) ---
    class DummyStage:
        def __init__(self):
            self.pos = {"x": 0, "y": 0, "z": 0}
        def get_pos(self, axis):
            return self.pos[axis]
        def set_pos(self, axis, val):
            self.pos[axis] = val

    stage = DummyStage()

    # --- Initialize camera ---
    cam = MockCamera(stage_ref=stage)
    cam.converter = converter

    # ====================================================
    # 1️⃣ Test: Single fiducial rendering (top-left)
    # ====================================================
    print("[TEST] Generating single top-left fiducial image...")

    img_single = np.zeros((cam.sensor_height, cam.sensor_width), dtype=np.uint16)
    block_id = list(layout["blocks"].keys())[0]
    block = layout["blocks"][block_id]
    fid = block["fiducials"]["top_left"]

    cam._render_fiducial(
        img_single,
        block_id,
        fid,
        "top_left",
        Y_center_nm=0,
        Z_center_nm=0,
        Y_min=-1e6, Y_max=1e6,
        Z_min=-1e6, Z_max=1e6,
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(img_single, cmap="gray", origin="lower")
    plt.title("Single Fiducial (Top Left)")
    plt.tight_layout()
    plt.savefig("fiducial_test_top_left.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    print("[TEST] Saved -> fiducial_test_top_left.png")

    # ====================================================
    # 2️⃣ Test: Full image acquisition (mock frame)
    # ====================================================
    print("[TEST] Generating full mock camera image via acquire_single_image()...")

    # Center the stage roughly around the layout center
    stage.set_pos("y", 0)
    stage.set_pos("z", 0)
    stage.set_pos("x", 0)

    img_full = cam.acquire_single_image()

    plt.figure(figsize=(8, 8))
    plt.imshow(img_full, cmap="gray", origin="lower")  # 'lower' shows correct physical orientation
    plt.title("Mock Camera - acquire_single_image() Output")
    plt.xlabel("Y (pixels)")
    plt.ylabel("Z (pixels)")
    plt.tight_layout()
    plt.savefig("mock_camera_acquire_single_image.png", dpi=200)
    plt.show()

    print("[TEST] Saved -> mock_camera_acquire_single_image.png")
    print("[TEST] Done.")
