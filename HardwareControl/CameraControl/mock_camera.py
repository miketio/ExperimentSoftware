# mock_camera_um.py
"""
Mock camera implementation that renders synthetic images based on stage position (in micrometers).
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
    All units are in micrometers (µm).
    """

    def __init__(self, layout_config_path: str = "config/mock_layout.json", stage_ref=None):
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
        ground_truth = self.layout["simulation_ground_truth"]
        self.converter.set_transformation(
            ground_truth["rotation_deg"],
            tuple(ground_truth["translation_um"])
        )

        # Stage reference
        self.stage = stage_ref

        # Camera parameters
        self.sensor_width = 2048
        self.sensor_height = 2048
        self.um_per_pixel = 0.3  # µm per pixel

        # Current settings
        self.exposure_time = 0.02  # seconds
        self.bit_depth_mode = "16-bit (low noise & high well capacity)"
        self.roi = None  # (left, top, width, height)
        self.is_streaming = False
        self._last_frame = None

        print(f"[MockCamera] Initialized (micrometer-based)")
        print(f"  Sensor: {self.sensor_width}x{self.sensor_height} px")
        print(f"  Scale: {self.um_per_pixel} µm/px")
        print(f"  FOV: {self.sensor_width * self.um_per_pixel:.1f} × "
              f"{self.sensor_height * self.um_per_pixel:.1f} µm")

    # =====================================================================
    # Connection Management
    # =====================================================================

    def connect(self) -> None:
        print("[MockCamera] Connected (mock)")

    def disconnect(self) -> None:
        self.is_streaming = False
        print("[MockCamera] Disconnected (mock)")

    def get_camera_info(self) -> dict:
        return {
            "model": "MockZyla",
            "serial": "MOCK-UM-12345",
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height,
        }

    def get_sensor_size(self) -> Tuple[int, int]:
        return (self.sensor_width, self.sensor_height)

    # =====================================================================
    # Configuration
    # =====================================================================

    def set_exposure_time(self, seconds: float) -> None:
        self.exposure_time = seconds
        print(f"[MockCamera] Exposure set to {seconds:.4f}s")

    def get_exposure_time(self) -> float:
        return self.exposure_time

    def set_bit_depth_mode(self, mode: str) -> None:
        self.bit_depth_mode = mode
        print(f"[MockCamera] Bit depth mode: {mode}")

    def set_roi(self, left: int, top: int, width: int, height: int) -> None:
        self.roi = (left, top, width, height)
        print(f"[MockCamera] ROI set: left={left}, top={top}, width={width}, height={height}")

    # =====================================================================
    # Image Acquisition
    # =====================================================================

    def acquire_single_image(self) -> np.ndarray:
        """Acquire a single synthetic image based on stage position (µm)."""
        if self.stage is None:
            raise RuntimeError("MockCamera: No stage reference set")

        Y_um = self.stage.get_pos("y")
        Z_um = self.stage.get_pos("z")
        X_um = self.stage.get_pos("x")

        img = self._render_image(Y_um, Z_um, X_um)

        if self.roi is not None:
            l, t, w, h = self.roi
            img = img[t:t + h, l:l + w].copy()

        if self._software_gain != 1.0:
            img = np.clip(img * self._software_gain, 0, 65535).astype(np.uint16)

        return img

    def start_streaming(self) -> None:
        self.is_streaming = True
        print("[MockCamera] Streaming started")

    def stop_streaming(self) -> None:
        self.is_streaming = False
        print("[MockCamera] Streaming stopped")

    def read_next_image(self) -> Optional[np.ndarray]:
        if not self.is_streaming:
            return None
        return self.acquire_single_image()

    # =====================================================================
    # Image Rendering
    # =====================================================================

    def _render_image(self, Y_center_um: float, Z_center_um: float, X_um: float) -> np.ndarray:
        """Render synthetic image at given camera position (all µm)."""
        rng = np.random.default_rng()
        base_level = rng.integers(300, 600, size=(self.sensor_height, self.sensor_width), dtype=np.uint16)
        sensor_noise = rng.normal(0, 50, size=(self.sensor_height, self.sensor_width))
        img = np.clip(base_level + sensor_noise, 0, 65535).astype(np.uint16)

        half_fov_Y = (self.sensor_width * self.um_per_pixel) / 2
        half_fov_Z = (self.sensor_height * self.um_per_pixel) / 2

        Y_min, Y_max = Y_center_um - half_fov_Y, Y_center_um + half_fov_Y
        Z_min, Z_max = Z_center_um - half_fov_Z, Z_center_um + half_fov_Z

        for block_id, block in self.layout["blocks"].items():
            for corner, local_pos in block["fiducials"].items():
                self._render_fiducial(img, block_id, local_pos, corner,
                                      Y_center_um, Z_center_um, Y_min, Y_max, Z_min, Z_max)
            for wg_id, wg in block["waveguides"].items():
                self._render_waveguide(img, block_id, wg,
                                       Y_center_um, Z_center_um, Y_min, Y_max, Z_min, Z_max)
            for gr_id, gr in block["gratings"].items():
                self._render_grating(img, block_id, gr,
                                     Y_center_um, Z_center_um, Y_min, Y_max, Z_min, Z_max)

        img = self._apply_focus_blur(img, X_um)
        exposure_scale = self.exposure_time / 0.02
        img = np.clip(img * exposure_scale, 0, 65535).astype(np.uint16)
        return img

    def _stage_to_pixel(self, Y_um: float, Z_um: float,
                        Y_center_um: float, Z_center_um: float) -> Tuple[int, int]:
        """Convert stage position (µm) to pixel coordinates."""
        dY = Y_um - Y_center_um
        dZ = Z_um - Z_center_um
        px = int(round(self.sensor_width / 2 + dY / self.um_per_pixel))
        py = int(round(self.sensor_height / 2 + dZ / self.um_per_pixel))
        return px, py

    # =====================================================================
    # Render Objects
    # =====================================================================

    def _render_fiducial(self, img, block_id, local_pos, corner,
                         Y_center_um, Z_center_um, Y_min, Y_max, Z_min, Z_max):
        """Render L-shaped fiducial marker (µm-based)."""
        Y_um, Z_um = self.converter.block_local_to_stage(block_id, local_pos[0], local_pos[1])
        if not (Y_min <= Y_um <= Y_max and Z_min <= Z_um <= Z_max):
            return

        size_um = 40
        thickness_px = 8
        brightness = 3000

        # Define arm directions
        if corner == "bottom_left":
            horiz = (size_um, 0.0)
            vert = (0.0, +size_um)
        elif corner == "top_left":
            horiz = (size_um, 0.0)
            vert = (0.0, -size_um)
        elif corner == "bottom_right":
            horiz = (-size_um, 0.0)
            vert = (0.0, +size_um)
        elif corner == "top_right":
            horiz = (-size_um, 0.0)
            vert = (0.0, -size_um)
        else:
            return

        u0, v0 = local_pos
        h_end = self.converter.block_local_to_stage(block_id, u0 + horiz[0], v0 + horiz[1])
        v_end = self.converter.block_local_to_stage(block_id, u0 + vert[0], v0 + vert[1])
        start_stage = self.converter.block_local_to_stage(block_id, u0, v0)

        p0 = self._stage_to_pixel(*start_stage, Y_center_um, Z_center_um)
        ph = self._stage_to_pixel(*h_end, Y_center_um, Z_center_um)
        pv = self._stage_to_pixel(*v_end, Y_center_um, Z_center_um)

        cv2.line(img, p0, ph, brightness, thickness=thickness_px)
        cv2.line(img, p0, pv, brightness, thickness=thickness_px)
        cv2.circle(img, p0, max(1, thickness_px // 2), brightness, -1)

    def _render_waveguide(self, img, block_id, wg,
                          Y_center_um, Z_center_um, Y_min, Y_max, Z_min, Z_max):
        """Render rectangular waveguide in µm."""
        u_start, u_end = wg["u_start"], wg["u_end"]
        v_center, width = wg["v_center"], wg["width"]
        v_bottom, v_top = v_center - width / 2, v_center + width / 2

        corners_stage = [
            self.converter.block_local_to_stage(block_id, u_start, v_bottom),
            self.converter.block_local_to_stage(block_id, u_end, v_bottom),
            self.converter.block_local_to_stage(block_id, u_end, v_top),
            self.converter.block_local_to_stage(block_id, u_start, v_top),
        ]

        if not any(Y_min <= Y <= Y_max and Z_min <= Z <= Z_max for Y, Z in corners_stage):
            return

        pixels = [self._stage_to_pixel(Y, Z, Y_center_um, Z_center_um) for Y, Z in corners_stage]
        self._fill_polygon(img, pixels, brightness=3000)

    def _render_grating(self, img, block_id, grating,
                        Y_center_um, Z_center_um, Y_min, Y_max, Z_min, Z_max):
        """Render grating coupler as bright spot (µm)."""
        local_pos = grating["position"]
        Y_um, Z_um = self.converter.block_local_to_stage(block_id, local_pos[0], local_pos[1])
        if not (Y_min <= Y_um <= Y_max and Z_min <= Z_um <= Z_max):
            return
        px, py = self._stage_to_pixel(Y_um, Z_um, Y_center_um, Z_center_um)
        self._draw_circle(img, px, py, radius=8, brightness=2000)

    # =====================================================================
    # Drawing Helpers
    # =====================================================================

    def _fill_polygon(self, img, pts, brightness=3000):
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(img, [pts], brightness)

    def _draw_circle(self, img, px, py, radius, brightness=2000):
        cv2.circle(img, (px, py), radius, brightness, -1)

    def _apply_focus_blur(self, img: np.ndarray, X_um: float) -> np.ndarray:
        """Apply Gaussian blur based on defocus in µm."""
        blur_coefficient = 0.001  # sigma per µm of defocus
        sigma = abs(X_um) * blur_coefficient
        if sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        return img

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from config.layout_config_generator_v2 import load_layout_config_v2, plot_layout_v2

    print("[TEST] Running µm-based MockCamera standalone test...")

    # ----------------------------------------------------
    # 1️⃣ Load layout and ground-truth transformation
    # ----------------------------------------------------
    layout_path = "config/mock_layout.json"
    layout = load_layout_config_v2(layout_path)
    converter = CoordinateTransform(layout)

    # Plot layout for visual verification
    plot_layout_v2(layout, "config/mock_layout_um.png")

    gt = layout["simulation_ground_truth"]
    converter.set_transformation(gt["rotation_deg"], tuple(gt["translation_um"]))

    # ----------------------------------------------------
    # 2️⃣ Dummy Stage (in µm)
    # ----------------------------------------------------
    class DummyStage:
        """Minimal stage mock that reports x, y, z in µm."""
        def __init__(self):
            self.pos = {"x": 0.0, "y": 0.0, "z": 0.0}

        def get_pos(self, axis: str) -> float:
            return self.pos[axis]

        def set_pos(self, axis: str, val: float) -> None:
            self.pos[axis] = val

    stage = DummyStage()

    # ----------------------------------------------------
    # 3️⃣ Initialize camera
    # ----------------------------------------------------
    cam = MockCamera(stage_ref=stage)
    cam.converter = converter

    # ----------------------------------------------------
    # 4️⃣ Test: Single fiducial rendering (top-left)
    # ----------------------------------------------------
    print("[TEST] Rendering single top-left fiducial in µm coordinates...")

    img_single = np.zeros((cam.sensor_height, cam.sensor_width), dtype=np.uint16)
    block_id = list(layout["blocks"].keys())[0]
    block = layout["blocks"][block_id]
    fid = block["fiducials"]["top_left"]

    cam._render_fiducial(
        img_single,
        block_id,
        fid,
        "top_left",
        Y_center_um=0.0,
        Z_center_um=0.0,
        Y_min=-1000.0, Y_max=1000.0,
        Z_min=-1000.0, Z_max=1000.0,
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(img_single, cmap="gray", origin="lower")
    plt.title("Single Fiducial (Top Left, µm-based)")
    plt.tight_layout()
    plt.savefig("fiducial_test_top_left_um.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    print("[TEST] Saved -> fiducial_test_top_left_um.png")

    # ----------------------------------------------------
    # 5️⃣ Test: Full image acquisition (mock frame)
    # ----------------------------------------------------
    print("[TEST] Generating full mock camera image via acquire_single_image()...")

    # Center the stage roughly around layout origin
    stage.set_pos("y", 0.0)
    stage.set_pos("z", 0.0)
    stage.set_pos("x", 0.0)

    img_full = cam.acquire_single_image()

    plt.figure(figsize=(8, 8))
    plt.imshow(img_full, cmap="gray", origin="lower")
    plt.title("Mock Camera - acquire_single_image() Output (µm-based)")
    plt.xlabel("Y (pixels)")
    plt.ylabel("Z (pixels)")
    plt.tight_layout()
    plt.savefig("mock_camera_acquire_single_image_um.png", dpi=200)
    plt.show()

    print("[TEST] Saved -> mock_camera_acquire_single_image_um.png")
    print("[TEST] Done (micrometer version).")
