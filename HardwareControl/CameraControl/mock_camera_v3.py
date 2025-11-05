#!/usr/bin/env python3
# mock_camera_um_v3.py
"""
Micrometer-based MockCamera using CameraLayout + CoordinateTransformV3.

Renders synthetic microscopy images of fiducials, waveguides, and gratings
based on simulated stage position (µm).
Intended as a realistic drop-in replacement for hardware camera.
"""

from __future__ import annotations
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

from HardwareControl.CameraControl.andorCameraBase import AndorCameraBase
from AlignmentSystem.coordinate_transform_v3 import CoordinateTransformV3
from config.layout_models import CameraLayout, Block


# ============================================================
# Mock Camera Implementation
# ============================================================
class MockCamera(AndorCameraBase):
    """Simulated camera producing synthetic images in µm coordinate space."""

    def __init__(self, layout_config_path: str = "config/mock_layout.json", stage_ref=None):
        super().__init__()

        # Load CameraLayout + build transform
        self.layout: CameraLayout = CameraLayout.from_json_file(layout_config_path)
        self.converter = CoordinateTransformV3(self.layout)


       # Apply full ground truth transformation (global + per-block)
        self.converter.use_ground_truth()
                
        self.stage = stage_ref

        # Camera specs
        self.sensor_width = 2048
        self.sensor_height = 2048
        self.um_per_pixel = 0.3

        # Runtime state
        self.exposure_time = 0.02
        self.bit_depth_mode = "16-bit (low noise & high well capacity)"
        self.roi = None
        self.is_streaming = False
        self._software_gain = 1.0
        self._last_frame = None

        print(f"[MockCameraV3] Initialized")
        print(f"  Sensor: {self.sensor_width}x{self.sensor_height}px, {self.um_per_pixel:.3f} µm/px")
        print(f"  FOV: {self.sensor_width*self.um_per_pixel:.1f} × {self.sensor_height*self.um_per_pixel:.1f} µm")

    # ========================================================
    # Connection
    # ========================================================
    def connect(self):
        print("[MockCameraV3] Connected (simulated)")

    def disconnect(self):
        self.is_streaming = False
        print("[MockCameraV3] Disconnected")

    def get_camera_info(self) -> dict:
        return {
            "model": "MockZyla-V3",
            "serial": "MOCKV3-001",
            "sensor_size_px": (self.sensor_width, self.sensor_height),
            "scale_um_per_px": self.um_per_pixel,
        }

    def get_sensor_size(self) -> Tuple[int, int]:
        return (self.sensor_width, self.sensor_height)

    # ========================================================
    # Configuration
    # ========================================================
    def set_exposure_time(self, seconds: float):
        self.exposure_time = seconds
        print(f"[MockCameraV3] Exposure set to {seconds:.3f}s")

    def get_exposure_time(self) -> float:
        return self.exposure_time

    def set_bit_depth_mode(self, mode: str):
        self.bit_depth_mode = mode
        print(f"[MockCameraV3] Bit depth mode: {mode}")

    def set_roi(self, left: int, top: int, width: int, height: int):
        self.roi = (left, top, width, height)
        print(f"[MockCameraV3] ROI set: {self.roi}")

    # ========================================================
    # Image Acquisition
    # ========================================================
    def acquire_single_image(self) -> np.ndarray:
        """Render a synthetic image based on current stage (µm)."""
        if self.stage is None:
            raise RuntimeError("MockCameraV3: stage reference required")

        Y_um = self.stage.get_pos("y")
        Z_um = self.stage.get_pos("z")
        X_um = self.stage.get_pos("x")

        img = self._render_image(Y_um, Z_um, X_um)

        if self.roi is not None:
            l, t, w, h = self.roi
            img = img[t:t+h, l:l+w].copy()

        if self._software_gain != 1.0:
            img = np.clip(img * self._software_gain, 0, 65535).astype(np.uint16)

        return img

    def start_streaming(self):
        self.is_streaming = True
        print("[MockCameraV3] Streaming started")

    def stop_streaming(self):
        self.is_streaming = False
        print("[MockCameraV3] Streaming stopped")

    def read_next_image(self) -> Optional[np.ndarray]:
        return self.acquire_single_image() if self.is_streaming else None

    # ========================================================
    # Rendering
    # ========================================================
    def _render_image(self, Y_center_um: float, Z_center_um: float, X_um: float) -> np.ndarray:
        """Render synthetic view for the current stage position (µm)."""
        rng = np.random.default_rng()
        base = rng.integers(300, 600, (self.sensor_height, self.sensor_width), dtype=np.uint16)
        noise = rng.normal(0, 50, (self.sensor_height, self.sensor_width))
        img = np.clip(base + noise, 0, 65535).astype(np.uint16)

        halfY = (self.sensor_width * self.um_per_pixel) / 2
        halfZ = (self.sensor_height * self.um_per_pixel) / 2
        Ymin, Ymax = Y_center_um - halfY, Y_center_um + halfY
        Zmin, Zmax = Z_center_um - halfZ, Z_center_um + halfZ

        for block_id, block in self.layout.blocks.items():
            for name, fid in block.fiducials.items():
                self._render_fiducial(img, block_id, fid.u, fid.v, name,
                                      Y_center_um, Z_center_um, Ymin, Ymax, Zmin, Zmax)
            for wg_id, wg in block.waveguides.items():
                self._render_waveguide(img, block_id, wg.u_start, wg.u_end, wg.v_center, wg.width,
                                       Y_center_um, Z_center_um, Ymin, Ymax, Zmin, Zmax)
            for gr_id, gr in block.gratings.items():
                pos = gr.position
                self._render_grating(img, block_id, pos.u, pos.v,
                                     Y_center_um, Z_center_um, Ymin, Ymax, Zmin, Zmax)

        img = self._apply_focus_blur(img, X_um)
        img = np.clip(img * (self.exposure_time / 0.02), 0, 65535).astype(np.uint16)
        return img

    def _stage_to_pixel(self, Y_um: float, Z_um: float,
                        Y_center_um: float, Z_center_um: float) -> Tuple[int, int]:
        """Convert stage coordinates (µm) to pixel indices."""
        px = int(round(self.sensor_width/2 + (Y_um - Y_center_um)/self.um_per_pixel))
        py = int(round(self.sensor_height/2 + (Z_um - Z_center_um)/self.um_per_pixel))
        return px, py

    # ========================================================
    # Render Objects
    # ========================================================
    def _render_fiducial(self, img, block_id, u_local, v_local, corner,
                         Yc, Zc, Ymin, Ymax, Zmin, Zmax):
        """Draw an L-shaped fiducial marker."""
        Y_um, Z_um = self.converter.block_local_to_stage(block_id, u_local, v_local)
        if not (Ymin <= Y_um <= Ymax and Zmin <= Z_um <= Zmax):
            return

        size_um = 40
        thickness_px = 8
        brightness = 3000

        directions = {
            "bottom_left": ((size_um, 0.0), (0.0, +size_um)),
            "top_left": ((size_um, 0.0), (0.0, -size_um)),
            "bottom_right": ((-size_um, 0.0), (0.0, +size_um)),
            "top_right": ((-size_um, 0.0), (0.0, -size_um)),
        }
        if corner not in directions:
            return
        horiz, vert = directions[corner]

        h_end = self.converter.block_local_to_stage(block_id, u_local + horiz[0], v_local + horiz[1])
        v_end = self.converter.block_local_to_stage(block_id, u_local + vert[0], v_local + vert[1])

        p0 = self._stage_to_pixel(Y_um, Z_um, Yc, Zc)
        ph = self._stage_to_pixel(*h_end, Yc, Zc)
        pv = self._stage_to_pixel(*v_end, Yc, Zc)

        cv2.line(img, p0, ph, brightness, thickness_px)
        cv2.line(img, p0, pv, brightness, thickness_px)
        cv2.circle(img, p0, thickness_px//2, brightness, -1)

    def _render_waveguide(self, img, block_id, u_start, u_end, v_center, width,
                          Yc, Zc, Ymin, Ymax, Zmin, Zmax):
        """Draw rectangular waveguide."""
        v_bottom, v_top = v_center - width/2, v_center + width/2
        corners = [
            self.converter.block_local_to_stage(block_id, u_start, v_bottom),
            self.converter.block_local_to_stage(block_id, u_end, v_bottom),
            self.converter.block_local_to_stage(block_id, u_end, v_top),
            self.converter.block_local_to_stage(block_id, u_start, v_top),
        ]
        if not any(Ymin <= Y <= Ymax and Zmin <= Z <= Zmax for Y, Z in corners):
            return
        pts = [self._stage_to_pixel(Y, Z, Yc, Zc) for Y, Z in corners]
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 3000)

    def _render_grating(self, img, block_id, u_local, v_local,
                        Yc, Zc, Ymin, Ymax, Zmin, Zmax):
        """Draw grating coupler as bright circle."""
        Y_um, Z_um = self.converter.block_local_to_stage(block_id, u_local, v_local)
        if not (Ymin <= Y_um <= Ymax and Zmin <= Z_um <= Zmax):
            return
        px, py = self._stage_to_pixel(Y_um, Z_um, Yc, Zc)
        cv2.circle(img, (px, py), 8, 2000, -1)

    # ========================================================
    # Focus & Effects
    # ========================================================
    def _apply_focus_blur(self, img, X_um):
        """Apply Gaussian blur proportional to defocus (µm)."""
        sigma = abs(X_um) * 0.001
        if sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        return img


# ============================================================
# Standalone test mode
# ============================================================
if __name__ == "__main__":

    print("[TEST] Running micrometer-based MockCameraV3 test...")

    # Dummy stage
    class DummyStage:
        def __init__(self):
            self.pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        def get_pos(self, axis): return self.pos[axis]
        def set_pos(self, axis, val): self.pos[axis] = val

    stage = DummyStage()
    cam = MockCamera(stage_ref=stage)

    # Test: render one fiducial
    block_id = list(cam.layout.blocks.keys())[0]
    block = cam.layout.blocks[block_id]
    fid = block.fiducials["top_left"]

    img_single = np.zeros((cam.sensor_height, cam.sensor_width), dtype=np.uint16)
    cam._render_fiducial(img_single, block_id, fid.u, fid.v, "top_left",
                         Yc=0, Zc=0, Ymin=-1000, Ymax=1000, Zmin=-1000, Zmax=1000)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_single, cmap="gray", origin="lower")
    plt.title("Single Fiducial (Top Left, µm-based)")
    plt.tight_layout()
    plt.savefig("fiducial_test_v3.png", dpi=200)
    plt.show()

    # Test: full synthetic frame
    print("[TEST] Generating full mock camera frame...")
    stage.set_pos("x", 0.0)
    stage.set_pos("y", 0.0)
    stage.set_pos("z", 0.0)
    img_full = cam.acquire_single_image()

    plt.figure(figsize=(8, 8))
    plt.imshow(img_full, cmap="gray", origin="lower")
    plt.title("MockCameraV3 - Full Frame (µm-based)")
    plt.xlabel("Y (px)")
    plt.ylabel("Z (px)")
    plt.tight_layout()
    plt.savefig("mock_camera_full_frame_v3.png", dpi=200)
    plt.show()

    print("[TEST] Done. Saved plots: fiducial_test_v3.png, mock_camera_full_frame_v3.png")
