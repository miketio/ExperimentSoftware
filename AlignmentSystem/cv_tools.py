"""
cv_tools.py - slimmed VisionTools that uses external GMarkerDetector.

This file replaces the previous heavy VisionTools implementation and keeps only
the utilities needed for integration with gmarker_detector.py.

Requirements:
 - gmarker_detector.py must provide `GMarkerDetector` with a `detect(image, expected_pos, search_radius)` API.
 - OpenCV (cv2) and numpy must be available.

API (important methods):
 - VisionTools.find_fiducial_auto(image, expected_position, search_radius, template=None)
     Tries GMarkerDetector first; falls back to simple multi-scale template matching (optional).
 - VisionTools.find_fiducial_gmarker(...)  (thin adapter to GMarkerDetector)
 - VisionTools.find_fiducial_template(...) (optional fallback)
 - VisionTools.visualize_detection(image, expected_pos, found_pos, search_radius, show=False)
     Returns annotated RGB uint8 image; if show=True will display with matplotlib.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import numpy as np
import cv2

# Try to import the external detector
try:
    from AlignmentSystem.gmarkerDetector import GMarkerDetector  # your separate file
    _GDETECTOR_AVAILABLE = True
except Exception:
    GMarkerDetector = None
    _GDETECTOR_AVAILABLE = False


@dataclass
class DetectionResult:
    """Structured (lightweight) result from fiducial detection."""
    position: Tuple[int, int]
    confidence: float
    method: str
    corner_score: float = 0.0
    centroid: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = None


class VisionTools:
    """Minimal VisionTools integrating the external GMarkerDetector."""

    def __init__(self, temp_image_path: str = "tempImage.npy", gmarker_detector: Optional[Any] = None):
        """
        Args:
            temp_image_path: path used by save/load_image helpers
            gmarker_detector: optionally pass an instantiated GMarkerDetector object
        """
        self.temp_image_path = temp_image_path
        self.last_image: Optional[np.ndarray] = None

        # Instantiate or attach GMarkerDetector if available
        if gmarker_detector is not None:
            self.gdet = gmarker_detector
            self.gdet_available = True
        elif _GDETECTOR_AVAILABLE and GMarkerDetector is not None:
            try:
                self.gdet = GMarkerDetector()
                self.gdet_available = True
            except Exception:
                self.gdet = None
                self.gdet_available = False
        else:
            self.gdet = None
            self.gdet_available = False

    # -------------------------
    # Image helpers
    # -------------------------
    def save_image(self, image: np.ndarray) -> str:
        """Save last image to temporary file (numpy .npy)."""
        np.save(self.temp_image_path, image)
        self.last_image = image
        return self.temp_image_path

    def load_image(self) -> Optional[np.ndarray]:
        """Load image from temporary file if present."""
        p = Path(self.temp_image_path)
        if p.exists():
            self.last_image = np.load(self.temp_image_path)
            return self.last_image
        return None

    def measure_intensity(self, image: Optional[np.ndarray] = None,
                         roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Return simple intensity statistics (for diagnostic)."""
        if image is None:
            image = self.load_image()
        if image is None:
            return {'error': 'No image available'}

        if roi is not None:
            x, y, w, h = roi
            image = image[y:y + h, x:x + w]

        return {
            'mean': float(np.mean(image)),
            'sum': float(np.sum(image)),
            'max': float(np.max(image)),
            'min': float(np.min(image)),
            'std': float(np.std(image))
        }

    def _normalize_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """
        Convert input image to uint8 grayscale robustly.
        Works well with uint16 mock images or uint8 images.
        """
        if image.dtype == np.uint16:
            # robust percentiles to avoid outliers
            lo, hi = np.percentile(image, [1, 99])
            if hi > lo:
                scaled = (image.astype(np.float32) - float(lo)) / (float(hi) - float(lo)) * 255.0
            else:
                scaled = image.astype(np.float32) * (255.0 / 65535.0)
            return np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            # general convert
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _prepare_image(self, image: np.ndarray, search_region: Optional[Tuple[int, int, int, int]] = None):
        """
        Crop (if search_region provided) and normalize to 8-bit grayscale.
        Returns (gray_image, offset) where offset is (x0,y0) of crop in full image coords.
        """
        offset = (0, 0)
        img = image
        if search_region:
            x, y, w, h = search_region
            # clamp bounds
            x = max(0, int(x))
            y = max(0, int(y))
            w = max(0, int(w))
            h = max(0, int(h))
            img = img[y:y + h, x:x + w].copy()
            offset = (x, y)

        gray = self._normalize_to_8bit(img)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        return gray, offset

    # -------------------------
    # GMarker adapter
    # -------------------------
    def find_fiducial_gmarker(self, image: np.ndarray,
                              expected_position: Optional[Tuple[int, int]] = None,
                              search_radius: int = 100) -> Optional[DetectionResult]:
        """
        Run the external GMarkerDetector.detect(...) and adapt its result.

        The external detector expects the full image and an expected center in full-image coords.
        """
        if not self.gdet_available or self.gdet is None:
            return None

        # choose expected position default to image center
        if expected_position is None:
            expected_position = (image.shape[1] // 2, image.shape[0] // 2)

        # call detector - it should handle normalization internally
        try:
            det = self.gdet.detect(image, expected_pos=(int(expected_position[0]), int(expected_position[1])),
                                   search_radius=int(search_radius))
        except Exception as e:
            # graceful fallback if external detector errors
            print(f"[VisionTools] GMarkerDetector.detect() raised: {e}")
            return None

        if not det:
            return None

        # det may be dict-like (as in provided implementation) or custom object
        if isinstance(det, dict):
            pos = det.get('position', None)
            if pos is None:
                return None
            pos = (int(pos[0]), int(pos[1]))
            conf = float(det.get('confidence', 0.0))
            method = det.get('method', 'gmarker')
            corner_score = float(det.get('contour_score', det.get('corner_score', 0.0)))
            centroid = det.get('centroid', None)
            metadata = det.get('metadata', {})
            return DetectionResult(position=pos, confidence=conf, method=f"gmarker_{method}",
                                   corner_score=corner_score, centroid=centroid, metadata=metadata)

        # if it's an object with attributes
        try:
            pos = getattr(det, 'position', None)
            if pos is None:
                return None
            pos = (int(pos[0]), int(pos[1]))
            conf = float(getattr(det, 'confidence', 0.0))
            method = getattr(det, 'method', 'gmarker')
            corner_score = float(getattr(det, 'corner_score', 0.0))
            centroid = getattr(det, 'centroid', None)
            metadata = getattr(det, 'metadata', {}) or {}
            return DetectionResult(position=pos, confidence=conf, method=f"gmarker_{method}",
                                   corner_score=corner_score, centroid=centroid, metadata=metadata)
        except Exception:
            return None

    # -------------------------
    # Simple fallback: multi-scale template matcher
    # -------------------------
    def find_fiducial_template(self, image: np.ndarray,
                               template: np.ndarray,
                               expected_position: Optional[Tuple[int, int]] = None,
                               search_radius: int = 100,
                               scales: Optional[List[float]] = None) -> Optional[DetectionResult]:
        """
        Lightweight multi-scale template matching fallback.
        Returns DetectionResult or None.

        Note: template is a grayscale image (uint8 or uint16). This matcher is simple and
        only used as a fallback when GMarkerDetector is unavailable or fails.
        """
        if template is None:
            return None

        if scales is None:
            scales = [0.8, 0.9, 1.0, 1.1, 1.25]

        # define search region around expected_position or full image
        if expected_position is None:
            ex = image.shape[1] // 2
            ey = image.shape[0] // 2
        else:
            ex, ey = int(expected_position[0]), int(expected_position[1])

        x1 = max(0, ex - search_radius)
        y1 = max(0, ey - search_radius)
        x2 = min(image.shape[1], ex + search_radius)
        y2 = min(image.shape[0], ey + search_radius)
        if x2 <= x1 or y2 <= y1:
            return None
        search_patch = image[y1:y2, x1:x2].copy()

        gray_patch = self._normalize_to_8bit(search_patch)
        tpl_gray = self._normalize_to_8bit(template)
        best_val = -1.0
        best_center = None
        best_scale = 1.0

        for s in scales:
            try:
                tpl_resized = cv2.resize(tpl_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            except Exception:
                continue
            th, tw = tpl_resized.shape[:2]
            if th < 6 or tw < 6 or th >= gray_patch.shape[0] or tw >= gray_patch.shape[1]:
                continue
            res = cv2.matchTemplate(gray_patch, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = float(max_val)
                center_x = int(max_loc[0] + tw // 2 + x1)
                center_y = int(max_loc[1] + th // 2 + y1)
                best_center = (center_x, center_y)
                best_scale = s

        if best_center is None:
            return None

        return DetectionResult(position=best_center, confidence=best_val,
                               method='template_fallback', corner_score=0.0,
                               metadata={'scale': best_scale})

    # -------------------------
    # Auto detection (public)
    # -------------------------
    def find_fiducial_auto(self, image: np.ndarray,
                           expected_position: Tuple[int, int],
                           search_radius: int = 100,
                           template: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Primary API used by the rest of the system.

        Strategy:
          1) Try GMarkerDetector (recommended)
          2) If not available or fails, try template matching fallback (if template provided)
          3) Return dict mimicking previous format for compatibility

        Returns:
            dict or None with keys: position, confidence, method, corner_score, centroid, metadata
        """
        # 1) GMarker
        if self.gdet_available:
            res = self.find_fiducial_gmarker(image, expected_position=expected_position, search_radius=search_radius)
            if res is not None:
                return {
                    'position': res.position,
                    'confidence': res.confidence,
                    'method': res.method,
                    'corner_score': res.corner_score,
                    'centroid': res.centroid,
                    'metadata': res.metadata or {'source': 'gmarker'}
                }

        # 2) Template fallback
        if template is not None:
            res_tpl = self.find_fiducial_template(image, template, expected_position=expected_position,
                                                  search_radius=search_radius)
            if res_tpl is not None:
                return {
                    'position': res_tpl.position,
                    'confidence': res_tpl.confidence,
                    'method': res_tpl.method,
                    'corner_score': res_tpl.corner_score,
                    'centroid': res_tpl.centroid,
                    'metadata': res_tpl.metadata or {'source': 'template_fallback'}
                }

        # Nothing found
        return None

    # -------------------------
    # Diagnostic helpers
    # -------------------------
    def calculate_focus_metric(self, image: np.ndarray) -> float:
        """Simple variance-of-Laplacian focus metric on normalized 8-bit image."""
        gray = self._normalize_to_8bit(image)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(lap))

    def visualize_detection(self, image: np.ndarray, expected_pos: Tuple[int, int],
                            found_pos: Optional[Tuple[int, int]], search_radius: int,
                            show: bool = False) -> np.ndarray:
        """
        Return an annotated RGB image (uint8) with expected and found markers drawn.
        If show=True, display with matplotlib.pyplot.show() (works in notebooks / GUI env).
        """
        # Normalized RGB for display
        vis_gray = self._normalize_to_8bit(image)
        vis_bgr = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)

        # Draw search radius (circle)
        ex, ey = int(expected_pos[0]), int(expected_pos[1])
        cv2.circle(vis_bgr, (ex, ey), int(search_radius), (255, 0, 0), 1, lineType=cv2.LINE_AA)

        # Expected marker (green cross)
        cv2.drawMarker(vis_bgr, (ex, ey), (0, 255, 0), cv2.MARKER_CROSS, 20, 2, line_type=cv2.LINE_AA)

        if found_pos is not None:
            fx, fy = int(found_pos[0]), int(found_pos[1])
            cv2.drawMarker(vis_bgr, (fx, fy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 2, line_type=cv2.LINE_AA)
            cv2.line(vis_bgr, (ex, ey), (fx, fy), (255, 255, 0), 1, lineType=cv2.LINE_AA)
            err = float(np.hypot(fx - ex, fy - ey))
            cv2.putText(vis_bgr, f"Err: {err:.1f}px", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(vis_bgr, "Found", (fx + 6, fy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

        if show:
            # show with matplotlib if available
            try:
                import matplotlib.pyplot as plt
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 6))
                plt.imshow(vis_rgb)
                plt.axis('off')
                plt.show()
            except Exception:
                # fallback to cv2.imshow (may fail on headless)
                try:
                    cv2.imshow("Detection", vis_bgr)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        # Return RGB image (uint8)
        return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    """
    Quick self-test for cv_tools.VisionTools.

    - Creates a synthetic uint16 image (mock-camera-like).
    - Inserts a synthetic 'Г' marker (using the external GMarkerDetector generator if available,
      otherwise draws an L-shape directly).
    - Runs VisionTools.find_fiducial_auto(...) and prints results.
    - Shows an interactive visualization with matplotlib (no files written).
    """

    import matplotlib.pyplot as plt
    import random
    import time

    # Create synthetic background image (uint16) similar to MockCamera output
    IMG_H, IMG_W = 1024, 1024
    rng = np.random.default_rng(int(time.time() % 2**31))
    background = rng.integers(8, 60, size=(IMG_H, IMG_W), dtype=np.uint16)

    # Instantiate VisionTools (it will attempt to instantiate GMarkerDetector if available)
    vt = VisionTools()

    # Choose marker placement (ensure it fits)
    cx, cy = 600, 420  # center where marker will be placed
    # Minor random perturbation for expected position to emulate imperfect guess
    expected_offset = (random.randint(-6, 6), random.randint(-6, 6))
    expected_pos = (cx + expected_offset[0], cy + expected_offset[1])

    # Try to construct a synthetic marker image to paste into the background
    tpl = None
    tpl_h = tpl_w = 200  # default template size if we need to draw it manually

    if vt.gdet_available and hasattr(vt.gdet, "_generate_gamma_template"):
        # Use gdet's internal generator when present (keeps shape realistic)
        try:
            # Many GMarkerDetector implementations take parameters (size, arm_len, thickness, orientation_deg)
            tpl = vt.gdet._generate_gamma_template(size=200, arm_len=80, thickness=14, orientation_deg=30)
            # tpl is 8-bit already in many implementations; scale to uint16 brightness like mock camera
            tpl_bin = (tpl > 16).astype(np.uint16) * 3000
            tpl_h, tpl_w = tpl_bin.shape
        except Exception:
            tpl = None

    if tpl is None:
        # Draw a simple L / Г marker into a small canvas
        tpl_h, tpl_w = 200, 200
        tpl_bin = np.zeros((tpl_h, tpl_w), dtype=np.uint16)
        arm_len = 80
        thickness = 14
        cx_t, cy_t = tpl_w // 2, tpl_h // 2

        # Horizontal arm to the right
        tpl_bin[max(0, cy_t - thickness // 2):min(tpl_h, cy_t + thickness // 2),
                cx_t: min(tpl_w, cx_t + arm_len)] = 3000
        # Vertical arm downward
        tpl_bin[cy_t: min(tpl_h, cy_t + arm_len),
                max(0, cx_t - thickness // 2): min(tpl_w, cx_t + thickness // 2)] = 3000

        # Add slight blur to emulate PSF
        tpl_bin = cv2.GaussianBlur(tpl_bin, (3, 3), 0)

    # Paste the template into the background at (cx, cy) center
    x1 = int(cx - tpl_w // 2)
    y1 = int(cy - tpl_h // 2)
    x2 = x1 + tpl_w
    y2 = y1 + tpl_h

    if x1 < 0 or y1 < 0 or x2 > IMG_W or y2 > IMG_H:
        raise RuntimeError("Template placement out of image bounds - adjust cx,cy or template size.")

    # Blend: add brightness where template has signal (safe clipping)
    region = background[y1:y2, x1:x2].astype(np.uint32)
    region = np.clip(region + tpl_bin.astype(np.uint32), 0, 65535).astype(np.uint16)
    background[y1:y2, x1:x2] = region

    # Now run detection
    print("\n" + "=" * 70)
    print("CV_TOOLS SELF-TEST (GMarker integration)")
    print("=" * 70)
    print(f"Image shape: {background.shape}, dtype: {background.dtype}")
    print(f"Marker inserted center at: ({cx}, {cy})")
    print(f"Expected search position (guessed): {expected_pos}")
    print(f"GMarkerDetector available: {vt.gdet_available}")
    print("-" * 70)

    # Provide the template (as uint8 or uint16) to find_fiducial_auto as fallback input if needed.
    # If we used vt.gdet._generate_gamma_template we had only tpl (8-bit). If we created tpl_bin, pass tpl_bin.
    template_arg = None
    if vt.gdet_available and hasattr(vt.gdet, "_generate_gamma_template") and tpl is not None:
        # If tpl came as 8-bit array from gdet generator, pass it (detector can use its own)
        template_arg = tpl
    else:
        template_arg = tpl_bin

    # Run auto finder (tries gmarker first then template fallback)
    start_t = time.time()
    detection = vt.find_fiducial_auto(background, expected_position=expected_pos, search_radius=200,
                                      template=template_arg)
    elapsed = time.time() - start_t

    if detection:
        pos = detection['position']
        conf = detection['confidence']
        method = detection['method']
        corner_score = detection.get('corner_score', None)
        metadata = detection.get('metadata', {})
        err_px = np.hypot(pos[0] - cx, pos[1] - cy)
        print("Detection result:")
        print(f"  Found position: {pos}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Method: {method}")
        print(f"  Corner score: {corner_score}")
        print(f"  Metadata: {metadata}")
        print(f"  Error to true center: {err_px:.2f} px")
        print(f"  Elapsed: {elapsed*1000:.1f} ms")
    else:
        print("No detection (returned None). Elapsed: {:.1f} ms".format(elapsed*1000))

    print("-" * 70)
    print("Showing annotated image (interactive)...")

    # Annotate for display
    vis_rgb = vt.visualize_detection(background, expected_pos, detection['position'] if detection else None,
                                     search_radius=200, show=False)

    # Display with matplotlib (blocking)
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_rgb)
    plt.title("GMarker detection — green=expected, red=found")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
