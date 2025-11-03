import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

@dataclass
class GMarkerResult:
    position: Tuple[int, int]
    confidence: float
    method: str
    scale: float = 1.0
    angle_deg: float = 0.0
    contour_score: float = 0.0
    hough_score: float = 0.0
    metadata: Dict = None


class GMarkerDetector:
    """
    Robust detector for 'Г' (Gamma / L-like) fiducials.

    Usage:
        det = GMarkerDetector()
        res = det.detect(image, expected_pos=(cx, cy), search_radius=120)
    """

    def __init__(self,
                 base_size: int = 256,
                 arm_length: int = 120,
                 arm_thickness: int = 16,
                 template_contrast: int = 4095):
        """
        base_size: template canvas size in pixels
        arm_length: nominal arm length in pixels (on template)
        arm_thickness: thickness in pixels
        """
        self.base_size = base_size
        self.arm_length = arm_length
        self.arm_thickness = arm_thickness
        self.template_contrast = template_contrast
        # pre-generate a canonical template oriented as 'Г' (corner at top-left)
        self.base_template = self._generate_gamma_template(
            size=self.base_size,
            arm_len=self.arm_length,
            thickness=self.arm_thickness,
            orientation_deg=0
        )

    # ----------------------
    # Utility / preproc
    # ----------------------
    def _normalize_to_8bit(self, img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint16:
            lo, hi = np.percentile(img, [1, 99])
            if hi > lo:
                scaled = (img.astype(np.float32) - lo) / (hi - lo) * 255.0
            else:
                scaled = img.astype(np.float32) * (255.0 / 65535.0)
            return np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _crop_search_region(self, img: np.ndarray, expected_pos: Tuple[int, int], radius: int):
        ex, ey = expected_pos
        x1 = max(0, ex - radius)
        y1 = max(0, ey - radius)
        x2 = min(img.shape[1], ex + radius)
        y2 = min(img.shape[0], ey + radius)
        if x2 <= x1 or y2 <= y1:
            return None, None
        patch = img[y1:y2, x1:x2].copy()
        return patch, (x1, y1)

    # ----------------------
    # Template generation
    # ----------------------
    def _generate_gamma_template(self, size: int, arm_len: int, thickness: int, orientation_deg: float = 0) -> np.ndarray:
        """
        Generate a synthetic 'Г' shaped binary template.
        orientation_deg = 0 means corner at (center) with arms extending right and down (like rotated L).
        """
        tpl = np.zeros((size, size), dtype=np.uint8)
        cx, cy = size // 2, size // 2

        # horizontal arm: from center to right
        x0 = cx
        x1 = min(size-1, cx + arm_len)
        y0 = cy - thickness // 2
        y1 = cy + thickness // 2
        tpl[max(0, y0):min(size, y1), x0:x1] = 255

        # vertical arm: from center downward
        x0 = cx - thickness // 2
        x1 = cx + thickness // 2
        y0 = cy
        y1 = min(size-1, cy + arm_len)
        tpl[y0:y1, max(0, x0):min(size, x1)] = 255

        # rotate to desired orientation
        if abs(orientation_deg) > 0.001:
            M = cv2.getRotationMatrix2D((cx, cy), orientation_deg, 1.0)
            tpl = cv2.warpAffine(tpl, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)

        # apply slight blur to approximate real imaging
        tpl = cv2.GaussianBlur(tpl, (3, 3), 0)
        return tpl

    # ----------------------
    # Candidate generation: multi-angle multi-scale template matching
    # ----------------------
    def _template_match_candidates(self, patch_gray: np.ndarray, scales: List[float], angles: List[float],
                                   top_k: int = 8, method=cv2.TM_CCOEFF_NORMED) -> List[Dict]:
        """
        Run template matching over scales & rotations; return candidate list with score & transform.
        Each candidate dict: {cx, cy, score, scale, angle, w, h}
        Coordinates are in patch coordinate system (px).
        """
        ph, pw = patch_gray.shape[:2]
        candidates = []

        for scale in scales:
            # resize base template to scale
            t_h = int(self.base_template.shape[0] * scale)
            t_w = int(self.base_template.shape[1] * scale)
            if t_h < 12 or t_w < 12 or t_h > ph or t_w > pw:
                continue
            tpl_scaled = cv2.resize(self.base_template, (t_w, t_h), interpolation=cv2.INTER_AREA)

            for angle in angles:
                # rotate template in its own canvas to avoid cropping
                cx_tpl, cy_tpl = t_w // 2, t_h // 2
                M = cv2.getRotationMatrix2D((cx_tpl, cy_tpl), angle, 1.0)
                tpl_rot = cv2.warpAffine(tpl_scaled, M, (t_w, t_h), flags=cv2.INTER_LINEAR, borderValue=0)

                # matchTemplate requires tpl smaller than image patch
                if tpl_rot.shape[0] >= ph or tpl_rot.shape[1] >= pw:
                    continue

                res = cv2.matchTemplate(patch_gray, tpl_rot, method)
                minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)

                if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    score = 1.0 - minv  # invert
                    loc = minloc
                else:
                    score = maxv
                    loc = maxloc

                # center of detected template in patch coordinates
                center_x = int(loc[0] + tpl_rot.shape[1] // 2)
                center_y = int(loc[1] + tpl_rot.shape[0] // 2)

                candidates.append({
                    'cx': center_x,
                    'cy': center_y,
                    'score': float(score),
                    'scale': float(scale),
                    'angle': float(angle),
                    'tpl_w': tpl_rot.shape[1],
                    'tpl_h': tpl_rot.shape[0]
                })

        # Sort and return top_k
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        # deduplicate nearby points (keep highest score per cluster)
        filtered = []
        taken = np.zeros(len(candidates), dtype=bool)
        for i, c in enumerate(candidates):
            if taken[i]:
                continue
            filtered.append(c)
            # suppress neighbors within 20px
            for j in range(i+1, len(candidates)):
                if np.hypot(candidates[j]['cx']-c['cx'], candidates[j]['cy']-c['cy']) < 20:
                    taken[j] = True
            if len(filtered) >= top_k:
                break

        return filtered

    # ----------------------
    # Refinement: contour L-shape scoring + Hough lines
    # ----------------------
    def _l_shape_score_and_corner(self, patch_gray: np.ndarray, center: Tuple[int, int], tpl_w: int, tpl_h: int) -> Tuple[float, Optional[Tuple[float, float]], float]:
        """
        Given a patch and coarse center, refine by thresholding, contour analysis and Hough lines.
        Returns (combined_score, (x,y) corner in patch coords or None, hough_score)
        """
        ph, pw = patch_gray.shape
        cx, cy = center

        # crop a small ROI around center
        r = int(max(tpl_w, tpl_h) * 0.9)
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(pw, cx + r)
        y2 = min(ph, cy + r)
        roi = patch_gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, None, 0.0

        # adaptive threshold (CLAHE + Otsu)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh = clahe.apply(roi)
        _, thr = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find contours
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None, 0.0

        # pick largest contour and compute L-shape score
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        box_x, box_y, box_w, box_h = cv2.boundingRect(largest)
        aspect = max(box_w, box_h) / (min(box_w, box_h) + 1e-9)
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.2) / 3.0)

        # approximate polygon complexity (L-shape tends to give ~6 vertices)
        eps = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)
        vertices_score = max(0.0, 1.0 - abs(len(approx) - 6) / 6.0)

        # convexity
        convex = cv2.isContourConvex(approx)
        convex_score = 0.0 if convex else 1.0

        contour_score = (0.4 * vertices_score + 0.4 * convex_score + 0.2 * aspect_score) * (area / (roi.shape[0]*roi.shape[1] + 1e-9))

        # Hough lines for corner refinement (on ROI)
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=int(min(box_w, box_h)*0.4), maxLineGap=10)
        best_hough_score = 0.0
        best_intersection = None

        if lines is not None and len(lines) >= 2:
            # iterate pairs to find approximately perpendicular intersection near ROI center
            for i in range(len(lines)):
                x1l, y1l, x2l, y2l = lines[i][0]
                ang1 = np.arctan2(y2l - y1l, x2l - x1l)
                for j in range(i+1, len(lines)):
                    x3l, y3l, x4l, y4l = lines[j][0]
                    ang2 = np.arctan2(y4l - y3l, x4l - x3l)
                    angle_diff = abs(abs(ang1 - ang2) - np.pi/2)
                    if angle_diff < np.pi/8:  # within 22.5 degrees
                        # compute intersection
                        inter = self._line_intersection((x1l, y1l, x2l, y2l), (x3l, y3l, x4l, y4l))
                        if inter is None:
                            continue
                        # score: line lengths * perpendicularity proximity
                        len1 = np.hypot(x2l - x1l, y2l - y1l)
                        len2 = np.hypot(x4l - x3l, y4l - y3l)
                        perp_score = 1.0 - (angle_diff / (np.pi/8))
                        # weight with area fraction
                        score = (len1 + len2) * perp_score
                        # prefer intersections near roi center
                        inter_x, inter_y = inter
                        dist_center = np.hypot(inter_x - roi.shape[1]/2, inter_y - roi.shape[0]/2)
                        proximity = np.exp(-dist_center / (max(roi.shape)/6.0))
                        final_score = score * proximity
                        if final_score > best_hough_score:
                            best_hough_score = final_score
                            # convert to patch coords
                            best_intersection = (inter_x + x1, inter_y + y1)

        # combine contour score and hough score (normalize heuristics)
        combined = 0.7 * contour_score + 0.3 * (best_hough_score / (1.0 + best_hough_score))
        return float(combined), best_intersection, float(best_hough_score)

    def _line_intersection(self, l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-9:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    # ----------------------
    # Non-maximum suppression of results
    # ----------------------
    def _nms_results(self, results: List[GMarkerResult], dist_thresh: float = 12.0) -> List[GMarkerResult]:
        if not results:
            return []
        res_sorted = sorted(results, key=lambda r: r.confidence, reverse=True)
        keep = []
        used = np.zeros(len(res_sorted), dtype=bool)
        for i, r in enumerate(res_sorted):
            if used[i]:
                continue
            keep.append(r)
            for j in range(i+1, len(res_sorted)):
                if used[j]:
                    continue
                d = np.hypot(r.position[0] - res_sorted[j].position[0], r.position[1] - res_sorted[j].position[1])
                if d < dist_thresh:
                    used[j] = True
        return keep

    # ----------------------
    # Public detect API
    # ----------------------
    def detect(self, image: np.ndarray, expected_pos: Tuple[int, int],
               search_radius: int = 150, scales: Optional[List[float]] = None,
               angles: Optional[List[float]] = None, top_k: int = 6) -> Optional[Dict]:
        """
        Detect 'Г' marker near expected_pos.

        Returns dict:
          { 'position': (x,y), 'confidence': float, 'method': 'gamma_template_consensus', 'angle_deg':float, 'scale':float, 'metadata': {...} }
        """
        gray = self._normalize_to_8bit(image)
        patch, offset = self._crop_search_region(gray, expected_pos, search_radius)
        if patch is None:
            return None

        # default scales & angles if not provided
        if scales is None:
            scales = [0.6, 0.8, 1.0, 1.2, 1.5]
        if angles is None:
            angles = list(range(-90, 181, 15))  # every 15 degrees

        candidates = self._template_match_candidates(patch, scales, angles, top_k=top_k)
        if not candidates:
            return None

        refined_results: List[GMarkerResult] = []

        for cand in candidates:
            cx, cy = cand['cx'], cand['cy']
            tpl_w, tpl_h = cand['tpl_w'], cand['tpl_h']
            tpl_score = cand['score']
            contour_score, refined_corner, hough_score = self._l_shape_score_and_corner(patch, (cx, cy), tpl_w, tpl_h)

            # if hough gives refined location, use it; otherwise use template center
            if refined_corner is not None:
                rx, ry = refined_corner
            else:
                rx, ry = cx + offset[0], cy + offset[1]

            # convert candidate center to full-image coords
            full_x = int(cx + offset[0])
            full_y = int(cy + offset[1])

            # combine metrics: template_score, contour_score, hough_score
            combined_conf = (0.5 * tpl_score) + (0.4 * contour_score) + (0.1 * np.tanh(hough_score / 100.0))
            combined_conf = float(np.clip(combined_conf, 0.0, 1.0))

            gr = GMarkerResult(
                position=(full_x, full_y),
                confidence=combined_conf,
                method='gamma_template_consensus',
                scale=cand['scale'],
                angle_deg=cand['angle'],
                contour_score=contour_score,
                hough_score=hough_score,
                metadata={'template_score': tpl_score, 'patch_offset': offset}
            )
            refined_results.append(gr)

        if not refined_results:
            return None

        # NMS and pick best
        refined_results = self._nms_results(refined_results, dist_thresh=20.0)
        best = max(refined_results, key=lambda r: r.confidence)

        return {
            'position': best.position,
            'confidence': float(best.confidence),
            'method': best.method,
            'angle_deg': float(best.angle_deg),
            'scale': float(best.scale),
            'contour_score': float(best.contour_score),
            'hough_score': float(best.hough_score),
            'metadata': best.metadata
        }


# ----------------------
# Quick self-test (synthetic)
# ----------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------
    # Self-test + display (no save)
    # ------------------------
    # Create synthetic image with background noise (uint16 like the mock camera)
    img_h, img_w = 800, 800
    img = np.random.randint(10, 40, (img_h, img_w), dtype=np.uint16)

    # instantiate detector with parameters similar to realistic markers
    det = GMarkerDetector(base_size=200, arm_length=80, arm_thickness=14)

    # Create a rotated template and paste it into the image at an arbitrary position
    tpl = det._generate_gamma_template(size=200, arm_len=80, thickness=14, orientation_deg=-90)
    tpl_bin = (tpl > 20).astype(np.uint16) * 3000  # brightness scaled like mock camera

    # paste center position (make sure it fits)
    cx, cy = 420, 300
    h, w = tpl_bin.shape
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = x1 + w, y1 + h

    if x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h:
        region = img[y1:y2, x1:x2].astype(np.uint16)
        # overlay: add template brightness where template is set
        region = np.clip(region + tpl_bin, 0, 65535).astype(np.uint16)
        img[y1:y2, x1:x2] = region
    else:
        print("Warning: template placement would fall outside image bounds. Adjust cx,cy.")

    # expected search center (small offset to emulate imperfect guess)
    expected = (cx + 5, cy - 2)

    # run detection
    result = det.detect(img, expected_pos=expected, search_radius=160)

    # Print nicely formatted results
    print("\n" + "=" * 60)
    print("G-MARKER DETECTION (SELF-TEST)")
    print("=" * 60)
    print(f"Template center inserted at: ({cx}, {cy})")
    print(f"Search expected position:   {expected}")
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print("-" * 60)

    if result:
        print("✅ Detection Succeeded")
        print(f"  Position (px):   {result['position']}")
        print(f"  Confidence:      {result['confidence']:.4f}")
        print(f"  Method:          {result['method']}")
        print(f"  Angle (deg):     {result.get('angle_deg', None)}")
        print(f"  Scale:           {result.get('scale', None)}")
        print(f"  Contour score:   {result.get('contour_score', None)}")
        print(f"  Hough score:     {result.get('hough_score', None)}")
        print(f"  Metadata:        {result.get('metadata', {})}")
    else:
        print("✗ Detection FAILED (no candidate found)")

    print("-" * 60)

    # Create a debug visualization for interactive inspection (do not save)
    vis = det._normalize_to_8bit(img)               # 8-bit grayscale
    vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # BGR for drawing

    # draw expected pos (green) and found pos (red)
    ex, ey = expected
    cv2.drawMarker(vis_color, (int(ex), int(ey)), (0, 255, 0), cv2.MARKER_CROSS, 30, 2, line_type=cv2.LINE_AA)

    if result:
        fx, fy = result['position']
        cv2.drawMarker(vis_color, (int(fx), int(fy)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 3, line_type=cv2.LINE_AA)
        # draw connecting line and annotate error
        cv2.line(vis_color, (int(ex), int(ey)), (int(fx), int(fy)), (255, 255, 0), 2, lineType=cv2.LINE_AA)
        err_px = np.hypot(fx - ex, fy - ey)
        cv2.putText(vis_color, f"Err: {err_px:.1f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Draw a rectangle showing the template placement (for debugging)
    rect_tl = (x1, y1)
    rect_br = (x2, y2)
    cv2.rectangle(vis_color, rect_tl, rect_br, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # Convert BGR -> RGB for matplotlib
    vis_rgb = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)

    # Show using matplotlib (interactive)
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_rgb)
    plt.title("G-marker detection (green=expected, red=found)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
