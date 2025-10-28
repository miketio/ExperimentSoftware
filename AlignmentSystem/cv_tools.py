# cv_tools.py
"""
Computer vision tools for fiducial detection and image analysis.
"""
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


class VisionTools:
    """Computer vision tools for alignment."""
    
    def __init__(self, temp_image_path: str = "tempImage.npy"):
        self.temp_image_path = temp_image_path
        self.last_image = None
        
    def save_image(self, image: np.ndarray) -> str:
        """Save image to temporary file."""
        np.save(self.temp_image_path, image)
        self.last_image = image
        return self.temp_image_path
    
    def load_image(self) -> Optional[np.ndarray]:
        """Load image from temporary file."""
        if Path(self.temp_image_path).exists():
            self.last_image = np.load(self.temp_image_path)
            return self.last_image
        return None
    
    def measure_intensity(self, image: Optional[np.ndarray] = None, 
                         roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """
        Measure intensity metrics from image.
        
        Args:
            image: Image array (if None, load from temp file)
            roi: (x, y, width, height) region of interest
        
        Returns:
            dict with intensity metrics
        """
        if image is None:
            image = self.load_image()
        
        if image is None:
            return {'error': 'No image available'}
        
        # Apply ROI if specified
        if roi is not None:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        return {
            'mean': float(np.mean(image)),
            'sum': float(np.sum(image)),
            'max': float(np.max(image)),
            'min': float(np.min(image)),
            'std': float(np.std(image))
        }
    
    def find_fiducial_template(self, image: np.ndarray, 
                              template: np.ndarray,
                              search_region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict]:
        """
        Find fiducial using template matching.
        
        Args:
            image: Full camera image
            template: Template of fiducial marker
            search_region: (x, y, w, h) to limit search
        
        Returns:
            dict with position and confidence, or None
        """
        # Apply search region
        if search_region:
            x, y, w, h = search_region
            search_img = image[y:y+h, x:x+w]
            offset = (x, y)
        else:
            search_img = image
            offset = (0, 0)
        
        # Normalize images
        search_norm = cv2.normalize(search_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        template_norm = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Template matching
        result = cv2.matchTemplate(search_norm, template_norm, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Adjust for template size and offset
        h, w = template.shape[:2]
        center_x = max_loc[0] + w // 2 + offset[0]
        center_y = max_loc[1] + h // 2 + offset[1]
        
        return {
            'position': (center_x, center_y),
            'confidence': float(max_val),
            'method': 'template_matching'
        }
    
    def find_fiducial_corners(self, image: np.ndarray,
                             search_region: Optional[Tuple[int, int, int, int]] = None,
                             threshold: float = 0.01) -> Optional[Dict]:
        """
        Find L-shaped fiducial using corner detection.
        
        Args:
            image: Camera image
            search_region: (x, y, w, h) to limit search
            threshold: Corner detection threshold
        
        Returns:
            dict with position and confidence, or None
        """
        # Apply search region
        if search_region:
            x, y, w, h = search_region
            search_img = image[y:y+h, x:x+w]
            offset = (x, y)
        else:
            search_img = image
            offset = (0, 0)
        
        # Normalize to 8-bit
        img_norm = cv2.normalize(search_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Harris corner detection
        corners = cv2.cornerHarris(img_norm, blockSize=2, ksize=3, k=0.04)
        
        # Find strongest corner
        corner_max = corners.max()
        if corner_max > threshold:
            max_loc = np.unravel_index(corners.argmax(), corners.shape)
            center_y = max_loc[0] + offset[1]
            center_x = max_loc[1] + offset[0]
            
            return {
                'position': (center_x, center_y),
                'confidence': float(corner_max),
                'method': 'corner_detection'
            }
        
        return None
    
    def find_fiducial_contours(
        self,
        image: np.ndarray,
        search_region: Optional[Tuple[int, int, int, int]] = None,
        invert: Optional[bool] = None  # None = auto detect
    ) -> Optional[Dict]:
        """
        Find fiducial using contour detection with distance transform.

        Args:
            image: Input grayscale or color image (8-bit or 16-bit)
            search_region: Optional (x, y, w, h) to crop before detection
            invert: True = detect dark objects, False = detect bright objects, None = auto

        Returns:
            dict with keys: 'position', 'centroid', 'confidence', 'corner_score', 'method'
            or None if detection fails.
        """

        # --- Crop to search region ---
        offset = (0, 0)
        img = image.copy()
        if search_region:
            x, y, w, h = search_region
            img = img[y:y+h, x:x+w]
            offset = (x, y)

        # --- Convert to grayscale ---
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # --- Convert 16-bit to 8-bit if needed ---
        if gray.dtype == np.uint16:
            gray = (gray / 16).astype(np.uint8)  # for 12-bit images (0-4095)

        # --- Auto detect invert if not specified ---
        if invert is None:
            mean_intensity = np.mean(gray)
            invert = mean_intensity > 127  # assume bright background if mean > 127

        if invert:
            gray = cv2.bitwise_not(gray)

        # --- Threshold to binary ---
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # --- Find contours ---
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # --- Choose largest contour ---
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area <= 0:
            return None

        # --- Filter out contours that are too large (like the entire image) ---
        h, w = binary.shape
        total_area = h * w
        if area > total_area * 0.9:  # If contour is 90% of the image, it's probably wrong
            return None

        # --- Compute centroid ---
        M = cv2.moments(largest)
        centroid = None
        if M.get("m00", 0) > 0:
            cx = int(M["m10"] / M["m00"]) + offset[0]
            cy = int(M["m01"] / M["m00"]) + offset[1]
            centroid = (cx, cy)

        # --- Create filled mask ---
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        # --- Ensure mask is non-empty and not too large ---
        non_zero_pixels = cv2.countNonZero(mask)
        if non_zero_pixels == 0 or non_zero_pixels > total_area * 0.9:
            return None

        # --- Distance transform ---
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # --- Create a mask for the inner region (excluding borders) ---
        h, w = dist.shape
        inner_mask = np.ones((h, w), dtype=bool)
        border_size = 2  # Exclude 2-pixel border to be more conservative
        inner_mask[:border_size, :] = False
        inner_mask[-border_size:, :] = False
        inner_mask[:, :border_size] = False
        inner_mask[:, -border_size:] = False
        
        # --- Apply the inner mask to the distance transform ---
        masked_dist = dist.copy()
        masked_dist[~inner_mask] = 0  # Zero out border regions

        # --- Find maximum distance within the valid region ---
        max_dist = np.max(masked_dist)
        
        # --- Check for valid finite values and avoid overflow ---
        if not np.isfinite(max_dist) or max_dist <= 0 or max_dist > 1000:  # Add sanity check
            return None

        # --- Get corner position (furthest from edge, avoiding image boundaries) ---
        max_idx = np.unravel_index(np.argmax(masked_dist), masked_dist.shape)
        corner_y, corner_x = max_idx
        corner_x += offset[0]
        corner_y += offset[1]
        
        # --- Ensure corner_score is within safe float32 range ---
        corner_score = min(float(max_dist), np.finfo(np.float32).max / 2)

        return {
            "position": (int(corner_x), int(corner_y)),
            "centroid": centroid,
            "confidence": area,
            "corner_score": corner_score,
            "method": "contour_detection"
        }

    def find_fiducial_auto(self, image: np.ndarray,
                        expected_position: Tuple[int, int],
                        search_radius: int = 100,
                        template: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Automatically find fiducial using multiple methods.
        """
        ex, ey = expected_position

        # Define search region
        x1 = max(0, ex - search_radius)
        y1 = max(0, ey - search_radius)
        x2 = min(image.shape[1], ex + search_radius)
        y2 = min(image.shape[0], ey + search_radius)
        search_region = (x1, y1, x2 - x1, y2 - y1)

        results = []

        # --- Template matching ---
        if template is not None:
            try:
                result = self.find_fiducial_template(image=image, template=template, search_region=search_region)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Template matching failed: {e}")

        # --- Corner detection ---
        try:
            result = self.find_fiducial_corners(image=image, search_region=search_region)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Corner detection failed: {e}")

        # --- Contour detection (bright) ---
        try:
            result = self.find_fiducial_contours(image=image, search_region=search_region, invert=False)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Contour detection (bright) failed: {e}")

        # --- Contour detection (dark) ---
        try:
            result = self.find_fiducial_contours(image=image, search_region=search_region, invert=True)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Contour detection (dark) failed: {e}")

        # --- Return best result by confidence ---
        if results:
            best = max(results, key=lambda r: r['confidence'])
            return best

        return None
        
    def calculate_focus_metric(self, image: np.ndarray) -> float:
        """
        Calculate focus metric (variance of Laplacian).
        
        Args:
            image: Camera image
        
        Returns:
            Focus metric (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.float64)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)

    def visualize_detection(self, image, expected_pos, found_pos, search_radius):
        """Visualize detection result."""
        # Normalize image to 8-bit for display
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to color for annotations
        img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        
        # Draw search region (blue circle)
        cv2.circle(img_color, expected_pos, search_radius, (255, 0, 0), 2)
        
        # Draw expected position (green crosshair)
        cv2.drawMarker(img_color, expected_pos, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Draw found position (red crosshair)
        cv2.drawMarker(img_color, found_pos, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
        
        # Draw line connecting them
        cv2.line(img_color, expected_pos, found_pos, (255, 255, 0), 2)
        
        # Add text labels
        cv2.putText(img_color, "Expected", (expected_pos[0] + 10, expected_pos[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_color, "Found", (found_pos[0] + 10, found_pos[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show in window
        cv2.imshow("Fiducial Detection Result", img_color)
        cv2.waitKey(10000)  # Show for 10 seconds
        cv2.destroyWindow("Fiducial Detection Result")

# Test/example usage
if __name__ == "__main__":
    print("Vision Tools Module")
    print("===================")
    
    # Create synthetic test image with L-shaped marker
    test_img = np.zeros((512, 512), dtype=np.uint16)
    
    # Draw L-shape at (100, 100)
    test_img[90:110, 90:200] = 4095  # Horizontal part
    test_img[90:200, 90:110] = 4095  # Vertical part
    
    expected_pos = (150, 150)
    search_radius = 100

    vt = VisionTools()
    
    # Test intensity measurement
    metrics = vt.measure_intensity(test_img)
    print(f"\nIntensity metrics: {metrics}")
    
    # Test fiducial detection
    result = vt.find_fiducial_auto(test_img, expected_position=expected_pos, search_radius=search_radius)
    if result:
        print(f"\nFiducial found: {result}")
    else:
        print("\nFiducial not found")
    
    # Test focus metric
    focus = vt.calculate_focus_metric(test_img)
    print(f"\nFocus metric: {focus:.2f}")

    # Visualize detection
    if result:
        vt.visualize_detection(test_img, expected_pos, result['position'], search_radius)