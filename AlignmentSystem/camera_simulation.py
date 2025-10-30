# camera_simulation_optimized.py
"""
Optimized camera simulation with performance improvements and debug output.
"""
import numpy as np
import cv2
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import math
from pathlib import Path
from AlignmentSystem.ascii_parser import ASCIIParser

class Camera:
    """Simulates a camera with limited field of view - OPTIMIZED VERSION."""

    def __init__(self, pixel_width=2048, pixel_height=2048, nm_per_pixel=1000):
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.nm_per_pixel = nm_per_pixel
        self.center_position = (0, 0)  # (Y, Z) in nm
        self.fov_width_nm = pixel_width * nm_per_pixel
        self.fov_height_nm = pixel_height * nm_per_pixel
        
        # Performance tracking
        self.last_render_stats = {
            'total_elements': 0,
            'rendered': 0,
            'skipped_outside_fov': 0,
            'by_type': {}
        }

    def move_to(self, Y_nm, Z_nm):
        """Move camera center to specified stage coordinates (in nm)."""
        self.center_position = (Y_nm, Z_nm)

    def get_fov_bounds(self):
        """Get field of view boundaries in stage coordinates (nm)."""
        Y_center, Z_center = self.center_position
        Y_min = Y_center - self.fov_width_nm / 2
        Y_max = Y_center + self.fov_width_nm / 2
        Z_min = Z_center - self.fov_height_nm / 2
        Z_max = Z_center + self.fov_height_nm / 2
        return (Y_min, Y_max, Z_min, Z_max)

    def stage_to_pixel(self, Y_nm, Z_nm):
        """Convert stage coordinates to pixel coordinates."""
        Y_min, Y_max, Z_min, Z_max = self.get_fov_bounds()
        px = int((Y_nm - Y_min) / self.nm_per_pixel)
        py = int((Z_nm - Z_min) / self.nm_per_pixel)
        return (px, py)

    def pixel_to_stage(self, px, py):
        """Convert pixel coordinates to stage coordinates."""
        Y_min, Y_max, Z_min, Z_max = self.get_fov_bounds()
        Y_nm = Y_min + px * self.nm_per_pixel
        Z_nm = Z_min + py * self.nm_per_pixel
        return (Y_nm, Z_nm)

    def take_image(self, layout_elements, debug=True):
        """
        Render layout elements visible in the FOV.
        OPTIMIZED: Pre-filters elements before rendering.
        
        Args:
            layout_elements: List of element dicts to render
            debug: If True, print rendering statistics
        """
        # Initialize image with noise
        image = np.zeros((self.pixel_height, self.pixel_width), dtype=np.uint16)
        image += np.random.randint(100, 200, image.shape, dtype=np.uint16)
        
        # Get FOV bounds once
        Y_min, Y_max, Z_min, Z_max = self.get_fov_bounds()
        
        # Reset stats
        self.last_render_stats = {
            'total_elements': len(layout_elements),
            'rendered': 0,
            'skipped_outside_fov': 0,
            'by_type': {}
        }
        
        if debug:
            print(f"\n📷 Camera rendering:")
            print(f"   FOV: Y=[{Y_min/1000:.1f}, {Y_max/1000:.1f}] µm, Z=[{Z_min/1000:.1f}, {Z_max/1000:.1f}] µm")
            print(f"   Total elements to check: {len(layout_elements)}")
        
        # Pre-filter and render elements
        for element in layout_elements:
            elem_type = element['type']
            
            # Initialize type counter
            if elem_type not in self.last_render_stats['by_type']:
                self.last_render_stats['by_type'][elem_type] = {'total': 0, 'rendered': 0}
            
            self.last_render_stats['by_type'][elem_type]['total'] += 1
            
            # Quick bounds check before rendering
            if self._is_element_in_fov(element, Y_min, Y_max, Z_min, Z_max):
                if elem_type == 'fiducial':
                    self._render_fiducial(image, element, Y_min, Y_max, Z_min, Z_max)
                elif elem_type == 'waveguide':
                    self._render_waveguide(image, element, Y_min, Y_max, Z_min, Z_max)
                elif elem_type == 'grating':
                    self._render_grating(image, element, Y_min, Y_max, Z_min, Z_max)
                elif elem_type == 'block_outline':
                    self._render_block_outline(image, element, Y_min, Y_max, Z_min, Z_max)
                
                self.last_render_stats['rendered'] += 1
                self.last_render_stats['by_type'][elem_type]['rendered'] += 1
            else:
                self.last_render_stats['skipped_outside_fov'] += 1
        
        if debug:
            self._print_render_stats()
        
        return image
    
    def _is_element_in_fov(self, element, Y_min, Y_max, Z_min, Z_max):
        """
        Quick check if element is potentially visible in FOV.
        Returns True if element might be visible, False if definitely outside.
        """
        elem_type = element['type']
        coords_stage = element.get('coords_stage')
        
        if coords_stage is None:
            return False
        
        # Handle different coordinate formats
        if elem_type in ['fiducial', 'grating']:
            # Single point (Y, Z) - coords_stage is tuple in µm
            Y_nm, Z_nm = coords_stage[0] * 1000, coords_stage[1] * 1000
            
            # Add tolerance for marker size
            tolerance = 20000  # 20 µm in nm
            return (Y_min - tolerance <= Y_nm <= Y_max + tolerance and
                   Z_min - tolerance <= Z_nm <= Z_max + tolerance)
        
        elif elem_type in ['waveguide', 'block_outline']:
            # Polygon - coords_stage is list of (Y, Z) tuples in µm
            # Convert to nm and check if any point or bounding box intersects FOV
            Y_coords = [c[0] * 1000 for c in coords_stage]
            Z_coords = [c[1] * 1000 for c in coords_stage]
            
            # Bounding box check
            elem_Y_min, elem_Y_max = min(Y_coords), max(Y_coords)
            elem_Z_min, elem_Z_max = min(Z_coords), max(Z_coords)
            
            # Check for any overlap
            return not (elem_Y_max < Y_min or elem_Y_min > Y_max or
                       elem_Z_max < Z_min or elem_Z_min > Z_max)
        
        return True  # Default: try to render
    
    def _print_render_stats(self):
        """Print rendering statistics."""
        stats = self.last_render_stats
        print(f"\n   Rendering stats:")
        print(f"   ✅ Rendered: {stats['rendered']}/{stats['total_elements']}")
        print(f"   ⏭️  Skipped (outside FOV): {stats['skipped_outside_fov']}")
        print(f"   By type:")
        for elem_type, counts in stats['by_type'].items():
            print(f"      {elem_type}: {counts['rendered']}/{counts['total']} rendered")

    # -------------------------------------------------------------------------
    # Individual renderers (unchanged but with minor optimizations)
    # -------------------------------------------------------------------------

    def _render_waveguide(self, img, element, Y_min, Y_max, Z_min, Z_max):
        """Render rectangular waveguide as bright filled polygon."""
        # coords_stage should be list of (Y, Z) tuples in µm
        coords_stage = element['coords_stage']
        
        # Convert to nm
        Y_coords_nm = [c[0] * 1000.0 for c in coords_stage]
        Z_coords_nm = [c[1] * 1000.0 for c in coords_stage]
        
        # Skip if fully outside (redundant but fast)
        if (max(Y_coords_nm) < Y_min or min(Y_coords_nm) > Y_max or
            max(Z_coords_nm) < Z_min or min(Z_coords_nm) > Z_max):
            return
        
        # Convert to pixel coords
        pixels = []
        for Y_nm, Z_nm in zip(Y_coords_nm, Z_coords_nm):
            px = int(round((Y_nm - Y_min) / self.nm_per_pixel))
            py = int(round((Z_nm - Z_min) / self.nm_per_pixel))
            # Clip to image bounds
            px = max(0, min(img.shape[1]-1, px))
            py = max(0, min(img.shape[0]-1, py))
            pixels.append((px, py))
        
        if len(pixels) < 3:
            return  # Need at least 3 points for polygon
        
        # Get bounding box
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        min_x, max_x = max(0, min(xs)), min(img.shape[1]-1, max(xs))
        min_y, max_y = max(0, min(ys)), min(img.shape[0]-1, max(ys))
        
        if min_x >= max_x or min_y >= max_y:
            return
        
        # Use matplotlib path for polygon fill
        y_grid, x_grid = np.mgrid[min_y:max_y+1, min_x:max_x+1]
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        path = MplPath(pixels)
        mask_small = path.contains_points(points).reshape(y_grid.shape)
        
        img[min_y:max_y+1, min_x:max_x+1][mask_small] = 3000

    def _render_fiducial(self, img, element, Y_min, Y_max, Z_min, Z_max):
        """Render L-shaped fiducial marker."""
        # coords_stage is (Y, Z) tuple in µm
        Y_stage_um, Z_stage_um = element['coords_stage']
        Y_stage = Y_stage_um * 1000.0  # Convert to nm
        Z_stage = Z_stage_um * 1000.0
        
        corner = element.get('corner', 'top_left')
        
        # Quick bounds check with tolerance
        tol = 15000  # nm
        if not (Y_min - tol <= Y_stage <= Y_max + tol and
                Z_min - tol <= Z_stage <= Z_max + tol):
            return
        
        # Convert to pixel coordinates
        px = int(round((Y_stage - Y_min) / self.nm_per_pixel))
        py = int(round((Z_stage - Z_min) / self.nm_per_pixel))
        
        # Draw L-shaped fiducial
        self._draw_fiducial(img, (px, py), size=40, thickness=8, brightness=3000, corner=corner)

    def _render_grating(self, img, element, Y_min, Y_max, Z_min, Z_max):
        """Render grating as a small bright circular spot."""
        # coords_stage is (Y, Z) tuple in µm
        Y_stage_um, Z_stage_um = element['coords_stage']
        Y_stage = Y_stage_um * 1000.0  # Convert to nm
        Z_stage = Z_stage_um * 1000.0
        
        tol_nm = 8 * self.nm_per_pixel
        if not (Y_min - tol_nm <= Y_stage <= Y_max + tol_nm and
                Z_min - tol_nm <= Z_stage <= Z_max + tol_nm):
            return
        
        px = int((Y_stage - Y_min) / self.nm_per_pixel)
        py = int((Z_stage - Z_min) / self.nm_per_pixel)
        
        radius = 8
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = xx**2 + yy**2 <= radius**2
        
        y0 = max(0, py - radius)
        y1 = min(img.shape[0], py + radius + 1)
        x0 = max(0, px - radius)
        x1 = min(img.shape[1], px + radius + 1)
        
        mask_y_start = max(0, radius - py)
        mask_x_start = max(0, radius - px)
        mask_y_end = mask_y_start + (y1 - y0)
        mask_x_end = mask_x_start + (x1 - x0)
        
        img[y0:y1, x0:x1][mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]] = 2000

    def _render_block_outline(self, img, element, Y_min, Y_max, Z_min, Z_max):
        """Render block outline as thin lines."""
        # coords_stage is list of (Y, Z) tuples in µm
        coords_stage_nm = [(c[0] * 1000.0, c[1] * 1000.0) for c in element['coords_stage']]
        
        for i in range(len(coords_stage_nm)):
            p1 = coords_stage_nm[i]
            p2 = coords_stage_nm[(i + 1) % len(coords_stage_nm)]
            
            px1 = int((p1[0] - Y_min) / self.nm_per_pixel)
            py1 = int((p1[1] - Z_min) / self.nm_per_pixel)
            px2 = int((p2[0] - Y_min) / self.nm_per_pixel)
            py2 = int((p2[1] - Z_min) / self.nm_per_pixel)
            
            self._draw_line(img, px1, py1, px2, py2, brightness=1500)

    def _draw_line(self, img, x0, y0, x1, y1, brightness=3000):
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        
        for _ in range(max(dx, dy) + 1):
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                img[y, x] = brightness
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _draw_fiducial(self, img, center, size=50, thickness=10, brightness=3000, corner=None):
        """
        Draw an L-shaped fiducial whose inner corner sits at `center` (pixel coords).
        """
        x, y = center
        arm = size
        th = max(1, int(thickness))

        def paint_rect(y0, y1, x0, x1):
            """Helper to clip and paint a rectangle."""
            y0c = max(0, y0)
            y1c = min(img.shape[0], y1)
            x0c = max(0, x0)
            x1c = min(img.shape[1], x1)
            if y1c > y0c and x1c > x0c:
                img[y0c:y1c, x0c:x1c] = brightness

        orient = (corner or 'top_left').lower()

        if orient == 'top_right':
            paint_rect(y - th//2, y + (th+1)//2, x - arm, x)
            paint_rect(y - arm, y, x - th//2, x + (th+1)//2)
        elif orient == 'top_left':
            paint_rect(y - th//2, y + (th+1)//2, x, x + arm)
            paint_rect(y - arm, y, x - th//2, x + (th+1)//2)
        elif orient == 'bottom_right':
            paint_rect(y - th//2, y + (th+1)//2, x - arm, x)
            paint_rect(y, y + arm, x - th//2, x + (th+1)//2)
        elif orient == 'bottom_left':
            paint_rect(y - th//2, y + (th+1)//2, x, x + arm)
            paint_rect(y, y + arm, x - th//2, x + (th+1)//2)
        else:
            # Fallback
            paint_rect(y - th//2, y + (th+1)//2, x, x + arm)
            paint_rect(y - arm, y, x - th//2, x + (th+1)//2)


# -----------------------------------------------------------------------------
# MAIN TEST: read ASCII file and simulate camera image
# -----------------------------------------------------------------------------

def main():
    ascii_file = Path("./AlignmentSystem/ascii_sample.ASC")  # <-- change to your actual file path

    if not ascii_file.exists():
        print(f"❌ ASCII file not found: {ascii_file}")
        return

    # Parse ASCII layout
    parser = ASCIIParser(ascii_file)
    parsed_data = parser.parse()

    print(f"✅ Parsed {ascii_file.name}")
    print(f"   Markers: {len(parsed_data['markers'])}")
    print(f"   Waveguides: {len(parsed_data['waveguides'])}")
    print(f"   Gratings: {len(parsed_data['gratings'])}")

    # Build layout elements for the camera
    layout_elements = []

    # Add markers
    for m in parsed_data['markers']:
        layout_elements.append({
            'type': 'fiducial',
            'coords_stage': m['position'],  # (u, v) in µm
            'corner': m.get('corner', None)
        })

    # Add waveguides
    for wg in parsed_data['waveguides']: 
        coords_stage = [
            (wg['u_start'], wg['v_bottom']),
            (wg['u_end'], wg['v_bottom']),
            (wg['u_end'], wg['v_top']),
            (wg['u_start'], wg['v_top'])
        ]
        layout_elements.append({
            'type': 'waveguide',
            'coords_stage': coords_stage
        })

    # Add gratings
    for g in parsed_data['gratings']:
        layout_elements.append({
            'type': 'grating',
            'coords_stage': g['position'],
            'side': g['side']
        })

    # Initialize camera
    cam = Camera(pixel_width=2048, pixel_height=2048, nm_per_pixel=300)

    # Optionally center on first waveguide if present
    if parsed_data['waveguides']:
        first_wg = parsed_data['waveguides'][0]
        cam_center_y = (first_wg['u_start'] + first_wg['u_end']) / 2 * 1000  # µm→nm
        cam_center_z = first_wg['v_center'] * 1000
        cam.move_to(cam_center_y, cam_center_z)
        print(f"📷 Centering camera on WG#1 at ({cam_center_y/1000:.2f}, {cam_center_z/1000:.2f}) µm")
    else:
        cam.move_to(0, 0)
        print("📷 No waveguides found, camera centered at (0,0)")

    # Simulate image
    img = cam.take_image(layout_elements)

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", vmin=0, vmax=4000)
    plt.title(f"Simulated Camera View: {ascii_file.name}")
    plt.xlabel("Y pixels")
    plt.ylabel("Z pixels")
    plt.show()


if __name__ == "__main__":
    main()
