#!/usr/bin/env python3
# alignment_search.py
"""
AlignmentSearcher: Handles fiducial search, detection, and visualization.

This class is responsible for:
- Grid search for fiducials with hardware interaction
- Pixel-level detection refinement
- Verification of fiducial centering
- Visualization of search progress and results

Separated from alignment logic to maintain clean separation of concerns.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple, Dict, List, Any


class AlignmentSearcher:
    """
    Handles fiducial search, detection, and visualization.
    Separated from HierarchicalAlignment to keep concerns separate.
    """
    
    def __init__(self, stage, camera, vision_tools):
        """
        Initialize searcher with hardware references.
        
        Args:
            stage: Stage controller (e.g., MockXYZStage)
            camera: Camera controller (e.g., MockCamera)
            vision_tools: VisionTools instance with CV detection
        """
        self.stage = stage
        self.camera = camera
        self.vt = vision_tools
    
    def search_for_fiducial(self, center_y_um: float, center_z_um: float, 
                            search_radius_um: float = 50.0,
                            step_um: float = 10.0,
                            label: str = "Fiducial",
                            plot_progress: bool = True) -> Optional[Dict[str, Any]]:
        """
        Grid search for fiducial with visualization (all units in µm).

        Args:
            center_y_um: Search center Y position (µm)
            center_z_um: Search center Z position (µm)
            search_radius_um: Search radius (µm)
            step_um: Grid step size (µm)
            label: Label for printing/plotting
            plot_progress: Whether to show grid visualization

        Returns:
            dict with:
                - stage_Y, stage_Z: Final refined stage position (µm)
                - pixel_pos: Detected pixel position in verification image
                - pixel_offset: Pixel offset from center in verification image
                - stage_offset_um: Stage offset from center (µm)
                - confidence: Detection confidence
                - method: Detection method used
                - image: Verification image (at refined position)
            or None if not found.
        """
        print(f"   Searching in {search_radius_um:.0f} µm radius with {step_um:.0f} µm steps...")

        y_positions = np.arange(center_y_um - search_radius_um,
                                center_y_um + search_radius_um + 0.5 * step_um,
                                step_um)
        z_positions = np.arange(center_z_um - search_radius_um,
                                center_z_um + search_radius_um + 0.5 * step_um,
                                step_um)

        best_result = None
        best_confidence = -1.0
        total_positions = len(y_positions) * len(z_positions)

        search_data = []
        checked = 0

        for Y in y_positions:
            for Z in z_positions:
                checked += 1
                if checked % 10 == 0:
                    print(f"   Progress: {checked}/{total_positions} positions checked...", end='\r')

                # Move stage (µm)
                self.stage.move_abs('y', Y)
                self.stage.move_abs('z', Z)

                # Capture image
                img = self.camera.acquire_single_image()

                img_center_x = img.shape[1] // 2
                img_center_y = img.shape[0] // 2
                img_center = (img_center_x, img_center_y)

                result = self.vt.find_fiducial_auto(img, expected_position=img_center, search_radius=150)

                search_data.append({
                    'Y': Y,
                    'Z': Z,
                    'image': img.copy(),
                    'detection': result,
                    'img_center': img_center
                })

                if result and result.get('confidence', 0.0) > best_confidence:
                    best_confidence = result['confidence']
                    found_px, found_py = result['position']

                    offset_px_x = found_px - img_center_x
                    offset_px_y = found_py - img_center_y

                    # Convert pixel offset to µm
                    offset_um_Y = offset_px_x * self.camera.um_per_pixel
                    offset_um_Z = offset_px_y * self.camera.um_per_pixel

                    actual_Y = Y + offset_um_Y
                    actual_Z = Z + offset_um_Z

                    best_result = {
                        'stage_Y': actual_Y,
                        'stage_Z': actual_Z,
                        'grid_Y': Y,
                        'grid_Z': Z,
                        'pixel_pos': result['position'],
                        'pixel_offset': (offset_px_x, offset_px_y),
                        'stage_offset_um': (offset_um_Y, offset_um_Z),
                        'confidence': result['confidence'],
                        'method': result['method'],
                        'search_image': img.copy()
                    }

        print()  # newline after progress

        # ========================================
        # Move to corrected position and verify
        # ========================================
        if best_result:
            print(f"   ✅ {label} found!")
            print(f"      Confidence: {best_result['confidence']:.3f}")
            print(f"      Grid position: Y={best_result['grid_Y']:.1f}, Z={best_result['grid_Z']:.1f} µm")
            print(f"      Pixel offset: {best_result['pixel_offset']}")
            print(f"      Stage offset: {best_result['stage_offset_um']} µm")
            print(f"      Final stage position: Y={best_result['stage_Y']:.1f}, Z={best_result['stage_Z']:.1f} µm")
            print(f"      📍 Moving stage to corrected position...")

            self.stage.move_abs('y', best_result['stage_Y'])
            self.stage.move_abs('z', best_result['stage_Z'])

            verification_img = self.camera.acquire_single_image()
            img_center = (verification_img.shape[1] // 2, verification_img.shape[0] // 2)
            verify_result = self.vt.find_fiducial_auto(verification_img,
                                                    expected_position=img_center,
                                                    search_radius=150)
            if verify_result:
                verify_px = verify_result['position']
                verify_offset_px = (verify_px[0] - img_center[0], verify_px[1] - img_center[1])
                verify_offset_um = (verify_offset_px[0] * self.camera.um_per_pixel,
                                    verify_offset_px[1] * self.camera.um_per_pixel)
                verify_error_um = np.hypot(verify_offset_um[0], verify_offset_um[1])

                print(f"      ✓ Verification: offset = {verify_offset_px} px = ({verify_offset_um[0]:.3f}, {verify_offset_um[1]:.3f}) µm")
                print(f"      ✓ Verification error: {verify_error_um:.3f} µm")

                verification_output = {
                    'stage_Y': best_result['stage_Y'],
                    'stage_Z': best_result['stage_Z'],
                    'pixel_pos': verify_result['position'],
                    'pixel_offset': verify_offset_px,
                    'stage_offset_um': verify_offset_um,
                    'confidence': verify_result['confidence'],
                    'method': verify_result['method'],
                    'image': verification_img.copy(),
                    'verification_error_um': verify_error_um
                }
            else:
                print(f"      ⚠️  Warning: Could not detect fiducial in verification image!")
                verification_output = None
        else:
            print(f"   ❌ {label} not found in search region")
            verification_output = None

        # ========================================
        # Visualization
        # ========================================
        if plot_progress:
            self._plot_search_progress(search_data=search_data, label=label,
                                    best_result=best_result, best_confidence=best_confidence)

        return verification_output

    def search_for_fiducials_in_block(self, block_id: int, 
                                    alignment_system,
                                    corners: List[str] = ['top_left', 'bottom_right'],
                                    search_radius_um: float = 20.0,
                                    step_um: float = 5.0,
                                    plot_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Search for multiple fiducials in a block (all units in µm).
        Uses alignment_system.predict_fiducial_position() for initialization.

        Args:
            block_id: Which block to search
            alignment_system: HierarchicalAlignment instance (for predictions)
            corners: List of corner names to find
            search_radius_um: Search radius around predicted position (µm)
            step_um: Grid step size (µm)
            plot_progress: Whether to visualize each search

        Returns:
            List of fiducial measurements (dicts with stage_Y, stage_Z, corner, etc.)
        """
        measurements = []

        print(f"\n   🔍 Searching for {len(corners)} fiducials in Block {block_id}...")

        for corner in corners:
            # Get predicted position from alignment system (convert nm → µm)
            try:
                pred_Y_nm, pred_Z_nm = alignment_system.predict_fiducial_position(block_id, corner)
                pred_Y_um = pred_Y_nm / 1000.0
                pred_Z_um = pred_Z_nm / 1000.0
                print(f"\n   Predicted position for Block {block_id} {corner}: "
                    f"({pred_Y_um:.2f}, {pred_Z_um:.2f}) µm")
            except Exception as e:
                print(f"\n   ⚠️  Could not predict position for Block {block_id} {corner}: {e}")
                print(f"   Using block design position as fallback...")
                block_center = alignment_system.layout['blocks'][block_id]['design_position']
                pred_Y_um = block_center[0]
                pred_Z_um = block_center[1]

            # Run fiducial search (now in µm)
            label = f"Block {block_id} {corner}"
            result = self.search_for_fiducial(
                center_y_um=pred_Y_um,
                center_z_um=pred_Z_um,
                search_radius_um=search_radius_um,
                step_um=step_um,
                label=label,
                plot_progress=plot_progress
            )

            if result:
                result['block_id'] = block_id
                result['corner'] = corner
                measurements.append(result)
            else:
                print(f"   ❌ Failed to find Block {block_id} {corner}")

        print(f"\n   ✅ Found {len(measurements)}/{len(corners)} fiducials in Block {block_id}")

        return measurements

    
    def verify_fiducial_centering(self, expected_y_um: float, expected_z_um: float, 
                                label: str = "Verification") -> Optional[Dict[str, Any]]:
        """
        Move to a position (in µm), capture image, and verify fiducial is centered.

        Args:
            expected_y_um: Expected Y position (µm)
            expected_z_um: Expected Z position (µm)
            label: Label for printing

        Returns:
            dict with verification results or None if not found
        """
        print(f"\n   🔍 Verifying fiducial at ({expected_y_um:.2f}, {expected_z_um:.2f}) µm...")

        # Move stage
        self.stage.move_abs('y', expected_y_um)
        self.stage.move_abs('z', expected_z_um)

        # Capture image
        img = self.camera.acquire_single_image()

        # Detect fiducial near image center
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        result = self.vt.find_fiducial_auto(img, expected_position=img_center, search_radius=150)

        if result:
            found_px = result['position']
            offset_px = (found_px[0] - img_center[0], found_px[1] - img_center[1])

            # Convert from pixels to µm
            offset_um = (offset_px[0] * self.camera.um_per_pixel,
                        offset_px[1] * self.camera.um_per_pixel)

            error_um = np.hypot(offset_um[0], offset_um[1])

            print(f"   ✓ {label}: Found with {error_um:.3f} µm error")
            print(f"     Pixel offset: {offset_px}")
            print(f"     Stage offset: ({offset_um[0]:.3f}, {offset_um[1]:.3f}) µm")

            return {
                'stage_Y': expected_y_um,
                'stage_Z': expected_z_um,
                'pixel_pos': found_px,
                'pixel_offset': offset_px,
                'stage_offset_um': offset_um,
                'confidence': result['confidence'],
                'method': result['method'],
                'image': img,
                'error_um': error_um
            }
        else:
            print(f"   ❌ {label}: Fiducial not found!")
            return None

    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def _plot_search_progress(self, search_data: List[Dict], label: str,
                             best_result: Optional[Dict], best_confidence: float):
        """
        Create grid visualization of all search images.
        
        Args:
            search_data: List of dicts with Y, Z, image, detection, img_center
            label: Title label
            best_result: Best detection result
            best_confidence: Best confidence score
        """
        print(f"\n   📊 Creating search visualization with {len(search_data)} images...")
        
        # Calculate grid dimensions
        n_images = len(search_data)
        n_cols = min(8, int(np.ceil(np.sqrt(n_images))))
        n_rows = int(np.ceil(n_images / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        fig.suptitle(f'Grid Search: {label} ({n_images} positions)', fontsize=14, fontweight='bold')
        
        for idx, data in enumerate(search_data):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            img = data['image']
            detection = data['detection']
            img_center = data['img_center']
            Y_stage = data['Y']
            Z_stage = data['Z']
            
            # Normalize image to 8-bit for better visualization
            img_norm = np.clip(img.astype(np.float32) / 4095.0 * 255, 0, 255).astype(np.uint8)
            
            # Zoom to center region
            zoom_size = 200
            cy, cx = img_center[1], img_center[0]
            y1, y2 = max(0, cy - zoom_size), min(img.shape[0], cy + zoom_size)
            x1, x2 = max(0, cx - zoom_size), min(img.shape[1], cx + zoom_size)
            zoomed = img_norm[y1:y2, x1:x2]
            
            # Convert to RGB for colored markers
            zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2RGB)
            
            # Calculate relative center in zoomed image
            rel_cx = cx - x1
            rel_cy = cy - y1
            
            if detection:
                # Mark detected position
                found_px, found_py = detection['position']
                rel_fx = found_px - x1
                rel_fy = found_py - y1
                
                # Draw markers
                cv2.drawMarker(zoomed_rgb, (int(rel_cx), int(rel_cy)), 
                            (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(zoomed_rgb, (int(rel_fx), int(rel_fy)), 
                            (255, 0, 0), cv2.MARKER_TILTED_CROSS, 25, 2)
                
                confidence = detection.get('confidence', 0.0)
                method = detection.get('method', 'unknown')[:8]
                
                # Color based on confidence
                if confidence > 0.7:
                    border_color = 'green'
                    title_color = 'darkgreen'
                elif confidence > 0.4:
                    border_color = 'orange'
                    title_color = 'darkorange'
                else:
                    border_color = 'yellow'
                    title_color = 'goldenrod'
                
                ax.imshow(zoomed_rgb, origin='lower')
                ax.set_title(f'✓ {method}\n{confidence:.2f}', 
                            fontsize=8, color=title_color, fontweight='bold')
                
                # Highlight border if this is the best result
                if best_result and detection['confidence'] == best_confidence:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('lime')
                        spine.set_linewidth(4)
            else:
                # No detection - mark center only
                cv2.drawMarker(zoomed_rgb, (int(rel_cx), int(rel_cy)), 
                            (128, 128, 128), cv2.MARKER_CROSS, 15, 1)
                ax.imshow(zoomed_rgb, origin='lower')
                ax.set_title('✗ not found', fontsize=8, color='gray')
            
            # Add axis labels with stage positions
            ax.set_xlabel(f'Y={Y_stage/1000:.1f}µm', fontsize=7)
            ax.set_ylabel(f'Z={Z_stage/1000:.1f}µm', fontsize=7)
            ax.tick_params(labelsize=6)
        
        # Hide unused subplots
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save with descriptive filename
        safe_label = label.replace(' ', '_').replace('/', '_')
        filename = f'search_grid_{safe_label}.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"   💾 Saved search visualization: {filename}")
        plt.show()
    
    def plot_calibration_residuals(self, design_points, measured_points, 
                                   predicted_points, label: str = "Calibration"):
        """
        Plot measured vs predicted positions with residual vectors.
        
        Args:
            design_points: List of (u, v) in design coords (µm)
            measured_points: List of (Y, Z) measured stage coords (nm)
            predicted_points: List of (Y, Z) predicted stage coords (nm)
            label: Title label
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Convert to arrays
        design_arr = np.array(design_points)
        measured_arr = np.array(measured_points) / 1000.0  # nm to µm
        predicted_arr = np.array(predicted_points) / 1000.0  # nm to µm
        
        # Left plot: Design coordinates
        ax1.plot(design_arr[:, 0], design_arr[:, 1], 'bo-', markersize=10, 
                label='Design', linewidth=2)
        ax1.set_xlabel('U (µm)', fontsize=12)
        ax1.set_ylabel('V (µm)', fontsize=12)
        ax1.set_title('Design Coordinates', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend()
        
        # Right plot: Stage coordinates (measured vs predicted)
        ax2.plot(measured_arr[:, 0], measured_arr[:, 1], 'rs', markersize=10, 
                label='Measured', linewidth=2)
        ax2.plot(predicted_arr[:, 0], predicted_arr[:, 1], 'g^', markersize=10,
                label='Predicted', linewidth=2)
        
        # Draw residual vectors
        for i in range(len(measured_arr)):
            ax2.arrow(predicted_arr[i, 0], predicted_arr[i, 1],
                     measured_arr[i, 0] - predicted_arr[i, 0],
                     measured_arr[i, 1] - predicted_arr[i, 1],
                     head_width=2, head_length=1, fc='orange', ec='orange',
                     alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Y (µm)', fontsize=12)
        ax2.set_ylabel('Z (µm)', fontsize=12)
        ax2.set_title(f'Stage Coordinates - {label}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.legend()
        
        # Calculate residuals
        residuals = measured_arr - predicted_arr
        residual_magnitudes = np.linalg.norm(residuals, axis=1)
        mean_residual = np.mean(residual_magnitudes)
        max_residual = np.max(residual_magnitudes)
        
        # Add text with statistics
        stats_text = f'Mean residual: {mean_residual:.3f} µm\nMax residual: {max_residual:.3f} µm'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        safe_label = label.replace(' ', '_').replace('/', '_')
        filename = f'calibration_residuals_{safe_label}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved residual plot: {filename}")
        plt.show()


if __name__ == "__main__":
    from HardwareControl.CameraControl.mock_camera_v3 import MockCamera
    from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
    from HardwareControl.SetupMotor.stageAdapter import StageAdapterUM
    from AlignmentSystem.cv_tools import VisionTools
    # Setup hardware
    stage_nm = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    stage = StageAdapterUM(stage_nm)
    camera = MockCamera("config/mock_layout.json", stage_ref=stage)
    camera.connect()
    stage.set_camera_observer(camera)
    vt = VisionTools()

    # Create searcher
    searcher = AlignmentSearcher(stage, camera, vt)

    # Test search
    result = searcher.search_for_fiducial(
        center_y_um=-100,
        center_z_um=100,
        search_radius_um=60,
        step_um=20,
        label="Test Fiducial",
        plot_progress=True
    )

    if result:
        print(f"Found at: ({result['stage_Y']}, {result['stage_Z']})")
        print(f"Error: {result['verification_error_um']:.1f} um")