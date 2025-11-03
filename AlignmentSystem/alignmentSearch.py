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
    
    def search_for_fiducial(self, center_y_nm: float, center_z_nm: float, 
                           search_radius_nm: float = 50000, 
                           step_nm: float = 10000,
                           label: str = "Fiducial", 
                           plot_progress: bool = True) -> Optional[Dict[str, Any]]:
        """
        Grid search for fiducial with visualization.
        
        Performs coarse grid search, then refines position based on pixel-level detection.
        After finding best position, moves stage there and captures verification image.
        
        Args:
            center_y_nm: Search center Y position (nm)
            center_z_nm: Search center Z position (nm)
            search_radius_nm: Search radius (nm)
            step_nm: Grid step size (nm)
            label: Label for printing/plotting
            plot_progress: Whether to show grid visualization
        
        Returns:
            dict with keys:
                - stage_Y, stage_Z: Final refined stage position (nm)
                - pixel_pos: Detected pixel position in verification image
                - pixel_offset: Pixel offset from center in verification image
                - stage_offset_nm: Stage offset from center (nm)
                - confidence: Detection confidence
                - method: Detection method used
                - image: Verification image (at refined position)
            or None if not found
        """
        print(f"   Searching in {search_radius_nm / 1000:.0f} µm radius with {step_nm / 1000:.0f} µm steps...")

        y_positions = np.arange(center_y_nm - search_radius_nm,
                                center_y_nm + search_radius_nm + 0.5 * step_nm,
                                step_nm)
        z_positions = np.arange(center_z_nm - search_radius_nm,
                                center_z_nm + search_radius_nm + 0.5 * step_nm,
                                step_nm)

        best_result = None
        best_confidence = -1.0
        total_positions = len(y_positions) * len(z_positions)
        
        # Store all images and detection results for visualization
        search_data = []
        
        checked = 0

        for Y in y_positions:
            for Z in z_positions:
                checked += 1
                if checked % 10 == 0:
                    print(f"   Progress: {checked}/{total_positions} positions checked...", end='\r')

                # Move stage
                self.stage.move_abs('y', int(round(Y)))
                self.stage.move_abs('z', int(round(Z)))

                # Capture image
                img = self.camera.acquire_single_image()

                # Look for fiducial near center
                img_center_x = img.shape[1] // 2
                img_center_y = img.shape[0] // 2
                img_center = (img_center_x, img_center_y)
                
                result = self.vt.find_fiducial_auto(img, expected_position=img_center, search_radius=150)

                # Store data for visualization
                search_data.append({
                    'Y': Y,
                    'Z': Z,
                    'image': img.copy(),
                    'detection': result,
                    'img_center': img_center
                })

                if result and result.get('confidence', 0.0) > best_confidence:
                    best_confidence = result['confidence']
                    
                    # Calculate actual stage position from pixel offset
                    found_px, found_py = result['position']
                    
                    # Pixel offset from center
                    offset_px_x = found_px - img_center_x
                    offset_px_y = found_py - img_center_y
                    
                    # Convert to nm
                    # From _stage_to_pixel: px corresponds to Y, py corresponds to Z
                    offset_nm_Y = offset_px_x * self.camera.nm_per_pixel
                    offset_nm_Z = offset_px_y * self.camera.nm_per_pixel
                    
                    # Actual stage position = grid position + pixel offset
                    actual_Y = Y + offset_nm_Y
                    actual_Z = Z + offset_nm_Z
                    
                    best_result = {
                        'stage_Y': int(round(actual_Y)),
                        'stage_Z': int(round(actual_Z)),
                        'grid_Y': Y,
                        'grid_Z': Z,
                        'pixel_pos': result['position'],
                        'pixel_offset': (offset_px_x, offset_px_y),
                        'stage_offset_nm': (offset_nm_Y, offset_nm_Z),
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
            print(f"      Grid position: Y={best_result['grid_Y']:.0f}, Z={best_result['grid_Z']:.0f} nm")
            print(f"      Pixel offset: {best_result['pixel_offset']}")
            print(f"      Stage offset (nm): {best_result['stage_offset_nm']}")
            print(f"      Final stage position: Y={best_result['stage_Y']}, Z={best_result['stage_Z']} nm")
            
            # Move stage to the corrected position where fiducial should be centered
            print(f"      📍 Moving stage to corrected position...")
            self.stage.move_abs('y', best_result['stage_Y'])
            self.stage.move_abs('z', best_result['stage_Z'])
            
            # Take verification image at corrected position
            verification_img = self.camera.acquire_single_image()
            
            # Verify centering
            img_center = (verification_img.shape[1] // 2, verification_img.shape[0] // 2)
            verify_result = self.vt.find_fiducial_auto(verification_img, 
                                                        expected_position=img_center, 
                                                        search_radius=150)
            
            if verify_result:
                verify_px = verify_result['position']
                verify_offset_px = (verify_px[0] - img_center[0], verify_px[1] - img_center[1])
                verify_offset_nm = (verify_offset_px[0] * self.camera.nm_per_pixel,
                                    verify_offset_px[1] * self.camera.nm_per_pixel)
                verify_error = np.hypot(verify_offset_nm[0], verify_offset_nm[1])
                
                print(f"      ✓ Verification: offset = {verify_offset_px} px = ({verify_offset_nm[0]:.0f}, {verify_offset_nm[1]:.0f}) nm")
                print(f"      ✓ Verification error: {verify_error:.1f} nm")
                
                # Update result with verification data
                verification_output = {
                    'stage_Y': best_result['stage_Y'],
                    'stage_Z': best_result['stage_Z'],
                    'pixel_pos': verify_result['position'],
                    'pixel_offset': verify_offset_px,
                    'stage_offset_nm': verify_offset_nm,
                    'confidence': verify_result['confidence'],
                    'method': verify_result['method'],
                    'image': verification_img.copy(),
                    'verification_error_nm': verify_error
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
                                      search_radius_nm: float = 20000, 
                                      step_nm: float = 5000,
                                      plot_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Search for multiple fiducials in a block.
        Uses alignment_system.predict_fiducial_position() for initialization.
        
        Args:
            block_id: Which block to search
            alignment_system: HierarchicalAlignment instance (for predictions)
            corners: List of corner names to find
            search_radius_nm: Search radius around predicted position
            step_nm: Grid step size
            plot_progress: Whether to visualize each search
        
        Returns:
            List of fiducial measurements (dicts with stage_Y, stage_Z, corner, etc.)
        """
        measurements = []
        
        print(f"\n   🔍 Searching for {len(corners)} fiducials in Block {block_id}...")
        
        for corner in corners:
            # Get predicted position from alignment system
            try:
                pred_Y, pred_Z = alignment_system.predict_fiducial_position(block_id, corner)
                print(f"\n   Predicted position for Block {block_id} {corner}: ({pred_Y:.0f}, {pred_Z:.0f}) nm")
            except Exception as e:
                print(f"\n   ⚠️  Could not predict position for Block {block_id} {corner}: {e}")
                print(f"   Using block design position as fallback...")
                # Fallback to design position
                block_center = alignment_system.layout['blocks'][block_id]['design_position']
                pred_Y = block_center[0] * 1000  # µm to nm
                pred_Z = block_center[1] * 1000
            
            # Search
            label = f"Block {block_id} {corner}"
            result = self.search_for_fiducial(
                center_y_nm=pred_Y,
                center_z_nm=pred_Z,
                search_radius_nm=search_radius_nm,
                step_nm=step_nm,
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
    
    def verify_fiducial_centering(self, expected_y_nm: float, expected_z_nm: float, 
                                  label: str = "Verification") -> Optional[Dict[str, Any]]:
        """
        Move to position, capture image, verify fiducial is centered.
        
        Args:
            expected_y_nm: Expected Y position (nm)
            expected_z_nm: Expected Z position (nm)
            label: Label for printing
        
        Returns:
            dict with verification results or None if not found
        """
        print(f"\n   🔍 Verifying fiducial at ({expected_y_nm:.0f}, {expected_z_nm:.0f}) nm...")
        
        # Move stage
        self.stage.move_abs('y', int(round(expected_y_nm)))
        self.stage.move_abs('z', int(round(expected_z_nm)))
        
        # Capture image
        img = self.camera.acquire_single_image()
        
        # Detect
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        result = self.vt.find_fiducial_auto(img, expected_position=img_center, search_radius=150)
        
        if result:
            found_px = result['position']
            offset_px = (found_px[0] - img_center[0], found_px[1] - img_center[1])
            offset_nm = (offset_px[0] * self.camera.nm_per_pixel,
                        offset_px[1] * self.camera.nm_per_pixel)
            error = np.hypot(offset_nm[0], offset_nm[1])
            
            print(f"   ✓ {label}: Found with {error:.1f} nm error")
            print(f"     Pixel offset: {offset_px}")
            print(f"     Stage offset: ({offset_nm[0]:.0f}, {offset_nm[1]:.0f}) nm")
            
            return {
                'stage_Y': int(round(expected_y_nm)),
                'stage_Z': int(round(expected_z_nm)),
                'pixel_pos': found_px,
                'pixel_offset': offset_px,
                'stage_offset_nm': offset_nm,
                'confidence': result['confidence'],
                'method': result['method'],
                'image': img,
                'error_nm': error
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
    from HardwareControl.CameraControl.mock_camera import MockCamera
    from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
    from AlignmentSystem.cv_tools import VisionTools
    # Setup hardware
    stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    camera = MockCamera("config/mock_layout.json", stage_ref=stage)
    camera.connect()
    stage.set_camera_observer(camera)
    vt = VisionTools()

    # Create searcher
    searcher = AlignmentSearcher(stage, camera, vt)

    # Test search
    result = searcher.search_for_fiducial(
        center_y_nm=-100000,
        center_z_nm=100000,
        search_radius_nm=60000,
        step_nm=20000,
        label="Test Fiducial",
        plot_progress=True
    )

    if result:
        print(f"Found at: ({result['stage_Y']}, {result['stage_Z']})")
        print(f"Error: {result['verification_error_nm']:.1f} nm")