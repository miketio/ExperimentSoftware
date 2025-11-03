#!/usr/bin/env python3
# test_alignment_cv_calibration.py (REFINED: keep workflow + visuals, fix bugs, ADD SEARCH VISUALIZATION)
"""
Critical alignment tests A-B with bug fixes and preserved visualization + interactivity.

What I fixed compared to the earlier broken run:
 - Do NOT re-create AlignmentTester mid-test (was losing captured images).
 - Compute fiducial global design coordinates consistently:
     global = block_center + (local - block_size/2)  # local is relative to bottom-left
 - Ensure units consistency: design points converted from µm -> nm before calibration.
 - Keep interactive prompts, image saving, and whole plotting pipeline intact for debugging.
 - ADDED: Visualization of all grid search images to debug fiducial detection
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import time

# Mock hardware
from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
from HardwareControl.CameraControl.mock_camera import MockCamera

# Alignment components
from config.layout_config_generator_v2 import load_layout_config_v2
from AlignmentSystem.coordinate_utils import CoordinateConverter
from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.coordinate_transform import CoordinateTransform


class AlignmentTester:
    """Complete alignment system tester (minimal wrapper)."""

    def __init__(self, layout_config: str = "config/mock_layout.json"):
        self.layout_config = layout_config
        self.layout = None

        # Mock hardware
        self.stage = None
        self.camera = None

        # Vision tools
        self.vt = None

        # Calibration & storage
        self.measured_fiducials = {}
        self.calibrated_converter = None
        self.gt_converter = None
        self.captured_images = {}

    def setup(self):
        """Initialize hardware and load layout."""
        print("\n" + "=" * 70)
        print("SETUP: Initialize Mock Hardware")
        print("=" * 70)

        # Load layout
        self.layout = load_layout_config_v2(self.layout_config)
        print(f"✅ Layout loaded: {self.layout['design_name']}")

        # Ground truth converter for simulation
        self.gt_converter = CoordinateConverter(self.layout)
        gt = self.layout['simulation_ground_truth']
        self.gt_converter.set_transformation(gt['rotation_deg'], tuple(gt['translation_nm']))
        print(f"✅ Ground truth set: {gt['rotation_deg']}° rotation, {gt['translation_nm']} nm translation")

        # Create mock hardware
        self.stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
        self.camera = MockCamera(self.layout_config, stage_ref=self.stage)
        self.camera.connect()
        self.stage.set_camera_observer(self.camera)
        self.camera.set_exposure_time(0.02)

        # Vision tools (integrates GMarkerDetector if available)
        self.vt = VisionTools()

        print(f"✅ Mock hardware initialized")
        print(f"   Camera FOV: {self.camera.sensor_width * self.camera.nm_per_pixel / 1000:.1f} µm")
        print(f"   Resolution: {self.camera.nm_per_pixel} nm/pixel")

    def search_for_fiducial(self, center_y_nm, center_z_nm, search_radius_nm=50000,
                        step_nm=10000, label="Fiducial", plot_progress=True):
        """
        Search for fiducial marker in a region using grid search.
        NOW WITH VISUALIZATION of all captured images!

        Returns:
            dict with 'stage_Y', 'stage_Z', 'pixel_pos', 'confidence', 'image', 'method' or None if not found
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
                    offset_nm_Y = offset_px_x * self.camera.nm_per_pixel
                    offset_nm_Z = offset_px_y * self.camera.nm_per_pixel
                    
                    # Actual stage position = grid position + pixel offset
                    actual_Y = Y + offset_nm_Y
                    actual_Z = Z + offset_nm_Z
                    
                    best_result = {
                        'stage_Y': int(round(actual_Y)),
                        'stage_Z': int(round(actual_Z)),
                        'pixel_pos': result['position'],
                        'pixel_offset': (offset_px_x, offset_px_y),
                        'stage_offset_nm': (offset_nm_Y, offset_nm_Z),
                        'confidence': result['confidence'],
                        'method': result['method'],
                        'image': img.copy()
                    }
        
        # ========================================
        # Print summary and move to corrected position
        # ========================================
        if best_result:
            print(f"   ✅ {label} found!")
            print(f"      Confidence: {best_result['confidence']:.3f}")
            print(f"      Grid position: Y={Y:.0f}, Z={Z:.0f} nm")
            print(f"      Pixel position: {best_result['pixel_pos']}")
            print(f"      Image center: ({img.shape[1]//2}, {img.shape[0]//2})")
            print(f"      Pixel offset: {best_result['pixel_offset']}")
            print(f"      Stage offset (nm): {best_result['stage_offset_nm']}")
            print(f"      Final stage position: Y={best_result['stage_Y']}, Z={best_result['stage_Z']} nm")
            
            # Move stage to the corrected position where fiducial is centered
            print(f"      📍 Moving stage to corrected position...")
            self.stage.move_abs('y', best_result['stage_Y'])
            self.stage.move_abs('z', best_result['stage_Z'])
            
            # Take verification image at corrected position
            verification_img = self.camera.acquire_single_image()
            best_result['verification_image'] = verification_img
            
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
                print(f"      ✓ Verification: offset = {verify_offset_px} px = ({verify_offset_nm[0]:.0f}, {verify_offset_nm[1]:.0f}) nm")
                best_result['verification_offset_px'] = verify_offset_px
                best_result['verification_offset_nm'] = verify_offset_nm
            else:
                print(f"      ⚠️  Warning: Could not detect fiducial in verification image!")
        else:
            print(f"   ❌ {label} not found in search region")
        # ========================================
        # Return verification-based dictionary
        # ========================================
        if best_result and 'verification_image' in best_result:
            verification_output = {
                'stage_Y': best_result['stage_Y'],
                'stage_Z': best_result['stage_Z'],
                'pixel_pos': verify_result['position'] if verify_result else None,
                'pixel_offset': verify_offset_px if verify_result else None,
                'stage_offset_nm': verify_offset_nm if verify_result else None,
                'confidence': verify_result['confidence'] if verify_result else None,
                'method': verify_result['method'] if verify_result else None,
                'image': best_result['verification_image'].copy(),
            }

        if plot_progress:
            self.plot_search_progress(search_data=search_data, label=label,
                                      best_result=best_result, best_confidence=best_confidence)
        
        if verification_output:    
            return verification_output
        else:
            print("   ⚠️  No verification image available, returning None.")
            return None



    def plot_search_progress(self, search_data=None, label=None, best_result=None, best_confidence=-1.0):
        # ========================================
        # VISUALIZATION: Create grid of all search images
        # ========================================
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
                
                # Draw markers (using cv2 to draw on array)
                cv2.drawMarker(zoomed_rgb, (int(rel_cx), int(rel_cy)), 
                            (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(zoomed_rgb, (int(rel_fx), int(rel_fy)), 
                            (255, 0, 0), cv2.MARKER_TILTED_CROSS, 25, 2)
                
                confidence = detection.get('confidence', 0.0)
                method = detection.get('method', 'unknown')[:8]  # truncate
                
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
            
            # Keep tick labels visible but make them smaller
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

# =============================================================================
# TEST A: CV DETECTION ON MOCK IMAGES
# =============================================================================
def test_a_cv_detection():
    """Test A: Verify CV tools can detect fiducials in mock camera images."""
    print("\n" + "=" * 70)
    print("TEST A: CV DETECTION ON MOCK IMAGES")
    print("=" * 70)

    tester = AlignmentTester()
    tester.setup()

    # Test fiducials to detect (kept same style as original)
    test_cases = [
        (1, 'top_left', 'Block 1 Top-Left'),
        (1, 'bottom_right', 'Block 1 Bottom-Right'),
        (20, 'top_left', 'Block 20 Top-Left'),
        (20, 'bottom_right', 'Block 20 Bottom-Right')
    ]

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, (block_id, corner, label) in enumerate(test_cases):
        print(f"\n{'─' * 70}")
        print(f"Test Case {idx + 1}: {label}")
        print('─' * 70)

        # Get ground truth position (stage coords in nm)
        gt_pos = tester.gt_converter.get_fiducial_stage_position(block_id, corner)
        print(f"🎯 Ground truth position (stage coords): ({gt_pos[0]}, {gt_pos[1]}) nm")

        # Move stage to that position
        print(f"🚗 Moving stage to fiducial...")
        tester.stage.move_abs('y', int(round(gt_pos[0])))
        tester.stage.move_abs('z', int(round(gt_pos[1])))
        # Capture image
        img = tester.camera.acquire_single_image()

        # Capture image
        print(f"📷 Capturing image...")
        img = tester.camera.acquire_single_image()
        print(f"   Image: {img.shape}, range=[{img.min()}, {img.max()}]")

        # Expected position (image center)
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        print(f"   Expected at image center: {img_center}")

        # Run CV detection
        search_radius = 150  # pixels
        print(f"🔍 Running CV detection (search radius={search_radius}px)...")

        detection_result = tester.vt.find_fiducial_auto(
            img,
            expected_position=img_center,
            search_radius=search_radius
        )

        # Save raw image for debugging (image key per block/corner)
        key = f"block{block_id}_{corner}"
        tester.captured_images[key] = img.copy()

        if detection_result:
            found_pos = detection_result['position']
            confidence = detection_result['confidence']
            method = detection_result['method']

            # Pixel error
            error_px = np.sqrt((found_pos[0] - img_center[0]) ** 2 +
                               (found_pos[1] - img_center[1]) ** 2)

            # Convert to stage coordinates (nm)
            found_stage_Y = gt_pos[0] + (found_pos[0] - img_center[0]) * tester.camera.nm_per_pixel
            found_stage_Z = gt_pos[1] + (found_pos[1] - img_center[1]) * tester.camera.nm_per_pixel

            error_stage = np.sqrt((found_stage_Y - gt_pos[0]) ** 2 +
                                  (found_stage_Z - gt_pos[1]) ** 2)

            print(f"✅ FIDUCIAL DETECTED!")
            print(f"   Method: {method}")
            print(f"   Found at: {found_pos} px")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Pixel error: {error_px:.2f} px")
            print(f"   Stage error: {error_stage:.1f} nm ({error_stage / 1000:.3f} µm)")

            results.append({
                'label': label,
                'success': True,
                'method': method,
                'pixel_error': error_px,
                'stage_error_nm': error_stage,
                'confidence': confidence
            })

            # Plot zoomed region centered on image center
            ax = axes[idx]
            zoom_size = 300
            cy, cx = img_center[1], img_center[0]
            zoomed = img[max(0, cy - zoom_size):min(img.shape[0], cy + zoom_size),
                         max(0, cx - zoom_size):min(img.shape[1], cx + zoom_size)]

            ax.imshow(zoomed, cmap='gray', vmin=0, vmax=3500, origin='lower')

            # Mark expected center and found position relative to zoom
            ax.plot(zoom_size, zoom_size, 'g+', markersize=25, markeredgewidth=3, label='Expected')
            found_rel_x = zoom_size + (found_pos[0] - img_center[0])
            found_rel_y = zoom_size + (found_pos[1] - img_center[1])
            ax.plot(found_rel_x, found_rel_y, 'rx', markersize=20, markeredgewidth=3, label='Detected')

            ax.set_title(f'{label}\n{method}, error={error_px:.1f}px', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            print(f"❌ DETECTION FAILED!")
            results.append({
                'label': label,
                'success': False,
                'method': None,
                'pixel_error': None,
                'stage_error_nm': None,
                'confidence': None
            })

            ax = axes[idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=3500, origin='lower')
            ax.set_title(f'{label}\n❌ DETECTION FAILED', fontweight='bold', color='red')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_a_cv_detection.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: test_a_cv_detection.png")
    plt.show()

    # Summary printing
    print(f"\n{'=' * 70}")
    print("TEST A SUMMARY")
    print('=' * 70)

    successful = sum(1 for r in results if r['success'])
    total = len(results)

    for r in results:
        if r['success']:
            print(f"✅ {r['label']}: {r['method']}, error={r['pixel_error']:.1f}px")
            print(f"   Confidence: {r['confidence']:.3f}")
        else:
            print(f"❌ {r['label']}: FAILED")

    print(f"\nSuccess rate: {successful}/{total} ({successful / total * 100:.0f}%)")

    if successful == total:
        avg_error = np.mean([r['pixel_error'] for r in results if r['success']])
        print(f"Average pixel error: {avg_error:.2f} px")
        print(f"\n🎉 TEST A PASSED - CV detection works on mock images!")
        return True, tester
    else:
        print(f"\n❌ TEST A FAILED - Some fiducials not detected")
        return False, tester


# =============================================================================
# TEST B: BLIND CALIBRATION (FIXED & consistent)
# =============================================================================
def test_b_blind_calibration():
    """
    Test B: Calibrate coordinate transformation WITHOUT using ground truth.
    Uses measured Block1 TL and Block20 BR.
    """
    print("\n" + "=" * 70)
    print("TEST B: BLIND CALIBRATION (NO GROUND TRUTH) - FIXED")
    print("=" * 70)

    tester = AlignmentTester()
    tester.setup()

    # Block size (µm)
    block_size = tester.layout['block_layout']['block_size']
    print(f"\n📐 Block size: {block_size} µm")

    # STEP 1: Block 1 Top-Left
    print(f"\n{'─' * 70}")
    print("STEP 1: Find Block 1 Top-Left Fiducial")
    print('─' * 70)

    # Block 1 design center and local top-left (local relative to bottom-left)
    block1_center = tester.layout['blocks'][1]['design_position']
    block1_tl_local = tester.layout['blocks'][1]['fiducials']['top_left']

    # Convert to global design coords (µm): center + (local - block_size/2)
    block1_tl_global_u = block1_center[0] + (block1_tl_local[0] - block_size / 2.0)
    block1_tl_global_v = block1_center[1] + (block1_tl_local[1] - block_size / 2.0)

    print(f"📍 Block 1 TL global design position: ({block1_tl_global_u}, {block1_tl_global_v}) µm")

    # Search region around origin (design assumed near origin)
    search_result_1 = tester.search_for_fiducial(
        center_y_nm=block1_tl_global_u * 1000.0,
        center_z_nm=block1_tl_global_v * 1000.0,
        search_radius_nm=10000,
        step_nm=5000,
        label="Block 1 TL"
    )

    if not search_result_1:
        print(f"❌ Failed to find Block 1 TL")
        return False, tester

    measured_block1_tl = (search_result_1['stage_Y'], search_result_1['stage_Z'])
    print(f"✅ Block 1 TL found at: ({measured_block1_tl[0]}, {measured_block1_tl[1]}) nm")

    tester.measured_fiducials['block1_top_left'] = measured_block1_tl
    tester.captured_images['block1_tl'] = search_result_1['image']

    # STEP 2: Block 20 Bottom-Right
    print(f"\n{'─' * 70}")
    print("STEP 2: Find Block 20 Bottom-Right Fiducial")
    print('─' * 70)

    block20_center = tester.layout['blocks'][20]['design_position']
    block20_br_local = tester.layout['blocks'][20]['fiducials']['bottom_right']

    # Convert to global design coords (µm)
    block20_br_global_u = block20_center[0] + (block20_br_local[0] - block_size / 2.0)
    block20_br_global_v = block20_center[1] + (block20_br_local[1] - block_size / 2.0)

    print(f"📍 Block 20 BR global design position: ({block20_br_global_u}, {block20_br_global_v}) µm")

    # Compute translation measured_stage - design (in nm) using Block1
    design_block1_tl_nm = (block1_tl_global_u * 1000.0, block1_tl_global_v * 1000.0)
    measured_block1_tl_nm = measured_block1_tl
    translation_nm = (measured_block1_tl_nm[0] - design_block1_tl_nm[0],
                      measured_block1_tl_nm[1] - design_block1_tl_nm[1])

    # Expected stage for block20 = design_block20_br_nm + translation_nm
    design_block20_br_nm = (block20_br_global_u * 1000.0, block20_br_global_v * 1000.0)
    expected_y = design_block20_br_nm[0] + translation_nm[0]
    expected_z = design_block20_br_nm[1] + translation_nm[1]

    print(f"   Expected stage position (from measured translation): ({expected_y:.0f}, {expected_z:.0f}) nm")
    print(f"\n🔍 Searching around expected position...")

    search_result_2 = tester.search_for_fiducial(
        center_y_nm=expected_y,
        center_z_nm=expected_z,
        search_radius_nm=100000,
        step_nm=20000,
        label="Block 20 BR"
    )

    if not search_result_2:
        print(f"❌ Failed to find Block 20 BR")
        return False, tester

    measured_block20_br = (search_result_2['stage_Y'], search_result_2['stage_Z'])
    print(f"✅ Block 20 BR found at: ({measured_block20_br[0]}, {measured_block20_br[1]}) nm")

    tester.measured_fiducials['block20_bottom_right'] = measured_block20_br
    tester.captured_images['block20_br'] = search_result_2['image']

    # STEP 3: Calibrate transformation
    print(f"\n{'─' * 70}")
    print("STEP 3: Calibrate Coordinate Transformation")
    print('─' * 70)

    # Design positions in µm -> convert to nm for calibration
    design_block1_tl = (block1_tl_global_u, block1_tl_global_v)  # µm
    design_block20_br = (block20_br_global_u, block20_br_global_v)  # µm

    design_points_nm = [(design_block1_tl[0] * 1000.0, design_block1_tl[1] * 1000.0),
                        (design_block20_br[0] * 1000.0, design_block20_br[1] * 1000.0)]
    measured_points_nm = [measured_block1_tl, measured_block20_br]

    print("Calibration points (design in µm, measured in nm):")
    print(f"  Point 1 design (µm): {design_block1_tl}  -> design (nm): {design_points_nm[0]}")
    print(f"  Point 1 measured (nm): {measured_points_nm[0]}")
    print(f"  Point 2 design (µm): {design_block20_br} -> design (nm): {design_points_nm[1]}")
    print(f"  Point 2 measured (nm): {measured_points_nm[1]}")

    # Calibrate
    calibrated_transform = CoordinateTransform()
    calibration_result = calibrated_transform.calibrate(
        measured_points=measured_points_nm,
        design_points=design_points_nm
    )

    print(f"\n✅ CALIBRATION COMPLETE")
    print(f"   Method: {calibration_result['method']}")
    print(f"   Rotation: {calibration_result['angle_deg']:.4f}°")
    print(f"   Translation (nm): ({calibration_result['translation_nm'][0]:.1f}, {calibration_result['translation_nm'][1]:.1f})")
    print(f"   Mean error: {calibration_result.get('mean_error_nm', 0.0):.3f} nm")
    print(f"   Max error: {calibration_result.get('max_error_nm', 0.0):.3f} nm")

    # STEP 4: Validate against ground truth (simulation only)
    print(f"\n{'─' * 70}")
    print("STEP 4: Validate Against Ground Truth")
    print('─' * 70)

    gt = tester.layout['simulation_ground_truth']
    gt_rotation = gt['rotation_deg']
    gt_translation = tuple(gt['translation_nm'])

    rotation_error = abs(calibration_result['angle_deg'] - gt_rotation)
    translation_error = np.hypot(
        calibration_result['translation_nm'][0] - gt_translation[0],
        calibration_result['translation_nm'][1] - gt_translation[1]
    )

    print(f"Ground truth: rotation={gt_rotation}°, translation={gt_translation} nm")
    print(f"Calibration error: rotation_err={rotation_error:.4f}°, translation_err={translation_error:.1f} nm")

    # store calibrated converter
    tester.calibrated_converter = CoordinateConverter(tester.layout)
    tester.calibrated_converter.set_transformation(
        calibration_result['angle_deg'],
        tuple(calibration_result['translation_nm'])
    )

    # STEP 5: Visualization - show captured images and calibration diagram
    print(f"\n{'─' * 70}")
    print("STEP 5: Generate Visualization")
    print('─' * 70)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Block 1 TL image
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = tester.captured_images['block1_tl']
    zoom_size = 300
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
    zoomed1 = img1[max(0, center[1] - zoom_size):min(img1.shape[0], center[1] + zoom_size),
                   max(0, center[0] - zoom_size):min(img1.shape[1], center[0] + zoom_size)]
    ax1.imshow(zoomed1, cmap='gray', vmin=0, vmax=3500, origin='lower')
    ax1.set_title('Block 1 Top-Left\n(Found)', fontweight='bold', fontsize=12)
    ax1.plot(zoom_size, zoom_size, 'r+', markersize=20, markeredgewidth=3)
    ax1.grid(True, alpha=0.3)

    # Block 20 BR image
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = tester.captured_images['block20_br']
    center = (img2.shape[1] // 2, img2.shape[0] // 2)
    zoomed2 = img2[max(0, center[1] - zoom_size):min(img2.shape[0], center[1] + zoom_size),
                   max(0, center[0] - zoom_size):min(img1.shape[1], center[0] + zoom_size)]
    ax2.imshow(zoomed2, cmap='gray', vmin=0, vmax=3500, origin='lower')
    ax2.set_title('Block 20 Bottom-Right\n(Found)', fontweight='bold', fontsize=12)
    ax2.plot(zoom_size, zoom_size, 'r+', markersize=20, markeredgewidth=3)
    ax2.grid(True, alpha=0.3)

    # Calibration diagram (design vs measured)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    # Design points (µm)
    d1 = design_block1_tl
    d2 = design_block20_br
    ax3.plot([d1[0], d2[0]], [d1[1], d2[1]], 'b-o', markersize=10, label='Design', linewidth=2)

    # Measured points (converted to µm)
    m1 = (measured_block1_tl[0] / 1000.0, measured_block1_tl[1] / 1000.0)
    m2 = (measured_block20_br[0] / 1000.0, measured_block20_br[1] / 1000.0)
    ax3.plot([m1[0], m2[0]], [m1[1], m2[1]], 'r-s', markersize=10, label='Measured', linewidth=2)

    ax3.set_xlabel('U (µm)', fontsize=10)
    ax3.set_ylabel('V (µm)', fontsize=10)
    ax3.set_title('Calibration Points\n(Design vs Measured)', fontweight='bold', fontsize=12)
    ax3.legend()

    # Results text area
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    rotation_ok = rotation_error < 1
    translation_ok = translation_error < 10000
    status = "✅ PASS" if (rotation_ok and translation_ok) else "❌ FAIL"

    results_text = f"""
{'=' * 100}
TEST B: BLIND CALIBRATION - RESULTS
{'=' * 100}

MEASURED FIDUCIALS:
  Block 1 TL:   Design ({design_block1_tl[0]:.1f}, {design_block1_tl[1]:.1f}) µm
                Measured ({measured_block1_tl[0]}, {measured_block1_tl[1]}) nm

  Block 20 BR:  Design ({design_block20_br[0]:.1f}, {design_block20_br[1]:.1f}) µm
                Measured ({measured_block20_br[0]}, {measured_block20_br[1]}) nm

CALIBRATION RESULTS:
  Method:               {calibration_result['method']}
  Rotation (calibrated): {calibration_result['angle_deg']:.4f}°
  Rotation (ground truth): {gt_rotation}°
  Rotation error:       {rotation_error:.4f}° {'✅' if rotation_ok else '❌'}

  Translation (calibrated): ({calibration_result['translation_nm'][0]:.1f}, {calibration_result['translation_nm'][1]:.1f}) nm
  Translation (ground truth): {gt_translation} nm
  Translation error:    {translation_error:.1f} nm {'✅' if translation_ok else '❌'}

  Mean residual:        {calibration_result.get('mean_error_nm', 0.0):.3f} nm
  Max residual:         {calibration_result.get('max_error_nm', 0.0):.3f} nm

STATUS: {status}
{'=' * 100}
    """

    ax4.text(0.01, 0.5, results_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.savefig('test_b_blind_calibration.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: test_b_blind_calibration.png")
    plt.show()

    # Final pass/fail
    if rotation_ok and translation_ok:
        print(f"\n🎉 TEST B PASSED - Calibration successful!")
        return True, tester
    else:
        print(f"\n❌ TEST B FAILED - Calibration error too large")
        if not rotation_ok:
            print(f"   Rotation error {rotation_error:.4f}° > 0.1° threshold")
        if not translation_ok:
            print(f"   Translation error {translation_error:.1f} nm > 1000 nm threshold")
        return False, tester

# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================

def main():
    """Run alignment CV detection and blind calibration tests sequentially."""
    print("\n" + "=" * 80)
    print("RUNNING ALIGNMENT SYSTEM TESTS")
    print("=" * 80)

    start_time = time.time()

    try:
        # ---------------------------------------------------------------
        # TEST A: CV Detection on Mock Images
        # ---------------------------------------------------------------
        print("\n🚀 Starting Test A: CV Detection on Mock Images...")
        success_a, tester_a = test_a_cv_detection()
        if not success_a:
            print("\n❌ Test A failed — stopping further tests.")
            print("=" * 80)
            return

        # ---------------------------------------------------------------
        # TEST B: Blind Calibration
        # ---------------------------------------------------------------
        print("\n🚀 Starting Test B: Blind Calibration...")
        success_b, tester_b = test_b_blind_calibration()

        # ---------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"Test A (CV Detection): {'✅ PASSED' if success_a else '❌ FAILED'}")
        print(f"Test B (Blind Calibration): {'✅ PASSED' if success_b else '❌ FAILED'}")
        print(f"\nTotal runtime: {total_time:.2f} seconds")

        if success_a and success_b:
            print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
        else:
            print("\n⚠️ Some tests failed. Check logs and saved plots for debugging.")

    except Exception as e:
        print("\n💥 FATAL ERROR during testing!")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


# Standard Python entry point
if __name__ == "__main__":
    main()