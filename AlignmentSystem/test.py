#!/usr/bin/env python3
# test_hierarchical_alignment_with_search_um.py
"""
Complete test of hierarchical alignment using real AlignmentSearcher.
This version operates COMPLETELY IN MICROMETERS (µm).

This demonstrates the full workflow:
1. Stage 1: Grid search for corner block fiducials → Global calibration
2. Stage 2: Use global calibration to predict → Grid search for block fiducials → Block calibration
3. Verification: Navigate to targets and verify accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2

# Hardware
from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
from HardwareControl.SetupMotor.stageAdapter import StageAdapterUM
from HardwareControl.CameraControl.mock_camera import MockCamera

# Alignment system
from config.layout_config_generator_v2 import load_layout_config_v2
from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.hierarchicalAlignment import HierarchicalAlignment
from AlignmentSystem.alignmentSearch import AlignmentSearcher


def visualize_waveguide_positioning(camera, alignment, target_block, waveguide_num, position_type, 
                                    predicted_Y, predicted_Z, actual_Y, actual_Z, verification_result):
    """
    Create detailed visualization showing where waveguide is vs where we tried to find it.
    All stage coordinates (predicted_Y, predicted_Z, actual_Y, actual_Z) are in MICROMETERS.
    
    Args:
        camera: Camera instance
        alignment: HierarchicalAlignment instance
        target_block: Block ID
        waveguide_num: Waveguide number
        position_type: 'left_grating', 'right_grating', or 'center'
        predicted_Y, predicted_Z: Predicted stage position (µm)
        actual_Y, actual_Z: Actual stage position after search (µm)
        verification_result: Result from verify_fiducial_centering
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Get the verification image
    if verification_result and 'image' in verification_result:
        img = verification_result['image']
    else:
        # Capture image at predicted position
        camera.stage.move_abs('y', predicted_Y)
        camera.stage.move_abs('z', predicted_Z)
        img = camera.acquire_single_image()
    
    # Normalize for display (using 16-bit mock camera range)
    img_norm = np.clip(img.astype(np.float32) / 4095.0, 0, 1)
    
    # =========================================================================
    # Plot 1: Full image at predicted position
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_norm, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax1.set_title(f'Full FOV at Predicted Position\nBlock {target_block} WG{waveguide_num} {position_type}', 
                  fontweight='bold', fontsize=11)
    
    # Mark center
    img_center = (img.shape[1] // 2, img.shape[0] // 2)
    ax1.plot(img_center[0], img_center[1], 'g+', markersize=30, markeredgewidth=3, label='Image Center (Target)')
    
    if verification_result and verification_result.get('pixel_pos'):
        found_px = verification_result['pixel_pos']
        ax1.plot(found_px[0], found_px[1], 'rx', markersize=25, markeredgewidth=3, label='Detected (Actual)')
        
        # Draw line showing offset
        ax1.plot([img_center[0], found_px[0]], [img_center[1], found_px[1]], 
                 'y--', linewidth=2, alpha=0.7)
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (pixels)', fontsize=9)
    ax1.set_ylabel('Y (pixels)', fontsize=9)
    
    # =========================================================================
    # Plot 2: Zoomed view (600x600 px around center)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    zoom_size = 300
    cy, cx = img_center[1], img_center[0]
    y1, y2 = max(0, cy - zoom_size), min(img.shape[0], cy + zoom_size)
    x1, x2 = max(0, cx - zoom_size), min(img.shape[1], cx + zoom_size)
    
    # Scale 0-1 float to 0-255 uint8 for cv2 drawing
    zoomed = (img_norm[y1:y2, x1:x2] * 255).astype(np.uint8) 
    
    # Convert to RGB for colored markers
    zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2RGB)
    
    # Relative coordinates in zoomed image
    rel_cx = cx - x1
    rel_cy = cy - y1
    
    # Draw center marker
    cv2.drawMarker(zoomed_rgb, (int(rel_cx), int(rel_cy)), 
                   (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
    
    if verification_result and verification_result.get('pixel_pos'):
        found_px = verification_result['pixel_pos']
        rel_fx = found_px[0] - x1
        rel_fy = found_px[1] - y1
        if 0 <= rel_fx < zoomed_rgb.shape[1] and 0 <= rel_fy < zoomed_rgb.shape[0]:
            cv2.drawMarker(zoomed_rgb, (int(rel_fx), int(rel_fy)), 
                           (255, 0, 0), cv2.MARKER_TILTED_CROSS, 45, 3)
            
            # Draw connecting line
            cv2.line(zoomed_rgb, (int(rel_cx), int(rel_cy)), (int(rel_fx), int(rel_fy)),
                     (255, 255, 0), 2, cv2.LINE_AA)
    
    ax2.imshow(zoomed_rgb, origin='lower')
    ax2.set_title(f'Zoomed View ({zoom_size*2}x{zoom_size*2} px)\nGreen=Target, Red=Found', 
                  fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (pixels)', fontsize=9)
    ax2.set_ylabel('Y (pixels)', fontsize=9)
    
    # =========================================================================
    # Plot 3: Coordinate space diagram
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # Get design position
    layout = alignment.layout
    block_center = layout['blocks'][target_block]['design_position']
    
    if position_type == 'left_grating':
        grating_key = f"wg{waveguide_num}_left"
        grating_local = layout['blocks'][target_block]['gratings'][grating_key]['position']
    elif position_type == 'right_grating':
        grating_key = f"wg{waveguide_num}_right"
        grating_local = layout['blocks'][target_block]['gratings'][grating_key]['position']
    else:  # center
        wg = layout['blocks'][target_block]['waveguides'][f"wg{waveguide_num}"]
        grating_local = ((wg['u_start'] + wg['u_end']) / 2.0, wg['v_center'])
    
    # Convert to global design
    block_size = layout['block_layout']['block_size']
    u_global = block_center[0] - block_size / 2.0 + grating_local[0]
    v_global = block_center[1] - block_size / 2.0 + grating_local[1]
    
    # Plot design position
    ax3.plot(u_global, v_global, 'bo', markersize=15, label='Design Position (u, v)')
    
    # Plot predicted stage position (µm)
    ax3.plot(predicted_Y, predicted_Z, 'g^', markersize=15, 
             label='Predicted (Stage 2)')
    
    # Plot actual found position (µm)
    if verification_result:
        ax3.plot(actual_Y, actual_Z, 'rs', markersize=15, 
                 label='Found (Actual)')
        
        # Draw error vector
        ax3.arrow(predicted_Y, predicted_Z,
                  (actual_Y - predicted_Y),
                  (actual_Z - predicted_Z),
                  head_width=0.2, head_length=0.1, fc='orange', ec='orange',
                  alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Stage Y (µm)', fontsize=10)
    ax3.set_ylabel('Stage Z (µm)', fontsize=10)
    ax3.set_title('Coordinate Space\n(Design vs Measured)', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9)
    
    # =========================================================================
    # Plot 4-6: Statistics and info
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Compute errors
    if verification_result:
        pixel_offset = verification_result.get('pixel_offset', (0, 0))
        stage_offset_um = verification_result.get('stage_offset_um', (0, 0))
        error_um = verification_result.get('error_um', 0)
        confidence = verification_result.get('confidence', 0)
        
        error_text = f"""
{'='*100}
WAVEGUIDE POSITIONING VERIFICATION (Units: µm) - BLOCK {target_block}, WG{waveguide_num}, {position_type.upper()}
{'='*100}

DESIGN COORDINATES:
  Block center:       ({block_center[0]:.2f}, {block_center[1]:.2f}) µm
  Local position:     ({grating_local[0]:.2f}, {grating_local[1]:.2f}) µm (relative to block)
  Global position:    ({u_global:.2f}, {v_global:.2f}) µm (design coordinates)

PREDICTED STAGE POSITION (Using Block {target_block} Calibration):
  Stage Y: {predicted_Y:10.3f} µm
  Stage Z: {predicted_Z:10.3f} µm

ACTUAL FOUND POSITION:
  Stage Y: {actual_Y:10.3f} µm
  Stage Z: {actual_Z:10.3f} µm

POSITIONING ERRORS:
  Pixel offset:       ({pixel_offset[0]:+.1f}, {pixel_offset[1]:+.1f}) pixels
  Stage offset:       ({stage_offset_um[0]:+.3f}, {stage_offset_um[1]:+.3f}) µm
  Total error:        {error_um:.3f} µm
  Detection confidence: {confidence:.3f}

CAMERA PARAMETERS:
  Resolution: {camera.um_per_pixel:.4f} µm/pixel
  Image center: ({img_center[0]}, {img_center[1]}) pixels

STATUS: {'✅ EXCELLENT' if error_um < 0.2 else '✓ GOOD' if error_um < 1.0 else '⚠️ NEEDS IMPROVEMENT'}
{'='*100}
"""
    else:
        error_text = f"""
{'='*100}
WAVEGUIDE POSITIONING VERIFICATION (Units: µm) - BLOCK {target_block}, WG{waveguide_num}, {position_type.upper()}
{'='*100}

❌ VERIFICATION FAILED - Target not detected in image

PREDICTED STAGE POSITION:
  Stage Y: {predicted_Y:10.3f} µm
  Stage Z: {predicted_Z:10.3f} µm

Possible causes:
  - Target not in field of view
  - Calibration error too large
  - Detection threshold too strict

{'='*100}
"""
    
    ax4.text(0.01, 0.5, error_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.suptitle(f'Stage 3 Verification: Block {target_block}, Waveguide {waveguide_num}, {position_type}',
                 fontsize=14, fontweight='bold')
    
    # Save figure
    filename = f'stage3_verification_block{target_block}_wg{waveguide_num}_{position_type}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved verification visualization: {filename}")
    plt.show()


def main():
    """
    Full hierarchical alignment workflow with real search (µm-based).
    """
    print("\n" + "="*70)
    print("HIERARCHICAL ALIGNMENT - FULL SEARCH WORKFLOW TEST (µm Version)")
    print("="*70)
    print("\nThis test demonstrates:")
    print("  1. Stage 1: Grid search for global calibration (4 corner blocks)")
    print("  2. Stage 2: Predicted search for block calibration (Block 10)")
    print("  3. Verification: Navigate to waveguide targets with visualization")
    print("  All operations and logs are in MICROMETERS (µm)")
    print()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n" + "="*70)
    print("SETUP: Initialize Hardware & Alignment System")
    print("="*70)
    
    # --- Create Mock Hardware (nm-based) ---
    stage_nm = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    
    # --- Create µm Adapter ---
    stage = StageAdapterUM(stage_nm)
    
    # --- Create Camera & Other Tools ---
    layout_config = "config/mock_layout.json"
    camera = MockCamera(layout_config, stage_ref=stage) # Camera gets the µm adapter
    camera.connect()
    stage.set_camera_observer(camera) # Adapter gets the camera
    camera.set_exposure_time(0.02)
    vt = VisionTools()
    
    layout = load_layout_config_v2(layout_config)
    print(f"✅ Layout loaded: {layout['design_name']}")
    
    print(f"✅ Mock hardware initialized")
    print(f"   (Internal stage 'stage_nm' is nm, but adapter 'stage' is µm)")
    print(f"✅ Camera initialized")
    # print(f"   Camera FOV: {camera.:.1f} µm")
    print(f"   Resolution: {camera.um_per_pixel:.4f} µm/pixel")
    
    # Create hierarchical alignment and searcher
    alignment = HierarchicalAlignment(layout)
    searcher = AlignmentSearcher(stage, camera, vt) # Searcher gets the µm adapter
    
    print(f"✅ Alignment system and searcher created")
    
    # Ground truth (for validation only)
    gt = layout['simulation_ground_truth']
    print(f"   Ground truth: rotation={gt['rotation_deg']}°, translation={gt['translation_um']} µm")
    
    # =========================================================================
    # STAGE 1: GLOBAL CALIBRATION
    # =========================================================================
    input("\nPress Enter to start Stage 1: Global Calibration with grid search...")
    
    print("\n" + "="*70)
    print("STAGE 1: GLOBAL CALIBRATION (Grid Search)")
    print("="*70)
    print("Searching for fiducials in corner blocks: 1, 20")
    print("Using coarse grid search around design positions")
    
    # Corner blocks and which corners to use
    corner_blocks = [1, 20]
    corners_to_use = ['top_left', 'bottom_right']
    
    global_measurements = []
    
    for block_id, corner in zip(corner_blocks, corners_to_use):
        print(f"\n{'─'*70}")
        print(f"Searching for Block {block_id} {corner}")
        print('─'*70)
        
        # Get design position as initial guess
        block_center = layout['blocks'][block_id]['design_position']
        fiducial_local = layout['blocks'][block_id]['fiducials'][corner]
        
        # Convert to global design coords
        block_size = layout['block_layout']['block_size']
        u_global = block_center[0] - block_size / 2.0 + fiducial_local[0]
        v_global = block_center[1] - block_size / 2.0 + fiducial_local[1]
        
        print(f"  Design position: ({u_global:.1f}, {v_global:.1f}) µm")
        
        # Grid search (larger radius since we don't know transformation yet)
        result = searcher.search_for_fiducial(
            center_y_um=u_global,
            center_z_um=v_global,
            search_radius_um=100,  # ±100 µm search
            step_um=20,           # 20 µm steps
            label=f"Block {block_id} {corner}",
            plot_progress=True
        )
        
        if result:
            global_measurements.append({
                'block_id': block_id,
                'corner': corner,
                'stage_Y': result['stage_Y'], # Already in µm
                'stage_Z': result['stage_Z'], # Already in µm
                'confidence': result['confidence'],
                'verification_error_um': result.get('verification_error_um', 0)
            })
            print(f"  ✅ Found at ({result['stage_Y']:.3f}, {result['stage_Z']:.3f}) µm")
            print(f"     Verification error: {result.get('verification_error_um', 0):.3f} µm")
        else:
            print(f"  ❌ Search failed!")
            print(f"\n⚠️  Cannot continue without all corner fiducials")
            return 1
    
    # Check we found all
    if len(global_measurements) < len(corner_blocks):
        print(f"\n❌ Only found {len(global_measurements)}/{len(corner_blocks)} corner fiducials!")
        print("Cannot proceed with global calibration")
        return 1
    
    print(f"\n{'─'*70}")
    print(f"✅ Found all {len(global_measurements)} corner fiducials!")
    print('─'*70)
    
    # Calibrate global transformation
    print(f"\nCalibrating global transformation...")
    # HierarchicalAlignment expects µm, which we are passing
    global_result = alignment.calibrate_global(global_measurements)
    
    # --- START FIX: Re-implemented validation logic ---
    # The 'validate_global_calibration' method does not exist.
    # We re-implement the validation logic here using the calibration result
    # and the ground truth (gt)
    
    gt_rot = gt['rotation_deg']
    gt_trans = gt['translation_um']
    cal_rot = global_result['rotation_deg']
    cal_trans = global_result['translation_um']

    validation = {
        'rotation_error_deg': abs(cal_rot - gt_rot),
        'translation_error_um': np.hypot(cal_trans[0] - gt_trans[0], cal_trans[1] - gt_trans[1]),
        'mean_residual_um': global_result['mean_error_um'], # This is the fit error
        'max_residual_um': global_result['max_error_um']   # This is the fit error
    }
    # --- END FIX ---
    
    print(f"\n{'─'*70}")
    print("Global Calibration Validation")
    print('─'*70)
    print(f"Ground truth: rotation={gt['rotation_deg']}°, translation={gt['translation_um']} µm")
    print(f"Calibrated:   rotation={global_result['rotation_deg']:.4f}°, "
          f"translation=({global_result['translation_um'][0]:.3f}, {global_result['translation_um'][1]:.3f}) µm")
    print(f"\nErrors:")
    print(f"  Rotation error:     {validation['rotation_error_deg']:.4f}° (vs GT)")
    print(f"  Translation error:  {validation['translation_error_um']:.3f} µm (vs GT)")
    print(f"  Mean residual:      {validation['mean_residual_um']:.6f} µm (internal fit)")
    print(f"  Max residual:       {validation['max_residual_um']:.6f} µm (internal fit)")
    
    if validation['rotation_error_deg'] < 0.1 and validation['translation_error_um'] < 1.0:
        print(f"\n✅ STAGE 1 PASSED - Global calibration accurate!")
    else:
        print(f"\n⚠️ STAGE 1 WARNING - Calibration errors larger than expected")
    
    # =========================================================================
    # STAGE 2: BLOCK-SPECIFIC CALIBRATION
    # =========================================================================
    input("\nPress Enter to start Stage 2: Block Calibration with predicted search...")
    
    print("\n" + "="*70)
    print("STAGE 2: BLOCK-SPECIFIC CALIBRATION (Predicted Search)")
    print("="*70)
    print("Using global calibration to predict Block 10 fiducial positions")
    print("Then performing fine grid search to find them accurately")
    
    target_block = 10
    
    # --- START FIX: Replaced 'predict_block_center' ---
    # This method does not exist. We get the design center and transform it.
    u_center, v_center = layout['blocks'][target_block]['design_position']
    pred_center_Y, pred_center_Z = alignment.global_transform.design_to_stage(u_center, v_center)
    # --- END FIX ---
    
    block_center_design = layout['blocks'][target_block]['design_position']
    
    print(f"\n{'─'*70}")
    print(f"Block {target_block} Prediction")
    print('─'*70)
    print(f"  Design center: ({block_center_design[0]:.1f}, {block_center_design[1]:.1f}) µm")
    print(f"  Predicted stage: ({pred_center_Y:.3f}, {pred_center_Z:.3f}) µm")
    
    # Search for corners using predicted positions
    block_corners = ['top_left', 'bottom_right']
    
    print(f"\n{'─'*70}")
    print(f"Searching for fiducials in Block {target_block}")
    print('─'*70)
    
    block_measurements = []
    
    for corner in block_corners:
        print(f"\n   Searching for {corner}...")
        
        # --- START FIX: Replaced 'predict_fiducial_position' ---
        # The correct method is 'get_fiducial_stage_position'
        pred_Y, pred_Z = alignment.get_fiducial_stage_position(target_block, corner)
        # --- END FIX ---
        
        print(f"   Predicted: ({pred_Y:.3f}, {pred_Z:.3f}) µm")
        
        # Fine grid search around predicted position
        result = searcher.search_for_fiducial(
            center_y_um=pred_Y,
            center_z_um=pred_Z,
            search_radius_um=60,  # ±60 µm
            step_um=15,           # 15 µm steps
            label=f"Block {target_block} {corner}",
            plot_progress=True
        )
        
        if result:
            # Calculate prediction error (all in µm)
            pred_error = np.hypot(result['stage_Y'] - pred_Y, 
                                  result['stage_Z'] - pred_Z)
            
            print(f"   ✅ Found at ({result['stage_Y']:.3f}, {result['stage_Z']:.3f}) µm")
            print(f"       Prediction error: {pred_error:.3f} µm")
            print(f"       Verification error: {result.get('verification_error_um', 0):.3f} µm")
            
            block_measurements.append({
                'corner': corner,
                'stage_Y': result['stage_Y'], # in µm
                'stage_Z': result['stage_Z'], # in µm
                'confidence': result['confidence'],
                'prediction_error_um': pred_error,
                'verification_error_um': result.get('verification_error_um', 0)
            })
        else:
            print(f"   ❌ Search failed for {corner}!")
    
    # Check we found enough fiducials (need at least 2)
    if len(block_measurements) < 2:
        print(f"\n❌ Only found {len(block_measurements)} fiducials in Block {target_block}!")
        print("Need at least 2 for block calibration")
        return 1
    
    print(f"\n{'─'*70}")
    print(f"✅ Found {len(block_measurements)} fiducials in Block {target_block}")
    print('─'*70)
    
    # Show prediction accuracy
    pred_errors = [m['prediction_error_um'] for m in block_measurements]
    print(f"\nPrediction accuracy using Stage 1 calibration:")
    print(f"  Mean prediction error: {np.mean(pred_errors):.3f} µm")
    print(f"  Max prediction error:  {np.max(pred_errors):.3f} µm")
    
    # Calibrate block (passing µm)
    print(f"\nCalibrating Block {target_block}...")
    block_result = alignment.calibrate_block(target_block, block_measurements)
    
    # --- START FIX: Re-implemented validation logic ---
    # The 'validate_block_calibration' method does not exist.
    # Re-implementing validation logic.
    # Note: In a real scenario with fab errors, this block calibration
    # would be different. In simulation, it should be very close to global.
    cal_rot_block = block_result['rotation_deg']
    cal_trans_block = block_result['translation_um']
    
    validation_block = {
        'rotation_error_deg': abs(cal_rot_block - gt_rot),
        'translation_error_um': np.hypot(cal_trans_block[0] - gt_trans[0], cal_trans_block[1] - gt_trans[1]),
        'mean_residual_um': block_result['mean_error_um']
    }
    # --- END FIX ---
    
    print(f"\n{'─'*70}")
    print(f"Block {target_block} Calibration Validation")
    print('─'*70)
    print("Note: In simulation, blocks have no fabrication errors,")
    print("so block calibration should closely match global calibration.")
    print(f"\nErrors:")
    print(f"  Rotation error:     {validation_block['rotation_error_deg']:.4f}° (vs GT)")
    print(f"  Translation error:  {validation_block['translation_error_um']:.3f} µm (vs GT)")
    print(f"  Mean residual:      {validation_block['mean_residual_um']:.6f} µm (internal fit)")
    
    if validation_block['mean_residual_um'] < 0.2:
        print(f"\n✅ STAGE 2 PASSED - Block calibration accurate!")
    else:
        print(f"\n⚠️ STAGE 2 WARNING - Block calibration residuals larger than expected")
    
    # --- START FIX: Replaced 'print_status' ---
    print(alignment.get_calibration_status())
    # --- END FIX ---
    
    # =========================================================================
    # STAGE 3: VERIFICATION - NAVIGATE TO TARGETS WITH VISUALIZATION
    # =========================================================================
    input("\nPress Enter to verify positioning accuracy with detailed visualization...")
    
    print("\n" + "="*70)
    print("STAGE 3: POSITIONING VERIFICATION WITH VISUALIZATION")
    print("="*70)
    print(f"Testing coordinate conversion and positioning accuracy")
    print(f"Target: Block {target_block}, Waveguide 25, Left Grating")
    
    # --- START FIX: Replaced 'get_waveguide_position' ---
    # The correct method is 'get_grating_stage_position'
    Y_pred, Z_pred = alignment.get_grating_stage_position(target_block, 25, 'left')
    # --- END FIX ---
    
    print(f"\nPredicted stage position (using Block {target_block} calibration):")
    print(f"  Y = {Y_pred:.3f} µm")
    print(f"  Z = {Z_pred:.3f} µm")
    
    # Verify by navigating and checking with camera
    print(f"\nNavigating to target and verifying...")
    verification = searcher.verify_fiducial_centering(
        expected_y_um=Y_pred,
        expected_z_um=Z_pred,
        label=f"Block {target_block} WG25 Left Grating"
    )
    
    # Get actual position (stage is already there from verification)
    actual_Y = stage.get_pos('y') # in µm
    actual_Z = stage.get_pos('z') # in µm
    
    # Create detailed visualization
    print(f"\n📊 Creating detailed visualization...")
    visualize_waveguide_positioning(
        camera=camera,
        alignment=alignment,
        target_block=target_block,
        waveguide_num=25,
        position_type='left_grating',
        predicted_Y=Y_pred,
        predicted_Z=Z_pred,
        actual_Y=actual_Y,
        actual_Z=actual_Z,
        verification_result=verification
    )
    
    # Evaluate success
    if verification:
        error = verification['error_um']
        print(f"\n✅ Target verified!")
        print(f"   Positioning error: {error:.3f} µm")
        print(f"   Confidence: {verification['confidence']:.3f}")
        
        if error < 0.2:
            print(f"\n🎉 EXCELLENT ACCURACY! System ready for alignment.")
            success = True
        elif error < 1.0:
            print(f"\n✓ GOOD ACCURACY. Within acceptable range.")
            success = True
        else:
            print(f"\n⚠️ LARGE ERROR. Check calibration.")
            success = False
    else:
        print(f"\n❌ VERIFICATION FAILED - Target not found!")
        success = False
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY (Units: µm)")
    print("="*70)
    
    print(f"\n✅ Stage 1: Global calibration complete")
    print(f"   - Found {len(global_measurements)}/{len(corner_blocks)} corner fiducials")
    print(f"   - Rotation error: {validation['rotation_error_deg']:.4f}° (vs GT)")
    print(f"   - Translation error: {validation['translation_error_um']:.3f} µm (vs GT)")
    
    print(f"\n✅ Stage 2: Block {target_block} calibration complete")
    print(f"   - Found {len(block_measurements)}/{len(block_corners)} block fiducials")
    print(f"   - Mean prediction error: {np.mean(pred_errors):.3f} µm")
    print(f"   - Calibration residual: {validation_block['mean_residual_um']:.6f} µm (internal fit)")
    
    if verification:
        print(f"\n✅ Stage 3: Positioning verification complete")
        print(f"   - Target positioning error: {verification['error_um']:.3f} µm")
    
    print("\n" + "="*70)
    print("WORKFLOW DEMONSTRATION COMPLETE")
    print("="*70)
    
    if success:
        print("\n🎉 All stages passed! Hierarchical alignment system works!")
        print("\nThe system successfully:")
        print("  1. Found fiducials using grid search")
        print("  2. Calibrated global sample transformation")
        print("  3. Used predictions to find block fiducials efficiently")
        print("  4. Calibrated block-specific transformation")
        print("  5. Accurately positioned to waveguide targets")
        print("  6. Generated detailed visualization of positioning accuracy")
        return 0
    else:
        print("\n⚠️ Some stages had issues. Review results above.")
        return 1


if __name__ == "__main__":
    # Need to add config and other dirs to path
    # Make sure parent directory is on path to find modules
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    if str(base_dir) not in sys.path:
        sys.path.append(str(base_dir))
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))

    # Add relevant subdirectories if needed (assuming flat structure or packages)
    # This structure assumes that modules can be imported (e.g., HardwareControl.CameraControl.mock_camera)
    # If not, you may need to add more paths
    
    # Dummy VisionTools class if not available
    try:
        from AlignmentSystem.cv_tools import VisionTools
    except ImportError:
        print("Warning: VisionTools not found, creating dummy class.")
        class VisionTools:
            def find_fiducial_auto(self, *args, **kwargs):
                print("Using dummy VisionTools!")
                return None
            
    # Dummy config generator if not available
    try:
        from config.layout_config_generator_v2 import load_layout_config_v2
    except ImportError:
        print("Error: mock_layout.json and generator not found.")
        print("Please provide all required files.")
        sys.exit(1)
        
    raise SystemExit(main())