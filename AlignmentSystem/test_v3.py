#!/usr/bin/env python3
# test_hierarchical_alignment_v3_um.py
"""
Full-mode µm-based integration test for HierarchicalAlignment.

Workflow:
  1) Stage 1 - Global calibration via grid search on corner fiducials
  2) Stage 2 - Predicted fine search for block fiducials + block calibration
  3) Stage 3 - Verify grating target and produce visualization

Important:
  - All coordinates and searches operate in MICROMETERS (µm).
  - Uses CoordinateTransformV3 and RuntimeLayout properly
"""

from pathlib import Path
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

# Project imports
from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
from HardwareControl.SetupMotor.stageAdapter import StageAdapterUM
from HardwareControl.CameraControl.mock_camera_v3 import MockCamera
from AlignmentSystem.hierarchicalAlignment_v3 import HierarchicalAlignment
from AlignmentSystem.alignmentSearch import AlignmentSearcher
from config.layout_models import RuntimeLayout, CameraLayout
from AlignmentSystem.cv_tools import VisionTools


def visualize_waveguide_positioning(camera, runtime_layout, target_block, waveguide_num, position_type,
                                    predicted_Y, predicted_Z, actual_Y, actual_Z, verification_result):
    """
    Create detailed visualization showing where waveguide is vs where we tried to find it.
    All stage coordinates are in MICROMETERS.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Acquire or reuse verification image
    if verification_result and 'image' in verification_result:
        img = verification_result['image']
    else:
        # Move stage to predicted position then capture
        camera.stage.move_abs('y', predicted_Y)
        camera.stage.move_abs('z', predicted_Z)
        img = camera.acquire_single_image()

    # Normalize for display
    img_norm = np.clip(img.astype(np.float32) / float(img.max() if img.max() > 0 else 1.0), 0, 1)

    # Plot 1: Full image at predicted position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_norm, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax1.set_title(f'Full FOV at Predicted Position\nBlock {target_block} WG{waveguide_num} {position_type}',
                  fontweight='bold', fontsize=11)
    img_center = (img.shape[1] // 2, img.shape[0] // 2)
    ax1.plot(img_center[0], img_center[1], 'g+', markersize=30, markeredgewidth=3, label='Image Center (Target)')

    if verification_result and verification_result.get('pixel_pos'):
        found_px = verification_result['pixel_pos']
        ax1.plot(found_px[0], found_px[1], 'rx', markersize=25, markeredgewidth=3, label='Detected (Actual)')
        ax1.plot([img_center[0], found_px[0]], [img_center[1], found_px[1]], 'y--', linewidth=2, alpha=0.7)

    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    # Plot 2: Zoomed view
    ax2 = fig.add_subplot(gs[0, 1])
    zoom_size = min(300, img.shape[0]//2, img.shape[1]//2)
    cy, cx = img_center[1], img_center[0]
    y1, y2 = max(0, cy - zoom_size), min(img.shape[0], cy + zoom_size)
    x1, x2 = max(0, cx - zoom_size), min(img.shape[1], cx + zoom_size)
    zoomed = (img_norm[y1:y2, x1:x2] * 255).astype(np.uint8)
    zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2RGB)
    rel_cx = cx - x1
    rel_cy = cy - y1
    cv2.drawMarker(zoomed_rgb, (int(rel_cx), int(rel_cy)), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)

    if verification_result and verification_result.get('pixel_pos'):
        found_px = verification_result['pixel_pos']
        rel_fx = found_px[0] - x1
        rel_fy = found_px[1] - y1
        if 0 <= rel_fx < zoomed_rgb.shape[1] and 0 <= rel_fy < zoomed_rgb.shape[0]:
            cv2.drawMarker(zoomed_rgb, (int(rel_fx), int(rel_fy)), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 45, 3)
            cv2.line(zoomed_rgb, (int(rel_cx), int(rel_cy)), (int(rel_fx), int(rel_fy)), (255, 255, 0), 2, cv2.LINE_AA)

    ax2.imshow(zoomed_rgb, origin='lower')
    ax2.set_title('Zoomed View (Green=Target, Red=Found)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Coordinate space diagram (stage coordinates, µm)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    # Get design position from RuntimeLayout
    block = runtime_layout.get_block(target_block)
    block_center = block.design_position
    block_size = runtime_layout.block_layout.block_size

    # Get grating or waveguide local position
    if position_type == 'left_grating':
        grating = block.get_grating(waveguide_num, 'left')
        grating_local = grating.position
    elif position_type == 'right_grating':
        grating = block.get_grating(waveguide_num, 'right')
        grating_local = grating.position
    else:  # center
        wg = block.get_waveguide(waveguide_num)
        grating_local = wg.center_position

    # ⚠️ FIX: Don't manually convert to global design coordinates
    # Instead, use the transform to get expected stage position from design
    from AlignmentSystem.coordinate_transform_v3 import CoordinateTransformV3
    transform = CoordinateTransformV3(runtime_layout)
    transform.sync_with_runtime()

    # Convert design local → stage using calibrated transform
    try:
        design_stage_Y, design_stage_Z = transform.block_local_to_stage(
            target_block, grating_local.u, grating_local.v
        )
    except Exception as e:
        print(f"Warning: Could not convert design to stage: {e}")
        # Fallback: use manual conversion (will be inaccurate)
        u_bl = block_center.u - block_size / 2.0
        v_bl = block_center.v - block_size / 2.0
        design_stage_Y = u_bl + grating_local.u
        design_stage_Z = v_bl + grating_local.v

    ax3.plot(design_stage_Y, design_stage_Z, 'bo', markersize=12, 
            label='Design Position (expected stage)')
    ax3.plot(predicted_Y, predicted_Z, 'g^', markersize=12, 
            label='Predicted (with calibration)')
    if verification_result:
        ax3.plot(actual_Y, actual_Z, 'rs', markersize=12, label='Found (Actual)')
        ax3.arrow(predicted_Y, predicted_Z, (actual_Y - predicted_Y), (actual_Z - predicted_Z),
                head_width=0.2, head_length=0.1, fc='orange', ec='orange', alpha=0.8, linewidth=2)

    ax3.set_xlabel('Stage Y (µm)')
    ax3.set_ylabel('Stage Z (µm)')
    ax3.set_title('Stage Coordinate Space')
    ax3.legend()

    # Plot 4: Text summary
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    if verification_result:
        pixel_offset = verification_result.get('pixel_offset', (0, 0))
        stage_offset_um = verification_result.get('stage_offset_um', (0, 0))
        error_um = verification_result.get('error_um', 0)
        confidence = verification_result.get('confidence', 0)
        status_line = '✅ EXCELLENT' if error_um < 0.2 else '✓ GOOD' if error_um < 1.0 else '⚠️ NEEDS IMPROVEMENT'
        
        # Calculate offsets for display
        predicted_to_design_offset = math.hypot(
            predicted_Y - design_stage_Y,
            predicted_Z - design_stage_Z
        )
        
        info_text = (
            f"Block center: ({block_center.u:.2f}, {block_center.v:.2f}) µm (global design)\n"
            f"Local pos:    ({grating_local.u:.2f}, {grating_local.v:.2f}) µm (block-relative)\n"
            f"Design → Stage (expected): ({design_stage_Y:.2f}, {design_stage_Z:.2f}) µm\n\n"
            f"PREDICTED STAGE: Y={predicted_Y:.3f} µm, Z={predicted_Z:.3f} µm\n"
            f"ACTUAL STAGE:    Y={actual_Y:.3f} µm, Z={actual_Z:.3f} µm\n\n"
            f"Prediction vs Design: {predicted_to_design_offset:.3f} µm\n"
            f"Pixel offset:    ({pixel_offset[0]:+.1f}, {pixel_offset[1]:+.1f}) px\n"
            f"Stage offset:    ({stage_offset_um[0]:+.3f}, {stage_offset_um[1]:+.3f}) µm\n"
            f"Total error:     {error_um:.3f} µm\n"
            f"Confidence:      {confidence:.3f}\n\n"
            f"Camera µm/px:    {camera.um_per_pixel:.4f}\n"
            f"STATUS: {status_line}\n"
        )
    else:
        info_text = "VERIFICATION FAILED - target not detected in image."

    ax4.text(0.01, 0.5, info_text, fontsize=10, family='monospace', verticalalignment='center')

    plt.suptitle(f'Stage 3 Verification: Block {target_block}, WG{waveguide_num}, {position_type}', fontsize=14)
    out_name = f"stage3_verification_block{target_block}_wg{waveguide_num}_{position_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved verification visualization: {out_name}")
    plt.show()


def main():
    # Paths
    layout_path = "config/mock_layout.json"
    
    # Initialize stage (internal stage uses nm units; adapter exposes µm)
    stage_nm = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    stage = StageAdapterUM(stage_nm)

    # Initialize camera (MockCamera expects layout path + stage reference)
    camera = MockCamera(layout_config_path=layout_path, stage_ref=stage)
    camera.connect()
    camera.set_exposure_time(0.02)

    vt = VisionTools()
    print("✅ Hardware & camera initialized (µm units)")

    # Load RuntimeLayout (for alignment - design only, no ground truth)
    runtime_layout = RuntimeLayout.from_json_file(layout_path)
    print(f"✅ RuntimeLayout loaded: {runtime_layout.design_name}")
    
    # Load CameraLayout (for validation only - has ground truth)
    camera_layout = CameraLayout.from_json_file(layout_path)
    print(f"✅ CameraLayout loaded (for validation): {camera_layout.design_name}")

    # Create HierarchicalAlignment with RuntimeLayout
    alignment = HierarchicalAlignment(runtime_layout)

    # Alignment searcher - uses the stage adapter (µm)
    searcher = AlignmentSearcher(stage, camera, vt)

    # Ground truth (from camera layout, for validation only)
    gt = camera_layout.ground_truth
    print(f"Ground truth (for validation only): rotation={gt.rotation_deg}°, "
          f"translation={gt.translation_um.to_tuple()} µm")

    # ----------------------------
    # STAGE 1: GLOBAL CALIBRATION
    # ----------------------------
    print("\n" + "="*70)
    print("STAGE 1: GLOBAL CALIBRATION")
    print("="*70)
    
    # Define which block–corner pairs to use
    pairs_to_use = [
        (1, 'top_left'),      # 1st block, top_left
        (20, 'bottom_right')  # 20th block, bottom_right
    ]
    global_measurements = []
    for block_id, corner in pairs_to_use:
        # Get block design position
        block = runtime_layout.get_block(block_id)
        block_center = block.design_position
        fid_local = block.get_fiducial(corner)
        block_size = runtime_layout.block_layout.block_size
        
        # Convert to global design coordinates
        u_bl = block_center.u - block_size / 2.0
        v_bl = block_center.v - block_size / 2.0
        u_global = u_bl + fid_local.u
        v_global = v_bl + fid_local.v

        print(f"\nSearching Block {block_id} {corner}, design pos ({u_global:.1f}, {v_global:.1f}) µm")

        result = searcher.search_for_fiducial(
            center_y_um=u_global,
            center_z_um=v_global,
            search_radius_um=60.0,
            step_um=20.0,
            label=f"Global corner B{block_id} {corner}",
            plot_progress=True
        )

        if not result:
            print(f"❌ Failed to find fiducial for Block {block_id} {corner}")
            return 1

        print(f"  Found at ({result['stage_Y']:.3f}, {result['stage_Z']:.3f}) µm "
                f"(confidence {result.get('confidence', 0):.3f})")
        
        global_measurements.append({
            'block_id': block_id,
            'corner': corner,
            'stage_Y': result['stage_Y'],
            'stage_Z': result['stage_Z'],
            'confidence': result.get('confidence', 0),
            'verification_error_um': result.get('verification_error_um', 0)
        })

    # Validate we found required corners
    if len(global_measurements) < 2:
        print("❌ Not enough corner fiducials found. Aborting.")
        return 1

    # Calibrate global transform
    print("\nCalibrating global transform from measured corner fiducials...")
    global_result = alignment.calibrate_global(global_measurements)
    print("Global calibration result:", global_result)

    # Validation vs ground truth
    gt_rot = gt.rotation_deg
    gt_trans = gt.translation_um.to_tuple()
    cal_rot = global_result['rotation_deg']
    cal_trans = global_result['translation_um']
    rot_err = abs(cal_rot - gt_rot)
    trans_err = math.hypot(cal_trans[0] - gt_trans[0], cal_trans[1] - gt_trans[1])
    print(f"\n✅ Validation vs GT:")
    print(f"  Rotation error: {rot_err:.6f}°")
    print(f"  Translation error: {trans_err:.6f} µm")
    print(f"  Fit mean residual: {global_result['mean_error_um']:.6f} µm")

    # ----------------------------
    # STAGE 2: BLOCK CALIBRATION
    # ----------------------------
    print("\n" + "="*70)
    print("STAGE 2: BLOCK-SPECIFIC CALIBRATION")
    print("="*70)
    
    target_block = 10
    print(f"Predicting block {target_block} fiducials using global calibration...")

    # Predict and search for block fiducials
    block_corners = ['top_left', 'bottom_right']
    block_measurements = []

    for corner in block_corners:
        # Predict fiducial position using alignment
        pred_Y, pred_Z = alignment.get_fiducial_stage_position(target_block, corner)
        print(f"Predicted {corner}: ({pred_Y:.3f}, {pred_Z:.3f}) µm - searching ±60 µm")

        res = searcher.search_for_fiducial(
            center_y_um=pred_Y,
            center_z_um=pred_Z,
            search_radius_um=60.0,
            step_um=15.0,
            label=f"Block {target_block} {corner}",
            plot_progress=True
        )

        if not res:
            print(f"  ❌ Search failed for block {target_block} corner {corner}")
            continue

        pred_err = math.hypot(res['stage_Y'] - pred_Y, res['stage_Z'] - pred_Z)
        print(f"  ✅ Found at ({res['stage_Y']:.3f}, {res['stage_Z']:.3f}) µm - "
              f"prediction error {pred_err:.3f} µm")
        
        block_measurements.append({
            'corner': corner,
            'stage_Y': res['stage_Y'],
            'stage_Z': res['stage_Z'],
            'confidence': res.get('confidence', 0),
            'prediction_error_um': pred_err,
            'verification_error_um': res.get('verification_error_um', 0)
        })
    if len(block_measurements) < 2:
        print("❌ Not enough block fiducials found for block calibration. Aborting.")
        return 1

    # ============================================================================
    # DIRECT CALCULATION TEST (no existing functions)
    # ============================================================================
    print("\n" + "="*70)
    print("DIRECT CALCULATION TEST - Stage 2 Block Calibration")
    print("="*70)
    
    # Extract measured stage positions (in µm)
    meas_tl_Y = block_measurements[0]['stage_Y']  # top_left Y
    meas_tl_Z = block_measurements[0]['stage_Z']  # top_left Z
    meas_br_Y = block_measurements[1]['stage_Y']  # bottom_right Y
    meas_br_Z = block_measurements[1]['stage_Z']  # bottom_right Z
    
    print(f"Measured fiducials in stage coords (µm):")
    print(f"  top_left:     ({meas_tl_Y:.3f}, {meas_tl_Z:.3f})")
    print(f"  bottom_right: ({meas_br_Y:.3f}, {meas_br_Z:.3f})")
    
    # Get design positions in block-local coords (µm)
    block_obj = runtime_layout.get_block(target_block)
    fid_tl_design = block_obj.get_fiducial('top_left')
    fid_br_design = block_obj.get_fiducial('bottom_right')
    
    design_tl_u = fid_tl_design.u  # block-local u
    design_tl_v = fid_tl_design.v  # block-local v
    design_br_u = fid_br_design.u  # block-local u
    design_br_v = fid_br_design.v  # block-local v
    
    print(f"\nDesign fiducials in block-local coords (µm):")
    print(f"  top_left:     ({design_tl_u:.3f}, {design_tl_v:.3f})")
    print(f"  bottom_right: ({design_br_u:.3f}, {design_br_v:.3f})")
    
    # Calculate block center from measured stage positions
    block_center_stage_Y = (meas_tl_Y + meas_br_Y) / 2.0
    block_center_stage_Z = (meas_tl_Z + meas_br_Z) / 2.0
    
    print(f"\nCalculated block center in stage coords (µm):")
    print(f"  Center: ({block_center_stage_Y:.3f}, {block_center_stage_Z:.3f})")
    
    # Calculate block angle from measured positions
    # Vector from top_left to bottom_right in stage coords
    vec_stage_Y = meas_br_Y - meas_tl_Y
    vec_stage_Z = meas_br_Z - meas_tl_Z
    
    # Vector from top_left to bottom_right in design coords
    vec_design_u = design_br_u - design_tl_u
    vec_design_v = design_br_v - design_tl_v
    
    # Angle in stage frame
    angle_stage_rad = math.atan2(vec_stage_Z, vec_stage_Y)
    angle_stage_deg = math.degrees(angle_stage_rad)
    
    # Angle in design frame
    angle_design_rad = math.atan2(vec_design_v, vec_design_u)
    angle_design_deg = math.degrees(angle_design_rad)
    
    # Block rotation = difference
    block_rotation_deg = angle_stage_deg - angle_design_deg
    block_rotation_rad = math.radians(block_rotation_deg)
    
    print(f"\nCalculated block rotation:")
    print(f"  Stage vector angle:  {angle_stage_deg:.6f}°")
    print(f"  Design vector angle: {angle_design_deg:.6f}°")
    print(f"  Block rotation:      {block_rotation_deg:.6f}°")
    
    # Now calculate block origin (bottom-left corner in stage coords)
    # Center in block-local coords
    block_size_um = runtime_layout.block_layout.block_size
    center_local_u = block_size_um / 2.0
    center_local_v = block_size_um / 2.0
    
    print(f"\nBlock size: {block_size_um:.3f} µm")
    print(f"Block center in local coords: ({center_local_u:.3f}, {center_local_v:.3f})")
    
    # Transform: stage = origin + R * local
    # So: origin = stage - R * local
    cos_theta = math.cos(block_rotation_rad)
    sin_theta = math.sin(block_rotation_rad)
    
    # Reverse transform to get origin
    block_origin_Y = block_center_stage_Y - (cos_theta * center_local_u - sin_theta * center_local_v)
    block_origin_Z = block_center_stage_Z - (sin_theta * center_local_u + cos_theta * center_local_v)
    
    print(f"\nCalculated block origin (bottom-left) in stage coords:")
    print(f"  Origin: ({block_origin_Y:.3f}, {block_origin_Z:.3f})")
    
    # Verification: Transform top_left design back to stage and compare
    verify_tl_stage_Y = block_origin_Y + (cos_theta * design_tl_u - sin_theta * design_tl_v)
    verify_tl_stage_Z = block_origin_Z + (sin_theta * design_tl_u + cos_theta * design_tl_v)
    
    verify_br_stage_Y = block_origin_Y + (cos_theta * design_br_u - sin_theta * design_br_v)
    verify_br_stage_Z = block_origin_Z + (sin_theta * design_br_u + cos_theta * design_br_v)
    
    error_tl = math.hypot(verify_tl_stage_Y - meas_tl_Y, verify_tl_stage_Z - meas_tl_Z)
    error_br = math.hypot(verify_br_stage_Y - meas_br_Y, verify_br_stage_Z - meas_br_Z)
    
    print(f"\n✅ Verification - Transform design back to stage:")
    print(f"  top_left predicted:  ({verify_tl_stage_Y:.3f}, {verify_tl_stage_Z:.3f})")
    print(f"  top_left measured:   ({meas_tl_Y:.3f}, {meas_tl_Z:.3f})")
    print(f"  Error: {error_tl:.6f} µm")
    print(f"  bottom_right predicted: ({verify_br_stage_Y:.3f}, {verify_br_stage_Z:.3f})")
    print(f"  bottom_right measured:  ({meas_br_Y:.3f}, {meas_br_Z:.3f})")
    print(f"  Error: {error_br:.6f} µm")
    
    # Now predict grating position using direct transform
    print(f"\n" + "="*70)
    print(f"DIRECT GRATING PREDICTION")
    print("="*70)
    
    # Get grating design position in block-local coords
    grating_obj = block_obj.get_grating(waveguide=25, side='left')
    grating_local_u = grating_obj.position.u
    grating_local_v = grating_obj.position.v
    
    print(f"Grating WG{25} left in block-local coords:")
    print(f"  ({grating_local_u:.3f}, {grating_local_v:.3f}) µm")
    
    # Transform to stage using our direct block transform
    grating_stage_Y_direct = block_origin_Y + (cos_theta * grating_local_u - sin_theta * grating_local_v)
    grating_stage_Z_direct = block_origin_Z + (sin_theta * grating_local_u + cos_theta * grating_local_v)
    
    print(f"\nDirect calculation predicted stage position:")
    print(f"  Y = {grating_stage_Y_direct:.3f} µm")
    print(f"  Z = {grating_stage_Z_direct:.3f} µm")
    
        # ============================================================================
    # CAPTURE IMAGE AT PREDICTED POSITION
    # ============================================================================
    print(f"\n📸 Moving stage to predicted position and capturing image...")
    stage.move_abs('y', grating_stage_Y_direct)
    stage.move_abs('z', grating_stage_Z_direct)
    
    # Acquire image at predicted position
    img_at_predicted = camera.acquire_single_image()
    
    # Create visualization with red cross at center
    fig_direct = plt.figure(figsize=(12, 10))
    img_norm = np.clip(img_at_predicted.astype(np.float32) / float(img_at_predicted.max() if img_at_predicted.max() > 0 else 1.0), 0, 1)
    
    plt.imshow(img_norm, cmap='gray', vmin=0, vmax=1, origin='lower')
    
    # Red cross at image center (where we predict the grating should be)
    img_center_x = img_at_predicted.shape[1] // 2
    img_center_y = img_at_predicted.shape[0] // 2
    plt.plot(img_center_x, img_center_y, 'r+', markersize=50, markeredgewidth=5, 
             label='Predicted Position (Direct Calc)', zorder=10)
    
    plt.title(f'Direct Calculation Result\nBlock {target_block} WG{25} Left Grating\n' + 
              f'Stage: Y={grating_stage_Y_direct:.3f} µm, Z={grating_stage_Z_direct:.3f} µm', 
              fontsize=14, fontweight='bold')
    plt.xlabel('X (pixels)', fontsize=12)
    plt.ylabel('Y (pixels)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Intensity (normalized)', fraction=0.046, pad=0.04)
    
    direct_plot_name = f"direct_calc_result_block{target_block}_wg{25}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(direct_plot_name, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved direct calculation visualization: {direct_plot_name}")
    plt.show()
    
    print(f"\n✅ Image captured at predicted position.")
    
    # ============================================================================
    # END DIRECT CALCULATION TEST
    # ============================================================================

    # Perform block-level calibration (old method for comparison)
    print(f"\n{'='*70}")
    print("OLD METHOD - Block Calibration (for comparison)")
    print(f"{'='*70}")
    block_result = alignment.calibrate_block(target_block, block_measurements)
    print("Block calibration result:", block_result)



if __name__ == "__main__":
    # Ensure repository root on path
    base = Path(__file__).resolve().parent.parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as exc:
        print(f"\n❌ Fatal error during test run: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)