#!/usr/bin/env python3
# test_hierarchical_alignment_v3_um.py
"""
Full-mode µm-based integration test for HierarchicalAlignment.

Workflow:
  1) Stage 1 - Global calibration via grid search on corner fiducials
  2) Stage 2 - Direct block calibration using precise fiducial measurements
  3) Stage 3 - Verify grating target and produce visualization

Important:
  - All coordinates and searches operate in MICROMETERS (µm).
  - Uses DIRECT block transform for sub-micron accuracy
"""

from pathlib import Path
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

# Project imports
from hardware_control.setup_motor.mock_stage import MockXYZStage
from hardware_control.setup_motor.stage_adapter import StageAdapterUM
from hardware_control.camera_control.mock_camera import MockCamera
from alignment_system.hierarchical_alignment import HierarchicalAlignment
from alignment_system.alignment_search import AlignmentSearcher
from config.layout_models import RuntimeLayout, CameraLayout
from alignment_system.cv_tools import VisionTools


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
    print("STAGE 2: BLOCK-SPECIFIC CALIBRATION (DIRECT METHOD)")
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

    # Perform block-level calibration using NEW direct method
    print(f"\nCalibrating block transform for block {target_block} (DIRECT METHOD)")
    block_result = alignment.calibrate_block(target_block, block_measurements)
    print("\n📊 Block calibration result:")
    print(f"  Rotation: {block_result['rotation_deg']:.6f}°")
    print(f"  Origin: {block_result['origin_stage_um']}")
    print(f"  Mean error: {block_result['mean_error_um']:.6f} µm")
    print(f"  Max error: {block_result['max_error_um']:.6f} µm")

    # Validation vs ground truth (block fabrication error)
    gt_block_err = camera_layout.ground_truth.get_block_error(target_block)
    block_rot_err = abs(block_result['rotation_deg'] - gt_block_err.rotation_deg)
    
    # Note: For origin validation, we need to compute expected origin from ground truth
    # This is complex - for now just show the values
    print(f"\n✅ Block validation vs GT fabrication error:")
    print(f"  GT block rotation: {gt_block_err.rotation_deg:.6f}°")
    print(f"  Measured rotation: {block_result['rotation_deg']:.6f}°")
    print(f"  Rotation error: {block_rot_err:.6f}°")
    print(f"  Fit mean residual: {block_result['mean_error_um']:.6f} µm")

    # ----------------------------
    # STAGE 3: VERIFICATION
    # ----------------------------
    print("\n" + "="*70)
    print("STAGE 3: POSITIONING VERIFICATION")
    print("="*70)
    
    waveguide_num = 25
    print(f"\nPredicting grating position for Block {target_block} WG{waveguide_num} left")
    print(f"Using NEW direct block transform method...")

    # Use the NEW get_grating_stage_position (uses direct transform)
    Y_pred, Z_pred = alignment.get_grating_stage_position(target_block, waveguide_num, 'left')
    print(f"\nPredicted stage position:")
    print(f"  Y = {Y_pred:.3f} µm")
    print(f"  Z = {Z_pred:.3f} µm")
    
    # Move to predicted position and capture image
    print(f"\n📸 Moving stage to predicted position and capturing image...")
    stage.move_abs('y', Y_pred)
    stage.move_abs('z', Z_pred)
    
    img_at_predicted = camera.acquire_single_image()
    
    # Create visualization with red cross at center
    fig_verification = plt.figure(figsize=(12, 10))
    img_norm = np.clip(img_at_predicted.astype(np.float32) / float(img_at_predicted.max() if img_at_predicted.max() > 0 else 1.0), 0, 1)
    
    plt.imshow(img_norm, cmap='gray', vmin=0, vmax=1, origin='lower')
    
    # Red cross at image center (where we predict the grating should be)
    img_center_x = img_at_predicted.shape[1] // 2
    img_center_y = img_at_predicted.shape[0] // 2
    plt.plot(img_center_x, img_center_y, 'r+', markersize=50, markeredgewidth=5, 
             label='Predicted Grating Position', zorder=10)
    
    plt.title(f'Stage 3 Verification (Direct Transform Method)\nBlock {target_block} WG{waveguide_num} Left Grating\n' + 
              f'Stage: Y={Y_pred:.3f} µm, Z={Z_pred:.3f} µm', 
              fontsize=14, fontweight='bold')
    plt.xlabel('X (pixels)', fontsize=12)
    plt.ylabel('Y (pixels)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Intensity (normalized)', fraction=0.046, pad=0.04)
    
    verification_plot_name = f"stage3_verification_block{target_block}_wg{waveguide_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(verification_plot_name, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved verification visualization: {verification_plot_name}")
    plt.show()
    
    print(f"\n✅ Image captured at predicted position.")
    print(f"   Visual inspection: Is the grating centered at the red cross?")
    success = True
    
    # Get grating design position in block-local coords
    block = runtime_layout.get_block(target_block)
    grating = block.get_grating(waveguide_num, 'left')
    grating_u = grating.position.u
    grating_v = grating.position.v

    # Create a transform using CameraLayout (has ground truth)
    from alignment_system.coordinate_transform_v3 import CoordinateTransformV3
    gt_transform = CoordinateTransformV3(camera_layout)
    gt_transform.use_ground_truth()

    # Get TRUE grating position (using ground truth transform)
    gt_Y, gt_Z = gt_transform.block_local_to_stage(target_block, grating_u, grating_v)

    # Calculate prediction error
    pos_error_um = math.hypot(Y_pred - gt_Y, Z_pred - gt_Z)
    # Calculate error
    pos_error_um = math.hypot(Y_pred - gt_Y, Z_pred - gt_Z)
    print(f"\n✅ Verification position error vs GT:")
    print(f"  GT stage position: Y={gt_Y:.3f} µm, Z={gt_Z:.3f} µm")
    print(f"  Measured position: Y={Y_pred:.3f} µm, Z={Z_pred:.3f} µm")
    print(f"  Total position error: {pos_error_um:.6f} µm")
    if pos_error_um > 5.0:
        print("❌ Position error exceeds 5 µm threshold. Verification failed.")
        success = False
    else:
        print("✅ Position error within acceptable range. Verification passed.")
    
    # Save calibration results
    output_path = f"results/test_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    alignment.save_calibration(output_path)

    return 0 if success else 2


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