#!/usr/bin/env python3
# test_hierarchical_alignment_with_search.py
"""
Complete test of hierarchical alignment using real AlignmentSearcher.

This demonstrates the full workflow:
1. Stage 1: Grid search for corner block fiducials → Global calibration
2. Stage 2: Use global calibration to predict → Grid search for block fiducials → Block calibration
3. Verification: Navigate to targets and verify accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Hardware
from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
from HardwareControl.CameraControl.mock_camera import MockCamera

# Alignment system
from config.layout_config_generator_v2 import load_layout_config_v2
from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.hierarchicalAlignment import HierarchicalAlignment
from AlignmentSystem.alignmentSearch import AlignmentSearcher


def main():
    """
    Full hierarchical alignment workflow with real search.
    """
    print("\n" + "="*70)
    print("HIERARCHICAL ALIGNMENT - FULL SEARCH WORKFLOW TEST")
    print("="*70)
    print("\nThis test demonstrates:")
    print("  1. Stage 1: Grid search for global calibration (4 corner blocks)")
    print("  2. Stage 2: Predicted search for block calibration (Block 10)")
    print("  3. Verification: Navigate to waveguide targets")
    print()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n" + "="*70)
    print("SETUP: Initialize Hardware & Alignment System")
    print("="*70)
    
    layout_config = "config/mock_layout.json"
    layout = load_layout_config_v2(layout_config)
    print(f"✅ Layout loaded: {layout['design_name']}")
    
    # Create mock hardware
    stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    camera = MockCamera(layout_config, stage_ref=stage)
    camera.connect()
    stage.set_camera_observer(camera)
    camera.set_exposure_time(0.02)
    vt = VisionTools()
    
    print(f"✅ Mock hardware initialized")
    print(f"   Camera FOV: {camera.sensor_width * camera.nm_per_pixel / 1000:.1f} µm")
    print(f"   Resolution: {camera.nm_per_pixel} nm/pixel")
    
    # Create hierarchical alignment and searcher
    alignment = HierarchicalAlignment(layout)
    searcher = AlignmentSearcher(stage, camera, vt)
    
    print(f"✅ Alignment system and searcher created")
    
    # Ground truth (for validation only)
    gt = layout['simulation_ground_truth']
    print(f"   Ground truth: rotation={gt['rotation_deg']}°, translation={gt['translation_nm']} nm")
    
    alignment.print_status()
    
    # =========================================================================
    # STAGE 1: GLOBAL CALIBRATION
    # =========================================================================
    input("\nPress Enter to start Stage 1: Global Calibration with grid search...")
    
    print("\n" + "="*70)
    print("STAGE 1: GLOBAL CALIBRATION (Grid Search)")
    print("="*70)
    print("Searching for fiducials in 4 corner blocks: 1, 5, 16, 20")
    print("Using coarse grid search around design positions")
    
    # Corner blocks and which corners to use
    corner_blocks = [1,20]
    corners_to_use = ['top_left',  'bottom_right']
    
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
        u_global = block_center[0] + (fiducial_local[0] - block_size / 2.0)
        v_global = block_center[1] + (fiducial_local[1] - block_size / 2.0)
        
        print(f"  Design position: ({u_global:.1f}, {v_global:.1f}) µm")
        
        # Initial guess: design position (no transformation yet)
        center_y_nm = u_global * 1000.0
        center_z_nm = v_global * 1000.0
        
        # Grid search (larger radius since we don't know transformation yet)
        result = searcher.search_for_fiducial(
            center_y_nm=center_y_nm,
            center_z_nm=center_z_nm,
            search_radius_nm=100000,  # ±100 µm search
            step_nm=20000,  # 20 µm steps
            label=f"Block {block_id} {corner}",
            plot_progress=True
        )
        
        if result:
            global_measurements.append({
                'block_id': block_id,
                'corner': corner,
                'stage_Y': result['stage_Y'],
                'stage_Z': result['stage_Z'],
                'confidence': result['confidence'],
                'verification_error_nm': result.get('verification_error_nm', 0)
            })
            print(f"  ✅ Found at ({result['stage_Y']}, {result['stage_Z']}) nm")
            print(f"     Verification error: {result.get('verification_error_nm', 0):.1f} nm")
        else:
            print(f"  ❌ Search failed!")
            print(f"\n⚠️  Cannot continue without all corner fiducials")
            return 1
    
    # Check we found all 4
    if len(global_measurements) < len(corner_blocks):
        print(f"\n❌ Only found {len(global_measurements)}/4 corner fiducials!")
        print("Cannot proceed with global calibration")
        return 1
    
    print(f"\n{'─'*70}")
    print(f"✅ Found all {len(global_measurements)} corner fiducials!")
    print('─'*70)
    
    # Calibrate global transformation
    print(f"\nCalibrating global transformation...")
    global_result = alignment.calibrate_global(global_measurements)
    
    # Validate
    validation = alignment.validate_global_calibration(
        gt['rotation_deg'],
        tuple(gt['translation_nm'])
    )
    
    print(f"\n{'─'*70}")
    print("Global Calibration Validation")
    print('─'*70)
    print(f"Ground truth: rotation={gt['rotation_deg']}°, translation={gt['translation_nm']} nm")
    print(f"Calibrated:   rotation={global_result['angle_deg']:.4f}°, "
          f"translation=({global_result['translation_nm'][0]:.1f}, {global_result['translation_nm'][1]:.1f}) nm")
    print(f"\nErrors:")
    print(f"  Rotation error:    {validation['rotation_error_deg']:.4f}°")
    print(f"  Translation error: {validation['translation_error_nm']:.1f} nm")
    print(f"  Mean residual:     {validation['mean_residual_nm']:.3f} nm")
    print(f"  Max residual:      {validation['max_residual_nm']:.3f} nm")
    
    if validation['rotation_error_deg'] < 0.1 and validation['translation_error_nm'] < 1000:
        print(f"\n✅ STAGE 1 PASSED - Global calibration accurate!")
    else:
        print(f"\n⚠️ STAGE 1 WARNING - Calibration errors larger than expected")
    
    alignment.print_status()
    
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
    
    # Predict block center
    pred_center_Y, pred_center_Z = alignment.predict_block_center(target_block)
    block_center_design = layout['blocks'][target_block]['design_position']
    
    print(f"\n{'─'*70}")
    print(f"Block {target_block} Prediction")
    print('─'*70)
    print(f"  Design center: ({block_center_design[0]:.1f}, {block_center_design[1]:.1f}) µm")
    print(f"  Predicted stage: ({pred_center_Y:.0f}, {pred_center_Z:.0f}) nm")
    
    # Search for all 4 corners using predicted positions
    block_corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    print(f"\n{'─'*70}")
    print(f"Searching for fiducials in Block {target_block}")
    print('─'*70)
    
    block_measurements = []
    
    for corner in block_corners:
        print(f"\n   Searching for {corner}...")
        
        # Use alignment system to predict position
        pred_Y, pred_Z = alignment.predict_fiducial_position(target_block, corner)
        print(f"   Predicted: ({pred_Y:.0f}, {pred_Z:.0f}) nm")
        
        # Fine grid search around predicted position
        result = searcher.search_for_fiducial(
            center_y_nm=pred_Y,
            center_z_nm=pred_Z,
            search_radius_nm=60000,  # ±30 µm (smaller now that we have prediction)
            step_nm=15000,  # 5 µm steps (finer grid)
            label=f"Block {target_block} {corner}",
            plot_progress=True
        )
        
        if result:
            # Calculate prediction error
            pred_error = np.hypot(result['stage_Y'] - pred_Y, 
                                 result['stage_Z'] - pred_Z)
            
            print(f"   ✅ Found at ({result['stage_Y']}, {result['stage_Z']}) nm")
            print(f"      Prediction error: {pred_error:.0f} nm")
            print(f"      Verification error: {result.get('verification_error_nm', 0):.1f} nm")
            
            block_measurements.append({
                'corner': corner,
                'stage_Y': result['stage_Y'],
                'stage_Z': result['stage_Z'],
                'confidence': result['confidence'],
                'prediction_error_nm': pred_error,
                'verification_error_nm': result.get('verification_error_nm', 0)
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
    pred_errors = [m['prediction_error_nm'] for m in block_measurements]
    print(f"\nPrediction accuracy using Stage 1 calibration:")
    print(f"  Mean prediction error: {np.mean(pred_errors):.1f} nm")
    print(f"  Max prediction error:  {np.max(pred_errors):.1f} nm")
    
    # Calibrate block
    print(f"\nCalibrating Block {target_block}...")
    block_result = alignment.calibrate_block(target_block, block_measurements)
    
    # Validate
    validation_block = alignment.validate_block_calibration(
        target_block,
        gt['rotation_deg'],
        tuple(gt['translation_nm'])
    )
    
    print(f"\n{'─'*70}")
    print(f"Block {target_block} Calibration Validation")
    print('─'*70)
    print("Note: In simulation, blocks have no fabrication errors,")
    print("so block calibration should closely match global calibration.")
    print(f"\nErrors:")
    print(f"  Rotation error:    {validation_block['rotation_error_deg']:.4f}°")
    print(f"  Translation error: {validation_block['translation_error_nm']:.1f} nm")
    print(f"  Mean residual:     {validation_block['mean_residual_nm']:.3f} nm")
    
    if validation_block['mean_residual_nm'] < 200:
        print(f"\n✅ STAGE 2 PASSED - Block calibration accurate!")
    else:
        print(f"\n⚠️ STAGE 2 WARNING - Block calibration residuals larger than expected")
    
    alignment.print_status()
    
    # =========================================================================
    # STAGE 3: VERIFICATION - NAVIGATE TO TARGETS
    # =========================================================================
    input("\nPress Enter to verify positioning accuracy...")
    
    print("\n" + "="*70)
    print("STAGE 3: POSITIONING VERIFICATION")
    print("="*70)
    print(f"Testing coordinate conversion and positioning accuracy")
    print(f"Target: Block {target_block}, Waveguide 25, Left Grating")
    
    # Get predicted position using block calibration
    Y_pred, Z_pred = alignment.get_waveguide_position(target_block, 25, 'left_grating')
    
    print(f"\nPredicted stage position (using Block {target_block} calibration):")
    print(f"  Y = {Y_pred:.0f} nm")
    print(f"  Z = {Z_pred:.0f} nm")
    
    # Verify by navigating and checking with camera
    print(f"\nNavigating to target...")
    verification = searcher.verify_fiducial_centering(
        expected_y_nm=Y_pred,
        expected_z_nm=Z_pred,
        label=f"Block {target_block} WG25 Left Grating"
    )
    
    if verification:
        error = verification['error_nm']
        print(f"\n✅ Target verified!")
        print(f"   Positioning error: {error:.1f} nm ({error/1000:.3f} µm)")
        print(f"   Confidence: {verification['confidence']:.3f}")
        
        if error < 200:
            print(f"\n🎉 EXCELLENT ACCURACY! System ready for alignment.")
            success = True
        elif error < 1000:
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
    print("TEST SUMMARY")
    print("="*70)
    
    print(f"\n✅ Stage 1: Global calibration complete")
    print(f"   - Found {len(global_measurements)}/4 corner fiducials")
    print(f"   - Rotation error: {validation['rotation_error_deg']:.4f}°")
    print(f"   - Translation error: {validation['translation_error_nm']:.1f} nm")
    
    print(f"\n✅ Stage 2: Block {target_block} calibration complete")
    print(f"   - Found {len(block_measurements)}/4 block fiducials")
    print(f"   - Mean prediction error: {np.mean(pred_errors):.1f} nm")
    print(f"   - Calibration residual: {validation_block['mean_residual_nm']:.3f} nm")
    
    if verification:
        print(f"\n✅ Stage 3: Positioning verification complete")
        print(f"   - Target positioning error: {verification['error_nm']:.1f} nm")
    
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
        return 0
    else:
        print("\n⚠️ Some stages had issues. Review results above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())