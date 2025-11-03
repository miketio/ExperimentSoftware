#!/usr/bin/env python3
# test_mock_camera_system.py
"""
Test script for MockCamera and MockStage integration.
Tests the complete mock hardware system with camera-stage coordination.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Import mock components
from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
from HardwareControl.CameraControl.mock_camera import MockCamera
from config.layout_config_generator_v2 import generate_layout_config_v2, load_layout_config_v2
from AlignmentSystem.coordinate_utils import CoordinateConverter


def test_1_generate_layout():
    """Test 1: Generate layout configuration."""
    print("\n" + "="*70)
    print("TEST 1: Generate Layout Configuration")
    print("="*70)
    
    ascii_file = "./AlignmentSystem/ascii_sample.ASC"
    output_file = "config/mock_layout.json"
    
    if not Path(ascii_file).exists():
        print(f"❌ ASCII file not found: {ascii_file}")
        return False
    
    try:
        layout = generate_layout_config_v2(
            ascii_file,
            output_file,
            simulated_rotation=0.0,
            simulated_translation=(10000, 0)
        )
        
        print(f"\n✅ Layout generated successfully!")
        print(f"   File: {output_file}")
        print(f"   Blocks: {layout['block_layout']['total_blocks']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_stage_camera_integration():
    """Test 2: Stage-Camera integration."""
    print("\n" + "="*70)
    print("TEST 2: Stage-Camera Integration")
    print("="*70)
    
    try:
        # Create mock stage
        stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
        print("✅ MockStage created")
        
        # Create mock camera with stage reference
        camera = MockCamera("config/mock_layout.json", stage_ref=stage)
        camera.connect()
        print("✅ MockCamera created and connected to stage")
        
        # Register camera as observer
        stage.set_camera_observer(camera)
        print("✅ Camera registered as stage observer")
        
        # Move stage and capture images
        positions = [(0, 0, 0), (100000, 50000, 0), (200000, 100000, 0)]
        
        for x, y, z in positions:
            print(f"\n📍 Moving stage to ({x}, {y}, {z}) nm...")
            stage.set_pos_all(x, y, z)
            
            # Camera automatically uses new position
            img = camera.acquire_single_image()
            print(f"   Image captured: {img.shape}, max={img.max()}")
        
        # Check position history
        history = stage.get_position_history()
        print(f"\n✅ Stage position history: {len(history)} moves recorded")
        
        camera.disconnect()
        stage.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_capture_fiducial_with_stage():
    """Test 3: Navigate to fiducial using stage, capture with camera."""
    print("\n" + "="*70)
    print("TEST 3: Navigate to Fiducial")
    print("="*70)
    
    try:
        layout = load_layout_config_v2("config/mock_layout.json")
        converter = CoordinateConverter(layout)
        gt = layout['simulation_ground_truth']
        converter.set_transformation(gt['rotation_deg'], tuple(gt['translation_nm']))
        
        # Get Block 1 top-left fiducial position
        tl_stage = converter.get_fiducial_stage_position(1, 'top_left')
        
        print(f"📍 Target: Block 1 Top-Left Fiducial")
        print(f"   Stage position: ({tl_stage[0]}, {tl_stage[1]}) nm")
        
        # Create mock hardware
        stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
        camera = MockCamera("config/mock_layout.json", stage_ref=stage)
        camera.connect()
        stage.set_camera_observer(camera)
        
        # Navigate to fiducial
        print(f"\n🎯 Navigating to fiducial...")
        stage.move_abs('y', tl_stage[0])
        stage.move_abs('z', tl_stage[1])
        
        # Set camera parameters
        camera.set_exposure_time(0.02)
        
        # Capture image
        print(f"\n📷 Capturing image...")
        img = camera.acquire_single_image()
        
        print(f"✅ Image captured:")
        print(f"   Shape: {img.shape}")
        print(f"   Max intensity: {img.max()}")
        print(f"   Very bright pixels (>2500): {np.count_nonzero(img > 2500)}")
        
        # Find brightest spot
        max_loc = np.unravel_index(img.argmax(), img.shape)
        center = (img.shape[1]//2, img.shape[0]//2)
        distance = np.sqrt((max_loc[1] - center[0])**2 + (max_loc[0] - center[1])**2)
        
        print(f"   Brightest pixel at: {max_loc}")
        print(f"   Image center: {center}")
        print(f"   Distance from center: {distance:.1f} pixels")
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Full image
        ax1 = axes[0]
        ax1.imshow(img, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax1.plot(max_loc[1], max_loc[0], 'r+', markersize=20, markeredgewidth=3, label='Brightest')
        ax1.plot(center[0], center[1], 'g+', markersize=15, markeredgewidth=2, label='Center')
        ax1.set_title('Full Image')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zoomed view
        ax2 = axes[1]
        zoom_size = 200
        cy, cx = center[1], center[0]
        zoomed = img[max(0, cy-zoom_size):min(img.shape[0], cy+zoom_size),
                     max(0, cx-zoom_size):min(img.shape[1], cx+zoom_size)]
        ax2.imshow(zoomed, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax2.set_title('Zoomed Center (L-shaped fiducial)')
        ax2.grid(True, alpha=0.3)
        
        # Stage position history
        ax3 = axes[2]
        ax3.axis('off')
        
        history = stage.get_position_history()
        history_text = f"Stage Movement History:\n{'─'*40}\n"
        for i, entry in enumerate(history[-5:], 1):  # Last 5 moves
            history_text += f"{i}. {entry['axis'].upper()}: {entry['old_position']}nm → {entry['new_position']}nm\n"
        
        history_text += f"\nFinal Position:\n"
        history_text += f"Y = {stage.get_pos('y')} nm\n"
        history_text += f"Z = {stage.get_pos('z')} nm\n"
        history_text += f"X = {stage.get_pos('x')} nm\n"
        
        ax3.text(0.1, 0.5, history_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('test_stage_camera_fiducial.png', dpi=150)
        print(f"\n💾 Saved: test_stage_camera_fiducial.png")
        plt.show()
        
        camera.disconnect()
        stage.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_scan_multiple_positions():
    """Test 4: Scan multiple positions (simulating grid scan)."""
    print("\n" + "="*70)
    print("TEST 4: Multi-Position Scan")
    print("="*70)
    
    try:
        layout = load_layout_config_v2("config/mock_layout.json")
        converter = CoordinateConverter(layout)
        gt = layout['simulation_ground_truth']
        converter.set_transformation(gt['rotation_deg'], tuple(gt['translation_nm']))
        
        # Get positions of several fiducials
        targets = [
            (1, 'top_left', 'Block 1 TL'),
            (1, 'bottom_right', 'Block 1 BR'),
            (10, 'top_left', 'Block 10 TL'),
            (20, 'bottom_right', 'Block 20 BR')
        ]
        
        # Create hardware
        stage = MockXYZStage()
        camera = MockCamera("config/mock_layout.json", stage_ref=stage)
        camera.connect()
        stage.set_camera_observer(camera)
        camera.set_exposure_time(0.02)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()
        
        for idx, (block_id, corner, label) in enumerate(targets):
            pos = converter.get_fiducial_stage_position(block_id, corner)
            
            print(f"\n📍 {label}: ({pos[0]}, {pos[1]}) nm")
            stage.move_abs('y', pos[0])
            stage.move_abs('z', pos[1])
            
            img = camera.acquire_single_image()
            
            # Plot
            ax = axes[idx]
            zoom_size = 300
            center = (img.shape[1]//2, img.shape[0]//2)
            zoomed = img[center[1]-zoom_size:center[1]+zoom_size,
                        center[0]-zoom_size:center[0]+zoom_size]
            
            ax.imshow(zoomed, cmap='gray', vmin=0, vmax=3500, origin='lower')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_multi_position_scan.png', dpi=150)
        print(f"\n💾 Saved: test_multi_position_scan.png")
        plt.show()
        
        # Save stage history
        stage.save_position_history("test_scan_history.csv")
        print(f"💾 Saved: test_scan_history.csv")
        
        camera.disconnect()
        stage.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_autofocus_simulation():
    """Test 5: Simulate autofocus by scanning X axis."""
    print("\n" + "="*70)
    print("TEST 5: Autofocus Simulation (X-axis scan)")
    print("="*70)
    
    try:
        layout = load_layout_config_v2("config/mock_layout.json")
        converter = CoordinateConverter(layout)
        gt = layout['simulation_ground_truth']
        converter.set_transformation(gt['rotation_deg'], tuple(gt['translation_nm']))
        
        # Get WG25 position
        wg25_pos = converter.get_grating_stage_position(10, 25, 'left')
        
        print(f"📍 Target: Block 10 WG25 Left Grating")
        print(f"   Position: ({wg25_pos[0]}, {wg25_pos[1]}) nm")
        
        # Create hardware
        stage = MockXYZStage()
        camera = MockCamera("config/mock_layout.json", stage_ref=stage)
        camera.connect()
        stage.set_camera_observer(camera)
        camera.set_exposure_time(0.02)
        
        # Navigate to WG25
        stage.move_abs('y', wg25_pos[0])
        stage.move_abs('z', wg25_pos[1])
        
        # Scan X axis (focus axis)
        X_positions = np.linspace(-10000, 10000, 21)  # -10µm to +10µm
        focus_metrics = []
        
        print(f"\n🔍 Scanning X axis for focus...")
        for X_nm in X_positions:
            stage.move_abs('x', int(X_nm))
            img = camera.acquire_single_image()
            
            # Calculate focus metric (variance of Laplacian)
            laplacian = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F)
            focus_metric = laplacian.var()
            focus_metrics.append(focus_metric)
            
            print(f"   X={X_nm:6.0f}nm → Focus={focus_metric:8.1f}")
        
        # Find best focus
        best_idx = np.argmax(focus_metrics)
        best_X = X_positions[best_idx]
        best_metric = focus_metrics[best_idx]
        
        print(f"\n✅ Best focus found:")
        print(f"   X = {best_X:.0f} nm")
        print(f"   Focus metric = {best_metric:.1f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Focus curve
        ax1.plot(X_positions/1000, focus_metrics, 'b-o', linewidth=2, markersize=6)
        ax1.axvline(x=0, color='r', linestyle='--', label='Optimal (X=0)')
        ax1.axvline(x=best_X/1000, color='g', linestyle='--', label=f'Found (X={best_X/1000:.1f}µm)')
        ax1.set_xlabel('X Position (µm)', fontsize=12)
        ax1.set_ylabel('Focus Metric (higher = sharper)', fontsize=12)
        ax1.set_title('Autofocus Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Images at different focus positions
        ax2.axis('off')
        
        # Show images at -10µm, 0µm, +10µm
        test_positions = [-10000, 0, 10000]
        images = []
        
        for X_nm in test_positions:
            stage.move_abs('x', int(X_nm))
            img = camera.acquire_single_image()
            center = (img.shape[1]//2, img.shape[0]//2)
            crop = img[center[1]-100:center[1]+100, center[0]-100:center[0]+100]
            images.append(crop)
        
        # Create composite image
        composite = np.hstack(images)
        
        ax2_img = fig.add_subplot(1, 2, 2)
        ax2_img.imshow(composite, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax2_img.set_title('Focus Comparison: X=-10µm, X=0µm, X=+10µm')
        ax2_img.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_autofocus_simulation.png', dpi=150)
        print(f"\n💾 Saved: test_autofocus_simulation.png")
        plt.show()
        
        camera.disconnect()
        stage.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MOCK CAMERA + STAGE INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Generate Layout", test_1_generate_layout),
        ("Stage-Camera Integration", test_2_stage_camera_integration),
        ("Navigate to Fiducial", test_3_capture_fiducial_with_stage),
        ("Multi-Position Scan", test_4_scan_multiple_positions),
        ("Autofocus Simulation", test_5_autofocus_simulation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'─'*70}")
        input(f"Press Enter to run: {name}...")
        result = test_func()
        results.append((name, result))
        
        if not result:
            print(f"\n❌ Test '{name}' failed!")
            cont = input("Continue with remaining tests? (y/n): ")
            if cont.lower() != 'y':
                break
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Mock hardware system is fully functional!")
        print("\n📁 Generated files:")
        print("   • config/mock_layout.json")
        print("   • test_stage_camera_fiducial.png")
        print("   • test_multi_position_scan.png")
        print("   • test_autofocus_simulation.png")
        print("   • test_scan_history.csv")


if __name__ == "__main__":
    main()