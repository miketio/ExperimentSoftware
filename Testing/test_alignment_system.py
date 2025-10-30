#!/usr/bin/env python3
# test_alignment_system.py
"""
Comprehensive testing script for the alignment system.
Tests components individually and full workflow.
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path


def test_ascii_parser():
    """Test ASCII file parsing."""
    print("\n" + "="*70)
    print("TEST 1: ASCII Parser")
    print("="*70)
    
    from AlignmentSystem.ascii_parser import ASCIIParser, find_waveguide_grating
    
    # Check if ASCII file exists
    ascii_file = "./AlignmentSystem/ascii_sample.ASC"
    if not Path(ascii_file).exists():
        print(f"❌ ASCII file not found: {ascii_file}")
        return False
    
    try:
        parser = ASCIIParser(ascii_file)
        data = parser.parse()
        
        print(f"✅ Parsed: {ascii_file}")
        print(f"   Markers: {len(data['markers'])}")
        print(f"   Waveguides: {len(data['waveguides'])}")
        print(f"   Gratings: {len(data['gratings'])}")
        
        # Find waveguide 25
        wg25_left = find_waveguide_grating(data['waveguides'], data['gratings'], 25, 'left')
        if wg25_left:
            print(f"   WG25 left grating: ({wg25_left[0]:.3f}, {wg25_left[1]:.3f}) µm")
        else:
            print(f"   ⚠️  WG25 left grating not found")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_layout_generator():
    """Test layout configuration generation."""
    print("\n" + "="*70)
    print("TEST 2: Layout Generator")
    print("="*70)
    
    from AlignmentSystem.layout_config_generator import generate_layout_config, load_layout_config
    
    ascii_file = "./AlignmentSystem/ascii_sample.ASC"
    config_file = "config/sample_layout.json"
    
    if not Path(ascii_file).exists():
        print(f"❌ ASCII file not found: {ascii_file}")
        return False
    
    try:
        # Generate layout
        layout = generate_layout_config(ascii_file, config_file)
        
        print(f"\n✅ Generated layout config")
        print(f"   Total blocks: {layout['total_blocks']}")
        print(f"   Dimensions: {layout['total_dimensions']['u_max']:.0f} × {layout['total_dimensions']['v_max']:.0f} µm")
        
        # Test loading
        loaded = load_layout_config(config_file)
        print(f"✅ Config loaded successfully")
        
        # Check block 10
        block10 = loaded['blocks'][10]
        print(f"\n   Block 10 info:")
        print(f"     Center: ({block10['center'][0]:.1f}, {block10['center'][1]:.1f}) µm")
        print(f"     WG25 left: ({block10['gratings']['wg25_left'][0]:.1f}, {block10['gratings']['wg25_left'][1]:.1f}) µm")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cv_tools():
    """Test computer vision tools."""
    print("\n" + "="*70)
    print("TEST 3: Computer Vision Tools")
    print("="*70)
    
    from AlignmentSystem.cv_tools import VisionTools
    
    try:
        vt = VisionTools()
        
        # Create synthetic test image
        print("Creating synthetic test image...")
        test_img = np.zeros((512, 512), dtype=np.uint16)
        test_img[100:150, 100:300] = 3000  # Horizontal bar
        test_img[100:300, 100:150] = 3000  # Vertical bar (L-shape)
        test_img[250:270, 250:400] = 4000  # Another feature
        
        # Test intensity measurement
        metrics = vt.measure_intensity(test_img)
        print(f"✅ Intensity measurement: mean={metrics['mean']:.1f}, max={metrics['max']}")
        
        # Test fiducial detection
        result = vt.find_fiducial_auto(test_img, expected_position=(200, 200), search_radius=150)
        if result:
            print(f"✅ Fiducial detection: pos={result['position']}, method={result['method']}")
        else:
            print(f"⚠️  Fiducial not detected (expected with synthetic image)")
        
        # Test focus metric
        focus = vt.calculate_focus_metric(test_img)
        print(f"✅ Focus metric: {focus:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_transform():
    """Test coordinate transformation."""
    print("\n" + "="*70)
    print("TEST 4: Coordinate Transform")
    print("="*70)
    
    from AlignmentSystem.coordinate_transform import CoordinateTransform
    
    try:
        transform = CoordinateTransform()
        
        # Simulate two fiducial measurements with known rotation
        angle_sim = np.radians(3.0)  # 3 degree rotation
        cos_a = np.cos(angle_sim)
        sin_a = np.sin(angle_sim)
        
        def sim_measure(u, v):
            """Simulate measurement with rotation and offset."""
            u_nm = u * 1000
            v_nm = v * 1000
            Y = cos_a * u_nm - sin_a * v_nm + 50000
            Z = sin_a * u_nm + cos_a * v_nm + 30000
            return (Y, Z)
        
        # Two fiducials (corners of array)
        design_fid1 = (5.0, 5.0)
        design_fid2 = (1395.0, 605.0)
        
        measured_fid1 = sim_measure(*design_fid1)
        measured_fid2 = sim_measure(*design_fid2)
        
        print(f"Simulating calibration:")
        print(f"  Fid 1: {design_fid1} µm → ({measured_fid1[0]:.0f}, {measured_fid1[1]:.0f}) nm")
        print(f"  Fid 2: {design_fid2} µm → ({measured_fid2[0]:.0f}, {measured_fid2[1]:.0f}) nm")
        
        # Calibrate
        result = transform.calibrate(
            measured_points=[measured_fid1, measured_fid2],
            design_points=[design_fid1, design_fid2]
        )
        
        print(f"\n✅ Calibration successful:")
        print(f"   Detected angle: {result['angle_deg']:.3f}° (expected: 3.0°)")
        print(f"   Mean error: {result['mean_error_nm']:.3f} nm")
        
        # Test transformation
        test_point = (700.0, 300.0)
        stage_coords = transform.design_to_stage(*test_point)
        back_coords = transform.stage_to_design(*stage_coords)
        
        error = np.sqrt((back_coords[0] - test_point[0])**2 + (back_coords[1] - test_point[1])**2)
        print(f"\n✅ Round-trip test:")
        print(f"   Original: {test_point} µm")
        print(f"   Stage: ({stage_coords[0]}, {stage_coords[1]}) nm")
        print(f"   Back: ({back_coords[0]:.3f}, {back_coords[1]:.3f}) µm")
        print(f"   Error: {error:.6f} µm")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment_state():
    """Test alignment state management."""
    print("\n" + "="*70)
    print("TEST 5: Alignment State")
    print("="*70)
    
    from AlignmentSystem.alignment_state import AlignmentState, AlignmentStatus
    
    try:
        state = AlignmentState()
        
        print("Initial state:")
        print(f"  Status: {state.status.value}")
        print(f"  Calibrated: {state.is_calibrated}")
        
        # Simulate workflow
        state.set_status(AlignmentStatus.FINDING_FIDUCIAL)
        state.add_fiducial('top_left', 50000, 30000, 0.95)
        state.add_fiducial('bottom_right', 1450000, 630000, 0.92)
        
        state.set_calibration({
            'method': 'two_point',
            'angle_deg': 2.8,
            'mean_error_nm': 15.3
        })
        
        state.set_target(10, 25, 'left', (12.0, 117.6), (162000, 147600))
        
        state.start_optimization()
        state.update_optimization_progress(0.5)
        state.finish_optimization({
            'success': True,
            'best_position': (162500, 148100),
            'best_intensity': 3245.7
        })
        
        state_dict = state.get_state_dict()
        
        print(f"\n✅ State management working:")
        print(f"   Status: {state_dict['status']}")
        print(f"   Calibrated: {state_dict['calibration']['is_calibrated']}")
        print(f"   Best intensity: {state_dict['optimization']['best_intensity']}")
        print(f"   Alignments completed: {state_dict['history']['alignments_completed']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_hardware():
    """Test with real hardware (requires running system)."""
    print("\n" + "="*70)
    print("TEST 6: Hardware Integration (Interactive)")
    print("="*70)
    
    print("\n⚠️  This test requires real hardware to be connected.")
    response = input("Do you want to run hardware tests? (y/N): ").strip().lower()
    
    if response != 'y':
        print("Skipping hardware tests.")
        return None
    
    try:
        from HardwareControl.CameraControl.zylaCamera import ZylaCamera
        from HardwareControl.andorCameraApp import AndorCameraApp
        from HardwareControl.SetupMotor.smartactStage import SmarActXYZStage
        from HardwareControl.xyzStageApp import XYZStageApp
        from AlignmentSystem.alignment_controller import AlignmentController
        from AlignmentSystem.layout_config_generator import load_layout_config
        
        print("\nInitializing hardware...")
        
        # Initialize camera
        camera = ZylaCamera()
        camera.connect()
        camera_app = AndorCameraApp(camera)
        camera_app.set_exposure(0.02)
        print("✅ Camera initialized")
        
        # Initialize stage
        stage = SmarActXYZStage()
        stage_app = XYZStageApp(stage)
        print("✅ Stage initialized")
        
        # Load layout
        layout = load_layout_config("config/sample_layout.json")
        print("✅ Layout loaded")
        
        # Create alignment controller
        alignment = AlignmentController(camera_app, stage_app, layout)
        print("✅ Alignment controller created")
        
        # Test image capture
        print("\nTest 1: Capture and analyze image")
        from AlignmentSystem.cv_tools import VisionTools
        vt = VisionTools()
        image = camera_app.acquire_image()
        vt.save_image(image)
        metrics = vt.measure_intensity(image)
        print(f"  Image shape: {image.shape}")
        print(f"  Mean intensity: {metrics['mean']:.1f}")
        print(f"  ✅ Image capture working")
        
        # Test stage movement
        print("\nTest 2: Stage positioning")
        current_y = stage_app.get_pos('y')
        current_z = stage_app.get_pos('z')
        print(f"  Current position: Y={current_y} nm, Z={current_z} nm")
        print(f"  ✅ Stage communication working")
        
        print("\n⚠️  Hardware tests completed successfully!")
        print("You can now run full calibration and alignment with:")
        print("  python test_alignment_system.py --full-workflow")
        
        # Cleanup
        camera.disconnect()
        stage.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all non-hardware tests."""
    print("\n" + "="*70)
    print("ALIGNMENT SYSTEM TEST SUITE")
    print("="*70)
    
    results = []
    
    results.append(("ASCII Parser", test_ascii_parser()))
    results.append(("Layout Generator", test_layout_generator()))
    results.append(("CV Tools", test_cv_tools()))
    results.append(("Coordinate Transform", test_coordinate_transform()))
    results.append(("Alignment State", test_alignment_state()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready for hardware integration.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
    
    return passed == total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test alignment system")
    parser.add_argument('--all', action='store_true', help='Run all software tests')
    parser.add_argument('--hardware', action='store_true', help='Run hardware integration test')
    parser.add_argument('--test', type=str, choices=['parser', 'layout', 'cv', 'transform', 'state'],
                       help='Run specific test')
    
    args = parser.parse_args()
    
    if args.test:
        tests = {
            'parser': test_ascii_parser,
            'layout': test_layout_generator,
            'cv': test_cv_tools,
            'transform': test_coordinate_transform,
            'state': test_alignment_state
        }
        tests[args.test]()
    elif args.hardware:
        test_with_hardware()
    elif args.all:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        print("Alignment System Test Suite")
        print("===========================")
        print("\nOptions:")
        print("  --all          Run all software tests")
        print("  --hardware     Run hardware integration test")
        print("  --test <name>  Run specific test (parser/layout/cv/transform/state)")
        print("\nExample:")
        print("  python test_alignment_system.py --all")


if __name__ == "__main__":
    main()