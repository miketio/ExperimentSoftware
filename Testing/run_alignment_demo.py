#!/usr/bin/env python3
# run_alignment_demo.py
"""
Interactive demo script for alignment system.
Guides user through setup and first alignment.
"""
import sys
import time
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def check_prerequisites():
    """Check if required files exist."""
    print_header("CHECKING PREREQUISITES")
    
    required_files = [
        ("ASCII file", "ArrayDquasiperiodic1.ASC"),
        ("Layout config", "config/sample_layout.json"),
    ]
    
    all_ok = True
    for name, path in required_files:
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name} NOT FOUND: {path}")
            all_ok = False
    
    return all_ok


def setup_layout():
    """Generate layout configuration if needed."""
    print_header("SETUP: LAYOUT CONFIGURATION")
    
    if Path("config/sample_layout.json").exists():
        print("✅ Layout configuration already exists.")
        response = input("\nRegenerate? (y/N): ").strip().lower()
        if response != 'y':
            return True
    
    ascii_file = "ArrayDquasiperiodic1.ASC"
    if not Path(ascii_file).exists():
        print(f"\n❌ ASCII file not found: {ascii_file}")
        print("Please provide the ASCII file for one block.")
        return False
    
    try:
        from AlignmentSystem.layout_config_generator import generate_layout_config
        
        print(f"\nGenerating layout from {ascii_file}...")
        layout = generate_layout_config(ascii_file, "config/sample_layout.json")
        print("\n✅ Layout configuration generated successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Failed to generate layout: {e}")
        return False


def run_software_tests():
    """Run software tests."""
    print_header("RUNNING SOFTWARE TESTS")
    
    print("\nThis will test all software components without hardware.")
    response = input("Run tests? (Y/n): ").strip().lower()
    
    if response == 'n':
        print("Skipping tests.")
        return True
    
    try:
        from Testing.test_alignment_system import run_all_tests
        success = run_all_tests()
        return success
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        return False


def demo_alignment_workflow():
    """Demonstrate alignment workflow (requires hardware)."""
    print_header("ALIGNMENT WORKFLOW DEMO")
    
    print("\n⚠️  This requires connected hardware (camera + stage).")
    print("Make sure:")
    print("  • Camera is connected and initialized")
    print("  • Stage is connected and homed")
    print("  • Sample is mounted")
    print("  • Experiment control system is running")
    
    response = input("\nProceed with hardware demo? (y/N): ").strip().lower()
    
    if response != 'y':
        print("\nSkipping hardware demo.")
        print("\nTo run alignment manually:")
        print("  1. Start your experiment control system")
        print("  2. Use the agent or Python API")
        print("  3. Run: calibrate_sample()")
        print("  4. Run: align_to_grating(10, 25, 'left')")
        return None
    
    try:
        print("\n" + "-"*70)
        print("INITIALIZING HARDWARE")
        print("-"*70)
        
        # Import hardware
        from HardwareControl.CameraControl.zylaCamera import ZylaCamera
        from HardwareControl.andorCameraApp import AndorCameraApp
        from HardwareControl.SetupMotor.smartactStage import SmarActXYZStage
        from HardwareControl.xyzStageApp import XYZStageApp
        from AlignmentSystem.alignment_controller import AlignmentController
        from AlignmentSystem.layout_config_generator import load_layout_config
        
        print("\n1. Initializing camera...")
        camera = ZylaCamera()
        camera.connect()
        camera_app = AndorCameraApp(camera)
        camera_app.set_exposure(0.02)
        print("   ✅ Camera ready")
        
        print("\n2. Initializing stage...")
        stage = SmarActXYZStage()
        stage_app = XYZStageApp(stage)
        print("   ✅ Stage ready")
        
        print("\n3. Loading layout...")
        layout = load_layout_config("config/sample_layout.json")
        print(f"   ✅ Loaded {layout['total_blocks']} blocks")
        
        print("\n4. Creating alignment controller...")
        alignment = AlignmentController(camera_app, stage_app, layout)
        print("   ✅ Controller ready")
        
        # Interactive workflow
        print("\n" + "-"*70)
        print("ALIGNMENT WORKFLOW")
        print("-"*70)
        
        # Step 1: Calibration
        print("\n📍 STEP 1: Sample Calibration")
        print("This will:")
        print("  • Find fiducial at block 1 (top-left corner)")
        print("  • Find fiducial at block 20 (bottom-right corner)")
        print("  • Calculate rotation and translation")
        
        response = input("\nRun calibration? (Y/n): ").strip().lower()
        if response != 'n':
            print("\nCalibrating...")
            calib_result = alignment.calibrate_sample(block1_id=1, block2_id=20)
            
            if calib_result['success']:
                print("\n✅ Calibration successful!")
                calib = calib_result['calibration']
                print(f"   Rotation: {calib['angle_deg']:.3f}°")
                print(f"   Error: {calib['mean_error_nm']:.1f} nm")
            else:
                print(f"\n❌ Calibration failed: {calib_result['error']}")
                return False
        
        # Step 2: Test alignment
        print("\n🎯 STEP 2: Align to Grating")
        print(f"Target: Block 10, Waveguide 25, Left side")
        
        response = input("\nRun alignment? (Y/n): ").strip().lower()
        if response != 'n':
            print("\nAligning...")
            print("This will take ~1-2 minutes (441 grid positions)...")
            
            align_result = alignment.align_to_grating(
                block_id=10,
                waveguide_number=25,
                side='left'
            )
            
            if align_result['success']:
                print("\n✅ Alignment successful!")
                print(f"   Best position: {align_result['best_position']}")
                print(f"   Best intensity: {align_result['best_intensity']:.1f}")
            else:
                print(f"\n❌ Alignment failed: {align_result['error']}")
                return False
        
        # Cleanup
        print("\n" + "-"*70)
        print("CLEANUP")
        print("-"*70)
        camera.disconnect()
        stage.close()
        print("✅ Hardware disconnected")
        
        print("\n" + "="*70)
        print("🎉 DEMO COMPLETE!")
        print("="*70)
        print("\nYou can now:")
        print("  • Use the agent for natural language control")
        print("  • Scan all 20 blocks: scan_all_center_gratings()")
        print("  • Integrate with your experiment workflow")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo workflow."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🔬  AUTOMATED ALIGNMENT SYSTEM - SETUP & DEMO  🎯         ║
║                                                               ║
║         Grating Coupler Alignment Automation                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠️  Missing required files. Please:")
        print("  1. Place ArrayDquasiperiodic1.ASC in current directory")
        print("  2. Run: python AlignmentSystem/layout_config_generator.py ArrayDquasiperiodic1.ASC")
        sys.exit(1)
    
    # Setup layout
    if not setup_layout():
        print("\n⚠️  Layout setup failed.")
        sys.exit(1)
    
    # Run tests
    if not run_software_tests():
        print("\n⚠️  Some tests failed. Fix issues before proceeding.")
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Hardware demo
    demo_alignment_workflow()
    
    print("\n" + "="*70)
    print("For more information, see: ALIGNMENT_SYSTEM_README.md")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)