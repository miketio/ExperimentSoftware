#!/usr/bin/env python3
"""
Camera Cleanup Script

Run this if you get AT_ERR_DEVICEINUSE (error 38).

This script:
1. Attempts to close any open Andor camera connections
2. Checks for orphaned Python processes
3. Provides diagnostic information

Usage:
    python cleanup_camera.py
"""

import sys
import time

# Import os here (after function definitions)
import os
# global os  # Make it accessible to other functions

def force_close_andor_camera():
    """
    Attempt to close Andor SDK3 camera connections.
    """
    print("=" * 70)
    print("Andor Camera Cleanup Script")
    print("=" * 70)
    
    # Step 1: Try to import and close via pylablib
    print("\n[1/4] Attempting to close camera via pylablib...")
    
    try:
        import pylablib.devices.Andor as Andor
        
        print("  ✅ pylablib imported successfully")
        
        # Try to connect and close
        try:
            print("  → Attempting to open camera...")
            cam = Andor.AndorSDK3Camera(idx=0)
            print("  ✅ Camera opened (it was available!)")
            
            print("  → Stopping any acquisition...")
            try:
                cam.stop_acquisition()
            except:
                pass
            
            print("  → Closing camera...")
            cam.close()
            print("  ✅ Camera closed successfully")
            
            time.sleep(0.5)
            
            print("\n✅ SUCCESS: Camera is now available")
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            if "AT_ERR_DEVICEINUSE" in error_msg or "38" in error_msg:
                print("  ❌ Camera is IN USE by another process")
                print(f"     Error: {error_msg}")
                return False
            
            elif "AT_ERR_NOTFOUND" in error_msg or "13" in error_msg:
                print("  ⚠️  No camera detected (check hardware)")
                return False
            
            else:
                print(f"  ❌ Unexpected error: {error_msg}")
                return False
    
    except ImportError as e:
        print(f"  ❌ pylablib not installed: {e}")
        print("     Install with: pip install pylablib")
        return False


def check_python_processes():
    """
    Check for other Python processes that might be using the camera.
    """
    print("\n[2/4] Checking for other Python processes...")
    
    try:
        import psutil
        
        current_pid = os.getpid()
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    if proc.info['pid'] != current_pid:
                        python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"  ⚠️  Found {len(python_processes)} other Python process(es):")
            for proc in python_processes:
                try:
                    cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'N/A'
                    print(f"     PID {proc.info['pid']}: {cmdline}")
                except:
                    print(f"     PID {proc.info['pid']}: <access denied>")
            
            print("\n  💡 TIP: Kill these processes in Task Manager if they're stuck")
        else:
            print("  ✅ No other Python processes found")
    
    except ImportError:
        print("  ⚠️  psutil not installed (optional check)")
        print("     Install with: pip install psutil")


def check_andor_software():
    """
    Check for Andor software that might be using the camera.
    """
    print("\n[3/4] Checking for Andor software...")
    
    try:
        import psutil
        
        andor_processes = []
        andor_names = ['solis', 'andor', 'atcore']
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info['name'].lower()
                if any(name in proc_name for name in andor_names):
                    andor_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if andor_processes:
            print(f"  ⚠️  Found {len(andor_processes)} Andor process(es):")
            for proc in andor_processes:
                print(f"     PID {proc.info['pid']}: {proc.info['name']}")
            
            print("\n  ❗ CLOSE THESE APPLICATIONS before running Python camera code")
        else:
            print("  ✅ No Andor software detected")
    
    except ImportError:
        print("  ⚠️  psutil not installed (skipping check)")


def print_solutions():
    """
    Print common solutions.
    """
    print("\n[4/4] Common Solutions:")
    print("=" * 70)
    print("""
If camera is still IN USE:

1️⃣  CLOSE OTHER SOFTWARE:
   • Solis (Andor's official software)
   • AndorView
   • Any Python IDEs with camera code running
   • Jupyter notebooks

2️⃣  KILL ORPHANED PROCESSES:
   • Open Task Manager (Ctrl+Shift+Esc)
   • Look for: python.exe, pythonw.exe
   • End any suspicious Python processes

3️⃣  POWER CYCLE CAMERA:
   • Unplug USB cable
   • Wait 10 seconds
   • Plug back in
   • Wait for Windows to detect it

4️⃣  RESTART COMPUTER:
   • Last resort if nothing else works

5️⃣  CHECK SDK INSTALLATION:
   • Ensure Andor SDK3 is properly installed
   • Try reinstalling drivers

═══════════════════════════════════════════════════════════════════════
""")


def main():
    """Main cleanup routine."""
    

    
    print("\n🔧 Starting camera cleanup...\n")
    
    # Try to close camera
    success = force_close_andor_camera()
    
    # Check for conflicts
    check_python_processes()
    check_andor_software()
    
    # Print solutions
    print_solutions()
    
    # Final status
    print("=" * 70)
    if success:
        print("✅ STATUS: Camera should now be available")
        print("\nYou can now run your application.")
    else:
        print("❌ STATUS: Camera is still IN USE")
        print("\nFollow the solutions above to resolve the issue.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Cleanup script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)