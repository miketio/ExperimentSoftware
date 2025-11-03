#!/usr/bin/env python3
# test_api_client.py
"""
Test client for the Experiment Control REST API.
Demonstrates how to interact with the API programmatically.
"""
import requests
import json
import time
from typing import Optional


class ExperimentAPIClient:
    """Client for interacting with the Experiment Control REST API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _get(self, endpoint: str) -> dict:
        """Make GET request."""
        response = self.session.get(f"{self.base_url}{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: dict) -> dict:
        """Make POST request."""
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================
    # System Methods
    # ========================================
    
    def health_check(self) -> dict:
        """Check system health."""
        return self._get("/health")
    
    def get_status(self) -> dict:
        """Get current position of all axes."""
        return self._get("/status")
    
    def get_position(self, axis: str) -> dict:
        """Get position of specific axis."""
        return self._get(f"/position/{axis}")
    
    # ========================================
    # Command Execution
    # ========================================
    
    def execute_command(self, command: str) -> dict:
        """Execute a CLI command."""
        return self._post("/command", {"command": command})
    
    # ========================================
    # Movement Methods
    # ========================================
    
    def move_absolute(self, axis: str, position: int) -> dict:
        """Move axis to absolute position."""
        return self._post("/move/absolute", {
            "axis": axis,
            "position": position
        })
    
    def move_relative(self, axis: str, shift: int) -> dict:
        """Move axis relative to current position."""
        return self._post("/move/relative", {
            "axis": axis,
            "shift": shift
        })
    
    # ========================================
    # Autofocus Methods
    # ========================================
    
    def run_autofocus(
        self,
        axis: str = "x",
        range: Optional[int] = None,
        step: Optional[int] = None,
        enable_plot: bool = True
    ) -> dict:
        """Run autofocus scan."""
        data = {
            "axis": axis,
            "enable_plot": enable_plot
        }
        if range is not None:
            data["range"] = range
        if step is not None:
            data["step"] = step
        
        return self._post("/autofocus", data)
    
    def get_autofocus_results(self) -> dict:
        """Get last autofocus results."""
        return self._get("/autofocus/results")
    
    def save_autofocus_results(self, filename: str = "autofocus_results.txt") -> dict:
        """Save autofocus results to file."""
        return self._post(f"/autofocus/save?filename={filename}", {})
    
    # ========================================
    # Camera Methods
    # ========================================
    
    def set_roi(
        self,
        left: Optional[int] = None,
        top: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> dict:
        """Set camera region of interest."""
        return self._post("/camera/roi", {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        })
    
    def get_camera_info(self) -> dict:
        """Get camera information."""
        return self._get("/camera/info")
    
    # ========================================
    # System Control
    # ========================================
    
    def stop_system(self) -> dict:
        """Stop the experiment control system."""
        return self._post("/stop", {})


# ========================================
# Example Usage & Tests
# ========================================

def test_basic_operations():
    """Test basic API operations."""
    print("="*70)
    print("TESTING EXPERIMENT CONTROL API")
    print("="*70)
    
    client = ExperimentAPIClient()
    
    try:
        # 1. Health check
        print("\n1. Health Check")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Camera: {health['camera_connected']}")
        print(f"   Stage: {health['stage_connected']}")
        
        # 2. Get current position
        print("\n2. Current Position")
        status = client.get_status()
        print(f"   X: {status['x']}nm")
        print(f"   Y: {status['y']}nm")
        print(f"   Z: {status['z']}nm")
        
        # 3. Move X axis
        print("\n3. Moving X axis to 5000nm...")
        result = client.move_absolute("x", 5000)
        print(f"   New position: {result['position']}nm")
        time.sleep(1)
        
        # 4. Relative move
        print("\n4. Moving X by +500nm...")
        result = client.move_relative("x", 500)
        print(f"   New position: {result['position']}nm")
        time.sleep(1)
        
        # 5. Get camera info
        print("\n5. Camera Information")
        camera_info = client.get_camera_info()
        print(f"   Model: {camera_info['model']}")
        print(f"   Exposure: {camera_info['exposure_time']}s")
        print(f"   Sensor: {camera_info['sensor_width']}x{camera_info['sensor_height']}")
        
        # 6. Execute CLI command
        print("\n6. Execute CLI command 'pos'")
        result = client.execute_command("pos")
        print(f"   Status: {result['status']}")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure the experiment control system is running:")
        print("  python dual_thread_with_api.py")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


def test_autofocus():
    """Test autofocus functionality."""
    print("="*70)
    print("TESTING AUTOFOCUS API")
    print("="*70)
    
    client = ExperimentAPIClient()
    
    try:
        print("\n1. Running autofocus on X-axis...")
        print("   (This will take some time...)")
        
        result = client.run_autofocus(
            axis="x",
            range=10000,
            step=500,
            enable_plot=False  # Disable plot for automated testing
        )
        
        print(f"\n   Best position: {result['best_position']}nm")
        print(f"   Focus metric: {result['best_metric']:.2f}")
        
        print("\n2. Saving autofocus results...")
        save_result = client.save_autofocus_results("test_autofocus.txt")
        print(f"   {save_result['message']}")
        
        print("\n" + "="*70)
        print("AUTOFOCUS TEST PASSED")
        print("="*70)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


def interactive_demo():
    """Interactive demonstration of API capabilities."""
    client = ExperimentAPIClient()
    
    print("="*70)
    print("INTERACTIVE API DEMO")
    print("="*70)
    print("\nAvailable commands:")
    print("  status     - Show current position")
    print("  move       - Move to position")
    print("  autofocus  - Run autofocus")
    print("  info       - Show camera info")
    print("  quit       - Exit demo")
    print("="*70)
    
    while True:
        try:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == 'quit':
                break
            
            elif cmd == 'status':
                status = client.get_status()
                print(f"Position - X: {status['x']}nm, Y: {status['y']}nm, Z: {status['z']}nm")
            
            elif cmd == 'move':
                axis = input("  Axis (x/y/z): ").strip().lower()
                position = int(input("  Position (nm): ").strip())
                result = client.move_absolute(axis, position)
                print(f"  Moved to {result['position']}nm")
            
            elif cmd == 'autofocus':
                axis = input("  Axis (x/y/z) [x]: ").strip().lower() or "x"
                print("  Running autofocus...")
                result = client.run_autofocus(axis=axis, enable_plot=False)
                print(f"  Best position: {result['best_position']}nm")
            
            elif cmd == 'info':
                info = client.get_camera_info()
                print(f"  Camera: {info['model']}")
                print(f"  Exposure: {info['exposure_time']}s")
            
            else:
                print(f"  Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main entry point."""
    import sys
    
    print("\n🧪 Experiment Control API Test Client\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_basic_operations()
        elif sys.argv[1] == "autofocus":
            test_autofocus()
        elif sys.argv[1] == "demo":
            interactive_demo()
        else:
            print("Usage:")
            print("  python test_api_client.py test       - Run basic tests")
            print("  python test_api_client.py autofocus  - Test autofocus")
            print("  python test_api_client.py demo       - Interactive demo")
    else:
        print("Choose mode:")
        print("  1 - Run basic tests")
        print("  2 - Test autofocus")
        print("  3 - Interactive demo")
        choice = input("\nSelect (1/2/3): ").strip()
        
        if choice == '1':
            test_basic_operations()
        elif choice == '2':
            test_autofocus()
        elif choice == '3':
            interactive_demo()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()