# dual_thread_with_fiducial_testing.py
"""
Enhanced multi-threaded application with fiducial detection testing.

NEW COMMANDS:
- find_fiducial x=<pixel_x> y=<pixel_y> [radius=<r>] - Test fiducial detection
- save_snapshot [name] - Save current camera frame for analysis
- load_snapshot [name] - Load saved snapshot
- test_region x=<x> y=<y> w=<w> h=<h> - Test detection in specific region
"""
import threading
import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
from pathlib import Path

from HardwareControl.CameraControl.zylaCamera import ZylaCamera
from HardwareControl.andorCameraApp import AndorCameraApp
from HardwareControl.SetupMotor.smartactStage import SmarActXYZStage
from HardwareControl.xyzStageApp import XYZStageApp
from HardwareControl.stage_commands import StageCommandProcessor
from RESTAPI.api_server import ExperimentAPI
from HardwareControl.autofocusController import AutofocusController
from AlignmentSystem.cv_tools import VisionTools


class EnhancedStageCommandProcessor(StageCommandProcessor):
    """Extended command processor with fiducial testing capabilities."""
    
    def __init__(self, stage_app, camera_app, autofocus, vision_tools):
        super().__init__(stage_app, camera_app, autofocus)
        self.vision_tools = vision_tools
        self.saved_snapshots = {}
        self.last_detection_result = None
        
    def process(self, command):
        """Process commands including new fiducial testing commands."""
        try:
            cmd = command.strip()
            if not cmd:
                return

            parts = cmd.split()
            key = parts[0].lower()

            # === NEW FIDUCIAL TESTING COMMANDS ===
            
            if key == 'find_fiducial':
                self._handle_find_fiducial(parts[1:])
                return
            
            if key == 'save_snapshot':
                self._handle_save_snapshot(parts[1:])
                return
            
            if key == 'load_snapshot':
                self._handle_load_snapshot(parts[1:])
                return
            
            if key == 'test_region':
                self._handle_test_region(parts[1:])
                return
            
            if key == 'show_detection':
                self._handle_show_detection()
                return
            
            if key == 'measure_intensity':
                self._handle_measure_intensity(parts[1:])
                return
            
            # Fall back to parent class for standard commands
            super().process(command)
            
        except Exception as e:
            print(f"[FIDUCIAL] Error processing command '{command}': {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_kwargs(self, parts):
        """Parse key=value arguments."""
        kwargs = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                k = k.strip().lower()
                v = v.strip()
                # Try to convert to int
                try:
                    v = int(v)
                except ValueError:
                    pass
                kwargs[k] = v
        return kwargs
    
    def _handle_find_fiducial(self, args):
        """
        Test fiducial detection at specified position.
        Usage: find_fiducial x=<px> y=<py> [radius=<r>] [template=<path>]
        """
        print("\n" + "="*70)
        print("FIDUCIAL DETECTION TEST")
        print("="*70)
        
        kwargs = self._parse_kwargs(args)
        
        # Get expected position
        x = kwargs.get('x')
        y = kwargs.get('y')
        
        if x is None or y is None:
            print("[FIDUCIAL] Error: Must specify x= and y= pixel coordinates")
            print("[FIDUCIAL] Usage: find_fiducial x=<px> y=<py> [radius=100]")
            return
        
        expected_pos = (int(x), int(y))
        search_radius = int(kwargs.get('radius', kwargs.get('r', 100)))
        
        print(f"[FIDUCIAL] Expected position: {expected_pos}")
        print(f"[FIDUCIAL] Search radius: {search_radius} pixels")
        
        # Capture current image
        print("[FIDUCIAL] Stopping camera stream...")
        self.camera_app.camera.stop_streaming()
        time.sleep(0.1)
        
        print("[FIDUCIAL] Capturing image...")
        image = self.camera_app.acquire_image()
        
        print(f"[FIDUCIAL] Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"[FIDUCIAL] Image range: {image.min()} to {image.max()}")
        
        # Optionally load template
        template = None
        if 'template' in kwargs:
            template_path = kwargs['template']
            if Path(template_path).exists():
                template = np.load(template_path) if template_path.endswith('.npy') else cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                print(f"[FIDUCIAL] Loaded template: {template.shape}")
            else:
                print(f"[FIDUCIAL] Warning: Template file not found: {template_path}")
        
        # Run detection
        print(f"[FIDUCIAL] Running auto-detection...")
        result = self.vision_tools.find_fiducial_auto(
            image,
            expected_pos,
            search_radius,
            template
        )
        
        if result:
            found_pos = result['position']
            confidence = result['confidence']
            method = result['method']
            
            # Calculate error
            error_x = found_pos[0] - expected_pos[0]
            error_y = found_pos[1] - expected_pos[1]
            error_dist = np.sqrt(error_x**2 + error_y**2)
            
            print(f"\n[FIDUCIAL] ✅ FOUND!")
            print(f"[FIDUCIAL]    Method: {method}")
            print(f"[FIDUCIAL]    Position: {found_pos}")
            print(f"[FIDUCIAL]    Confidence: {confidence:.4f}")
            print(f"[FIDUCIAL]    Method: {method}")
            print(f"[FIDUCIAL]    Error: Δx={error_x}, Δy={error_y}, dist={error_dist:.1f} pixels")
            
            self.last_detection_result = {
                'image': image,
                'result': result,
                'expected': expected_pos,
                'search_radius': search_radius
            }
            
            # Visualize
            self._visualize_detection(image, expected_pos, found_pos, search_radius)
            
        else:
            print(f"\n[FIDUCIAL] ❌ NOT FOUND")
            print(f"[FIDUCIAL]    No fiducial detected in search region")
            
            self.last_detection_result = {
                'image': image,
                'result': None,
                'expected': expected_pos,
                'search_radius': search_radius
            }
        
        print("="*70 + "\n")
        
        # Restart camera stream
        print("[FIDUCIAL] Restarting camera stream...")
        self.camera_app.camera.start_streaming()
        time.sleep(0.2)
    
    def _visualize_detection(self, image, expected_pos, found_pos, search_radius):
        """Visualize detection result."""
        self.vision_tools.visualize_detection(
            image,expected_pos, found_pos, search_radius
        )
    
    def _handle_save_snapshot(self, args):
        """
        Save current camera frame for later analysis.
        Usage: save_snapshot [name]
        """
        name = args[0] if args else "default"
        
        print("[FIDUCIAL] Stopping camera stream...")
        self.camera_app.camera.stop_streaming()
        time.sleep(0.1)
        
        print("[FIDUCIAL] Capturing snapshot...")
        image = self.camera_app.acquire_image()
        
        # Save to memory
        self.saved_snapshots[name] = image
        
        # Save to file
        filename = f"snapshot_{name}.npy"
        np.save(filename, image)
        
        print(f"[FIDUCIAL] ✅ Snapshot '{name}' saved")
        print(f"[FIDUCIAL]    Shape: {image.shape}")
        print(f"[FIDUCIAL]    File: {filename}")
        
        print("[FIDUCIAL] Restarting camera stream...")
        self.camera_app.camera.start_streaming()
        time.sleep(0.2)
    
    def _handle_load_snapshot(self, args):
        """
        Load snapshot from file.
        Usage: load_snapshot [name]
        """
        name = args[0] if args else "default"
        filename = f"snapshot_{name}.npy"
        
        if not Path(filename).exists():
            print(f"[FIDUCIAL] ❌ Snapshot file not found: {filename}")
            return
        
        image = np.load(filename)
        self.saved_snapshots[name] = image
        
        print(f"[FIDUCIAL] ✅ Snapshot '{name}' loaded")
        print(f"[FIDUCIAL]    Shape: {image.shape}")
        print(f"[FIDUCIAL]    Range: {image.min()} to {image.max()}")
    
    def _handle_test_region(self, args):
        """
        Test detection in a specific rectangular region.
        Usage: test_region x=<x> y=<y> w=<w> h=<h>
        """
        kwargs = self._parse_kwargs(args)
        
        x = kwargs.get('x')
        y = kwargs.get('y')
        w = kwargs.get('w', kwargs.get('width'))
        h = kwargs.get('h', kwargs.get('height'))
        
        if None in (x, y, w, h):
            print("[FIDUCIAL] Error: Must specify x, y, w, h")
            print("[FIDUCIAL] Usage: test_region x=<x> y=<y> w=<w> h=<h>")
            return
        
        search_region = (int(x), int(y), int(w), int(h))
        
        print(f"[FIDUCIAL] Testing region: {search_region}")
        
        # Capture image
        self.camera_app.camera.stop_streaming()
        time.sleep(0.1)
        image = self.camera_app.acquire_image()
        
        # Test all methods
        print("\n[FIDUCIAL] Testing corner detection...")
        result_corner = self.vision_tools.find_fiducial_corners(image, search_region)
        if result_corner:
            print(f"   ✅ Found at {result_corner['position']}, confidence={result_corner['confidence']:.4f}")
        else:
            print(f"   ❌ Not found")
        
        print("\n[FIDUCIAL] Testing contour detection (bright)...")
        result_contour = self.vision_tools.find_fiducial_contours(image, search_region, invert=False)
        if result_contour:
            print(f"   ✅ Found at {result_contour['position']}, confidence={result_contour['confidence']:.4f}")
        else:
            print(f"   ❌ Not found")
        
        print("\n[FIDUCIAL] Testing contour detection (dark)...")
        result_contour_inv = self.vision_tools.find_fiducial_contours(image, search_region, invert=True)
        if result_contour_inv:
            print(f"   ✅ Found at {result_contour_inv['position']}, confidence={result_contour_inv['confidence']:.4f}")
        else:
            print(f"   ❌ Not found")
        
        self.camera_app.camera.start_streaming()
        time.sleep(0.2)
    
    def _handle_show_detection(self):
        """Show last detection result in detail."""
        if not self.last_detection_result:
            print("[FIDUCIAL] No detection result available. Run find_fiducial first.")
            return
        
        result = self.last_detection_result
        image = result['image']
        detection = result['result']
        expected = result['expected']
        
        if detection:
            print(f"\n[FIDUCIAL] Last Detection:")
            print(f"   Expected: {expected}")
            print(f"   Found: {detection['position']}")
            print(f"   Method: {detection['method']}")
            print(f"   Confidence: {detection['confidence']:.4f}")
            
            # Re-visualize
            self._visualize_detection(
                image, 
                expected, 
                detection['position'], 
                result['search_radius']
            )
        else:
            print(f"[FIDUCIAL] Last detection failed at position {expected}")
    
    def _handle_measure_intensity(self, args):
        """
        Measure intensity in region.
        Usage: measure_intensity [x=<x> y=<y> w=<w> h=<h>]
        """
        kwargs = self._parse_kwargs(args)
        
        roi = None
        if all(k in kwargs for k in ('x', 'y', 'w', 'h')):
            roi = (kwargs['x'], kwargs['y'], kwargs['w'], kwargs['h'])
        
        # Capture image
        self.camera_app.camera.stop_streaming()
        time.sleep(0.1)
        image = self.camera_app.acquire_image()
        
        metrics = self.vision_tools.measure_intensity(image, roi)
        
        print(f"\n[FIDUCIAL] Intensity Metrics:")
        print(f"   Mean: {metrics['mean']:.2f}")
        print(f"   Sum: {metrics['sum']:.0f}")
        print(f"   Max: {metrics['max']}")
        print(f"   Min: {metrics['min']}")
        print(f"   Std: {metrics['std']:.2f}")
        
        self.camera_app.camera.start_streaming()
        time.sleep(0.2)
    
    def _print_help(self):
        """Override parent help to include new commands."""
        super()._print_help()
        
        print('\n--- FIDUCIAL TESTING COMMANDS ---')
        print('find_fiducial x=<px> y=<py> [radius=<r>]  - Test fiducial detection')
        print('save_snapshot [name]                       - Save current frame')
        print('load_snapshot [name]                       - Load saved snapshot')
        print('test_region x=<x> y=<y> w=<w> h=<h>       - Test detection in region')
        print('show_detection                             - Show last detection result')
        print('measure_intensity [x=<x> y=<y> w=<w> h=<h>] - Measure intensity in ROI')
        print('\nExamples:')
        print('  find_fiducial x=512 y=256 radius=150')
        print('  save_snapshot test1')
        print('  test_region x=400 y=200 w=300 h=300')
        print('='*70 + '\n')


class DualThreadApp:
    """Manages camera stream, stage control, vision tools, and REST API."""
    
    def __init__(self, enable_api=True, api_port=5000):
        self.camera = None
        self.camera_app = None
        self.stage = None
        self.stage_app = None
        self.autofocus = None
        self.vision_tools = None  # NEW
        
        self.camera_thread = None
        self.stage_thread = None
        self.input_thread = None
        self.api_thread = None
        
        self.stop_event = threading.Event()
        self.command_queue = []
        self.queue_lock = threading.Lock()
        
        # API settings
        self.enable_api = enable_api
        self.api_port = api_port
        self.api_server = None
        
    def initialize_camera(self):
        """Initialize camera and app in the main thread."""
        print("[INIT] Initializing camera...")
        self.camera = ZylaCamera()
        self.camera.connect()
        self.camera_app = AndorCameraApp(self.camera)
        
        self.camera_app.set_gain_mode("16-bit (low noise & high well capacity)")
        self.camera_app.set_exposure(0.02)
        print("[INIT] Camera initialized successfully")
        
    def initialize_stage(self):
        """Initialize stage and app in the main thread."""
        print("[INIT] Initializing stage...")
        self.stage = SmarActXYZStage()
        self.stage_app = XYZStageApp(self.stage)
        
        self.autofocus = AutofocusController(self.camera_app, self.stage_app)
        
        # Initialize vision tools (NEW)
        print("[INIT] Initializing vision tools...")
        self.vision_tools = VisionTools()
        
        # Use enhanced command processor
        self.stage_cmd_processor = EnhancedStageCommandProcessor(
            self.stage_app, 
            self.camera_app, 
            self.autofocus,
            self.vision_tools  # NEW
        )

        print("[INIT] Stage initialized successfully")
    
    def initialize_api(self):
        """Initialize REST API server."""
        if not self.enable_api:
            return
            
        print("[INIT] Initializing REST API server...")
        self.api_server = ExperimentAPI(self)
        print(f"[INIT] REST API will be available at http://localhost:{self.api_port}")
        
    def camera_stream_thread(self):
        """Thread 1: Run camera live stream."""
        print("[CAMERA] Starting live stream...")
        print("[CAMERA] Note: OpenCV window 'q' key is disabled, use CLI 'quit' command")
        try:
            self.camera.start_streaming()
            
            window_title = "Andor Live View (Type 'quit' in terminal to stop)"

            while not self.stop_event.is_set():
                image_data = self.camera.read_next_image()
                if image_data is None:
                    time.sleep(0.01)
                    continue
                
                combined = self.camera_app._render_frame(
                    image_data, 
                    max_width=1550, 
                    max_height=800
                )
                
                cv2.imshow(window_title, combined)
                
                key = cv2.waitKey(1) & 0xFF
                
            cv2.destroyAllWindows()
            self.camera.stop_streaming()
            
        except Exception as e:
            print(f"[CAMERA] Error: {e}")
        finally:
            print("[CAMERA] Live stream stopped")
            
    def stage_control_thread(self):
        """Thread 2: Process stage movement commands from queue."""
        print("[STAGE] Ready to receive movement commands")
        
        try:
            while not self.stop_event.is_set():
                command = None
                with self.queue_lock:
                    if self.command_queue:
                        command = self.command_queue.pop(0)
                
                if command:
                    self.stage_cmd_processor.process(command)
                else:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"[STAGE] Error: {e}")
        finally:
            print("[STAGE] Control stopped")
    
    def input_thread_func(self):
        """Thread 3: Handle CLI input."""
        print("[INPUT] CLI ready. Type 'help' for commands.")
        
        try:
            while not self.stop_event.is_set():
                try:
                    user_input = input(">> ").strip()
                    
                    if not user_input:
                        continue
                        
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("[INPUT] Stopping application...")
                        self.stop_event.set()
                        break
                        
                    with self.queue_lock:
                        self.command_queue.append(user_input)
                        
                except EOFError:
                    print("\n[INPUT] EOF received, stopping...")
                    self.stop_event.set()
                    break
                except KeyboardInterrupt:
                    print("\n[INPUT] Interrupted, stopping...")
                    self.stop_event.set()
                    break
                    
        except Exception as e:
            print(f"[INPUT] Error: {e}")
        finally:
            print("[INPUT] Input handler stopped")
    
    def api_server_thread(self):
        """Thread 4: Run REST API server."""
        if not self.enable_api or not self.api_server:
            return
            
        print(f"[API] Starting REST API server on port {self.api_port}...")
        try:
            config = uvicorn.Config(
                self.api_server.get_app(),
                host="0.0.0.0",
                port=self.api_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            print(f"[API] Error: {e}")
        finally:
            print("[API] Server stopped")
            
    def run(self):
        """Main execution: start all threads."""
        try:
            # Initialize hardware
            self.initialize_camera()
            self.initialize_stage()
            self.initialize_api()
            
            print("\n" + "="*70)
            print("MULTI-THREAD APP WITH FIDUCIAL TESTING")
            print("="*70)
            print("Thread 1: Camera live stream")
            print("Thread 2: Stage control + Fiducial testing")
            print("Thread 3: CLI input handler")
            if self.enable_api:
                print(f"Thread 4: REST API server (port {self.api_port})")
            print("="*70)
            self.stage_cmd_processor._print_help()
            
            if self.enable_api:
                print(f"\n{'='*70}")
                print("REST API ENDPOINTS")
                print('='*70)
                print(f"API Documentation: http://localhost:{self.api_port}/docs")
                print(f"Health Check:      http://localhost:{self.api_port}/health")
                print(f"Current Status:    http://localhost:{self.api_port}/status")
                print('='*70 + '\n')
            
            # Create and start threads
            self.camera_thread = threading.Thread(
                target=self.camera_stream_thread,
                name="CameraThread",
                daemon=True
            )
            
            self.stage_thread = threading.Thread(
                target=self.stage_control_thread,
                name="StageThread",
                daemon=True
            )
            
            self.input_thread = threading.Thread(
                target=self.input_thread_func,
                name="InputThread",
                daemon=True
            )
            
            if self.enable_api:
                self.api_thread = threading.Thread(
                    target=self.api_server_thread,
                    name="APIThread",
                    daemon=True
                )
            
            # Start all threads
            self.camera_thread.start()
            time.sleep(0.5)
            self.stage_thread.start()
            self.input_thread.start()
            
            if self.enable_api:
                self.api_thread.start()
                time.sleep(1)
            
            # Wait for stop signal
            while not self.stop_event.is_set():
                time.sleep(0.1)
            
            print("\n[MAIN] Stopping all threads...")
            time.sleep(1)
            
            print("[MAIN] All threads completed")
            
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user (Ctrl+C)")
            self.stop_event.set()
            time.sleep(1)
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        print("\n[CLEANUP] Closing hardware connections...")
        
        if self.camera_app:
            try:
                self.camera_app.close()
                print("[CLEANUP] Camera closed")
            except Exception as e:
                print(f"[CLEANUP] Error closing camera: {e}")
                
        if self.stage_app:
            try:
                self.stage_app.close()
                print("[CLEANUP] Stage closed")
            except Exception as e:
                print(f"[CLEANUP] Error closing stage: {e}")
                
        plt.close('all')
        
        print("[CLEANUP] Done")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-threaded microscopy control with fiducial testing")
    parser.add_argument('--no-api', action='store_true', help='Disable REST API')
    parser.add_argument('--api-port', type=int, default=5000, help='API server port')
    
    args = parser.parse_args()
    
    app = DualThreadApp(
        enable_api=not args.no_api,
        api_port=args.api_port
    )
    
    app.run()