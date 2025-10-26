# dual_thread_with_api.py
"""
Multi-threaded application with REST API integration.

Threads:
- Thread 1: Camera live stream (GUI)
- Thread 2: Stage control via CLI commands (including autofocus)
- Thread 3: CLI input handler
- Thread 4: REST API server (NEW)

All threads run simultaneously without blocking each other.
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "SetupMotor"))
sys.path.append(os.path.join(os.path.dirname(__file__), "CameraControl"))
import threading
import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
from HardwareControl.CameraControl.zylaCamera import ZylaCamera
from HardwareControl.andorCameraApp import AndorCameraApp
from HardwareControl.SetupMotor.smartactStage import SmarActXYZStage
from HardwareControl.xyzStageApp import XYZStageApp
from HardwareControl.stage_commands import StageCommandProcessor
from RESTAPI.api_server import ExperimentAPI


class AutofocusController:
    """Handles autofocus operations with live plotting."""
    
    def __init__(self, camera_app, stage_app):
        self.camera_app = camera_app
        self.stage_app = stage_app
        
        # Autofocus parameters
        self.default_range = 10000  # nm
        self.default_step = 500     # nm
        self.axis = 'x'             # default axis for autofocus
        
        # Results storage
        self.positions = []
        self.metrics = []
        self.best_position = None
        self.best_metric = None
        
        # Live plotting
        self.plot_enabled = True
        self.fig = None
        self.ax = None
        self.line = None
        self.is_plotting = False
    
    def calculate_focus_metric(self, image):
        """Calculate Variance of Laplacian (sharpness metric)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
            
    def run_autofocus(self, axis='x', scan_range=None, step_size=None, enable_plot=True):
        """Run autofocus scan on specified axis."""
        self.axis = axis.lower()
        if self.axis not in ['x', 'y', 'z']:
            print(f"[AUTOFOCUS] Error: Invalid axis '{axis}'")
            return False
            
        scan_range = scan_range or self.default_range
        step_size = step_size or self.default_step
        self.plot_enabled = enable_plot
        
        # Clear previous results
        self.positions = []
        self.metrics = []
        self.best_position = None
        self.best_metric = None
        
        try:
            current_pos = self.stage_app.get_pos(self.axis)
            start_pos = current_pos - scan_range // 2
            end_pos = current_pos + scan_range // 2
            
            print(f"\n[AUTOFOCUS] ========================================")
            print(f"[AUTOFOCUS] Starting autofocus on {self.axis.upper()}-axis")
            print(f"[AUTOFOCUS] Range: {start_pos}nm to {end_pos}nm")
            print(f"[AUTOFOCUS] Step: {step_size}nm")
            print(f"[AUTOFOCUS] Current position: {current_pos}nm")
            print(f"[AUTOFOCUS] ========================================\n")
            
            print("[AUTOFOCUS] Stopping camera stream...")
            self.camera_app.camera.stop_streaming()
            time.sleep(0.1)

            # Scan loop
            num_steps = int((end_pos - start_pos) / step_size) + 1
            for i, pos in enumerate(range(start_pos, end_pos + 1, step_size)):
                self.stage_app.move_abs(self.axis, pos)
                time.sleep(0.1)
                
                self.camera_app.set_roi(400, 1200, 1300, 1000)
                image = self.camera_app.acquire_image()
                
                metric = self.calculate_focus_metric(image)
                
                self.positions.append(pos)
                self.metrics.append(metric)
                
                progress = (i + 1) / num_steps * 100
                print(f"[AUTOFOCUS] [{progress:5.1f}%] {self.axis.upper()}={pos:6d}nm -> focus={metric:8.2f}")
                    
            # Find best position
            best_idx = np.argmax(self.metrics)
            self.best_position = self.positions[best_idx]
            self.best_metric = self.metrics[best_idx]
            
            print(f"\n[AUTOFOCUS] ========================================")
            print(f"[AUTOFOCUS] Scan complete!")
            print(f"[AUTOFOCUS] Best focus at {self.axis.upper()}={self.best_position}nm")
            print(f"[AUTOFOCUS] Focus metric: {self.best_metric:.2f}")
            print(f"[AUTOFOCUS] ========================================")
            
            print(f"[AUTOFOCUS] Moving to optimal position...")
            self.stage_app.move_abs(self.axis, self.best_position)
            
            return True
            
        except Exception as e:
            print(f"[AUTOFOCUS] Error during autofocus: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            print("[AUTOFOCUS] Restarting camera stream...")
            try:
                self.camera_app.set_roi()
                self.camera_app.camera.start_streaming()
                time.sleep(0.2)
            except Exception as e:
                print(f"[AUTOFOCUS] Failed to restart stream: {e}")
            print("[AUTOFOCUS] Done!\n")
    
    def save_results(self, filename="autofocus_results.txt"):
        """Save autofocus results to file."""
        if not self.positions:
            print("[AUTOFOCUS] No results to save")
            return
            
        try:
            with open(filename, 'w') as f:
                f.write(f"Autofocus Results\n")
                f.write(f"Axis: {self.axis.upper()}\n")
                f.write(f"Best Position: {self.best_position}nm\n")
                f.write(f"Best Metric: {self.best_metric:.2f}\n\n")
                f.write(f"Position (nm)\tFocus Metric\n")
                for pos, metric in zip(self.positions, self.metrics):
                    f.write(f"{pos}\t{metric:.2f}\n")
            print(f"[AUTOFOCUS] Results saved to {filename}")
        except Exception as e:
            print(f"[AUTOFOCUS] Error saving results: {e}")


class DualThreadApp:
    """Manages camera stream, stage control, and REST API in separate threads."""
    
    def __init__(self, enable_api=True, api_port=5000):
        self.camera = None
        self.camera_app = None
        self.stage = None
        self.stage_app = None
        self.autofocus = None
        
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
        
        self.stage_cmd_processor = StageCommandProcessor(
            self.stage_app, 
            self.camera_app, 
            self.autofocus
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
            print("MULTI-THREAD APPLICATION WITH REST API")
            print("="*70)
            print("Thread 1: Camera live stream")
            print("Thread 2: Stage control (processes commands)")
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
                time.sleep(1)  # Give API server time to start
            
            # Wait for stop signal
            while not self.stop_event.is_set():
                time.sleep(0.1)
            
            print("\n[MAIN] Stopping all threads...")
            time.sleep(1)  # Give threads time to clean up
            
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
                
        # Close any open matplotlib windows
        plt.close('all')
        
        print("[CLEANUP] Done")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Microscopy Experiment Control with REST API"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable REST API server"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=5000,
        help="REST API server port (default: 5000)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-THREAD: CAMERA + STAGE + AUTOFOCUS + REST API")
    print("="*70)
    print("This will run:")
    print("  1. Camera live stream (GUI window)")
    print("  2. Stage control via CLI commands")
    print("  3. Autofocus with live plotting")
    print("  4. Interactive command prompt")
    if not args.no_api:
        print(f"  5. REST API server (port {args.api_port})")
    print("\nType 'quit' or press Ctrl+C to stop.")
    print("="*70 + "\n")
    
    app = DualThreadApp(
        enable_api=not args.no_api,
        api_port=args.api_port
    )
    app.run()
    
    print("\n" + "="*70)
    print("APPLICATION TERMINATED")
    print("="*70)


if __name__ == "__main__":
    main()