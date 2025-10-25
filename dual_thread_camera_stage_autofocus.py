# dual_thread_camera_stage_autofocus.py
"""
Dual-threaded application that runs:
- Thread 1: Camera live stream (GUI)
- Thread 2: Stage control via CLI commands (including autofocus)
- Thread 3: CLI input handler

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
from matplotlib.animation import FuncAnimation
from CameraControl.zylaCamera import ZylaCamera
from CameraControl.andorCameraApp import AndorCameraApp
from SetupMotor.smartactStage import SmarActXYZStage
from SetupMotor.xyzStageApp import XYZStageApp
from stage_commands import StageCommandProcessor

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
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
            
    def run_autofocus(self, axis='x', scan_range=None, step_size=None, enable_plot=False):
        """
        Run autofocus scan on specified axis.
        
        Args:
            axis: 'x', 'y', or 'z'
            scan_range: total range to scan in nm (default: self.default_range)
            step_size: step size in nm (default: self.default_step)
            enable_plot: whether to show live plot
        """
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
            # Get current position
            current_pos = self.stage_app.get_pos(self.axis)
            start_pos = current_pos - scan_range // 2
            end_pos = current_pos + scan_range // 2
            
            print(f"\n[AUTOFOCUS] ========================================")
            print(f"[AUTOFOCUS] Starting autofocus on {self.axis.upper()}-axis")
            print(f"[AUTOFOCUS] Range: {start_pos}nm to {end_pos}nm")
            print(f"[AUTOFOCUS] Step: {step_size}nm")
            print(f"[AUTOFOCUS] Current position: {current_pos}nm")
            print(f"[AUTOFOCUS] ========================================\n")
            
            # # Setup live plot
            # if self.plot_enabled:
            #     self.setup_live_plot()

            print("[AUTOFOCUS] Stopping camera stream...")
            self.camera_app.camera.stop_streaming()
            time.sleep(0.1)  # Allow SDK to settle

            # Scan loop
            num_steps = int((end_pos - start_pos) / step_size) + 1
            for i, pos in enumerate(range(start_pos, end_pos + 1, step_size)):
                # Move to position
                self.stage_app.move_abs(self.axis, pos)
                time.sleep(0.1)  # Wait for stage to settle
                
                # Acquire image
                self.camera_app.set_roi(400, 1200, 1300, 1000)
                image = self.camera_app.acquire_image()
                
                # Calculate focus metric
                metric = self.calculate_focus_metric(image)
                
                # Store results
                self.positions.append(pos)
                self.metrics.append(metric)
                
                # Print progress
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
            
            # Move to best position
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
    """Manages camera stream and stage control in separate threads."""
    
    def __init__(self):
        self.camera = None
        self.camera_app = None
        self.stage = None
        self.stage_app = None
        self.autofocus = None
        
        self.camera_thread = None
        self.stage_thread = None
        self.input_thread = None
        
        self.stop_event = threading.Event()
        self.command_queue = []
        self.queue_lock = threading.Lock()     
        
    def initialize_camera(self):
        """Initialize camera and app in the main thread."""
        print("[INIT] Initializing camera...")
        self.camera = ZylaCamera()
        self.camera.connect()
        self.camera_app = AndorCameraApp(self.camera)

        
        # Configure camera settings
        self.camera_app.set_gain_mode("16-bit (low noise & high well capacity)")
        self.camera_app.set_exposure(0.02)
        print("[INIT] Camera initialized successfully")
        
    def initialize_stage(self):
        """Initialize stage and app in the main thread."""
        print("[INIT] Initializing stage...")
        self.stage = SmarActXYZStage()
        self.stage_app = XYZStageApp(self.stage)
        
        # Initialize autofocus controller
        self.autofocus = AutofocusController(self.camera_app, self.stage_app)
        
        # Initialize the external command processor (keeps things tidy)
        self.stage_cmd_processor = StageCommandProcessor(self.stage_app, 
                                                         self.camera_app, 
                                                         self.autofocus)

        print("[INIT] Stage initialized successfully")
        
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
                
                # Render frame
                combined = self.camera_app._render_frame(
                    image_data, 
                    max_width=1550, 
                    max_height=800
                )
                
                cv2.imshow(window_title, combined)
                
                # Use very short waitKey to keep responsive
                key = cv2.waitKey(30) & 0xFF
                # Ignore 'q' key, only respond to stop_event
                
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
                # Check for commands
                command = None
                with self.queue_lock:
                    if self.command_queue:
                        command = self.command_queue.pop(0)
                
                if command:
                    self.stage_cmd_processor.process(command)
                else:
                    time.sleep(0.1)  # No commands, wait a bit
                    
        except Exception as e:
            print(f"[STAGE] Error: {e}")
        finally:
            print("[STAGE] Control stopped")
       
            
    def print_help(self):
        """Print available commands."""
        # We now delegate to the command processor's help output for consistency
        if self.stage_cmd_processor:
            self.stage_cmd_processor._print_help()
        else:
            # Fallback help (very terse)
            print("Type 'help' for command list (stage command processor not initialized yet)")
                
    def input_thread_func(self):
        """Thread 3: Handle CLI input."""
        print("[INPUT] CLI ready. Type 'help' for commands.")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Non-blocking input with timeout
                    user_input = input(">> ").strip()
                    
                    if not user_input:
                        continue
                        
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("[INPUT] Stopping application...")
                        self.stop_event.set()
                        break
                        
                    # Add command to queue
                    with self.queue_lock:
                        self.command_queue.append(user_input)
                        
                except EOFError:
                    # Handle Ctrl+D
                    print("\n[INPUT] EOF received, stopping...")
                    self.stop_event.set()
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print("\n[INPUT] Interrupted, stopping...")
                    self.stop_event.set()
                    break
                    
        except Exception as e:
            print(f"[INPUT] Error: {e}")
        finally:
            print("[INPUT] Input handler stopped")
            
    def run(self):
        """Main execution: start all threads."""
        try:
            # Initialize hardware
            self.initialize_camera()
            self.initialize_stage()
            
            print("\n" + "="*70)
            print("DUAL THREAD APPLICATION STARTING")
            print("="*70)
            print("Thread 1: Camera live stream")
            print("Thread 2: Stage control (processes commands)")
            print("Thread 3: CLI input handler")
            print("="*70)
            self.print_help()
            
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
            
            # Start all threads
            self.camera_thread.start()
            time.sleep(0.5)  # Let camera window open first
            self.stage_thread.start()
            self.input_thread.start()
            
            # Wait for stop signal (from input thread or Ctrl+C)
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
    print("="*70)
    print("DUAL THREAD: CAMERA STREAM + STAGE CONTROL + AUTOFOCUS")
    print("="*70)
    print("This will run:")
    print("  1. Camera live stream (GUI window)")
    print("  2. Stage control via CLI commands")
    print("  3. Autofocus with live plotting")
    print("  4. Interactive command prompt")
    print("\nType 'quit' or press Ctrl+C to stop.")
    print("="*70 + "\n")
    
    app = DualThreadApp()
    app.run()
    
    print("\n" + "="*70)
    print("APPLICATION TERMINATED")
    print("="*70)


if __name__ == "__main__":
    main()