import time
import numpy as np
import cv2

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
                
                # self.camera_app.set_roi(400, 1200, 1300, 1000)
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

