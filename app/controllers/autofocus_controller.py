"""
Autofocus Controller for GUI

Wraps AutofocusController to work with PyQt6 signals and threading.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
import time
from typing import Optional


class AutofocusWorker(QThread):
    """
    Worker thread for running autofocus scan.
    
    Emits progress updates during scan and final results.
    """
    
    # Signals
    started = pyqtSignal(str)  # axis
    progress = pyqtSignal(float, float, float)  # position (µm), metric, progress%
    complete = pyqtSignal(float, float)  # best_position (µm), best_metric
    failed = pyqtSignal(str)  # error message
    plot_data = pyqtSignal(list, list)  # positions, metrics (for live plotting)
    
    def __init__(self, camera, stage, axis='x', scan_range_um=10.0, 
                 step_um=0.5, enable_plot=False):
        """
        Initialize autofocus worker.
        
        Args:
            camera: Camera application instance
            stage: Stage adapter (µm interface)
            axis: Axis to scan ('x', 'y', 'z')
            scan_range_um: Scan range in micrometers
            step_um: Step size in micrometers
            enable_plot: Enable live plotting
        """
        super().__init__()
        self.camera = camera
        self.stage = stage
        self.axis = axis.lower()
        self.scan_range_um = scan_range_um
        self.step_um = step_um
        self.enable_plot = enable_plot
        
        self.cancelled = False
        
        # Results
        self.positions = []
        self.metrics = []
        self.best_position = None
        self.best_metric = None
    
    def cancel(self):
        """Cancel the autofocus scan."""
        self.cancelled = True
    
    def calculate_focus_metric(self, image: np.ndarray) -> float:
        """
        Calculate Variance of Laplacian (sharpness metric).
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            Focus metric (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
    
    def run(self):
        """Main autofocus scan loop."""
        try:
            self.started.emit(self.axis)
            
            # Get current position
            current_pos = self.stage.get_pos(self.axis)
            start_pos = current_pos - self.scan_range_um / 2
            end_pos = current_pos + self.scan_range_um / 2
            
            print(f"[Autofocus] Starting scan on {self.axis.upper()}-axis")
            print(f"  Range: {start_pos:.3f} to {end_pos:.3f} µm")
            print(f"  Step: {self.step_um:.3f} µm")
            
            # Clear previous results
            self.positions = []
            self.metrics = []
            
            # Scan loop
            num_steps = int((end_pos - start_pos) / self.step_um) + 1
            
            for i, pos in enumerate(np.arange(start_pos, end_pos + self.step_um/2, self.step_um)):
                if self.cancelled:
                    print("[Autofocus] Cancelled by user")
                    return
                
                # Move stage
                self.stage.move_abs(self.axis, pos)
                time.sleep(0.05)  # Small settle time
                
                # Acquire image
                try:
                    image = self.camera.acquire_image()
                except Exception as e:
                    print(f"[Autofocus] Warning: Failed to acquire image: {e}")
                    continue
                
                # Calculate focus metric
                metric = self.calculate_focus_metric(image)
                
                self.positions.append(pos)
                self.metrics.append(metric)
                
                # Calculate progress
                progress = (i + 1) / num_steps * 100
                
                # Emit progress
                self.progress.emit(pos, metric, progress)
                
                # Emit plot data periodically
                if self.enable_plot and (i % 2 == 0 or i == num_steps - 1):
                    self.plot_data.emit(self.positions.copy(), self.metrics.copy())
                
                print(f"[Autofocus] [{progress:5.1f}%] {self.axis.upper()}={pos:7.3f}µm -> focus={metric:8.2f}")
            
            # Find best position
            if len(self.metrics) == 0:
                self.failed.emit("No valid measurements collected")
                return
            
            best_idx = np.argmax(self.metrics)
            self.best_position = self.positions[best_idx]
            self.best_metric = self.metrics[best_idx]
            
            print(f"[Autofocus] Best focus at {self.axis.upper()}={self.best_position:.3f}µm (metric: {self.best_metric:.2f})")
            
            # Move to best position
            self.stage.move_abs(self.axis, self.best_position)
            time.sleep(0.1)
            
            # Emit completion
            self.complete.emit(self.best_position, self.best_metric)
            
        except Exception as e:
            print(f"[Autofocus] Error: {e}")
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class AutofocusController:
    """
    High-level autofocus controller for GUI.
    
    Manages autofocus worker thread and provides convenience methods.
    """
    
    def __init__(self, camera, stage, signals):
        """
        Initialize controller.
        
        Args:
            camera: Camera application instance
            stage: Stage adapter (µm interface)
            signals: SystemSignals instance
        """
        self.camera = camera
        self.stage = stage
        self.signals = signals
        
        self.worker = None
        self.is_running = False
        
        # Results
        self.last_axis = None
        self.last_best_position = None
        self.last_best_metric = None
        self.last_positions = []
        self.last_metrics = []
    
    def run_autofocus(self, axis='x', scan_range_um=10.0, step_um=0.5, 
                     enable_plot=False) -> bool:
        """
        Start autofocus scan.
        
        Args:
            axis: Axis to scan ('x', 'y', 'z')
            scan_range_um: Scan range in micrometers
            step_um: Step size in micrometers
            enable_plot: Enable live plotting
        
        Returns:
            True if started successfully, False if already running
        """
        if self.is_running:
            self.signals.warning_occurred.emit(
                "Autofocus Running",
                "Autofocus scan is already in progress"
            )
            return False
        
        # Validate inputs
        if axis.lower() not in ['x', 'y', 'z']:
            self.signals.error_occurred.emit(
                "Invalid Axis",
                f"Axis must be 'x', 'y', or 'z', got '{axis}'"
            )
            return False
        
        # Create worker
        self.worker = AutofocusWorker(
            camera=self.camera,
            stage=self.stage,
            axis=axis,
            scan_range_um=scan_range_um,
            step_um=step_um,
            enable_plot=enable_plot
        )
        
        # Connect signals
        self.worker.started.connect(self._on_started)
        self.worker.progress.connect(self._on_progress)
        self.worker.complete.connect(self._on_complete)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        
        if enable_plot:
            self.worker.plot_data.connect(self._on_plot_data)
        
        # Start worker
        self.worker.start()
        self.is_running = True
        
        return True
    
    def cancel(self):
        """Cancel running autofocus."""
        if self.worker is not None and self.is_running:
            self.worker.cancel()
            self.signals.autofocus_cancelled.emit()
    
    def _on_started(self, axis: str):
        """Handle autofocus start."""
        self.last_axis = axis
        self.signals.autofocus_started.emit(axis)
        self.signals.busy_started.emit(f"Autofocus ({axis.upper()}-axis)")
        self.signals.status_message.emit(f"Running autofocus on {axis.upper()}-axis...")
    
    def _on_progress(self, position: float, metric: float, progress: float):
        """Handle progress update."""
        self.signals.autofocus_progress.emit(position, metric, progress)
    
    def _on_complete(self, best_position: float, best_metric: float):
        """Handle autofocus completion."""
        self.last_best_position = best_position
        self.last_best_metric = best_metric
        
        if self.worker:
            self.last_positions = self.worker.positions.copy()
            self.last_metrics = self.worker.metrics.copy()
        
        self.signals.autofocus_complete.emit(best_position, best_metric)
        self.signals.status_message.emit(
            f"Autofocus complete: {self.last_axis.upper()}={best_position:.3f}µm (metric: {best_metric:.1f})"
        )
    
    def _on_failed(self, error: str):
        """Handle autofocus failure."""
        self.signals.autofocus_failed.emit(error)
        self.signals.error_occurred.emit("Autofocus Failed", error)
    
    def _on_finished(self):
        """Handle worker thread completion."""
        self.is_running = False
        self.signals.busy_ended.emit()
        self.worker = None
    
    def _on_plot_data(self, positions, metrics):
        """Handle plot data update (for live plotting)."""
        # Store for potential live plot widget
        pass
    
    def get_last_results(self) -> dict:
        """
        Get results from last autofocus scan.
        
        Returns:
            dict with 'axis', 'best_position', 'best_metric', 'positions', 'metrics'
        """
        return {
            'axis': self.last_axis,
            'best_position': self.last_best_position,
            'best_metric': self.last_best_metric,
            'positions': self.last_positions.copy(),
            'metrics': self.last_metrics.copy()
        }
    
    def has_results(self) -> bool:
        """Check if results are available."""
        return self.last_best_position is not None