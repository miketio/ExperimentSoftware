# app/controllers/filter_controller.py
"""
Filter Stage Controller

Manages K-space filtering sweeps with image capture.
Works with both real FilterStage and MockFilterStage.
Includes camera stream conflict resolution.
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict


class FilterSweepWorker(QThread):
    """Worker thread for filter sweep with image capture."""
    
    progress = pyqtSignal(int, int, str)  # current, total, status
    image_captured = pyqtSignal(int, float, object)  # index, position_um, image
    complete = pyqtSignal(dict)  # results
    error = pyqtSignal(str)  # error message
    
    def __init__(
        self,
        filter_stage,
        camera,
        start_um: float,
        end_um: float,
        step_um: float,
        output_dir: str,
        settle_time_s: float = 0.5
    ):
        super().__init__()
        self.filter_stage = filter_stage
        self.camera = camera
        self.start_nm = int(start_um * 1000)
        self.end_nm = int(end_um * 1000)
        self.step_nm = int(step_um * 1000)
        self.output_dir = output_dir
        self.settle_time = settle_time_s
        self.cancelled = False
    
    def cancel(self):
        """Cancel the sweep."""
        self.cancelled = True
    
    def run(self):
        """Execute sweep."""
        try:
            # Calculate positions
            positions_nm = list(range(self.start_nm, self.end_nm + self.step_nm, self.step_nm))
            total = len(positions_nm)
            
            self.progress.emit(0, total, f"Starting sweep ({total} positions)...")
            
            # Create output directory
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results = []
            failed_count = 0
            
            for idx, target_nm in enumerate(positions_nm):
                if self.cancelled:
                    self.error.emit("Sweep cancelled by user")
                    return
                
                # Move to position
                target_um = target_nm / 1000.0
                self.progress.emit(
                    idx + 1, total,
                    f"Moving to {target_um:.3f} µm..."
                )
                
                self.filter_stage.move_abs(target_nm)
                
                # Wait for settling
                import time
                time.sleep(self.settle_time)
                
                # Read back actual position
                actual_nm = self.filter_stage.get_position()
                actual_um = actual_nm / 1000.0
                
                # Capture image
                try:
                    image = self.camera.acquire_single_image()
                    
                    # Save image
                    image_filename = f"img_{idx:04d}_pos_{actual_nm}nm.tif"
                    image_path = output_path / image_filename
                    
                    import tifffile
                    tifffile.imwrite(str(image_path), image)
                    
                    results.append({
                        'index': idx,
                        'target_nm': target_nm,
                        'actual_nm': actual_nm,
                        'image_file': str(image_path)
                    })
                    
                    # Emit progress with image
                    self.image_captured.emit(idx, actual_um, image)
                    
                except Exception as e:
                    failed_count += 1
                    print(f"⚠️  Failed to capture at {actual_nm}nm: {e}")
                    # Continue with next position instead of aborting
            
            # Save metadata
            metadata = {
                'sweep_config': {
                    'start_um': self.start_nm / 1000.0,
                    'end_um': self.end_nm / 1000.0,
                    'step_um': self.step_nm / 1000.0,
                    'total_positions': total
                },
                'results': results,
                'successful_captures': len(results),
                'failed_captures': failed_count,
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            metadata_file = output_path / "sweep_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Complete
            self.complete.emit({
                'positions': len(results),
                'failed': failed_count,
                'output_dir': str(output_path),
                'metadata_file': str(metadata_file)
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class FilterController(QObject):
    """
    Controller for filter stage operations.
    
    Works with both real FilterStage and MockFilterStage.
    Handles camera stream coordination during sweeps.
    """
    
    def __init__(self, state, signals, filter_stage, camera, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.filter_stage = filter_stage
        self.camera = camera
        
        self.worker = None
        self.is_running = False
        
        # Filter stage limits (µm)
        self.limit_min = 0.0
        self.limit_max = 100.0
        
        # Detect hardware type
        hardware_type = type(filter_stage).__name__
        print(f"[FilterController] Initialized with {hardware_type}")
        
        # Initial position update
        self._update_position()
    
    def _update_position(self):
        """Update state with current filter position."""
        try:
            pos_nm = self.filter_stage.get_position()
            self.state.filter.position_nm = pos_nm
            # Emit signal if you have one for filter position
            # self.signals.filter_position_changed.emit(pos_nm)
        except Exception as e:
            print(f"[FilterController] Warning: Failed to read position: {e}")
    
    def move_to_position(self, pos_um: float) -> bool:
        """Move filter to specific position."""
        if self.is_running:
            self.signals.warning_occurred.emit(
                "Sweep Running",
                "Cannot move while sweep is in progress"
            )
            return False
        
        # Check limits
        if not (self.limit_min <= pos_um <= self.limit_max):
            self.signals.error_occurred.emit(
                "Out of Range",
                f"Position {pos_um:.3f}µm outside limits [{self.limit_min:.1f}, {self.limit_max:.1f}]µm"
            )
            return False
        
        try:
            pos_nm = int(pos_um * 1000)
            self.filter_stage.move_abs(pos_nm)
            
            actual_nm = self.filter_stage.get_position()
            actual_um = actual_nm / 1000.0
            
            self.signals.status_message.emit(
                f"Filter moved to {actual_um:.3f}µm"
            )
            
            self._update_position()
            return True
            
        except Exception as e:
            self.signals.error_occurred.emit("Move Failed", str(e))
            return False
    
    def home(self) -> bool:
        """Move filter to home position (0 µm)."""
        return self.move_to_position(0.0)
    
    def run_sweep(
        self,
        start_um: float,
        end_um: float,
        step_um: float,
        output_dir: Optional[str] = None,
        settle_time_s: float = 0.5
    ) -> bool:
        """Start filter sweep."""
        if self.is_running:
            self.signals.warning_occurred.emit(
                "Sweep Running",
                "Sweep already in progress"
            )
            return False
        
        # Validate range
        if not (self.limit_min <= start_um <= self.limit_max):
            self.signals.error_occurred.emit("Invalid Range", f"Start position out of limits")
            return False
        
        if not (self.limit_min <= end_um <= self.limit_max):
            self.signals.error_occurred.emit("Invalid Range", f"End position out of limits")
            return False
        
        if step_um <= 0:
            self.signals.error_occurred.emit("Invalid Step", "Step size must be positive")
            return False
        
        # Default output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/filter_sweep_{timestamp}"
        
        # ✅ CRITICAL: Stop camera stream to prevent conflicts
        print("[FilterController] Stopping camera stream before sweep")
        self.signals.request_stop_camera_stream.emit()
        
        # Small delay to ensure stream stops
        import time
        time.sleep(0.3)
        
        # Create worker
        self.worker = FilterSweepWorker(
            filter_stage=self.filter_stage,
            camera=self.camera,
            start_um=start_um,
            end_um=end_um,
            step_um=step_um,
            output_dir=output_dir,
            settle_time_s=settle_time_s
        )
        
        # Connect signals
        self.worker.progress.connect(self._on_progress)
        self.worker.image_captured.connect(self._on_image_captured)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)
        
        # Start
        self.worker.start()
        self.is_running = True
        
        self.signals.busy_started.emit(f"Filter Sweep ({start_um:.1f} to {end_um:.1f}µm)")
        self.signals.status_message.emit("Filter sweep started...")
        
        return True
    
    def cancel_sweep(self):
        """Cancel running sweep."""
        if self.worker and self.is_running:
            print("[FilterController] Cancelling sweep")
            self.worker.cancel()
    
    def _on_progress(self, current: int, total: int, status: str):
        """Handle progress update."""
        self.signals.status_message.emit(f"[{current}/{total}] {status}")
    
    def _on_image_captured(self, index: int, position_um: float, image):
        """Handle image capture."""
        # Could emit signal for live preview if needed
        pass
    
    def _on_complete(self, results: dict):
        """Handle sweep completion."""
        self.is_running = False
        
        # ✅ CRITICAL: Restart camera stream after sweep
        print("[FilterController] Restarting camera stream after sweep")
        self.signals.request_start_camera_stream.emit()
        
        self.signals.busy_ended.emit()
        
        # Update position
        self._update_position()
        
        # Show results
        if results['failed'] > 0:
            self.signals.warning_occurred.emit(
                "Sweep Complete with Errors",
                f"✅ {results['positions']} images saved\n"
                f"⚠️  {results['failed']} captures failed\n\n"
                f"Output: {results['output_dir']}"
            )
        else:
            self.signals.status_message.emit(
                f"✅ Sweep complete! {results['positions']} images saved to {results['output_dir']}"
            )
        
        print(f"[FilterController] Sweep saved to: {results['output_dir']}")
    
    def _on_error(self, error: str):
        """Handle error."""
        self.is_running = False
        
        # ✅ CRITICAL: Restart camera stream even on error
        print("[FilterController] Restarting camera stream after error")
        self.signals.request_start_camera_stream.emit()
        
        self.signals.busy_ended.emit()
        self.signals.error_occurred.emit("Sweep Failed", error)
    
    def get_status(self) -> dict:
        """Get filter stage status."""
        try:
            return self.filter_stage.get_status()
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def print_status(self):
        """Print filter stage status to console."""
        try:
            self.filter_stage.print_status()
        except Exception as e:
            print(f"[FilterController] Error getting status: {e}")