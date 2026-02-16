# app/controllers/filter_controller.py
"""
Filter Stage Controller - WITH CAMERA CONFIG PRESERVATION

CRITICAL FIXES:
- Saves camera settings before sweep
- Restores settings before EACH image capture
- Stores camera mode (Fourier/Real space) in metadata
- Ensures consistent imaging throughout sweep
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict


class FilterSweepWorker(QThread):
    """Worker thread for filter sweep with CAMERA CONFIG PRESERVATION."""
    
    progress = pyqtSignal(int, int, str)  # current, total, status
    image_captured = pyqtSignal(int, float, object)  # index, position_um, image
    complete = pyqtSignal(dict)  # results
    error = pyqtSignal(str)  # error message
    
    def __init__(
        self,
        filter_stage,
        camera,
        camera_thread,  # NEW: Need camera thread reference
        start_um: float,
        end_um: float,
        step_um: float,
        output_dir: str,
        settle_time_s: float = 0.5,
    ):
        super().__init__()
        self.filter_stage = filter_stage
        self.camera = camera
        self.camera_thread = camera_thread
        self.start_nm = int(start_um * 1000)
        self.end_nm = int(end_um * 1000)
        self.step_nm = int(step_um * 1000)
        self.output_dir = output_dir
        self.settle_time = settle_time_s
        self.cancelled = False
        
        # ✅ NEW: Store camera configuration
        self.camera_config = {}
    
    def cancel(self):
        """Cancel the sweep."""
        self.cancelled = True
    
    def _capture_camera_config(self):
        """
        Capture current camera configuration to preserve during sweep.
        
        Returns dict with all relevant settings.
        """
        config = {}
        
        # Exposure time
        if hasattr(self.camera, 'get_exposure_time'):
            try:
                config['exposure_time_s'] = self.camera.get_exposure_time()
                print(f"[FilterSweep] 💾 Saved exposure: {config['exposure_time_s']:.4f}s")
            except Exception as e:
                print(f"[FilterSweep] ⚠️  Could not read exposure: {e}")
        
        # ROI
        if hasattr(self.camera, 'roi'):
            config['roi'] = self.camera.roi
            if config['roi']:
                print(f"[FilterSweep] 💾 Saved ROI: {config['roi']}")
            else:
                print(f"[FilterSweep] 💾 ROI: Full sensor")
        
        # Sensor size
        if hasattr(self.camera, 'get_sensor_size'):
            try:
                config['sensor_size'] = self.camera.get_sensor_size()
            except:
                pass
        
        # Pixel scale
        if hasattr(self.camera, 'um_per_pixel'):
            config['um_per_pixel'] = self.camera.um_per_pixel
        
        # Bit depth mode (if available)
        if hasattr(self.camera, 'bit_depth_mode'):
            config['bit_depth_mode'] = self.camera.bit_depth_mode
        
        # ✅ CRITICAL: Fourier mode state
        if self.camera_thread and hasattr(self.camera_thread, 'color_manager'):
            config['fourier_mode'] = self.camera_thread.color_manager.fourier_mode
            print(f"[FilterSweep] 💾 Fourier mode: {config['fourier_mode']}")
        
        return config
    
    def _apply_camera_config(self, config: dict):
        """
        Apply camera configuration before image capture.
        
        This ensures EVERY image uses the same settings.
        """
        # Exposure time
        if 'exposure_time_s' in config:
            try:
                if hasattr(self.camera, 'set_exposure_time'):
                    self.camera.set_exposure_time(config['exposure_time_s'])
            except Exception as e:
                print(f"[FilterSweep] ⚠️  Could not set exposure: {e}")
        
        # ROI (if changed during sweep)
        if 'roi' in config:
            try:
                if hasattr(self.camera, 'set_roi'):
                    roi = config['roi']
                    if roi:
                        self.camera.set_roi(*roi)
                    else:
                        # Reset to full sensor
                        sensor_w, sensor_h = config.get('sensor_size', (2048, 2048))
                        self.camera.set_roi(0, 0, sensor_w, sensor_h)
            except Exception as e:
                print(f"[FilterSweep] ⚠️  Could not set ROI: {e}")
    
    def run(self):
        """Execute sweep with camera config preservation."""
        try:
            # ✅ STEP 1: Capture camera configuration
            self.progress.emit(0, 0, "Saving camera configuration...")
            self.camera_config = self._capture_camera_config()
            
            if not self.camera_config:
                self.error.emit("Failed to read camera configuration")
                return
            
            # Calculate positions
            positions_nm = list(range(self.start_nm, self.end_nm + self.step_nm, self.step_nm))
            total = len(positions_nm)
            
            self.progress.emit(0, total, f"Starting sweep ({total} positions)...")
            
            # Create output directory
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results = []
            failed_count = 0
            max_intensity_idx = -1
            max_intensity = 0
            
            # ✅ STEP 2: Loop with config restoration
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
                
                # ✅ STEP 3: Restore camera config before capture
                self._apply_camera_config(self.camera_config)
                time.sleep(0.05)  # Brief settle
                
                # Capture image
                try:
                    image = self.camera.acquire_single_image()
                    
                    # Track brightest image
                    mean_intensity = float(np.mean(image))
                    if mean_intensity > max_intensity:
                        max_intensity = mean_intensity
                        max_intensity_idx = idx
                    
                    # Save image
                    image_filename = f"img_{idx:04d}_pos_{actual_nm}nm.tif"
                    image_path = output_path / image_filename
                    
                    import tifffile
                    tifffile.imwrite(str(image_path), image)
                    
                    results.append({
                        'index': idx,
                        'target_nm': target_nm,
                        'actual_nm': actual_nm,
                        'image_file': str(image_path),
                        'mean_intensity': mean_intensity,
                        'min_intensity': float(np.min(image)),
                        'max_intensity': float(np.max(image))
                    })
                    
                    # Emit progress with image
                    self.image_captured.emit(idx, actual_um, image)
                    
                except Exception as e:
                    failed_count += 1
                    print(f"⚠️ Failed to capture at {actual_nm}nm: {e}")
                    
                    results.append({
                        'index': idx,
                        'target_nm': target_nm,
                        'actual_nm': actual_nm,
                        'image_file': None,
                        'error': str(e)
                    })
            
            # ✅ STEP 4: Save enhanced metadata
            metadata = {
                'sweep_config': {
                    'start_um': self.start_nm / 1000.0,
                    'end_um': self.end_nm / 1000.0,
                    'step_um': self.step_nm / 1000.0,
                    'total_positions': total,
                    'settle_time_s': self.settle_time
                },
                'camera_config': self.camera_config,  # ✅ NEW: Full camera settings
                'acquisition_mode': 'fourier_space' if self.camera_config.get('fourier_mode', False) else 'real_space',
                'results': results,
                'successful_captures': len([r for r in results if 'error' not in r]),
                'failed_captures': failed_count,
                'brightest_image_index': max_intensity_idx,
                'intensity_range': {
                    'min': float(np.min([r.get('min_intensity', 0) for r in results if 'min_intensity' in r])),
                    'max': float(np.max([r.get('max_intensity', 0) for r in results if 'max_intensity' in r])),
                    'mean_of_means': float(np.mean([r.get('mean_intensity', 0) for r in results if 'mean_intensity' in r]))
                },
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
                'metadata_file': str(metadata_file),
                'brightest_idx': max_intensity_idx
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class FilterController(QObject):
    """
    Controller for filter stage operations.
    
    NOW WITH: Camera configuration preservation during sweeps
    """
    
    def __init__(self, state, signals, filter_stage, camera, parent=None):
        super().__init__(parent)
        self.state = state
        self.signals = signals
        self.filter_stage = filter_stage
        self.camera = camera
        self.camera_thread = None  # ✅ NEW: Will be set by main window
        
        self.worker = None
        self.is_running = False
        
        # Extended limits to ±15mm
        self.limit_min = -15000.0  # µm
        self.limit_max = 15000.0   # µm
        
        hardware_type = type(filter_stage).__name__
        print(f"[FilterController] Initialized with {hardware_type}")
        print(f"[FilterController] Range limits: {self.limit_min/1000:.1f} to {self.limit_max/1000:.1f} mm")
        
        self._update_position()
    
    def set_camera_thread(self, camera_thread):
        """
        ✅ NEW: Set camera thread reference for config preservation.
        
        Call this from main window after camera thread is created.
        """
        self.camera_thread = camera_thread
        print("[FilterController] Camera thread reference set")
    
    def _update_position(self):
        """Update state with current filter position."""
        try:
            pos_nm = self.filter_stage.get_position()
            self.state.filter.position_nm = pos_nm
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
        
        if not (self.limit_min <= pos_um <= self.limit_max):
            self.signals.error_occurred.emit(
                "Out of Range",
                f"Position {pos_um:.3f}µm outside limits [{self.limit_min:.1f}, {self.limit_max:.1f}]µm"
            )
            return False
        
        try:
            pos_nm = int(pos_um * 1000)
            print(f"[FilterController] Moving to {pos_um:.3f} µm ({pos_nm} nm)")
            
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
            import traceback
            traceback.print_exc()
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
        """
        Start filter sweep WITH CAMERA CONFIG PRESERVATION.
        
        ✅ NEW: Automatically saves and restores camera settings
        """
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
        
        # ✅ CHECK: Camera thread must be set
        if self.camera_thread is None:
            print("[FilterController] ⚠️  Warning: Camera thread not set, config preservation may fail")
        
        # Default output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/filter_sweep_{timestamp}"
        
        # Stop camera stream to prevent conflicts
        print("[FilterController] Stopping camera stream before sweep")
        self.signals.request_stop_camera_stream.emit()
        
        # Small delay to ensure stream stops
        import time
        time.sleep(0.3)
        
        # ✅ Create worker WITH camera thread reference
        self.worker = FilterSweepWorker(
            filter_stage=self.filter_stage,
            camera=self.camera,
            camera_thread=self.camera_thread,  # ✅ NEW: Pass camera thread
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
        pass
    
    def _on_complete(self, results: dict):
        """Handle sweep completion."""
        self.is_running = False
        
        # Restart camera stream after sweep
        print("[FilterController] Restarting camera stream after sweep")
        self.signals.request_start_camera_stream.emit()
        
        self.signals.busy_ended.emit()
        
        # Update position
        self._update_position()
        
        # Show results with brightest image info
        brightest_idx = results.get('brightest_idx', -1)
        brightest_msg = f"\n📸 Brightest image: #{brightest_idx}" if brightest_idx >= 0 else ""
        
        if results['failed'] > 0:
            self.signals.warning_occurred.emit(
                "Sweep Complete with Errors",
                f"✅ {results['positions']} images saved\n"
                f"⚠️ {results['failed']} captures failed\n"
                f"{brightest_msg}\n\n"
                f"Output: {results['output_dir']}"
            )
        else:
            self.signals.status_message.emit(
                f"✅ Sweep complete! {results['positions']} images saved to {results['output_dir']}"
                f"{brightest_msg}"
            )
        
        print(f"[FilterController] Sweep saved to: {results['output_dir']}")
    
    def _on_error(self, error: str):
        """Handle error."""
        self.is_running = False
        
        # Restart camera stream even on error
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