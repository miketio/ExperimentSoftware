# app/controllers/filter_controller.py
"""
Filter Stage Controller - WITH MULTI-POSITION SWEEP

NEW FEATURES:
- Multi-position sweep capability
- Automatic stage navigation between positions
- Subfolder organization by position name
- Progress tracking for position + sweep

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
from typing import Optional, Dict, List
import time


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
        camera_thread,
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
        
        # Store camera configuration
        self.camera_config = {}
    
    def cancel(self):
        """Cancel the sweep."""
        self.cancelled = True
    
    def _capture_camera_config(self):
        """Capture current camera configuration to preserve during sweep."""
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
        
        # Fourier mode state
        if self.camera_thread and hasattr(self.camera_thread, 'color_manager'):
            config['fourier_mode'] = self.camera_thread.color_manager.fourier_mode
            print(f"[FilterSweep] 💾 Fourier mode: {config['fourier_mode']}")
        
        return config
    
    def run(self):
        """Execute sweep: stop stream once, snap each image, restart stream once."""
        try:
            # Capture camera configuration
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

            # Set exposure once before the loop — streaming is already stopped
            # by FilterController.run_sweep() before this worker starts.
            if 'exposure_time_s' in self.camera_config:
                if hasattr(self.camera, 'set_exposure_time'):
                    self.camera.set_exposure_time(self.camera_config['exposure_time_s'])
                    print(f"[FilterSweep] ✅ Exposure set to {self.camera_config['exposure_time_s']:.4f}s")

            results = []
            failed_count = 0
            max_intensity_idx = -1
            max_intensity = 0

            import tifffile
            import json

            for idx, target_nm in enumerate(positions_nm):
                if self.cancelled:
                    self.error.emit("Sweep cancelled by user")
                    return

                target_um = target_nm / 1000.0
                self.progress.emit(idx + 1, total, f"Moving to {target_um:.3f} µm...")

                self.filter_stage.move_abs(target_nm)
                time.sleep(self.settle_time)

                actual_nm = self.filter_stage.get_position()
                actual_um = actual_nm / 1000.0

                try:
                    image = self.camera.acquire_single_image()

                    mean_intensity = float(np.mean(image))
                    if mean_intensity > max_intensity:
                        max_intensity = mean_intensity
                        max_intensity_idx = idx

                    image_filename = f"img_{idx:04d}_pos_{actual_nm}nm.tif"
                    image_path = output_path / image_filename
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

                    self.image_captured.emit(idx, actual_um, image)

                except Exception as e:
                    failed_count += 1
                    print(f"[FilterSweep] ⚠️  Failed to capture at {actual_nm}nm: {e}")

                self.progress.emit(idx + 1, total, f"Captured {idx+1}/{total}")

            # Build intensity stats across all images
            all_means = [r['mean_intensity'] for r in results]
            all_mins  = [r['min_intensity']  for r in results]
            all_maxs  = [r['max_intensity']  for r in results]
            acquisition_mode = 'fourier_space' if self.camera_config.get('fourier_mode') else 'real_space'

            # Save metadata
            metadata_file = output_path / "sweep_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'sweep_config': {
                        'start_um':        self.start_nm / 1000.0,
                        'end_um':          self.end_nm   / 1000.0,
                        'step_um':         self.step_nm  / 1000.0,
                        'total_positions': total,
                        'settle_time_s':   self.settle_time
                    },
                    'camera_config':        self.camera_config,
                    'acquisition_mode':     acquisition_mode,
                    'results':              results,
                    'successful_captures':  len(results),
                    'failed_captures':      failed_count,
                    'brightest_image_index': max_intensity_idx,
                    'intensity_range': {
                        'min':          min(all_mins)  if all_mins  else 0,
                        'max':          max(all_maxs)  if all_maxs  else 0,
                        'mean_of_means': sum(all_means) / len(all_means) if all_means else 0
                    },
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            self.complete.emit({
                'output_dir':   str(output_path),
                'positions':    len(results),
                'failed':       failed_count,
                'brightest_idx': max_intensity_idx,
                'metadata_file': str(metadata_file)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Sweep failed: {e}")


class MultiPositionSweepWorker(QThread):
    """✅ NEW: Worker thread for multi-position sweep."""
    
    # Progress: (pos_idx, pos_total, sweep_idx, sweep_total, status)
    progress = pyqtSignal(int, int, int, int, str)
    position_complete = pyqtSignal(str, dict)  # position_name, results
    complete = pyqtSignal(dict)  # final results
    error = pyqtSignal(str)
    
    def __init__(
        self,
        filter_stage,
        camera,
        camera_thread,
        stage,  # XYZ stage for navigation
        positions: List[Dict],  # List of {name, x, y, z}
        start_um: float,
        end_um: float,
        step_um: float,
        base_output_dir: str,
        settle_time_s: float = 0.5,
    ):
        super().__init__()
        self.filter_stage = filter_stage
        self.camera = camera
        self.camera_thread = camera_thread
        self.stage = stage
        self.positions = positions
        self.start_nm = int(start_um * 1000)
        self.end_nm = int(end_um * 1000)
        self.step_nm = int(step_um * 1000)
        self.base_output_dir = base_output_dir
        self.settle_time = settle_time_s
        self.cancelled = False
        self.camera_config = {}
    
    def cancel(self):
        """Cancel the sweep."""
        self.cancelled = True
    
    def _capture_camera_config(self):
        """Capture camera configuration."""
        config = {}
        
        if hasattr(self.camera, 'get_exposure_time'):
            try:
                config['exposure_time_s'] = self.camera.get_exposure_time()
            except:
                pass
        
        if hasattr(self.camera, 'roi'):
            config['roi'] = self.camera.roi
        
        if hasattr(self.camera, 'um_per_pixel'):
            config['um_per_pixel'] = self.camera.um_per_pixel
        
        if self.camera_thread and hasattr(self.camera_thread, 'color_manager'):
            config['fourier_mode'] = self.camera_thread.color_manager.fourier_mode
        
        return config
    
    def run(self):
        """Execute multi-position sweep: stop stream once, snap all images, restart once."""
        try:
            total_positions = len(self.positions)
            all_results = {}

            # Capture camera config once
            self.progress.emit(0, total_positions, 0, 0, "Saving camera configuration...")
            self.camera_config = self._capture_camera_config()

            # Set exposure once — streaming already stopped before this worker starts
            if 'exposure_time_s' in self.camera_config:
                if hasattr(self.camera, 'set_exposure_time'):
                    self.camera.set_exposure_time(self.camera_config['exposure_time_s'])
                    print(f"[MultiPositionSweep] ✅ Exposure set to {self.camera_config['exposure_time_s']:.4f}s")

            import tifffile
            import json

            for pos_idx, pos_data in enumerate(self.positions):
                if self.cancelled:
                    self.error.emit("Multi-position sweep cancelled by user")
                    return

                pos_name = pos_data['name']
                pos_x = pos_data['x']
                pos_y = pos_data['y']
                pos_z = pos_data['z']

                output_dir = Path(self.base_output_dir) / pos_name
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n[MultiPositionSweep] ====== Position {pos_idx+1}/{total_positions}: {pos_name} ======")

                self.progress.emit(pos_idx + 1, total_positions, 0, 0, f"Moving to {pos_name}...")

                try:
                    print(f"[MultiPositionSweep] Moving stage to X={pos_x:.3f}, Y={pos_y:.3f}, Z={pos_z:.3f} µm")
                    self.stage.move_abs('y', pos_y)
                    self.stage.move_abs('x', pos_x)
                    self.stage.move_abs('z', pos_z)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[MultiPositionSweep] ⚠️  Failed to move to {pos_name}: {e}")
                    all_results[pos_name] = {'error': str(e)}
                    continue

                self.progress.emit(pos_idx + 1, total_positions, 0, 0, f"Running sweep at {pos_name}...")

                try:
                    positions_nm = list(range(self.start_nm, self.end_nm + self.step_nm, self.step_nm))
                    total_sweep = len(positions_nm)

                    results = []
                    failed_count = 0
                    max_intensity_idx = -1
                    max_intensity = 0

                    for sweep_idx, target_nm in enumerate(positions_nm):
                        if self.cancelled:
                            self.error.emit("Multi-position sweep cancelled")
                            return

                        target_um = target_nm / 1000.0
                        self.progress.emit(
                            pos_idx + 1, total_positions,
                            sweep_idx + 1, total_sweep,
                            f"{pos_name}: {target_um:.1f}µm ({sweep_idx+1}/{total_sweep})"
                        )

                        self.filter_stage.move_abs(target_nm)
                        time.sleep(self.settle_time)

                        actual_nm = self.filter_stage.get_position()

                        try:
                            image = self.camera.acquire_single_image()

                            mean_intensity = float(np.mean(image))
                            if mean_intensity > max_intensity:
                                max_intensity = mean_intensity
                                max_intensity_idx = sweep_idx

                            image_filename = f"img_{sweep_idx:04d}_pos_{actual_nm}nm.tif"
                            image_path = output_dir / image_filename
                            tifffile.imwrite(str(image_path), image)

                            results.append({
                                'index':          sweep_idx,
                                'target_nm':      target_nm,
                                'actual_nm':      actual_nm,
                                'image_file':     str(image_path),
                                'mean_intensity': mean_intensity,
                                'min_intensity':  float(np.min(image)),
                                'max_intensity':  float(np.max(image))
                            })

                        except Exception as e:
                            failed_count += 1
                            print(f"[MultiPositionSweep] ⚠️  Failed capture: {e}")

                    # Intensity stats
                    all_means = [r['mean_intensity'] for r in results]
                    all_mins  = [r['min_intensity']  for r in results]
                    all_maxs  = [r['max_intensity']  for r in results]
                    acquisition_mode = 'fourier_space' if self.camera_config.get('fourier_mode') else 'real_space'

                    # Save metadata
                    metadata_file = output_dir / "sweep_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump({
                            'position': {
                                'name': pos_name,
                                'x_um': pos_x,
                                'y_um': pos_y,
                                'z_um': pos_z
                            },
                            'sweep_config': {
                                'start_um':        self.start_nm / 1000.0,
                                'end_um':          self.end_nm   / 1000.0,
                                'step_um':         self.step_nm  / 1000.0,
                                'total_positions': total_sweep,
                                'settle_time_s':   self.settle_time
                            },
                            'camera_config':         self.camera_config,
                            'acquisition_mode':      acquisition_mode,
                            'results':               results,
                            'successful_captures':   len(results),
                            'failed_captures':       failed_count,
                            'brightest_image_index': max_intensity_idx,
                            'intensity_range': {
                                'min':           min(all_mins)  if all_mins  else 0,
                                'max':           max(all_maxs)  if all_maxs  else 0,
                                'mean_of_means': sum(all_means) / len(all_means) if all_means else 0
                            },
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)

                    position_results = {
                        'output_dir':      str(output_dir),
                        'images_captured': len(results),
                        'images_failed':   failed_count,
                        'brightest_idx':   max_intensity_idx,
                        'metadata_file':   str(metadata_file)
                    }
                    all_results[pos_name] = position_results
                    self.position_complete.emit(pos_name, position_results)
                    print(f"[MultiPositionSweep] ✅ {pos_name}: {len(results)} images saved")

                except Exception as e:
                    print(f"[MultiPositionSweep] ❌ Sweep failed at {pos_name}: {e}")
                    all_results[pos_name] = {'error': str(e)}

            total_images = sum(r.get('images_captured', 0) for r in all_results.values())
            successful_positions = sum(1 for r in all_results.values() if 'error' not in r)

            self.complete.emit({
                'base_output_dir':       self.base_output_dir,
                'total_positions':       total_positions,
                'successful_positions':  successful_positions,
                'total_images':          total_images,
                'position_results':      all_results
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Multi-position sweep failed: {e}")


class FilterController(QObject):
    """Controller for filter stage operations."""
    
    def __init__(self, state, signals, filter_stage, camera, stage):
        super().__init__()
        self.state = state
        self.signals = signals
        self.filter_stage = filter_stage
        self.camera = camera
        self.stage = stage  # ✅ NEW: XYZ stage reference for multi-position
        self.camera_thread = None
        
        self.worker = None
        self.multi_worker = None  # ✅ NEW: Multi-position worker
        self.is_running = False
        
        # Position limits (±15mm)
        self.limit_min = -15000.0  # µm
        self.limit_max = 15000.0   # µm
        
        self._update_position()
    
    def set_camera_thread(self, camera_thread):
        """Set camera thread reference for config preservation."""
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
        """Start filter sweep at current position."""
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
        
        # Stop camera stream
        print("[FilterController] Stopping camera stream before sweep")
        self.signals.request_stop_camera_stream.emit()
        time.sleep(0.3)
        
        # Create worker
        self.worker = FilterSweepWorker(
            filter_stage=self.filter_stage,
            camera=self.camera,
            camera_thread=self.camera_thread,
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
    
    def run_multi_position_sweep(
        self,
        positions: List[Dict],
        start_um: float,
        end_um: float,
        step_um: float,
        output_dir: Optional[str] = None,
        settle_time_s: float = 0.5
    ) -> bool:
        """
        ✅ NEW: Run filter sweep at multiple XYZ positions.
        
        Args:
            positions: List of dicts with keys: name, x, y, z
            start_um: Filter sweep start (µm)
            end_um: Filter sweep end (µm)
            step_um: Filter sweep step (µm)
            output_dir: Base output directory (subfolders created per position)
            settle_time_s: Settle time per filter position
        """
        if self.is_running:
            self.signals.warning_occurred.emit(
                "Sweep Running",
                "Sweep already in progress"
            )
            return False
        
        if len(positions) == 0:
            self.signals.error_occurred.emit("No Positions", "No positions provided")
            return False
        
        # Validate range
        if not (self.limit_min <= start_um <= self.limit_max):
            self.signals.error_occurred.emit("Invalid Range", "Start position out of limits")
            return False
        
        if not (self.limit_min <= end_um <= self.limit_max):
            self.signals.error_occurred.emit("Invalid Range", "End position out of limits")
            return False
        
        if step_um <= 0:
            self.signals.error_occurred.emit("Invalid Step", "Step size must be positive")
            return False
        
        # Default output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/multi_position_sweep_{timestamp}"
        
        # Stop camera stream
        print("[FilterController] Stopping camera stream before multi-position sweep")
        self.signals.request_stop_camera_stream.emit()
        time.sleep(0.3)
        
        # Create multi-position worker
        self.multi_worker = MultiPositionSweepWorker(
            filter_stage=self.filter_stage,
            camera=self.camera,
            camera_thread=self.camera_thread,
            stage=self.stage,
            positions=positions,
            start_um=start_um,
            end_um=end_um,
            step_um=step_um,
            base_output_dir=output_dir,
            settle_time_s=settle_time_s
        )
        
        # Connect signals
        self.multi_worker.progress.connect(self._on_multi_progress)
        self.multi_worker.position_complete.connect(self._on_position_complete)
        self.multi_worker.complete.connect(self._on_multi_complete)
        self.multi_worker.error.connect(self._on_error)
        
        # Start
        self.multi_worker.start()
        self.is_running = True
        
        self.signals.busy_started.emit(f"Multi-Position Sweep ({len(positions)} positions)")
        self.signals.status_message.emit(f"Starting multi-position sweep at {len(positions)} positions...")
        
        return True
    
    def cancel_sweep(self):
        """Cancel running sweep."""
        if self.worker and self.is_running:
            print("[FilterController] Cancelling single sweep")
            self.worker.cancel()
        elif self.multi_worker and self.is_running:
            print("[FilterController] Cancelling multi-position sweep")
            self.multi_worker.cancel()
    
    def _on_progress(self, current: int, total: int, status: str):
        """Handle single sweep progress."""
        self.signals.status_message.emit(f"[{current}/{total}] {status}")
    
    def _on_multi_progress(self, pos_idx: int, pos_total: int, sweep_idx: int, sweep_total: int, status: str):
        """✅ NEW: Handle multi-position sweep progress."""
        if sweep_total > 0:
            msg = f"[Position {pos_idx}/{pos_total}] [Image {sweep_idx}/{sweep_total}] {status}"
        else:
            msg = f"[Position {pos_idx}/{pos_total}] {status}"
        self.signals.status_message.emit(msg)
    
    def _on_position_complete(self, position_name: str, results: dict):
        """✅ NEW: Handle individual position completion."""
        if 'error' in results:
            print(f"[FilterController] ⚠️  Position {position_name} failed: {results['error']}")
        else:
            print(f"[FilterController] ✅ Position {position_name}: {results['images_captured']} images")
    
    def _on_image_captured(self, index: int, position_um: float, image):
        """Handle image capture."""
        pass
    
    def _on_complete(self, results: dict):
        """Handle single sweep completion."""
        self.is_running = False
        
        # Restart camera stream
        print("[FilterController] Restarting camera stream after sweep")
        self.signals.request_start_camera_stream.emit()
        
        self.signals.busy_ended.emit()
        self._update_position()
        
        # Show results
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
    
    def _on_multi_complete(self, results: dict):
        """✅ NEW: Handle multi-position sweep completion."""
        self.is_running = False
        
        # Restart camera stream
        print("[FilterController] Restarting camera stream after multi-position sweep")
        self.signals.request_start_camera_stream.emit()
        
        self.signals.busy_ended.emit()
        self._update_position()
        
        # Show summary
        total_pos = results['total_positions']
        success_pos = results['successful_positions']
        total_imgs = results['total_images']
        
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            None,
            "Multi-Position Sweep Complete",
            f"✅ Multi-Position Sweep Complete!\n\n"
            f"Positions processed: {success_pos}/{total_pos}\n"
            f"Total images captured: {total_imgs}\n\n"
            f"Output directory:\n{results['base_output_dir']}\n\n"
            f"Each position has its own subfolder."
        )
        
        print(f"[FilterController] Multi-position sweep complete:")
        print(f"  Positions: {success_pos}/{total_pos}")
        print(f"  Images: {total_imgs}")
        print(f"  Output: {results['base_output_dir']}")
    
    def _on_error(self, error_msg: str):
        """Handle sweep error."""
        self.is_running = False
        # Restart camera stream
        print("[FilterController] Restarting camera stream after error")
        self.signals.request_start_camera_stream.emit()
        # ✅ CRITICAL: Emit busy_ended so buttons re-enable
        self.signals.busy_ended.emit()
        self._update_position()
        self.signals.error_occurred.emit("Sweep Error", error_msg)
        print(f"[FilterController] Sweep error: {error_msg}")
    
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