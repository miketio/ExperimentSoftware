# app/controllers/alignment_worker.py - UPDATED
"""
Alignment Worker Thread - WITH FIDUCIAL SYNC

Changes:
1. Automatically add captured fiducials to RuntimeLayout for manual panel display
"""

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import numpy as np
import time
from typing import Optional, List, Dict, Any

from alignment_system.hierarchical_alignment import HierarchicalAlignment
from alignment_system.alignment_search import AlignmentSearcher
from alignment_system.cv_tools import VisionTools


class AlignmentWorker(QThread):
    """Worker thread for alignment procedures."""
    
    # Signals
    progress_updated = pyqtSignal(int, int, str, object)
    block_found = pyqtSignal(int, str, float, float, float, object)
    calibration_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, camera, stage, runtime_layout, alignment_system=None, parent=None):
        super().__init__(parent)
        
        self.camera = camera
        self.stage = stage
        self.runtime_layout = runtime_layout
        self.vision_tools = VisionTools()
        self.searcher = AlignmentSearcher(stage=stage, camera=camera, vision_tools=self.vision_tools)
        self.alignment = alignment_system
        
        self.cancel_flag = False
        self.mutex = QMutex()
        self.task = None
        self.task_params = {}
    
    def configure_global_alignment(self, corner_pairs = [
                        (1, 'top_left'),      
                        (20, 'bottom_right')  
                    ], search_radius_um=100.0, step_um=20.0):
        self.task = 'global'
        self.task_params = {
            'corner_pairs': corner_pairs,
            'search_radius_um': search_radius_um,
            'step_um': step_um
        }
    
    def configure_block_alignment(self, block_id, corners=['top_left', 'bottom_right'], 
                                 search_radius_um=60.0, step_um=15.0):
        self.task = 'block'
        self.task_params = {
            'block_id': block_id,
            'corners': corners,
            'search_radius_um': search_radius_um,
            'step_um': step_um
        }
    
    def configure_batch_alignment(self, block_ids, search_radius_um=60.0, step_um=15.0):
        self.task = 'batch'
        self.task_params = {
            'block_ids': block_ids,
            'search_radius_um': search_radius_um,
            'step_um': step_um
        }
    
    def cancel(self):
        with QMutexLocker(self.mutex):
            self.cancel_flag = True
    
    def is_cancelled(self):
        with QMutexLocker(self.mutex):
            return self.cancel_flag
    
    def run(self):
        try:
            if self.alignment is None:
                self.alignment = HierarchicalAlignment(self.runtime_layout)
            
            if self.task == 'global':
                self._run_global_alignment()
            elif self.task == 'block':
                self._run_block_alignment()
            elif self.task == 'batch':
                self._run_batch_alignment()
            else:
                self.error_occurred.emit("Configuration Error", f"Unknown task: {self.task}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error_occurred.emit("Alignment Failed", f"{str(e)}\n\n{tb}")
        finally:
            print("[AlignmentWorker] Thread finished")
    
    def _run_global_alignment(self):
        corner_pairs = self.task_params['corner_pairs']
        search_radius = self.task_params['search_radius_um']
        step = self.task_params['step_um']
        
        print(f"[AlignmentWorker] Starting global alignment")
        
        if not self.runtime_layout.has_block_1_position():
            self.error_occurred.emit(
                "Block 1 Not Set",
                "You must set Block 1 position before global calibration.\n\n"
                "Go to: Calibration > Set Block 1 Position"
            )
            return
        
        block1_Y_measured, block1_Z_measured = self.runtime_layout.get_block_1_position()
        block1 = self.runtime_layout.get_block(1)
        block1_Y_design = block1.design_position.u
        block1_Z_design = block1.design_position.v
        
        print(f"[AlignmentWorker] Block 1 reference:")
        print(f"  Design:   ({block1_Y_design:.3f}, {block1_Z_design:.3f}) µm")
        print(f"  Measured: ({block1_Y_measured:.3f}, {block1_Z_measured:.3f}) µm")
        
        total_steps = len(corner_pairs)
        current_step = 0
        measurements = []
        
        for block_id, corner in corner_pairs:
            if self.is_cancelled():
                print("[AlignmentWorker] Cancelled by user")
                self.error_occurred.emit("Cancelled", "Alignment cancelled by user")
                return
            
            block = self.runtime_layout.get_block(block_id)
            block_size = self.runtime_layout.block_layout.block_size
            
            current_step += 1
            
            self.progress_updated.emit(
                current_step, total_steps,
                f"Searching Block {block_id} {corner}...", None
            )
            
            fiducial = block.get_fiducial(corner)
            u_center = block.design_position.u
            v_center = block.design_position.v
            
            u_bl = u_center - block_size / 2.0
            v_bl = v_center - block_size / 2.0
            u_global_design = u_bl + fiducial.u
            v_global_design = v_bl + fiducial.v
            
            offset_Y = block1_Y_measured - block1_Y_design
            offset_Z = block1_Z_measured - block1_Z_design
            
            search_center_Y = u_global_design + offset_Y
            search_center_Z = v_global_design + offset_Z
            
            result = self.searcher.search_for_fiducial(
                center_y_um=search_center_Y,
                center_z_um=search_center_Z,
                search_radius_um=search_radius,
                step_um=step,
                label=f"Block {block_id} {corner}",
                plot_progress=False,
                cancel_callback=self.is_cancelled
            )
            
            if result is None:
                self.error_occurred.emit("Search Failed", 
                                    f"Could not find Block {block_id} {corner}")
                return

            measurements.append({
                'block_id': block_id,
                'corner': corner,
                'stage_Y': result['stage_Y'],
                'stage_Z': result['stage_Z'],
                'confidence': result['confidence'],
                'verification_error_um': result.get('verification_error_um', 0)
            })
            
            # ✅ NEW: Also add to RuntimeLayout's captured fiducials
            self.runtime_layout.add_captured_fiducial(
                block_id, corner, 
                result['stage_Y'], 
                result['stage_Z']
            )
            
            self.block_found.emit(
                block_id, corner,
                result['stage_Y'], result['stage_Z'],
                result.get('verification_error_um', 0),
                result.get('image')
            )
            
            print(f"    ✓ Found at ({result['stage_Y']:.3f}, {result['stage_Z']:.3f}) µm")
            
            if self.is_cancelled():
                print("[AlignmentWorker] Cancelled after search")
                self.error_occurred.emit("Cancelled", "Alignment cancelled by user")
                return
        
        if self.is_cancelled():
            return
        
        self.progress_updated.emit(total_steps, total_steps, 
                                  "Calculating global transformation...", None)
        
        print(f"  Calibrating with {len(measurements)} fiducials...")
        calib_result = self.alignment.calibrate_global(measurements)
        
        print(f"  ✓ Global calibration complete:")
        print(f"    Rotation: {calib_result['rotation_deg']:.6f}°")
        print(f"    Translation: {calib_result['translation_um']} µm")
        
        self.runtime_layout.set_global_calibration(
            rotation=calib_result['rotation_deg'],
            translation=calib_result['translation_um'],
            calibration_error=calib_result['mean_error_um'],
            num_points=len(measurements)
        )
        
        self.calibration_complete.emit({
            'type': 'global',
            'measurements': measurements,
            'calibration': calib_result
        })
    
    def _run_block_alignment(self):
        block_id = self.task_params['block_id']
        corners = self.task_params['corners']
        search_radius = self.task_params['search_radius_um']
        step = self.task_params['step_um']
        
        print(f"[AlignmentWorker] Starting block {block_id} alignment")
        
        if not self.runtime_layout.is_globally_calibrated():
            self.error_occurred.emit("Not Ready", 
                                   "Global calibration required before block alignment")
            return
        
        total_steps = len(corners)
        current_step = 0
        measurements = []
        
        for corner in corners:
            if self.is_cancelled():
                return
            
            current_step += 1
            self.progress_updated.emit(current_step, total_steps,
                                      f"Searching Block {block_id} {corner}...", None)
            
            try:
                pred_Y, pred_Z = self.alignment.get_fiducial_stage_position(block_id, corner)
                print(f"  Predicted {corner}: ({pred_Y:.3f}, {pred_Z:.3f}) µm")
            except Exception as e:
                self.error_occurred.emit("Prediction Failed", f"Could not predict position: {e}")
                return
            
            result = self.searcher.search_for_fiducial(
                center_y_um=pred_Y,
                center_z_um=pred_Z,
                search_radius_um=search_radius,
                step_um=step,
                label=f"Block {block_id} {corner}",
                plot_progress=False,
                cancel_callback=self.is_cancelled
            )
            
            if result is None:
                self.error_occurred.emit("Search Failed", 
                                       f"Could not find Block {block_id} {corner}")
                return
            
            pred_error = np.hypot(result['stage_Y'] - pred_Y, result['stage_Z'] - pred_Z)
            
            measurements.append({
                'corner': corner,
                'stage_Y': result['stage_Y'],
                'stage_Z': result['stage_Z']
            })
            
            # ✅ NEW: Also add to RuntimeLayout's captured fiducials
            self.runtime_layout.add_captured_fiducial(
                block_id, corner,
                result['stage_Y'],
                result['stage_Z']
            )
            
            self.block_found.emit(block_id, corner,
                                result['stage_Y'], result['stage_Z'],
                                result.get('verification_error_um', 0),
                                result.get('image'))
            
            print(f"    ✓ Found at ({result['stage_Y']:.3f}, {result['stage_Z']:.3f}) µm")
            print(f"      Prediction error: {pred_error:.3f} µm")
        
        if self.is_cancelled():
            return
        
        self.progress_updated.emit(total_steps, total_steps,
                                  f"Calculating Block {block_id} transformation...", None)
        
        calib_result = self.alignment.calibrate_block(block_id, measurements)
        
        print(f"  ✓ Block {block_id} calibration complete:")
        print(f"    Mean error: {calib_result['mean_error_um']:.6f} µm")
        
        self.runtime_layout.set_block_calibration(
            block_id=block_id,
            rotation=calib_result['rotation_deg'],
            translation=calib_result['origin_stage_um'],
            calibration_error=calib_result['mean_error_um'],
            num_points=2
        )
        
        self.calibration_complete.emit({
            'type': 'block',
            'block_id': block_id,
            'measurements': measurements,
            'calibration': calib_result
        })
    
    def _run_batch_alignment(self):
        block_ids = self.task_params['block_ids']
        print(f"[AlignmentWorker] Batch alignment for {len(block_ids)} blocks")
        
        for i, block_id in enumerate(block_ids):
            if self.is_cancelled():
                return
            
            self.progress_updated.emit(i, len(block_ids),
                                      f"Calibrating Block {block_id}...", None)
            
            self.configure_block_alignment(block_id)
            self._run_block_alignment()
        
        self.calibration_complete.emit({
            'type': 'batch',
            'block_ids': block_ids,
            'completed': len(block_ids)
        })