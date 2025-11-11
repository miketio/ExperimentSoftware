# app/controllers/alignment_controller.py
"""
Alignment Controller

Manages alignment procedures with GUI integration.
Coordinates worker thread, progress dialog, and state updates.
"""

from PyQt6.QtCore import QObject
from typing import List, Optional

from app.widgets.alignment_progress_dialog import AlignmentProgressDialog
from app.controllers.alignment_worker import AlignmentWorker
from app.system_state import SystemState, AlignmentStatus
from app.signals import SystemSignals


class AlignmentController(QObject):
    """
    Controller for alignment procedures.
    
    Responsibilities:
    - Start/stop alignment worker thread
    - Show progress dialog
    - Update SystemState after completion
    - Emit appropriate signals
    """
    
    def __init__(
        self,
        state: SystemState,
        signals: SystemSignals,
        camera,
        stage,
        runtime_layout,
        parent=None
    ):
        """
        Initialize alignment controller.
        
        Args:
            state: System state (GUI state tracking)
            signals: System signals (for UI updates)
            camera: Camera instance
            stage: Stage adapter (µm interface)
            runtime_layout: RuntimeLayout instance
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self.state = state
        self.signals = signals
        self.camera = camera
        self.stage = stage
        self.runtime_layout = runtime_layout
        
        # Create HierarchicalAlignment instance for predictions
        from AlignmentSystem.hierarchicalAlignment_v3 import HierarchicalAlignment
        self.alignment_system = HierarchicalAlignment(runtime_layout)
        
        # Worker and dialog
        self.worker = None
        self.progress_dialog = None
    
    # =========================================================================
    # Global Alignment
    # =========================================================================
    
    def start_global_alignment(
        self,
        corner_pairs: List[tuple] = [
            (1, 'top_left'),      # Block 1
            (20, 'bottom_right')  # Block 20    
        ],
        search_radius_um: float = 100.0,
        step_um: float = 20.0
    ):
        """
        Start global alignment procedure.
        
        Args:
            corner_pairs: Block IDs and fiducial to use for calibration
            search_radius_um: Search radius around design position
            step_um: Grid step size
        """
        print(f"[AlignmentController] Starting global alignment")
        
        # Check preconditions
        if not self.state.camera_connected or not self.state.stage_connected:
            self.signals.error_occurred.emit(
                "Hardware Not Ready",
                "Camera and stage must be connected"
            )
            return
        
        # Create progress dialog
        self.progress_dialog = AlignmentProgressDialog(
            title="Global Alignment",
            parent=None  # Will be set by caller
        )
        
        # Create worker
        self.worker = AlignmentWorker(
            camera=self.camera,
            stage=self.stage,
            runtime_layout=self.runtime_layout
        )
        
        # Configure task
        self.worker.configure_global_alignment(
            corner_pairs=corner_pairs,
            search_radius_um=search_radius_um,
            step_um=step_um
        )
        
        # Connect worker signals
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.block_found.connect(self._on_block_found)
        self.worker.calibration_complete.connect(self._on_calibration_complete)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        
        # Connect dialog cancel to worker
        self.progress_dialog.rejected.connect(self.worker.cancel)
        
        # Start worker
        self.worker.start()
        
        # Show dialog (blocks until closed)
        self.progress_dialog.exec()
    
    # =========================================================================
    # Block Alignment
    # =========================================================================
    
    def start_block_alignment(
        self,
        block_id: int,
        corners: List[str] = ['top_left', 'bottom_right'],
        search_radius_um: float = 60.0,
        step_um: float = 15.0
    ):
        """
        Start block-level alignment.
        
        Args:
            block_id: Block to calibrate
            corners: Which corners to find
            search_radius_um: Search radius around predicted position
            step_um: Grid step size
        """
        print(f"[AlignmentController] Starting block {block_id} alignment")
        
        # Check preconditions
        if not self.state.global_calibrated:
            self.signals.error_occurred.emit(
                "Not Ready",
                "Global calibration required before block alignment"
            )
            return
        
        # Create progress dialog
        self.progress_dialog = AlignmentProgressDialog(
            title=f"Block {block_id} Alignment",
            parent=None
        )
        
        # Create worker
        self.worker = AlignmentWorker(
            camera=self.camera,
            stage=self.stage,
            runtime_layout=self.runtime_layout
        )
        
        # Configure task
        self.worker.configure_block_alignment(
            block_id=block_id,
            corners=corners,
            search_radius_um=search_radius_um,
            step_um=step_um
        )
        
        # Connect signals
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.block_found.connect(self._on_block_found)
        self.worker.calibration_complete.connect(self._on_calibration_complete)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        
        self.progress_dialog.rejected.connect(self.worker.cancel)
        
        # Start and show
        self.worker.start()
        self.progress_dialog.exec()
    
    # =========================================================================
    # Batch Alignment
    # =========================================================================
    
    def start_batch_alignment(
        self,
        block_ids: List[int],
        search_radius_um: float = 60.0,
        step_um: float = 15.0
    ):
        """
        Start batch block alignment.
        
        Args:
            block_ids: List of blocks to calibrate
            search_radius_um: Search radius
            step_um: Grid step size
        """
        print(f"[AlignmentController] Starting batch alignment for {len(block_ids)} blocks")
        
        if not self.state.global_calibrated:
            self.signals.error_occurred.emit(
                "Not Ready",
                "Global calibration required before batch alignment"
            )
            return
        
        # Create dialog
        self.progress_dialog = AlignmentProgressDialog(
            title=f"Batch Alignment ({len(block_ids)} blocks)",
            parent=None
        )
        
        # Create worker
        self.worker = AlignmentWorker(
            camera=self.camera,
            stage=self.stage,
            runtime_layout=self.runtime_layout
        )
        
        # Configure
        self.worker.configure_batch_alignment(
            block_ids=block_ids,
            search_radius_um=search_radius_um,
            step_um=step_um
        )
        
        # Connect
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.block_found.connect(self._on_block_found)
        self.worker.calibration_complete.connect(self._on_calibration_complete)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        
        self.progress_dialog.rejected.connect(self.worker.cancel)
        
        # Start
        self.worker.start()
        self.progress_dialog.exec()
    
    # =========================================================================
    # Signal Handlers (from worker)
    # =========================================================================
    
    def _on_progress_updated(self, current: int, total: int, status: str, thumbnail):
        """Handle progress update from worker."""
        if self.progress_dialog:
            self.progress_dialog.update_progress(current, total, status)
    
    def _on_block_found(
        self,
        block_id: int,
        corner: str,
        Y_um: float,
        Z_um: float,
        error_um: float,
        image
    ):
        """Handle fiducial found from worker."""
        if self.progress_dialog:
            self.progress_dialog.add_fiducial_thumbnail(
                block_id, corner, image, error_um
            )
            self.progress_dialog.append_log(
                f"✓ Block {block_id} {corner}: ({Y_um:.3f}, {Z_um:.3f}) µm, error={error_um:.3f}µm"
            )
    
    def _on_calibration_complete(self, results: dict):
        """Handle calibration completion from worker."""
        result_type = results['type']
        
        if result_type == 'global':
            # Update SystemState
            calib = results['calibration']
            self.state.global_calibrated = True
            self.state.global_calibration_params = calib
            
            # Update block statuses to GLOBAL_ONLY
            for block_id in self.state.blocks.keys():
                self.state.set_block_status(
                    block_id,
                    AlignmentStatus.GLOBAL_ONLY,
                    error=None,
                    fiducials_found=0
                )
            
            # Mark dialog complete
            if self.progress_dialog:
                self.progress_dialog.mark_complete(success=True)
                self.progress_dialog.append_log(
                    f"\n✓ Global calibration complete!\n"
                    f"  Rotation: {calib['rotation_deg']:.6f}°\n"
                    f"  Translation: {calib['translation_um']} µm\n"
                    f"  Mean error: {calib['mean_error_um']:.6f} µm"
                )
            
            # Emit signal
            self.signals.global_alignment_complete.emit(results)
            self.signals.status_message.emit(
                f"Global alignment complete (error: {calib['mean_error_um']:.3f}µm)"
            )
        
        elif result_type == 'block':
            # Update SystemState
            block_id = results['block_id']
            calib = results['calibration']
            
            self.state.set_block_status(
                block_id,
                AlignmentStatus.BLOCK_CALIBRATED,
                error=calib['mean_error_um'],
                fiducials_found=len(results['measurements'])
            )
            
            # Mark dialog complete
            if self.progress_dialog:
                self.progress_dialog.mark_complete(success=True)
                self.progress_dialog.append_log(
                    f"\n✓ Block {block_id} calibration complete!\n"
                    f"  Mean error: {calib['mean_error_um']:.6f} µm"
                )
            
            # Emit signal
            self.signals.block_alignment_complete.emit(block_id, results)
            self.signals.status_message.emit(
                f"Block {block_id} aligned (error: {calib['mean_error_um']:.3f}µm)"
            )
        
        elif result_type == 'batch':
            # Mark dialog complete
            if self.progress_dialog:
                self.progress_dialog.mark_complete(success=True)
                self.progress_dialog.append_log(
                    f"\n✓ Batch alignment complete!\n"
                    f"  {results['completed']} blocks calibrated"
                )
            
            # Emit signal
            self.signals.batch_alignment_complete.emit(results)
            self.signals.status_message.emit(
                f"Batch alignment complete ({results['completed']} blocks)"
            )
    
    def _on_error(self, title: str, message: str):
        """Handle error from worker."""
        if self.progress_dialog:
            self.progress_dialog.mark_complete(success=False)
            self.progress_dialog.append_log(f"\n✗ ERROR: {message}")
        
        self.signals.error_occurred.emit(title, message)
    
    def _on_worker_finished(self):
        """Handle worker thread completion."""
        print("[AlignmentController] Worker finished")
        
        # Cleanup
        if self.worker:
            self.worker.deleteLater()
            self.worker = None