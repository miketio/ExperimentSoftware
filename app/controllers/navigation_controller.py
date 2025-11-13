# app/controllers/navigation_controller.py
"""
Navigation Controller - COMPLETE IMPLEMENTATION

Handles stage movement to predicted waveguide/grating positions.
Uses HierarchicalAlignment for position prediction.
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal
import time
from typing import Optional, Tuple

from app.system_state import SystemState
from app.signals import SystemSignals


class NavigationWorker(QThread):
    """Worker thread for stage navigation with autofocus support."""
    
    progress = pyqtSignal(str)  # Status message
    complete = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(
        self,
        stage,
        alignment_system,  # HierarchicalAlignment instance
        block_id: int,
        waveguide: int,
        side: str,
        autofocus: bool = False,
        autofocus_controller = None,
        beam_offset_um: Optional[Tuple[float, float]] = None
    ):
        super().__init__()
        self.stage = stage
        self.alignment = alignment_system
        self.block_id = block_id
        self.waveguide = waveguide
        self.side = side
        self.autofocus = autofocus
        self.autofocus_controller = autofocus_controller
        
        self.cancelled = False
        self.beam_offset_um = beam_offset_um

    def cancel(self):
        """Cancel navigation."""
        self.cancelled = True
        print("[NavigationWorker] Cancellation requested")
    
    def run(self):
        """Execute navigation sequence."""
        try:
            # Step 1: Get predicted position
            self.progress.emit(f"Calculating position for Block {self.block_id} WG{self.waveguide} {self.side}...")
            
            try:
                # Handle "center" side - use waveguide center position
                if self.side == 'center':
                    print(f"[NavigationWorker] Using center position for WG{self.waveguide}")
                    
                    # Get waveguide center position directly
                    block = self.alignment.layout.get_block(self.block_id)
                    waveguide = block.get_waveguide(self.waveguide)
                    
                    # Use block_local_to_stage from CoordinateTransformV3
                    from AlignmentSystem.coordinate_transform_v3 import CoordinateTransformV3
                    transform = CoordinateTransformV3(self.alignment.layout)
                    transform.sync_with_runtime()
                    
                    # Waveguide center is at (u_center, v_center) in local coords
                    target_y, target_z = transform.block_local_to_stage(
                        self.block_id,
                        waveguide.center_position.u,
                        waveguide.center_position.v
                    )
                else:
                    print(f"[NavigationWorker] Using {self.side} grating for WG{self.waveguide}")
                    
                    # Use get_grating_stage_position for left/right
                    target_y, target_z = self.alignment.get_grating_stage_position(
                        block_id=self.block_id,
                        waveguide=self.waveguide,
                        side=self.side
                    )
                    
                print(f"[NavigationWorker] Target position calculated: Y={target_y:.3f}, Z={target_z:.3f} µm")
                
            except Exception as e:
                import traceback
                print(f"[NavigationWorker] Position prediction failed: {e}")
                traceback.print_exc()
                self.error.emit(f"Position prediction failed: {e}")
                return
            
            if self.cancelled:
                print("[NavigationWorker] Cancelled after position calculation")
                return
            
            # Step 2: Move to position
            self.progress.emit(f"Moving to Y={target_y:.3f}µm, Z={target_z:.3f}µm...")
            
            print(f"[NavigationWorker] Moving Y to {target_y:.3f} µm")
            print(f"[NavigationWorker] Moving Z to {target_z:.3f} µm")
            
            # Move Y and Z (preserve X focus position)
            self.stage.move_abs('y', target_y)
            if self.cancelled:
                print("[NavigationWorker] Cancelled after Y move")
                return
            
            self.stage.move_abs('z', target_z)
            if self.cancelled:
                print("[NavigationWorker] Cancelled after Z move")
                return
            
            # Small settle time
            time.sleep(0.1)

            # Step 2.5: APPLY BEAM OFFSET (NEW!)
            if self.beam_offset_um is not None:
                offset_y, offset_z = self.beam_offset_um
                
                self.progress.emit(
                    f"Applying beam offset: ΔY={offset_y:.3f}µm, ΔZ={offset_z:.3f}µm..."
                )
                
                print(f"[NavigationWorker] Applying beam offset: "
                      f"Y+={offset_y:.3f}µm, Z+={offset_z:.3f}µm")
                
                # Relative move to shift from center to beam
                self.stage.move_rel('y', -1*offset_y)
                if self.cancelled:
                    return
                
                self.stage.move_rel('z', offset_z)
                if self.cancelled:
                    return
                
                time.sleep(0.1)  # Settle
                
            # Step 3: Optional autofocus
            if self.autofocus and self.autofocus_controller:
                if self.cancelled:
                    return
                
                self.progress.emit("Running autofocus...")
                print("[NavigationWorker] Starting autofocus")
                
                # Run autofocus synchronously
                success = self.autofocus_controller.run_autofocus(
                    axis='x',
                    scan_range_um=10.0,
                    step_um=0.5,
                    enable_plot=False
                )
                
                if not success:
                    self.progress.emit("⚠️ Autofocus skipped (already running)")
                    print("[NavigationWorker] Autofocus skipped")
                else:
                    # Wait for autofocus to complete
                    while self.autofocus_controller.is_running:
                        if self.cancelled:
                            self.autofocus_controller.cancel()
                            print("[NavigationWorker] Cancelled during autofocus")
                            return
                        time.sleep(0.1)
                    print("[NavigationWorker] Autofocus complete")
            
            if self.cancelled:
                return
            
            # Complete
            print(f"[NavigationWorker] Navigation complete to Block {self.block_id} WG{self.waveguide} {self.side}")
            self.complete.emit()
            
        except Exception as e:
            import traceback
            print(f"[NavigationWorker] Fatal error: {e}")
            traceback.print_exc()
            self.error.emit(f"Navigation failed: {e}")


class NavigationController(QObject):
    """
    Controller for waveguide/grating navigation.
    
    Responsibilities:
    - Calculate target positions using HierarchicalAlignment
    - Move stage safely
    - Optional autofocus after movement
    - Update navigation state
    """
    
    def __init__(
        self,
        state: SystemState,
        signals: SystemSignals,
        stage,
        alignment_system,  # HierarchicalAlignment instance
        autofocus_controller = None,
        parent=None
    ):
        super().__init__(parent)
        
        self.state = state
        self.signals = signals
        self.stage = stage
        self.alignment = alignment_system
        self.autofocus = autofocus_controller
        
        self.worker = None
        
        print("[NavigationController] Initialized")
    
    def navigate_to_grating(
        self,
        block_id: int,
        waveguide: int,
        side: str,
        autofocus: bool = False
    ) -> bool:
        """
        Navigate to grating position.
        
        Args:
            block_id: Block ID
            waveguide: Waveguide number (1-50)
            side: 'left', 'center', or 'right'
            autofocus: Run autofocus after arrival
        
        Returns:
            True if navigation started, False if preconditions failed
        """
        print(f"[NavigationController] Navigate to Block {block_id} WG{waveguide} {side}")
        
        # Check preconditions
        if not self.state.global_calibrated:
            self.signals.error_occurred.emit(
                "Not Calibrated",
                "Global calibration required before navigation.\n\n"
                "Please run 'Global Alignment' first."
            )
            return False
        
        # CHECK: Is this specific block calibrated?
        runtime_layout = self.alignment.layout
        if not runtime_layout.is_block_calibrated(block_id):
            self.signals.warning_occurred.emit(
                "Block Not Calibrated",
                f"Block {block_id} has not been calibrated.\n\n"
                f"Navigation will use global calibration only, "
                f"which may be less accurate (±60µm).\n\n"
                f"For best accuracy, calibrate this block first."
            )
            # Allow navigation but warn user
        
        # Check if currently navigating
        if self.worker is not None and self.worker.isRunning():
            self.signals.warning_occurred.emit(
                "Navigation In Progress",
                "Please wait for current navigation to complete."
            )
            return False
        
        # Validate inputs
        if side not in ['left', 'center', 'right']:
            self.signals.error_occurred.emit(
                "Invalid Side",
                f"Grating side must be 'left', 'center', or 'right', got '{side}'"
            )
            return False
        
        # Get RuntimeLayout to check block
        try:
            block = runtime_layout.get_block(block_id)
            
            # Check if waveguide exists
            if waveguide not in block.list_waveguides():
                self.signals.error_occurred.emit(
                    "Invalid Waveguide",
                    f"Waveguide {waveguide} not found in Block {block_id}.\n\n"
                    f"Available: {block.list_waveguides()}"
                )
                return False
            
        except Exception as e:
            self.signals.error_occurred.emit(
                "Block Validation Failed",
                f"Could not validate block/waveguide: {e}"
            )
            return False
        
        # Create worker
        self.worker = NavigationWorker(
            stage=self.stage,
            alignment_system=self.alignment,
            block_id=block_id,
            waveguide=waveguide,
            side=side,
            autofocus=autofocus,
            autofocus_controller=self.autofocus
        )
        
        # Connect signals
        self.worker.progress.connect(
            lambda msg: self.signals.status_message.emit(msg)
        )
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(
            lambda e: self._on_error(e)
        )
        
        # Emit started signal
        self.signals.navigation_started.emit(block_id, waveguide, side)
        self.signals.busy_started.emit(f"Navigation to Block {block_id} WG{waveguide} {side}")
        
        print(f"[NavigationController] Starting worker thread")
        
        # Start movement
        self.worker.start()
        
        return True
    
    def cancel_navigation(self):
        """Cancel ongoing navigation."""
        if self.worker is not None and self.worker.isRunning():
            print("[NavigationController] Cancelling navigation")
            self.worker.cancel()
            self.signals.status_message.emit("Navigation cancelled")
    
    def _on_complete(self):
        """Handle navigation completion."""
        print("[NavigationController] Navigation complete")
        
        # Add to history
        x, y, z = self.state.get_stage_position()
        self.state.navigation.add_to_history(x, y, z)
        
        # Emit completion
        self.signals.navigation_complete.emit()
        self.signals.busy_ended.emit()
        self.signals.status_message.emit("✅ Navigation complete")
        
        # Cleanup worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def _on_error(self, error: str):
        """Handle navigation error."""
        print(f"[NavigationController] Error: {error}")
        
        self.signals.navigation_failed.emit(error)
        self.signals.busy_ended.emit()
        self.signals.error_occurred.emit("Navigation Failed", error)
        
        # Cleanup worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def navigate_to_grating_with_beam_offset(
        self,
        block_id: int,
        waveguide: int,
        side: str,
        beam_offset_um: Tuple[float, float],
        autofocus: bool = False
    ) -> bool:
        """
        Navigate to grating, then apply beam offset.
        
        Args:
            beam_offset_um: (Y_offset, Z_offset) in µm to shift from center to beam
        """
        print(f"[NavigationController] Navigate with beam offset: "
            f"Block {block_id} WG{waveguide} {side} + offset {beam_offset_um}")
        
        # Same preconditions as navigate_to_grating
        if not self.state.global_calibrated:
            self.signals.error_occurred.emit(
                "Not Calibrated",
                "Global calibration required before navigation."
            )
            return False
        
        # Create worker WITH beam offset
        self.worker = NavigationWorker(
            stage=self.stage,
            alignment_system=self.alignment,
            block_id=block_id,
            waveguide=waveguide,
            side=side,
            autofocus=autofocus,
            autofocus_controller=self.autofocus,
            beam_offset_um=beam_offset_um  # NEW parameter
        )
        
        # Connect signals
        self.worker.progress.connect(
            lambda msg: self.signals.status_message.emit(msg)
        )
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(
            lambda e: self._on_error(e)
        )
        
        # Emit started signal
        self.signals.navigation_started.emit(block_id, waveguide, side)
        self.signals.busy_started.emit(f"Navigation to Block {block_id} WG{waveguide} {side}")
        
        print(f"[NavigationController] Starting worker thread")
        
        # Start movement
        self.worker.start()
        
        return True