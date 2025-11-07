# app/controllers/navigation_controller.py
"""
Navigation Controller

Handles stage movement to predicted positions (waveguides, gratings).
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal
import numpy as np
from typing import Optional

from app.system_state import SystemState
from app.signals import SystemSignals


class NavigationWorker(QThread):
    """Worker thread for stage navigation."""
    
    progress = pyqtSignal(str)  # Status message
    complete = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(
        self,
        stage,
        target_y: float,
        target_z: float,
        current_x: float,
        autofocus: bool = False
    ):
        super().__init__()
        self.stage = stage
        self.target_y = target_y
        self.target_z = target_z
        self.current_x = current_x
        self.autofocus = autofocus
    
    def run(self):
        """Execute navigation."""
        try:
            # Move Y and Z (keep X constant initially)
            self.progress.emit(f"Moving to Y={self.target_y:.2f}µm, Z={self.target_z:.2f}µm...")
            
            self.stage.move_abs('y', self.target_y)
            self.stage.move_abs('z', self.target_z)
            
            # Autofocus if requested
            if self.autofocus:
                self.progress.emit("Running autofocus...")
                # TODO: Implement autofocus call
                # For now, just wait
                import time
                time.sleep(0.5)
            
            self.complete.emit()
            
        except Exception as e:
            self.error.emit(str(e))


class NavigationController(QObject):
    """
    Controller for stage navigation.
    
    Responsibilities:
    - Calculate target positions using alignment system
    - Move stage safely
    - Optional autofocus after movement
    - Update state
    """
    
    def __init__(
        self,
        state: SystemState,
        signals: SystemSignals,
        stage,
        alignment_system,
        parent=None
    ):
        super().__init__(parent)
        
        self.state = state
        self.signals = signals
        self.stage = stage
        self.alignment = alignment_system
        
        self.worker = None
    
    def navigate_to_grating(
        self,
        block_id: int,
        waveguide: int,
        side: str,
        autofocus: bool = False
    ):
        """
        Navigate to grating position.
        
        Args:
            block_id: Block ID
            waveguide: Waveguide number
            side: 'left', 'center', or 'right'
            autofocus: Run autofocus after arrival
        """
        print(f"[NavigationController] Navigate to Block {block_id} WG{waveguide} {side}")
        
        # Check calibration
        if not self.state.global_calibrated:
            self.signals.error_occurred.emit(
                "Not Calibrated",
                "Global calibration required before navigation"
            )
            return
        
        # Get predicted position
        try:
            Y, Z = self.alignment.get_grating_stage_position(
                block_id, waveguide, side
            )
            print(f"  Predicted position: Y={Y:.3f}µm, Z={Z:.3f}µm")
        except Exception as e:
            self.signals.error_occurred.emit(
                "Prediction Failed",
                f"Could not predict position: {e}"
            )
            return
        
        # Get current X position
        current_x = self.state.stage_position['x']
        
        # Start worker
        self.worker = NavigationWorker(
            stage=self.stage,
            target_y=Y,
            target_z=Z,
            current_x=current_x,
            autofocus=autofocus
        )
        
        self.worker.progress.connect(
            lambda msg: self.signals.status_message.emit(msg)
        )
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(
            lambda e: self.signals.navigation_failed.emit(e)
        )
        
        # Emit started signal
        self.signals.navigation_started.emit(block_id, waveguide, side)
        
        # Start movement
        self.worker.start()
    
    def _on_complete(self):
        """Handle navigation completion."""
        print("[NavigationController] Navigation complete")
        
        # Add to history
        x, y, z = self.state.get_stage_position()
        self.state.navigation.add_to_history(x, y, z)
        
        # Emit completion
        self.signals.navigation_complete.emit()
        self.signals.status_message.emit("Navigation complete")
        
        # Cleanup worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None