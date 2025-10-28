# alignment_state.py
"""
State management for alignment workflow.
Tracks calibration status, current target, and optimization results.
"""
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class AlignmentStatus(Enum):
    """Alignment workflow states."""
    IDLE = "idle"
    FINDING_FIDUCIAL = "finding_fiducial"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    NAVIGATING = "navigating"
    OPTIMIZING = "optimizing"
    ALIGNED = "aligned"
    ERROR = "error"


class AlignmentState:
    """
    Shared state between alignment controller and agent.
    Thread-safe state tracking for alignment workflow.
    """
    
    def __init__(self):
        # Status
        self.status = AlignmentStatus.IDLE
        self.error_message = None
        
        # Calibration
        self.is_calibrated = False
        self.calibration_result = None
        self.fiducials_found = {}  # corner -> (Y, Z, confidence)
        
        # Current target
        self.current_block = None
        self.current_waveguide = None
        self.current_side = None
        self.target_design_coords = None  # (u, v) in µm
        self.target_stage_coords = None   # (Y, Z) in nm
        
        # Optimization
        self.optimization_running = False
        self.optimization_progress = 0.0
        self.optimization_result = None
        self.best_position = None  # (Y, Z) in nm
        self.best_intensity = None
        
        # History
        self.alignment_history = []  # List of completed alignments
        
        # Timestamps
        self.last_calibration_time = None
        self.last_alignment_time = None
    
    def set_status(self, status: AlignmentStatus, message: Optional[str] = None):
        """Update status and optional error message."""
        self.status = status
        if status == AlignmentStatus.ERROR:
            self.error_message = message
        else:
            self.error_message = None
    
    def add_fiducial(self, corner: str, Y: int, Z: int, confidence: float):
        """Record found fiducial position."""
        self.fiducials_found[corner] = {
            'Y': Y,
            'Z': Z,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def set_calibration(self, result: Dict):
        """Store calibration results."""
        self.is_calibrated = True
        self.calibration_result = result
        self.last_calibration_time = datetime.now().isoformat()
        self.set_status(AlignmentStatus.CALIBRATED)
    
    def set_target(self, block_id: int, waveguide_number: int, side: str,
                   design_coords: Tuple[float, float],
                   stage_coords: Tuple[int, int]):
        """Set current alignment target."""
        self.current_block = block_id
        self.current_waveguide = waveguide_number
        self.current_side = side
        self.target_design_coords = design_coords
        self.target_stage_coords = stage_coords
    
    def start_optimization(self):
        """Mark optimization as started."""
        self.optimization_running = True
        self.optimization_progress = 0.0
        self.set_status(AlignmentStatus.OPTIMIZING)
    
    def update_optimization_progress(self, progress: float):
        """Update optimization progress (0.0 to 1.0)."""
        self.optimization_progress = progress
    
    def finish_optimization(self, result: Dict):
        """Store optimization results."""
        self.optimization_running = False
        self.optimization_progress = 1.0
        self.optimization_result = result
        
        if result.get('success'):
            self.best_position = result['best_position']
            self.best_intensity = result['best_intensity']
            self.set_status(AlignmentStatus.ALIGNED)
            
            # Add to history
            self.alignment_history.append({
                'block': self.current_block,
                'waveguide': self.current_waveguide,
                'side': self.current_side,
                'position': self.best_position,
                'intensity': self.best_intensity,
                'timestamp': datetime.now().isoformat()
            })
            
            self.last_alignment_time = datetime.now().isoformat()
        else:
            self.set_status(AlignmentStatus.ERROR, result.get('error'))
    
    def reset(self):
        """Reset state to initial conditions."""
        self.status = AlignmentStatus.IDLE
        self.error_message = None
        self.current_block = None
        self.current_waveguide = None
        self.current_side = None
        self.target_design_coords = None
        self.target_stage_coords = None
        self.optimization_running = False
        self.optimization_progress = 0.0
        self.optimization_result = None
        self.best_position = None
        self.best_intensity = None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary for agent/API."""
        return {
            'status': self.status.value,
            'error_message': self.error_message,
            
            'calibration': {
                'is_calibrated': self.is_calibrated,
                'result': self.calibration_result,
                'fiducials_found': self.fiducials_found,
                'last_calibration_time': self.last_calibration_time
            },
            
            'current_target': {
                'block': self.current_block,
                'waveguide': self.current_waveguide,
                'side': self.current_side,
                'design_coords': self.target_design_coords,
                'stage_coords': self.target_stage_coords
            },
            
            'optimization': {
                'running': self.optimization_running,
                'progress': self.optimization_progress,
                'result': self.optimization_result,
                'best_position': self.best_position,
                'best_intensity': self.best_intensity
            },
            
            'history': {
                'alignments_completed': len(self.alignment_history),
                'last_alignment_time': self.last_alignment_time,
                'recent_alignments': self.alignment_history[-5:] if self.alignment_history else []
            }
        }


# Global state instance (singleton pattern)
_global_state = None

def get_alignment_state() -> AlignmentState:
    """Get the global alignment state instance."""
    global _global_state
    if _global_state is None:
        _global_state = AlignmentState()
    return _global_state


# Test/example usage
if __name__ == "__main__":
    print("Alignment State Module")
    print("======================")
    
    state = get_alignment_state()
    
    # Simulate workflow
    print("\n1. Initial state:")
    print(f"   Status: {state.status.value}")
    
    print("\n2. Finding fiducials...")
    state.set_status(AlignmentStatus.FINDING_FIDUCIAL)
    state.add_fiducial('top_left', 50000, 30000, 0.95)
    state.add_fiducial('bottom_right', 1450000, 630000, 0.92)
    
    print("\n3. Calibrating...")
    state.set_status(AlignmentStatus.CALIBRATING)
    state.set_calibration({
        'method': 'two_point',
        'angle_deg': 2.8,
        'mean_error_nm': 15.3
    })
    
    print(f"   Calibrated: {state.is_calibrated}")
    print(f"   Angle: {state.calibration_result['angle_deg']}°")
    
    print("\n4. Setting target (Block 10, WG 25, left)...")
    state.set_target(10, 25, 'left', (12.0, 117.6), (162000, 147600))
    
    print("\n5. Optimizing...")
    state.start_optimization()
    for i in range(5):
        state.update_optimization_progress(i / 4)
        print(f"   Progress: {state.optimization_progress * 100:.0f}%")
    
    state.finish_optimization({
        'success': True,
        'best_position': (162500, 148100),
        'best_intensity': 3245.7
    })
    
    print(f"\n6. Final state:")
    state_dict = state.get_state_dict()
    print(f"   Status: {state_dict['status']}")
    print(f"   Best position: {state_dict['optimization']['best_position']}")
    print(f"   Best intensity: {state_dict['optimization']['best_intensity']}")
    print(f"   Alignments completed: {state_dict['history']['alignments_completed']}")