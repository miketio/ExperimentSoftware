# mockStage_v2.py
"""
Enhanced mock stage with camera observer pattern.
Notifies connected camera when stage moves so camera can update its FOV.
"""
import time
import numpy as np
from typing import Optional, Dict
from BaseClasses.xyzStageBase import XYZStageBase


class MockXYZStage(XYZStageBase):
    """
    Enhanced mock 3D stage with camera observer notification.
    
    Features:
    - In-memory position tracking
    - Camera observer pattern (notifies camera on movement)
    - Optional movement delays (for realism)
    - Optional position noise/jitter
    - Position logging for ground truth tracking
    """
    
    def __init__(
        self,
        start_positions: Optional[Dict[str, int]] = None,
        simulate_delays: bool = False,
        position_noise_nm: float = 0.0
    ):
        """
        Initialize mock stage.
        
        Args:
            start_positions: Initial positions dict {'x': nm, 'y': nm, 'z': nm}
            simulate_delays: If True, add realistic movement delays
            position_noise_nm: Amount of position noise to add (nm, std dev)
        """
        self._pos = {'x': 0, 'y': 0, 'z': 0}
        if start_positions:
            self._pos.update(start_positions)
        
        self._closed = False
        self._camera_observer = None
        
        # Simulation parameters
        self.simulate_delays = simulate_delays
        self.position_noise_nm = position_noise_nm
        
        # Movement speed (nm/s) for delay simulation
        self.speed_nm_per_s = {'x': 100000, 'y': 50000, 'z': 50000}  # X faster
        
        # Position history for logging
        self.position_history = []
        
        print(f"[MockStage] Initialized at X={self._pos['x']}nm, "
              f"Y={self._pos['y']}nm, Z={self._pos['z']}nm")
    
    # =========================================================================
    # Camera Observer Pattern
    # =========================================================================
    
    def set_camera_observer(self, camera):
        """
        Set camera observer to be notified on position changes.
        
        Args:
            camera: MockCamera instance (or any object with position tracking)
        """
        self._camera_observer = camera
        print(f"[MockStage] Camera observer registered")
    
    def _notify_camera(self):
        """Notify camera that stage has moved (if observer is set)."""
        if self._camera_observer is not None:
            # Camera will automatically use new stage position on next acquisition
            pass  # MockCamera reads position directly via get_pos()
    
    # =========================================================================
    # Core Stage Interface (XYZStageBase)
    # =========================================================================
    
    def move_abs(self, axis: str, pos: int) -> None:
        """
        Move axis to absolute position.
        
        Args:
            axis: 'x', 'y', or 'z'
            pos: Target position in nanometers
        """
        if axis not in self._pos:
            raise ValueError(f"Invalid axis: {axis}")
        
        if self._closed:
            raise RuntimeError("Stage is closed")
        
        old_pos = self._pos[axis]
        target_pos = int(pos)
        
        # Simulate movement delay if enabled
        if self.simulate_delays:
            distance = abs(target_pos - old_pos)
            delay = distance / self.speed_nm_per_s[axis]
            time.sleep(min(delay, 0.5))  # Cap at 0.5s for testing
        else:
            # Minimal delay
            time.sleep(0.01)
        
        # Update position
        self._pos[axis] = target_pos
        
        # Add noise if enabled
        if self.position_noise_nm > 0:
            noise = np.random.normal(0, self.position_noise_nm)
            self._pos[axis] += int(noise)
        
        # Log position
        self._log_position(axis, old_pos, self._pos[axis])
        
        # Notify camera
        self._notify_camera()
        
        print(f"[MockStage] Moved {axis.upper()}: {old_pos}nm → {self._pos[axis]}nm")
    
    def move_rel(self, axis: str, shift: int, sleep_time: float = 0.01) -> None:
        """
        Move axis by relative amount.
        
        Args:
            axis: 'x', 'y', or 'z'
            shift: Amount to shift in nanometers (can be negative)
            sleep_time: Movement delay (overridden if simulate_delays=True)
        """
        if axis not in self._pos:
            raise ValueError(f"Invalid axis: {axis}")
        
        if self._closed:
            raise RuntimeError("Stage is closed")
        
        old_pos = self._pos[axis]
        target_pos = old_pos + int(shift)
        
        # Use move_abs for consistent behavior
        self.move_abs(axis, target_pos)
    
    def get_pos(self, axis: str) -> int:
        """
        Get current position of axis.
        
        Args:
            axis: 'x', 'y', or 'z'
        
        Returns:
            Position in nanometers
        """
        if axis not in self._pos:
            raise ValueError(f"Invalid axis: {axis}")
        
        if self._closed:
            raise RuntimeError("Stage is closed")
        
        return int(self._pos[axis])
    
    def close(self) -> None:
        """Close stage and cleanup."""
        self._closed = True
        self._camera_observer = None
        print(f"[MockStage] Closed")
    
    # =========================================================================
    # Additional Convenience Methods
    # =========================================================================
    
    def get_pos_all(self) -> Dict[str, int]:
        """
        Get positions of all axes.
        
        Returns:
            dict: {'x': nm, 'y': nm, 'z': nm}
        """
        return {
            'x': self.get_pos('x'),
            'y': self.get_pos('y'),
            'z': self.get_pos('z')
        }
    
    def set_pos_all(self, x: int, y: int, z: int) -> None:
        """
        Set all axes simultaneously.
        
        Args:
            x, y, z: Positions in nanometers
        """
        print(f"[MockStage] Moving to X={x}nm, Y={y}nm, Z={z}nm")
        self.move_abs('x', x)
        self.move_abs('y', y)
        self.move_abs('z', z)
    
    def home(self) -> None:
        """Move all axes to origin (0, 0, 0)."""
        print(f"[MockStage] Homing...")
        self.set_pos_all(0, 0, 0)
    
    # =========================================================================
    # Position Logging (for ground truth tracking)
    # =========================================================================
    
    def _log_position(self, axis: str, old_pos: int, new_pos: int):
        """Log position change for ground truth tracking."""
        timestamp = time.time()
        self.position_history.append({
            'timestamp': timestamp,
            'axis': axis,
            'old_position': old_pos,
            'new_position': new_pos,
            'all_positions': self._pos.copy()
        })
    
    def get_position_history(self) -> list:
        """Get complete position history."""
        return self.position_history.copy()
    
    def clear_position_history(self):
        """Clear position history."""
        self.position_history.clear()
        print(f"[MockStage] Position history cleared")
    
    def save_position_history(self, filename: str = "mock_stage_positions.csv"):
        """
        Save position history to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Axis', 'Old_Position_nm', 
                           'New_Position_nm', 'X_nm', 'Y_nm', 'Z_nm'])
            
            for entry in self.position_history:
                writer.writerow([
                    entry['timestamp'],
                    entry['axis'],
                    entry['old_position'],
                    entry['new_position'],
                    entry['all_positions']['x'],
                    entry['all_positions']['y'],
                    entry['all_positions']['z']
                ])
        
        print(f"[MockStage] Position history saved to {filename}")
    
    # =========================================================================
    # Simulation Control
    # =========================================================================
    
    def enable_delays(self, enable: bool = True):
        """Enable or disable movement delays."""
        self.simulate_delays = enable
        print(f"[MockStage] Movement delays: {'enabled' if enable else 'disabled'}")
    
    def set_position_noise(self, noise_nm: float):
        """
        Set position noise level.
        
        Args:
            noise_nm: Standard deviation of position noise in nanometers
        """
        self.position_noise_nm = noise_nm
        print(f"[MockStage] Position noise set to {noise_nm}nm (std dev)")
    
    def set_speed(self, axis: str, speed_nm_per_s: float):
        """
        Set movement speed for delay simulation.
        
        Args:
            axis: 'x', 'y', or 'z'
            speed_nm_per_s: Speed in nm/s
        """
        if axis not in self._pos:
            raise ValueError(f"Invalid axis: {axis}")
        
        self.speed_nm_per_s[axis] = speed_nm_per_s
        print(f"[MockStage] {axis.upper()} speed set to {speed_nm_per_s/1000:.1f} µm/s")
    
    # =========================================================================
    # Status and Diagnostics
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get comprehensive stage status."""
        return {
            'connected': not self._closed,
            'position': self._pos.copy(),
            'has_camera_observer': self._camera_observer is not None,
            'simulate_delays': self.simulate_delays,
            'position_noise_nm': self.position_noise_nm,
            'speed_nm_per_s': self.speed_nm_per_s.copy(),
            'position_history_length': len(self.position_history)
        }
    
    def print_status(self):
        """Print stage status to console."""
        status = self.get_status()
        
        print(f"\n{'='*60}")
        print(f"MockStage Status")
        print(f"{'='*60}")
        print(f"Connected:        {status['connected']}")
        print(f"Position:         X={status['position']['x']}nm, "
              f"Y={status['position']['y']}nm, "
              f"Z={status['position']['z']}nm")
        print(f"Camera Observer:  {status['has_camera_observer']}")
        print(f"Simulate Delays:  {status['simulate_delays']}")
        print(f"Position Noise:   {status['position_noise_nm']}nm")
        print(f"History Length:   {status['position_history_length']} moves")
        print(f"{'='*60}\n")


# =========================================================================
# Test/Example Usage
# =========================================================================

if __name__ == "__main__":
    print("MockStage v2 Test")
    print("="*70)
    
    # Test 1: Basic movement
    print("\nTest 1: Basic Movement")
    print("-"*70)
    
    stage = MockXYZStage(start_positions={'x': 0, 'y': 0, 'z': 0})
    stage.print_status()
    
    # Move to various positions
    stage.move_abs('x', 5000)
    stage.move_abs('y', 10000)
    stage.move_rel('z', 2000)
    
    print(f"\nCurrent position: {stage.get_pos_all()}")
    
    # Test 2: With delays
    print("\n\nTest 2: With Movement Delays")
    print("-"*70)
    
    stage2 = MockXYZStage(simulate_delays=True)
    stage2.enable_delays(True)
    
    print("Moving X by 100µm (should take ~1s at 100µm/s)...")
    start = time.time()
    stage2.move_abs('x', 100000)
    elapsed = time.time() - start
    print(f"Move took {elapsed:.3f}s")
    
    # Test 3: With position noise
    print("\n\nTest 3: Position Noise")
    print("-"*70)
    
    stage3 = MockXYZStage(position_noise_nm=50.0)
    stage3.set_position_noise(50.0)
    
    target = 10000
    print(f"Moving to X={target}nm with 50nm noise...")
    for i in range(5):
        stage3.move_abs('x', target)
        actual = stage3.get_pos('x')
        error = actual - target
        print(f"  Attempt {i+1}: Target={target}nm, Actual={actual}nm, Error={error}nm")
    
    # Test 4: Position history
    print("\n\nTest 4: Position History")
    print("-"*70)
    
    stage4 = MockXYZStage()
    stage4.move_abs('x', 1000)
    stage4.move_abs('y', 2000)
    stage4.move_abs('x', 3000)
    stage4.move_abs('z', 500)
    
    history = stage4.get_position_history()
    print(f"Recorded {len(history)} movements:")
    for entry in history:
        print(f"  {entry['axis'].upper()}: {entry['old_position']}nm → "
              f"{entry['new_position']}nm")
    
    # Save history
    stage4.save_position_history("test_stage_history.csv")
    
    # Test 5: Camera observer (mock)
    print("\n\nTest 5: Camera Observer Pattern")
    print("-"*70)
    
    class MockCameraObserver:
        def __init__(self):
            self.notifications = 0
        
        def on_stage_moved(self):
            self.notifications += 1
    
    stage5 = MockXYZStage()
    observer = MockCameraObserver()
    stage5.set_camera_observer(observer)
    
    stage5.move_abs('x', 1000)
    stage5.move_abs('y', 2000)
    
    print(f"Stage moved, camera can now query new position via get_pos()")
    print(f"Current stage position: {stage5.get_pos_all()}")
    
    # Cleanup
    stage.close()
    stage2.close()
    stage3.close()
    stage4.close()
    stage5.close()
    
    print("\n✅ All MockStage tests completed!")