# hardware_control/setup_motor/mock_filter_stage.py
"""
Mock Filter Stage - Simulates 1D filter hardware

Provides same API as FilterStage but without real hardware.
Used for testing and development.
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
from datetime import datetime


class MockFilterStage:
    """
    Mock single-axis filter stage for K-space filtering.
    
    Simulates FilterStage behavior without hardware:
    - Position tracking
    - Movement delays
    - Sweep operations
    - Metadata logging
    
    API matches FilterStage exactly.
    """
    
    def __init__(
        self,
        locator: str = "mock:filter:0",
        axis_channel: int = 0,
        simulate_delays: bool = True,
        position_noise_nm: float = 0.0
    ):
        """
        Initialize mock filter stage.
        
        Args:
            locator: Mock locator string (for compatibility)
            axis_channel: Channel index (mock, ignored)
            simulate_delays: If True, add realistic movement delays
            position_noise_nm: Amount of position noise (nm, std dev)
        """
        self.locator = locator
        self.axis_channel = axis_channel
        self._closed = False
        
        # Mock position (nanometers)
        self._position_nm = 0
        
        # Simulation parameters
        self.simulate_delays = simulate_delays
        self.position_noise_nm = position_noise_nm
        
        # Movement speed (nm/s) for delay simulation
        self.speed_nm_per_s = 50000  # 50 µm/s
        
        # Mock hardware info
        self.num_channels = 1
        
        # Position history
        self.position_history: List[Tuple[float, int]] = []
        
        print(f"[MockFilterStage] Initialized (MOCK mode)")
        print(f"  Locator: {locator}")
        print(f"  Axis channel: {axis_channel}")
    
    # =========================================================================
    # Basic Movement (matches FilterStage API)
    # =========================================================================
    
    def move_abs(self, pos_nm: int, hold_time_ms: int = 0):
        """
        Move to absolute position (simulated).
        
        Args:
            pos_nm: Target position in nanometers
            hold_time_ms: Hold time after movement (milliseconds, ignored in mock)
        """
        if self._closed:
            raise RuntimeError("MockFilterStage is closed")
        
        old_pos = self._position_nm
        target_pos = int(pos_nm)
        
        # Simulate movement delay if enabled
        if self.simulate_delays:
            distance = abs(target_pos - old_pos)
            delay = distance / self.speed_nm_per_s
            time.sleep(min(delay, 0.5))  # Cap at 0.5s
        else:
            time.sleep(0.01)  # Minimal delay
        
        # Update position
        self._position_nm = target_pos
        
        # Add noise if enabled
        if self.position_noise_nm > 0:
            noise = np.random.normal(0, self.position_noise_nm)
            self._position_nm += int(noise)
        
        # Log position
        self.position_history.append((time.time(), self._position_nm))
        
        # Settle time
        time.sleep(0.2)
        
        print(f"[MockFilterStage] Moved: {old_pos}nm → {self._position_nm}nm ({self._position_nm/1000:.3f}µm)")
    
    def move_rel(self, shift_nm: int, hold_time_ms: int = 0):
        """
        Move relative to current position (simulated).
        
        Args:
            shift_nm: Distance to move in nanometers
            hold_time_ms: Hold time after movement (milliseconds, ignored in mock)
        """
        target = self._position_nm + int(shift_nm)
        self.move_abs(target, hold_time_ms)
    
    def get_position(self) -> int:
        """
        Get current position (simulated).
        
        Returns:
            Position in nanometers
        """
        if self._closed:
            raise RuntimeError("MockFilterStage is closed")
        
        return int(self._position_nm)
    
    # =========================================================================
    # Sweep Operation (matches FilterStage API)
    # =========================================================================
    
    def run_sweep(
        self,
        start_nm: int,
        end_nm: int,
        step_nm: int,
        camera,
        output_dir: str,
        settle_time_s: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Run 1D sweep and capture images (MOCK).
        
        Args:
            start_nm: Start position (nanometers)
            end_nm: End position (nanometers)
            step_nm: Step size (nanometers)
            camera: Camera instance with acquire_single_image() method
            output_dir: Directory to save images and metadata
            settle_time_s: Wait time after each move (seconds)
            progress_callback: Optional function(current, total) for progress
        
        Returns:
            dict with sweep results:
                - positions: List of positions (nm)
                - image_files: List of saved image paths
                - metadata_file: Path to metadata JSON
                - start_time: ISO timestamp
                - duration_s: Total sweep time
        """
        print(f"[MockFilterStage] Starting MOCK sweep:")
        print(f"  Range: {start_nm} to {end_nm} nm ({(end_nm-start_nm)/1000:.3f} µm)")
        print(f"  Step: {step_nm} nm ({step_nm/1000:.3f} µm)")
        
        # Calculate positions
        positions = list(range(start_nm, end_nm + step_nm, step_nm))
        num_positions = len(positions)
        
        print(f"  Total positions: {num_positions}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        start_time = datetime.now()
        image_files = []
        actual_positions = []
        
        try:
            for idx, target_pos in enumerate(positions):
                # Move to position
                self.move_abs(target_pos)
                
                # Wait for settling
                time.sleep(settle_time_s)
                
                # Read back actual position
                actual_pos = self.get_position()
                actual_positions.append(actual_pos)
                
                # Capture image
                try:
                    image = camera.acquire_single_image()
                    
                    # Save image
                    image_filename = f"img_{idx:04d}_pos_{actual_pos}nm.tif"
                    image_path = output_path / image_filename
                    
                    # Save as 16-bit TIFF
                    import tifffile
                    tifffile.imwrite(str(image_path), image)
                    
                    image_files.append(str(image_path))
                    
                    print(f"  [{idx+1}/{num_positions}] "
                          f"Pos={actual_pos}nm ({actual_pos/1000:.3f}µm) "
                          f"→ {image_filename}")
                
                except Exception as e:
                    print(f"  ⚠️ Failed to capture at position {actual_pos}nm: {e}")
                    image_files.append(None)
                
                # Progress callback
                if progress_callback:
                    progress_callback(idx + 1, num_positions)
        
        except KeyboardInterrupt:
            print("\n[MockFilterStage] ⚠️ Sweep interrupted by user")
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save metadata
        metadata = {
            'sweep_config': {
                'start_nm': start_nm,
                'end_nm': end_nm,
                'step_nm': step_nm,
                'requested_positions': num_positions
            },
            'actual_data': {
                'target_positions_nm': positions,
                'actual_positions_nm': actual_positions,
                'image_files': image_files
            },
            'timing': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'settle_time_s': settle_time_s
            },
            'hardware': {
                'mode': 'MOCK',
                'locator': self.locator,
                'axis_channel': self.axis_channel
            }
        }
        
        # Save metadata JSON
        import json
        metadata_file = output_path / "sweep_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[MockFilterStage] ✅ MOCK sweep complete!")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Images saved: {len([f for f in image_files if f is not None])}/{num_positions}")
        print(f"  Output: {output_dir}")
        
        return {
            'positions': actual_positions,
            'image_files': image_files,
            'metadata_file': str(metadata_file),
            'start_time': start_time.isoformat(),
            'duration_s': duration
        }
    
    # =========================================================================
    # Convenience Methods (matches FilterStage API)
    # =========================================================================
    
    def home(self):
        """Move to zero position (simulated)."""
        print("[MockFilterStage] Moving to home (0 nm)...")
        self.move_abs(0)
    
    def get_status(self) -> Dict:
        """Get mock stage status (matches FilterStage API)."""
        status_names = {
            0: 'STOPPED',
            1: 'STEPPING',
            2: 'SCANNING',
            3: 'HOLDING',
            4: 'TARGET'
        }
        
        return {
            'connected': not self._closed,
            'mode': 'MOCK',
            'position_nm': self._position_nm,
            'position_um': self._position_nm / 1000.0,
            'status': 'STOPPED',  # Mock always stopped when queried
            'locator': self.locator,
            'channel': self.axis_channel
        }
    
    def print_status(self):
        """Print stage status to console (matches FilterStage API)."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("MockFilterStage Status")
        print("="*60)
        print(f"Mode:      {status['mode']}")
        print(f"Locator:   {status.get('locator', 'N/A')}")
        print(f"Channel:   {status.get('channel', 'N/A')}")
        print(f"Connected: {status.get('connected', False)}")
        print(f"Position:  {status['position_nm']} nm ({status['position_um']:.3f} µm)")
        print(f"Status:    {status.get('status', 'UNKNOWN')}")
        print("="*60 + "\n")
    
    # =========================================================================
    # Simulation Control (extras not in real FilterStage)
    # =========================================================================
    
    def enable_delays(self, enable: bool = True):
        """Enable or disable movement delays."""
        self.simulate_delays = enable
        print(f"[MockFilterStage] Movement delays: {'enabled' if enable else 'disabled'}")
    
    def set_position_noise(self, noise_nm: float):
        """
        Set position noise level.
        
        Args:
            noise_nm: Standard deviation of position noise in nanometers
        """
        self.position_noise_nm = noise_nm
        print(f"[MockFilterStage] Position noise set to {noise_nm}nm (std dev)")
    
    def set_speed(self, speed_nm_per_s: float):
        """
        Set movement speed for delay simulation.
        
        Args:
            speed_nm_per_s: Speed in nm/s
        """
        self.speed_nm_per_s = speed_nm_per_s
        print(f"[MockFilterStage] Speed set to {speed_nm_per_s/1000:.1f} µm/s")
    
    def get_position_history(self) -> List[Tuple[float, int]]:
        """
        Get position history.
        
        Returns:
            List of (timestamp, position_nm) tuples
        """
        return self.position_history.copy()
    
    def clear_position_history(self):
        """Clear position history."""
        self.position_history.clear()
        print(f"[MockFilterStage] Position history cleared")
    
    # =========================================================================
    # Cleanup (matches FilterStage API)
    # =========================================================================
    
    def close(self):
        """Close mock stage (matches FilterStage API)."""
        if self._closed:
            return
        
        self._closed = True
        print("[MockFilterStage] Closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Example Usage / Testing
# =============================================================================

if __name__ == "__main__":
    print("MockFilterStage Test")
    print("="*70)
    
    # Test 1: Basic movement
    print("\nTest 1: Basic Movement")
    print("-"*70)
    
    stage = MockFilterStage()
    stage.print_status()
    
    # Move to various positions
    stage.move_abs(10000)
    print(f"Position: {stage.get_position()}nm")
    
    stage.move_rel(5000)
    print(f"Position: {stage.get_position()}nm")
    
    stage.home()
    print(f"Position: {stage.get_position()}nm")
    
    # Test 2: With delays
    print("\n\nTest 2: With Movement Delays")
    print("-"*70)
    
    stage2 = MockFilterStage(simulate_delays=True)
    
    print("Moving to 100µm (should take ~2s at 50µm/s)...")
    start = time.time()
    stage2.move_abs(100000)
    elapsed = time.time() - start
    print(f"Move took {elapsed:.3f}s")
    
    # Test 3: Position noise
    print("\n\nTest 3: Position Noise")
    print("-"*70)
    
    stage3 = MockFilterStage(position_noise_nm=50.0)
    
    target = 10000
    print(f"Moving to {target}nm with 50nm noise...")
    for i in range(5):
        stage3.move_abs(target)
        actual = stage3.get_position()
        error = actual - target
        print(f"  Attempt {i+1}: Target={target}nm, Actual={actual}nm, Error={error}nm")
    
    # Test 4: Status
    print("\n\nTest 4: Status")
    print("-"*70)
    
    stage4 = MockFilterStage()
    stage4.move_abs(25000)
    stage4.print_status()
    
    # Cleanup
    stage.close()
    stage2.close()
    stage3.close()
    stage4.close()
    
    print("\n✅ All MockFilterStage tests completed!")