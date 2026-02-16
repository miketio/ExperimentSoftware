# hardware_control/setup_motor/filter_stage.py
"""
Filter Stage - Single-axis MCS controller for K-space filtering

UPDATED:
- Extended range to ±15000 µm (±15,000,000 nm)
- Clears hardware position limits on initialization
- Multi-step movement for large distances to avoid issues
- Position verification after moves
"""

import ctypes as ct
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict
from datetime import datetime

import hardware_control.setup_motor.MCSControl_PythonWrapper as mcs


class FilterStage:
    """
    Single-axis MCS stage for K-space filtering.
    
    UPDATED: Extended range ±15000 µm with position limit management
    
    Features:
    - 1D sweep with configurable range and step
    - Image capture at each position
    - Metadata logging
    - Progress callbacks
    - Hardware position limit clearing
    - Multi-step movement for large distances
    
    Usage:
        stage = FilterStage(locator="usb:id:1234")
        
        # Run sweep
        results = stage.run_sweep(
            start_nm=-15000000,
            end_nm=15000000,
            step_nm=1000,
            camera=camera,
            output_dir="data/sweep_001"
        )
    """
    
    # ✅ UPDATED: Extended limits to ±15mm (±15,000,000 nm)
    POSITION_LIMIT_MIN_NM = -15_000_000  # -15mm
    POSITION_LIMIT_MAX_NM = 15_000_000   # +15mm
    
    # Multi-step movement threshold (if move > this, split into steps)
    LARGE_MOVE_THRESHOLD_NM = 1_000_000  # 1mm
    LARGE_MOVE_STEP_NM = 500_000         # 500µm per step
    
    def __init__(
        self,
        locator: str,
        axis_channel: int = 0,
        options: str = "sync,reset"
    ):
        """
        Initialize filter stage.
        
        Args:
            locator: MCS device locator (e.g., "usb:id:1234")
            axis_channel: Channel index for the filter axis (default: 0)
            options: MCS open options (default: "sync,reset")
        """
        self.locator = locator
        self.axis_channel = axis_channel
        self._closed = False
        
        # MCS handle
        self.mcsHandle = ct.c_ulong()
        
        # Open system
        locator_bytes = locator.encode('utf-8')
        options_bytes = options.encode('utf-8')
        
        status = mcs.SA_OpenSystem(self.mcsHandle, locator_bytes, options_bytes)
        self._exit_if_error(status)
        
        # Query info
        num_channels = ct.c_ulong()
        status = mcs.SA_GetNumberOfChannels(self.mcsHandle, num_channels)
        self._exit_if_error(status)
        
        self.num_channels = num_channels.value
        
        print(f"[FilterStage] Initialized on {locator}")
        print(f"  Channels: {self.num_channels}")
        print(f"  Axis channel: {axis_channel}")
        
        # Check sensor
        sensor_type = ct.c_ulong()
        status = mcs.SA_GetSensorType_S(
            self.mcsHandle,
            ct.c_ulong(axis_channel),
            sensor_type
        )
        if status == mcs.SA_OK:
            print(f"  Sensor type: {sensor_type.value}")
        
        # ✅ CRITICAL: Clear hardware position limits
        self._configure_position_limits()
    
    def _configure_position_limits(self):
        """
        Configure (or remove) hardware position limits.
        
        SmarAct stages often have default limits that prevent full range movement.
        We set them to our desired range.
        """
        print(f"[FilterStage] Configuring position limits...")
        
        channel = ct.c_ulong(self.axis_channel)
        
        # First, check current limits
        min_pos = ct.c_long()
        max_pos = ct.c_long()
        
        status = mcs.SA_GetPositionLimit_S(
            self.mcsHandle,
            channel,
            min_pos,
            max_pos
        )
        
        if status == mcs.SA_OK:
            print(f"  Current limits: {min_pos.value} to {max_pos.value} nm")
            print(f"  Current limits: {min_pos.value/1e6:.3f} to {max_pos.value/1e6:.3f} mm")
        
        # Set new limits to our desired range
        print(f"  Setting new limits: {self.POSITION_LIMIT_MIN_NM} to {self.POSITION_LIMIT_MAX_NM} nm")
        print(f"  Setting new limits: {self.POSITION_LIMIT_MIN_NM/1e6:.3f} to {self.POSITION_LIMIT_MAX_NM/1e6:.3f} mm")
        
        status = mcs.SA_SetPositionLimit_S(
            self.mcsHandle,
            channel,
            ct.c_long(self.POSITION_LIMIT_MIN_NM),
            ct.c_long(self.POSITION_LIMIT_MAX_NM)
        )
        
        if status == mcs.SA_OK:
            print(f"  ✅ Position limits updated successfully")
        else:
            # Non-fatal - some sensors don't support limits
            print(f"  ⚠️ Could not set position limits (error {status})")
            print(f"     This may be normal for some sensor types")
        
        # Verify
        status = mcs.SA_GetPositionLimit_S(
            self.mcsHandle,
            channel,
            min_pos,
            max_pos
        )
        
        if status == mcs.SA_OK:
            print(f"  Verified limits: {min_pos.value} to {max_pos.value} nm")
    
    def _exit_if_error(self, status: int):
        """Raise RuntimeError if status != SA_OK."""
        if status == mcs.SA_OK:
            return
        
        try:
            err_buf = ct.create_string_buffer(256)
            mcs.SA_GetStatusInfo(ct.c_ulong(status), err_buf)
            msg = err_buf.value.decode('utf-8', errors='ignore')
        except:
            msg = f"MCS error code {status}"
        
        raise RuntimeError(f"FilterStage error: {msg}")
    
    # =========================================================================
    # Basic Movement (UPDATED with multi-step support)
    # =========================================================================
    
    def move_abs(self, pos_nm: int, hold_time_ms: int = 0, verify: bool = True):
        """
        Move to absolute position with optional multi-step for large distances.
        
        Args:
            pos_nm: Target position in nanometers
            hold_time_ms: Hold time after movement (milliseconds)
            verify: If True, verify final position is within tolerance
        """
        # Validate position is within limits
        if not (self.POSITION_LIMIT_MIN_NM <= pos_nm <= self.POSITION_LIMIT_MAX_NM):
            raise ValueError(
                f"Position {pos_nm} nm ({pos_nm/1e6:.3f} mm) outside limits "
                f"[{self.POSITION_LIMIT_MIN_NM/1e6:.3f}, {self.POSITION_LIMIT_MAX_NM/1e6:.3f}] mm"
            )
        
        # Get current position
        current_pos = self.get_position()
        distance = abs(pos_nm - current_pos)
        
        print(f"[FilterStage] Moving from {current_pos/1000:.3f} to {pos_nm/1000:.3f} µm "
              f"(distance: {distance/1000:.3f} µm)")
        
        # ✅ For large moves, split into steps to avoid issues
        if distance > self.LARGE_MOVE_THRESHOLD_NM:
            print(f"  Large move detected ({distance/1e6:.3f} mm) - using multi-step approach")
            self._move_abs_multistep(current_pos, pos_nm, hold_time_ms)
        else:
            # Direct move for small distances
            self._move_abs_direct(pos_nm, hold_time_ms)
        
        # ✅ Verify final position
        if verify:
            final_pos = self.get_position()
            error = abs(final_pos - pos_nm)
            
            if error > 1000:  # 1µm tolerance
                print(f"  ⚠️ Position error: {error} nm ({error/1000:.3f} µm)")
                print(f"     Target: {pos_nm} nm, Actual: {final_pos} nm")
            else:
                print(f"  ✅ Position verified: {final_pos} nm (error: {error} nm)")
    
    def _move_abs_direct(self, pos_nm: int, hold_time_ms: int = 0):
        """Direct absolute move (internal)."""
        channel = ct.c_ulong(self.axis_channel)
        position = ct.c_long(int(pos_nm))
        hold = ct.c_ulong(hold_time_ms)
        
        status = mcs.SA_GotoPositionAbsolute_S(
            self.mcsHandle,
            channel,
            position,
            hold
        )
        self._exit_if_error(status)
        
        # Wait for move to complete
        self._wait_for_stop()
    
    def _move_abs_multistep(self, start_nm: int, target_nm: int, hold_time_ms: int = 0):
        """
        Move in multiple steps for large distances.
        
        This helps avoid issues with:
        - Position limit violations
        - Motor step size limitations
        - Communication timeouts
        """
        direction = 1 if target_nm > start_nm else -1
        distance = abs(target_nm - start_nm)
        
        # Calculate number of steps
        num_steps = int(np.ceil(distance / self.LARGE_MOVE_STEP_NM))
        
        print(f"  Multi-step move: {num_steps} steps of ~{self.LARGE_MOVE_STEP_NM/1000:.1f} µm")
        
        # Generate intermediate positions
        positions = np.linspace(start_nm, target_nm, num_steps + 1)[1:]  # Exclude start
        
        for i, pos in enumerate(positions):
            pos_int = int(pos)
            print(f"    Step {i+1}/{num_steps}: {pos_int/1000:.3f} µm")
            
            self._move_abs_direct(pos_int, hold_time_ms if i == len(positions)-1 else 0)
            
            # Brief pause between steps
            if i < len(positions) - 1:
                time.sleep(0.1)
    
    def _wait_for_stop(self, timeout_s: float = 30.0):
        """
        Wait for stage to stop moving.
        
        Args:
            timeout_s: Maximum time to wait (seconds)
        """
        channel = ct.c_ulong(self.axis_channel)
        status_val = ct.c_ulong()
        
        start_time = time.time()
        
        while True:
            status = mcs.SA_GetStatus_S(self.mcsHandle, channel, status_val)
            self._exit_if_error(status)
            
            # Check if stopped
            if status_val.value == mcs.SA_STOPPED_STATUS or status_val.value == mcs.SA_TARGET_STATUS:
                break
            
            # Check timeout
            if time.time() - start_time > timeout_s:
                raise RuntimeError(f"Move timeout after {timeout_s}s")
            
            time.sleep(0.05)  # 50ms polling
        
        # Final settle time
        time.sleep(0.2)
    
    def move_rel(self, shift_nm: int, hold_time_ms: int = 0):
        """
        Move relative to current position.
        
        Args:
            shift_nm: Distance to move in nanometers
            hold_time_ms: Hold time after movement (milliseconds)
        """
        current_pos = self.get_position()
        target_pos = current_pos + shift_nm
        
        # Use absolute move with validation
        self.move_abs(target_pos, hold_time_ms)
    
    def get_position(self) -> int:
        """
        Get current position.
        
        Returns:
            Position in nanometers
        """
        channel = ct.c_ulong(self.axis_channel)
        position = ct.c_long()
        
        status = mcs.SA_GetPosition_S(
            self.mcsHandle,
            channel,
            position
        )
        self._exit_if_error(status)
        
        return int(position.value)
    
    # =========================================================================
    # Sweep Operation
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
        Run 1D sweep and capture images.
        
        Args:
            start_nm: Start position (nanometers)
            end_nm: End position (nanometers)
            step_nm: Step size (nanometers)
            camera: Camera instance with acquire_single_image() method
            output_dir: Directory to save images and metadata
            settle_time_s: Wait time after each move (seconds)
            progress_callback: Optional function(current, total) for progress
        
        Returns:
            dict with sweep results
        """
        print(f"[FilterStage] Starting sweep:")
        print(f"  Range: {start_nm} to {end_nm} nm ({(end_nm-start_nm)/1e6:.3f} mm)")
        print(f"  Step: {step_nm} nm ({step_nm/1000:.3f} µm)")
        
        # Validate range
        if not (self.POSITION_LIMIT_MIN_NM <= start_nm <= self.POSITION_LIMIT_MAX_NM):
            raise ValueError(f"Start position {start_nm} nm outside limits")
        
        if not (self.POSITION_LIMIT_MIN_NM <= end_nm <= self.POSITION_LIMIT_MAX_NM):
            raise ValueError(f"End position {end_nm} nm outside limits")
        
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
                # Move to position (with multi-step support)
                self.move_abs(target_pos, verify=False)  # Skip verification in loop for speed
                
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
            print("\n[FilterStage] ⚠️ Sweep interrupted by user")
        
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
                'mcs_locator': self.locator,
                'axis_channel': self.axis_channel,
                'position_limits_nm': {
                    'min': self.POSITION_LIMIT_MIN_NM,
                    'max': self.POSITION_LIMIT_MAX_NM
                }
            }
        }
        
        # Save metadata JSON
        import json
        metadata_file = output_path / "sweep_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[FilterStage] ✅ Sweep complete!")
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
    # Convenience Methods
    # =========================================================================
    
    def home(self):
        """Move to zero position."""
        print("[FilterStage] Moving to home (0 nm)...")
        self.move_abs(0)
    
    def get_status(self) -> Dict:
        """Get stage status."""
        try:
            pos = self.get_position()
            
            # Query stage status
            channel = ct.c_ulong(self.axis_channel)
            status = ct.c_ulong()
            mcs.SA_GetStatus_S(self.mcsHandle, channel, status)
            
            status_names = {
                mcs.SA_STOPPED_STATUS: 'STOPPED',
                mcs.SA_STEPPING_STATUS: 'STEPPING',
                mcs.SA_SCANNING_STATUS: 'SCANNING',
                mcs.SA_HOLDING_STATUS: 'HOLDING',
                mcs.SA_TARGET_STATUS: 'TARGET',
                mcs.SA_MOVE_DELAY_STATUS: 'MOVE_DELAY',
                mcs.SA_CALIBRATING_STATUS: 'CALIBRATING',
                mcs.SA_FINDING_REF_STATUS: 'FINDING_REF'
            }
            
            status_name = status_names.get(status.value, f'UNKNOWN({status.value})')
            
            # Get position limits
            min_pos = ct.c_long()
            max_pos = ct.c_long()
            mcs.SA_GetPositionLimit_S(self.mcsHandle, channel, min_pos, max_pos)
            
            return {
                'connected': not self._closed,
                'position_nm': pos,
                'position_um': pos / 1000.0,
                'status': status_name,
                'locator': self.locator,
                'channel': self.axis_channel,
                'limits_nm': {
                    'min': min_pos.value,
                    'max': max_pos.value
                },
                'limits_um': {
                    'min': min_pos.value / 1000.0,
                    'max': max_pos.value / 1000.0
                }
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def print_status(self):
        """Print stage status to console."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("FilterStage Status")
        print("="*60)
        print(f"Locator:   {status.get('locator', 'N/A')}")
        print(f"Channel:   {status.get('channel', 'N/A')}")
        print(f"Connected: {status.get('connected', False)}")
        
        if status.get('connected'):
            print(f"Position:  {status['position_nm']} nm ({status['position_um']:.3f} µm)")
            print(f"Status:    {status.get('status', 'UNKNOWN')}")
            
            if 'limits_nm' in status:
                lim = status['limits_nm']
                print(f"Limits:    {lim['min']} to {lim['max']} nm")
                print(f"           ({lim['min']/1e6:.3f} to {lim['max']/1e6:.3f} mm)")
        else:
            print(f"Error:     {status.get('error', 'Unknown')}")
        
        print("="*60 + "\n")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self):
        """Close MCS connection."""
        if self._closed:
            return
        
        try:
            status = mcs.SA_CloseSystem(self.mcsHandle)
            self._exit_if_error(status)
        finally:
            self._closed = True
            print("[FilterStage] Closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("FilterStage Test - Extended Range")
    print("="*70)
    
    # Note: Replace with actual MCS locator
    
    try:
        # Create filter stage (use actual locator from discovery)
        stage = FilterStage(locator="usb:id:1234", axis_channel=0)
        
        # Print status
        stage.print_status()
        
        # Test movement
        print("\nTest: Movement across extended range")
        print("-"*70)
        
        # Test positions (µm → nm)
        test_positions_um = [0, 100, -100, 1000, -1000, 5000, -5000]
        
        for pos_um in test_positions_um:
            pos_nm = int(pos_um * 1000)
            print(f"\nMoving to {pos_um} µm ({pos_nm} nm)...")
            
            try:
                stage.move_abs(pos_nm)
                actual_nm = stage.get_position()
                actual_um = actual_nm / 1000
                error_um = abs(actual_um - pos_um)
                
                print(f"  Target: {pos_um} µm")
                print(f"  Actual: {actual_um:.3f} µm")
                print(f"  Error:  {error_um:.3f} µm")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        # Home
        print("\nReturning to home...")
        stage.home()
        
        stage.close()
        print("\n✅ Test complete!")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()